from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import venv
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_STAMP = VENV_DIR / ".requirements.sha256"

MODEL_CANDIDATES = (
    PROJECT_ROOT / "models" / "best_model.keras",
    PROJECT_ROOT / "models" / "best_model.h5",
    PROJECT_ROOT / "models" / "Pretrained_model.h5",
)
CLASS_NAMES_FILE = PROJECT_ROOT / "models" / "class_names.json"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/yasminehedfi/plant-disease-keras/resolve/main/plant_disease_model.keras"
)
DEFAULT_IDX2LABEL_URL = (
    "https://huggingface.co/yasminehedfi/plant-disease-keras/resolve/main/idx2label.json"
)
DEFAULT_LABELS_TXT_URL = (
    "https://huggingface.co/yasminehedfi/plant-disease-keras/resolve/main/labels.txt"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PlantCare cross-platform launcher")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Set up environment and validate files, then exit without starting Flask.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Preferred server port. If busy, launcher auto-selects a free port.",
    )
    return parser.parse_args()


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT), env=env)


def _ensure_venv() -> Path:
    venv_python = _venv_python()
    if venv_python.exists():
        return venv_python

    print(f"[INFO] Creating virtual environment at: {VENV_DIR}")
    builder = venv.EnvBuilder(with_pip=True, clear=False, upgrade=False)
    builder.create(str(VENV_DIR))

    if not venv_python.exists():
        raise RuntimeError(f"Virtual environment created but python not found at: {venv_python}")
    return venv_python


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _install_requirements(python_exe: Path) -> None:
    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing requirements file: {REQUIREMENTS_FILE}")

    current_hash = _sha256(REQUIREMENTS_FILE)
    previous_hash = ""
    if REQUIREMENTS_STAMP.exists():
        previous_hash = REQUIREMENTS_STAMP.read_text(encoding="utf-8").strip()

    if previous_hash == current_hash:
        print("[INFO] Requirements unchanged. Skipping dependency install.")
        return

    _run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
    REQUIREMENTS_STAMP.write_text(current_hash, encoding="utf-8")


def _ensure_runtime_dirs() -> None:
    for rel in ("models", "uploads", "results", "static/images"):
        (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)


def _download_file(url: str, destination: Path) -> None:
    print(f"[INFO] Downloading {url} -> {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())


def _download_json(url: str) -> dict[str, str]:
    with urlopen(url, timeout=120) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object for class mapping.")
    return data


def _save_class_names_from_idx2label(idx2label: dict[str, str], output_path: Path) -> None:
    class_items: list[tuple[int, str]] = []
    for key, value in idx2label.items():
        try:
            class_index = int(key)
        except ValueError:
            continue
        class_items.append((class_index, str(value)))

    if not class_items:
        raise ValueError("No numeric class indices found in idx2label mapping.")

    class_names = [label for _, label in sorted(class_items, key=lambda item: item[0])]
    output_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def _download_class_names_fallback(output_path: Path) -> None:
    print(f"[INFO] Downloading default class mapping from: {DEFAULT_IDX2LABEL_URL}")
    idx2label = _download_json(DEFAULT_IDX2LABEL_URL)
    _save_class_names_from_idx2label(idx2label, output_path)
    print(f"[INFO] Saved class names to: {output_path}")


def _download_class_names_from_labels_txt(output_path: Path) -> None:
    print(f"[INFO] Downloading fallback labels from: {DEFAULT_LABELS_TXT_URL}")
    with urlopen(DEFAULT_LABELS_TXT_URL, timeout=120) as response:
        lines = response.read().decode("utf-8").splitlines()

    class_names = [line.strip() for line in lines if line.strip()]
    if not class_names:
        raise ValueError("labels.txt was empty.")

    output_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"[INFO] Saved class names to: {output_path}")


def _ensure_model_files() -> None:
    has_model = any(candidate.exists() for candidate in MODEL_CANDIDATES)
    if not has_model:
        model_url = os.environ.get("PLANTCARE_MODEL_URL", "").strip() or DEFAULT_MODEL_URL
        try:
            _download_file(model_url, MODEL_CANDIDATES[0])
            has_model = True
        except Exception as exc:
            raise FileNotFoundError(
                "No local model file and automatic model download failed. "
                "Set PLANTCARE_MODEL_URL to a valid .keras/.h5 file or place "
                "best_model.keras in models/."
            ) from exc

    if not has_model:
        raise FileNotFoundError(
            "No model file found. Put one of these files in models/: "
            "best_model.keras, best_model.h5, Pretrained_model.h5. "
            "Or set PLANTCARE_MODEL_URL to auto-download at startup."
        )

    if CLASS_NAMES_FILE.exists():
        return

    class_url = os.environ.get("PLANTCARE_CLASS_NAMES_URL", "").strip()
    if class_url:
        _download_file(class_url, CLASS_NAMES_FILE)
        return

    try:
        _download_class_names_fallback(CLASS_NAMES_FILE)
        return
    except Exception:
        _download_class_names_from_labels_txt(CLASS_NAMES_FILE)


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _pick_port(preferred_port: int) -> int:
    if _is_port_free(preferred_port):
        return preferred_port

    for candidate in range(preferred_port + 1, preferred_port + 101):
        if _is_port_free(candidate):
            print(
                f"[WARN] Port {preferred_port} is busy. Using free port {candidate} instead."
            )
            return candidate

    raise RuntimeError(
        f"No free port found in range {preferred_port}-{preferred_port + 100}. "
        "Set a custom port with --port."
    )


def _start_app(python_exe: Path, cli_port: int | None = None) -> None:
    env = os.environ.copy()
    requested_port = cli_port or int(env.get("PORT", "5000"))
    selected_port = _pick_port(requested_port)
    env["PORT"] = str(selected_port)
    env.setdefault("FLASK_DEBUG", "0")
    print(f"[INFO] Starting PlantCare AI at: http://127.0.0.1:{selected_port}")
    _run([str(python_exe), "app.py"], env=env)


def main() -> None:
    args = _parse_args()

    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ is required.")

    _ensure_runtime_dirs()
    venv_python = _ensure_venv()
    _install_requirements(venv_python)
    _ensure_model_files()
    if args.check:
        print("[INFO] Environment check passed.")
        return
    _start_app(venv_python, cli_port=args.port)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
