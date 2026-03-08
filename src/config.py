import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"


def _path_from_env(env_name: str, default: Path) -> Path:
    raw_value = os.environ.get(env_name, "").strip()
    if not raw_value:
        return default

    env_path = Path(raw_value).expanduser()
    if env_path.is_absolute():
        return env_path
    return BASE_DIR / env_path


MODEL_PATH = _path_from_env("PLANTCARE_MODEL_PATH", MODELS_DIR / "best_model.keras")
CLASS_NAMES_PATH = _path_from_env("PLANTCARE_CLASS_NAMES_PATH", MODELS_DIR / "class_names.json")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
