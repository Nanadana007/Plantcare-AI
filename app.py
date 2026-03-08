from __future__ import annotations

import os
import secrets
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from PIL import Image
from werkzeug.utils import secure_filename

from src.config import UPLOADS_DIR

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "plantcare-dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
app.config["STATIC_FOLDER"] = "static"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
Path(app.config["STATIC_FOLDER"], "images").mkdir(parents=True, exist_ok=True)

predictor: Any | None = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model() -> None:
    global predictor
    if predictor is None:
        from src.predict import PlantDiseasePredictor

        predictor = PlantDiseasePredictor()
        print("Model loaded successfully.")


def predict_image(image_path: str | Path) -> dict[str, str | float]:
    if predictor is None:
        load_model()
    assert predictor is not None
    return predictor.predict(image_path)


def _humanize_label(text: str) -> str:
    text = text.replace("_", " ").replace("  ", " ").strip()
    if not text:
        return "Unknown"
    return text.title()


def _format_prediction(prediction: dict[str, str | float]) -> dict[str, str | float | bool]:
    if not bool(prediction.get("is_supported", True)):
        confidence = float(prediction.get("confidence", 0.0))
        recommendation = str(prediction.get("recommendation", "Unsupported image for this model."))
        raw_prediction = str(prediction.get("raw_prediction", "Unknown"))
        return {
            "raw_label": raw_prediction,
            "confidence": round(confidence, 2),
            "recommendation": recommendation,
            "plant_name": "Unsupported Image",
            "condition_name": "Out of Model Scope",
            "is_healthy": False,
            "is_supported": False,
            "status_text": "Unsupported Input",
        }

    label = str(prediction.get("label", "Unknown"))
    confidence = float(prediction.get("confidence", 0.0))
    recommendation = str(prediction.get("recommendation", "No recommendation available."))

    if "___" in label:
        plant_raw, condition_raw = label.split("___", 1)
    else:
        plant_raw, condition_raw = label, "Unknown condition"

    plant_name = _humanize_label(plant_raw)
    condition_name = _humanize_label(condition_raw)
    is_healthy = "healthy" in condition_name.lower()

    return {
        "raw_label": label,
        "confidence": round(confidence, 2),
        "recommendation": recommendation,
        "plant_name": plant_name,
        "condition_name": condition_name,
        "is_healthy": is_healthy,
        "is_supported": True,
        "status_text": "Healthy Plant" if is_healthy else "Disease Detected",
    }


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", active_page="home")


@app.route("/about")
def about():
    return render_template("about.html", active_page="about")


@app.route("/upload")
def upload():
    return render_template("upload.html", active_page="upload")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image_file = request.files["file"]
    if image_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"error": "Only PNG, JPG, and JPEG files are supported"}), 400

    safe_name = secure_filename(image_file.filename)
    temp_name = f"{uuid.uuid4().hex}_{safe_name}"
    temp_path = Path(app.config["UPLOAD_FOLDER"]) / temp_name
    image_file.save(temp_path)

    try:
        prediction = predict_image(temp_path)

        static_filename = f"upload_{secrets.token_hex(8)}.jpg"
        static_path = Path(app.config["STATIC_FOLDER"]) / "images" / static_filename
        Image.open(temp_path).convert("RGB").save(static_path)

        session["prediction"] = prediction
        session["image_path"] = f"images/{static_filename}"

        temp_path.unlink(missing_ok=True)
        return jsonify({"success": True})
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/result")
def result():
    prediction = session.get("prediction")
    image_path = session.get("image_path")

    if not prediction:
        return redirect(url_for("upload"))

    analysis = _format_prediction(prediction)
    return render_template(
        "result.html",
        active_page="upload",
        prediction=prediction,
        analysis=analysis,
        image_path=image_path,
    )


@app.route("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    try:
        load_model()
    except Exception as exc:
        print(f"[WARN] Model preload failed: {exc}")
        print("[INFO] App started without model preload. Add model files to models/ to enable predictions.")
    debug_mode = os.environ.get("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes"}
    app.run(
        debug=debug_mode,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
    )
