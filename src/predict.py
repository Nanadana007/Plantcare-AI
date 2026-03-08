from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.config import CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH, MODELS_DIR
from src.recommendations import get_recommendation


class HuberLossLayer(tf.keras.layers.Layer):
    """Compatibility layer for externally trained models that include this custom layer."""

    def call(self, y_true, y_pred):
        return tf.keras.losses.Huber()(y_true, y_pred)


class CustomScaleLayer(tf.keras.layers.Layer):
    """Compatibility layer used by some external plant disease models."""

    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            left = tf.cast(inputs[0], tf.float32)
            right = tf.cast(inputs[1], tf.float32)
            return left + (right * self.scale)
        tensor = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        tensor = tf.cast(tensor, tf.float32)
        return tensor * self.scale

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and input_shape:
            first = input_shape[0]
            if isinstance(first, tf.TensorShape):
                return first
            if isinstance(first, (list, tuple)):
                return tf.TensorShape(first)
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


class PlantDiseasePredictor:
    SUPPORTED_SPECIES = (
        "Apple",
        "Blueberry",
        "Cherry",
        "Corn",
        "Grape",
        "Orange",
        "Peach",
        "Pepper",
        "Potato",
        "Raspberry",
        "Soybean",
        "Squash",
        "Strawberry",
        "Tomato",
    )

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        class_names_path: str | Path = CLASS_NAMES_PATH,
        image_size: tuple[int, int] = IMAGE_SIZE,
    ) -> None:
        self.model_path = self._resolve_model_path(Path(model_path))
        self.class_names_path = self._resolve_class_names_path(Path(class_names_path))
        self.image_size = image_size

        custom_objects = {
            "HuberLossLayer": HuberLossLayer,
            "CustomScaleLayer": CustomScaleLayer,
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            self.model = tf.keras.models.load_model(
                self.model_path,
                compile=False,
            )
        self.image_size = self._resolve_image_size(self.image_size)
        self.class_names = self._load_class_names()
        self._validate_plant_classes()
        self._validate_output_dim()

    def _resolve_model_path(self, requested_path: Path) -> Path:
        candidates = [
            requested_path,
            MODELS_DIR / "best_model.keras",
            MODELS_DIR / "best_model.h5",
            MODELS_DIR / "Pretrained_model.h5",
            MODELS_DIR / "mobilenetv2_best.keras",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            "Model not found. Expected one of: "
            + ", ".join(str(path) for path in candidates)
        )

    def _resolve_class_names_path(self, requested_path: Path) -> Path:
        if requested_path.exists():
            return requested_path

        class_indices_path = MODELS_DIR / "class_indices.json"
        if class_indices_path.exists():
            with class_indices_path.open("r", encoding="utf-8") as f:
                class_indices = json.load(f)
            class_names = [
                name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])
            ]
            with requested_path.open("w", encoding="utf-8") as f:
                json.dump(class_names, f, indent=2)
            return requested_path

        config_path = MODELS_DIR / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                model_config = json.load(f)
            classes = model_config.get("classes")
            if isinstance(classes, list) and classes:
                with requested_path.open("w", encoding="utf-8") as f:
                    json.dump(classes, f, indent=2)
                return requested_path

        raise FileNotFoundError(f"Class names not found: {requested_path}")

    def _load_class_names(self) -> list[str]:
        with self.class_names_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_image_size(self, fallback_size: tuple[int, int]) -> tuple[int, int]:
        input_shape = self.model.input_shape
        if (
            isinstance(input_shape, tuple)
            and len(input_shape) == 4
            and isinstance(input_shape[1], int)
            and isinstance(input_shape[2], int)
        ):
            return (input_shape[1], input_shape[2])
        return fallback_size

    def _validate_output_dim(self) -> None:
        output_shape: Any = self.model.output_shape
        output_dim = output_shape[-1] if isinstance(output_shape, tuple) else None
        if isinstance(output_dim, int) and output_dim != len(self.class_names):
            raise ValueError(
                f"Class count mismatch: model outputs {output_dim}, "
                f"but class_names has {len(self.class_names)} entries."
            )

    def _validate_plant_classes(self) -> None:
        has_plant_format = any("___" in class_name for class_name in self.class_names)
        if len(self.class_names) < 20 or not has_plant_format:
            raise ValueError(
                "Invalid model for PlantCare AI. Expected plant-disease class names (e.g., "
                "'Tomato___healthy') and >=20 classes."
            )

    def predict(self, image_path: str | Path) -> dict[str, str | float]:
        image_tensor = self._load_image(image_path)
        probabilities = self.model.predict(image_tensor, verbose=0)[0].astype(np.float64)

        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        top2 = np.partition(probabilities, -2)[-2:]
        margin = float(top2[-1] - top2[-2])
        label = self.class_names[class_index]
        agreement, tta_mean_confidence = self._tta_consistency(image_tensor, class_index)
        image_std = float(np.std(image_tensor))

        highly_uncertain = (
            confidence < 0.60
            and margin < 0.06
            and agreement < 0.45
            and tta_mean_confidence < 0.60
        )
        visually_blank = image_std < 10.0
        is_supported = not highly_uncertain and not visually_blank

        if not is_supported:
            return {
                "label": "Unsupported Image",
                "raw_prediction": label,
                "confidence": round(confidence * 100, 2),
                "recommendation": (
                    "Image appears outside PlantCare model scope. Upload a close-up of a single leaf "
                    "from supported crops: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, "
                    "Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato."
                ),
                "is_supported": False,
            }

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "recommendation": get_recommendation(label),
            "is_supported": True,
        }

    def _load_image(self, image_path: str | Path) -> np.ndarray:
        image = tf.keras.utils.load_img(image_path, target_size=self.image_size)
        image_array = tf.keras.utils.img_to_array(image)
        return np.expand_dims(image_array, axis=0)

    def _tta_consistency(self, image_tensor: np.ndarray, base_class_index: int) -> tuple[float, float]:
        x = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
        variants = [
            x,
            tf.image.flip_left_right(x),
            tf.image.flip_up_down(x),
            tf.image.adjust_brightness(x, delta=0.08),
            tf.image.adjust_contrast(x, contrast_factor=1.08),
        ]
        batch = tf.concat(variants, axis=0)
        probs = self.model.predict(batch, verbose=0)
        top_indices = np.argmax(probs, axis=1)
        agreement = float(np.mean(top_indices == base_class_index))
        mean_confidence = float(np.mean(np.max(probs, axis=1)))
        return agreement, mean_confidence
