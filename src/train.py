from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import BATCH_SIZE, IMAGE_SIZE, RESULTS_DIR, SEED
from src.data import build_datasets, count_images
from src.model import build_model, compile_model, unfreeze_for_finetuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PlantCare AI using MobileNetV2 transfer learning")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR), help="Directory for plots and reports")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img_height", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--img_width", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--epochs_head", type=int, default=8)
    parser.add_argument("--epochs_finetune", type=int, default=8)
    parser.add_argument("--fine_tune_last_n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_class_names(class_names: list[str], models_dir: Path) -> Path:
    class_file = models_dir / "class_names.json"
    with class_file.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    return class_file


def get_callbacks(model_path: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
        ),
    ]


def merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for metric, values in history.history.items():
            merged.setdefault(metric, []).extend(values)
    return merged


def plot_history(history: dict[str, list[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.get("accuracy", []), label="train_accuracy")
    axes[0].plot(history.get("val_accuracy", []), label="val_accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("loss", []), label="train_loss")
    axes[1].plot(history.get("val_loss", []), label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_sample_images(train_ds: tf.data.Dataset, class_names: list[str], output_path: Path) -> None:
    images, labels = next(iter(train_ds.take(1)))
    fig = plt.figure(figsize=(12, 8))

    for i in range(min(9, len(images))):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(class_names[int(labels[i])], fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_class_distribution(train_ds: tf.data.Dataset, class_names: list[str], output_path: Path) -> None:
    counts = np.zeros(len(class_names), dtype=np.int64)
    for _, labels in train_ds:
        batch_counts = np.bincount(labels.numpy(), minlength=len(class_names))
        counts[: len(batch_counts)] += batch_counts

    distribution_df = pd.DataFrame({"class": class_names, "count": counts})
    distribution_df = distribution_df.sort_values("count", ascending=False)

    fig = plt.figure(figsize=(12, max(6, len(class_names) * 0.15)))
    sns.barplot(data=distribution_df, x="count", y="class", orient="h")
    plt.title("Training Class Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    distribution_df.to_csv(output_path.with_suffix(".csv"), index=False)


def evaluate_model(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    class_names: list[str],
    output_dir: Path,
) -> None:
    metric_values = model.evaluate(val_ds, verbose=1)
    metric_names = model.metrics_names
    metrics = {name: float(value) for name, value in zip(metric_names, metric_values)}

    with (output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    image_size = (args.img_height, args.img_width)

    print(f"[INFO] Counting images in: {args.data_dir}")
    print(f"[INFO] Total images found: {count_images(args.data_dir)}")

    train_ds, val_ds, class_names = build_datasets(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        seed=args.seed,
    )

    print(f"[INFO] Classes detected ({len(class_names)}):")
    print("\n".join(class_names))

    class_file = save_class_names(class_names, models_dir)
    print(f"[INFO] Saved class names to: {class_file}")

    save_sample_images(train_ds, class_names, results_dir / "sample_images.png")
    save_class_distribution(train_ds, class_names, results_dir / "class_distribution.png")

    model, base_model = build_model(num_classes=len(class_names), input_shape=(*image_size, 3))

    model_path = models_dir / "best_model.keras"
    callbacks = get_callbacks(model_path)

    compile_model(model, learning_rate=1e-3)
    print("[INFO] Starting head training...")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=callbacks,
        verbose=1,
    )

    print("[INFO] Starting fine-tuning...")
    unfreeze_for_finetuning(base_model, fine_tune_last_n=args.fine_tune_last_n)
    compile_model(model, learning_rate=1e-5)
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        callbacks=callbacks,
        verbose=1,
    )

    merged_history = merge_histories(history_head, history_finetune)
    history_df = pd.DataFrame(merged_history)
    history_df.to_csv(results_dir / "training_history.csv", index=False)
    plot_history(merged_history, results_dir / "training_curves.png")

    best_model = tf.keras.models.load_model(model_path)
    evaluate_model(best_model, val_ds, class_names, results_dir)
    print(f"[INFO] Training complete. Best model saved at: {model_path}")


if __name__ == "__main__":
    main()
