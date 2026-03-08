from __future__ import annotations

from pathlib import Path

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def _resolve_split_dirs(data_dir: Path) -> tuple[Path | None, Path | None]:
    train_dir = data_dir / "train"
    if not train_dir.exists():
        return None, None

    for valid_name in ("valid", "val", "validation"):
        valid_dir = data_dir / valid_name
        if valid_dir.exists():
            return train_dir, valid_dir

    return train_dir, None


def build_datasets(
    data_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Build train and validation datasets from directory.

    Supported formats:
    1) root/class_name/*.jpg (uses validation_split)
    2) root/train/class_name/*.jpg + root/valid/class_name/*.jpg
    """
    data_root = Path(data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    train_dir, valid_dir = _resolve_split_dirs(data_root)

    if train_dir and valid_dir:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            valid_dir,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_root,
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_root,
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
        )

    class_names = list(train_ds.class_names)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names


def count_images(data_dir: str | Path) -> int:
    data_root = Path(data_dir)
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    total = 0
    for pattern in patterns:
        total += len(list(data_root.rglob(pattern)))
    return total
