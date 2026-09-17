#!/usr/bin/env python3
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import pandas as pd
import tensorflow as tf

IMAGE_SIZE = (32, 32)
NUM_CLASSES = 43
AUTOTUNE = tf.data.AUTOTUNE


def _read_image(path: tf.Tensor, label: tf.Tensor):
    data = tf.io.read_file(path)
    image = tf.io.decode_image(
        data, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return image, label


def make_keras_dataset(
    csv_path: str | Path,
    data_root: str | Path,
    training: bool,
    batch_size: int = 32,
    local_epochs: int = 1,
    seed: int = 20260720,
):
    df = pd.read_csv(csv_path)
    required = {"Path", "ClassId"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 必须包含 {required}")

    labels = df["ClassId"].astype("int32")
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError("ClassId 必须位于 0~42")

    root = Path(data_root).resolve()
    paths = [
        str(root.joinpath(*str(p).replace("\\", "/").split("/")))
        for p in df["Path"]
    ]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels.to_numpy()))
    if training:
        ds = ds.shuffle(
            min(len(df), 4096),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        ds = ds.repeat(local_epochs)

    ds = ds.map(
        _read_image,
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds.prefetch(AUTOTUNE)


def make_tff_dataset(*args, **kwargs):
    ds = make_keras_dataset(*args, **kwargs)
    return ds.map(
        lambda image, label: OrderedDict(x=image, y=label),
        num_parallel_calls=AUTOTUNE,
    )


def build_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),

        tf.keras.layers.Conv2D(
            32,
            kernel_size=3,
            padding="same",
            activation="relu",
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64,
            kernel_size=3,
            padding="same",
            activation="relu",
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128,
            kernel_size=3,
            padding="same",
            activation="relu",
        ),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dropout(0.25),

        # 43 类分类 logits
        tf.keras.layers.Dense(43),
    ])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    ds = make_tff_dataset(
        args.csv,
        args.data_root,
        training=False,
    )
    batch = next(iter(ds))
    print("x:", batch["x"].shape, batch["x"].dtype)
    print("y:", batch["y"].shape, batch["y"].dtype)
    print("element_spec:", ds.element_spec)

    model = build_cnn()
    logits = model(batch["x"])
    print("logits:", logits.shape)