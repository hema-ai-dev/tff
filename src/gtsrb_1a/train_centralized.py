#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 1A：GTSRB Centralized 43 类分类基线。"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from gtsrb_data_pipeline import NUM_CLASSES, build_cnn, make_keras_dataset


def parse_args():
    p = argparse.ArgumentParser(description="GTSRB Centralized baseline")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def prepare_output(path: Path, resume: bool, overwrite: bool):
    if resume and overwrite:
        raise ValueError("--resume 与 --overwrite 不能同时使用")
    if overwrite and path.exists():
        shutil.rmtree(path)
    if resume and not path.exists():
        raise FileNotFoundError(f"恢复目录不存在：{path}")
    if not resume and path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"输出目录非空：{path}\n重新训练加 --overwrite；恢复加 --resume。"
        )
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (path / "manifests").mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(parents=True, exist_ok=True)


def check_manifest(df: pd.DataFrame, name: str):
    required = {"Path", "ClassId"}
    if not required.issubset(df.columns):
        raise ValueError(f"{name} 必须包含 Path 和 ClassId")
    if df.empty:
        raise ValueError(f"{name} 为空")
    labels = df["ClassId"].astype(int)
    expected = list(range(NUM_CLASSES))
    actual = sorted(labels.unique().tolist())
    if actual != expected:
        raise ValueError(f"{name} 类别不完整：{actual}")


def stratified_split(df: pd.DataFrame, val_ratio: float, seed: int):
    train_parts, val_parts = [], []
    for class_id, group in df.groupby("ClassId", sort=True):
        group = group.sample(frac=1.0, random_state=seed + int(class_id))
        n_val = max(1, int(round(len(group) * val_ratio)))
        n_val = min(n_val, len(group) - 1)
        val_parts.append(group.iloc[:n_val])
        train_parts.append(group.iloc[n_val:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

    key = "sample_id" if "sample_id" in df.columns else "Path"
    if set(train_df[key]).intersection(set(val_df[key])):
        raise RuntimeError("训练集与验证集发生重叠")
    if len(train_df) + len(val_df) != len(df):
        raise RuntimeError("训练/验证样本总数不一致")
    return train_df, val_df


def make_manifests(split_root: Path, out_dir: Path, val_ratio: float, seed: int, resume: bool):
    train_out = out_dir / "manifests" / "train.csv"
    val_out = out_dir / "manifests" / "val.csv"
    test_out = out_dir / "manifests" / "test.csv"

    if resume:
        for p in (train_out, val_out, test_out):
            if not p.is_file():
                raise FileNotFoundError(f"恢复所需文件不存在：{p}")
        return train_out, val_out, test_out

    full_train = pd.read_csv(split_root / "centralized_train.csv")
    test_df = pd.read_csv(split_root / "global_test.csv")
    check_manifest(full_train, "centralized_train.csv")
    check_manifest(test_df, "global_test.csv")

    train_df, val_df = stratified_split(full_train, val_ratio, seed)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)
    return train_out, val_out, test_out


def compile_model(lr: float):
    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


class SaveEpoch(tf.keras.callbacks.Callback):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        payload = {
            "completed_epochs": int(epoch + 1),
            "last_logs": {k: float(v) for k, v in (logs or {}).items()},
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def completed_epochs(path: Path):
    if not path.is_file():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8")).get("completed_epochs", 0))


def callbacks(out_dir: Path, resume: bool):
    ckpt = out_dir / "checkpoints"
    # TensorFlow/Keras 2.14.1 的 ModelCheckpoint 在部分环境中
    # 保存原生 .keras 格式时会传入不兼容的 options 参数。
    # 使用完整模型 HDF5 格式可避开该兼容性问题，并保留模型结构、
    # 权重和优化器状态，便于 --resume。
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt / "best.h5"), monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt / "last.h5"), save_best_only=False, verbose=0
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "metrics.csv"), append=resume),
        tf.keras.callbacks.TensorBoard(log_dir=str(out_dir / "tensorboard")),
        SaveEpoch(out_dir / "run_state.json"),
    ]


def save_curves(metrics_csv: Path, plot_dir: Path):
    df = pd.read_csv(metrics_csv)
    if {"loss", "val_loss"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["loss"], label="train_loss")
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Centralized GTSRB Loss")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(plot_dir / "loss_curve.png", dpi=180); plt.close()
    if {"accuracy", "val_accuracy"}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["accuracy"], label="train_accuracy")
        plt.plot(df["epoch"], df["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Centralized GTSRB Accuracy")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(plot_dir / "accuracy_curve.png", dpi=180); plt.close()


def predict_all(model, dataset):
    true_parts, pred_parts = [], []
    for images, labels in dataset:
        logits = model(images, training=False)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        true_parts.append(labels.numpy())
        pred_parts.append(preds.numpy())
    return np.concatenate(true_parts), np.concatenate(pred_parts)


def save_class_metrics(y_true, y_pred, out_dir: Path):
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES).numpy()
    pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
    ).to_csv(out_dir / "confusion_matrix.csv")

    totals = cm.sum(axis=1)
    correct = np.diag(cm)
    acc = np.divide(correct, totals, out=np.zeros(NUM_CLASSES, dtype=float), where=totals != 0)
    pd.DataFrame({
        "class_id": np.arange(NUM_CLASSES),
        "num_test_samples": totals,
        "num_correct": correct,
        "class_accuracy": acc,
    }).to_csv(out_dir / "per_class_accuracy.csv", index=False)
    return float((y_true == y_pred).mean()), float(acc.mean())


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs、batch-size、learning-rate 参数非法")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio 必须位于 (0,1)")

    data_root = args.data_root.expanduser().resolve()
    split_root = args.split_root.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for p in (data_root, split_root, split_root / "centralized_train.csv", split_root / "global_test.csv"):
        if not p.exists():
            raise FileNotFoundError(f"必要路径不存在：{p}")

    prepare_output(out_dir, args.resume, args.overwrite)
    set_seed(args.seed)
    train_csv, val_csv, test_csv = make_manifests(
        split_root, out_dir, args.val_ratio, args.seed, args.resume
    )

    counts = {
        "train": len(pd.read_csv(train_csv)),
        "val": len(pd.read_csv(val_csv)),
        "test": len(pd.read_csv(test_csv)),
    }
    print("=" * 72)
    print("Stage 1A - Centralized GTSRB 43-class Classification")
    print("=" * 72)
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("Samples:", counts)
    print("Output:", out_dir)

    train_ds = make_keras_dataset(train_csv, data_root, True, args.batch_size, 1, args.seed)
    val_ds = make_keras_dataset(val_csv, data_root, False, args.batch_size, 1, args.seed)
    test_ds = make_keras_dataset(test_csv, data_root, False, args.batch_size, 1, args.seed)

    state_path = out_dir / "run_state.json"
    last_path = out_dir / "checkpoints" / "last.h5"
    best_path = out_dir / "checkpoints" / "best.h5"

    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint：{last_path}")
        model = tf.keras.models.load_model(last_path)
        initial_epoch = completed_epochs(state_path)
        print(f"[RESUME] initial_epoch={initial_epoch}")
    else:
        model = compile_model(args.learning_rate)
        initial_epoch = 0

    if initial_epoch < args.epochs:
        model.summary()
        started = time.perf_counter()
        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=args.epochs,
            callbacks=callbacks(out_dir, args.resume),
        )
        (out_dir / "training_time_seconds.txt").write_text(
            f"{time.perf_counter() - started:.6f}\n", encoding="utf-8"
        )
    else:
        print("目标 epoch 已经完成，跳过训练并直接评估。")

    selected = best_path if best_path.is_file() else last_path
    eval_model = tf.keras.models.load_model(selected)
    test_result = {k: float(v) for k, v in eval_model.evaluate(test_ds, return_dict=True).items()}
    y_true, y_pred = predict_all(eval_model, test_ds)
    overall, macro = save_class_metrics(y_true, y_pred, out_dir)
    test_result.update({
        "overall_accuracy_from_predictions": overall,
        "macro_class_accuracy": macro,
        "selected_checkpoint": str(selected),
        "num_test_samples": int(len(y_true)),
    })
    (out_dir / "test_metrics.json").write_text(
        json.dumps(test_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if (out_dir / "metrics.csv").is_file():
        save_curves(out_dir / "metrics.csv", out_dir / "plots")

    config = {
        "data_root": str(data_root),
        "split_root": str(split_root),
        "output_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_classes": NUM_CLASSES,
        "samples": counts,
        "tensorflow_version": tf.__version__,
        "python_version": sys.version,
        "model_output": "43 logits",
        "loss": "SparseCategoricalCrossentropy(from_logits=True)",
        "metric": "SparseCategoricalAccuracy",
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nCentralized 基线完成：")
    print(json.dumps(test_result, ensure_ascii=False, indent=2))
    print("metrics.csv:", out_dir / "metrics.csv")
    print("best checkpoint:", best_path)
    print("TensorBoard:", out_dir / "tensorboard")


if __name__ == "__main__":
    main()
