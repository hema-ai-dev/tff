#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from gtsrb_data_pipeline import NUM_CLASSES, build_cnn, make_keras_dataset


def parse_args():
    p = argparse.ArgumentParser("GTSRB Local-only baseline")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--clients", nargs="+", default=["all"])
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


def resolve_clients(raw, clients_dir: Path):
    available = sorted(
        int(p.stem.split("_")[1])
        for p in clients_dir.glob("client_*_train.csv")
    )
    if not available:
        raise FileNotFoundError(f"未找到客户端 CSV：{clients_dir}")
    if [x.lower() for x in raw] == ["all"]:
        return available
    ids = sorted({int(x) for x in raw})
    missing = sorted(set(ids) - set(available))
    if missing:
        raise ValueError(f"不存在的客户端：{missing}")
    return ids


def check_manifest(df: pd.DataFrame, name: str):
    required = {"Path", "ClassId"}
    if not required.issubset(df.columns):
        raise ValueError(f"{name} 必须包含 Path 和 ClassId")
    classes = sorted(df["ClassId"].astype(int).unique().tolist())
    if classes != list(range(NUM_CLASSES)):
        raise ValueError(f"{name} 未完整包含 0~{NUM_CLASSES - 1} 类")


def stratified_split(df: pd.DataFrame, ratio: float, seed: int):
    train_parts, val_parts = [], []
    for class_id, group in df.groupby("ClassId", sort=True):
        group = group.sample(frac=1, random_state=seed + int(class_id))
        n_val = max(1, int(round(len(group) * ratio)))
        n_val = min(n_val, len(group) - 1)
        val_parts.append(group.iloc[:n_val])
        train_parts.append(group.iloc[n_val:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed + 1).reset_index(drop=True)

    key = "sample_id" if "sample_id" in df.columns else "Path"
    if set(train_df[key]).intersection(set(val_df[key])):
        raise RuntimeError("本地训练集与验证集存在重叠")
    return train_df, val_df


def make_model(lr: float):
    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


class SaveState(tf.keras.callbacks.Callback):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        payload = {
            "completed_epochs": int(epoch + 1),
            "last_logs": {k: float(v) for k, v in (logs or {}).items()},
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def callbacks(out: Path, resume: bool):
    ckpt = out / "checkpoints"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt / "best.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt / "last.h5"),
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(
            str(out / "metrics.csv"),
            append=resume,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(out / "tensorboard")
        ),
        SaveState(out / "run_state.json"),
    ]


def load_completed_epochs(path: Path):
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8"))["completed_epochs"])


def save_plots(metrics_csv: Path, out: Path, client_id: int):
    df = pd.read_csv(metrics_csv)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Local-only Client {client_id} Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["accuracy"], label="train_accuracy")
    plt.plot(df["epoch"], df["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Local-only Client {client_id} Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "accuracy_curve.png", dpi=180)
    plt.close(fig)


def predict_all(model, dataset):
    true_parts, pred_parts = [], []
    for images, labels in dataset:
        logits = model(images, training=False)
        pred = tf.argmax(logits, axis=1, output_type=tf.int32)
        true_parts.append(labels.numpy())
        pred_parts.append(pred.numpy())
    return np.concatenate(true_parts), np.concatenate(pred_parts)


def save_class_metrics(y_true, y_pred, out: Path):
    cm = tf.math.confusion_matrix(
        y_true, y_pred, num_classes=NUM_CLASSES, dtype=tf.int64
    ).numpy()
    pd.DataFrame(
        cm,
        index=[f"true_{i}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
    ).to_csv(out / "confusion_matrix.csv")

    totals = cm.sum(axis=1)
    correct = np.diag(cm)
    per_class = np.divide(
        correct,
        totals,
        out=np.zeros_like(correct, dtype=float),
        where=totals != 0,
    )
    pd.DataFrame({
        "class_id": np.arange(NUM_CLASSES),
        "num_test_samples": totals,
        "num_correct": correct,
        "class_accuracy": per_class,
    }).to_csv(out / "per_class_accuracy.csv", index=False)

    return float((y_true == y_pred).mean()), float(per_class.mean())


def prepare_manifests(
    client_csv: Path,
    global_test_csv: Path,
    manifest_dir: Path,
    val_ratio: float,
    seed: int,
    resume: bool,
):
    train_csv = manifest_dir / "train.csv"
    val_csv = manifest_dir / "val.csv"
    test_csv = manifest_dir / "test.csv"

    if resume:
        for p in (train_csv, val_csv, test_csv):
            if not p.exists():
                raise FileNotFoundError(f"恢复训练缺少：{p}")
        return train_csv, val_csv, test_csv

    client_df = pd.read_csv(client_csv)
    test_df = pd.read_csv(global_test_csv)
    check_manifest(client_df, client_csv.name)
    check_manifest(test_df, global_test_csv.name)
    train_df, val_df = stratified_split(client_df, val_ratio, seed)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    return train_csv, val_csv, test_csv


def train_client(client_id: int, args, data_root, split_root, output_root):
    client_csv = split_root / "clients" / f"client_{client_id}_train.csv"
    test_source = split_root / "global_test.csv"
    out = output_root / f"client_{client_id}"

    if args.overwrite and out.exists():
        shutil.rmtree(out)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} 非空；使用 --overwrite 或 --resume")
    if args.resume and not out.exists():
        raise FileNotFoundError(f"恢复目录不存在：{out}")

    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    train_csv, val_csv, test_csv = prepare_manifests(
        client_csv, test_source, out / "manifests",
        args.val_ratio, args.seed, args.resume
    )

    train_count = len(pd.read_csv(train_csv))
    val_count = len(pd.read_csv(val_csv))
    source_count = len(pd.read_csv(client_csv))
    test_count = len(pd.read_csv(test_csv))

    print("\n" + "=" * 72)
    print(f"Local-only client_{client_id}")
    print("=" * 72)
    print(f"本地原始样本：{source_count}")
    print(f"本地训练样本：{train_count}")
    print(f"本地验证样本：{val_count}")
    print(f"全局测试样本：{test_count}")

    train_ds = make_keras_dataset(
        train_csv, data_root, training=True,
        batch_size=args.batch_size, local_epochs=1, seed=args.seed
    )
    val_ds = make_keras_dataset(
        val_csv, data_root, training=False,
        batch_size=args.batch_size, seed=args.seed
    )
    test_ds = make_keras_dataset(
        test_csv, data_root, training=False,
        batch_size=args.batch_size, seed=args.seed
    )

    last_path = out / "checkpoints" / "last.h5"
    if args.resume:
        model = tf.keras.models.load_model(last_path)
        initial_epoch = load_completed_epochs(out / "run_state.json")
        print(f"[RESUME] initial_epoch={initial_epoch}")
    else:
        # 每个客户端重置为同一随机种子，保证初始权重一致。
        set_seed(args.seed)
        model = make_model(args.learning_rate)
        initial_epoch = 0

    if initial_epoch < args.epochs:
        start = time.perf_counter()
        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=args.epochs,
            callbacks=callbacks(out, args.resume),
            verbose=1,
        )
        (out / "training_time_seconds.txt").write_text(
            f"{time.perf_counter() - start:.6f}\n",
            encoding="utf-8",
        )

    best_path = out / "checkpoints" / "best.h5"
    selected = best_path if best_path.exists() else last_path
    best_model = tf.keras.models.load_model(selected)
    eval_result = best_model.evaluate(test_ds, return_dict=True, verbose=1)

    y_true, y_pred = predict_all(best_model, test_ds)
    overall, macro = save_class_metrics(y_true, y_pred, out)

    history = pd.read_csv(out / "metrics.csv")
    best_row = history.loc[history["val_accuracy"].idxmax()]

    result = {
        "client_id": client_id,
        "source_local_samples": source_count,
        "train_samples": train_count,
        "val_samples": val_count,
        "test_samples": test_count,
        "best_epoch": int(best_row["epoch"]) + 1,
        "best_val_accuracy": float(best_row["val_accuracy"]),
        "test_loss": float(eval_result["loss"]),
        "test_accuracy": float(eval_result["accuracy"]),
        "overall_accuracy_from_predictions": overall,
        "macro_class_accuracy": macro,
        "selected_checkpoint": str(selected),
    }
    (out / "test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_plots(out / "metrics.csv", out / "plots", client_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume 与 --overwrite 不能同时使用")

    data_root = args.data_root.expanduser().resolve()
    split_root = args.split_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    clients_dir = split_root / "clients"
    client_ids = resolve_clients(args.clients, clients_dir)

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow: {tf.__version__}")
    print(f"Clients: {client_ids}")
    print("Aggregation: none")
    print("Initialization: same seed for all clients")

    results = [
        train_client(
            client_id, args, data_root, split_root, output_root
        )
        for client_id in client_ids
    ]

    df = pd.DataFrame(results).sort_values("client_id")
    df.to_csv(output_root / "local_only_summary.csv", index=False)

    accuracy = df["test_accuracy"].to_numpy(float)
    macro = df["macro_class_accuracy"].to_numpy(float)
    aggregate = {
        "num_completed_clients": len(df),
        "client_ids": df["client_id"].astype(int).tolist(),
        "mean_test_accuracy": float(accuracy.mean()),
        "std_test_accuracy_population": float(accuracy.std(ddof=0)),
        "min_test_accuracy": float(accuracy.min()),
        "max_test_accuracy": float(accuracy.max()),
        "mean_macro_class_accuracy": float(macro.mean()),
        "std_macro_class_accuracy_population": float(macro.std(ddof=0)),
    }
    if len(df) > 1:
        aggregate["std_test_accuracy_sample"] = float(
            accuracy.std(ddof=1)
        )
        aggregate["std_macro_class_accuracy_sample"] = float(
            macro.std(ddof=1)
        )

    (output_root / "local_only_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nLocal-only summary:")
    print(df[[
        "client_id", "source_local_samples", "best_epoch",
        "best_val_accuracy", "test_accuracy", "macro_class_accuracy"
    ]].to_string(index=False))
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
