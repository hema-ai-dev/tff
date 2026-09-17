#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1C-A1: GTSRB + TFF 0.87.0 加权 FedProx

核心设置
--------
- 5 个客户端每轮全部参与；
- 每个客户端仅访问自己的训练清单；
- 共享同一个 43 类 CNN；
- 客户端本地训练后上传模型更新；
- 服务器按客户端实际训练样本数加权聚合；
- 服务器优化器使用 SGDM(lr=1.0)，对应原始 FedAvg 更新；
- 每轮记录 TFF 训练指标和全局验证指标；
- 官方 Test.csv 只在训练结束后用于最终测试；
- 保存 TFF 全局 ModelWeights（NPZ），支持按通信轮次恢复。

冒烟测试
--------
python src/gtsrb_1c/fedprox/train_fedprox_stage1c.py \
  --data-root data/gtsrb \
  --partition-dir data/gtsrb/federated_dirichlet_alpha05_5clients_seed20260720 \
  --output-dir outputs/stage1c/fedprox/fedprox_alpha05_mu001_smoke_seed20260720 \
  --experiment-name fedprox_alpha05_mu001_smoke_seed20260720 \
  --rounds 3 \
  --local-epochs 1 \
  --batch-size 64 \
  --client-optimizer adam \
  --client-learning-rate 0.001 \
  --server-learning-rate 1.0 \
  --proximal-strength 0.001 \
  --seed 20260720 \
  --overwrite

正式训练
--------
python src/gtsrb_1c/fedprox/train_fedprox_stage1c.py \
  --data-root data/gtsrb \
  --partition-dir data/gtsrb/federated_dirichlet_alpha05_5clients_seed20260720 \
  --output-dir outputs/stage1c/fedprox/fedprox_alpha05_mu001_seed20260720 \
  --experiment-name fedprox_alpha05_mu001_seed20260720 \
  --rounds 160 \
  --local-epochs 1 \
  --batch-size 64 \
  --client-optimizer adam \
  --client-learning-rate 0.001 \
  --server-learning-rate 1.0 \
  --proximal-strength 0.001 \
  --seed 20260720 \
  --overwrite

恢复训练
--------
将 --overwrite 改成 --resume，并保持其他训练参数不变。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

# train_fedavg_stage1b.py 位于 src/gtsrb_1b，而数据流水线仍复用
# Stage 1A 中已经验证过的 src/gtsrb_1a/gtsrb_data_pipeline.py。
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
STAGE1A_DIR = SRC_DIR / "gtsrb_1a"
STAGE1B_DIR = SRC_DIR / "gtsrb_1b"

for import_dir in (CURRENT_DIR, STAGE1A_DIR, STAGE1B_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from gtsrb_data_pipeline import (
    NUM_CLASSES,
    build_cnn,
    make_keras_dataset,
    make_tff_dataset,
)
from stage1b_partition_loader import (
    describe_partition,
    load_partition_csvs,
)


# ---------------------------------------------------------------------------
# 参数和环境
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GTSRB 5 客户端加权 FedProx 分类实验。"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
    "--partition-dir",
    type=Path,
    required=True,
    help=(
        "Directory containing clients/client_0_train.csv ... "
        "client_4_train.csv and split_metadata.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
    "--experiment-name",
    type=str,
    default=None,
    help="Optional experiment name recorded in config.",
    )

    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument(
        "--client-optimizer",
        choices=("adam", "sgd"),
        default="adam",
    )
    parser.add_argument(
        "--client-learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--client-momentum",
        type=float,
        default=0.0,
        help="仅在 client-optimizer=sgd 时使用。",
    )
    parser.add_argument(
        "--server-learning-rate",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--proximal-strength",
        type=float,
        default=0.0,
        help=(
            "FedProx proximal strength mu. Must be nonnegative. "
            "mu=0.0 is the FedAvg degeneration check."
        ),
    )

    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--max-concurrent-calls",
        type=int,
        default=1,
        help="CPU 环境建议保持 1，避免本地并发占用过多内存。",
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.resume and args.overwrite:
        raise ValueError("--resume 与 --overwrite 不能同时使用。")
    if args.rounds < 1:
        raise ValueError("--rounds 必须至少为 1。")
    if args.local_epochs < 1:
        raise ValueError("--local-epochs 必须至少为 1。")
    if args.batch_size < 1:
        raise ValueError("--batch-size 必须至少为 1。")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio 必须位于 (0, 1)。")
    if args.client_learning_rate <= 0:
        raise ValueError("--client-learning-rate 必须大于 0。")
    if args.server_learning_rate <= 0:
        raise ValueError("--server-learning-rate 必须大于 0。")
    if args.proximal_strength < 0:
        raise ValueError("--proximal-strength 必须大于或等于 0。")
    if not 0.0 <= args.client_momentum < 1.0:
        raise ValueError("--client-momentum 必须位于 [0, 1)。")
    if args.eval_every < 1:
        raise ValueError("--eval-every 必须至少为 1。")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every 必须至少为 1。")


def set_global_seed(seed: int) -> None:
    """设置可复现随机种子。

    注意：
    TFF 会在独立 TensorFlow graph 中追踪 model_fn。TensorFlow 2.14
    开启全局 op determinism 后，Keras Dropout 在该追踪 graph 内可能
    被判定为“没有设置随机种子”，从而直接抛出 RuntimeError。

    因此 FedAvg 进程不在这里强制 enable_op_determinism()；仍保留
    Python、NumPy 和 TensorFlow 的固定随机种子。model_fn 内还会再次
    设置 graph-level seed。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_tff_context(max_concurrent_calls: int) -> None:
    try:
        tff.backends.native.set_sync_local_cpp_execution_context(
            max_concurrent_computation_calls=max_concurrent_calls
        )
        print(
            "[INFO] TFF local C++ context: "
            f"max_concurrent_calls={max_concurrent_calls}"
        )
    except TypeError:
        tff.backends.native.set_sync_local_cpp_execution_context()
        print("[INFO] TFF local C++ context enabled.")
    except AttributeError:
        print("[WARN] 使用 TFF 默认执行上下文。")


def prepare_output_dir(
    output_dir: Path,
    resume: bool,
    overwrite: bool,
) -> None:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    if resume:
        if not output_dir.exists():
            raise FileNotFoundError(
                f"恢复目录不存在：{output_dir}"
            )
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"输出目录非空：{output_dir}\n"
            "重新训练请加 --overwrite；恢复请加 --resume。"
        )

    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "global_models").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 数据清单
# ---------------------------------------------------------------------------

def validate_manifest(
    df: pd.DataFrame,
    name: str,
    *,
    require_all_classes: bool = False,
) -> None:
    """验证训练/验证/测试清单。

    Non-IID 客户端本来就可能缺少部分标签，因此客户端清单只检查标签
    是否位于 0~42；只有全局验证集和官方测试集才要求覆盖全部 43 类。
    """
    required = {"Path", "ClassId"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} 缺少必要列：{sorted(missing)}")
    if df.empty:
        raise ValueError(f"{name} 为空。")
    if df["Path"].isna().any() or df["ClassId"].isna().any():
        raise ValueError(f"{name} 包含空的 Path 或 ClassId。")
    if df["Path"].astype(str).duplicated().any():
        duplicate_count = int(df["Path"].astype(str).duplicated().sum())
        raise ValueError(f"{name} 包含 {duplicate_count} 个重复 Path。")

    labels = df["ClassId"].astype(int)
    invalid = labels[(labels < 0) | (labels >= NUM_CLASSES)]
    if not invalid.empty:
        raise ValueError(
            f"{name} 含有超出 0~{NUM_CLASSES - 1} 的标签："
            f"{sorted(invalid.unique().tolist())}"
        )

    if require_all_classes:
        classes = sorted(labels.unique().tolist())
        expected = list(range(NUM_CLASSES))
        if classes != expected:
            raise ValueError(
                f"{name} 未完整包含 0~{NUM_CLASSES - 1} 类。\n"
                f"实际类别：{classes}"
            )


def stratified_train_val_split(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    与 Local-only 脚本使用相同的分层拆分规则。
    """
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []

    for class_id, group in df.groupby("ClassId", sort=True):
        group = group.sample(
            frac=1.0,
            random_state=seed + int(class_id),
        )
        n_val = max(1, int(round(len(group) * val_ratio)))
        n_val = min(n_val, len(group) - 1)

        val_parts.append(group.iloc[:n_val].copy())
        train_parts.append(group.iloc[n_val:].copy())

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)

    train_df = train_df.sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    val_df = val_df.sample(
        frac=1.0, random_state=seed + 1
    ).reset_index(drop=True)

    key = "sample_id" if "sample_id" in df.columns else "Path"
    overlap = set(train_df[key]).intersection(set(val_df[key]))
    if overlap:
        raise RuntimeError(
            f"本地训练集和验证集重叠：{len(overlap)} 个样本。"
        )
    if len(train_df) + len(val_df) != len(df):
        raise RuntimeError("训练/验证拆分后的总样本数不正确。")

    return train_df, val_df


def create_or_load_manifests(
    partition_dir: Path,
    data_root: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    resume: bool,
) -> tuple[list[Path], Path, Path, list[int]]:
    manifest_dir = output_dir / "manifests"
    global_val_csv = manifest_dir / "global_val.csv"
    global_test_csv = manifest_dir / "global_test.csv"
    metadata_json = manifest_dir / "manifest_metadata.json"

    # 先通过 Stage 1B loader 做总样本数、客户端数量、重复路径等审计。
    client_csv_map = load_partition_csvs(
        partition_dir=partition_dir,
        num_clients=5,
        expected_total_samples=39209,
        expected_num_classes=NUM_CLASSES,
    )
    source_client_csvs = [
        client_csv_map[f"client_{client_id}"]
        for client_id in range(5)
    ]

    train_csvs = [
        manifest_dir / f"client_{i}_train.csv"
        for i in range(len(source_client_csvs))
    ]
    val_csvs = [
        manifest_dir / f"client_{i}_val.csv"
        for i in range(len(source_client_csvs))
    ]

    if resume:
        required = train_csvs + val_csvs + [
            global_val_csv,
            global_test_csv,
            metadata_json,
        ]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(
                    f"恢复训练缺少 manifest：{path}"
                )
        metadata = json.loads(
            metadata_json.read_text(encoding="utf-8")
        )
        saved_partition_dir = Path(metadata["source_partition_dir"]).resolve()
        if saved_partition_dir != partition_dir.resolve():
            raise ValueError(
                "恢复时传入的 --partition-dir 与原实验不一致。\n"
                f"原实验：{saved_partition_dir}\n"
                f"当前参数：{partition_dir.resolve()}"
            )
        client_train_sizes = [
            int(x) for x in metadata["client_train_sizes"]
        ]
        return train_csvs, global_val_csv, global_test_csv, client_train_sizes

    all_val_parts: list[pd.DataFrame] = []
    client_train_sizes: list[int] = []
    client_val_sizes: list[int] = []

    for client_id, source_csv in enumerate(source_client_csvs):
        source_df = pd.read_csv(source_csv)
        validate_manifest(
            source_df,
            source_csv.name,
            require_all_classes=False,
        )

        train_df, val_df = stratified_train_val_split(
            source_df,
            val_ratio=val_ratio,
            seed=seed,
        )
        validate_manifest(
            train_df,
            f"client_{client_id}_train.csv",
            require_all_classes=False,
        )
        validate_manifest(
            val_df,
            f"client_{client_id}_val.csv",
            require_all_classes=False,
        )
        train_df.to_csv(train_csvs[client_id], index=False)
        val_df.to_csv(val_csvs[client_id], index=False)

        all_val_parts.append(val_df)
        client_train_sizes.append(int(len(train_df)))
        client_val_sizes.append(int(len(val_df)))

    global_val_df = pd.concat(
        all_val_parts, ignore_index=True
    ).sample(
        frac=1.0,
        random_state=seed + 999,
    ).reset_index(drop=True)
    validate_manifest(
        global_val_df,
        "global_val.csv",
        require_all_classes=True,
    )
    global_val_df.to_csv(global_val_csv, index=False)

    # Dirichlet 划分目录不保存 global_test.csv；官方测试集始终来自
    # 原始 GTSRB data-root/Test.csv，且从不参与客户端划分。
    official_test_path = data_root / "Test.csv"
    if not official_test_path.is_file():
        raise FileNotFoundError(f"找不到官方测试清单：{official_test_path}")
    official_test = pd.read_csv(official_test_path)
    validate_manifest(
        official_test,
        "Test.csv",
        require_all_classes=True,
    )
    official_test.to_csv(global_test_csv, index=False)

    metadata = {
        "source_partition_dir": str(partition_dir.resolve()),
        "official_test_csv": str(official_test_path.resolve()),
        "num_clients": len(train_csvs),
        "client_train_sizes": client_train_sizes,
        "client_val_sizes": client_val_sizes,
        "global_val_samples": int(len(global_val_df)),
        "global_test_samples": int(len(official_test)),
        "val_ratio": val_ratio,
        "seed": seed,
    }
    metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return train_csvs, global_val_csv, global_test_csv, client_train_sizes


# ---------------------------------------------------------------------------
# TFF 模型和学习过程
# ---------------------------------------------------------------------------

def build_client_optimizer(args: argparse.Namespace):
    if args.client_optimizer == "adam":
        return tff.learning.optimizers.build_adam(
            learning_rate=args.client_learning_rate
        )

    momentum = (
        None if args.client_momentum == 0.0
        else args.client_momentum
    )
    return tff.learning.optimizers.build_sgdm(
        learning_rate=args.client_learning_rate,
        momentum=momentum,
    )


def build_learning_process(
    input_spec: Any,
    args: argparse.Namespace,
):
    def model_fn():
        # TFF 会在独立 graph 中多次追踪 model_fn，因此需要在 graph 内
        # 显式设置随机种子，确保 Dropout 等随机算子可正常创建。
        tf.random.set_seed(args.seed)
        keras_model = build_cnn()
        return tff.learning.models.from_keras_model(
            keras_model=keras_model,
            input_spec=input_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(
                    name="accuracy"
                )
            ],
        )

    return tff.learning.algorithms.build_weighted_fed_prox(
        model_fn=model_fn,
        proximal_strength=float(args.proximal_strength),
        client_optimizer_fn=build_client_optimizer(args),
        server_optimizer_fn=tff.learning.optimizers.build_sgdm(
            learning_rate=args.server_learning_rate
        ),
        client_weighting=tff.learning.ClientWeighting.NUM_EXAMPLES,
    )


def make_federated_train_data(
    client_train_csvs: list[Path],
    data_root: Path,
    args: argparse.Namespace,
    round_num: int,
) -> list[tf.data.Dataset]:
    """为指定通信轮次创建客户端数据。

    使用 round_num 构造轮次相关随机种子。这样从 checkpoint 恢复后，
    同一轮会得到相同的数据打乱种子，而不会从第 1 轮的 shuffle 序列重启。
    """
    datasets = []
    for client_id, csv_path in enumerate(client_train_csvs):
        dataset = make_tff_dataset(
            csv_path=csv_path,
            data_root=data_root,
            training=True,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            seed=args.seed + round_num * 1000 + client_id,
        )
        datasets.append(dataset)
    return datasets


# ---------------------------------------------------------------------------
# 指标、模型导出和 checkpoint
# ---------------------------------------------------------------------------

def scalar_value(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(()).item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value


def flatten_nested(
    value: Any,
    prefix: str = "",
) -> dict[str, Any]:
    """
    将 TFF metrics 的嵌套 OrderedDict / Struct 展平。
    """
    result: dict[str, Any] = {}

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten_nested(child, child_prefix))
        return result

    if hasattr(value, "_asdict"):
        return flatten_nested(value._asdict(), prefix)

    try:
        elements = list(tff.structure.iter_elements(value))
    except (TypeError, AttributeError):
        elements = []

    if elements:
        for index, (name, child) in enumerate(elements):
            key = name if name is not None else str(index)
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten_nested(child, child_prefix))
        return result

    result[prefix or "value"] = scalar_value(value)
    return result


def find_metric(
    flat_metrics: dict[str, Any],
    metric_name: str,
) -> float | None:
    """
    优先查找 client_work/train 下的 loss 或 accuracy。
    """
    candidates: list[tuple[int, str, float]] = []

    for key, value in flat_metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue

        normalized = key.lower()
        if normalized.split("/")[-1] != metric_name.lower():
            continue

        score = 0
        if "client_work" in normalized:
            score += 4
        if "/train/" in f"/{normalized}/":
            score += 4
        if "evaluation" in normalized:
            score -= 2
        candidates.append((score, key, number))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][2]



def normalize_model_weights(raw_weights):
    """兼容 TFF 0.87.0 中 get_model_weights 的不同返回形式。

    build_weighted_fed_avg 在当前环境中直接返回 ModelWeights。
    只有当返回值确实是 tff.structure.Struct 时，才调用
    ModelWeights.from_tff_result()。
    """
    if isinstance(raw_weights, tff.learning.models.ModelWeights):
        return raw_weights

    if (
        hasattr(raw_weights, "trainable")
        and hasattr(raw_weights, "non_trainable")
    ):
        return tff.learning.models.ModelWeights(
            trainable=raw_weights.trainable,
            non_trainable=raw_weights.non_trainable,
        )

    return tff.learning.models.ModelWeights.from_tff_result(raw_weights)


def make_evaluation_model() -> tf.keras.Model:
    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )
    return model


def assign_state_to_keras_model(
    learning_process,
    state,
    keras_model: tf.keras.Model,
) -> None:
    raw_weights = learning_process.get_model_weights(state)
    model_weights = normalize_model_weights(raw_weights)
    model_weights.assign_weights_to(keras_model)


def model_size_bytes(model: tf.keras.Model) -> int:
    return int(
        sum(
            np.prod(variable.shape.as_list())
            * variable.dtype.size
            for variable in model.weights
        )
    )


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> dict[str, float]:
    result = model.evaluate(
        dataset,
        return_dict=True,
        verbose=0,
    )
    return {key: float(value) for key, value in result.items()}


def predict_all(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []

    for images, labels in dataset:
        logits = model(images, training=False)
        predictions = tf.argmax(
            logits,
            axis=1,
            output_type=tf.int32,
        )
        true_parts.append(labels.numpy())
        pred_parts.append(predictions.numpy())

    return np.concatenate(true_parts), np.concatenate(pred_parts)


def save_classification_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    confusion = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=NUM_CLASSES,
        dtype=tf.int64,
    ).numpy()

    pd.DataFrame(
        confusion,
        index=[f"true_{i}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
    ).to_csv(output_dir / "confusion_matrix.csv")

    totals = confusion.sum(axis=1)
    correct = np.diag(confusion)
    per_class = np.divide(
        correct,
        totals,
        out=np.zeros_like(correct, dtype=float),
        where=totals != 0,
    )

    pd.DataFrame(
        {
            "class_id": np.arange(NUM_CLASSES),
            "num_test_samples": totals,
            "num_correct": correct,
            "class_accuracy": per_class,
        }
    ).to_csv(
        output_dir / "per_class_accuracy.csv",
        index=False,
    )

    return {
        "overall_accuracy_from_predictions": float(
            (y_true == y_pred).mean()
        ),
        "macro_class_accuracy": float(per_class.mean()),
    }


def append_metrics_row(
    metrics_csv: Path,
    row: dict[str, Any],
) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(
        metrics_csv,
        mode="a",
        header=not metrics_csv.exists(),
        index=False,
    )


def save_plots(metrics_csv: Path, plot_dir: Path) -> None:
    df = pd.read_csv(metrics_csv)

    if {"round", "train_loss", "val_loss"}.issubset(df.columns):
        fig = plt.figure(figsize=(8, 5))
        plt.plot(df["round"], df["train_loss"], label="train_loss")
        plt.plot(df["round"], df["val_loss"], label="val_loss")
        plt.xlabel("Communication Round")
        plt.ylabel("Loss")
        plt.title("FedProx GTSRB Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "loss_curve.png", dpi=180)
        plt.close(fig)

    if {"round", "train_accuracy", "val_accuracy"}.issubset(df.columns):
        fig = plt.figure(figsize=(8, 5))
        plt.plot(
            df["round"],
            df["train_accuracy"],
            label="train_accuracy",
        )
        plt.plot(
            df["round"],
            df["val_accuracy"],
            label="val_accuracy",
        )
        plt.xlabel("Communication Round")
        plt.ylabel("Accuracy")
        plt.title("FedProx GTSRB Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "accuracy_curve.png", dpi=180)
        plt.close(fig)


def save_model_weights_checkpoint(
    learning_process,
    state,
    checkpoint_path: Path,
) -> None:
    """将 TFF 全局模型权重保存为纯 NumPy NPZ。

    不使用 FileProgramStateManager，因为它依赖 SavedModel；在当前
    TensorFlow 2.14.1 环境中，SavedModel 序列化可能触发
    `_DictWrapper` / typing_extensions 兼容错误。

    当前实验的服务器优化器是无 momentum 的 SGDM，聚合器是无状态的
    NUM_EXAMPLES 加权平均，因此恢复全局模型权重即可恢复此 FedProx 基线。
    """
    raw_weights = learning_process.get_model_weights(state)
    model_weights = normalize_model_weights(
        raw_weights
    ).convert_variables_to_arrays()

    trainable = [np.asarray(value) for value in model_weights.trainable]
    non_trainable = [
        np.asarray(value) for value in model_weights.non_trainable
    ]

    payload: dict[str, np.ndarray] = {
        "num_trainable": np.asarray(len(trainable), dtype=np.int64),
        "num_non_trainable": np.asarray(
            len(non_trainable), dtype=np.int64
        ),
    }
    for index, value in enumerate(trainable):
        payload[f"trainable_{index:04d}"] = value
    for index, value in enumerate(non_trainable):
        payload[f"non_trainable_{index:04d}"] = value

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix(".tmp.npz")
    with temp_path.open("wb") as file:
        np.savez_compressed(file, **payload)
    temp_path.replace(checkpoint_path)


def load_model_weights_checkpoint(
    checkpoint_path: Path,
) -> tff.learning.models.ModelWeights:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到模型权重 checkpoint：{checkpoint_path}"
        )

    with np.load(checkpoint_path, allow_pickle=False) as payload:
        num_trainable = int(payload["num_trainable"])
        num_non_trainable = int(payload["num_non_trainable"])
        trainable = [
            np.array(payload[f"trainable_{index:04d}"])
            for index in range(num_trainable)
        ]
        non_trainable = [
            np.array(payload[f"non_trainable_{index:04d}"])
            for index in range(num_non_trainable)
        ]

    return tff.learning.models.ModelWeights(
        trainable=trainable,
        non_trainable=non_trainable,
    )


def save_round_state(
    state_path: Path,
    completed_round: int,
    checkpoint_path: Path,
    output_dir: Path,
) -> None:
    try:
        relative_checkpoint = checkpoint_path.relative_to(output_dir)
    except ValueError:
        relative_checkpoint = checkpoint_path

    payload = {
        "completed_round": int(completed_round),
        "weights_checkpoint": str(relative_checkpoint),
        "checkpoint_format": "TFF ModelWeights stored as NumPy NPZ",
    }
    temp_path = state_path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(state_path)


def load_round_state(
    state_path: Path,
    output_dir: Path,
) -> tuple[int, Path]:
    if not state_path.is_file():
        raise FileNotFoundError(
            f"找不到恢复状态文件：{state_path}"
        )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    completed_round = int(payload["completed_round"])
    checkpoint_path = Path(payload["weights_checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = output_dir / checkpoint_path
    return completed_round, checkpoint_path


def trim_metrics_after_round(
    metrics_csv: Path,
    completed_round: int,
) -> None:
    """删除可能在 checkpoint 提交前写入的多余指标行。"""
    if not metrics_csv.is_file():
        return
    frame = pd.read_csv(metrics_csv)
    if "round" not in frame.columns:
        return
    trimmed = frame.loc[frame["round"] <= completed_round].copy()
    trimmed.to_csv(metrics_csv, index=False)


def get_existing_best_validation(
    metrics_csv: Path,
) -> float:
    if not metrics_csv.is_file():
        return float("-inf")
    df = pd.read_csv(metrics_csv)
    if "val_accuracy" not in df.columns:
        return float("-inf")
    values = pd.to_numeric(df["val_accuracy"], errors="coerce")
    if not values.notna().any():
        return float("-inf")
    return float(values.max())


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate_args(args)

    data_root = args.data_root.expanduser().resolve()
    partition_dir = args.partition_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for path in (
        data_root,
        data_root / "Train.csv",
        data_root / "Test.csv",
        partition_dir,
        partition_dir / "clients",
        partition_dir / "split_metadata.json",
        partition_dir / "partition_manifest.csv",
    ):
        if not path.exists():
            raise FileNotFoundError(f"必要路径不存在：{path}")

    partition_metadata = describe_partition(partition_dir)

    prepare_output_dir(
        output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    set_global_seed(args.seed)
    configure_tff_context(args.max_concurrent_calls)

    (
        client_train_csvs,
        global_val_csv,
        global_test_csv,
        client_train_sizes,
    ) = create_or_load_manifests(
        partition_dir=partition_dir,
        data_root=data_root,
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        resume=args.resume,
    )

    prototype_federated_data = make_federated_train_data(
        client_train_csvs,
        data_root,
        args,
        round_num=0,
    )
    if len(prototype_federated_data) != 5:
        print(
            f"[WARN] 当前客户端数量为 {len(prototype_federated_data)}，"
            "并非预期的 5。"
        )

    input_spec = prototype_federated_data[0].element_spec
    learning_process = build_learning_process(input_spec, args)

    val_dataset = make_keras_dataset(
        csv_path=global_val_csv,
        data_root=data_root,
        training=False,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    test_dataset = make_keras_dataset(
        csv_path=global_test_csv,
        data_root=data_root,
        training=False,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    initial_state = learning_process.initialize()

    metrics_csv = output_dir / "metrics.csv"
    round_state_path = output_dir / "run_state.json"
    checkpoint_dir = output_dir / "checkpoints"
    best_model_path = (
        output_dir / "global_models" / "best_global_model.h5"
    )
    last_model_path = (
        output_dir / "global_models" / "last_global_model.h5"
    )

    eval_model = make_evaluation_model()

    if args.resume:
        completed_round, checkpoint_path = load_round_state(
            round_state_path,
            output_dir,
        )
        restored_weights = load_model_weights_checkpoint(
            checkpoint_path
        )
        state = learning_process.set_model_weights(
            initial_state,
            restored_weights,
        )
        trim_metrics_after_round(metrics_csv, completed_round)
        assign_state_to_keras_model(
            learning_process,
            state,
            eval_model,
        )
        eval_model.save(last_model_path)
        start_round = completed_round + 1
        print(
            f"[RESUME] 已恢复 round {completed_round}，"
            f"下一轮为 {start_round}。"
        )
        print(f"[RESUME] 权重文件：{checkpoint_path}")
    else:
        state = initial_state
        completed_round = 0
        start_round = 1
        initial_checkpoint = (
            checkpoint_dir / "model_weights_round_000000.npz"
        )
        save_model_weights_checkpoint(
            learning_process,
            state,
            initial_checkpoint,
        )
        save_round_state(
            round_state_path,
            completed_round=0,
            checkpoint_path=initial_checkpoint,
            output_dir=output_dir,
        )
        assign_state_to_keras_model(
            learning_process,
            state,
            eval_model,
        )
        eval_model.save(last_model_path)
        print("[CHECKPOINT] 已保存初始全局模型 round 0。")

    bytes_per_global_model = model_size_bytes(eval_model)
    num_clients = len(prototype_federated_data)

    # 估算值：每轮向每个客户端下发一次模型，并上传一次同尺寸更新。
    estimated_round_comm_bytes = (
        2 * num_clients * bytes_per_global_model
    )

    best_val_accuracy = get_existing_best_validation(metrics_csv)

    client_partition_summary = pd.DataFrame([
        {
            "client_id": client_id,
            "total_partition_samples": item["total_samples"],
            "train_samples_after_val_split": client_train_sizes[int(client_id.split("_")[1])],
            "present_classes": item["present_classes"],
            "nonzero_classes": item["nonzero_classes"],
            "largest_class_id": item["largest_class_id"],
            "largest_class_ratio": item["largest_class_ratio"],
            "js_divergence_to_global": item["js_divergence_to_global"],
        }
        for client_id, item in partition_metadata["client_summary"].items()
    ])
    client_partition_summary.to_csv(
        output_dir / "client_partition_summary.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print("Stage 1C-A1 - Weighted FedProx under Dirichlet Non-IID")
    print("=" * 78)
    print(f"实验名称                : {args.experiment_name or output_dir.name}")
    print(f"数据划分目录            : {partition_dir}")
    print(f"划分类型                : {partition_metadata['partition_type']}")
    print(f"Dirichlet alpha         : {partition_metadata['alpha']}")
    print(
        "Mean JS divergence      : "
        f"{partition_metadata['mean_js_divergence_to_global']:.6f}"
    )
    print(f"TensorFlow              : {tf.__version__}")
    print(f"TFF                     : {tff.__version__}")
    print(f"客户端数量              : {num_clients}")
    print(f"各客户端训练样本数      : {client_train_sizes}")
    print(f"每轮参与客户端          : {num_clients} / {num_clients}")
    print(f"本地 epochs             : {args.local_epochs}")
    print(f"batch size              : {args.batch_size}")
    print(f"client optimizer        : {args.client_optimizer}")
    print(f"client learning rate    : {args.client_learning_rate}")
    print(f"server learning rate    : {args.server_learning_rate}")
    print("聚合权重                : NUM_EXAMPLES")
    print(f"目标通信轮数            : {args.rounds}")
    print(f"单个全局模型大小估算    : {bytes_per_global_model / 2**20:.3f} MiB")
    print(
        "每轮上下行通信估算      : "
        f"{estimated_round_comm_bytes / 2**20:.3f} MiB"
    )
    print(f"输出目录                : {output_dir}")
    print()

    if start_round <= args.rounds:
        for round_num in range(start_round, args.rounds + 1):
            # 每轮重新构造数据集，使用和 round 绑定的随机种子。
            federated_train_data = make_federated_train_data(
                client_train_csvs,
                data_root,
                args,
                round_num=round_num,
            )
            round_start = time.perf_counter()

            output = learning_process.next(
                state,
                federated_train_data,
            )
            state = output.state
            round_seconds = time.perf_counter() - round_start

            flat_tff_metrics = flatten_nested(output.metrics)
            train_loss = find_metric(
                flat_tff_metrics, "loss"
            )
            train_accuracy = find_metric(
                flat_tff_metrics, "accuracy"
            )

            assign_state_to_keras_model(
                learning_process,
                state,
                eval_model,
            )

            val_metrics = {"loss": np.nan, "accuracy": np.nan}
            if (
                round_num % args.eval_every == 0
                or round_num == args.rounds
            ):
                val_metrics = evaluate_model(
                    eval_model,
                    val_dataset,
                )

                if val_metrics["accuracy"] > best_val_accuracy:
                    best_val_accuracy = val_metrics["accuracy"]
                    eval_model.save(best_model_path)
                    print(
                        f"[BEST] round={round_num}, "
                        f"val_accuracy={best_val_accuracy:.6f}"
                    )

            eval_model.save(last_model_path)

            row: dict[str, Any] = {
                "round": round_num,
                "round_seconds": round_seconds,
                "num_clients": num_clients,
                "total_client_train_samples": int(
                    sum(client_train_sizes)
                ),
                "local_epochs": args.local_epochs,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "model_bytes": bytes_per_global_model,
                "estimated_round_communication_bytes": (
                    estimated_round_comm_bytes
                ),
                "estimated_cumulative_communication_bytes": (
                    round_num * estimated_round_comm_bytes
                ),
            }

            for key, value in flat_tff_metrics.items():
                safe_key = (
                    "tff_"
                    + key.replace("/", "__")
                    .replace(" ", "_")
                )
                if safe_key not in row:
                    row[safe_key] = value

            append_metrics_row(metrics_csv, row)

            if (
                round_num % args.checkpoint_every == 0
                or round_num == args.rounds
            ):
                round_checkpoint = checkpoint_dir / (
                    f"model_weights_round_{round_num:06d}.npz"
                )
                save_model_weights_checkpoint(
                    learning_process,
                    state,
                    round_checkpoint,
                )
                # JSON 最后写入；只有它成功更新后，该轮才算已提交。
                save_round_state(
                    round_state_path,
                    completed_round=round_num,
                    checkpoint_path=round_checkpoint,
                    output_dir=output_dir,
                )

            print(
                f"Round {round_num:03d}/{args.rounds} | "
                f"time={round_seconds:.2f}s | "
                f"train_loss={train_loss} | "
                f"train_acc={train_accuracy} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_acc={val_metrics['accuracy']:.6f}"
            )
    else:
        print(
            f"[INFO] checkpoint 已完成 round {completed_round}，"
            f"不低于目标 {args.rounds}，跳过训练。"
        )

    if not best_model_path.is_file():
        # 极端情况下没有触发验证评估，则使用最后模型。
        shutil.copy2(last_model_path, best_model_path)

    best_model = tf.keras.models.load_model(best_model_path)
    test_metrics = evaluate_model(best_model, test_dataset)
    y_true, y_pred = predict_all(best_model, test_dataset)
    extra_metrics = save_classification_outputs(
        y_true,
        y_pred,
        output_dir,
    )

    history = pd.read_csv(metrics_csv)
    valid_rows = history.dropna(subset=["val_accuracy"])
    if valid_rows.empty:
        best_round = None
        recorded_best_val = None
    else:
        best_row = valid_rows.loc[
            valid_rows["val_accuracy"].idxmax()
        ]
        best_round = int(best_row["round"])
        recorded_best_val = float(best_row["val_accuracy"])

    final_results = {
        "stage": "1C-A1",
        "algorithm": "Weighted FedProx",
        "proximal_strength": float(args.proximal_strength),
        "experiment_name": args.experiment_name or output_dir.name,
        "partition_dir": str(partition_dir),
        "partition_type": partition_metadata["partition_type"],
        "dirichlet_alpha": partition_metadata["alpha"],
        "partition_base_seed": partition_metadata["base_seed"],
        "partition_derived_seed": partition_metadata["derived_seed"],
        "mean_js_divergence_to_global": (
            partition_metadata["mean_js_divergence_to_global"]
        ),
        "num_clients": num_clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "client_optimizer": args.client_optimizer,
        "client_learning_rate": args.client_learning_rate,
        "server_optimizer": "SGDM",
        "server_learning_rate": args.server_learning_rate,
        "client_weighting": "NUM_EXAMPLES",
        "best_round": best_round,
        "best_val_accuracy": recorded_best_val,
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        **extra_metrics,
        "num_test_samples": int(len(y_true)),
        "best_model": str(best_model_path),
        "last_model": str(last_model_path),
        "estimated_model_bytes": bytes_per_global_model,
        "estimated_round_communication_bytes": (
            estimated_round_comm_bytes
        ),
        "estimated_total_communication_bytes": (
            args.rounds * estimated_round_comm_bytes
        ),
        "checkpoint_mode": (
            "global ModelWeights NPZ; exact for stateless server SGDM "
            "and stateless weighted-mean aggregator"
        ),
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(
            final_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "stage": "1C-A1",
        "project_stage": "1C-A1",
        "algorithm": "Weighted FedProx",
        "algorithm_version": "tff_builtin_weighted_fedprox_v1",
        "algorithm_specific_hparams": {
            "proximal_strength": float(args.proximal_strength),
        },
        "resolved_data_root": str(data_root),
        "resolved_partition_dir": str(partition_dir),
        "resolved_output_dir": str(output_dir),
        "partition": {
            "partition_type": partition_metadata["partition_type"],
            "alpha": partition_metadata["alpha"],
            "base_seed": partition_metadata["base_seed"],
            "derived_seed": partition_metadata["derived_seed"],
            "successful_attempt": partition_metadata["successful_attempt"],
            "mean_js_divergence_to_global": (
                partition_metadata["mean_js_divergence_to_global"]
            ),
            "client_summary": partition_metadata["client_summary"],
        },
        "tensorflow_version": tf.__version__,
        "tff_version": tff.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "num_classes": NUM_CLASSES,
        "client_train_sizes": client_train_sizes,
        "model_output": "43 logits",
        "loss": "SparseCategoricalCrossentropy(from_logits=True)",
        "metric": "SparseCategoricalAccuracy",
        "checkpoint_mode": (
            "global ModelWeights NPZ; FileProgramStateManager disabled "
            "because of TensorFlow SavedModel _DictWrapper incompatibility"
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_plots(metrics_csv, output_dir / "plots")

    print()
    print("=" * 78)
    print("FedProx 基线完成")
    print("=" * 78)
    print(json.dumps(final_results, ensure_ascii=False, indent=2))
    print()
    print(f"每轮指标       : {metrics_csv}")
    print(f"最佳全局模型   : {best_model_path}")
    print(f"最新全局模型   : {last_model_path}")
    print(f"权重检查点目录 : {output_dir / 'checkpoints'}")
    print(f"测试结果       : {output_dir / 'test_metrics.json'}")
    print(f"混淆矩阵       : {output_dir / 'confusion_matrix.csv'}")
    print(f"各类别准确率   : {output_dir / 'per_class_accuracy.csv'}")
    print(f"客户端划分摘要 : {output_dir / 'client_partition_summary.csv'}")
    print(f"曲线目录       : {output_dir / 'plots'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)