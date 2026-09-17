#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GTSRB -> 5 客户端分层 IID 划分脚本

适用阶段：
    Stage 1A: Centralized / Local-only / FedAvg 基线验证

设计原则：
1. 以 Train.csv / Test.csv 为唯一索引来源，不直接遍历 Train/、train/ 等目录。
2. 对每个 ClassId 单独随机打乱，再近似平均分配给 5 个客户端。
3. 不复制图片，只输出 CSV 清单，节省磁盘空间。
4. 官方 Test.csv 不参与客户端划分，作为三组实验共同的 global_test.csv。
5. 自动检查：
   - 43 个类别是否完整；
   - CSV 路径对应图片是否存在；
   - Train.csv 是否有重复图片路径；
   - 各客户端是否互斥；
   - 各客户端并集是否等于完整训练集；
   - 每个客户端是否都包含全部 43 类。
6. 输出划分统计、元数据和类别分布图，便于论文和实验记录。

示例：
    python scripts/split_gtsrb_iid.py \
        --data-root data/gtsrb \
        --num-clients 5 \
        --num-classes 43 \
        --seed 20260720
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_CANDIDATES = (
    "ClassId",
    "ClassID",
    "class_id",
    "label",
    "Label",
)

PATH_CANDIDATES = (
    "Path",
    "path",
    "image_path",
    "filepath",
    "FilePath",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 GTSRB 官方训练集划分为多个近似均衡的分层 IID 客户端。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="GTSRB 根目录，例如：data/gtsrb",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="客户端数量，默认 5。",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=43,
        help="类别数量，GTSRB 默认 43。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260720,
        help="随机种子，固定后可以复现完全相同的划分。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "输出目录。默认写入："
            "<data-root>/federated_iid_<num_clients>clients_seed<seed>"
        ),
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="跳过图片文件存在性检查。首次运行不建议使用。",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不生成客户端类别分布图。",
    )
    return parser.parse_args()


def find_csv_case_insensitive(root: Path, expected_name: str) -> Path:
    """
    在 root 下查找文件名大小写可能不同的 CSV。
    例如 expected_name='Train.csv'，允许找到 train.csv。
    """
    exact = root / expected_name
    if exact.is_file():
        return exact

    candidates = [
        path
        for path in root.iterdir()
        if path.is_file() and path.name.lower() == expected_name.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"在目录 {root} 中找不到 {expected_name}。"
        )

    raise RuntimeError(
        f"在目录 {root} 中发现多个同名但大小写不同的文件：{candidates}"
    )


def find_column(
    columns: Iterable[str],
    candidates: Iterable[str],
    column_type: str,
) -> str:
    columns = list(columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"无法识别{column_type}列。\n"
        f"当前 CSV 列名：{columns}\n"
        f"支持的候选列名：{list(candidates)}"
    )


def normalize_relative_path(value: object) -> str:
    """
    将 CSV 中的路径统一为 Linux/Windows 都能处理的相对路径格式。
    只规范斜杠，不修改 Train/train 的大小写。
    """
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def absolute_image_path(data_root: Path, relative_path: str) -> Path:
    return data_root.joinpath(*relative_path.split("/"))


def validate_dataframe_basic(
    df: pd.DataFrame,
    csv_name: str,
    label_col: str,
    path_col: str,
) -> None:
    if df.empty:
        raise ValueError(f"{csv_name} 为空。")

    if df[label_col].isna().any():
        count = int(df[label_col].isna().sum())
        raise ValueError(f"{csv_name} 的 {label_col} 中有 {count} 个空标签。")

    if df[path_col].isna().any():
        count = int(df[path_col].isna().sum())
        raise ValueError(f"{csv_name} 的 {path_col} 中有 {count} 个空路径。")

    duplicated = df[path_col].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, path_col].head(10).tolist()
        raise ValueError(
            f"{csv_name} 中发现重复图片路径，共 {int(duplicated.sum())} 行。\n"
            f"示例：{examples}"
        )


def validate_class_ids(
    df: pd.DataFrame,
    label_col: str,
    num_classes: int,
    dataset_name: str,
) -> None:
    actual = sorted(df[label_col].unique().tolist())
    expected = list(range(num_classes))

    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            f"{dataset_name} 类别编号不符合预期。\n"
            f"缺失类别：{missing}\n"
            f"异常类别：{unexpected}\n"
            f"实际类别：{actual}"
        )


def validate_image_files(
    df: pd.DataFrame,
    data_root: Path,
    path_col: str,
    dataset_name: str,
    max_examples: int = 20,
) -> None:
    missing: list[str] = []

    for rel_path in df[path_col].astype(str):
        image_path = absolute_image_path(data_root, rel_path)
        if not image_path.is_file():
            missing.append(str(image_path))
            if len(missing) >= max_examples:
                break

    if missing:
        joined = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            f"{dataset_name} 中存在图片路径找不到对应文件。\n"
            f"前 {len(missing)} 个示例：\n{joined}\n\n"
            "你的目录同时存在 Train/train、Test/test 时，不要手动猜测目录；"
            "应以 CSV 的 Path 列为准，并确保路径大小写完全一致。"
        )


def make_stratified_iid_assignment(
    train_df: pd.DataFrame,
    label_col: str,
    num_clients: int,
    seed: int,
) -> pd.DataFrame:
    """
    按类别进行分层 IID 划分。

    对每个类别：
    1. 随机打乱样本；
    2. 按 num_clients 近似平均切分；
    3. 轮换“余数样本”优先分配到的客户端，
       避免每个类别的余数总是落到 client_0、client_1。
    """
    rng = np.random.default_rng(seed)
    assigned_parts: list[pd.DataFrame] = []

    class_ids = sorted(train_df[label_col].unique().tolist())

    for class_order, class_id in enumerate(class_ids):
        class_df = train_df.loc[train_df[label_col] == class_id].copy()

        shuffled_indices = class_df.index.to_numpy(copy=True)
        rng.shuffle(shuffled_indices)

        # np.array_split 会把余数优先分给前几个分片。
        # 通过按类别轮换分片到客户端的映射，平衡客户端总样本数。
        shards = np.array_split(shuffled_indices, num_clients)
        rotation = class_order % num_clients

        for shard_position, shard_indices in enumerate(shards):
            client_id = (shard_position + rotation) % num_clients

            part = train_df.loc[shard_indices].copy()
            part["client_id"] = int(client_id)
            assigned_parts.append(part)

    assigned = pd.concat(assigned_parts, axis=0)

    # 每个客户端内部再次打乱，避免 CSV 按类别成块排列。
    shuffled_clients: list[pd.DataFrame] = []
    for client_id in range(num_clients):
        client_df = assigned.loc[assigned["client_id"] == client_id].copy()
        client_df = client_df.sample(
            frac=1.0,
            random_state=seed + client_id + 1,
        )
        shuffled_clients.append(client_df)

    assigned = pd.concat(shuffled_clients, axis=0)
    return assigned


def validate_assignment(
    original_train_df: pd.DataFrame,
    assigned_df: pd.DataFrame,
    label_col: str,
    num_clients: int,
    num_classes: int,
) -> None:
    """
    检查划分是否满足 Stage 1A 的基本要求。
    """
    if len(assigned_df) != len(original_train_df):
        raise RuntimeError(
            "划分后的总样本数与原始训练集不一致："
            f"{len(assigned_df)} != {len(original_train_df)}"
        )

    if assigned_df["_source_index"].duplicated().any():
        duplicated_count = int(
            assigned_df["_source_index"].duplicated(keep=False).sum()
        )
        raise RuntimeError(
            f"客户端划分发生样本重复，共涉及 {duplicated_count} 行。"
        )

    original_indices = set(
        original_train_df["_source_index"].astype(int).tolist()
    )
    assigned_indices = set(
        assigned_df["_source_index"].astype(int).tolist()
    )

    if original_indices != assigned_indices:
        missing = original_indices - assigned_indices
        extra = assigned_indices - original_indices
        raise RuntimeError(
            "客户端并集没有完整覆盖原始训练集："
            f"missing={len(missing)}, extra={len(extra)}"
        )

    actual_client_ids = sorted(
        assigned_df["client_id"].unique().astype(int).tolist()
    )
    expected_client_ids = list(range(num_clients))

    if actual_client_ids != expected_client_ids:
        raise RuntimeError(
            f"客户端编号异常：actual={actual_client_ids}, "
            f"expected={expected_client_ids}"
        )

    expected_classes = set(range(num_classes))
    for client_id in range(num_clients):
        client_df = assigned_df.loc[
            assigned_df["client_id"] == client_id
        ]
        client_classes = set(client_df[label_col].astype(int).unique())

        if client_classes != expected_classes:
            missing_classes = sorted(expected_classes - client_classes)
            raise RuntimeError(
                f"client_{client_id} 缺失类别：{missing_classes}"
            )


def build_long_summary(
    assigned_df: pd.DataFrame,
    label_col: str,
    num_clients: int,
    num_classes: int,
) -> pd.DataFrame:
    """
    长表：
        client_id, class_id, num_samples,
        fraction_in_client, fraction_of_class
    """
    client_totals = (
        assigned_df.groupby("client_id")
        .size()
        .to_dict()
    )
    class_totals = (
        assigned_df.groupby(label_col)
        .size()
        .to_dict()
    )

    rows: list[dict[str, float | int]] = []

    for client_id in range(num_clients):
        client_df = assigned_df.loc[
            assigned_df["client_id"] == client_id
        ]
        counts = (
            client_df[label_col]
            .value_counts()
            .reindex(range(num_classes), fill_value=0)
        )

        for class_id in range(num_classes):
            count = int(counts.loc[class_id])
            rows.append(
                {
                    "client_id": client_id,
                    "class_id": class_id,
                    "num_samples": count,
                    "fraction_in_client": (
                        count / int(client_totals[client_id])
                    ),
                    "fraction_of_class": (
                        count / int(class_totals[class_id])
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_wide_summary(
    assigned_df: pd.DataFrame,
    label_col: str,
    num_clients: int,
    num_classes: int,
) -> pd.DataFrame:
    """
    宽表：每行一个客户端，每列一个类别。
    """
    rows: list[dict[str, int]] = []

    for client_id in range(num_clients):
        client_df = assigned_df.loc[
            assigned_df["client_id"] == client_id
        ]
        counts = (
            client_df[label_col]
            .value_counts()
            .reindex(range(num_classes), fill_value=0)
        )

        row: dict[str, int] = {
            "client_id": client_id,
            "num_samples": int(len(client_df)),
            "num_present_classes": int((counts > 0).sum()),
            "min_samples_per_class": int(counts.min()),
            "max_samples_per_class": int(counts.max()),
        }
        for class_id in range(num_classes):
            row[f"class_{class_id}"] = int(counts.loc[class_id])

        rows.append(row)

    return pd.DataFrame(rows)


def sha256_of_dataframe_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> str:
    """
    生成划分指纹，用于确认未来加载的是不是同一份划分。
    """
    payload = (
        df[columns]
        .sort_values(columns)
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def save_distribution_plot(
    wide_summary: pd.DataFrame,
    num_classes: int,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[警告] 未安装 matplotlib，跳过类别分布图生成。",
            file=sys.stderr,
        )
        return

    class_columns = [f"class_{i}" for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(16, 7))
    for _, row in wide_summary.iterrows():
        client_id = int(row["client_id"])
        values = row[class_columns].to_numpy(dtype=int)
        ax.plot(
            range(num_classes),
            values,
            marker="o",
            markersize=2,
            linewidth=1,
            label=f"client_{client_id}",
        )

    ax.set_title("GTSRB Stratified IID Split: Samples per Class")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Number of samples")
    ax.set_xticks(range(num_classes))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(
            f"--data-root 不是有效目录：{data_root}"
        )

    if args.num_clients < 2:
        raise ValueError("--num-clients 必须至少为 2。")

    if args.num_classes < 2:
        raise ValueError("--num-classes 必须至少为 2。")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root
        / (
            f"federated_iid_{args.num_clients}"
            f"clients_seed{args.seed}"
        )
    )

    train_csv_path = find_csv_case_insensitive(
        data_root,
        "Train.csv",
    )
    test_csv_path = find_csv_case_insensitive(
        data_root,
        "Test.csv",
    )

    print("=" * 72)
    print("GTSRB Stage 1A: 分层 IID 客户端划分")
    print("=" * 72)
    print(f"数据根目录 : {data_root}")
    print(f"训练 CSV   : {train_csv_path}")
    print(f"测试 CSV   : {test_csv_path}")
    print(f"客户端数量 : {args.num_clients}")
    print(f"类别数量   : {args.num_classes}")
    print(f"随机种子   : {args.seed}")
    print(f"输出目录   : {output_dir}")
    print()

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_label_col = find_column(
        train_df.columns,
        LABEL_CANDIDATES,
        "训练集标签",
    )
    train_path_col = find_column(
        train_df.columns,
        PATH_CANDIDATES,
        "训练集图片路径",
    )
    test_label_col = find_column(
        test_df.columns,
        LABEL_CANDIDATES,
        "测试集标签",
    )
    test_path_col = find_column(
        test_df.columns,
        PATH_CANDIDATES,
        "测试集图片路径",
    )

    # 统一数据类型与路径格式。
    train_df[train_label_col] = train_df[train_label_col].astype(int)
    test_df[test_label_col] = test_df[test_label_col].astype(int)
    train_df[train_path_col] = train_df[train_path_col].map(
        normalize_relative_path
    )
    test_df[test_path_col] = test_df[test_path_col].map(
        normalize_relative_path
    )

    validate_dataframe_basic(
        train_df,
        "Train.csv",
        train_label_col,
        train_path_col,
    )
    validate_dataframe_basic(
        test_df,
        "Test.csv",
        test_label_col,
        test_path_col,
    )

    validate_class_ids(
        train_df,
        train_label_col,
        args.num_classes,
        "训练集",
    )
    validate_class_ids(
        test_df,
        test_label_col,
        args.num_classes,
        "测试集",
    )

    # 检查训练集和测试集是否出现相同相对路径。
    train_paths = set(train_df[train_path_col].astype(str))
    test_paths = set(test_df[test_path_col].astype(str))
    overlap_paths = train_paths.intersection(test_paths)
    if overlap_paths:
        examples = sorted(overlap_paths)[:10]
        raise ValueError(
            "Train.csv 与 Test.csv 出现相同图片路径，可能存在数据泄漏。\n"
            f"示例：{examples}"
        )

    if not args.skip_image_check:
        print("[1/5] 检查 Train.csv 对应图片...")
        validate_image_files(
            train_df,
            data_root,
            train_path_col,
            "训练集",
        )

        print("[2/5] 检查 Test.csv 对应图片...")
        validate_image_files(
            test_df,
            data_root,
            test_path_col,
            "测试集",
        )
    else:
        print("[1/5] 已跳过图片存在性检查。")
        print("[2/5] 已跳过图片存在性检查。")

    # source_index 是 Train.csv 中的原始行号，用于验证互斥性和完整性。
    train_working = train_df.copy()
    train_working.insert(
        0,
        "_source_index",
        np.arange(len(train_working), dtype=np.int64),
    )

    print("[3/5] 进行按类别分层的 IID 划分...")
    assigned_df = make_stratified_iid_assignment(
        train_df=train_working,
        label_col=train_label_col,
        num_clients=args.num_clients,
        seed=args.seed,
    )

    print("[4/5] 验证客户端互斥性、完整性和类别覆盖...")
    validate_assignment(
        original_train_df=train_working,
        assigned_df=assigned_df,
        label_col=train_label_col,
        num_clients=args.num_clients,
        num_classes=args.num_classes,
    )

    print("[5/5] 保存 CSV、统计信息和元数据...")
    clients_dir = output_dir / "clients"
    output_dir.mkdir(parents=True, exist_ok=True)
    clients_dir.mkdir(parents=True, exist_ok=True)

    # Centralized 使用完整 Train.csv。
    centralized_df = train_df.copy()
    centralized_df.insert(
        0,
        "sample_id",
        np.arange(len(centralized_df), dtype=np.int64),
    )
    centralized_df.to_csv(
        output_dir / "centralized_train.csv",
        index=False,
    )

    # 三组基线共用官方 Test.csv。
    global_test_df = test_df.copy()
    global_test_df.insert(
        0,
        "sample_id",
        np.arange(len(global_test_df), dtype=np.int64),
    )
    global_test_df.to_csv(
        output_dir / "global_test.csv",
        index=False,
    )

    # 每个客户端保存一个独立 CSV。
    for client_id in range(args.num_clients):
        client_df = assigned_df.loc[
            assigned_df["client_id"] == client_id
        ].copy()

        client_df = client_df.rename(
            columns={"_source_index": "sample_id"}
        )

        # 将 client_id 放在第一列，便于检查。
        preferred_columns = [
            "client_id",
            "sample_id",
        ]
        remaining_columns = [
            col
            for col in client_df.columns
            if col not in preferred_columns
        ]
        client_df = client_df[
            preferred_columns + remaining_columns
        ]

        client_df.to_csv(
            clients_dir / f"client_{client_id}_train.csv",
            index=False,
        )

    long_summary = build_long_summary(
        assigned_df=assigned_df,
        label_col=train_label_col,
        num_clients=args.num_clients,
        num_classes=args.num_classes,
    )
    wide_summary = build_wide_summary(
        assigned_df=assigned_df,
        label_col=train_label_col,
        num_clients=args.num_clients,
        num_classes=args.num_classes,
    )

    long_summary.to_csv(
        output_dir / "split_summary_long.csv",
        index=False,
    )
    wide_summary.to_csv(
        output_dir / "split_summary_wide.csv",
        index=False,
    )

    assignment_for_hash = assigned_df[
        ["_source_index", "client_id", train_label_col]
    ].copy()
    split_sha256 = sha256_of_dataframe_columns(
        assignment_for_hash,
        ["_source_index", "client_id", train_label_col],
    )

    client_sizes = {
        str(client_id): int(
            (assigned_df["client_id"] == client_id).sum()
        )
        for client_id in range(args.num_clients)
    }

    metadata = {
        "dataset": "GTSRB",
        "stage": "1A",
        "split_type": "stratified_iid",
        "description": (
            "Each class is independently shuffled and distributed "
            "approximately evenly across all clients."
        ),
        "data_root": str(data_root),
        "train_csv": str(train_csv_path),
        "test_csv": str(test_csv_path),
        "output_dir": str(output_dir),
        "num_clients": args.num_clients,
        "num_classes": args.num_classes,
        "seed": args.seed,
        "num_train_samples": int(len(train_df)),
        "num_test_samples": int(len(test_df)),
        "client_num_samples": client_sizes,
        "train_label_column": train_label_col,
        "train_path_column": train_path_col,
        "test_label_column": test_label_col,
        "test_path_column": test_path_col,
        "all_clients_have_all_classes": True,
        "clients_are_disjoint": True,
        "client_union_equals_full_train": True,
        "global_test_is_official_test_csv": True,
        "split_sha256": split_sha256,
    }

    with (
        output_dir / "split_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    readme_text = f"""GTSRB Stage 1A IID Split

Split type:
    Stratified IID

Clients:
    {args.num_clients}

Classes:
    {args.num_classes}

Random seed:
    {args.seed}

Files:
    centralized_train.csv
        Complete official training set for the Centralized baseline.

    global_test.csv
        Complete official test set shared by Centralized, Local-only,
        and FedAvg.

    clients/client_0_train.csv ... client_{args.num_clients - 1}_train.csv
        Per-client training manifests for Local-only and FedAvg.

    split_summary_long.csv
        One row per client/class pair.

    split_summary_wide.csv
        One row per client, with one column per class.

    split_metadata.json
        Reproducibility metadata and split fingerprint.

Important:
    Image files are not copied.
    Each CSV Path value is relative to:
        {data_root}
"""
    (output_dir / "README.txt").write_text(
        readme_text,
        encoding="utf-8",
    )

    if not args.no_plot:
        save_distribution_plot(
            wide_summary=wide_summary,
            num_classes=args.num_classes,
            output_path=(
                output_dir / "client_class_distribution.png"
            ),
        )

    print()
    print("=" * 72)
    print("划分完成")
    print("=" * 72)
    print(f"训练样本总数：{len(train_df)}")
    print(f"测试样本总数：{len(test_df)}")
    print()

    for client_id in range(args.num_clients):
        client_df = assigned_df.loc[
            assigned_df["client_id"] == client_id
        ]
        class_counts = client_df[train_label_col].value_counts()

        print(
            f"client_{client_id}: "
            f"{len(client_df)} samples, "
            f"{client_df[train_label_col].nunique()} classes, "
            f"min/class={int(class_counts.min())}, "
            f"max/class={int(class_counts.max())}"
        )

    size_values = list(client_sizes.values())
    print()
    print(
        "客户端样本数最大差值："
        f"{max(size_values) - min(size_values)}"
    )
    print(f"划分指纹 SHA256：{split_sha256}")
    print(f"输出目录：{output_dir}")
    print()
    print("已验证：")
    print("  [OK] 客户端之间没有重复样本")
    print("  [OK] 五个客户端并集等于完整训练集")
    print("  [OK] 每个客户端都包含全部类别")
    print("  [OK] 官方测试集没有参与训练划分")


if __name__ == "__main__":
    main()