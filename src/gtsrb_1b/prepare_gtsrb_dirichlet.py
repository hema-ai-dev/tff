#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GTSRB Stage 1B-2: reproducible Dirichlet label-skew partitioning for 5 clients.

Outputs for each alpha:
- clients/client_0_train.csv ... client_4_train.csv
- client_distribution.csv
- client_distribution_wide.csv
- partition_manifest.csv
- split_metadata.json
- client_class_distribution_count.png
- client_class_distribution_ratio.png

The official test set is never repartitioned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible Dirichlet Non-IID partitions for GTSRB."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/gtsrb"),
        help="GTSRB root directory. Default: data/gtsrb",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Path to Train.csv. Default: <dataset-root>/Train.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/gtsrb"),
        help="Parent directory for generated partitions. Default: data/gtsrb",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[10.0, 1.0, 0.5, 0.1],
        help="Dirichlet alpha values. Default: 10 1.0 0.5 0.1",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of clients. Default: 5",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=43,
        help="Expected number of classes. Default: 43",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260720,
        help="Base random seed. Default: 20260720",
    )
    parser.add_argument(
        "--min-samples-per-client",
        type=int,
        default=500,
        help="Minimum total samples per client. Default: 500",
    )
    parser.add_argument(
        "--min-classes-per-client",
        type=int,
        default=5,
        help="Minimum number of present classes per client. Default: 5",
    )
    parser.add_argument(
        "--present-class-min-samples",
        type=int,
        default=5,
        help="A class counts as present only if client has at least this many samples. Default: 5",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5000,
        help="Maximum rejection-sampling attempts per alpha. Default: 5000",
    )
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="Verify that all paths in Train.csv exist under dataset-root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    return parser.parse_args()


def alpha_tag(alpha: float) -> str:
    """Convert alpha to a stable directory-friendly tag."""
    if math.isclose(alpha, round(alpha)):
        return str(int(round(alpha)))
    return str(alpha).replace(".", "")


def stable_seed(base_seed: int, alpha: float) -> int:
    """Derive a deterministic per-alpha seed from base seed and alpha."""
    payload = f"{base_seed}|{alpha:.12g}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little")


def validate_input(
    df: pd.DataFrame,
    dataset_root: Path,
    num_classes: int,
    verify_files: bool,
) -> None:
    required = {"Path", "ClassId"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Train.csv is missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Train.csv is empty.")

    if df["Path"].isna().any() or df["ClassId"].isna().any():
        raise ValueError("Train.csv contains missing Path or ClassId values.")

    if df["Path"].duplicated().any():
        dup_count = int(df["Path"].duplicated().sum())
        raise ValueError(f"Train.csv contains {dup_count} duplicated image paths.")

    labels = sorted(df["ClassId"].astype(int).unique().tolist())
    expected = list(range(num_classes))
    if labels != expected:
        raise ValueError(
            f"Unexpected class labels.\nExpected: {expected}\nFound: {labels}"
        )

    if verify_files:
        missing_files: List[str] = []
        for rel_path in df["Path"].astype(str):
            full_path = dataset_root / rel_path
            if not full_path.is_file():
                missing_files.append(rel_path)
                if len(missing_files) >= 20:
                    break
        if missing_files:
            raise FileNotFoundError(
                "Some image files referenced by Train.csv do not exist. "
                f"First examples: {missing_files}"
            )


def generate_partition(
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    rng: np.random.Generator,
) -> List[List[int]]:
    """
    For each class:
      1) shuffle its sample indices;
      2) draw a Dirichlet vector;
      3) draw exact integer client counts with multinomial;
      4) assign every sample exactly once.
    """
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id).copy()
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(
            np.full(num_clients, alpha, dtype=np.float64)
        )
        counts = rng.multinomial(len(class_indices), proportions)

        start = 0
        for client_id, count in enumerate(counts):
            end = start + int(count)
            if end > start:
                client_indices[client_id].extend(
                    class_indices[start:end].tolist()
                )
            start = end

        if start != len(class_indices):
            raise RuntimeError("Internal split error: class samples were not fully assigned.")

    for indices in client_indices:
        rng.shuffle(indices)

    return client_indices


def distribution_matrix(
    client_indices: List[List[int]],
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((len(client_indices), num_classes), dtype=np.int64)
    for client_id, indices in enumerate(client_indices):
        if indices:
            matrix[client_id] = np.bincount(
                labels[np.asarray(indices, dtype=np.int64)],
                minlength=num_classes,
            )
    return matrix


def validate_partition(
    client_indices: List[List[int]],
    labels: np.ndarray,
    num_classes: int,
    min_samples_per_client: int,
    min_classes_per_client: int,
    present_class_min_samples: int,
) -> Tuple[bool, Dict[str, object]]:
    total_samples = len(labels)
    flat = [idx for indices in client_indices for idx in indices]

    unique_count = len(set(flat))
    complete = len(flat) == total_samples
    no_duplicates = unique_count == total_samples
    correct_index_range = all(0 <= idx < total_samples for idx in flat)

    matrix = distribution_matrix(client_indices, labels, num_classes)
    sample_counts = matrix.sum(axis=1)
    present_class_counts = (matrix >= present_class_min_samples).sum(axis=1)

    min_samples_ok = bool(np.all(sample_counts >= min_samples_per_client))
    min_classes_ok = bool(np.all(present_class_counts >= min_classes_per_client))

    valid = (
        complete
        and no_duplicates
        and correct_index_range
        and min_samples_ok
        and min_classes_ok
    )

    diagnostics = {
        "valid": valid,
        "total_assigned": int(len(flat)),
        "total_expected": int(total_samples),
        "unique_assigned": int(unique_count),
        "complete": complete,
        "no_duplicates": no_duplicates,
        "correct_index_range": correct_index_range,
        "sample_counts": sample_counts.astype(int).tolist(),
        "present_class_counts": present_class_counts.astype(int).tolist(),
        "min_samples_ok": min_samples_ok,
        "min_classes_ok": min_classes_ok,
    }
    return valid, diagnostics


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence with natural logarithm."""
    p = p.astype(np.float64)
    q = q.astype(np.float64)

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return float("nan")

    p /= p_sum
    q /= q_sum
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def save_heatmap(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    normalized: bool,
) -> None:
    values = matrix.astype(np.float64)

    if normalized:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_sums,
            out=np.zeros_like(values),
            where=row_sums != 0,
        )

    fig, ax = plt.subplots(figsize=(18, 4.8))
    image = ax.imshow(values, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Client ID")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(i) for i in range(matrix.shape[1])], fontsize=7)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([f"client_{i}" for i in range(matrix.shape[0])])

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Within-client ratio" if normalized else "Sample count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    df: pd.DataFrame,
    client_indices: List[List[int]],
    matrix: np.ndarray,
    output_dir: Path,
    alpha: float,
    seed: int,
    derived_seed: int,
    attempt: int,
    args: argparse.Namespace,
) -> None:
    clients_dir = output_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)

    manifest_parts: List[pd.DataFrame] = []

    for client_id, indices in enumerate(client_indices):
        client_df = df.iloc[indices].copy()
        client_df.insert(0, "client_id", f"client_{client_id}")
        client_df.insert(1, "source_row_index", indices)

        client_csv = clients_dir / f"client_{client_id}_train.csv"
        client_df.to_csv(client_csv, index=False)

        manifest_parts.append(
            client_df[["source_row_index", "client_id", "Path", "ClassId"]].copy()
        )

    manifest_df = pd.concat(manifest_parts, ignore_index=True)
    manifest_df = manifest_df.sort_values("source_row_index").reset_index(drop=True)
    manifest_df.to_csv(output_dir / "partition_manifest.csv", index=False)

    long_rows = []
    for client_id in range(args.num_clients):
        for class_id in range(args.num_classes):
            long_rows.append(
                {
                    "client_id": f"client_{client_id}",
                    "class_id": class_id,
                    "sample_count": int(matrix[client_id, class_id]),
                }
            )

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(output_dir / "client_distribution.csv", index=False)

    wide_df = pd.DataFrame(
        matrix,
        index=[f"client_{i}" for i in range(args.num_clients)],
        columns=[f"class_{i}" for i in range(args.num_classes)],
    )
    wide_df.insert(0, "total_samples", matrix.sum(axis=1))
    wide_df.insert(
        1,
        "present_classes",
        (matrix >= args.present_class_min_samples).sum(axis=1),
    )
    wide_df.to_csv(output_dir / "client_distribution_wide.csv", index_label="client_id")

    global_counts = np.bincount(
        df["ClassId"].astype(int).to_numpy(),
        minlength=args.num_classes,
    )
    client_js = [
        js_divergence(matrix[i], global_counts)
        for i in range(args.num_clients)
    ]

    per_client_summary = {}
    for client_id in range(args.num_clients):
        total = int(matrix[client_id].sum())
        max_count = int(matrix[client_id].max())
        per_client_summary[f"client_{client_id}"] = {
            "total_samples": total,
            "present_classes": int(
                (matrix[client_id] >= args.present_class_min_samples).sum()
            ),
            "nonzero_classes": int((matrix[client_id] > 0).sum()),
            "largest_class_id": int(np.argmax(matrix[client_id])),
            "largest_class_count": max_count,
            "largest_class_ratio": float(max_count / total) if total else 0.0,
            "js_divergence_to_global": float(client_js[client_id]),
        }

    metadata = {
        "stage": "1B-2",
        "dataset": "GTSRB",
        "partition_type": "dirichlet_label_skew",
        "alpha": alpha,
        "base_seed": seed,
        "derived_seed": derived_seed,
        "successful_attempt": attempt,
        "num_clients": args.num_clients,
        "num_classes": args.num_classes,
        "total_samples": int(len(df)),
        "constraints": {
            "min_samples_per_client": args.min_samples_per_client,
            "min_classes_per_client": args.min_classes_per_client,
            "present_class_min_samples": args.present_class_min_samples,
            "max_attempts": args.max_attempts,
        },
        "global_class_counts": {
            str(i): int(global_counts[i]) for i in range(args.num_classes)
        },
        "client_summary": per_client_summary,
        "mean_js_divergence_to_global": float(np.mean(client_js)),
        "files": {
            "manifest": "partition_manifest.csv",
            "distribution_long": "client_distribution.csv",
            "distribution_wide": "client_distribution_wide.csv",
            "count_heatmap": "client_class_distribution_count.png",
            "ratio_heatmap": "client_class_distribution_ratio.png",
            "clients_directory": "clients",
        },
    }

    with (output_dir / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    save_heatmap(
        matrix,
        output_dir / "client_class_distribution_count.png",
        title=f"GTSRB Dirichlet Partition: alpha={alpha} (sample counts)",
        normalized=False,
    )
    save_heatmap(
        matrix,
        output_dir / "client_class_distribution_ratio.png",
        title=f"GTSRB Dirichlet Partition: alpha={alpha} (within-client ratios)",
        normalized=True,
    )


def main() -> None:
    args = parse_args()

    if args.num_clients < 2:
        raise ValueError("--num-clients must be at least 2.")
    if args.num_classes < 2:
        raise ValueError("--num-classes must be at least 2.")
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("All alpha values must be greater than 0.")
    if args.present_class_min_samples < 1:
        raise ValueError("--present-class-min-samples must be at least 1.")

    dataset_root = args.dataset_root.resolve()
    train_csv = (
        args.train_csv.resolve()
        if args.train_csv is not None
        else (dataset_root / "Train.csv").resolve()
    )
    output_root = args.output_root.resolve()

    if not train_csv.is_file():
        raise FileNotFoundError(f"Train.csv not found: {train_csv}")

    print(f"[INFO] Reading: {train_csv}")
    df = pd.read_csv(train_csv)
    df["ClassId"] = df["ClassId"].astype(int)

    validate_input(
        df=df,
        dataset_root=dataset_root,
        num_classes=args.num_classes,
        verify_files=args.verify_files,
    )

    labels = df["ClassId"].to_numpy(dtype=np.int64)

    print(f"[INFO] Samples: {len(df):,}")
    print(f"[INFO] Classes: {args.num_classes}")
    print(f"[INFO] Clients: {args.num_clients}")
    print(f"[INFO] Alphas: {args.alphas}")

    for alpha in args.alphas:
        tag = alpha_tag(alpha)
        output_dir = (
            output_root
            / f"federated_dirichlet_alpha{tag}_{args.num_clients}clients_seed{args.seed}"
        )

        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}\n"
                "Use --overwrite only if you intentionally want to replace it."
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        derived = stable_seed(args.seed, alpha)
        rng = np.random.default_rng(derived)

        print("\n" + "=" * 80)
        print(f"[INFO] Generating alpha={alpha}")
        print(f"[INFO] Derived seed={derived}")
        print(f"[INFO] Output={output_dir}")

        accepted_indices = None
        accepted_matrix = None
        accepted_diagnostics = None
        accepted_attempt = None

        for attempt in range(1, args.max_attempts + 1):
            client_indices = generate_partition(
                labels=labels,
                num_clients=args.num_clients,
                num_classes=args.num_classes,
                alpha=alpha,
                rng=rng,
            )

            valid, diagnostics = validate_partition(
                client_indices=client_indices,
                labels=labels,
                num_classes=args.num_classes,
                min_samples_per_client=args.min_samples_per_client,
                min_classes_per_client=args.min_classes_per_client,
                present_class_min_samples=args.present_class_min_samples,
            )

            if valid:
                accepted_indices = client_indices
                accepted_matrix = distribution_matrix(
                    client_indices,
                    labels,
                    args.num_classes,
                )
                accepted_diagnostics = diagnostics
                accepted_attempt = attempt
                break

            if attempt == 1 or attempt % 250 == 0:
                print(
                    f"[RETRY] attempt={attempt}, "
                    f"samples={diagnostics['sample_counts']}, "
                    f"present_classes={diagnostics['present_class_counts']}"
                )

        if accepted_indices is None or accepted_matrix is None:
            raise RuntimeError(
                f"Could not generate a valid partition for alpha={alpha} "
                f"within {args.max_attempts} attempts.\n"
                "Try reducing --min-samples-per-client or "
                "--min-classes-per-client."
            )

        save_outputs(
            df=df,
            client_indices=accepted_indices,
            matrix=accepted_matrix,
            output_dir=output_dir,
            alpha=alpha,
            seed=args.seed,
            derived_seed=derived,
            attempt=accepted_attempt,
            args=args,
        )

        print(f"[OK] alpha={alpha} accepted at attempt={accepted_attempt}")
        print(f"[OK] client sample counts: {accepted_diagnostics['sample_counts']}")
        print(
            "[OK] present class counts: "
            f"{accepted_diagnostics['present_class_counts']}"
        )
        print(f"[OK] saved to: {output_dir}")

    print("\n[DONE] All requested Dirichlet partitions were generated successfully.")


if __name__ == "__main__":
    main()
