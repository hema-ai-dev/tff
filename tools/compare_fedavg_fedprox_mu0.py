#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict comparison for FedAvg and FedProx(mu=0).

Checks:
1. Core round metrics.
2. FedProx proximal_strength.
3. Client/train/validation manifests.
4. Every common NPZ checkpoint.
5. Final test metrics.
6. Classification CSV outputs.

Exit codes:
0: pass
1: comparison failed
2: incomplete or invalid inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_COLUMNS = (
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
)

FINAL_METRIC_KEYS = (
    "best_round",
    "best_val_accuracy",
    "test_loss",
    "test_accuracy",
    "overall_accuracy_from_predictions",
    "macro_class_accuracy",
    "num_test_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly compare FedAvg and FedProx(mu=0) outputs."
        )
    )
    parser.add_argument(
        "--fedavg-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fedprox-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help=(
            "Require checkpoint arrays to be exactly equal. "
            "Without this flag, numerical allclose is sufficient."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_metrics(directory: Path) -> pd.DataFrame:
    path = directory / "metrics.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    if "round" not in frame.columns:
        raise ValueError(
            f"{path} does not contain a round column."
        )

    return frame


def compare_core_metrics(
    fedavg_dir: Path,
    fedprox_dir: Path,
    atol: float,
    rtol: float,
    failures: list[str],
) -> None:
    fedavg = read_metrics(fedavg_dir)
    fedprox = read_metrics(fedprox_dir)

    fedavg_rounds = fedavg["round"].astype(int).tolist()
    fedprox_rounds = fedprox["round"].astype(int).tolist()

    print()
    print("[1] Core metrics")
    print("-" * 88)

    if fedavg_rounds != fedprox_rounds:
        failures.append(
            "The two runs do not contain the same round sequence."
        )
        print(
            f"[FAIL] rounds: {fedavg_rounds} != {fedprox_rounds}"
        )
        return

    merged = fedavg.merge(
        fedprox,
        on="round",
        suffixes=("_fedavg", "_fedprox"),
        how="inner",
    )

    for column in METRIC_COLUMNS:
        left = f"{column}_fedavg"
        right = f"{column}_fedprox"

        if left not in merged.columns or right not in merged.columns:
            failures.append(
                f"Missing metric column: {column}"
            )
            print(f"[FAIL] {column}: missing column")
            continue

        x = pd.to_numeric(
            merged[left],
            errors="coerce",
        ).to_numpy(dtype=float)

        y = pd.to_numeric(
            merged[right],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite_mask = np.isfinite(x) & np.isfinite(y)

        if not finite_mask.any():
            failures.append(
                f"No finite paired values for {column}."
            )
            print(f"[FAIL] {column}: no finite values")
            continue

        diff = np.abs(
            x[finite_mask] - y[finite_mask]
        )

        close = np.allclose(
            x[finite_mask],
            y[finite_mask],
            atol=atol,
            rtol=rtol,
        )

        exact = np.array_equal(
            x[finite_mask],
            y[finite_mask],
        )

        print(
            f"{column:16s} "
            f"max_abs_diff={diff.max():.12g} "
            f"mean_abs_diff={diff.mean():.12g} "
            f"exact={exact} "
            f"allclose={close}"
        )

        if not close:
            failures.append(
                f"Metric mismatch: {column}"
            )


def compare_fedprox_config(
    fedprox_dir: Path,
    failures: list[str],
    warnings: list[str],
) -> None:
    print()
    print("[2] FedProx configuration")
    print("-" * 88)

    config_path = fedprox_dir / "config.json"
    config = read_json(config_path)

    proximal_strength = config.get(
        "proximal_strength"
    )

    print(
        "proximal_strength = "
        f"{proximal_strength}"
    )

    try:
        proximal_strength_value = float(
            proximal_strength
        )
    except (TypeError, ValueError):
        failures.append(
            "FedProx config has no valid proximal_strength."
        )
        return

    if proximal_strength_value != 0.0:
        failures.append(
            "FedProx run is not a mu=0 run."
        )

    stage = config.get(
        "project_stage",
        config.get("stage"),
    )

    print(f"stage = {stage}")

    if stage != "1C-A1":
        warnings.append(
            "FedProx config uses legacy or missing stage metadata."
        )


def compare_manifests(
    fedavg_dir: Path,
    fedprox_dir: Path,
    failures: list[str],
) -> None:
    print()
    print("[3] Manifest files")
    print("-" * 88)

    fedavg_manifest_dir = fedavg_dir / "manifests"
    fedprox_manifest_dir = fedprox_dir / "manifests"

    if not fedavg_manifest_dir.is_dir():
        failures.append(
            f"Missing manifest directory: {fedavg_manifest_dir}"
        )
        return

    if not fedprox_manifest_dir.is_dir():
        failures.append(
            f"Missing manifest directory: {fedprox_manifest_dir}"
        )
        return

    fedavg_files = {
        path.name: path
        for path in fedavg_manifest_dir.glob("*")
        if path.is_file()
    }

    fedprox_files = {
        path.name: path
        for path in fedprox_manifest_dir.glob("*")
        if path.is_file()
    }

    common_names = sorted(
        set(fedavg_files) & set(fedprox_files)
    )

    if not common_names:
        failures.append(
            "No common manifest files were found."
        )
        return

    for name in common_names:
        left_hash = sha256_file(
            fedavg_files[name]
        )
        right_hash = sha256_file(
            fedprox_files[name]
        )

        identical = left_hash == right_hash

        print(
            f"{name:32s} identical={identical}"
        )

        if not identical:
            failures.append(
                f"Manifest mismatch: {name}"
            )

    only_fedavg = sorted(
        set(fedavg_files) - set(fedprox_files)
    )
    only_fedprox = sorted(
        set(fedprox_files) - set(fedavg_files)
    )

    if only_fedavg:
        failures.append(
            f"Manifest files only in FedAvg: {only_fedavg}"
        )

    if only_fedprox:
        failures.append(
            f"Manifest files only in FedProx: {only_fedprox}"
        )


def load_npz_arrays(
    path: Path,
) -> dict[str, np.ndarray]:
    with np.load(
        path,
        allow_pickle=False,
    ) as payload:
        return {
            key: np.array(payload[key])
            for key in payload.files
        }


def compare_checkpoints(
    fedavg_dir: Path,
    fedprox_dir: Path,
    atol: float,
    rtol: float,
    require_exact: bool,
    failures: list[str],
) -> None:
    print()
    print("[4] NPZ checkpoints")
    print("-" * 88)

    fedavg_checkpoint_dir = (
        fedavg_dir / "checkpoints"
    )
    fedprox_checkpoint_dir = (
        fedprox_dir / "checkpoints"
    )

    fedavg_files = {
        path.name: path
        for path in fedavg_checkpoint_dir.glob(
            "model_weights_round_*.npz"
        )
    }

    fedprox_files = {
        path.name: path
        for path in fedprox_checkpoint_dir.glob(
            "model_weights_round_*.npz"
        )
    }

    common_names = sorted(
        set(fedavg_files) & set(fedprox_files)
    )

    if not common_names:
        failures.append(
            "No common NPZ checkpoints were found."
        )
        return

    if set(fedavg_files) != set(fedprox_files):
        failures.append(
            "The two runs do not have the same checkpoint names."
        )

    for name in common_names:
        left = load_npz_arrays(
            fedavg_files[name]
        )
        right = load_npz_arrays(
            fedprox_files[name]
        )

        if set(left) != set(right):
            failures.append(
                f"{name}: NPZ keys are different."
            )
            print(
                f"{name:40s} keys_equal=False"
            )
            continue

        checkpoint_exact = True
        checkpoint_close = True
        max_abs_diff = 0.0
        squared_l2 = 0.0

        for key in sorted(left):
            x = left[key]
            y = right[key]

            if x.shape != y.shape:
                checkpoint_exact = False
                checkpoint_close = False
                failures.append(
                    f"{name}, {key}: shape mismatch "
                    f"{x.shape} != {y.shape}"
                )
                continue

            exact = np.array_equal(x, y)
            close = np.allclose(
                x,
                y,
                atol=atol,
                rtol=rtol,
            )

            checkpoint_exact = (
                checkpoint_exact and exact
            )
            checkpoint_close = (
                checkpoint_close and close
            )

            if (
                np.issubdtype(x.dtype, np.number)
                and np.issubdtype(y.dtype, np.number)
            ):
                diff = (
                    x.astype(np.float64)
                    - y.astype(np.float64)
                )

                if diff.size:
                    max_abs_diff = max(
                        max_abs_diff,
                        float(np.max(np.abs(diff))),
                    )

                    squared_l2 += float(
                        np.sum(diff * diff)
                    )

        l2_difference = float(
            np.sqrt(squared_l2)
        )

        print(
            f"{name:40s} "
            f"exact={checkpoint_exact} "
            f"allclose={checkpoint_close} "
            f"max_abs_diff={max_abs_diff:.12g} "
            f"l2_diff={l2_difference:.12g}"
        )

        if not checkpoint_close:
            failures.append(
                f"Checkpoint mismatch: {name}"
            )
        elif require_exact and not checkpoint_exact:
            failures.append(
                f"Checkpoint is not exactly equal: {name}"
            )


def compare_test_metrics(
    fedavg_dir: Path,
    fedprox_dir: Path,
    atol: float,
    rtol: float,
    failures: list[str],
    warnings: list[str],
) -> None:
    print()
    print("[5] Final test metrics")
    print("-" * 88)

    fedavg_path = (
        fedavg_dir / "test_metrics.json"
    )
    fedprox_path = (
        fedprox_dir / "test_metrics.json"
    )

    fedavg = read_json(fedavg_path)
    fedprox = read_json(fedprox_path)

    for key in FINAL_METRIC_KEYS:
        if key not in fedavg or key not in fedprox:
            warnings.append(
                f"Final metric key missing: {key}"
            )
            print(
                f"{key:36s} skipped"
            )
            continue

        left = fedavg[key]
        right = fedprox[key]

        if isinstance(left, (int, float)) and isinstance(
            right,
            (int, float),
        ):
            equal = bool(
                np.isclose(
                    float(left),
                    float(right),
                    atol=atol,
                    rtol=rtol,
                )
            )
        else:
            equal = left == right

        print(
            f"{key:36s} "
            f"fedavg={left} "
            f"fedprox={right} "
            f"equal={equal}"
        )

        if not equal:
            failures.append(
                f"Final metric mismatch: {key}"
            )

    fedprox_stage = fedprox.get(
        "project_stage",
        fedprox.get("stage"),
    )

    if fedprox_stage != "1C-A1":
        warnings.append(
            "Legacy FedProx test_metrics.json stage metadata detected. "
            "Do not modify the historical result manually; "
            "future runs will use the corrected trainer."
        )


def compare_classification_outputs(
    fedavg_dir: Path,
    fedprox_dir: Path,
    failures: list[str],
) -> None:
    print()
    print("[6] Classification outputs")
    print("-" * 88)

    filenames = (
        "confusion_matrix.csv",
        "per_class_accuracy.csv",
    )

    for filename in filenames:
        left = fedavg_dir / filename
        right = fedprox_dir / filename

        if not left.is_file() or not right.is_file():
            failures.append(
                f"Missing classification output: {filename}"
            )
            print(
                f"{filename:32s} missing"
            )
            continue

        identical = (
            sha256_file(left)
            == sha256_file(right)
        )

        print(
            f"{filename:32s} identical={identical}"
        )

        if not identical:
            failures.append(
                f"Classification output mismatch: {filename}"
            )


def main() -> None:
    args = parse_args()

    fedavg_dir = (
        args.fedavg_dir.expanduser().resolve()
    )
    fedprox_dir = (
        args.fedprox_dir.expanduser().resolve()
    )

    if not fedavg_dir.is_dir():
        print(
            f"[ERROR] Directory not found: {fedavg_dir}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if not fedprox_dir.is_dir():
        print(
            f"[ERROR] Directory not found: {fedprox_dir}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    failures: list[str] = []
    warnings: list[str] = []

    print("=" * 88)
    print("Strict FedAvg vs FedProx(mu=0) equivalence check")
    print("=" * 88)
    print(f"FedAvg : {fedavg_dir}")
    print(f"FedProx: {fedprox_dir}")
    print(
        f"Tolerance: atol={args.atol}, rtol={args.rtol}, "
        f"require_exact={args.require_exact}"
    )

    try:
        compare_core_metrics(
            fedavg_dir,
            fedprox_dir,
            args.atol,
            args.rtol,
            failures,
        )

        compare_fedprox_config(
            fedprox_dir,
            failures,
            warnings,
        )

        compare_manifests(
            fedavg_dir,
            fedprox_dir,
            failures,
        )

        compare_checkpoints(
            fedavg_dir,
            fedprox_dir,
            args.atol,
            args.rtol,
            args.require_exact,
            failures,
        )

        compare_test_metrics(
            fedavg_dir,
            fedprox_dir,
            args.atol,
            args.rtol,
            failures,
            warnings,
        )

        compare_classification_outputs(
            fedavg_dir,
            fedprox_dir,
            failures,
        )

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"[ERROR] {error}",
            file=sys.stderr,
        )
        raise SystemExit(2) from error

    print()
    print("=" * 88)
    print("Summary")
    print("=" * 88)

    for warning in warnings:
        print(f"[WARN] {warning}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")

        print()
        print(
            "[FAILED] FedAvg and FedProx(mu=0) "
            "did not pass strict equivalence."
        )
        raise SystemExit(1)

    print()
    print(
        "[PASS] FedAvg and FedProx(mu=0) "
        "passed strict equivalence."
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()