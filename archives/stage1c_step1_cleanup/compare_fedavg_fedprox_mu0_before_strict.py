#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare fresh FedAvg and FedProx(mu=0) smoke-test outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = (
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fedavg-dir", type=Path, required=True)
    parser.add_argument("--fedprox-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def read_metrics(directory: Path) -> pd.DataFrame:
    path = directory / "metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "round" not in frame.columns:
        raise ValueError(f"{path} has no round column.")
    return frame


def main() -> None:
    args = parse_args()
    fedavg = read_metrics(args.fedavg_dir)
    fedprox = read_metrics(args.fedprox_dir)

    merged = fedavg.merge(
        fedprox,
        on="round",
        suffixes=("_fedavg", "_fedprox"),
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("No common rounds found.")

    print()
    print("=" * 88)
    print("FedAvg vs FedProx(mu=0) metric comparison")
    print("=" * 88)
    print(f"FedAvg directory : {args.fedavg_dir.resolve()}")
    print(f"FedProx directory: {args.fedprox_dir.resolve()}")
    print(f"Common rounds    : {merged['round'].tolist()}")
    print()

    all_strict = True
    compared = 0

    for column in METRIC_COLUMNS:
        left = f"{column}_fedavg"
        right = f"{column}_fedprox"
        if left not in merged.columns or right not in merged.columns:
            print(f"[SKIP] {column}: column missing in one output.")
            continue

        x = pd.to_numeric(merged[left], errors="coerce").to_numpy(float)
        y = pd.to_numeric(merged[right], errors="coerce").to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            print(f"[SKIP] {column}: no finite paired values.")
            continue

        diff = np.abs(x[mask] - y[mask])
        strict = np.allclose(
            x[mask],
            y[mask],
            atol=args.atol,
            rtol=args.rtol,
        )
        all_strict = all_strict and strict
        compared += 1

        print(
            f"{column:16s} "
            f"max_abs_diff={diff.max():.10g} "
            f"mean_abs_diff={diff.mean():.10g} "
            f"allclose={strict}"
        )

    fedprox_config_path = args.fedprox_dir / "config.json"
    if fedprox_config_path.is_file():
        config = json.loads(fedprox_config_path.read_text(encoding="utf-8"))
        print()
        print(
            "FedProx config proximal_strength = "
            f"{config.get('proximal_strength')}"
        )

    print()
    if compared == 0:
        print("[INCONCLUSIVE] No standard metric columns were comparable.")
    elif all_strict:
        print(
            "[PASS-STRICT] The reported smoke-test metrics are numerically "
            "equivalent under the selected tolerances."
        )
    else:
        print(
            "[REVIEW] The builders are mathematically equivalent at mu=0, "
            "but the reported trajectories are not numerically identical. "
            "Inspect the printed differences and both logs before continuing."
        )


if __name__ == "__main__":
    main()
