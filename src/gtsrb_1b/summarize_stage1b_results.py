#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["iid", "alpha10", "alpha1", "alpha05", "alpha01"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1b-root",
        type=Path,
        default=Path("outputs/stage1b"),
    )
    parser.add_argument(
        "--stage1a-iid-dir",
        type=Path,
        default=Path("outputs/stage1a/fedavg_nobn_seed20260720"),
        help="Optional Stage 1A IID FedAvg directory.",
    )
    parser.add_argument(
        "--partition-summary",
        type=Path,
        default=Path("data/gtsrb/stage1b_partition_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage1b/comparison"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def convergence_round(metrics: pd.DataFrame, fraction: float = 0.95):
    values = pd.to_numeric(metrics.get("val_accuracy"), errors="coerce")
    if values is None or not values.notna().any():
        return None
    best = float(values.max())
    threshold = fraction * best
    hit = metrics.loc[values >= threshold, "round"]
    return int(hit.iloc[0]) if not hit.empty else None


def load_experiment(tag: str, directory: Path, mean_js=None, alpha=None):
    metrics_path = directory / "metrics.csv"
    test_path = directory / "test_metrics.json"

    if not metrics_path.is_file() or not test_path.is_file():
        print(f"[SKIP] incomplete experiment: {directory}")
        return None

    metrics = pd.read_csv(metrics_path)
    test = read_json(test_path)

    val_values = pd.to_numeric(metrics["val_accuracy"], errors="coerce")
    train_values = pd.to_numeric(metrics["train_accuracy"], errors="coerce")

    result = {
        "tag": tag,
        "alpha": alpha,
        "mean_js_divergence": mean_js,
        "rounds": int(test.get("rounds", metrics["round"].max())),
        "best_round": test.get("best_round"),
        "convergence_round_95pct": convergence_round(metrics, 0.95),
        "best_val_accuracy": test.get("best_val_accuracy"),
        "final_train_accuracy": float(train_values.dropna().iloc[-1])
        if train_values.notna().any() else np.nan,
        "test_accuracy": test.get("test_accuracy"),
        "macro_class_accuracy": test.get("macro_class_accuracy"),
        "test_loss": test.get("test_loss"),
        "estimated_total_communication_bytes": test.get(
            "estimated_total_communication_bytes"
        ),
        "directory": str(directory.resolve()),
    }
    return result


def plot_curves(experiments, output_dir: Path):
    # Accuracy
    fig = plt.figure(figsize=(9, 5.5))
    plotted = False
    for tag, directory in experiments:
        path = directory / "metrics.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "round" not in df or "val_accuracy" not in df:
            continue
        plt.plot(df["round"], df["val_accuracy"], label=tag)
        plotted = True
    if plotted:
        plt.xlabel("Communication Round")
        plt.ylabel("Validation Accuracy")
        plt.title("Stage 1B FedAvg Validation Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "validation_accuracy_comparison.png", dpi=200)
    plt.close(fig)

    # Loss
    fig = plt.figure(figsize=(9, 5.5))
    plotted = False
    for tag, directory in experiments:
        path = directory / "metrics.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "round" not in df or "val_loss" not in df:
            continue
        plt.plot(df["round"], df["val_loss"], label=tag)
        plotted = True
    if plotted:
        plt.xlabel("Communication Round")
        plt.ylabel("Validation Loss")
        plt.title("Stage 1B FedAvg Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "validation_loss_comparison.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    partition_map = {}
    if args.partition_summary.is_file():
        p = pd.read_csv(args.partition_summary)
        for row in p.to_dict("records"):
            alpha = float(row["alpha"])
            if np.isclose(alpha, 10.0):
                tag = "alpha10"
            elif np.isclose(alpha, 1.0):
                tag = "alpha1"
            elif np.isclose(alpha, 0.5):
                tag = "alpha05"
            elif np.isclose(alpha, 0.1):
                tag = "alpha01"
            else:
                continue
            partition_map[tag] = {
                "alpha": alpha,
                "mean_js": float(row["mean_js_divergence"]),
            }

    experiments = []

    if args.stage1a_iid_dir.is_dir():
        experiments.append(("iid", args.stage1a_iid_dir.resolve()))

    for tag in ["alpha10", "alpha1", "alpha05", "alpha01"]:
        directory = (
            args.stage1b_root
            / f"fedavg_{tag}_seed20260720"
        ).resolve()
        experiments.append((tag, directory))

    rows = []
    for tag, directory in experiments:
        info = partition_map.get(tag, {})
        row = load_experiment(
            tag,
            directory,
            mean_js=info.get("mean_js", 0.0 if tag == "iid" else None),
            alpha=info.get("alpha"),
        )
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No completed experiments found.")

    summary = pd.DataFrame(rows)
    summary["order"] = summary["tag"].map(
        {name: index for index, name in enumerate(ORDER)}
    )
    summary = summary.sort_values("order").drop(columns="order")
    summary.to_csv(output_dir / "stage1b_fedavg_summary.csv", index=False)

    print(summary.to_string(index=False))

    plot_curves(experiments, output_dir)

    # Test accuracy bar chart
    chart = summary.dropna(subset=["test_accuracy"])
    if not chart.empty:
        fig = plt.figure(figsize=(8, 5))
        plt.bar(chart["tag"], chart["test_accuracy"])
        plt.xlabel("Partition")
        plt.ylabel("Test Accuracy")
        plt.title("FedAvg Test Accuracy under Different Non-IID Levels")
        plt.tight_layout()
        fig.savefig(output_dir / "test_accuracy_comparison.png", dpi=200)
        plt.close(fig)

    # Mean JS vs accuracy
    scatter = summary.dropna(
        subset=["mean_js_divergence", "test_accuracy"]
    )
    if len(scatter) >= 2:
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(
            scatter["mean_js_divergence"],
            scatter["test_accuracy"],
        )
        for row in scatter.itertuples():
            plt.annotate(
                row.tag,
                (row.mean_js_divergence, row.test_accuracy),
            )
        plt.xlabel("Mean JS Divergence")
        plt.ylabel("Test Accuracy")
        plt.title("Data Heterogeneity vs FedAvg Test Accuracy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "js_vs_test_accuracy.png", dpi=200)
        plt.close(fig)

    print(f"\nSaved comparison outputs to: {output_dir}")


if __name__ == "__main__":
    main()
