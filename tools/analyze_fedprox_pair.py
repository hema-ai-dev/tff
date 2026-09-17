#!/usr/bin/env python3
"""Analyze one paired FedProx experiment: mu=0 versus mu=0.1.

Outputs:
- per-class comparison CSV
- confusion-pair comparison CSV
- classification summary JSON
- pair summary Markdown
- learning-curve and class-delta plots

The loader is intentionally tolerant of several confusion_matrix.csv layouts:
plain 43x43, header row, index column, or both.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUM_CLASSES = 43


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu0-dir", required=True, type=Path)
    parser.add_argument("--mu0p1-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--prefix", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    copied = df.copy()
    copied.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in copied.columns
    ]
    return copied


def find_column(columns: list[str], aliases: list[str], contains: str | None = None) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    if contains is not None:
        candidates = [column for column in columns if contains in column]
        if candidates:
            return candidates[0]
    return None


def confusion_candidates(path: Path, expected_total: int | None) -> list[tuple[float, np.ndarray]]:
    frames: list[pd.DataFrame] = []
    for kwargs in (
        {"header": None},
        {},
        {"index_col": 0},
    ):
        try:
            frames.append(pd.read_csv(path, **kwargs))
        except Exception:
            pass

    scored: list[tuple[float, np.ndarray]] = []
    for frame in frames:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
        array = numeric.to_numpy(dtype=float)
        rows, cols = array.shape

        if rows < NUM_CLASSES or cols < NUM_CLASSES:
            continue

        for row_start in range(rows - NUM_CLASSES + 1):
            for col_start in range(cols - NUM_CLASSES + 1):
                block = array[
                    row_start : row_start + NUM_CLASSES,
                    col_start : col_start + NUM_CLASSES,
                ]
                if not np.isfinite(block).all():
                    continue
                if (block < 0).any():
                    continue

                integer_error = float(np.abs(block - np.rint(block)).sum())
                total = float(block.sum())
                total_error = 0.0 if expected_total is None else abs(total - expected_total)
                score = total_error * 1000.0 + integer_error
                scored.append((score, np.rint(block).astype(np.int64)))

    return scored


def load_confusion_matrix(path: Path, expected_total: int | None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    candidates = confusion_candidates(path, expected_total)
    if not candidates:
        raise AssertionError(f"无法把 {path} 解释为 {NUM_CLASSES}x{NUM_CLASSES} 混淆矩阵。")
    candidates.sort(key=lambda item: item[0])
    matrix = candidates[0][1]

    if expected_total is not None and int(matrix.sum()) != int(expected_total):
        raise AssertionError(
            f"{path} 解析后的样本总数为 {matrix.sum()}，预期为 {expected_total}。"
        )
    return matrix


def load_per_class(path: Path, confusion: np.ndarray) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    raw = normalize_columns(pd.read_csv(path))
    columns = list(raw.columns)

    class_column = find_column(
        columns,
        ["class_id", "class", "label", "true_class", "category_id"],
    )
    accuracy_column = find_column(
        columns,
        ["class_accuracy", "accuracy", "per_class_accuracy", "test_accuracy"],
        contains="accuracy",
    )
    count_column = find_column(
        columns,
        ["num_test_samples", "support", "sample_count", "count", "n"],
    )

    if class_column is None:
        raw = raw.reset_index().rename(columns={"index": "class_id"})
        class_column = "class_id"

    result = pd.DataFrame()
    result["class_id"] = pd.to_numeric(raw[class_column], errors="coerce")
    result = result.dropna(subset=["class_id"])
    result["class_id"] = result["class_id"].astype(int)

    row_counts = confusion.sum(axis=1).astype(int)
    derived_accuracy = np.divide(
        np.diag(confusion),
        row_counts,
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=row_counts > 0,
    )

    if count_column is not None:
        result["num_test_samples"] = pd.to_numeric(raw.loc[result.index, count_column], errors="coerce")
    else:
        result["num_test_samples"] = result["class_id"].map(
            {index: int(value) for index, value in enumerate(row_counts)}
        )

    if accuracy_column is not None:
        result["class_accuracy"] = pd.to_numeric(raw.loc[result.index, accuracy_column], errors="coerce")
    else:
        result["class_accuracy"] = result["class_id"].map(
            {index: float(value) for index, value in enumerate(derived_accuracy)}
        )

    result = result.sort_values("class_id").drop_duplicates("class_id").reset_index(drop=True)

    if result["class_id"].tolist() != list(range(NUM_CLASSES)):
        raise AssertionError(
            f"{path} 没有完整覆盖 class_id 0..{NUM_CLASSES - 1}。"
        )

    # Use confusion-derived counts as the authoritative count and verify accuracy.
    result["num_test_samples"] = row_counts
    confusion_accuracy = derived_accuracy
    csv_accuracy = result["class_accuracy"].to_numpy(dtype=float)

    if not np.allclose(csv_accuracy, confusion_accuracy, atol=1e-9, rtol=1e-7):
        max_diff = float(np.max(np.abs(csv_accuracy - confusion_accuracy)))
        raise AssertionError(
            f"{path} 与混淆矩阵推导的类别准确率不一致，max_abs_diff={max_diff}"
        )

    return result


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    metrics = normalize_columns(pd.read_csv(path))
    if "round" not in metrics.columns:
        raise AssertionError(f"{path} 缺少 round 列。")
    metrics["round"] = pd.to_numeric(metrics["round"], errors="raise").astype(int)
    return metrics


def compare_initial_checkpoints(mu0_dir: Path, mu0p1_dir: Path) -> dict[str, Any]:
    left_path = mu0_dir / "checkpoints/model_weights_round_000000.npz"
    right_path = mu0p1_dir / "checkpoints/model_weights_round_000000.npz"

    result: dict[str, Any] = {
        "available": left_path.is_file() and right_path.is_file(),
        "keys_equal": None,
        "exact": None,
        "max_abs_diff": None,
    }
    if not result["available"]:
        return result

    with np.load(left_path, allow_pickle=False) as left_payload:
        left = {key: np.asarray(left_payload[key]) for key in left_payload.files}
    with np.load(right_path, allow_pickle=False) as right_payload:
        right = {key: np.asarray(right_payload[key]) for key in right_payload.files}

    keys_equal = set(left) == set(right)
    exact = keys_equal
    max_abs_diff = 0.0

    if keys_equal:
        for key in sorted(left):
            if left[key].shape != right[key].shape:
                exact = False
                max_abs_diff = math.inf
                break
            exact = exact and np.array_equal(left[key], right[key])
            if np.issubdtype(left[key].dtype, np.number):
                diff = np.abs(
                    left[key].astype(np.float64) - right[key].astype(np.float64)
                )
                if diff.size:
                    max_abs_diff = max(max_abs_diff, float(diff.max()))

    result.update(
        {
            "keys_equal": bool(keys_equal),
            "exact": bool(exact),
            "max_abs_diff": float(max_abs_diff),
        }
    )
    return result


def compare_configs(mu0_config: dict[str, Any], mu0p1_config: dict[str, Any]) -> dict[str, Any]:
    comparable_keys = [
        "partition_dir",
        "partition_type",
        "dirichlet_alpha",
        "partition_base_seed",
        "partition_derived_seed",
        "num_clients",
        "rounds",
        "local_epochs",
        "batch_size",
        "client_optimizer",
        "client_learning_rate",
        "client_momentum",
        "server_optimizer",
        "server_learning_rate",
        "client_weighting",
    ]

    rows = []
    all_equal = True
    for key in comparable_keys:
        left = mu0_config.get(key)
        right = mu0p1_config.get(key)
        equal = left == right
        all_equal = all_equal and equal
        rows.append({"key": key, "mu0": left, "mu0p1": right, "equal": equal})

    return {"all_equal": all_equal, "rows": rows}


def safe_number(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    return None if value is None else float(value)


def plot_line(
    mu0_metrics: pd.DataFrame,
    mu0p1_metrics: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> bool:
    if column not in mu0_metrics.columns or column not in mu0p1_metrics.columns:
        return False

    left = pd.to_numeric(mu0_metrics[column], errors="coerce")
    right = pd.to_numeric(mu0p1_metrics[column], errors="coerce")
    if left.notna().sum() == 0 or right.notna().sum() == 0:
        return False

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mu0_metrics["round"], left, label="mu=0")
    ax.plot(mu0p1_metrics["round"], right, label="mu=0.1")
    ax.set_title(title)
    ax.set_xlabel("Federated round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mu0_summary = read_json(args.mu0_dir / "run_summary.json")
    mu0p1_summary = read_json(args.mu0p1_dir / "run_summary.json")
    mu0_config = read_json(args.mu0_dir / "config.json")
    mu0p1_config = read_json(args.mu0p1_dir / "config.json")

    expected_total_mu0 = int(mu0_summary["num_test_samples"])
    expected_total_mu0p1 = int(mu0p1_summary["num_test_samples"])

    confusion_mu0 = load_confusion_matrix(
        args.mu0_dir / "confusion_matrix.csv",
        expected_total_mu0,
    )
    confusion_mu0p1 = load_confusion_matrix(
        args.mu0p1_dir / "confusion_matrix.csv",
        expected_total_mu0p1,
    )

    per_class_mu0 = load_per_class(
        args.mu0_dir / "per_class_accuracy.csv",
        confusion_mu0,
    )
    per_class_mu0p1 = load_per_class(
        args.mu0p1_dir / "per_class_accuracy.csv",
        confusion_mu0p1,
    )

    if not np.array_equal(
        per_class_mu0["num_test_samples"].to_numpy(),
        per_class_mu0p1["num_test_samples"].to_numpy(),
    ):
        raise AssertionError("两个配对实验的类别测试样本数不一致。")

    comparison = per_class_mu0.rename(
        columns={
            "num_test_samples": "num_test_samples_mu0",
            "class_accuracy": "class_accuracy_mu0",
        }
    ).merge(
        per_class_mu0p1.rename(
            columns={
                "num_test_samples": "num_test_samples_mu0p1",
                "class_accuracy": "class_accuracy_mu0p1",
            }
        ),
        on="class_id",
        validate="one_to_one",
    )

    comparison["accuracy_delta_percentage_points"] = 100.0 * (
        comparison["class_accuracy_mu0p1"] - comparison["class_accuracy_mu0"]
    )
    comparison["correct_mu0"] = np.rint(
        comparison["class_accuracy_mu0"] * comparison["num_test_samples_mu0"]
    ).astype(int)
    comparison["correct_mu0p1"] = np.rint(
        comparison["class_accuracy_mu0p1"] * comparison["num_test_samples_mu0p1"]
    ).astype(int)
    comparison["correct_delta"] = comparison["correct_mu0p1"] - comparison["correct_mu0"]

    per_class_path = args.output_dir / f"{args.prefix}_per_class_mu0_vs_mu0p1.csv"
    comparison.to_csv(per_class_path, index=False)

    error_rows = []
    for true_class in range(NUM_CLASSES):
        for predicted_class in range(NUM_CLASSES):
            if true_class == predicted_class:
                continue
            errors_mu0 = int(confusion_mu0[true_class, predicted_class])
            errors_mu0p1 = int(confusion_mu0p1[true_class, predicted_class])
            change = errors_mu0p1 - errors_mu0
            error_rows.append(
                {
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "errors_mu0": errors_mu0,
                    "errors_mu0p1": errors_mu0p1,
                    "error_change_mu0p1_minus_mu0": change,
                    "errors_reduced_by_fedprox": -change,
                }
            )

    confusion_comparison = pd.DataFrame(error_rows).sort_values(
        ["errors_reduced_by_fedprox", "errors_mu0"],
        ascending=[False, False],
    )
    confusion_path = args.output_dir / f"{args.prefix}_confusion_pair_comparison.csv"
    confusion_comparison.to_csv(confusion_path, index=False)

    tolerance = 1e-12
    deltas = comparison["accuracy_delta_percentage_points"]
    improved = comparison[deltas > tolerance]
    declined = comparison[deltas < -tolerance]
    unchanged = comparison[deltas.abs() <= tolerance]

    config_comparison = compare_configs(mu0_config, mu0p1_config)
    initial_checkpoint = compare_initial_checkpoints(args.mu0_dir, args.mu0p1_dir)

    confusion_accuracy_mu0 = float(np.trace(confusion_mu0) / confusion_mu0.sum())
    confusion_accuracy_mu0p1 = float(np.trace(confusion_mu0p1) / confusion_mu0p1.sum())

    for label, summary, confusion_accuracy in (
        ("mu0", mu0_summary, confusion_accuracy_mu0),
        ("mu0p1", mu0p1_summary, confusion_accuracy_mu0p1),
    ):
        reported = safe_number(summary, "overall_accuracy_from_predictions")
        if reported is None:
            reported = safe_number(summary, "test_accuracy")
        if reported is None or not math.isclose(
            reported, confusion_accuracy, rel_tol=1e-8, abs_tol=1e-10
        ):
            raise AssertionError(
                f"{label} 的混淆矩阵准确率 {confusion_accuracy} 与摘要 {reported} 不一致。"
            )

    metrics_mu0 = load_metrics(args.mu0_dir / "metrics.csv")
    metrics_mu0p1 = load_metrics(args.mu0p1_dir / "metrics.csv")

    plot_files: list[str] = []

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(comparison["class_id"], comparison["accuracy_delta_percentage_points"])
    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"FedProx class-accuracy change, alpha={args.alpha:g}")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Accuracy change (percentage points)")
    ax.set_xticks(range(NUM_CLASSES))
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    class_delta_plot = plots_dir / f"{args.prefix}_per_class_delta.png"
    fig.savefig(class_delta_plot, dpi=180)
    plt.close(fig)
    plot_files.append(str(class_delta_plot))

    line_specs = [
        ("val_accuracy", "Validation accuracy", "Accuracy", "val_accuracy_curve"),
        ("val_loss", "Validation loss", "Loss", "val_loss_curve"),
        ("train_accuracy", "Training accuracy", "Accuracy", "train_accuracy_curve"),
        ("train_loss", "Training loss", "Loss", "train_loss_curve"),
        (
            "global_model_delta_norm",
            "Global model update norm",
            "L2 norm",
            "global_delta_norm_curve",
        ),
        (
            "relative_global_model_delta_norm",
            "Relative global model update norm",
            "Relative L2 norm",
            "relative_global_delta_norm_curve",
        ),
    ]
    for column, title, ylabel, suffix in line_specs:
        destination = plots_dir / f"{args.prefix}_{suffix}.png"
        if plot_line(
            metrics_mu0,
            metrics_mu0p1,
            column,
            f"{title}, alpha={args.alpha:g}",
            ylabel,
            destination,
        ):
            plot_files.append(str(destination))

    confusion_delta = confusion_mu0p1.astype(float) - confusion_mu0.astype(float)
    np.fill_diagonal(confusion_delta, 0.0)
    limit = float(np.max(np.abs(confusion_delta))) if confusion_delta.size else 1.0
    if limit == 0.0:
        limit = 1.0
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(
        confusion_delta,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_title(f"Off-diagonal confusion change (mu=0.1 minus mu=0), alpha={args.alpha:g}")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.colorbar(image, ax=ax, label="Error-count change")
    fig.tight_layout()
    confusion_plot = plots_dir / f"{args.prefix}_confusion_delta_heatmap.png"
    fig.savefig(confusion_plot, dpi=180)
    plt.close(fig)
    plot_files.append(str(confusion_plot))

    def metric_delta(key: str, scale: float = 1.0) -> float | None:
        left = safe_number(mu0_summary, key)
        right = safe_number(mu0p1_summary, key)
        if left is None or right is None:
            return None
        return scale * (right - left)

    summary = {
        "alpha": float(args.alpha),
        "mu0_dir": str(args.mu0_dir),
        "mu0p1_dir": str(args.mu0p1_dir),
        "pair_config_comparable": config_comparison["all_equal"],
        "config_comparison": config_comparison,
        "initial_checkpoint_comparison": initial_checkpoint,
        "mu0": {
            "proximal_strength": safe_number(mu0_summary, "proximal_strength"),
            "best_round": int(mu0_summary["best_round"]),
            "best_val_accuracy": safe_number(mu0_summary, "best_val_accuracy"),
            "test_loss": safe_number(mu0_summary, "test_loss"),
            "test_accuracy": safe_number(mu0_summary, "test_accuracy"),
            "overall_accuracy": safe_number(mu0_summary, "overall_accuracy_from_predictions"),
            "macro_class_accuracy": safe_number(mu0_summary, "macro_class_accuracy"),
            "num_test_samples": int(mu0_summary["num_test_samples"]),
            "confusion_accuracy": confusion_accuracy_mu0,
        },
        "mu0p1": {
            "proximal_strength": safe_number(mu0p1_summary, "proximal_strength"),
            "best_round": int(mu0p1_summary["best_round"]),
            "best_val_accuracy": safe_number(mu0p1_summary, "best_val_accuracy"),
            "test_loss": safe_number(mu0p1_summary, "test_loss"),
            "test_accuracy": safe_number(mu0p1_summary, "test_accuracy"),
            "overall_accuracy": safe_number(mu0p1_summary, "overall_accuracy_from_predictions"),
            "macro_class_accuracy": safe_number(mu0p1_summary, "macro_class_accuracy"),
            "num_test_samples": int(mu0p1_summary["num_test_samples"]),
            "confusion_accuracy": confusion_accuracy_mu0p1,
        },
        "paired_deltas": {
            "best_val_accuracy_percentage_points": metric_delta(
                "best_val_accuracy", 100.0
            ),
            "test_accuracy_percentage_points": metric_delta(
                "test_accuracy", 100.0
            ),
            "macro_class_accuracy_percentage_points": metric_delta(
                "macro_class_accuracy", 100.0
            ),
            "test_loss": metric_delta("test_loss"),
        },
        "class_level": {
            "zero_accuracy_classes_mu0": comparison.loc[
                comparison["class_accuracy_mu0"] <= tolerance, "class_id"
            ].astype(int).tolist(),
            "zero_accuracy_classes_mu0p1": comparison.loc[
                comparison["class_accuracy_mu0p1"] <= tolerance, "class_id"
            ].astype(int).tolist(),
            "improved_classes": int(len(improved)),
            "declined_classes": int(len(declined)),
            "unchanged_classes": int(len(unchanged)),
            "mean_class_delta_percentage_points": float(deltas.mean()),
            "worst_class_mu0": int(
                comparison.loc[comparison["class_accuracy_mu0"].idxmin(), "class_id"]
            ),
            "worst_class_accuracy_mu0": float(comparison["class_accuracy_mu0"].min()),
            "worst_class_mu0p1": int(
                comparison.loc[comparison["class_accuracy_mu0p1"].idxmin(), "class_id"]
            ),
            "worst_class_accuracy_mu0p1": float(
                comparison["class_accuracy_mu0p1"].min()
            ),
            "top_10_improved": comparison.nlargest(
                10, "accuracy_delta_percentage_points"
            ).to_dict(orient="records"),
            "top_10_declined": comparison.nsmallest(
                10, "accuracy_delta_percentage_points"
            ).to_dict(orient="records"),
            "top_15_confusions_reduced": confusion_comparison.nlargest(
                15, "errors_reduced_by_fedprox"
            ).to_dict(orient="records"),
            "top_15_confusions_increased": confusion_comparison.nsmallest(
                15, "errors_reduced_by_fedprox"
            ).to_dict(orient="records"),
        },
        "artifacts": {
            "per_class_comparison_csv": str(per_class_path),
            "confusion_pair_comparison_csv": str(confusion_path),
            "plots": plot_files,
        },
    }

    summary_path = args.output_dir / f"{args.prefix}_classification_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown_lines = [
        f"# FedProx paired analysis: alpha={args.alpha:g}",
        "",
        "## Paired global metrics",
        "",
        "| Metric | mu=0 | mu=0.1 | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| Best validation accuracy | "
            f"{100*summary['mu0']['best_val_accuracy']:.4f}% | "
            f"{100*summary['mu0p1']['best_val_accuracy']:.4f}% | "
            f"{summary['paired_deltas']['best_val_accuracy_percentage_points']:+.4f} pp |"
        ),
        (
            f"| Test accuracy | "
            f"{100*summary['mu0']['test_accuracy']:.4f}% | "
            f"{100*summary['mu0p1']['test_accuracy']:.4f}% | "
            f"{summary['paired_deltas']['test_accuracy_percentage_points']:+.4f} pp |"
        ),
        (
            f"| Macro class accuracy | "
            f"{100*summary['mu0']['macro_class_accuracy']:.4f}% | "
            f"{100*summary['mu0p1']['macro_class_accuracy']:.4f}% | "
            f"{summary['paired_deltas']['macro_class_accuracy_percentage_points']:+.4f} pp |"
        ),
        (
            f"| Test loss | "
            f"{summary['mu0']['test_loss']:.6f} | "
            f"{summary['mu0p1']['test_loss']:.6f} | "
            f"{summary['paired_deltas']['test_loss']:+.6f} |"
        ),
        "",
        "## Class-level findings",
        "",
        f"- Improved classes: {len(improved)}",
        f"- Declined classes: {len(declined)}",
        f"- Unchanged classes: {len(unchanged)}",
        f"- Mean class change: {deltas.mean():+.4f} percentage points",
        (
            "- Zero-accuracy classes, mu=0: "
            f"{summary['class_level']['zero_accuracy_classes_mu0']}"
        ),
        (
            "- Zero-accuracy classes, mu=0.1: "
            f"{summary['class_level']['zero_accuracy_classes_mu0p1']}"
        ),
        "",
        "## Audit status",
        "",
        f"- Pair configuration comparable: {summary['pair_config_comparable']}",
        f"- Initial checkpoint exact: {initial_checkpoint.get('exact')}",
        f"- mu=0 confusion accuracy: {confusion_accuracy_mu0:.12f}",
        f"- mu=0.1 confusion accuracy: {confusion_accuracy_mu0p1:.12f}",
        "",
        "## Output files",
        "",
        f"- `{per_class_path}`",
        f"- `{confusion_path}`",
        f"- `{summary_path}`",
    ]
    pair_markdown_path = args.output_dir / f"{args.prefix}_pair_summary.md"
    pair_markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    if not config_comparison["all_equal"]:
        unequal = [row["key"] for row in config_comparison["rows"] if not row["equal"]]
        raise AssertionError(f"配对配置不一致：{unequal}")
    if initial_checkpoint["available"] and not initial_checkpoint["exact"]:
        raise AssertionError("两个实验的 Round 0 checkpoint 不完全一致。")

    print("=" * 88)
    print(f"[PASS] alpha={args.alpha:g} 配对分析完成")
    print(f"per_class_csv              = {per_class_path}")
    print(f"confusion_comparison_csv   = {confusion_path}")
    print(f"classification_summary     = {summary_path}")
    print(f"pair_summary_markdown      = {pair_markdown_path}")
    print(f"plots_dir                  = {plots_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
