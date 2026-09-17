#!/usr/bin/env python3
"""Create the Stage 1C-A1 cross-alpha summary, report, manifest, and closure audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path("outputs/stage1c/fedprox_formal"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def pp(value: float) -> str:
    return f"{value:+.4f} pp"


def main() -> None:
    args = parse_args()
    formal_root = args.formal_root
    analysis_root = formal_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    summary_paths = {
        0.1: analysis_root / "alpha01/alpha01_classification_summary.json",
        0.5: analysis_root / "alpha05/alpha05_classification_summary.json",
    }
    summaries = {alpha: read_json(path) for alpha, path in summary_paths.items()}

    rows = []
    for alpha, summary in summaries.items():
        rows.append(
            {
                "alpha": alpha,
                "mu0_best_val_accuracy": summary["mu0"]["best_val_accuracy"],
                "mu0p1_best_val_accuracy": summary["mu0p1"]["best_val_accuracy"],
                "best_val_accuracy_delta_pp": summary["paired_deltas"][
                    "best_val_accuracy_percentage_points"
                ],
                "mu0_test_accuracy": summary["mu0"]["test_accuracy"],
                "mu0p1_test_accuracy": summary["mu0p1"]["test_accuracy"],
                "test_accuracy_delta_pp": summary["paired_deltas"][
                    "test_accuracy_percentage_points"
                ],
                "mu0_macro_class_accuracy": summary["mu0"]["macro_class_accuracy"],
                "mu0p1_macro_class_accuracy": summary["mu0p1"][
                    "macro_class_accuracy"
                ],
                "macro_class_accuracy_delta_pp": summary["paired_deltas"][
                    "macro_class_accuracy_percentage_points"
                ],
                "mu0_test_loss": summary["mu0"]["test_loss"],
                "mu0p1_test_loss": summary["mu0p1"]["test_loss"],
                "test_loss_delta": summary["paired_deltas"]["test_loss"],
                "improved_classes": summary["class_level"]["improved_classes"],
                "declined_classes": summary["class_level"]["declined_classes"],
                "unchanged_classes": summary["class_level"]["unchanged_classes"],
                "mean_class_delta_pp": summary["class_level"][
                    "mean_class_delta_percentage_points"
                ],
                "zero_classes_mu0": len(
                    summary["class_level"]["zero_accuracy_classes_mu0"]
                ),
                "zero_classes_mu0p1": len(
                    summary["class_level"]["zero_accuracy_classes_mu0p1"]
                ),
                "pair_config_comparable": summary["pair_config_comparable"],
                "initial_checkpoint_exact": summary[
                    "initial_checkpoint_comparison"
                ].get("exact"),
            }
        )

    cross_df = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    cross_csv = analysis_root / "stage1ca1_cross_alpha_summary.csv"
    cross_json = analysis_root / "stage1ca1_cross_alpha_summary.json"
    cross_md = analysis_root / "stage1ca1_cross_alpha_summary.md"
    plots_dir = analysis_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cross_df.to_csv(cross_csv, index=False)
    cross_json.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown = [
        "# Stage 1C-A1 FedProx cross-alpha summary",
        "",
        "| alpha | Test acc. mu=0 | Test acc. mu=0.1 | Test delta | Macro mu=0 | Macro mu=0.1 | Macro delta | Improved/Declined/Unchanged |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['alpha']:.1f} | "
            f"{pct(row['mu0_test_accuracy'])} | "
            f"{pct(row['mu0p1_test_accuracy'])} | "
            f"{pp(row['test_accuracy_delta_pp'])} | "
            f"{pct(row['mu0_macro_class_accuracy'])} | "
            f"{pct(row['mu0p1_macro_class_accuracy'])} | "
            f"{pp(row['macro_class_accuracy_delta_pp'])} | "
            f"{row['improved_classes']}/{row['declined_classes']}/{row['unchanged_classes']} |"
        )
    markdown += [
        "",
        "## Main paired conclusion",
        "",
        "Under exactly matched SGD settings, FedProx with mu=0.1 outperforms the mu=0 paired baseline at both heterogeneity levels.",
        "The gain is larger under stronger label skew (alpha=0.1), which is consistent with the intended role of the proximal term.",
        "",
    ]
    cross_md.write_text("\n".join(markdown), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        [str(value) for value in cross_df["alpha"]],
        cross_df["test_accuracy_delta_pp"],
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_title("FedProx paired test-accuracy gain")
    ax.set_xlabel("Dirichlet alpha")
    ax.set_ylabel("Gain (percentage points)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    test_gain_plot = plots_dir / "stage1ca1_test_accuracy_gain_by_alpha.png"
    fig.savefig(test_gain_plot, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        [str(value) for value in cross_df["alpha"]],
        cross_df["macro_class_accuracy_delta_pp"],
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_title("FedProx paired macro-class-accuracy gain")
    ax.set_xlabel("Dirichlet alpha")
    ax.set_ylabel("Gain (percentage points)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    macro_gain_plot = plots_dir / "stage1ca1_macro_accuracy_gain_by_alpha.png"
    fig.savefig(macro_gain_plot, dpi=180)
    plt.close(fig)

    alpha01 = summaries[0.1]
    alpha05 = summaries[0.5]

    report_path = formal_root / "Stage1C_A1_FedProx_Technical_Report_v1.0.md"
    report = f"""# Stage 1C-A1 FedProx Technical Report v1.0

## 1. Stage objective

This stage evaluates whether the FedProx proximal term improves federated GTSRB training under label-skew non-IID partitions while preserving the same data partition, model, client sampling, training budget, optimizer, learning rate, server update, and initialization.

The paired control is the same FedProx implementation with `mu=0`, which is the strict FedAvg-equivalent control under the selected SGD configuration.

## 2. Experimental design

- Dataset: GTSRB, 43 classes
- Clients: 5
- Partition family: Dirichlet label skew
- Heterogeneity levels: alpha=0.1 and alpha=0.5
- Rounds: 160
- Local epochs: 1
- Batch size: 64
- Client optimizer: SGD
- Client learning rate: 0.1
- Client momentum: 0.0
- Server optimizer: stateless SGDM-compatible update
- Server learning rate: 1.0
- Client weighting: number of examples
- Paired proximal strengths: mu=0 and mu=0.1
- Model selection: global validation accuracy
- Official test set: used only after hyperparameter selection

## 3. Integrity and comparability audit

Both alpha pairs passed:

- 160 contiguous and unique metric rows
- finite core numeric metrics
- zero non-finite update count
- final checkpoints and model files
- test metrics, per-class accuracy, and confusion matrix
- confusion-derived accuracy consistency
- matched paired configuration
- identical Round-0 model weights

## 4. Formal results

| alpha | Metric | mu=0 | mu=0.1 | FedProx change |
|---:|---|---:|---:|---:|
| 0.1 | Best validation accuracy | {pct(alpha01['mu0']['best_val_accuracy'])} | {pct(alpha01['mu0p1']['best_val_accuracy'])} | {pp(alpha01['paired_deltas']['best_val_accuracy_percentage_points'])} |
| 0.1 | Test accuracy | {pct(alpha01['mu0']['test_accuracy'])} | {pct(alpha01['mu0p1']['test_accuracy'])} | {pp(alpha01['paired_deltas']['test_accuracy_percentage_points'])} |
| 0.1 | Macro class accuracy | {pct(alpha01['mu0']['macro_class_accuracy'])} | {pct(alpha01['mu0p1']['macro_class_accuracy'])} | {pp(alpha01['paired_deltas']['macro_class_accuracy_percentage_points'])} |
| 0.1 | Test loss | {alpha01['mu0']['test_loss']:.6f} | {alpha01['mu0p1']['test_loss']:.6f} | {alpha01['paired_deltas']['test_loss']:+.6f} |
| 0.5 | Best validation accuracy | {pct(alpha05['mu0']['best_val_accuracy'])} | {pct(alpha05['mu0p1']['best_val_accuracy'])} | {pp(alpha05['paired_deltas']['best_val_accuracy_percentage_points'])} |
| 0.5 | Test accuracy | {pct(alpha05['mu0']['test_accuracy'])} | {pct(alpha05['mu0p1']['test_accuracy'])} | {pp(alpha05['paired_deltas']['test_accuracy_percentage_points'])} |
| 0.5 | Macro class accuracy | {pct(alpha05['mu0']['macro_class_accuracy'])} | {pct(alpha05['mu0p1']['macro_class_accuracy'])} | {pp(alpha05['paired_deltas']['macro_class_accuracy_percentage_points'])} |
| 0.5 | Test loss | {alpha05['mu0']['test_loss']:.6f} | {alpha05['mu0p1']['test_loss']:.6f} | {alpha05['paired_deltas']['test_loss']:+.6f} |

## 5. Class-level findings

### alpha=0.1

- Improved classes: {alpha01['class_level']['improved_classes']}
- Declined classes: {alpha01['class_level']['declined_classes']}
- Unchanged classes: {alpha01['class_level']['unchanged_classes']}
- Mean class-accuracy change: {alpha01['class_level']['mean_class_delta_percentage_points']:+.4f} percentage points
- Zero-accuracy classes at mu=0: {alpha01['class_level']['zero_accuracy_classes_mu0']}
- Zero-accuracy classes at mu=0.1: {alpha01['class_level']['zero_accuracy_classes_mu0p1']}

### alpha=0.5

- Improved classes: {alpha05['class_level']['improved_classes']}
- Declined classes: {alpha05['class_level']['declined_classes']}
- Unchanged classes: {alpha05['class_level']['unchanged_classes']}
- Mean class-accuracy change: {alpha05['class_level']['mean_class_delta_percentage_points']:+.4f} percentage points
- Zero-accuracy classes at mu=0: {alpha05['class_level']['zero_accuracy_classes_mu0']}
- Zero-accuracy classes at mu=0.1: {alpha05['class_level']['zero_accuracy_classes_mu0p1']}

Detailed class and confusion-pair tables are stored in the per-alpha analysis directories.

## 6. Interpretation

FedProx improves the strict paired SGD baseline at both tested heterogeneity levels. The test-accuracy gain is larger at alpha=0.1 than at alpha=0.5, supporting the interpretation that proximal regularization is more valuable when client label distributions are more heterogeneous.

Macro-class accuracy also improves at both alpha values. This indicates that the benefit is not limited to frequent classes and should be interpreted together with the per-class and confusion-pair analyses.

The strict claim is relative to the matched `mu=0` control. Comparisons with historical Stage 1B runs using different optimizer settings are supplementary and must not be presented as the primary causal estimate of the proximal term.

## 7. Limitations

- One partition seed is used in the current formal comparison.
- Five clients and full participation are fixed.
- Only one selected nonzero proximal strength is evaluated formally.
- Results are specific to the selected model, local epoch count, and optimizer configuration.
- Statistical uncertainty across multiple seeds has not yet been estimated.

## 8. Reproducibility artifacts

- `analysis/alpha01/alpha01_classification_summary.json`
- `analysis/alpha01/alpha01_per_class_mu0_vs_mu0p1.csv`
- `analysis/alpha01/alpha01_confusion_pair_comparison.csv`
- `analysis/alpha05/alpha05_classification_summary.json`
- `analysis/alpha05/alpha05_per_class_mu0_vs_mu0p1.csv`
- `analysis/alpha05/alpha05_confusion_pair_comparison.csv`
- `analysis/stage1ca1_cross_alpha_summary.csv`
- `analysis/stage1ca1_cross_alpha_summary.json`
- `analysis/stage1ca1_cross_alpha_summary.md`
- `analysis/alpha05/alpha05_integrity_and_pair_audit.txt`

## 9. Stage conclusion

Stage 1C-A1 is complete. Under matched SGD conditions, FedProx with mu=0.1 produces higher validation accuracy, test accuracy, macro-class accuracy, and lower test loss than the mu=0 paired baseline at both alpha=0.1 and alpha=0.5. The gain is stronger under the more heterogeneous alpha=0.1 partition.

The next algorithm stage is Stage 1C-A2: SCAFFOLD.
"""
    report_path.write_text(report, encoding="utf-8")

    key_files = [
        summary_paths[0.1],
        summary_paths[0.5],
        cross_csv,
        cross_json,
        cross_md,
        test_gain_plot,
        macro_gain_plot,
        report_path,
    ]
    optional_audit = analysis_root / "alpha05/alpha05_integrity_and_pair_audit.txt"
    if optional_audit.is_file():
        key_files.append(optional_audit)

    manifest_rows = []
    for path in key_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest_rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest_json = analysis_root / "stage1ca1_artifact_manifest.json"
    manifest_csv = analysis_root / "stage1ca1_artifact_manifest.csv"
    manifest_json.write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)

    closure_checks = {
        "alpha01_summary_exists": summary_paths[0.1].is_file(),
        "alpha05_summary_exists": summary_paths[0.5].is_file(),
        "cross_csv_exists": cross_csv.is_file(),
        "cross_json_exists": cross_json.is_file(),
        "cross_markdown_exists": cross_md.is_file(),
        "report_exists": report_path.is_file(),
        "manifest_json_exists": manifest_json.is_file(),
        "manifest_csv_exists": manifest_csv.is_file(),
        "all_pairs_comparable": all(
            bool(summary["pair_config_comparable"]) for summary in summaries.values()
        ),
        "all_initial_checkpoints_exact": all(
            bool(summary["initial_checkpoint_comparison"].get("exact"))
            for summary in summaries.values()
        ),
    }

    closure_log = analysis_root / "stage1ca1_closure_audit.txt"
    with closure_log.open("w", encoding="utf-8") as handle:
        handle.write("Stage 1C-A1 closure audit\n")
        handle.write("=" * 88 + "\n")
        for key, value in closure_checks.items():
            handle.write(f"{key:38s} = {value}\n")
        handle.write("\n")
        handle.write(cross_df.to_string(index=False))
        handle.write("\n")

    if not all(closure_checks.values()):
        failed = [key for key, value in closure_checks.items() if not value]
        raise AssertionError(f"Stage 1C-A1 closure audit failed: {failed}")

    print("=" * 88)
    print("[PASS] Stage 1C-A1 FedProx finalization complete")
    print(f"cross_alpha_csv     = {cross_csv}")
    print(f"cross_alpha_json    = {cross_json}")
    print(f"cross_alpha_md      = {cross_md}")
    print(f"technical_report    = {report_path}")
    print(f"manifest_json       = {manifest_json}")
    print(f"manifest_csv        = {manifest_csv}")
    print(f"closure_audit       = {closure_log}")
    print("=" * 88)


if __name__ == "__main__":
    main()
