

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RUNS = {
    "alpha05_mu0": {
        "directory": Path(
            "outputs/stage1c/fedprox_formal/alpha05/"
            "sgd_lr0p1_mu0_alpha05_160r_seed20260720"
        ),
        "expected_mu": 0.0,
    },
    "alpha05_mu0p1": {
        "directory": Path(
            "outputs/stage1c/fedprox_formal/alpha05/"
            "sgd_lr0p1_mu0p1_alpha05_160r_seed20260720"
        ),
        "expected_mu": 0.1,
    },
}

EXPECTED_ROUNDS = 160
EXPECTED_ALPHA = 0.5
EXPECTED_TEST_SAMPLES = 12630
NUM_CLASSES = 43
ATOL = 1e-6


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_value(key: str, *documents: dict[str, Any]) -> Any:
    for document in documents:
        if key in document and document[key] is not None:
            return document[key]
    return None


def finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def normalize_class_id(value: Any) -> int | None:
    """把 0、"0"、"class_0"、"pred_0" 等标签统一解析为类别编号。"""
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        class_id = int(value)
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        class_id = int(value)
    else:
        import re

        match = re.search(r"(-?\d+)\s*$", str(value).strip())
        if match is None:
            return None
        class_id = int(match.group(1))

    if 0 <= class_id < NUM_CLASSES:
        return class_id
    return None


def valid_confusion_candidate(array: np.ndarray) -> bool:
    return bool(
        array.shape == (NUM_CLASSES, NUM_CLASSES)
        and np.isfinite(array).all()
        and (array >= 0).all()
        and np.allclose(array, np.rint(array), atol=ATOL, rtol=0)
    )


def load_confusion_matrix(path: Path) -> np.ndarray:
    """读取常见的宽表或长表混淆矩阵 CSV。

    支持：
    1. 43x43 纯数值矩阵；
    2. 带行名、列名、索引列或 total 行/列的宽表；
    3. true_class / predicted_class / count 三列长表；
    4. class_0、pred_0 等字符串类别标签。
    """
    observed: list[str] = []

    # 先尝试长表格式，例如 true_class,predicted_class,count。
    normal = pd.read_csv(path)
    observed.append(
        f"header=0 shape={normal.shape}, columns={normal.columns.tolist()}"
    )

    lower_to_original = {
        str(column).strip().lower(): column for column in normal.columns
    }

    def find_named_column(candidates: list[str]) -> Any | None:
        for candidate in candidates:
            if candidate in lower_to_original:
                return lower_to_original[candidate]
        return None

    true_column = find_named_column(
        ["true_class", "true_label", "actual_class", "actual_label", "y_true"]
    )
    predicted_column = find_named_column(
        [
            "predicted_class",
            "predicted_label",
            "prediction",
            "pred_class",
            "y_pred",
        ]
    )
    count_column = find_named_column(
        ["count", "errors", "num_samples", "frequency", "value"]
    )

    if true_column is not None and predicted_column is not None:
        long_frame = normal.copy()
        long_frame["__true_id"] = long_frame[true_column].map(normalize_class_id)
        long_frame["__predicted_id"] = long_frame[predicted_column].map(
            normalize_class_id
        )

        if count_column is None:
            long_frame["__count"] = 1.0
        else:
            long_frame["__count"] = pd.to_numeric(
                long_frame[count_column], errors="coerce"
            )

        long_frame = long_frame.dropna(
            subset=["__true_id", "__predicted_id", "__count"]
        )
        if not long_frame.empty:
            pivot = long_frame.pivot_table(
                index="__true_id",
                columns="__predicted_id",
                values="__count",
                aggfunc="sum",
                fill_value=0.0,
            )
            pivot = pivot.reindex(
                index=range(NUM_CLASSES),
                columns=range(NUM_CLASSES),
                fill_value=0.0,
            )
            array = pivot.to_numpy(dtype=np.float64)
            if valid_confusion_candidate(array):
                return array

    # 再尝试宽表。header=None 能保留原始表头，便于从带标签的 CSV 中
    # 搜索真正的 43x43 数值区域。
    frames = [normal, pd.read_csv(path, header=None)]
    candidate_arrays: list[np.ndarray] = []

    for frame_index, frame in enumerate(frames):
        frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
        observed.append(f"frame[{frame_index}] cleaned_shape={frame.shape}")

        numeric = frame.apply(pd.to_numeric, errors="coerce")
        row_count, column_count = numeric.shape

        if row_count < NUM_CLASSES or column_count < NUM_CLASSES:
            continue

        # 在整个表中搜索所有可能的 43x43 数值窗口。
        for row_start in range(row_count - NUM_CLASSES + 1):
            for column_start in range(column_count - NUM_CLASSES + 1):
                window = numeric.iloc[
                    row_start : row_start + NUM_CLASSES,
                    column_start : column_start + NUM_CLASSES,
                ]
                if window.isna().any().any():
                    continue

                array = window.to_numpy(dtype=np.float64)
                if valid_confusion_candidate(array):
                    candidate_arrays.append(array)

    if candidate_arrays:
        # 优先选择样本总数与正式测试集一致的候选区域。
        exact_total = [
            array
            for array in candidate_arrays
            if np.isclose(
                float(array.sum()),
                float(EXPECTED_TEST_SAMPLES),
                atol=ATOL,
                rtol=0,
            )
        ]
        if exact_total:
            return exact_total[0]

        # 没有精确总数时返回第一个候选，后续总数审计会给出更明确错误。
        return candidate_arrays[0]

    raise AssertionError(
        f"无法把 {path} 解释为 {NUM_CLASSES}x{NUM_CLASSES} 混淆矩阵。"
        f"检测信息: {'; '.join(observed)}。"
        "请运行 `head -n 5 confusion_matrix.csv` 检查实际格式。"
    )


def choose_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise AssertionError(
        f"缺少列，候选={candidates}，实际列={frame.columns.tolist()}"
    )


def audit_one(name: str, directory: Path, expected_mu: float) -> dict[str, Any]:
    print("\n" + "=" * 96)
    print(name)
    print(directory)
    print("=" * 96)

    required = [
        directory / "config.json",
        directory / "run_state.json",
        directory / "run_summary.json",
        directory / "metrics.csv",
        directory / "test_metrics.json",
        directory / "per_class_accuracy.csv",
        directory / "confusion_matrix.csv",
        directory / "global_models" / "best_global_model.h5",
        directory / "global_models" / "last_global_model.h5",
        directory / "checkpoints" / "model_weights_round_000160.npz",
    ]

    for path in required:
        present = path.is_file()
        print(f"{str(path.relative_to(directory)):58s} present={present}")
        assert present, f"缺少文件: {path}"

    config = read_json(directory / "config.json")
    state = read_json(directory / "run_state.json")
    summary = read_json(directory / "run_summary.json")
    metrics = pd.read_csv(directory / "metrics.csv")
    per_class = pd.read_csv(directory / "per_class_accuracy.csv")
    confusion = load_confusion_matrix(directory / "confusion_matrix.csv")

    rounds = pd.to_numeric(metrics["round"], errors="raise").astype(int).tolist()
    expected_rounds = list(range(1, EXPECTED_ROUNDS + 1))

    assert len(metrics) == EXPECTED_ROUNDS
    assert rounds == expected_rounds
    assert metrics["round"].is_unique
    assert int(state["completed_round"]) == EXPECTED_ROUNDS

    core_columns = [
        "round",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    ]
    optional_diagnostics = [
        "global_weight_l2_norm_before",
        "global_weight_l2_norm_after",
        "global_model_delta_norm",
        "relative_global_model_delta_norm",
    ]
    checked_columns = core_columns + [
        column for column in optional_diagnostics if column in metrics.columns
    ]
    numeric_core = metrics[checked_columns].apply(pd.to_numeric, errors="coerce")
    assert np.isfinite(numeric_core.to_numpy(dtype=np.float64)).all()

    non_finite_columns = [
        column for column in metrics.columns if "update_non_finite" in column
    ]
    non_finite_total = 0.0
    if non_finite_columns:
        non_finite_total = float(
            metrics[non_finite_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            .sum()
        )
    assert non_finite_total == 0.0

    actual_mu = float(first_value("proximal_strength", summary, config))
    actual_alpha = float(first_value("dirichlet_alpha", summary, config))
    assert np.isclose(actual_mu, expected_mu, atol=1e-12, rtol=0)
    assert np.isclose(actual_alpha, EXPECTED_ALPHA, atol=1e-12, rtol=0)

    assert summary.get("final_test_skipped") is False

    for key in [
        "test_loss",
        "test_accuracy",
        "macro_class_accuracy",
        "overall_accuracy_from_predictions",
        "num_test_samples",
        "best_val_accuracy",
        "best_round",
    ]:
        assert summary.get(key) is not None, f"run_summary 缺少非空字段: {key}"

    for key in [
        "test_loss",
        "test_accuracy",
        "macro_class_accuracy",
        "overall_accuracy_from_predictions",
        "best_val_accuracy",
    ]:
        assert finite_number(summary[key]), f"字段不是有限数: {key}={summary[key]}"

    assert int(summary["num_test_samples"]) == EXPECTED_TEST_SAMPLES

    best_index = pd.to_numeric(metrics["val_accuracy"], errors="raise").idxmax()
    best_metric_round = int(metrics.loc[best_index, "round"])
    best_metric_accuracy = float(metrics.loc[best_index, "val_accuracy"])
    assert best_metric_round == int(summary["best_round"])
    assert np.isclose(
        best_metric_accuracy,
        float(summary["best_val_accuracy"]),
        atol=ATOL,
        rtol=0,
    )

    class_id_column = choose_column(per_class, ["class_id", "class", "label"])
    samples_column = choose_column(
        per_class,
        ["num_test_samples", "num_samples", "support", "count"],
    )
    accuracy_column = choose_column(
        per_class,
        ["class_accuracy", "accuracy", "per_class_accuracy"],
    )

    assert len(per_class) == NUM_CLASSES
    class_ids = pd.to_numeric(per_class[class_id_column], errors="raise").astype(int)
    assert class_ids.is_unique
    assert sorted(class_ids.tolist()) == list(range(NUM_CLASSES))

    class_samples = pd.to_numeric(per_class[samples_column], errors="raise")
    class_accuracy = pd.to_numeric(per_class[accuracy_column], errors="raise")
    assert int(class_samples.sum()) == EXPECTED_TEST_SAMPLES
    assert np.isfinite(class_accuracy.to_numpy(dtype=np.float64)).all()
    assert ((class_accuracy >= 0.0) & (class_accuracy <= 1.0)).all()

    assert confusion.shape == (NUM_CLASSES, NUM_CLASSES)
    assert np.isfinite(confusion).all()
    assert (confusion >= 0).all()
    assert int(round(float(confusion.sum()))) == EXPECTED_TEST_SAMPLES

    confusion_accuracy = float(np.trace(confusion) / confusion.sum())
    macro_from_classes = float(class_accuracy.mean())

    assert np.isclose(
        confusion_accuracy,
        float(summary["overall_accuracy_from_predictions"]),
        atol=ATOL,
        rtol=0,
    )
    assert np.isclose(
        confusion_accuracy,
        float(summary["test_accuracy"]),
        atol=ATOL,
        rtol=0,
    )
    assert np.isclose(
        macro_from_classes,
        float(summary["macro_class_accuracy"]),
        atol=ATOL,
        rtol=0,
    )

    print(f"completed_round             = {state['completed_round']}")
    print(f"metrics_rows                = {len(metrics)}")
    print(f"rounds_contiguous           = {rounds == expected_rounds}")
    print(f"all_core_finite             = True")
    print(f"non_finite_update_total     = {non_finite_total}")
    print(f"dirichlet_alpha             = {actual_alpha}")
    print(f"proximal_strength           = {actual_mu}")
    print(f"best_round                  = {summary['best_round']}")
    print(f"best_val_accuracy           = {float(summary['best_val_accuracy']):.12f}")
    print(f"test_loss                   = {float(summary['test_loss']):.12f}")
    print(f"test_accuracy               = {float(summary['test_accuracy']):.12f}")
    print(f"macro_class_accuracy        = {float(summary['macro_class_accuracy']):.12f}")
    print(f"confusion_accuracy          = {confusion_accuracy:.12f}")
    print(f"num_test_samples            = {summary['num_test_samples']}")
    print(f"[PASS] {name} 完整性审计通过。")

    return {
        "name": name,
        "directory": str(directory),
        "config": config,
        "state": state,
        "summary": summary,
        "metrics": metrics,
        "per_class": per_class,
        "confusion": confusion,
        "class_id_column": class_id_column,
        "samples_column": samples_column,
        "accuracy_column": accuracy_column,
    }


def comparable_value(result: dict[str, Any], key: str) -> Any:
    return first_value(key, result["summary"], result["config"])


def main() -> None:
    audited: dict[str, dict[str, Any]] = {}
    for name, specification in RUNS.items():
        audited[name] = audit_one(
            name=name,
            directory=specification["directory"],
            expected_mu=specification["expected_mu"],
        )

    left = audited["alpha05_mu0"]
    right = audited["alpha05_mu0p1"]

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

    print("\n" + "=" * 96)
    print("配对实验可比性检查")
    print("=" * 96)

    for key in comparable_keys:
        left_value = comparable_value(left, key)
        right_value = comparable_value(right, key)

        if left_value is None and right_value is None:
            print(f"{key:30s} skipped: 两边均未记录")
            continue

        if isinstance(left_value, (int, float)) and isinstance(
            right_value, (int, float)
        ):
            equal = bool(np.isclose(float(left_value), float(right_value), atol=1e-12, rtol=0))
        else:
            equal = str(left_value) == str(right_value)

        print(
            f"{key:30s} equal={equal} | "
            f"mu0={left_value!r} | mu0p1={right_value!r}"
        )
        assert equal, f"两组实验不可比，字段 {key} 不一致"

    baseline = left["summary"]
    fedprox = right["summary"]

    test_delta_pp = 100.0 * (
        float(fedprox["test_accuracy"]) - float(baseline["test_accuracy"])
    )
    macro_delta_pp = 100.0 * (
        float(fedprox["macro_class_accuracy"])
        - float(baseline["macro_class_accuracy"])
    )
    best_val_delta_pp = 100.0 * (
        float(fedprox["best_val_accuracy"])
        - float(baseline["best_val_accuracy"])
    )
    test_loss_delta = float(fedprox["test_loss"]) - float(baseline["test_loss"])

    print("\n" + "=" * 96)
    print("审计后的初步正式对比")
    print("=" * 96)
    print(f"best validation accuracy delta = {best_val_delta_pp:+.6f} percentage points")
    print(f"test accuracy delta           = {test_delta_pp:+.6f} percentage points")
    print(f"macro accuracy delta          = {macro_delta_pp:+.6f} percentage points")
    print(f"test loss delta               = {test_loss_delta:+.12f}")
    print("\n[PASS] α=0.5 的 μ=0 与 μ=0.1 均通过完整性与配对可比性审计。")


if __name__ == "__main__":
    main()