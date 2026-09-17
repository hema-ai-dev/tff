#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit exact per-client FedProx drift by running the existing TFF FedProx
LearningProcess with one client at a time from the same round-start checkpoint.

This tool does NOT modify the main trainer. It reuses:
- the verified Stage 1C FedProx builder,
- the saved client train manifests,
- the exact round-start global ModelWeights,
- the same dataset seed rule used by the main trainer.

Primary outputs:
- client_diagnostics.csv
- reconstruction_summary.json
- audit_config.json

Important interpretation:
- distance_to_global_model and model_delta_norm are exact per-client end-of-local-
  training distances inferred through the same TFF FedProx process.
- final_proximal_penalty = 0.5 * mu * ||w_client - w_global||^2 is the penalty
  evaluated at the final client model. It is NOT the mean per-batch proximal loss.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEDPROX_DIR = PROJECT_ROOT / "src" / "gtsrb_1c" / "fedprox"

if str(FEDPROX_DIR) not in sys.path:
    sys.path.insert(0, str(FEDPROX_DIR))

import train_fedprox_stage1c as trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit per-client FedProx drift from an existing run."
    )
    parser.add_argument(
        "--reference-run-dir",
        type=Path,
        required=True,
        help=(
            "Existing FedProx output directory containing config.json, "
            "manifests/, and checkpoints/model_weights_round_*.npz."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for client_diagnostics.csv and audit summaries.",
    )
    parser.add_argument(
        "--round-num",
        type=int,
        default=1,
        help=(
            "Audit this communication round. The tool loads checkpoint "
            "round_num-1 and reconstructs checkpoint round_num."
        ),
    )
    parser.add_argument(
        "--max-concurrent-calls",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --overwrite to replace the audit output."
            )
        import shutil
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def checkpoint_path(run_dir: Path, round_num: int) -> Path:
    return (
        run_dir
        / "checkpoints"
        / f"model_weights_round_{round_num:06d}.npz"
    )


def model_weights_to_trainable_arrays(
    model_weights: tff.learning.models.ModelWeights,
) -> list[np.ndarray]:
    arrays = model_weights.convert_variables_to_arrays()
    return [
        np.asarray(value).copy()
        for value in arrays.trainable
    ]


def get_state_trainable_arrays(
    learning_process,
    state,
) -> list[np.ndarray]:
    raw_weights = learning_process.get_model_weights(state)
    normalized = trainer.normalize_model_weights(raw_weights)
    return model_weights_to_trainable_arrays(normalized)


def arrays_l2_norm(arrays: list[np.ndarray]) -> float:
    squared_sum = 0.0
    for value in arrays:
        array = np.asarray(value, dtype=np.float64)
        squared_sum += float(np.sum(array * array))
    return float(np.sqrt(squared_sum))


def arrays_l2_distance(
    left: list[np.ndarray],
    right: list[np.ndarray],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Tensor count mismatch: {len(left)} != {len(right)}"
        )

    squared_sum = 0.0

    for index, (x_value, y_value) in enumerate(zip(left, right)):
        x = np.asarray(x_value, dtype=np.float64)
        y = np.asarray(y_value, dtype=np.float64)

        if x.shape != y.shape:
            raise ValueError(
                f"Tensor {index} shape mismatch: {x.shape} != {y.shape}"
            )

        diff = x - y
        squared_sum += float(np.sum(diff * diff))

    return float(np.sqrt(squared_sum))


def subtract_arrays(
    left: list[np.ndarray],
    right: list[np.ndarray],
) -> list[np.ndarray]:
    if len(left) != len(right):
        raise ValueError("Tensor count mismatch while subtracting arrays.")

    result: list[np.ndarray] = []

    for index, (x_value, y_value) in enumerate(zip(left, right)):
        x = np.asarray(x_value)
        y = np.asarray(y_value)

        if x.shape != y.shape:
            raise ValueError(
                f"Tensor {index} shape mismatch: {x.shape} != {y.shape}"
            )

        result.append(x.astype(np.float64) - y.astype(np.float64))

    return result


def add_arrays(
    left: list[np.ndarray],
    right: list[np.ndarray],
) -> list[np.ndarray]:
    if len(left) != len(right):
        raise ValueError("Tensor count mismatch while adding arrays.")

    return [
        np.asarray(x, dtype=np.float64)
        + np.asarray(y, dtype=np.float64)
        for x, y in zip(left, right)
    ]


def weighted_mean_array_lists(
    array_lists: list[list[np.ndarray]],
    weights: list[int],
) -> list[np.ndarray]:
    if not array_lists:
        raise ValueError("No client arrays were provided.")

    if len(array_lists) != len(weights):
        raise ValueError("Client array count does not match weight count.")

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("Total client weight must be positive.")

    num_tensors = len(array_lists[0])
    for client_arrays in array_lists:
        if len(client_arrays) != num_tensors:
            raise ValueError("Client tensor counts are inconsistent.")

    result: list[np.ndarray] = []

    for tensor_index in range(num_tensors):
        accumulator = np.zeros_like(
            np.asarray(array_lists[0][tensor_index]),
            dtype=np.float64,
        )

        for client_arrays, weight in zip(array_lists, weights):
            accumulator += (
                np.asarray(
                    client_arrays[tensor_index],
                    dtype=np.float64,
                )
                * float(weight)
            )

        result.append(accumulator / total_weight)

    return result


def compare_array_lists(
    left: list[np.ndarray],
    right: list[np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if len(left) != len(right):
        return {
            "tensor_count_equal": False,
            "exact": False,
            "allclose": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "l2_diff": None,
        }

    all_exact = True
    all_close = True
    max_abs_diff = 0.0
    abs_sum = 0.0
    value_count = 0
    squared_l2 = 0.0

    for index, (x_value, y_value) in enumerate(zip(left, right)):
        x = np.asarray(x_value)
        y = np.asarray(y_value)

        if x.shape != y.shape:
            return {
                "tensor_count_equal": True,
                "shape_mismatch_tensor": index,
                "exact": False,
                "allclose": False,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "l2_diff": None,
            }

        exact = np.array_equal(x, y)
        close = np.allclose(x, y, atol=atol, rtol=rtol)

        all_exact = all_exact and exact
        all_close = all_close and close

        diff = (
            x.astype(np.float64)
            - y.astype(np.float64)
        )
        absolute = np.abs(diff)

        if absolute.size:
            max_abs_diff = max(
                max_abs_diff,
                float(np.max(absolute)),
            )
            abs_sum += float(np.sum(absolute))
            value_count += int(absolute.size)
            squared_l2 += float(np.sum(diff * diff))

    return {
        "tensor_count_equal": True,
        "exact": bool(all_exact),
        "allclose": bool(all_close),
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": (
            float(abs_sum / value_count)
            if value_count
            else 0.0
        ),
        "l2_diff": float(np.sqrt(squared_l2)),
    }


def build_process_args(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        client_optimizer=str(config["client_optimizer"]),
        client_learning_rate=float(config["client_learning_rate"]),
        client_momentum=float(config.get("client_momentum", 0.0)),
        server_learning_rate=float(config["server_learning_rate"]),
        proximal_strength=float(config["proximal_strength"]),
        seed=int(config["seed"]),
        batch_size=int(config["batch_size"]),
        local_epochs=int(config["local_epochs"]),
    )


def validate_reference_config(
    config: dict[str, Any],
    run_dir: Path,
    round_num: int,
) -> None:
    required = [
        "client_optimizer",
        "client_learning_rate",
        "server_learning_rate",
        "proximal_strength",
        "seed",
        "batch_size",
        "local_epochs",
        "resolved_data_root",
    ]
    missing = [
        key for key in required if key not in config
    ]
    if missing:
        raise ValueError(
            f"Reference config is missing fields: {missing}"
        )

    if round_num < 1:
        raise ValueError("--round-num must be at least 1.")

    before = checkpoint_path(run_dir, round_num - 1)
    after = checkpoint_path(run_dir, round_num)

    if not before.is_file():
        raise FileNotFoundError(before)
    if not after.is_file():
        raise FileNotFoundError(after)


def discover_client_manifests(run_dir: Path) -> list[tuple[int, Path]]:
    manifest_dir = run_dir / "manifests"
    files = sorted(manifest_dir.glob("client_*_train.csv"))

    discovered: list[tuple[int, Path]] = []

    for path in files:
        stem_parts = path.stem.split("_")
        try:
            client_id = int(stem_parts[1])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Unexpected client manifest name: {path.name}"
            ) from error

        discovered.append((client_id, path))

    if not discovered:
        raise FileNotFoundError(
            f"No client train manifests found in {manifest_dir}"
        )

    expected_ids = list(range(len(discovered)))
    actual_ids = [client_id for client_id, _ in discovered]

    if actual_ids != expected_ids:
        raise ValueError(
            f"Client IDs are not contiguous: {actual_ids}"
        )

    return discovered


def main() -> None:
    args = parse_args()

    reference_run_dir = (
        args.reference_run_dir
        .expanduser()
        .resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()

    if not reference_run_dir.is_dir():
        raise FileNotFoundError(reference_run_dir)

    prepare_output_dir(output_dir, args.overwrite)

    config = read_json(reference_run_dir / "config.json")
    validate_reference_config(
        config,
        reference_run_dir,
        args.round_num,
    )

    process_args = build_process_args(config)
    data_root = Path(
        config["resolved_data_root"]
    ).expanduser().resolve()

    client_manifests = discover_client_manifests(
        reference_run_dir
    )

    trainer.set_global_seed(process_args.seed)
    trainer.configure_tff_context(
        args.max_concurrent_calls
    )

    prototype_dataset = trainer.make_tff_dataset(
        csv_path=client_manifests[0][1],
        data_root=data_root,
        training=True,
        batch_size=process_args.batch_size,
        local_epochs=process_args.local_epochs,
        seed=(
            process_args.seed
            + args.round_num * 1000
            + client_manifests[0][0]
        ),
    )

    input_spec = prototype_dataset.element_spec
    learning_process = trainer.build_learning_process(
        input_spec,
        process_args,
    )

    initialized_state = learning_process.initialize()

    before_weights = trainer.load_model_weights_checkpoint(
        checkpoint_path(
            reference_run_dir,
            args.round_num - 1,
        )
    )
    reference_after_weights = (
        trainer.load_model_weights_checkpoint(
            checkpoint_path(
                reference_run_dir,
                args.round_num,
            )
        )
    )

    global_before_arrays = model_weights_to_trainable_arrays(
        before_weights
    )
    reference_after_arrays = model_weights_to_trainable_arrays(
        reference_after_weights
    )

    global_weight_l2_norm = arrays_l2_norm(
        global_before_arrays
    )

    rows: list[dict[str, Any]] = []
    client_deltas: list[list[np.ndarray]] = []
    client_weights: list[int] = []

    print("=" * 88)
    print("FedProx exact per-client drift audit")
    print("=" * 88)
    print(f"Reference run : {reference_run_dir}")
    print(f"Round         : {args.round_num}")
    print(f"Mu            : {process_args.proximal_strength}")
    print(f"Clients       : {len(client_manifests)}")
    print(f"Global norm   : {global_weight_l2_norm:.10f}")
    print()

    for client_id, manifest_path in client_manifests:
        manifest = pd.read_csv(manifest_path)
        client_num_examples = int(len(manifest))

        dataset_seed = (
            process_args.seed
            + args.round_num * 1000
            + client_id
        )

        dataset = trainer.make_tff_dataset(
            csv_path=manifest_path,
            data_root=data_root,
            training=True,
            batch_size=process_args.batch_size,
            local_epochs=process_args.local_epochs,
            seed=dataset_seed,
        )

        client_start_state = (
            learning_process.set_model_weights(
                initialized_state,
                before_weights,
            )
        )

        start_time = time.perf_counter()
        output = learning_process.next(
            client_start_state,
            [dataset],
        )
        elapsed = time.perf_counter() - start_time

        client_after_arrays = get_state_trainable_arrays(
            learning_process,
            output.state,
        )
        client_delta = subtract_arrays(
            client_after_arrays,
            global_before_arrays,
        )

        distance = arrays_l2_distance(
            client_after_arrays,
            global_before_arrays,
        )
        relative_distance = (
            distance
            / max(
                global_weight_l2_norm,
                np.finfo(np.float64).eps,
            )
        )
        squared_distance = distance * distance
        final_proximal_penalty = (
            0.5
            * process_args.proximal_strength
            * squared_distance
        )

        flat_metrics = trainer.flatten_nested(
            output.metrics
        )
        task_loss = trainer.find_metric(
            flat_metrics,
            "loss",
        )
        train_accuracy = trainer.find_metric(
            flat_metrics,
            "accuracy",
        )

        row: dict[str, Any] = {
            "round": int(args.round_num),
            "client_id": int(client_id),
            "client_num_examples": client_num_examples,
            "dataset_seed": int(dataset_seed),
            "proximal_strength": float(
                process_args.proximal_strength
            ),
            "task_loss": task_loss,
            "train_accuracy": train_accuracy,
            "global_weight_l2_norm": global_weight_l2_norm,
            "distance_to_global_model": distance,
            "model_delta_norm": distance,
            "relative_distance_to_global_model": relative_distance,
            "squared_distance_to_global_model": squared_distance,
            "final_proximal_penalty": final_proximal_penalty,
            "audit_seconds": elapsed,
        }

        for key, value in flat_metrics.items():
            safe_key = (
                "tff_"
                + key.replace("/", "__")
                .replace(" ", "_")
            )
            if safe_key not in row:
                row[safe_key] = trainer.scalar_value(value)

        rows.append(row)
        client_deltas.append(client_delta)
        client_weights.append(client_num_examples)

        print(
            f"client_{client_id} | "
            f"n={client_num_examples} | "
            f"task_loss={task_loss} | "
            f"train_acc={train_accuracy} | "
            f"distance={distance:.10f} | "
            f"relative={relative_distance:.10f} | "
            f"final_penalty={final_proximal_penalty:.10f} | "
            f"time={elapsed:.2f}s"
        )

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(
        output_dir / "client_diagnostics.csv",
        index=False,
    )

    weighted_delta = weighted_mean_array_lists(
        client_deltas,
        client_weights,
    )
    reconstructed_after = add_arrays(
        global_before_arrays,
        weighted_delta,
    )

    reconstruction = compare_array_lists(
        reconstructed_after,
        reference_after_arrays,
        atol=args.atol,
        rtol=args.rtol,
    )

    weighted_distance = float(
        np.average(
            diagnostics["distance_to_global_model"],
            weights=diagnostics["client_num_examples"],
        )
    )
    weighted_relative_distance = float(
        np.average(
            diagnostics[
                "relative_distance_to_global_model"
            ],
            weights=diagnostics["client_num_examples"],
        )
    )
    weighted_final_penalty = float(
        np.average(
            diagnostics["final_proximal_penalty"],
            weights=diagnostics["client_num_examples"],
        )
    )

    summary = {
        "reference_run_dir": str(reference_run_dir),
        "round": int(args.round_num),
        "proximal_strength": float(
            process_args.proximal_strength
        ),
        "num_clients": int(len(client_manifests)),
        "total_client_examples": int(sum(client_weights)),
        "global_weight_l2_norm": global_weight_l2_norm,
        "weighted_mean_client_distance": weighted_distance,
        "weighted_mean_relative_client_distance": (
            weighted_relative_distance
        ),
        "weighted_mean_final_proximal_penalty": (
            weighted_final_penalty
        ),
        "min_client_distance": float(
            diagnostics["distance_to_global_model"].min()
        ),
        "max_client_distance": float(
            diagnostics["distance_to_global_model"].max()
        ),
        "std_client_distance_population": float(
            diagnostics["distance_to_global_model"].std(ddof=0)
        ),
        "reconstruction": reconstruction,
        "interpretation_note": (
            "final_proximal_penalty is evaluated at the final client model; "
            "it is not the average per-batch proximal loss."
        ),
    }

    (output_dir / "reconstruction_summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit_config = {
        "tool": "audit_fedprox_client_drift",
        "reference_run_dir": str(reference_run_dir),
        "output_dir": str(output_dir),
        "round": int(args.round_num),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "tensorflow_version": tf.__version__,
        "tff_version": tff.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "source_run_config": config,
    }
    (output_dir / "audit_config.json").write_text(
        json.dumps(
            audit_config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("Reconstruction check")
    print("=" * 88)
    print(json.dumps(reconstruction, indent=2))
    print()
    print("=" * 88)
    print("Weighted summary")
    print("=" * 88)
    print(
        f"weighted_mean_client_distance="
        f"{weighted_distance:.10f}"
    )
    print(
        f"weighted_mean_relative_client_distance="
        f"{weighted_relative_distance:.10f}"
    )
    print(
        f"weighted_mean_final_proximal_penalty="
        f"{weighted_final_penalty:.10f}"
    )
    print()
    print(
        f"[DONE] {output_dir / 'client_diagnostics.csv'}"
    )


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
