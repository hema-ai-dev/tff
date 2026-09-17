#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_partition_csvs(
    partition_dir: str | Path,
    num_clients: int = 5,
    expected_total_samples: int = 39209,
    expected_num_classes: int = 43,
) -> Dict[str, Path]:
    """
    Validate a saved Stage 1B partition and return client_id -> CSV path.
    """
    partition_dir = Path(partition_dir).resolve()
    clients_dir = partition_dir / "clients"
    metadata_path = partition_dir / "split_metadata.json"
    manifest_path = partition_dir / "partition_manifest.csv"

    if not partition_dir.is_dir():
        raise FileNotFoundError(f"Partition directory not found: {partition_dir}")
    if not clients_dir.is_dir():
        raise FileNotFoundError(f"Clients directory not found: {clients_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if int(metadata["num_clients"]) != num_clients:
        raise ValueError(
            f"Expected {num_clients} clients, found {metadata['num_clients']}"
        )
    if int(metadata["num_classes"]) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, "
            f"found {metadata['num_classes']}"
        )
    if int(metadata["total_samples"]) != expected_total_samples:
        raise ValueError(
            f"Expected {expected_total_samples} samples, "
            f"found {metadata['total_samples']}"
        )

    client_csvs: Dict[str, Path] = {}
    total_rows = 0
    all_paths: List[str] = []

    for client_id in range(num_clients):
        name = f"client_{client_id}"
        csv_path = clients_dir / f"{name}_train.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing client CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        required = {"Path", "ClassId"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        labels = df["ClassId"].astype(int)
        if labels.min() < 0 or labels.max() >= expected_num_classes:
            raise ValueError(f"Invalid class label found in {csv_path}")

        total_rows += len(df)
        all_paths.extend(df["Path"].astype(str).tolist())
        client_csvs[name] = csv_path

    if total_rows != expected_total_samples:
        raise ValueError(
            f"Client CSV total is {total_rows}, expected {expected_total_samples}"
        )

    unique_paths = len(set(all_paths))
    if unique_paths != expected_total_samples:
        raise ValueError(
            f"Unique client paths = {unique_paths}; "
            f"expected {expected_total_samples}. Duplicate paths exist."
        )

    manifest = pd.read_csv(manifest_path)
    if len(manifest) != expected_total_samples:
        raise ValueError(
            f"Manifest rows = {len(manifest)}, "
            f"expected {expected_total_samples}"
        )

    return client_csvs


def describe_partition(partition_dir: str | Path) -> dict:
    partition_dir = Path(partition_dir).resolve()
    metadata_path = partition_dir / "split_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)
