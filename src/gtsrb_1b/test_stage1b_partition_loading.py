#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stage1b_partition_loader import load_partition_csvs, describe_partition


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    client_csvs = load_partition_csvs(args.partition_dir)
    metadata = describe_partition(args.partition_dir)

    print("=" * 80)
    print("Partition validation passed")
    print("partition_dir:", args.partition_dir.resolve())
    print("alpha:", metadata["alpha"])
    print("mean_js_divergence:", metadata["mean_js_divergence_to_global"])

    total = 0
    for client_id, csv_path in client_csvs.items():
        df = pd.read_csv(csv_path)
        total += len(df)
        print(
            client_id,
            "samples=", len(df),
            "classes=", df["ClassId"].nunique(),
            "csv=", csv_path,
        )

    print("total_samples:", total)
    print("=" * 80)


if __name__ == "__main__":
    main()
