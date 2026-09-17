#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create the Stage 1C FedProx trainer from the verified Stage 1B FedAvg trainer.

Run from the project root:
    python tools/bootstrap_stage1c_fedprox.py

Optional:
    python tools/bootstrap_stage1c_fedprox.py --overwrite
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"[PATCH ERROR] {label}: expected exactly 1 match, found {count}."
        )
    return text.replace(old, new, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing src/gtsrb_1b.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()

    source = project_root / "src/gtsrb_1b/train_fedavg_stage1b.py"
    target_dir = project_root / "src/gtsrb_1c/fedprox"
    target = target_dir / "train_fedprox_stage1c.py"
    backup_dir = project_root / "archives/stage1c_before_fedprox"
    backup = backup_dir / "train_fedavg_stage1b.py"

    if not source.is_file():
        raise FileNotFoundError(f"Missing verified Stage 1B trainer: {source}")

    if target.exists() and not args.overwrite:
        raise FileExistsError(
            f"Target already exists: {target}\n"
            "Use --overwrite only when you intentionally want to regenerate it."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not backup.exists():
        shutil.copy2(source, backup)
        print(f"[BACKUP] {backup}")
    else:
        print(f"[BACKUP] Existing backup kept: {backup}")

    text = source.read_text(encoding="utf-8")

    required_markers = [
        "build_weighted_fed_avg",
        "--partition-dir",
        "--server-learning-rate",
        "Weighted FedAvg",
        "stage1b_partition_loader",
    ]
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        raise RuntimeError(
            "The Stage 1B script does not match the expected verified baseline. "
            f"Missing markers: {missing}"
        )

    text = text.replace(
        "Stage 1B: GTSRB + TFF 0.87.0 加权 FedAvg 基线",
        "Stage 1C-A1: GTSRB + TFF 0.87.0 加权 FedProx",
        1,
    )
    text = text.replace(
        'description="GTSRB 5 客户端加权 FedAvg 分类基线。"',
        'description="GTSRB 5 客户端加权 FedProx 分类实验。"',
        1,
    )

    old_import_block = """CURRENT_DIR = Path(__file__).resolve().parent
STAGE1A_DIR = CURRENT_DIR.parent / "gtsrb_1a"

for import_dir in (CURRENT_DIR, STAGE1A_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))
"""
    new_import_block = """CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
STAGE1A_DIR = SRC_DIR / "gtsrb_1a"
STAGE1B_DIR = SRC_DIR / "gtsrb_1b"

for import_dir in (CURRENT_DIR, STAGE1A_DIR, STAGE1B_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))
"""
    text = replace_once(
        text, old_import_block, new_import_block, "import path block"
    )

    server_arg = """    parser.add_argument(
        "--server-learning-rate",
        type=float,
        default=1.0,
    )
"""
    prox_arg = server_arg + """    parser.add_argument(
        "--proximal-strength",
        type=float,
        default=0.0,
        help=(
            "FedProx proximal strength mu. Must be nonnegative. "
            "mu=0.0 is the FedAvg degeneration check."
        ),
    )
"""
    text = replace_once(
        text, server_arg, prox_arg, "proximal-strength CLI argument"
    )

    server_validation = """    if args.server_learning_rate <= 0:
        raise ValueError("--server-learning-rate 必须大于 0。")
"""
    prox_validation = server_validation + """    if args.proximal_strength < 0:
        raise ValueError("--proximal-strength 必须大于或等于 0。")
"""
    text = replace_once(
        text, server_validation, prox_validation, "proximal-strength validation"
    )

    text = replace_once(
        text,
        "return tff.learning.algorithms.build_weighted_fed_avg(",
        "return tff.learning.algorithms.build_weighted_fed_prox(",
        "FedAvg -> FedProx builder",
    )
    text = replace_once(
        text,
        "        model_fn=model_fn,\n"
        "        client_optimizer_fn=build_client_optimizer(args),",
        "        model_fn=model_fn,\n"
        "        proximal_strength=float(args.proximal_strength),\n"
        "        client_optimizer_fn=build_client_optimizer(args),",
        "proximal strength builder argument",
    )

    text = text.replace(
        '"algorithm": "Weighted FedAvg",',
        '"algorithm": "Weighted FedProx",\n'
        '        "proximal_strength": float(args.proximal_strength),',
        1,
    )
    text = text.replace(
        'plt.title("FedAvg GTSRB Loss")',
        'plt.title("FedProx GTSRB Loss")',
    )
    text = text.replace(
        'plt.title("FedAvg GTSRB Accuracy")',
        'plt.title("FedProx GTSRB Accuracy")',
    )
    text = text.replace(
        'print("Stage 1B - Weighted FedAvg under Dirichlet Non-IID")',
        'print("Stage 1C-A1 - Weighted FedProx under Dirichlet Non-IID")',
    )
    text = text.replace("此 FedAvg 基线", "此 FedProx 基线")

    if "build_weighted_fed_avg(" in text:
        raise RuntimeError(
            "An unexpected build_weighted_fed_avg call remains after patching."
        )
    if text.count("build_weighted_fed_prox(") != 1:
        raise RuntimeError(
            "Expected exactly one build_weighted_fed_prox call."
        )
    if "proximal_strength=float(args.proximal_strength)" not in text:
        raise RuntimeError("FedProx proximal strength was not inserted.")

    target.write_text(text, encoding="utf-8")
    py_compile.compile(str(target), doraise=True)

    print(f"[CREATED] {target}")
    print("[CHECK] Python syntax compilation passed.")
    print("[NEXT] Run: bash scripts/stage1c/run_mu0_equivalence.sh")


if __name__ == "__main__":
    main()