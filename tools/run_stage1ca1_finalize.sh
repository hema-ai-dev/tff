#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/tff_mnist_fedavg
mkdir -p \
  outputs/stage1c/fedprox_formal/analysis/alpha01 \
  outputs/stage1c/fedprox_formal/analysis/alpha05

python tools/analyze_fedprox_pair.py \
  --mu0-dir outputs/stage1c/fedprox_formal/alpha01/sgd_lr0p1_mu0_alpha01_160r_seed20260720 \
  --mu0p1-dir outputs/stage1c/fedprox_formal/alpha01/sgd_lr0p1_mu0p1_alpha01_160r_seed20260720 \
  --output-dir outputs/stage1c/fedprox_formal/analysis/alpha01 \
  --alpha 0.1 \
  --prefix alpha01 \
  | tee outputs/stage1c/fedprox_formal/analysis/alpha01/alpha01_pair_analysis_log.txt

python tools/analyze_fedprox_pair.py \
  --mu0-dir outputs/stage1c/fedprox_formal/alpha05/sgd_lr0p1_mu0_alpha05_160r_seed20260720 \
  --mu0p1-dir outputs/stage1c/fedprox_formal/alpha05/sgd_lr0p1_mu0p1_alpha05_160r_seed20260720 \
  --output-dir outputs/stage1c/fedprox_formal/analysis/alpha05 \
  --alpha 0.5 \
  --prefix alpha05 \
  | tee outputs/stage1c/fedprox_formal/analysis/alpha05/alpha05_pair_analysis_log.txt

python tools/finalize_stage1ca1_fedprox.py \
  --formal-root outputs/stage1c/fedprox_formal \
  | tee outputs/stage1c/fedprox_formal/analysis/stage1ca1_finalize_log.txt

echo
echo "[DONE] Inspect these files:"
echo "outputs/stage1c/fedprox_formal/analysis/stage1ca1_closure_audit.txt"
echo "outputs/stage1c/fedprox_formal/Stage1C_A1_FedProx_Technical_Report_v1.0.md"
