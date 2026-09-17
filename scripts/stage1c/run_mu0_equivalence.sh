#!/usr/bin/env bash
set -euo pipefail

# Stage 1C-A1: FedProx mu=0 degeneration/equivalence check.
# Run from the project root:
#   bash scripts/stage1c/run_mu0_equivalence.sh

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROUNDS="${ROUNDS:-3}"
TAG="${TAG:-alpha05}"
SEED="${SEED:-20260720}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_RATIO="${VAL_RATIO:-0.1}"
CLIENT_LR="${CLIENT_LR:-0.001}"
SERVER_LR="${SERVER_LR:-1.0}"
EVAL_EVERY="${EVAL_EVERY:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
MAX_CONCURRENT_CALLS="${MAX_CONCURRENT_CALLS:-1}"

FEDAVG_SCRIPT="${PROJECT_ROOT}/src/gtsrb_1b/train_fedavg_stage1b.py"
FEDPROX_SCRIPT="${PROJECT_ROOT}/src/gtsrb_1c/fedprox/train_fedprox_stage1c.py"
DATA_ROOT="${PROJECT_ROOT}/data/gtsrb"
PARTITION_DIR="${DATA_ROOT}/federated_dirichlet_${TAG}_5clients_seed${SEED}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/stage1c/equivalence"
FEDAVG_OUT="${OUTPUT_ROOT}/fedavg_${TAG}_${ROUNDS}r_seed${SEED}"
FEDPROX_OUT="${OUTPUT_ROOT}/fedprox_mu0_${TAG}_${ROUNDS}r_seed${SEED}"
LOG_DIR="${OUTPUT_ROOT}/logs"

for path in "${FEDAVG_SCRIPT}" "${FEDPROX_SCRIPT}" "${DATA_ROOT}/Test.csv" \
            "${PARTITION_DIR}/split_metadata.json" \
            "${PARTITION_DIR}/partition_manifest.csv"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --partition-dir "${PARTITION_DIR}"
  --rounds "${ROUNDS}"
  --local-epochs "${LOCAL_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --val-ratio "${VAL_RATIO}"
  --client-optimizer adam
  --client-learning-rate "${CLIENT_LR}"
  --server-learning-rate "${SERVER_LR}"
  --eval-every "${EVAL_EVERY}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --seed "${SEED}"
  --max-concurrent-calls "${MAX_CONCURRENT_CALLS}"
  --overwrite
)

echo "============================================================================"
echo "[1/3] Fresh FedAvg control run"
echo "============================================================================"
"${PYTHON_BIN}" "${FEDAVG_SCRIPT}" \
  --output-dir "${FEDAVG_OUT}" \
  --experiment-name "stage1c_equivalence_fedavg_${TAG}_${ROUNDS}r_seed${SEED}" \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/fedavg_${TAG}_${ROUNDS}r_seed${SEED}.log"

echo "============================================================================"
echo "[2/3] Fresh FedProx run with mu=0"
echo "============================================================================"
"${PYTHON_BIN}" "${FEDPROX_SCRIPT}" \
  --output-dir "${FEDPROX_OUT}" \
  --experiment-name "stage1c_equivalence_fedprox_mu0_${TAG}_${ROUNDS}r_seed${SEED}" \
  --proximal-strength 0.0 \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/fedprox_mu0_${TAG}_${ROUNDS}r_seed${SEED}.log"

echo "============================================================================"
echo "[3/3] Compare metrics"
echo "============================================================================"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/compare_fedavg_fedprox_mu0.py" \
  --fedavg-dir "${FEDAVG_OUT}" \
  --fedprox-dir "${FEDPROX_OUT}"

echo "[DONE] mu=0 equivalence check finished."
