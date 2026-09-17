#!/usr/bin/env bash
set -euo pipefail

# Stage 1B formal FedAvg runs for GTSRB Dirichlet Non-IID.
# Run from project root:
#   bash scripts/run_stage1b_fedavg_all.sh
#
# Optional environment overrides:
#   ROUNDS=100 CHECKPOINT_EVERY=5 EVAL_EVERY=1 bash scripts/run_stage1b_fedavg_all.sh

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROUNDS="${ROUNDS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_RATIO="${VAL_RATIO:-0.1}"
CLIENT_LR="${CLIENT_LR:-0.001}"
SERVER_LR="${SERVER_LR:-1.0}"
SEED="${SEED:-20260720}"
EVAL_EVERY="${EVAL_EVERY:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
MAX_CONCURRENT_CALLS="${MAX_CONCURRENT_CALLS:-1}"

TRAIN_SCRIPT="${PROJECT_ROOT}/src/gtsrb_1b/train_fedavg_stage1b.py"
DATA_ROOT="${PROJECT_ROOT}/data/gtsrb"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/stage1b"

mkdir -p "${OUTPUT_ROOT}/logs"

declare -a TAGS=("alpha10" "alpha1" "alpha05" "alpha01")

for TAG in "${TAGS[@]}"; do
  PARTITION_DIR="${DATA_ROOT}/federated_dirichlet_${TAG}_5clients_seed${SEED}"
  OUTPUT_DIR="${OUTPUT_ROOT}/fedavg_${TAG}_seed${SEED}"
  LOG_FILE="${OUTPUT_ROOT}/logs/fedavg_${TAG}_seed${SEED}.log"

  if [[ ! -d "${PARTITION_DIR}" ]]; then
    echo "[ERROR] Missing partition directory: ${PARTITION_DIR}" >&2
    exit 1
  fi

  echo "================================================================================"
  echo "[RUN] ${TAG}"
  echo "[RUN] partition: ${PARTITION_DIR}"
  echo "[RUN] output:    ${OUTPUT_DIR}"
  echo "[RUN] log:       ${LOG_FILE}"
  echo "================================================================================"

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --data-root "${DATA_ROOT}" \
    --partition-dir "${PARTITION_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --experiment-name "stage1b_fedavg_${TAG}_seed${SEED}" \
    --rounds "${ROUNDS}" \
    --local-epochs "${LOCAL_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --val-ratio "${VAL_RATIO}" \
    --client-optimizer adam \
    --client-learning-rate "${CLIENT_LR}" \
    --server-learning-rate "${SERVER_LR}" \
    --eval-every "${EVAL_EVERY}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --seed "${SEED}" \
    --max-concurrent-calls "${MAX_CONCURRENT_CALLS}" \
    --overwrite \
    2>&1 | tee "${LOG_FILE}"

  echo "[DONE] ${TAG}"
done

echo "[DONE] All Stage 1B FedAvg experiments completed."
