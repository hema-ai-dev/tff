#!/usr/bin/env bash
set -euo pipefail

# Resume Stage 1B formal runs to a new target round count.
# Example:
#   TARGET_ROUNDS=160 bash scripts/resume_stage1b_fedavg_all.sh

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_ROUNDS="${TARGET_ROUNDS:-160}"
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
  LOG_FILE="${OUTPUT_ROOT}/logs/fedavg_${TAG}_seed${SEED}_resume_to_${TARGET_ROUNDS}.log"

  if [[ ! -f "${OUTPUT_DIR}/run_state.json" ]]; then
    echo "[SKIP] No checkpoint state for ${TAG}: ${OUTPUT_DIR}/run_state.json"
    continue
  fi

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --data-root "${DATA_ROOT}" \
    --partition-dir "${PARTITION_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --experiment-name "stage1b_fedavg_${TAG}_seed${SEED}" \
    --rounds "${TARGET_ROUNDS}" \
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
    --resume \
    2>&1 | tee "${LOG_FILE}"
done
