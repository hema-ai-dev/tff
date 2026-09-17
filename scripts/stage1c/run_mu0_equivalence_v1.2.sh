#!/usr/bin/env bash
set -euo pipefail

# Stage 1C-A1: FedAvg vs FedProx(mu=0) equivalence check.
# v1.2 removes "| tee", because TFF subprocesses may keep the pipe open
# after the main trainer has already completed.
#
# Run from the project root:
#   bash scripts/stage1c/run_mu0_equivalence_v1.2.sh

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
COMPARE_SCRIPT="${PROJECT_ROOT}/tools/compare_fedavg_fedprox_mu0.py"

DATA_ROOT="${PROJECT_ROOT}/data/gtsrb"
PARTITION_DIR="${DATA_ROOT}/federated_dirichlet_${TAG}_5clients_seed${SEED}"

OUTPUT_ROOT="${PROJECT_ROOT}/outputs/stage1c/equivalence"
FEDAVG_OUT="${OUTPUT_ROOT}/fedavg_${TAG}_${ROUNDS}r_seed${SEED}"
FEDPROX_OUT="${OUTPUT_ROOT}/fedprox_mu0_${TAG}_${ROUNDS}r_seed${SEED}"
LOG_DIR="${OUTPUT_ROOT}/logs"

FEDAVG_LOG="${LOG_DIR}/fedavg_${TAG}_${ROUNDS}r_seed${SEED}.log"
FEDPROX_LOG="${LOG_DIR}/fedprox_mu0_${TAG}_${ROUNDS}r_seed${SEED}.log"

for path in \
    "${FEDAVG_SCRIPT}" \
    "${FEDPROX_SCRIPT}" \
    "${COMPARE_SCRIPT}" \
    "${DATA_ROOT}/Test.csv" \
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

is_complete() {
  local output_dir="$1"
  [[ -f "${output_dir}/test_metrics.json" ]] &&
  [[ -f "${output_dir}/metrics.csv" ]] &&
  [[ -f "${output_dir}/run_state.json" ]] &&
  [[ -f "${output_dir}/global_models/last_global_model.h5" ]]
}

run_logged() {
  local log_file="$1"
  shift

  echo "[LOG] ${log_file}"
  set +e
  PYTHONUNBUFFERED=1 "$@" >"${log_file}" 2>&1
  local status=$?
  set -e

  echo
  echo "------------------------------ log tail ------------------------------"
  tail -n 80 "${log_file}" || true
  echo "---------------------------------------------------------------------"
  echo

  if [[ ${status} -ne 0 ]]; then
    echo "[ERROR] Command failed with exit code ${status}." >&2
    exit "${status}"
  fi
}

echo "============================================================================"
echo "[1/3] Fresh FedAvg control run"
echo "============================================================================"

if is_complete "${FEDAVG_OUT}"; then
  echo "[SKIP] Existing FedAvg output is complete:"
  echo "       ${FEDAVG_OUT}"
else
  run_logged "${FEDAVG_LOG}" \
    "${PYTHON_BIN}" "${FEDAVG_SCRIPT}" \
    --output-dir "${FEDAVG_OUT}" \
    --experiment-name "stage1c_equivalence_fedavg_${TAG}_${ROUNDS}r_seed${SEED}" \
    "${COMMON_ARGS[@]}"
fi

echo "============================================================================"
echo "[2/3] Fresh FedProx run with mu=0"
echo "============================================================================"

if is_complete "${FEDPROX_OUT}"; then
  echo "[SKIP] Existing FedProx(mu=0) output is complete:"
  echo "       ${FEDPROX_OUT}"
else
  run_logged "${FEDPROX_LOG}" \
    "${PYTHON_BIN}" "${FEDPROX_SCRIPT}" \
    --output-dir "${FEDPROX_OUT}" \
    --experiment-name "stage1c_equivalence_fedprox_mu0_${TAG}_${ROUNDS}r_seed${SEED}" \
    --proximal-strength 0.0 \
    "${COMMON_ARGS[@]}"
fi

echo "============================================================================"
echo "[3/3] Compare metrics"
echo "============================================================================"

"${PYTHON_BIN}" "${COMPARE_SCRIPT}" \
  --fedavg-dir "${FEDAVG_OUT}" \
  --fedprox-dir "${FEDPROX_OUT}"

echo
echo "[DONE] mu=0 equivalence check finished."
