#!/usr/bin/env bash
# 啟動 AACDR 平行訓練（在 Docker 容器內執行）
#
# 用法：
#   bash /workspace/AACDR/scripts/run_parallel_train.sh [workers] [total_models]
#
# 範例（8 個 worker 各跑 100/8 個模型）：
#   bash /workspace/AACDR/scripts/run_parallel_train.sh 8 100

set -euo pipefail

PROJECT_ROOT="/workspace/AACDR"
CODE_DIR="${PROJECT_ROOT}/code"
CKPT_DIR="${PROJECT_ROOT}/ckpt"
RUNS_DIR="${CKPT_DIR}/runs"
LOG_DIR="${CKPT_DIR}/parallel_logs"

WORKERS="${1:-8}"
TOTAL_MODELS="${2:-100}"
DESCRIPTION="${3:-Reproduce}"
RUN_ID="${4:-0}"
SEED="${5:-0}"
DEVICE="${AACDR_DEVICE:-cuda}"

mkdir -p "${RUNS_DIR}" "${LOG_DIR}"

echo "Parallel AACDR training"
echo "  workers=${WORKERS}"
echo "  total_models=${TOTAL_MODELS}"
echo "  device=${DEVICE}"
echo "  description=${DESCRIPTION}"

# 將 100 個模型盡量平均分配給 workers
chunk=$(( (TOTAL_MODELS + WORKERS - 1) / WORKERS ))

pids=()
for ((worker=0; worker<WORKERS; worker++)); do
  start=$(( worker * chunk ))
  end=$(( start + chunk ))
  if (( start >= TOTAL_MODELS )); then
    break
  fi
  if (( end > TOTAL_MODELS )); then
    end=${TOTAL_MODELS}
  fi

  log_file="${LOG_DIR}/worker_${worker}.log"
  echo "Launch worker ${worker}: models [${start}, ${end}) -> ${log_file}"

  (
    cd "${CODE_DIR}"
    python -u parallel_train.py \
      --description "${DESCRIPTION}" \
      --id "${RUN_ID}" \
      --seed "${SEED}" \
      --model_start "${start}" \
      --model_end "${end}" \
      --device "${DEVICE}" \
      --worker_id "${worker}"
  ) > "${log_file}" 2>&1 &

  pids+=($!)
done

echo "Launched ${#pids[@]} workers. Waiting..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if (( fail != 0 )); then
  echo "One or more workers failed. Check ${LOG_DIR}/worker_*.log"
  exit 1
fi

echo "All workers finished. Merging logs..."
python "${PROJECT_ROOT}/scripts/merge_parallel_logs.py" \
  --runs-dir "${RUNS_DIR}" \
  --total-models "${TOTAL_MODELS}" \
  --out-log "${CKPT_DIR}/log_${SEED}_${RUN_ID}_${DESCRIPTION}.txt"

python "${PROJECT_ROOT}/scripts/summarize_results.py" \
  --log "${CKPT_DIR}/log_${SEED}_${RUN_ID}_${DESCRIPTION}.txt" \
  --reference "${PROJECT_ROOT}/supplymentmaterials/100_random_initializations.txt" \
  --out-dir "${CKPT_DIR}"

echo "Done."
