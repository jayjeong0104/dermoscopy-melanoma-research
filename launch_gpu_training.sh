#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="${SKIN_CANCER_OUTPUT_ROOT:-$project_dir/artifacts/multitask_v2}"
warm_start_model="${SKIN_CANCER_WARM_START_MODEL:-$project_dir/artifacts/multitask_v2/run_20260817_093453/best_head_model.keras}"
pid_file="$output_root/active_training.pid"

worker_main() {
  worker_status_file="$1"

  finish() {
    local status=$?
    {
      printf 'exit_code=%s\n' "$status"
      printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    } > "$worker_status_file"

    if [[ -f "$pid_file" ]] && [[ "$(<"$pid_file")" == "$$" ]]; then
      rm -f "$pid_file"
    fi
  }
  trap finish EXIT

  export KERAS_HOME="${KERAS_HOME:-$project_dir/.keras}"
  export PYTHONUNBUFFERED=1

  printf 'Started: %s\n' "$(date --iso-8601=seconds)"
  printf 'PID: %s\n' "$$"
  printf 'Project: %s\n' "$project_dir"
  printf 'Output root: %s\n' "$output_root"
  printf 'Warm start: %s\n' "$warm_start_model"
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

  if [[ ! -f "$warm_start_model" ]]; then
    printf 'Warm-start model not found: %s\n' "$warm_start_model" >&2
    exit 1
  fi

  bash "$project_dir/run_wsl_python.sh" "$project_dir/train_multitask.py" \
    --manifest "$project_dir/data/combined_ham_bcn/combined_manifest.csv" \
    --output-root "$output_root" \
    --backbone efficientnetv2b1 \
    --image-size 300 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --data-workers 1 \
    --prefetch-batches 1 \
    --head-epochs 2 \
    --fine-tune-epochs 12 \
    --fine-tune-layers 100 \
    --learning-rate 6e-5 \
    --fine-tune-learning-rate 5e-6 \
    --diagnosis-loss-weight 0.30 \
    --metadata-modality-dropout 0.50 \
    --risk-features image_only \
    --risk-loss focal \
    --focal-gamma 1.5 \
    --warm-start-model "$warm_start_model" \
    --weights none \
    --mixed-precision auto \
    --fit-verbose 2
}

if [[ "${1:-}" == "--worker" ]]; then
  worker_main "$2"
  exit
fi

mkdir -p "$output_root"
if [[ -f "$pid_file" ]]; then
  existing_pid="$(<"$pid_file")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    printf 'Training is already running with PID %s.\n' "$existing_pid" >&2
    exit 2
  fi
  rm -f "$pid_file"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$output_root/gpu_training_$timestamp.log"
status_file="$output_root/gpu_training_$timestamp.exit"
rm -f "$status_file"

nohup setsid bash "$0" --worker "$status_file" > "$log_file" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pid_file"

sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  printf 'Training failed during startup. Log: %s\n' "$log_file" >&2
  tail -n 40 "$log_file" >&2 || true
  exit 1
fi

printf 'Training PID: %s\n' "$pid"
printf 'Training log: %s\n' "$log_file"
printf 'Exit status: %s\n' "$status_file"
