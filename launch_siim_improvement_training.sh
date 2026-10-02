#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

data_root="/mnt/d/SkinCancerData/SIIM_ISIC_2020"
image_dir="$data_root/train"
manifest_dir="$project_dir/data/combined_ham_bcn_derm12345_siim2020"
manifest="$manifest_dir/combined_manifest.csv"
folds="$manifest_dir/training_folds.csv"
output_root="$project_dir/artifacts/melanoma_siim2020_v2"
warm_start="${SKIN_CANCER_WARM_START:-$project_dir/artifacts/melanoma_v1/run_20260822_111527/best_model.keras}"
resume_run="${SKIN_CANCER_RESUME_RUN:-}"
pid_file="$output_root/active_training.pid"

prepare_data() {
  local image_count
  image_count="$(find "$image_dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)"
  if [[ "$image_count" != "33126" ]]; then
    printf 'Expected 33126 SIIM images, found %s. Run download_siim_isic2020.ps1 first.\n' "$image_count" >&2
    exit 1
  fi

  bash "$project_dir/run_wsl_python.sh" "$project_dir/prepare_siim_isic2020.py" \
    --image-dir "$image_dir" \
    --output-dir "$manifest_dir" \
    --hash-workers 4
  bash "$project_dir/run_wsl_python.sh" "$project_dir/build_siim_training_folds.py" \
    --manifest "$manifest" \
    --output "$folds" \
    --seed 42
}

worker_main() {
  # EXIT traps run after function-local variables leave scope, so keep the
  # status path in shell scope for reliable cleanup on interruption.
  worker_log_status="$1"
  local resume_args=()
  if [[ -n "$resume_run" ]]; then
    if [[ ! -d "$resume_run" ]]; then
      printf 'Resume run directory is missing: %s\n' "$resume_run" >&2
      exit 1
    fi
    resume_args=(--resume-run-dir "$resume_run")
  fi
  finish() {
    local status=$?
    {
      printf 'exit_code=%s\n' "$status"
      printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    } > "$worker_log_status"
    if [[ -f "$pid_file" ]] && [[ "$(<"$pid_file")" == "$$" ]]; then
      rm -f "$pid_file"
    fi
  }
  trap finish EXIT

  export KERAS_HOME="${KERAS_HOME:-$project_dir/.keras}"
  export PYTHONUNBUFFERED=1
  prepare_data

  if [[ ! -f "$warm_start" ]]; then
    printf 'Warm-start model is missing: %s\n' "$warm_start" >&2
    exit 1
  fi

  printf 'Running helper and model smoke tests.\n'
  bash "$project_dir/run_wsl_python.sh" -m unittest -v test_train_multitask_helpers.py
  bash "$project_dir/run_wsl_python.sh" "$project_dir/smoke_test_multitask_v2.py"

  printf 'Starting source-robust SIIM improvement run at %s\n' "$(date --iso-8601=seconds)"
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  bash "$project_dir/run_wsl_python.sh" "$project_dir/train_multitask.py" \
    --manifest "$manifest" \
    --output-root "$output_root" \
    "${resume_args[@]}" \
    --fold-assignments "$folds" \
    --test-fold 0 \
    --validation-fold 1 \
    --binary-target melanoma \
    --backbone efficientnetv2b1 \
    --image-size 300 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --data-workers 2 \
    --prefetch-batches 2 \
    --head-epochs 3 \
    --fine-tune-epochs 15 \
    --fine-tune-layers 140 \
    --learning-rate 5e-5 \
    --fine-tune-learning-rate 3e-6 \
    --diagnosis-loss-weight 0.20 \
    --sampling-alpha 0.0 \
    --positive-sampling-fraction 0.35 \
    --metadata-modality-dropout 0.50 \
    --risk-features image_only \
    --risk-loss bce \
    --minimum-operating-recall 0.95 \
    --tta-transforms 8 \
    --warm-start-model "$warm_start" \
    --weights none \
    --mixed-precision auto \
    --finalization-mode fresh_process \
    --evaluation-batch-size 2 \
    --fit-verbose 2

  local latest_run
  if [[ -n "$resume_run" ]]; then
    latest_run="$resume_run"
  else
    shopt -s nullglob
    local runs=("$output_root"/run_*)
    latest_run="$(printf '%s\n' "${runs[@]}" | sort | tail -n 1)"
  fi
  if [[ -z "$latest_run" ]]; then
    printf 'No completed run directory was produced.\n' >&2
    exit 1
  fi

  bash "$project_dir/run_wsl_python.sh" "$project_dir/evaluate_derm12345.py" \
    --run-dir "$latest_run" \
    --dataset-root "/mnt/d/SkinCancerData/DERM12345/test" \
    --training-manifest "$manifest" \
    --batch-size 2 \
    --hash-workers 2
  bash "$project_dir/run_wsl_python.sh" "$project_dir/evaluate_multitask_ph2.py" \
    --run-dir "$latest_run" \
    --ph2-root "$project_dir/PH2Dataset" \
    --batch-size 2
  bash "$project_dir/run_wsl_python.sh" "$project_dir/assess_market_readiness.py" \
    --run-dir "$latest_run" \
    --manifest "$manifest"
}

if [[ "${1:-}" == "--worker" ]]; then
  worker_main "$2"
  exit
fi

mkdir -p "$output_root"
worker_is_active() {
  local pid="$1"
  local command_line
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$command_line" == *"launch_siim_improvement_training.sh --worker"* ]]
}
if [[ -f "$pid_file" ]]; then
  existing_pid="$(<"$pid_file")"
  if worker_is_active "$existing_pid"; then
    printf 'SIIM improvement run is already active with PID %s.\n' "$existing_pid" >&2
    exit 2
  fi
  rm -f "$pid_file"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$output_root/training_$timestamp.log"
status_file="$output_root/training_$timestamp.exit"
nohup setsid bash "$0" --worker "$status_file" > "$log_file" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pid_file"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  printf 'SIIM improvement worker failed during startup. Log: %s\n' "$log_file" >&2
  tail -n 60 "$log_file" >&2 || true
  exit 1
fi
printf 'Training PID: %s\n' "$pid"
printf 'Training log: %s\n' "$log_file"
printf 'Exit status: %s\n' "$status_file"
