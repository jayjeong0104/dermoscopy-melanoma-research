#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
source_root="$project_dir/artifacts/multitask_v2"
source_run="${SKIN_CANCER_SOURCE_RUN:-$source_root/run_20260817_182119}"
output_root="${SKIN_CANCER_MELANOMA_OUTPUT_ROOT:-$project_dir/artifacts/melanoma_v1}"
queue_pid_file="$output_root/queued_training.pid"
training_manifest="$project_dir/data/combined_ham_bcn_derm12345/combined_manifest.csv"
fold_assignments="$project_dir/data/combined_ham_bcn_derm12345/grouped_folds.csv"

select_warm_start() {
  local candidate
  for candidate in \
    "$source_run/best_model.keras" \
    "$source_run/best_finetune_model.keras" \
    "$source_run/best_head_model.keras"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

wait_for_source_training() {
  local active_pid_file="$source_root/active_training.pid"
  while [[ -f "$active_pid_file" ]]; do
    local active_pid
    active_pid="$(<"$active_pid_file")"
    if ! kill -0 "$active_pid" 2>/dev/null; then
      break
    fi
    printf 'Waiting for source training PID %s at %s\n' "$active_pid" "$(date --iso-8601=seconds)"
    sleep 60
  done
  sleep 3
}

prepare_derm12345_test() {
  local archive="/mnt/d/SkinCancerData/DERM12345/derm12345_test.zip"
  local destination="/mnt/d/SkinCancerData/DERM12345/test"
  local expected_md5="eaf822e5a3951875b7466047426b2358"
  if [[ ! -f "$archive" ]]; then
    printf 'DERM12345 archive is not present; external evaluation preparation skipped.\n'
    return 0
  fi

  local actual_md5
  actual_md5="$(md5sum "$archive" | awk '{print $1}')"
  if [[ "$actual_md5" != "$expected_md5" ]]; then
    printf 'DERM12345 archive checksum mismatch: %s\n' "$actual_md5" >&2
    return 1
  fi

  local image_count=0
  if [[ -d "$destination" ]]; then
    image_count="$(find "$destination" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)"
  fi
  if [[ "$image_count" == "2485" ]]; then
    printf 'DERM12345 test images are already extracted and verified by count.\n'
    return 0
  fi

  printf 'Extracting the verified DERM12345 test archive.\n'
  mkdir -p "$destination"
  python3 -m zipfile -e "$archive" "$destination"
  image_count="$(find "$destination" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)"
  if [[ "$image_count" != "2485" ]]; then
    printf 'Expected 2485 DERM12345 test images, found %s.\n' "$image_count" >&2
    return 1
  fi
  printf 'Extracted and counted %s DERM12345 test images.\n' "$image_count"
}

prepare_derm12345_training() {
  local destination="/mnt/d/SkinCancerData/DERM12345/train"
  local archives=(
    "/mnt/d/SkinCancerData/DERM12345/derm12345_train_part_1.zip"
    "/mnt/d/SkinCancerData/DERM12345/derm12345_train_part_2.zip"
  )
  local checksums=(
    "22678b59d31da37883159ab5e42679ac"
    "83c043205737fc03f889ba3e65ea835f"
  )
  local index
  for index in 0 1; do
    if [[ ! -f "${archives[$index]}" ]]; then
      printf 'Required DERM12345 training archive is missing: %s\n' "${archives[$index]}" >&2
      return 1
    fi
    local actual_md5
    actual_md5="$(md5sum "${archives[$index]}" | awk '{print $1}')"
    if [[ "$actual_md5" != "${checksums[$index]}" ]]; then
      printf 'DERM12345 training archive checksum mismatch: %s\n' "${archives[$index]}" >&2
      return 1
    fi
  done

  local image_count=0
  if [[ -d "$destination" ]]; then
    image_count="$(find "$destination" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)"
  fi
  if [[ "$image_count" != "9860" ]]; then
    printf 'Extracting verified DERM12345 training archives.\n'
    mkdir -p "$destination"
    for index in 0 1; do
      python3 -m zipfile -e "${archives[$index]}" "$destination"
    done
    image_count="$(find "$destination" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)"
  fi
  if [[ "$image_count" != "9860" ]]; then
    printf 'Expected 9860 DERM12345 training images, found %s.\n' "$image_count" >&2
    return 1
  fi
  printf 'Preparing exact-deduplicated patient-grouped training manifest.\n'
  bash "$project_dir/run_wsl_python.sh" "$project_dir/prepare_derm12345_training.py" \
    --image-dir "$destination" \
    --output-dir "$project_dir/data/combined_ham_bcn_derm12345" \
    --hash-workers 2
  bash "$project_dir/run_wsl_python.sh" "$project_dir/build_grouped_folds.py" \
    --manifest "$training_manifest" \
    --output "$fold_assignments" \
    --folds 5 \
    --seed 42
}

worker_main() {
  # The EXIT trap runs after this function returns, so this must outlive local scope.
  worker_status_file="$1"

  finish() {
    local status=$?
    {
      printf 'exit_code=%s\n' "$status"
      printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    } > "$worker_status_file"
    if [[ -f "$queue_pid_file" ]] && [[ "$(<"$queue_pid_file")" == "$$" ]]; then
      rm -f "$queue_pid_file"
    fi
  }
  trap finish EXIT

  export KERAS_HOME="${KERAS_HOME:-$project_dir/.keras}"
  export PYTHONUNBUFFERED=1

  printf 'Queue started: %s\n' "$(date --iso-8601=seconds)"
  printf 'Queue PID: %s\n' "$$"
  wait_for_source_training
  prepare_derm12345_test
  prepare_derm12345_training

  local warm_start_model
  if ! warm_start_model="$(select_warm_start)"; then
    printf 'No usable checkpoint found under %s\n' "$source_run" >&2
    exit 1
  fi
  printf 'Selected warm start: %s\n' "$warm_start_model"
  printf 'Running helper tests before GPU training.\n'
  bash "$project_dir/run_wsl_python.sh" -m unittest -v test_train_multitask_helpers.py
  bash "$project_dir/run_wsl_python.sh" "$project_dir/smoke_test_multitask_v2.py"

  printf 'Melanoma training started: %s\n' "$(date --iso-8601=seconds)"
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  bash "$project_dir/run_wsl_python.sh" "$project_dir/train_multitask.py" \
    --manifest "$training_manifest" \
    --output-root "$output_root" \
    --fold-assignments "$fold_assignments" \
    --test-fold 0 \
    --validation-fold 1 \
    --binary-target melanoma \
    --backbone efficientnetv2b1 \
    --image-size 300 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --data-workers 1 \
    --prefetch-batches 1 \
    --head-epochs 4 \
    --fine-tune-epochs 15 \
    --fine-tune-layers 100 \
    --learning-rate 1e-4 \
    --fine-tune-learning-rate 5e-6 \
    --diagnosis-loss-weight 0.25 \
    --sampling-alpha 0.50 \
    --positive-sampling-fraction 0.45 \
    --metadata-modality-dropout 0.50 \
    --risk-features image_only \
    --risk-loss focal \
    --focal-gamma 1.5 \
    --minimum-operating-recall 0.95 \
    --tta-transforms 8 \
    --warm-start-model "$warm_start_model" \
    --reset-risk-head \
    --weights none \
    --mixed-precision auto \
    --finalization-mode fresh_process \
    --evaluation-batch-size 2 \
    --fit-verbose 2

  shopt -s nullglob
  local runs=("$output_root"/run_*)
  if (( ${#runs[@]} == 0 )); then
    printf 'Training completed without a timestamped run directory.\n' >&2
    exit 1
  fi
  local latest_run
  latest_run="$(printf '%s\n' "${runs[@]}" | sort | tail -n 1)"
  printf 'Evaluating locked DERM12345 test split with run: %s\n' "$latest_run"
  bash "$project_dir/run_wsl_python.sh" "$project_dir/evaluate_derm12345.py" \
    --run-dir "$latest_run" \
    --dataset-root "/mnt/d/SkinCancerData/DERM12345/test" \
    --training-manifest "$training_manifest" \
    --batch-size 2 \
    --hash-workers 2
  printf 'Evaluating development-only PH2 cohort.\n'
  bash "$project_dir/run_wsl_python.sh" "$project_dir/evaluate_multitask_ph2.py" \
    --run-dir "$latest_run" \
    --ph2-root "$project_dir/PH2Dataset" \
    --batch-size 2

  printf 'Generating fail-closed market-readiness assessment.\n'
  bash "$project_dir/run_wsl_python.sh" "$project_dir/assess_market_readiness.py" \
    --run-dir "$latest_run" \
    --manifest "$training_manifest"
}

if [[ "${1:-}" == "--worker" ]]; then
  worker_main "$2"
  exit
fi

mkdir -p "$output_root"
if [[ -f "$queue_pid_file" ]]; then
  existing_pid="$(<"$queue_pid_file")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    printf 'Melanoma training is already queued with PID %s.\n' "$existing_pid" >&2
    exit 2
  fi
  rm -f "$queue_pid_file"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$output_root/queued_training_$timestamp.log"
status_file="$output_root/queued_training_$timestamp.exit"
rm -f "$status_file"

nohup setsid bash "$0" --worker "$status_file" > "$log_file" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$queue_pid_file"

sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  printf 'Melanoma queue failed during startup. Log: %s\n' "$log_file" >&2
  tail -n 40 "$log_file" >&2 || true
  exit 1
fi

printf 'Queue PID: %s\n' "$pid"
printf 'Queue log: %s\n' "$log_file"
printf 'Exit status: %s\n' "$status_file"
