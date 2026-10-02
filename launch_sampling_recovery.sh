#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment="$project_dir/artifacts/sampling_recovery_20260908"
mkdir -p "$experiment"
cd "$project_dir"
export PYTHONUNBUFFERED=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
nohup setsid bash "$project_dir/run_wsl_python.sh" "$project_dir/run_sampling_recovery.py" \
  --experiment-dir "$experiment" >> "$experiment/driver.log" 2>&1 < /dev/null &
pid=$!
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  wait "$pid"
  printf 'Driver exited. Inspect %s\n' "$experiment/driver.log"
  exit 0
fi
printf 'Sampling study driver PID: %s\nLog: %s\n' "$pid" "$experiment/driver.log"
