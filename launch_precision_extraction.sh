#!/usr/bin/env bash
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="$project/artifacts/precision_rebuild_20261006"
mkdir -p "$output"
cd "$project"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=2 TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
nohup setsid bash "$project/run_wsl_python.sh" "$project/extract_clean_reference.py" \
  >> "$output/extraction.log" 2>&1 < /dev/null &
pid=$!
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  wait "$pid"
  exit 0
fi
printf 'Clean feature extraction PID: %s\nLog: %s\n' "$pid" "$output/extraction.log"
