#!/usr/bin/env bash
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# > 1 )); then
  printf 'Usage: %s [reference|natural_lesion]\n' "$0" >&2
  exit 2
fi
recipe="${1:-reference}"
case "$recipe" in
  reference) output="$project/artifacts/clean_melanoma_20261005" ;;
  natural_lesion) output="$project/artifacts/natural_lesion_20261006" ;;
  *) printf 'Unknown recipe: %s. Choose reference or natural_lesion.\n' "$recipe" >&2; exit 2 ;;
esac
mkdir -p "$output"
cd "$project"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=2 TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
nohup setsid bash "$project/run_wsl_python.sh" "$project/run_clean_training.py" \
  --recipe "$recipe" --experiment-dir "$output" >> "$output/driver.log" 2>&1 < /dev/null &
pid=$!
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  wait "$pid"
  exit 0
fi
printf 'Clean training driver PID: %s\nLog: %s\n' "$pid" "$output/training.log"
