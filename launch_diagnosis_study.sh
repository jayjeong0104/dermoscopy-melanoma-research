#!/usr/bin/env bash
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="$project/artifacts/diagnosis_recovery_20261001"
python_bin="/mnt/d/SkinCancerData/venvs/skin-cancer-tf/bin/python"
cd "$project"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=2 CUDA_VISIBLE_DEVICES=""
"$python_bin" "$project/run_diagnosis_study.py" --prepare-only --experiment-dir "$output"
nohup setsid "$python_bin" "$project/queue_diagnosis_study.py" >> "$output/queue.log" 2>&1 < /dev/null &
pid=$!
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  wait "$pid"
  exit 0
fi
printf 'Diagnosis continuation PID: %s\nLog: %s\n' "$pid" "$output/queue.log"
