#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${SKIN_CANCER_VENV:-/mnt/d/SkinCancerData/venvs/skin-cancer-tf}"

if (( EUID == 0 )); then
  sudo_cmd=()
else
  sudo_cmd=(sudo)
fi

"${sudo_cmd[@]}" apt-get update
"${sudo_cmd[@]}" apt-get install -y python3-venv python3-pip
python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$project_dir/requirements-gpu.txt"

SKIN_CANCER_VENV="$venv_dir" bash "$project_dir/run_wsl_python.sh" - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)
if not gpus:
    raise SystemExit("TensorFlow did not detect a GPU.")
PY

printf '\nGPU environment ready: %s\n' "$venv_dir"
