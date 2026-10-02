#!/usr/bin/env bash
set -euo pipefail

# Separate environment: never upgrade the working TensorFlow/CUDA installation.
env_dir="/mnt/d/SkinCancerData/venvs/skin-cancer-features"
if [[ ! -x "$env_dir/bin/python" ]]; then
  /usr/bin/python3 -m venv "$env_dir"
fi
"$env_dir/bin/python" -m pip install --no-cache-dir 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu118
"$env_dir/bin/python" -m pip install --no-cache-dir 'timm==0.9.16' 'gdown==5.2.0' 'numpy==2.2.6' 'pandas==2.3.2' 'scikit-learn==1.7.1'
"$env_dir/bin/python" -m pip check
