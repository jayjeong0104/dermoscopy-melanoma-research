#!/usr/bin/env bash
set -euo pipefail

venv_dir="${SKIN_CANCER_VENV:-/mnt/d/SkinCancerData/venvs/skin-cancer-tf}"
python_bin="$venv_dir/bin/python"

if [[ ! -x "$python_bin" ]]; then
  printf 'Python environment not found: %s\n' "$python_bin" >&2
  exit 1
fi

site_packages="$($python_bin -c 'import site; print(site.getsitepackages()[0])')"
mapfile -t cuda_lib_dirs < <(
  find "$site_packages/nvidia" -mindepth 2 -maxdepth 3 -type d -name lib -print | sort
)

if (( ${#cuda_lib_dirs[@]} == 0 )); then
  printf 'No NVIDIA runtime libraries found under %s\n' "$site_packages" >&2
  exit 1
fi

cuda_library_path="$(IFS=:; printf '%s' "${cuda_lib_dirs[*]}")"
export LD_LIBRARY_PATH="$cuda_library_path:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$python_bin" "$@"
