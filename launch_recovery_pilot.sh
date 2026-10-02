#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment="$project_dir/artifacts/recovery_pilot_20260906"
mkdir -p "$experiment"
export PYTHONUNBUFFERED=1
cd "$project_dir"

# The driver owns a kernel lock and checks for other training/evaluation processes.
nohup setsid bash "$project_dir/run_wsl_python.sh" "$project_dir/run_recovery_pilot.py" \
  --experiment-dir "$experiment" >> "$experiment/driver.log" 2>&1 < /dev/null &
pid=$!
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  if wait "$pid"; then
    printf 'Pilot exited successfully. Report/status: %s\n' "$experiment/status.json"
    exit 0
  fi
  printf 'Pilot startup failed. Inspect %s\n' "$experiment/driver.log" >&2
  exit 1
fi
printf 'Pilot driver PID: %s\nLog: %s\nStatus: %s\n' \
  "$pid" "$experiment/driver.log" "$experiment/status.json"
