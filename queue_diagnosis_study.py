"""Single dependency-bound continuation; never overlaps the current GPU study."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from artifact_io import atomic_output

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/diagnosis_recovery_20261001"
DEPENDENCY = ROOT / "artifacts/context_recovery_20261001"


def save_state(value):
    with atomic_output(OUTPUT / "status.json") as temporary:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")


def check_code():
    plan = json.loads((OUTPUT / "experiment_plan.json").read_text())
    for name, expected in plan["code_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Queued experiment code changed: {name}")


def dependency_state(directory):
    status = json.loads((directory / "status.json").read_text())
    if status["status"] == "complete":
        marker = json.loads((directory / "complete.json").read_text())
        if hashlib.sha256((directory / "validation_report.json").read_bytes()).hexdigest() != marker["report_sha256"]:
            raise ValueError("Dependency completed report changed.")
        return "ready", None
    if status["status"] != "running":
        raise RuntimeError(f"Dependency is {status['status']}; not starting a follow-up.")
    pid = int(status.get("driver_pid", status["pid"]))
    return "waiting", pid


def existing_child():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if (str(OUTPUT).encode() in command and any(
                Path(os.fsdecode(arg)).name == "run_diagnosis_study.py" for arg in command if arg)):
            return int(entry.name)
    return None


def main():
    import fcntl
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_code()
        child_pid = existing_child()
        if child_pid is not None:
            print(f"Diagnosis child is already active: PID {child_pid}; not starting a duplicate.", flush=True)
            return
        if (OUTPUT / "complete.json").exists():
            complete = json.loads((OUTPUT / "complete.json").read_text())
            for name, expected in complete["outputs_sha256"].items():
                if hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() != expected:
                    raise ValueError("Completed diagnosis output changed.")
            print("Follow-up already complete; not queueing again.", flush=True)
            return
        try:
            deadline = time.monotonic() + 12 * 60 * 60
            announced = False
            while True:
                state, pid = dependency_state(DEPENDENCY)
                if state == "ready":
                    break
                command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
                if not any(Path(os.fsdecode(arg)).name == "run_context_study.py" for arg in command if arg):
                    raise RuntimeError("Dependency driver is no longer active; resume it before this follow-up.")
                if not announced:
                    save_state({"status": "queued", "phase": "waiting_for_context_study", "queue_pid": os.getpid(), "dependency_pid": pid})
                    print(f"Queued diagnosis study behind context driver {pid}; no fitting until it completes.", flush=True)
                    announced = True
                if time.monotonic() >= deadline:
                    raise TimeoutError("Dependency wait exceeded 12 hours; no training started.")
                time.sleep(30)
            check_code()
            with (OUTPUT / "driver.log").open("a") as log:
                child = subprocess.Popen([sys.executable, str(ROOT / "run_diagnosis_study.py"),
                    "--experiment-dir", str(OUTPUT)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    save_state({"status": "starting", "pid": child.pid, "queue_pid": os.getpid()})
                    print(f"Starting diagnosis study: PID {child.pid}", flush=True)
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    child.wait()
                    raise
            if code:
                raise RuntimeError(f"Diagnosis study exited {code}; see driver.log.")
            print("Diagnosis study completed; see validation_report.json.", flush=True)
        except BaseException as error:
            save_state({"status": "failed", "phase": "dependency_or_followup", "error": str(error), "queue_pid": os.getpid()})
            raise


if __name__ == "__main__":
    main()
