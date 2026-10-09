"""Record the reviewed I/O-only patch without changing the original study plan."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from compare_model_runs import file_hash
from foundation_common import ROOT, save_json
from run_foundation_experiment import assert_idle, verify_plan


ORIGINAL = {
    "foundation_common.py": "9c3cfccf7843f80396972f0ffb71c51971eeffbb85400ac85616a685ba0c9456",
    "train_foundation_head.py": "4420106484fd39f26b7fb3ed284a84ef3d4628469710d28e8b9bc7cd6fd5ec40",
    "run_foundation_experiment.py": "877e90696138f772165388ea6aa8456ebd68b2ae65b21cd0faafd3df47241058",
}


def register(experiment: Path) -> None:
    import fcntl
    with (ROOT / "artifacts/recovery_pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle()
        record = experiment / "io_repair.json"
        if record.exists():
            verify_plan(experiment)
            print("Existing I/O repair verified; no artifacts overwritten.")
            return
        plan_path = experiment / "experiment_plan.json"
        plan = json.loads(plan_path.read_text())
        for name, expected in plan["code_sha256"].items():
            if file_hash(experiment / "code_snapshot" / name) != expected:
                raise ValueError(f"Original source snapshot changed: {name}")
            if name in ORIGINAL:
                if expected != ORIGINAL[name]:
                    raise ValueError(f"Unsupported prior source revision: {name}")
            elif file_hash(ROOT / name) != expected:
                raise ValueError(f"Unrelated experiment source changed: {name}")
        for name, expected in plan["input_sha256"].items():
            if file_hash(Path(name)) != expected:
                raise ValueError(f"Experiment input changed: {name}")
        for marker in (experiment / "phase_done").glob("*.json"):
            completed = json.loads(marker.read_text())
            for name, expected in completed.get("outputs", {}).items():
                if file_hash(experiment / name) != expected:
                    raise ValueError(f"Completed artifact changed: {name}")
        snapshot = experiment / "io_repair_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name in (*ORIGINAL, "artifact_io.py"):
            target = snapshot / name
            if target.exists() and file_hash(target) != file_hash(ROOT / name):
                raise ValueError(f"An earlier repair snapshot differs: {name}")
            if not target.exists():
                shutil.copy2(ROOT / name, target)
        for name in ("status.json", "driver.log", "train_panderm_base.log"):
            target = snapshot / name
            if not target.exists():
                shutil.copy2(experiment / name, target)
        save_json(record, {
            "version": 1, "scope": "artifact_io_only",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "original_plan_sha256": file_hash(plan_path),
            "reason": "PermissionError replacing a history JSON in the OneDrive-backed WSL directory.",
            "changes": ["Unique same-directory temporary files and fsync before publication",
                        "Retry access/sharing/busy replacement errors for at most 60 seconds",
                        "Atomic JSON, feature chunk, classifier checkpoint and prediction writes",
                        "Terminate child if its initial status cannot be published"],
            "code_overrides": {name: {"before_sha256": original,
                               "after_sha256": file_hash(ROOT / name)} for name, original in ORIGINAL.items()},
            "additional_code_sha256": {"artifact_io.py": file_hash(ROOT / "artifact_io.py")},
            "registration_script_sha256": file_hash(Path(__file__)),
            "protocol_changed": None,
            "protocol_review_status": "unverified_by_registration",
            "protocol_review_limitation": "Hashes preserve the registered bytes; registration "
            "does not prove that arbitrary edits in the allowed files are semantically I/O-only.",
            "cached_features_modified": False,
            "resume_policy": "Reuse completed extraction and completed head choices; restart only incomplete small head fit.",
        })
        verify_plan(experiment)
        print("Repair bytes registered. Original plan, inputs and completed outputs verified unchanged; "
              "semantic I/O-only scope and protocol equivalence remain unverified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    register(args.experiment_dir.resolve())
