"""Keep epoch-level recovery and validation-driven callback state together."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import tensorflow as tf


class CheckpointEarlyStopping(tf.keras.callbacks.EarlyStopping):
    """Use the shared best checkpoint, including when its epoch preceded recovery."""

    def __init__(self, checkpoint: Path, **kwargs):
        super().__init__(restore_best_weights=False, **kwargs)
        self.checkpoint = Path(checkpoint)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if self.checkpoint.exists():
            # Stages recompile their optimizer; restore the coherent best checkpoint
            # here rather than retaining another full model copy in host memory.
            self.model.load_weights(self.checkpoint)


class TrainingStateBackup(tf.keras.callbacks.BackupAndRestore):
    """Run LAST: the backup must contain the LR and patience after this epoch."""

    FIELDS = ("best", "wait", "cooldown_counter", "stopped_epoch", "best_epoch")

    def __init__(self, backup_dir: Path, controllers: dict[str, tf.keras.callbacks.Callback]):
        super().__init__(backup_dir=str(backup_dir), delete_checkpoint=True)
        self.state_metadata = Path(backup_dir) / "training_metadata.json"
        self.controllers = controllers

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if not self.state_metadata.exists():
            return
        metadata = json.loads(self.state_metadata.read_text(encoding="utf-8"))
        states = metadata.get("validation_callback_state")
        if states is None:
            warnings.warn("Legacy training backup has no validation callback state; "
                          "patience counters restart and continuation is not exact.", RuntimeWarning)
            return
        if set(states) != set(self.controllers):
            raise ValueError("Recovery callback configuration does not match its backup.")
        for name, callback in self.controllers.items():
            state = states[name]
            if state["monitor"] != callback.monitor:
                raise ValueError("Recovery validation monitor does not match its backup.")
            for field, value in state["values"].items():
                if field not in self.FIELDS:
                    raise ValueError(f"Unsupported recovery callback field: {field}")
                setattr(callback, field, value)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        metadata = json.loads(self.state_metadata.read_text(encoding="utf-8"))
        states = {}
        for name, callback in self.controllers.items():
            values = {}
            for field in self.FIELDS:
                if hasattr(callback, field):
                    value = getattr(callback, field)
                    values[field] = value.item() if hasattr(value, "item") else value
            states[name] = {"monitor": callback.monitor, "values": values}
        metadata["validation_callback_state"] = states
        temporary = self.state_metadata.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(self.state_metadata)
