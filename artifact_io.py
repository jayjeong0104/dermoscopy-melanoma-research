"""Atomic artifact publication with bounded retries for Windows sharing locks."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import tempfile
import time
import warnings


def replace_with_retry(source: Path, destination: Path, timeout: float = 60.) -> None:
    deadline = time.monotonic() + timeout
    delay = .1
    attempts = 0
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            transient = error.errno in {errno.EACCES, errno.EPERM, errno.EBUSY} or getattr(
                error, "winerror", None) in {5, 32, 33}
            remaining = deadline - time.monotonic()
            if not transient or remaining <= 0:
                error.add_note(f"Artifact publication failed; prepared file retained at {source}. "
                               f"The previous {destination} was not deleted.")
                raise
            if attempts == 0:
                warnings.warn(f"Artifact replacement temporarily denied: {destination}; "
                              f"retrying for at most {timeout:g}s.", RuntimeWarning, stacklevel=2)
            time.sleep(min(delay, remaining))
            delay = min(2., delay * 2)
            attempts += 1


@contextmanager
def atomic_output(destination: Path):
    """Publish only a fully written file; never truncate an existing checkpoint."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep .keras/.npz suffixes so serializers select the intended format.
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.stem}.",
        suffix=f".pending{destination.suffix}", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # Cleanup must not hide the original serialization error.
        raise
    # Leave the complete temporary artifact available if publication stays blocked.
    replace_with_retry(temporary, destination)


def save_model(model, path: Path) -> None:
    with atomic_output(path) as temporary:
        model.save(temporary)


def save_csv(frame, path: Path) -> None:
    with atomic_output(path) as temporary:
        frame.to_csv(temporary, index=False)
