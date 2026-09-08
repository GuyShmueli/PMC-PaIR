from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from .errors import PipelineError


if os.name == "nt":
    import msvcrt
elif os.name == "posix":
    import fcntl


def _acquire_nonblocking(handle: BinaryIO) -> None:
    if os.name == "posix":
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    if os.name == "nt":
        # ``msvcrt.locking`` locks bytes from the current file position, so the
        # lock file needs a stable byte even when it has just been created.
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    raise PipelineError(f"Unsupported operating system for run locking: {os.name!r}")


def _release(handle: BinaryIO) -> None:
    if os.name == "posix":
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    raise PipelineError(f"Unsupported operating system for run locking: {os.name!r}")


@contextmanager
def run_lock(output_dir: str | Path) -> Iterator[None]:
    """Prevent concurrent CLI commands from using one run on POSIX or Windows."""

    run_dir = Path(output_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory is missing: {run_dir}")
    lock_path = run_dir / ".pipeline.lock"
    with lock_path.open("a+b") as handle:
        try:
            _acquire_nonblocking(handle)
        except OSError as exc:
            raise PipelineError(
                f"Another pipeline command is already using this run: {run_dir}"
            ) from exc
        try:
            yield
        finally:
            _release(handle)


__all__ = ["run_lock"]
