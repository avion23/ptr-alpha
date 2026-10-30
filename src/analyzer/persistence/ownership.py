"""Single-host writer ownership and explicit immutable read publication."""
from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path


class OwnershipError(RuntimeError):
    pass


def _path(db_path) -> Path:
    return Path(db_path).expanduser().resolve()


def _lock_path(db_path) -> Path:
    return _path(db_path).with_suffix(_path(db_path).suffix + ".owner.lock")


def publication_pointer(db_path) -> Path:
    return _path(db_path).with_suffix(_path(db_path).suffix + ".published.json")


@contextmanager
def database_owner(db_path, purpose):
    """Acquire before connecting; never wait behind a competing writer."""
    if str(db_path) == ":memory:":
        yield None
        return
    path = _path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_path(path).open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OwnershipError(f"Database already has a writer owner: {path}") from exc
        try:
            yield lease
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def publication_state(db_path) -> dict:
    if str(db_path) == ":memory:":
        return {"published": False, "owner_active": False}
    pointer = publication_pointer(db_path)
    state = {"published": False}
    if pointer.exists():
        state = json.loads(pointer.read_text())
        path = Path(state["path"])
        if not path.is_file():
            raise OwnershipError(f"Published snapshot is missing: {path}")
        state["published"] = True
    lock = _lock_path(db_path)
    active = False
    if lock.exists():
        with lock.open("r") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                active = True
            else:
                fcntl.flock(lease, fcntl.LOCK_UN)
    return {**state, "owner_active": active}


def resolve_read_path(db_path) -> Path:
    state = publication_state(db_path)
    if state["published"]:
        return Path(state["path"])
    if state["owner_active"]:
        raise OwnershipError("Writer active and no published snapshot exists")
    return _path(db_path)
