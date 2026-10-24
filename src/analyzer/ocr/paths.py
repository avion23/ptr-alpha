"""Default OCR storage paths, independent of the operational wrapper location."""
import os
from pathlib import Path


def default_data_dir() -> Path:
    """Honor configured storage; source-tree drains default to that tree's data.

    Installed applications use the same CWD-relative ``data`` default as the
    analyzer CLI, or an explicit DATA__DATA_DIR for CWD-independent storage.
    """
    configured = os.environ.get("DATA__DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "pyproject.toml").is_file():
        return source_root / "data"
    return Path("data").resolve()


DATA_DIR = default_data_dir()
STAGING_ROOT = DATA_DIR / "staging" / "glm-backlog"
