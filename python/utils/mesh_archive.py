"""Where the 10k benchmark meshes live, and a clear error when they are absent.

The meshes are not in the repository; they are a separate download. Scripts
that need them default to ``dataset/bubble_gym/meshes10k`` (or
``BUBBLEGYM_MESH_ROOT``), and without this check a missing archive surfaced as
a bare "No such file or directory" with nothing pointing at the download.
"""

from __future__ import annotations

from pathlib import Path

MESH_ARCHIVE_URL = (
    "https://drive.google.com/file/d/1RaK-cJ7NlHzVyHUncHQMwzrxlvt204PB/view?usp=sharing"
)
ENV_VAR = "BUBBLEGYM_MESH_ROOT"


class MeshArchiveNotFound(FileNotFoundError):
    """The benchmark mesh archive is not where the script was told to look."""


def require_mesh_root(root: Path | str) -> Path:
    """Return ``root`` if it is a directory, else raise with the download pointer."""
    root = Path(root)
    if root.is_dir():
        return root
    raise MeshArchiveNotFound(
        f"Mesh archive not found at {root}.\n"
        f"The 10,000 benchmark meshes are distributed separately from the "
        f"repository. Download them from\n    {MESH_ARCHIVE_URL}\n"
        f"unpack so the VOF/ and LBM/ folders sit side by side, then either "
        f"place them at dataset/bubble_gym/meshes10k or set {ENV_VAR} to "
        f"that folder."
    )
