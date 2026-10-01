from __future__ import annotations

import hashlib
from pathlib import Path


def ensure_input_video(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Input video does not exist: {resolved}")
    return resolved


def job_id_for(path: Path) -> str:
    """Create a stable, inexpensive job id without hashing a multi-GB video."""
    video = ensure_input_video(path)
    stat = video.stat()
    identity = f"{video}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(identity).hexdigest()[:12]


def job_directory(root: Path, video: Path) -> Path:
    directory = root.expanduser().resolve() / f"{video.stem}-{job_id_for(video)}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
