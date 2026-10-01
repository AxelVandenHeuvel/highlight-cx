from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from football_highlights.media import MediaError, render_interval
from football_highlights.models import HighlightManifest


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "highlight"


def concat_clips(
    clips: list[Path],
    output_path: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    runner: Any = subprocess.run,
) -> None:
    if not clips:
        raise ValueError("at least one clip is required")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_file = output_path.parent / "concat.txt"
    lines = []
    for clip in clips:
        escaped = str(clip.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]
    try:
        runner(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MediaError("ffmpeg concat failed") from exc


def render_manifest(
    source_path: Path,
    manifest: HighlightManifest,
    output_directory: Path,
    *,
    use_videotoolbox: bool = True,
) -> tuple[list[Path], Path]:
    """Render full broadcast frames over a blurred vertical background."""

    if not manifest.segments:
        raise ValueError("manifest contains no highlight segments")

    clips_directory = output_directory / "clips"
    clips_directory.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for index, segment in enumerate(manifest.segments, start=1):
        clip = clips_directory / f"{index:02d}-{_slug(segment.event_type)}.mp4"
        render_interval(
            source_path,
            clip,
            segment.source_start,
            segment.source_end,
            use_videotoolbox=use_videotoolbox,
        )
        clips.append(clip)

    reel = output_directory / "reel.mp4"
    concat_clips(clips, reel)
    return clips, reel
