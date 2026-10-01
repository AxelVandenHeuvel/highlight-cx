from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_highlights.media import MediaError, VideoMetadata


@dataclass(frozen=True)
class ScorebugRegion:
    """A resolution-independent crop rectangle expressed as 0..1 fractions."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("scorebug region values must be between 0 and 1")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("scorebug width and height must be positive")
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("scorebug region must fit inside the video frame")

    def pixels(self, metadata: VideoMetadata) -> tuple[int, int, int, int]:
        x = round(self.x * metadata.width)
        y = round(self.y * metadata.height)
        width = round(self.width * metadata.width)
        height = round(self.height * metadata.height)
        # FFmpeg's crop filter is most predictable with even dimensions.
        return x // 2 * 2, y // 2 * 2, width // 2 * 2, height // 2 * 2


FOX_BOTTOM_CENTER = ScorebugRegion(x=0.25, y=0.75, width=0.50, height=0.25)


def extract_scorebug_sample(
    source: Path,
    output: Path,
    timestamp: float,
    metadata: VideoMetadata,
    *,
    region: ScorebugRegion = FOX_BOTTOM_CENTER,
    ffmpeg_bin: str = "ffmpeg",
    runner: Any = subprocess.run,
) -> None:
    if timestamp < 0 or timestamp > metadata.duration:
        raise ValueError("timestamp must fall within the source video")
    x, y, width, height = region.pixels(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        f"crop={width}:{height}:{x}:{y}",
        "-y",
        str(output),
    ]
    try:
        runner(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MediaError(f"failed to extract scorebug sample at {timestamp:.3f}s") from exc
