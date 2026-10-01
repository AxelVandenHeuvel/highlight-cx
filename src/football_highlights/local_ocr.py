"""Local scorebug OCR for validating live broadcast footage."""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_highlights.media import VideoMetadata
from football_highlights.scorebug import FOX_BOTTOM_CENTER

_CLOCK = re.compile(r"(?<!\d)(\d{1,2})\s*:\s*([0-5]\d)(?!\d)")
_QUARTERS = {
    "1ST": 1,
    "2ND": 2,
    "3RD": 3,
    "4TH": 4,
    "OT": 5,
    "1OT": 5,
}


@dataclass(frozen=True)
class ParsedScorebug:
    quarter: int
    clock_seconds: int
    confidence: float


@dataclass(frozen=True)
class ParsedScore:
    left: int
    right: int
    confidence: float


def parse_scorebug_tokens(tokens: Iterable[tuple[str, float]]) -> ParsedScorebug | None:
    """Parse OCR text/confidence pairs without depending on a specific OCR engine."""

    quarter: int | None = None
    quarter_confidence = 0.0
    clock_seconds: int | None = None
    clock_confidence = 0.0
    for raw_text, confidence in tokens:
        text = re.sub(r"\s+", "", raw_text.upper())
        normalized = text.replace("IST", "1ST").replace("|ST", "1ST")
        if normalized in _QUARTERS and confidence > quarter_confidence:
            quarter = _QUARTERS[normalized]
            quarter_confidence = confidence
        match = _CLOCK.search(text)
        if match and confidence > clock_confidence:
            minutes, seconds = (int(value) for value in match.groups())
            if minutes <= 15:
                clock_seconds = minutes * 60 + seconds
                clock_confidence = confidence
    if quarter is None or clock_seconds is None:
        return None
    return ParsedScorebug(quarter, clock_seconds, min(quarter_confidence, clock_confidence))


class RapidOcrBackend:
    """Small offline OCR backend; model files are installed with the analysis extra."""

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("install the analysis extra: uv sync --extra analysis") from exc
        self._engine = RapidOCR()

    def __call__(self, image: Any) -> list[tuple[str, float]]:
        source = str(image) if isinstance(image, Path) else image
        result, _ = self._engine(source)
        return [(str(item[1]), float(item[2])) for item in (result or [])]

    def read_score(self, image: Any) -> ParsedScore | None:
        source = str(image) if isinstance(image, Path) else image
        result, _ = self._engine(source)
        if not result:
            return None
        width = float(image.shape[1]) if hasattr(image, "shape") else 960.0
        height = float(image.shape[0]) if hasattr(image, "shape") else 270.0
        sides: dict[str, tuple[int, float]] = {}
        for box, raw_text, raw_confidence in result:
            text = re.sub(r"\D", "", str(raw_text))
            if not text or len(text) > 2:
                continue
            center_x = sum(float(point[0]) for point in box) / (4 * width)
            center_y = sum(float(point[1]) for point in box) / (4 * height)
            if not 0.30 <= center_y <= 0.72:
                continue
            side = (
                "left" if 0.22 <= center_x <= 0.43 else "right" if 0.57 <= center_x <= 0.78 else ""
            )
            if side and float(raw_confidence) > sides.get(side, (-1, 0.0))[1]:
                sides[side] = (int(text), float(raw_confidence))
        if "left" not in sides or "right" not in sides:
            return None
        return ParsedScore(
            sides["left"][0], sides["right"][0], min(sides["left"][1], sides["right"][1])
        )

    def read_scorebug(self, image: Any, timestamp: float) -> dict[str, Any] | None:
        parsed = parse_scorebug_tokens(self(image))
        if parsed is None:
            return None
        return {
            "timestamp": timestamp,
            "visible": True,
            "quarter": parsed.quarter,
            "clock_seconds": parsed.clock_seconds,
            "confidence": parsed.confidence,
        }


def scan_scorebug(
    source: Path,
    metadata: VideoMetadata,
    *,
    interval_seconds: float = 20.0,
    ocr: Callable[[Path], list[tuple[str, float]]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[tuple[float, ParsedScorebug]]:
    """Sample a broadcast and OCR its scorebug entirely on the local machine."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    backend = ocr or RapidOcrBackend()
    x, y, width, height = FOX_BOTTOM_CENTER.pixels(metadata)
    timestamps = [
        min(index * interval_seconds, metadata.duration)
        for index in range(int(metadata.duration // interval_seconds) + 1)
    ]
    readings: list[tuple[float, ParsedScorebug]] = []
    with tempfile.TemporaryDirectory(prefix="nfl-scorebug-") as temporary:
        image = Path(temporary) / "scorebug.png"
        for index, timestamp in enumerate(timestamps, start=1):
            command = [
                "ffmpeg",
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
                str(image),
            ]
            completed = subprocess.run(command, capture_output=True, check=False)
            if completed.returncode == 0 and image.exists():
                parsed = parse_scorebug_tokens(backend(image))
                if parsed is not None:
                    readings.append((timestamp, parsed))
            if progress is not None:
                progress(index, len(timestamps))
    return readings
