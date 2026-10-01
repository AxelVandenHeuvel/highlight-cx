"""Small, dependency-free data models for football highlight jobs.

The models deliberately contain no media-processing logic.  They are the
serializable contract between analysis and rendering stages.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """Raised when a model contains invalid or incomplete data."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number")
    return float(value)


def _nonnegative(value: Any, name: str) -> float:
    result = _number(value, name)
    if result < 0:
        raise ValidationError(f"{name} must be nonnegative")
    return result


@dataclass
class MediaInfo:
    """Basic metadata for the source video."""

    source_path: str
    duration_seconds: float
    width: int
    height: int
    fps: float

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, str) or not self.source_path.strip():
            raise ValidationError("source_path must be a non-empty string")
        self.duration_seconds = _nonnegative(self.duration_seconds, "duration_seconds")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValidationError("width must be a positive integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise ValidationError("height must be a positive integer")
        self.fps = _number(self.fps, "fps")
        if self.fps <= 0:
            raise ValidationError("fps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MediaInfo:
        return cls(**dict(data))


@dataclass
class HighlightSegment:
    """A source-video interval selected for inclusion in a highlight reel."""

    event_type: str
    source_start: float
    source_end: float
    title: str
    importance: float
    reasons: list[str] = field(default_factory=list)
    play_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, str) or not self.event_type.strip():
            raise ValidationError("event_type must be a non-empty string")
        self.source_start = _nonnegative(self.source_start, "source_start")
        self.source_end = _nonnegative(self.source_end, "source_end")
        if self.source_end <= self.source_start:
            raise ValidationError("source_end must be greater than source_start")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValidationError("title must be a non-empty string")
        self.importance = _number(self.importance, "importance")
        if not 0 <= self.importance <= 1:
            raise ValidationError("importance must be between 0 and 1")
        if isinstance(self.reasons, str) or not isinstance(self.reasons, list):
            raise ValidationError("reasons must be a list of strings")
        if any(not isinstance(reason, str) for reason in self.reasons):
            raise ValidationError("reasons must be a list of strings")
        if self.play_id is not None and not isinstance(self.play_id, str):
            raise ValidationError("play_id must be a string or null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HighlightSegment:
        payload = dict(data)
        # Accept manifests produced by older tracked-crop versions while
        # intentionally discarding their obsolete rendering instructions.
        payload.pop("layout", None)
        payload.pop("focal_x_anchors", None)
        return cls(**payload)


@dataclass
class HighlightManifest:
    """Serializable analysis result and edit decision list."""

    media_info: MediaInfo
    segments: list[HighlightSegment] = field(default_factory=list)
    game_id: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.media_info, MediaInfo):
            raise ValidationError("media_info must be a MediaInfo instance")
        if not isinstance(self.segments, list) or any(
            not isinstance(segment, HighlightSegment) for segment in self.segments
        ):
            raise ValidationError("segments must be a list of HighlightSegment instances")
        if self.game_id is not None and not isinstance(self.game_id, str):
            raise ValidationError("game_id must be a string or null")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValidationError("version must be a positive integer")
        self.validate()

    def validate(self) -> None:
        """Validate the manifest and all of its segments."""
        self.media_info.__post_init__()
        for segment in self.segments:
            segment.__post_init__()
            if segment.source_end > self.media_info.duration_seconds:
                raise ValidationError("segment source_end cannot exceed media duration_seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "game_id": self.game_id,
            "media_info": self.media_info.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
        }

    def to_json(self, *, indent: int = 2) -> str:
        self.validate()
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HighlightManifest:
        payload = dict(data)
        media_info = payload.pop("media_info")
        segments = payload.pop("segments", [])
        return cls(
            media_info=media_info
            if isinstance(media_info, MediaInfo)
            else MediaInfo.from_dict(media_info),
            segments=[
                segment
                if isinstance(segment, HighlightSegment)
                else HighlightSegment.from_dict(segment)
                for segment in segments
            ],
            **payload,
        )

    @classmethod
    def from_json(cls, text: str) -> HighlightManifest:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid manifest JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValidationError("manifest JSON must contain an object")
        return cls.from_dict(data)

    def save(self, path: str | Path, *, indent: int = 2) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> HighlightManifest:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def save_manifest(manifest: HighlightManifest, path: str | Path, *, indent: int = 2) -> None:
    """Save a manifest as UTF-8 JSON."""
    manifest.save(path, indent=indent)


def load_manifest(path: str | Path) -> HighlightManifest:
    """Load and validate a manifest from UTF-8 JSON."""
    return HighlightManifest.load(path)
