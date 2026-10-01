"""Local FFprobe/FFmpeg helpers for rendering football highlight clips."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MediaError(RuntimeError):
    """Raised when local media inspection or rendering fails."""


class MediaValidationError(ValueError):
    """Raised when a probed input does not contain usable media."""


@dataclass(frozen=True)
class VideoMetadata:
    """The minimum metadata required by the rendering pipeline."""

    duration: float
    width: int
    height: int
    frame_rate: float
    has_audio: bool
    video_codec: str | None = None


def _frame_rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def _metadata_from_probe(payload: dict[str, Any]) -> VideoMetadata:
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise MediaValidationError("ffprobe output contains no video stream")

    format_data = payload.get("format") or {}
    try:
        duration = float(format_data.get("duration", video.get("duration", 0)))
        width = int(video.get("width", 0))
        height = int(video.get("height", 0))
        frame_rate = _frame_rate(video.get("r_frame_rate") or video.get("avg_frame_rate"))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise MediaValidationError("ffprobe output contains invalid video metadata") from exc

    return VideoMetadata(
        duration=duration,
        width=width,
        height=height,
        frame_rate=frame_rate,
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
        video_codec=video.get("codec_name"),
    )


def probe_video(
    input_path: str | Path,
    *,
    ffprobe_bin: str = "ffprobe",
    runner: Any = subprocess.run,
) -> VideoMetadata:
    """Probe a local video using ffprobe and return normalized metadata."""

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]
    try:
        completed = runner(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MediaError(f"ffprobe failed for {input_path}") from exc
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise MediaError("ffprobe returned invalid JSON") from exc
    return validate_metadata(_metadata_from_probe(payload))


def validate_metadata(metadata: VideoMetadata, *, require_audio: bool = False) -> VideoMetadata:
    """Validate metadata needed for rendering and return it for convenient chaining."""

    errors: list[str] = []
    if metadata.duration <= 0:
        errors.append("duration must be greater than zero")
    if metadata.width <= 0 or metadata.height <= 0:
        errors.append("video dimensions must be greater than zero")
    if metadata.frame_rate <= 0:
        errors.append("frame rate must be greater than zero")
    if require_audio and not metadata.has_audio:
        errors.append("an audio stream is required")
    if errors:
        raise MediaValidationError("; ".join(errors))
    return metadata


def build_render_command(
    input_path: str | Path,
    output_path: str | Path,
    start: float,
    end: float,
    *,
    use_videotoolbox: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    audio_bitrate: str = "192k",
) -> list[str]:
    """Build an argv list for one 9:16 interval with a blurred background."""

    if start < 0 or end <= start:
        raise ValueError("end must be greater than start, and start cannot be negative")

    duration = end - start
    video_encoder = "h264_videotoolbox" if use_videotoolbox else "libx264"
    encoder_options = (
        ["-b:v", "8M", "-allow_sw", "1"]
        if use_videotoolbox
        else ["-preset", "medium", "-crf", "18"]
    )
    filter_graph = (
        "[0:v]split=2[bgsrc][fgsrc];"
        "[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=40:20,eq=brightness=-0.08:saturation=1.15[bg];"
        "[fgsrc]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(input_path),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-map",
        "0:a:0?",
        "-c:v",
        video_encoder,
        *encoder_options,
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]


def render_interval(
    input_path: str | Path,
    output_path: str | Path,
    start: float,
    end: float,
    *,
    use_videotoolbox: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    runner: Any = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Render an interval locally, falling back to libx264 when requested."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    command = build_render_command(
        input_path,
        output_path,
        start,
        end,
        use_videotoolbox=use_videotoolbox,
        ffmpeg_bin=ffmpeg_bin,
    )
    try:
        return runner(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as first_error:
        if not use_videotoolbox:
            raise MediaError("ffmpeg render failed") from first_error
        fallback = build_render_command(
            input_path,
            output_path,
            start,
            end,
            use_videotoolbox=False,
            ffmpeg_bin=ffmpeg_bin,
        )
        try:
            return runner(fallback, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as fallback_error:
            raise MediaError(
                "ffmpeg render failed with VideoToolbox and libx264"
            ) from fallback_error
    except OSError as exc:
        raise MediaError("ffmpeg executable could not be started") from exc


__all__ = [
    "MediaError",
    "MediaValidationError",
    "VideoMetadata",
    "build_render_command",
    "probe_video",
    "render_interval",
    "validate_metadata",
]
