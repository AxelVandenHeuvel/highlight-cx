import json
import subprocess
from pathlib import Path

import pytest

from football_highlights.media import (
    MediaError,
    MediaValidationError,
    VideoMetadata,
    build_render_command,
    probe_video,
    render_interval,
    validate_metadata,
)


def test_probe_video_uses_argv_and_parses_metadata() -> None:
    calls = []
    payload = {
        "format": {"duration": "3600.5"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 3840,
                "height": 2160,
                "r_frame_rate": "60000/1001",
            },
            {"codec_type": "audio"},
        ],
    }

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    metadata = probe_video(Path("game.mkv"), ffprobe_bin="local-ffprobe", runner=runner)

    assert metadata == VideoMetadata(3600.5, 3840, 2160, pytest.approx(60000 / 1001), True, "hevc")
    assert calls[0][0] == [
        "local-ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "game.mkv",
    ]
    assert calls[0][1] == {"check": True, "capture_output": True, "text": True}


def test_probe_video_rejects_missing_video_stream() -> None:
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"streams": []}), stderr=""
        )

    with pytest.raises(MediaValidationError, match="no video stream"):
        probe_video("game.mkv", runner=runner)


def test_validate_metadata_can_require_audio() -> None:
    metadata = VideoMetadata(10, 1920, 1080, 30, False)
    with pytest.raises(MediaValidationError, match="audio stream"):
        validate_metadata(metadata, require_audio=True)


def test_build_render_command_uses_blurred_background_and_videotoolbox() -> None:
    command = build_render_command("game.mkv", "clip.mp4", 12.5, 27.75)

    assert isinstance(command, list)
    assert command[0] == "ffmpeg"
    assert command[command.index("-ss") + 1] == "12.500"
    assert command[command.index("-t") + 1] == "15.250"
    assert command[command.index("-c:v") + 1] == "h264_videotoolbox"
    assert "boxblur=40:20" in command[command.index("-filter_complex") + 1]
    assert "scale=1080:1920" in command[command.index("-filter_complex") + 1]
    assert command[command.index("-map") + 3] == "0:a:0?"


def test_render_interval_falls_back_to_libx264() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command, stderr="VideoToolbox unavailable")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = render_interval("game.mkv", "clip.mp4", 1, 3, runner=runner)

    assert result.returncode == 0
    assert len(calls) == 2
    assert "h264_videotoolbox" in calls[0]
    assert "libx264" in calls[1]
    assert all(isinstance(item, str) for command in calls for item in command)


def test_render_interval_wraps_ffmpeg_failure_without_fallback() -> None:
    def runner(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(MediaError, match="ffmpeg render failed"):
        render_interval("game.mkv", "clip.mp4", 1, 3, use_videotoolbox=False, runner=runner)
