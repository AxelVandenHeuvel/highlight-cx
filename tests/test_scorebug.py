import subprocess
from pathlib import Path

import pytest

from football_highlights.media import VideoMetadata
from football_highlights.scorebug import ScorebugRegion, extract_scorebug_sample


def test_region_converts_to_even_pixel_coordinates() -> None:
    metadata = VideoMetadata(100, 1920, 1080, 60, True)
    region = ScorebugRegion(0.25, 0.75, 0.50, 0.25)
    assert region.pixels(metadata) == (480, 810, 960, 270)


def test_region_rejects_out_of_bounds_values() -> None:
    with pytest.raises(ValueError):
        ScorebugRegion(0.8, 0.8, 0.4, 0.4)


def test_extract_scorebug_builds_crop_command(tmp_path: Path) -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    metadata = VideoMetadata(100, 1920, 1080, 60, True)
    output = tmp_path / "sample.jpg"
    extract_scorebug_sample(Path("game.mkv"), output, 12.5, metadata, runner=runner)

    command = calls[0][0]
    assert command[command.index("-ss") + 1] == "12.500"
    assert command[command.index("-vf") + 1] == "crop=960:270:480:810"


def test_extract_scorebug_rejects_timestamp_after_video() -> None:
    metadata = VideoMetadata(100, 1920, 1080, 60, True)
    with pytest.raises(ValueError):
        extract_scorebug_sample(Path("game.mkv"), Path("sample.jpg"), 101, metadata)
