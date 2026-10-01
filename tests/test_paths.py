from pathlib import Path

import pytest

from football_highlights.paths import ensure_input_video, job_directory, job_id_for


def test_job_id_is_stable_for_unchanged_file(tmp_path: Path) -> None:
    video = tmp_path / "game.mkv"
    video.write_bytes(b"test-video")
    assert job_id_for(video) == job_id_for(video)


def test_job_directory_uses_video_stem(tmp_path: Path) -> None:
    video = tmp_path / "GB@NYJ.mkv"
    video.write_bytes(b"test-video")
    output = job_directory(tmp_path / "output", video)
    assert output.is_dir()
    assert output.name.startswith("GB@NYJ-")


def test_missing_video_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ensure_input_video(tmp_path / "missing.mkv")
