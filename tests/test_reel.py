from pathlib import Path

from football_highlights import reel
from football_highlights.models import HighlightManifest, HighlightSegment, MediaInfo
from football_highlights.reel import concat_clips


def test_concat_clips_uses_concat_demuxer(tmp_path: Path) -> None:
    clips = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))

    output = tmp_path / "reel.mp4"
    concat_clips(clips, output, runner=runner)

    command = calls[0][0]
    assert command[command.index("-f") + 1] == "concat"
    assert command[command.index("-c") + 1] == "copy"
    assert (tmp_path / "concat.txt").read_text().count("file '") == 2


def test_render_manifest_uses_full_frame_blurred_renderer(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        reel,
        "render_interval",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(reel, "concat_clips", lambda _clips, _output: None)
    manifest = HighlightManifest(
        media_info=MediaInfo("game.mkv", 100, 1920, 1080, 30),
        segments=[HighlightSegment("touchdown", 40, 42, "Touchdown", 1)],
    )

    clips, output = reel.render_manifest(tmp_path / "game.mkv", manifest, tmp_path / "output")

    assert clips == [tmp_path / "output/clips/01-touchdown.mp4"]
    assert output == tmp_path / "output/reel.mp4"
    assert calls[0][0][2:4] == (40, 42)
