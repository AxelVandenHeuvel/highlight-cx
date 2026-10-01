import json

import pytest

from football_highlights.models import (
    HighlightManifest,
    HighlightSegment,
    MediaInfo,
    ValidationError,
    load_manifest,
    save_manifest,
)


def make_manifest() -> HighlightManifest:
    return HighlightManifest(
        game_id="GB@NYJ",
        media_info=MediaInfo(
            source_path="/tmp/GB@NYJ.mkv",
            duration_seconds=7200,
            width=1920,
            height=1080,
            fps=59.94,
        ),
        segments=[
            HighlightSegment(
                event_type="touchdown",
                source_start=4182.4,
                source_end=4207.8,
                title="Fourth-quarter touchdown",
                importance=0.93,
                reasons=["score change", "late game"],
                play_id="123",
            )
        ],
    )


def test_segment_validates_required_highlight_fields() -> None:
    segment = make_manifest().segments[0]
    assert segment.to_dict() == {
        "event_type": "touchdown",
        "source_start": 4182.4,
        "source_end": 4207.8,
        "title": "Fourth-quarter touchdown",
        "importance": 0.93,
        "reasons": ["score change", "late game"],
        "play_id": "123",
    }


@pytest.mark.parametrize(
    ("start", "end", "importance"),
    [(-1, 2, 0.5), (2, -1, 0.5), (3, 2, 0.5), (1, 2, -0.01), (1, 2, 1.01)],
)
def test_segment_rejects_invalid_times_and_importance(start, end, importance) -> None:
    with pytest.raises(ValidationError):
        HighlightSegment("play", start, end, "Play", importance)


def test_manifest_round_trips_through_json(tmp_path) -> None:
    original = make_manifest()
    path = tmp_path / "manifest.json"

    save_manifest(original, path)
    loaded = load_manifest(path)

    assert loaded == original
    assert json.loads(path.read_text()) == original.to_dict()


def test_manifest_rejects_segment_outside_media_duration() -> None:
    media = MediaInfo("game.mkv", 10, 1920, 1080, 30)
    segment = HighlightSegment("field_goal", 8, 11, "Field goal", 0.7)

    with pytest.raises(ValidationError, match="duration"):
        HighlightManifest(media, [segment])


def test_optional_play_id_and_default_values() -> None:
    segment = HighlightSegment("interception", 10, 20, "Interception", 1)
    assert segment.play_id is None
    assert segment.reasons == []


def test_old_tracking_fields_are_ignored_when_loading() -> None:
    segment = HighlightSegment.from_dict(
        {
            "event_type": "interception",
            "source_start": 10,
            "source_end": 20,
            "title": "Interception",
            "importance": 1,
            "layout": "tracked",
            "focal_x_anchors": {"1": 0.5},
        }
    )

    assert segment.event_type == "interception"
    assert "layout" not in segment.to_dict()
