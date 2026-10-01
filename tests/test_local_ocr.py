from football_highlights.local_ocr import parse_scorebug_tokens


def test_parse_scorebug_tokens() -> None:
    result = parse_scorebug_tokens([("14:52", 0.98), ("1ST", 0.82), ("1ST & 10", 0.93)])
    assert result is not None
    assert result.quarter == 1
    assert result.clock_seconds == 14 * 60 + 52
    assert result.confidence == 0.82


def test_parse_scorebug_rejects_partial_reading() -> None:
    assert parse_scorebug_tokens([("14:52", 0.98)]) is None
