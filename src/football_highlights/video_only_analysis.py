"""Discover football highlights using only signals contained in the video."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from football_highlights.audio_events import AudioEventConfig, detect_audio_events_from_video
from football_highlights.highlight_ranking import rank_non_scoring_highlights
from football_highlights.live_play import (
    LivePlayConfig,
    sample_video_around_candidate,
    validate_live_play,
)
from football_highlights.local_ocr import RapidOcrBackend
from football_highlights.media import probe_video
from football_highlights.models import HighlightManifest, HighlightSegment, MediaInfo
from football_highlights.transcript_events import (
    FasterWhisperTranscriptAdapter,
    TranscriptEvent,
    TranscriptSegment,
    detect_transcript_events,
)

_IMPORTANCE = {
    "touchdown": 1.0,
    "interception": 0.97,
    "fumble": 0.94,
    "blocked_kick": 0.94,
    "field_goal": 0.86,
    "fourth_down": 0.84,
    "first_down": 0.68,
    "big_play": 0.78,
}

_NFL_TEAM_NAMES = {
    "49ers",
    "bears",
    "bengals",
    "bills",
    "broncos",
    "browns",
    "buccaneers",
    "cardinals",
    "chargers",
    "chiefs",
    "colts",
    "commanders",
    "cowboys",
    "dolphins",
    "eagles",
    "falcons",
    "giants",
    "jaguars",
    "jets",
    "lions",
    "packers",
    "panthers",
    "patriots",
    "raiders",
    "rams",
    "ravens",
    "saints",
    "seahawks",
    "steelers",
    "texans",
    "titans",
    "vikings",
}


def _merge_windows(points: list[float], duration: float) -> list[tuple[float, float]]:
    windows = sorted((max(0.0, point - 16.0), min(duration, point + 18.0)) for point in points)
    merged: list[tuple[float, float]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1] + 2.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _extract_window(source: Path, destination: Path, start: float, end: float) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{end - start:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)


def _transcribe_windows(
    source: Path,
    windows: list[tuple[float, float]],
    transcriber: FasterWhisperTranscriptAdapter,
    progress: Callable[[str], None] | None = None,
) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    with tempfile.TemporaryDirectory(prefix="nfl-transcript-") as temporary:
        clip = Path(temporary) / "window.wav"
        for index, (start, end) in enumerate(windows, start=1):
            _extract_window(source, clip, start, end)
            for segment in transcriber.transcribe(clip):
                segments.append(
                    TranscriptSegment(
                        start=start + segment.start,
                        end=start + segment.end,
                        text=segment.text,
                        confidence=segment.confidence,
                    )
                )
            if progress:
                progress(f"transcribed candidate window {index}/{len(windows)}")
    return segments


def _dedupe_semantic_events(events: list[TranscriptEvent]) -> list[TranscriptEvent]:
    """Suppress replay recaps and repeated commentary about the same result."""

    accepted: list[TranscriptEvent] = []
    for event in sorted(events, key=lambda item: item.start):
        if event.is_replay_reference:
            continue
        if event.event_type == "touchdown" and "touchdown" not in event.reasons:
            # Goal-line discussion alone is common before a snap and is not
            # sufficient evidence that the play scored.
            continue
        # Window/replay duplicates are already resolved by transcript_events.
        # Do not collapse distinct plays merely because they share a type and
        # occur within 90 seconds (common for first downs and chunk gains).
        accepted.append(event)
    return accepted


def _is_crucial_first_down(event: TranscriptEvent) -> bool:
    """Keep high-leverage first downs rather than every chain movement."""

    return bool(
        re.search(
            r"\b(?:third|3rd|fourth|4th)\s+(?:and|down)|keep(?:s|ing)? the drive alive|"
            r"move(?:s|d)? the chains|crucial|critical|huge|big first down|needed\b",
            event.text,
            re.IGNORECASE,
        )
    )


def _infer_game_teams(transcript: list[TranscriptSegment]) -> set[str]:
    counts = {
        team: sum(segment.text.lower().count(team) for segment in transcript)
        for team in _NFL_TEAM_NAMES
    }
    return {team for team, count in sorted(counts.items(), key=lambda item: -item[1])[:2] if count}


def _mentions_other_game(event: TranscriptEvent, game_teams: set[str]) -> bool:
    mentioned = {team for team in _NFL_TEAM_NAMES if team in event.text.lower()}
    return bool(mentioned - game_teams)


def _is_requested_event(event: TranscriptEvent, scoring_plays_only: bool) -> bool:
    """Keep made-score candidates by default, or every supported event on request."""

    if not scoring_plays_only:
        return True
    return event.event_type in {"touchdown", "field_goal"}


def _score_matches_event(
    event_type: str,
    score_delta: tuple[int, int] | None,
) -> bool:
    """Require the scoreboard change expected for explicit scoring candidates."""

    if event_type == "touchdown":
        return score_delta is not None and max(score_delta) >= 6
    if event_type == "field_goal":
        return score_delta is not None and max(score_delta) == 3
    return True


def _score_change_after_play(
    source: Path,
    live_start: float,
    live_end: float,
    ocr: RapidOcrBackend,
) -> tuple[int, int] | None:
    """Return the observed left/right score delta around a verified live play."""

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return None
    before: list[tuple[int, int]] = []
    after: list[tuple[int, int]] = []
    try:
        # Broadcast graphics are frequently hidden by lower thirds, cuts, and
        # celebration shots.  Scan a modest interval on both sides instead of
        # trusting a few exact frames; the modal score filters transient OCR.
        points = [max(0.0, live_start - offset) for offset in range(20, 1, -2)]
        points.extend(live_end + offset for offset in range(3, 43, 4))
        for timestamp in points:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            crop = frame[round(height * 0.75) : height, round(width * 0.25) : round(width * 0.75)]
            score = ocr.read_score(crop)
            if score is None or score.confidence < 0.45:
                continue
            pair = (score.left, score.right)
            (before if timestamp < live_start else after).append(pair)
    finally:
        capture.release()
    if not before or not after:
        return None
    baseline = max(set(before), key=before.count)
    final = max(after, key=lambda pair: (pair[0] + pair[1], after.count(pair)))
    delta = (final[0] - baseline[0], final[1] - baseline[1])
    if min(delta) < 0 or max(delta) > 8 or delta == (0, 0):
        return None
    return delta


def _goalpost_visible(frame: np.ndarray) -> bool:
    """Detect the two tall yellow uprights in a field-goal broadcast angle."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array((18, 120, 120)), np.array((40, 255, 255)))
    _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(yellow)
    height, width = frame.shape[:2]
    uprights = [
        (int(x), int(y), int(component_width), int(component_height))
        for x, y, component_width, component_height, area in stats[1:]
        if component_height >= height * 0.38
        and component_height / max(1, component_width) >= 6.0
        and area >= height * width * 0.0015
    ]
    if len(uprights) < 2:
        return False
    centers = sorted(x + component_width / 2 for x, _y, component_width, _h in uprights)
    adjacent = zip(centers, centers[1:], strict=False)
    return any(width * 0.12 <= right - left <= width * 0.55 for left, right in adjacent)


def _find_goalpost_window(
    source: Path,
    candidate_time: float,
    duration: float,
) -> tuple[float, float] | None:
    """Locate the actual kick around score/commentary evidence."""

    start = max(0.0, candidate_time - 55.0)
    end = min(duration, candidate_time + 30.0)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return None
    hits: list[float] = []
    try:
        timestamp = start
        while timestamp <= end:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if ok and _goalpost_visible(frame):
                hits.append(timestamp)
            timestamp += 0.5
    finally:
        capture.release()
    if not hits:
        return None
    runs: list[list[float]] = []
    for timestamp in hits:
        if runs and timestamp - runs[-1][-1] <= 1.25:
            runs[-1].append(timestamp)
        else:
            runs.append([timestamp])
    run = min(runs, key=lambda values: abs(sum(values) / len(values) - candidate_time))
    if len(run) < 2:
        return None
    clip_start = max(0.0, run[0] - 10.0)
    clip_end = min(duration, run[-1] + 3.0, clip_start + 26.0)
    return clip_start, clip_end


def _scan_goalpost_windows(
    source: Path,
    duration: float,
    *,
    interval_seconds: float = 2.0,
) -> list[tuple[float, float]]:
    """Find consecutive broadcast shots that visibly contain both uprights."""

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return []
    hits: list[float] = []
    try:
        timestamp = 0.0
        while timestamp <= duration:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if ok and _goalpost_visible(frame):
                hits.append(timestamp)
            timestamp += interval_seconds
    finally:
        capture.release()
    runs: list[list[float]] = []
    for timestamp in hits:
        if runs and timestamp - runs[-1][-1] <= interval_seconds * 2.0:
            runs[-1].append(timestamp)
        else:
            runs.append([timestamp])
    return [(run[0], run[-1]) for run in runs]


def analyze_video_only(
    source: Path,
    *,
    whisper_model: str = "tiny.en",
    max_audio_candidates: int = 80,
    max_highlights: int = 24,
    scoring_plays_only: bool = False,
    transcript_cache: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> HighlightManifest:
    """Create an automatic manifest without PBP, URLs, or authored timestamps."""

    metadata = probe_video(source)
    if progress:
        progress("scanning complete game audio")
    audio_events = detect_audio_events_from_video(
        source,
        config=AudioEventConfig(activation_threshold=0.48),
    )
    strongest = sorted(audio_events, key=lambda item: item.confidence, reverse=True)[
        :max_audio_candidates
    ]
    windows = _merge_windows([item.peak_time for item in strongest], metadata.duration)
    if progress:
        progress(f"found {len(audio_events)} audio peaks; transcribing {len(windows)} windows")

    if transcript_cache is not None and transcript_cache.exists():
        transcript = [
            TranscriptSegment(**item)
            for item in json.loads(transcript_cache.read_text(encoding="utf-8"))
        ]
        if progress:
            progress(f"loaded {len(transcript)} cached transcript segments")
    else:
        transcriber = FasterWhisperTranscriptAdapter(model_size=whisper_model)
        transcript = _transcribe_windows(source, windows, transcriber, progress)
        if transcript_cache is not None:
            transcript_cache.parent.mkdir(parents=True, exist_ok=True)
            transcript_cache.write_text(
                json.dumps(
                    [
                        {
                            "start": item.start,
                            "end": item.end,
                            "text": item.text,
                            "confidence": item.confidence,
                        }
                        for item in transcript
                    ],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    game_teams = _infer_game_teams(transcript)
    events = _dedupe_semantic_events(detect_transcript_events(transcript))
    events = [
        event
        for event in events
        if not _mentions_other_game(event, game_teams)
        and (
            event.event_type != "big_play"
            or (event.yards is not None and 20 <= event.yards <= 109)
        )
        and _is_requested_event(event, scoring_plays_only)
    ]
    field_goal_events = [
        event for event in events if event.event_type == "field_goal" and event.outcome != "missed"
    ]
    non_scoring_types = {"interception", "fumble", "big_play", "first_down", "fourth_down"}
    scoring_candidates = [
        event
        for event in events
        if event.event_type != "field_goal" and event.event_type not in non_scoring_types
    ]
    non_scoring_events = [
        event
        for event in events
        if event.event_type in non_scoring_types
        and (event.event_type != "first_down" or _is_crucial_first_down(event))
    ]
    ranked_non_scoring = rank_non_scoring_highlights(non_scoring_events)
    ranking_score = {
        (item.candidate.event_type, item.candidate.start, item.candidate.end): item.score
        for item in ranked_non_scoring
    }
    ranked = sorted(
        [*scoring_candidates, *non_scoring_events],
        key=lambda item: ranking_score.get(
            (item.event_type, item.start, item.end),
            _IMPORTANCE.get(item.event_type, 0.5) * item.confidence,
        ),
        reverse=True,
    )

    segments: list[HighlightSegment] = []
    ocr = RapidOcrBackend()
    for index, event in enumerate(ranked, start=1):
        is_non_scoring = event.event_type in non_scoring_types
        peak_radius = 35.0 if is_non_scoring else 120.0
        nearby = [
            item for item in audio_events if abs(item.peak_time - event.end) <= peak_radius
        ]
        strongest_peaks = sorted(
            nearby,
            key=lambda item: (-item.confidence, abs(item.peak_time - event.end)),
        )[:3]
        closest_peaks = sorted(
            nearby,
            key=lambda item: (abs(item.peak_time - event.end), -item.confidence),
        )[:6]
        search_points = [event.end]
        # A 1 fps visual sample can otherwise land entirely on cuts at one
        # phase of the timeline.  Probe just before each reaction peak too.
        for item in closest_peaks:
            search_points.extend((max(0.0, int(item.peak_time) - 2.0), item.peak_time))
        search_points.extend(item.peak_time for item in strongest_peaks)
        search_points = list(dict.fromkeys(round(point, 2) for point in search_points))

        accepted = None
        score_delta = None
        for point in search_points:
            sampled = sample_video_around_candidate(
                source,
                point,
                window_before=24.0 if is_non_scoring else 18.0,
                window_after=8.0 if is_non_scoring else 12.0,
                sample_fps=1.0,
                scorebug_ocr=ocr.read_scorebug,
            )
            live = validate_live_play(
                sampled.frames,
                sampled.scorebug_readings,
                candidate_timestamp=point,
                config=LivePlayConfig(
                    max_live_seconds=32.0,
                    min_scorebug_coverage=0.60,
                    max_missing_scorebug_seconds=8.0,
                ),
            )
            if not live.accepted or live.live_start is None or live.live_end is None:
                continue
            if is_non_scoring and (
                live.live_end < event.start - 30.0 or live.live_start > event.end + 10.0
            ):
                continue
            if any(abs(segment.source_start - live.live_start) < 20.0 for segment in segments):
                continue
            candidate_delta = _score_change_after_play(source, live.live_start, live.live_end, ocr)
            if not _score_matches_event(event.event_type, candidate_delta):
                continue
            accepted = live
            score_delta = candidate_delta
            break
        if accepted is None:
            if progress:
                progress(f"rejected non-live candidate {index}/{len(ranked)}: {event.event_type}")
            continue
        live = accepted
        event_type = event.event_type
        if score_delta is not None and max(score_delta) >= 6:
            event_type = "touchdown"
        elif score_delta is not None and max(score_delta) == 3:
            event_type = "field_goal"
        segments.append(
            HighlightSegment(
                event_type=event_type,
                source_start=max(0.0, live.live_start),
                source_end=min(metadata.duration, live.live_end + 0.75),
                title=(
                    event.text
                    if event_type == event.event_type
                    else f"{event_type.replace('_', ' ').title()} detected from live video"
                ),
                importance=min(
                    1.0,
                    ranking_score.get(
                        (event.event_type, event.start, event.end),
                        _IMPORTANCE.get(event.event_type, 0.5) * event.confidence,
                    ),
                ),
                reasons=[
                    "local transcript detection",
                    *event.reasons,
                    "strict live-play validation passed",
                    *([f"score changed by {score_delta}"] if score_delta else []),
                    *live.reasons,
                ],
            )
        )
        if progress:
            progress(f"accepted live highlight {len(segments)}/{max_highlights}")
        if len(segments) >= max_highlights:
            break

    if len(segments) < max_highlights:
        if progress:
            progress("scanning complete game for field-goal uprights")
        goalpost_windows = _scan_goalpost_windows(source, metadata.duration)
        for goalpost_start, goalpost_end in goalpost_windows:
            score_delta = _score_change_after_play(
                source,
                max(0.0, goalpost_start - 2.0),
                goalpost_end + 2.0,
                ocr,
            )
            if not _score_matches_event("field_goal", score_delta):
                continue
            clip_start = max(0.0, goalpost_start - 10.0)
            clip_end = min(metadata.duration, goalpost_end + 3.0, clip_start + 26.0)
            segments.append(
                HighlightSegment(
                    event_type="field_goal",
                    source_start=clip_start,
                    source_end=clip_end,
                    title="Field goal detected from video",
                    importance=_IMPORTANCE["field_goal"],
                    reasons=[
                        "goalpost sequence detected in live broadcast",
                        f"score changed by {score_delta}",
                    ],
                )
            )
            if len(segments) >= max_highlights:
                break

    # In overtime the broadcast can end before the final scorebug update is
    # shown.  Local commentary plus the actual upright camera sequence provides
    # an independent fallback for those game-ending kicks.
    for event in field_goal_events:
        if len(segments) >= max_highlights:
            break
        window = _find_goalpost_window(source, event.end, metadata.duration)
        if window is None:
            continue
        clip_start, clip_end = window
        if any(abs(segment.source_start - clip_start) < 20.0 for segment in segments):
            continue
        segments.append(
            HighlightSegment(
                event_type="field_goal",
                source_start=clip_start,
                source_end=clip_end,
                title=event.text,
                importance=min(1.0, _IMPORTANCE["field_goal"] * event.confidence),
                reasons=[
                    "local field-goal commentary",
                    *event.reasons,
                    "goalpost sequence detected in live broadcast",
                ],
            )
        )

    segments.sort(key=lambda item: item.source_start)
    return HighlightManifest(
        game_id=source.stem,
        media_info=MediaInfo(
            source_path=str(source.resolve()),
            duration_seconds=metadata.duration,
            width=metadata.width,
            height=metadata.height,
            fps=metadata.frame_rate,
        ),
        segments=segments,
    )
