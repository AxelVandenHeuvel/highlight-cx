"""Video-only visual event discovery for football broadcasts.

This module deliberately has no play-by-play dependency.  It combines cheap
signals available in a broadcast video with an injectable local scorebug OCR
callback.  The pure helpers are useful in tests and for replacing the OpenCV
sampler with another local decoder later.

The sampler emits observations; :func:`detect_visual_events` turns those
observations into timestamped candidates.  The OCR callback receives a cropped
scorebug image and its source timestamp and may return a ``ScoreReading``, a
mapping with ``home_score``/``away_score`` keys, or ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # OpenCV is a normal project dependency, but keep pure helpers importable.
    import cv2
except ImportError:  # pragma: no cover - only relevant to unusual minimal installs
    cv2 = None  # type: ignore[assignment]


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class ScoreReading:
    """A scorebug reading produced by a local OCR backend."""

    timestamp: float
    home_score: int
    away_score: int
    confidence: float = 1.0
    quarter: int | None = None
    clock_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if self.home_score < 0 or self.away_score < 0:
            raise ValueError("scores must be non-negative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.quarter is not None and self.quarter < 1:
            raise ValueError("quarter must be positive")


@dataclass(frozen=True)
class VisualObservation:
    """Signals calculated for one sampled video frame."""

    timestamp: float
    motion_score: float = 0.0
    scene_change_score: float = 0.0
    field_action_score: float = 0.0
    celebration_score: float = 0.0
    green_field_ratio: float = 0.0
    scorebug_visible: bool = True

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        for name in (
            "motion_score",
            "scene_change_score",
            "field_action_score",
            "celebration_score",
            "green_field_ratio",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class EventCandidate:
    """A timestamped event hypothesis produced from the video."""

    timestamp: float
    kind: str
    confidence: float
    source_start: float
    source_end: float
    reasons: tuple[str, ...] = ()
    score_delta: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.timestamp < 0 or self.source_start < 0:
            raise ValueError("candidate timestamps must be non-negative")
        if self.source_end <= self.source_start:
            raise ValueError("source_end must be after source_start")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


# More descriptive aliases for callers that prefer explicit names.
ScoreChange = EventCandidate
VisualEventCandidate = EventCandidate


@dataclass(frozen=True)
class VisualEventConfig:
    """Thresholds for deterministic candidate generation."""

    min_ocr_confidence: float = 0.55
    score_change_max_jump: int = 8
    score_change_pre_roll: float = 5.0
    score_change_post_roll: float = 4.0
    field_action_threshold: float = 0.46
    field_action_min_seconds: float = 1.5
    field_action_max_seconds: float = 14.0
    field_action_gap_seconds: float = 1.25
    scene_change_threshold: float = 0.42
    celebration_threshold: float = 0.52
    replay_post_seconds: float = 2.0
    merge_window_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_ocr_confidence <= 1:
            raise ValueError("min_ocr_confidence must be between 0 and 1")
        for name in (
            "score_change_pre_roll",
            "score_change_post_roll",
            "field_action_min_seconds",
            "field_action_max_seconds",
            "field_action_gap_seconds",
            "replay_post_seconds",
            "merge_window_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.field_action_max_seconds < self.field_action_min_seconds:
            raise ValueError("field_action_max_seconds must cover min duration")


@dataclass(frozen=True)
class VideoSampleResult:
    """Signals and OCR readings emitted by :func:`sample_video`."""

    observations: tuple[VisualObservation, ...]
    score_readings: tuple[ScoreReading, ...]
    fps: float
    width: int
    height: int
    duration: float


OCRCallback = Callable[[np.ndarray, float], ScoreReading | Mapping[str, Any] | None]


def _coerce_score_reading(
    value: ScoreReading | Mapping[str, Any] | Sequence[Any] | None,
    timestamp: float,
) -> ScoreReading | None:
    """Normalize common OCR callback return shapes without tying to an OCR engine."""

    if value is None:
        return None
    if isinstance(value, ScoreReading):
        if math.isclose(value.timestamp, timestamp):
            return value
        return ScoreReading(
            timestamp,
            value.home_score,
            value.away_score,
            value.confidence,
            value.quarter,
            value.clock_seconds,
        )
    if isinstance(value, Mapping):
        home = value.get("home_score", value.get("home"))
        away = value.get("away_score", value.get("away"))
        if home is None or away is None:
            return None
        try:
            return ScoreReading(
                timestamp=timestamp,
                home_score=int(home),
                away_score=int(away),
                confidence=float(value.get("confidence", 1.0)),
                quarter=(int(value["quarter"]) if value.get("quarter") is not None else None),
                clock_seconds=(
                    int(value["clock_seconds"]) if value.get("clock_seconds") is not None else None
                ),
            )
        except (TypeError, ValueError):
            return None
    if len(value) >= 2:
        try:
            confidence = float(value[2]) if len(value) >= 3 else 1.0
            return ScoreReading(timestamp, int(value[0]), int(value[1]), confidence)
        except (TypeError, ValueError):
            return None
    return None


def _accepted_scores(
    readings: Iterable[ScoreReading],
    config: VisualEventConfig,
) -> list[ScoreReading]:
    """Sort, filter, and de-duplicate OCR observations deterministically."""

    accepted: list[ScoreReading] = []
    for reading in sorted(readings, key=lambda item: item.timestamp):
        if reading.confidence < config.min_ocr_confidence:
            continue
        if accepted and (
            math.isclose(reading.timestamp, accepted[-1].timestamp, abs_tol=1e-6)
            or (reading.home_score, reading.away_score)
            == (accepted[-1].home_score, accepted[-1].away_score)
            and reading.timestamp - accepted[-1].timestamp < 0.5
        ):
            if reading.confidence > accepted[-1].confidence:
                accepted[-1] = reading
            continue
        accepted.append(reading)
    return accepted


def detect_score_changes(
    readings: Iterable[ScoreReading],
    *,
    config: VisualEventConfig | None = None,
) -> list[EventCandidate]:
    """Find stable scoreboard changes, rejecting likely OCR glitches."""

    active = config or VisualEventConfig()
    scores = _accepted_scores(readings, active)
    candidates: list[EventCandidate] = []
    previous: ScoreReading | None = None
    for current in scores:
        if previous is None:
            previous = current
            continue
        home_delta = current.home_score - previous.home_score
        away_delta = current.away_score - previous.away_score
        valid = (
            home_delta >= 0
            and away_delta >= 0
            and home_delta + away_delta > 0
            and home_delta + away_delta <= active.score_change_max_jump
        )
        if valid:
            changed_team = "home" if home_delta else "away"
            delta = home_delta or away_delta
            subtype = {
                2: "safety",
                3: "field_goal",
                6: "touchdown",
                7: "touchdown_or_extra_point",
                8: "touchdown_or_two_point",
            }.get(delta, "score_change")
            confidence = _bounded(min(previous.confidence, current.confidence) * 0.95)
            candidates.append(
                EventCandidate(
                    timestamp=current.timestamp,
                    kind="score_change",
                    confidence=confidence,
                    source_start=max(0.0, current.timestamp - active.score_change_pre_roll),
                    source_end=current.timestamp + active.score_change_post_roll,
                    reasons=(
                        f"{changed_team} score +{delta}",
                        f"score-change:{subtype}",
                    ),
                    score_delta=(home_delta, away_delta),
                )
            )
        # Keep the latest valid scoreboard as the baseline.  A transient OCR
        # glitch therefore cannot create a false multi-score cascade.
        if home_delta >= 0 and away_delta >= 0:
            previous = current
    return candidates


def _contiguous_runs(
    observations: Sequence[VisualObservation],
    predicate: Callable[[VisualObservation], bool],
    max_gap_seconds: float,
) -> list[list[VisualObservation]]:
    runs: list[list[VisualObservation]] = []
    current: list[VisualObservation] = []
    for observation in sorted(observations, key=lambda item: item.timestamp):
        if not predicate(observation):
            if current:
                runs.append(current)
                current = []
            continue
        if current and observation.timestamp - current[-1].timestamp > max_gap_seconds:
            runs.append(current)
            current = []
        current.append(observation)
    if current:
        runs.append(current)
    return runs


def detect_sustained_field_action(
    observations: Iterable[VisualObservation],
    *,
    config: VisualEventConfig | None = None,
) -> list[EventCandidate]:
    """Find sustained on-field motion, which is useful around snaps and plays."""

    active = config or VisualEventConfig()
    ordered = sorted(observations, key=lambda item: item.timestamp)
    candidates: list[EventCandidate] = []
    for run in _contiguous_runs(
        ordered,
        lambda item: item.field_action_score >= active.field_action_threshold,
        active.field_action_gap_seconds,
    ):
        start = run[0].timestamp
        end = run[-1].timestamp
        sample_gap = run[-1].timestamp - run[-2].timestamp if len(run) > 1 else 0.0
        duration = max(sample_gap, end - start)
        if duration < active.field_action_min_seconds:
            continue
        end = min(end + max(sample_gap, 0.25), start + active.field_action_max_seconds)
        mean_action = sum(item.field_action_score for item in run) / len(run)
        confidence = _bounded(0.45 * mean_action + 0.25 * min(1.0, duration / 5.0) + 0.30)
        candidates.append(
            EventCandidate(
                timestamp=(start + end) / 2,
                kind="sustained_field_action",
                confidence=confidence,
                source_start=start,
                source_end=end,
                reasons=("sustained field motion", f"action-score:{mean_action:.2f}"),
            )
        )
    return candidates


def detect_replay_transitions(
    observations: Iterable[VisualObservation],
    *,
    config: VisualEventConfig | None = None,
) -> list[EventCandidate]:
    """Find hard broadcast cuts likely to enter a replay or commentary shot."""

    active = config or VisualEventConfig()
    ordered = sorted(observations, key=lambda item: item.timestamp)
    candidates: list[EventCandidate] = []
    for index, observation in enumerate(ordered):
        if observation.scene_change_score < active.scene_change_threshold:
            continue
        following = ordered[index + 1 : index + 4]
        no_scorebug = following and (
            sum(not item.scorebug_visible for item in following) / len(following)
        )
        low_field = (
            following and sum(item.field_action_score for item in following) / len(following) < 0.35
        )
        if not (no_scorebug or low_field):
            continue
        confidence = _bounded(
            0.50 * observation.scene_change_score
            + 0.25 * (1.0 if no_scorebug else 0.0)
            + 0.25 * (1.0 if low_field else 0.0)
        )
        reasons = [f"shot-change:{observation.scene_change_score:.2f}"]
        if no_scorebug:
            reasons.append("scorebug disappeared")
        if low_field:
            reasons.append("field action dropped")
        candidates.append(
            EventCandidate(
                timestamp=observation.timestamp,
                kind="replay_transition",
                confidence=confidence,
                source_start=max(0.0, observation.timestamp - 0.25),
                source_end=observation.timestamp + active.replay_post_seconds,
                reasons=tuple(reasons),
            )
        )
    return candidates


def detect_celebration_cuts(
    observations: Iterable[VisualObservation],
    *,
    config: VisualEventConfig | None = None,
) -> list[EventCandidate]:
    """Find cut-and-celebration patterns without assuming a PBP event list."""

    active = config or VisualEventConfig()
    ordered = sorted(observations, key=lambda item: item.timestamp)
    candidates: list[EventCandidate] = []
    for index, observation in enumerate(ordered):
        if observation.scene_change_score < active.scene_change_threshold:
            continue
        following = ordered[index + 1 : index + 4]
        if not following:
            continue
        celebration = max(item.celebration_score for item in following)
        if celebration < active.celebration_threshold:
            continue
        confidence = _bounded(0.55 * observation.scene_change_score + 0.45 * celebration)
        candidates.append(
            EventCandidate(
                timestamp=observation.timestamp,
                kind="celebration_cut",
                confidence=confidence,
                source_start=max(0.0, observation.timestamp - 1.0),
                source_end=following[-1].timestamp + 1.0,
                reasons=(
                    f"celebration-score:{celebration:.2f}",
                    f"shot-change:{observation.scene_change_score:.2f}",
                ),
            )
        )
    return candidates


_KIND_PRIORITY = {
    "score_change": 4,
    "sustained_field_action": 3,
    "celebration_cut": 2,
    "replay_transition": 1,
}


def fuse_candidates(
    candidates: Iterable[EventCandidate],
    *,
    merge_window_seconds: float = 5.0,
) -> list[EventCandidate]:
    """Merge nearby evidence into stable candidates with deterministic confidence."""

    if merge_window_seconds < 0:
        raise ValueError("merge_window_seconds must be non-negative")
    ordered = sorted(candidates, key=lambda item: (item.source_start, -item.confidence, item.kind))
    groups: list[list[EventCandidate]] = []
    for candidate in ordered:
        latest_end = max(item.source_end for item in groups[-1]) if groups else -math.inf
        if groups and candidate.source_start <= latest_end + merge_window_seconds:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    fused: list[EventCandidate] = []
    for group in groups:
        primary = max(group, key=lambda item: (_KIND_PRIORITY.get(item.kind, 0), item.confidence))
        confidence = 1.0
        for item in group:
            confidence *= 1.0 - item.confidence
        confidence = _bounded(1.0 - confidence)
        reasons: list[str] = []
        for item in sorted(group, key=lambda item: (_KIND_PRIORITY.get(item.kind, 0), item.kind)):
            for reason in item.reasons:
                if reason not in reasons:
                    reasons.append(reason)
        delta = next((item.score_delta for item in group if item.score_delta != (0, 0)), (0, 0))
        fused.append(
            EventCandidate(
                timestamp=primary.timestamp,
                kind=primary.kind,
                confidence=confidence,
                source_start=min(item.source_start for item in group),
                source_end=max(item.source_end for item in group),
                reasons=tuple(reasons),
                score_delta=delta,
            )
        )
    return sorted(fused, key=lambda item: item.timestamp)


def detect_visual_events(
    observations: Iterable[VisualObservation],
    score_readings: Iterable[ScoreReading] = (),
    *,
    config: VisualEventConfig | None = None,
) -> list[EventCandidate]:
    """Generate and fuse video-only event candidates from sampled signals."""

    active = config or VisualEventConfig()
    frames = tuple(observations)
    raw: list[EventCandidate] = []
    raw.extend(detect_score_changes(score_readings, config=active))
    raw.extend(detect_sustained_field_action(frames, config=active))
    raw.extend(detect_replay_transitions(frames, config=active))
    raw.extend(detect_celebration_cuts(frames, config=active))
    return fuse_candidates(raw, merge_window_seconds=active.merge_window_seconds)


def _scorebug_crop(frame: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, crop_width, crop_height = region
    left = max(0, min(width - 1, int(round(x * width))))
    top = max(0, min(height - 1, int(round(y * height))))
    right = max(left + 1, min(width, int(round((x + crop_width) * width))))
    bottom = max(top + 1, min(height, int(round((y + crop_height) * height))))
    return frame[top:bottom, left:right]


def _call_ocr(ocr: OCRCallback, crop: np.ndarray, timestamp: float) -> ScoreReading | None:
    try:
        value = ocr(crop, timestamp)
    except TypeError:
        # A one-argument callback is convenient in tests and for tiny OCR wrappers.
        value = ocr(crop)  # type: ignore[call-arg]
    return _coerce_score_reading(value, timestamp)


def sample_video(
    source: str | Path,
    *,
    sample_fps: float = 2.0,
    ocr: OCRCallback | None = None,
    scorebug_region: tuple[float, float, float, float] = (0.25, 0.75, 0.50, 0.25),
) -> VideoSampleResult:
    """Sample a local video with OpenCV and optionally run local scorebug OCR.

    The sampler is intentionally conservative and inexpensive: it downscales
    frames before calculating absolute motion and histogram-like scene changes.
    It does not call a network service or require a play-by-play file.
    """

    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if len(scorebug_region) != 4 or any(value < 0 for value in scorebug_region):
        raise ValueError("scorebug_region must contain four non-negative fractions")
    if scorebug_region[0] + scorebug_region[2] > 1 or scorebug_region[1] + scorebug_region[3] > 1:
        raise ValueError("scorebug_region must fit inside the frame")
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV is required for video sampling")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f"could not open video: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = frame_count / fps if frame_count else 0.0
        step = max(1, round(fps / sample_fps))
        observations: list[VisualObservation] = []
        readings: list[ScoreReading] = []
        previous_small: np.ndarray | None = None
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % step:
                frame_index += 1
                continue
            timestamp = frame_index / fps
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            green = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(green, (30, 35, 25), (95, 255, 255))
            green_ratio = float(np.count_nonzero(green_mask)) / green_mask.size
            if previous_small is None:
                motion = 0.0
                scene_change = 0.0
            else:
                difference = cv2.absdiff(gray, previous_small)
                raw_change = float(np.mean(difference)) / 255.0
                motion = _bounded(raw_change * 4.0)
                scene_change = _bounded(raw_change * 2.5)
            field_action = _bounded(motion * (0.45 + 0.9 * green_ratio))
            celebration = _bounded(scene_change * 0.55 + motion * 0.25 + (1.0 - green_ratio) * 0.20)
            scorebug_visible = True
            if ocr is not None:
                reading = _call_ocr(ocr, _scorebug_crop(frame, scorebug_region), timestamp)
                if reading is not None:
                    readings.append(reading)
                else:
                    scorebug_visible = False
            observations.append(
                VisualObservation(
                    timestamp=timestamp,
                    motion_score=motion,
                    scene_change_score=scene_change,
                    field_action_score=field_action,
                    celebration_score=celebration,
                    green_field_ratio=green_ratio,
                    scorebug_visible=scorebug_visible,
                )
            )
            previous_small = gray
            frame_index += 1
        if observations:
            duration = max(duration, observations[-1].timestamp)
        return VideoSampleResult(tuple(observations), tuple(readings), fps, width, height, duration)
    finally:
        capture.release()


def analyze_video(
    source: str | Path,
    *,
    sample_fps: float = 2.0,
    ocr: OCRCallback | None = None,
    scorebug_region: tuple[float, float, float, float] = (0.25, 0.75, 0.50, 0.25),
    config: VisualEventConfig | None = None,
) -> list[EventCandidate]:
    """Run local OpenCV/OCR sampling and return timestamped event candidates."""

    result = sample_video(
        source,
        sample_fps=sample_fps,
        ocr=ocr,
        scorebug_region=scorebug_region,
    )
    return detect_visual_events(result.observations, result.score_readings, config=config)
