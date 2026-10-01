"""Strict live-play validation for video-only football highlight discovery.

The detector in this module is intentionally a gate, not a highlight scorer.
Audio excitement or transcript keywords may propose a candidate, but a
candidate is accepted only when the sampled video contains the broadcast
evidence of a real play:

* a scorebug is visible through the live interval;
* a wide green-field, low-motion pre-snap formation exists;
* coordinated motion follows that formation (the snap/onset);
* the shot stays continuous and is not a replay/commentary cut; and
* an injected/local OCR reading shows the game clock moving.

All decision logic is dependency-free.  ``sample_video_around_candidate`` is
the optional local OpenCV adapter that turns a video window into the signals
consumed by :func:`validate_live_play`.  OCR is injected so callers can use
RapidOCR, Tesseract, or a deterministic test double without coupling this
module to one OCR engine.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Keep the pure validator importable in environments without OpenCV.
    import cv2
except ImportError:  # pragma: no cover - exercised only by minimal installs
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:  # pragma: no cover - NumPy is a normal project dependency
    np = None  # type: ignore[assignment]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class ScorebugReading:
    """One local OCR result for the scorebug.

    ``clock_seconds`` is seconds remaining in the quarter.  A reading may
    report visibility without a readable clock, which is useful when OCR is
    temporarily imperfect; clock movement still requires two valid readings.
    """

    timestamp: float
    visible: bool = True
    quarter: int | None = None
    clock_seconds: float | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        timestamp = _finite(self.timestamp, "timestamp")
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if not isinstance(self.visible, bool):
            raise ValueError("visible must be a bool")
        if self.quarter is not None:
            if isinstance(self.quarter, bool) or not isinstance(self.quarter, int):
                raise ValueError("quarter must be an integer or None")
            if not 1 <= self.quarter <= 5:
                raise ValueError("quarter must be between 1 and 5")
        if self.clock_seconds is not None:
            clock = _finite(self.clock_seconds, "clock_seconds")
            if not 0 <= clock <= 900:
                raise ValueError("clock_seconds must be between 0 and 900")
        confidence = _finite(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


# Names that make the type easy to discover for callers using different
# terminology.
GameClockReading = ScorebugReading
ScorebugOCRReading = ScorebugReading


@dataclass(frozen=True, slots=True)
class FrameSignal:
    """Signals calculated for one sampled source-video frame.

    Scores are normalized to 0..1.  ``scorebug_visible=None`` means that the
    sampler has no OCR evidence for that frame; strict validation treats that
    as missing evidence rather than assuming the scorebug is present.
    """

    timestamp: float
    green_field_ratio: float = 0.0
    motion_score: float = 0.0
    coordinated_motion_score: float = 0.0
    shot_change_score: float = 0.0
    scorebug_visible: bool | None = None
    replay_indicator: bool = False
    quarter: int | None = None
    clock_seconds: float | None = None

    def __post_init__(self) -> None:
        timestamp = _finite(self.timestamp, "timestamp")
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        for name in (
            "green_field_ratio",
            "motion_score",
            "coordinated_motion_score",
            "shot_change_score",
        ):
            value = _finite(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.scorebug_visible is not None and not isinstance(self.scorebug_visible, bool):
            raise ValueError("scorebug_visible must be a bool or None")
        if not isinstance(self.replay_indicator, bool):
            raise ValueError("replay_indicator must be a bool")
        if self.quarter is not None:
            if isinstance(self.quarter, bool) or not isinstance(self.quarter, int):
                raise ValueError("quarter must be an integer or None")
            if not 1 <= self.quarter <= 5:
                raise ValueError("quarter must be between 1 and 5")
        if self.clock_seconds is not None:
            clock = _finite(self.clock_seconds, "clock_seconds")
            if not 0 <= clock <= 900:
                raise ValueError("clock_seconds must be between 0 and 900")


SampledFrame = FrameSignal
FrameObservation = FrameSignal


@dataclass(frozen=True, slots=True)
class LivePlayConfig:
    """Deterministic thresholds for the strict live-play gate."""

    min_green_field_ratio: float = 0.38
    max_pre_snap_motion: float = 0.22
    min_coordinated_motion: float = 0.34
    min_snap_motion: float = 0.30
    min_pre_snap_seconds: float = 0.75
    max_snap_search_seconds: float = 3.0
    max_sample_gap_seconds: float = 1.25
    scorebug_min_confidence: float = 0.45
    min_scorebug_coverage: float = 0.75
    max_shot_change_score: float = 0.68
    max_missing_scorebug_seconds: float = 0.75
    min_clock_delta_seconds: float = 0.25
    min_live_seconds: float = 0.50
    max_live_seconds: float = 20.0
    post_play_quiet_seconds: float = 1.25

    def __post_init__(self) -> None:
        for name in (
            "min_green_field_ratio",
            "max_pre_snap_motion",
            "min_coordinated_motion",
            "min_snap_motion",
            "scorebug_min_confidence",
            "min_scorebug_coverage",
            "max_shot_change_score",
        ):
            value = _finite(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in (
            "min_pre_snap_seconds",
            "max_snap_search_seconds",
            "max_sample_gap_seconds",
            "max_missing_scorebug_seconds",
            "min_clock_delta_seconds",
            "min_live_seconds",
            "max_live_seconds",
            "post_play_quiet_seconds",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_pre_snap_seconds > self.max_snap_search_seconds:
            raise ValueError("min_pre_snap_seconds cannot exceed max_snap_search_seconds")
        if self.min_live_seconds > self.max_live_seconds:
            raise ValueError("min_live_seconds cannot exceed max_live_seconds")


@dataclass(frozen=True, slots=True)
class LivePlayResult:
    """Explainable result returned by :func:`validate_live_play`."""

    accepted: bool
    confidence: float
    live_start: float | None
    live_end: float | None
    snap_timestamp: float | None
    reasons: tuple[str, ...]
    evidence: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        confidence = _finite(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        for name, value in (
            ("live_start", self.live_start),
            ("live_end", self.live_end),
            ("snap_timestamp", self.snap_timestamp),
        ):
            if value is not None and _finite(value, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.live_start is not None
            and self.live_end is not None
            and self.live_end <= self.live_start
        ):
            raise ValueError("live_end must be greater than live_start")
        if not self.reasons:
            raise ValueError("reasons must not be empty")
        for name, value in self.evidence:
            if not name or not 0 <= value <= 1:
                raise ValueError("evidence must contain named 0..1 scores")


@dataclass(frozen=True, slots=True)
class SampledVideo:
    """Result of optional local OpenCV sampling around a candidate."""

    frames: tuple[FrameSignal, ...]
    scorebug_readings: tuple[ScorebugReading, ...]
    fps: float
    width: int
    height: int
    duration: float


def _ordered_unique[T](values: Iterable[T], key: Callable[[T], float]) -> list[T]:
    ordered = sorted(values, key=key)
    result: list[T] = []
    last: float | None = None
    for value in ordered:
        current = key(value)
        if last is not None and math.isclose(current, last, abs_tol=1e-7):
            continue
        result.append(value)
        last = current
    return result


def _coerce_ocr(value: Any, timestamp: float) -> ScorebugReading | None:
    if value is None:
        return None
    if isinstance(value, ScorebugReading):
        if math.isclose(value.timestamp, timestamp, abs_tol=1e-7):
            return value
        return ScorebugReading(
            timestamp=timestamp,
            visible=value.visible,
            quarter=value.quarter,
            clock_seconds=value.clock_seconds,
            confidence=value.confidence,
        )
    if isinstance(value, Mapping):
        visible = value.get("visible", value.get("scorebug_visible", True))
        try:
            return ScorebugReading(
                timestamp=timestamp,
                visible=bool(visible),
                quarter=(int(value["quarter"]) if value.get("quarter") is not None else None),
                clock_seconds=(
                    float(value["clock_seconds"])
                    if value.get("clock_seconds") is not None
                    else None
                ),
                confidence=float(value.get("confidence", 1.0)),
            )
        except (TypeError, ValueError):
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < 1:
            return None
        try:
            clock = float(value[0]) if value[0] is not None else None
            quarter = int(value[1]) if len(value) > 1 and value[1] is not None else None
            confidence = float(value[2]) if len(value) > 2 else 1.0
            return ScorebugReading(timestamp, True, quarter, clock, confidence)
        except (TypeError, ValueError):
            return None
    return None


def _reading_for_frame(
    frame: FrameSignal,
    readings: Sequence[ScorebugReading],
    tolerance: float,
) -> ScorebugReading | None:
    matches = [item for item in readings if abs(item.timestamp - frame.timestamp) <= tolerance]
    if not matches:
        return None
    return min(matches, key=lambda item: (abs(item.timestamp - frame.timestamp), -item.confidence))


def _effective_scorebug(
    frame: FrameSignal,
    reading: ScorebugReading | None,
    config: LivePlayConfig,
) -> bool:
    if frame.scorebug_visible is not None:
        return frame.scorebug_visible
    return (
        reading is not None
        and reading.visible
        and reading.confidence >= config.scorebug_min_confidence
    )


def _clock_readings(
    frames: Sequence[FrameSignal],
    readings: Iterable[ScorebugReading],
    config: LivePlayConfig,
) -> list[ScorebugReading]:
    result = list(readings)
    for frame in frames:
        if frame.clock_seconds is not None:
            result.append(
                ScorebugReading(
                    timestamp=frame.timestamp,
                    visible=frame.scorebug_visible is not False,
                    quarter=frame.quarter,
                    clock_seconds=frame.clock_seconds,
                    confidence=1.0,
                )
            )
    return [
        item
        for item in _ordered_unique(result, lambda value: value.timestamp)
        if item.visible and item.confidence >= config.scorebug_min_confidence
    ]


def _pre_snap(frame: FrameSignal, scorebug: bool, config: LivePlayConfig) -> bool:
    return (
        scorebug
        and not frame.replay_indicator
        and frame.green_field_ratio >= config.min_green_field_ratio
        and frame.motion_score <= config.max_pre_snap_motion
    )


def _runs(
    frames: Sequence[FrameSignal],
    predicate: Callable[[FrameSignal], bool],
    max_gap: float,
) -> list[list[FrameSignal]]:
    runs: list[list[FrameSignal]] = []
    current: list[FrameSignal] = []
    for frame in frames:
        if not predicate(frame):
            if current:
                runs.append(current)
                current = []
            continue
        if current and frame.timestamp - current[-1].timestamp > max_gap:
            runs.append(current)
            current = []
        current.append(frame)
    if current:
        runs.append(current)
    return runs


def _clock_progress(
    readings: Sequence[ScorebugReading],
    start: float,
    end: float,
    config: LivePlayConfig,
) -> tuple[bool, float, str]:
    in_window = [item for item in readings if start - 1.0 <= item.timestamp <= end + 1.5]
    in_window.sort(key=lambda item: item.timestamp)
    for previous, current in zip(in_window, in_window[1:], strict=False):
        same_quarter = (
            previous.quarter is None
            or current.quarter is None
            or previous.quarter == current.quarter
        )
        if (
            same_quarter
            and previous.clock_seconds is not None
            and current.clock_seconds is not None
            and current.timestamp > previous.timestamp
            and previous.clock_seconds - current.clock_seconds >= config.min_clock_delta_seconds
        ):
            delta = previous.clock_seconds - current.clock_seconds
            return (
                True,
                _bounded(min(1.0, delta / 2.0)),
                (
                    "game clock moved from "
                    f"{previous.clock_seconds:g}s to {current.clock_seconds:g}s"
                ),
            )
    return False, 0.0, "game clock did not move during the candidate"


def _coverage(
    frames: Sequence[FrameSignal],
    readings: Sequence[ScorebugReading],
    start: float,
    end: float,
    config: LivePlayConfig,
) -> tuple[float, float]:
    in_window = [item for item in frames if start <= item.timestamp <= end]
    if not in_window:
        return 0.0, math.inf
    visible = 0
    missing_duration = 0.0
    previous: FrameSignal | None = None
    for frame in in_window:
        reading = _reading_for_frame(frame, readings, config.max_sample_gap_seconds / 2)
        present = _effective_scorebug(frame, reading, config)
        visible += int(present)
        if previous is not None and not present:
            missing_duration += min(
                config.max_sample_gap_seconds,
                max(0.0, frame.timestamp - previous.timestamp),
            )
        previous = frame
    return visible / len(in_window), missing_duration


def _find_end(
    frames: Sequence[FrameSignal],
    onset_index: int,
    config: LivePlayConfig,
) -> tuple[float, str]:
    onset = frames[onset_index]
    max_end = onset.timestamp + config.max_live_seconds
    quiet_start: float | None = None
    previous = onset
    for index in range(onset_index + 1, len(frames)):
        frame = frames[index]
        if frame.timestamp > max_end:
            return max_end, "maximum live-play duration reached"
        if frame.timestamp - previous.timestamp > config.max_sample_gap_seconds:
            return previous.timestamp, "sample gap ended the continuous live shot"
        if frame.replay_indicator or frame.shot_change_score > config.max_shot_change_score:
            return frame.timestamp, "replay/shot transition ended the live shot"
        # A single OCR miss is not itself a shot cut.  Keep sampling so the
        # coverage gate can distinguish a transient miss from a sustained
        # dropout.  When the scorebug disappears with a hard cut or replay
        # indicator, the branches above trim the interval at that boundary.
        if frame.motion_score <= config.max_pre_snap_motion:
            quiet_start = frame.timestamp if quiet_start is None else quiet_start
            if frame.timestamp - quiet_start >= config.post_play_quiet_seconds:
                return quiet_start, "post-play motion stayed quiet"
        else:
            quiet_start = None
        previous = frame
    return min(previous.timestamp, max_end), "sampled live shot ended without a replay cut"


def validate_live_play(
    frames: Iterable[FrameSignal],
    scorebug_readings: Iterable[ScorebugReading] = (),
    *,
    candidate_timestamp: float | None = None,
    config: LivePlayConfig | None = None,
) -> LivePlayResult:
    """Strictly accept or reject a candidate using sampled video evidence.

    The input may be unsorted; the caller's objects are never mutated.  If a
    candidate timestamp is supplied, the valid pre-snap/onset sequence nearest
    that timestamp is selected.  A rejection still returns the best detected
    interval when one exists, making false positives auditable.
    """

    active = config or LivePlayConfig()
    ordered = _ordered_unique(frames, lambda item: item.timestamp)
    if candidate_timestamp is not None:
        candidate_timestamp = _finite(candidate_timestamp, "candidate_timestamp")
        if candidate_timestamp < 0:
            raise ValueError("candidate_timestamp must be non-negative")
    readings = _clock_readings(ordered, scorebug_readings, active)
    if not ordered:
        return LivePlayResult(False, 0.0, None, None, None, ("no sampled frames",))

    frame_readings = {
        frame.timestamp: _reading_for_frame(frame, readings, active.max_sample_gap_seconds / 2)
        for frame in ordered
    }
    pre_runs = _runs(
        ordered,
        lambda frame: _pre_snap(
            frame,
            _effective_scorebug(frame, frame_readings[frame.timestamp], active),
            active,
        ),
        active.max_sample_gap_seconds,
    )
    valid_pre_runs = [
        run
        for run in pre_runs
        if run[-1].timestamp - run[0].timestamp >= active.min_pre_snap_seconds
    ]

    reasons: list[str] = []
    evidence: dict[str, float] = {}
    if valid_pre_runs:
        reasons.append("wide green-field pre-snap formation detected")
        evidence["pre_snap_formation"] = 1.0
    else:
        reasons.append("rejected: no stable wide green-field pre-snap formation")
        evidence["pre_snap_formation"] = 0.0

    onset_options: list[tuple[float, int, list[FrameSignal]]] = []
    for run in valid_pre_runs:
        start = run[-1].timestamp
        for index, frame in enumerate(ordered):
            if frame.timestamp <= start or frame.timestamp - start > active.max_snap_search_seconds:
                continue
            if (
                frame.motion_score >= active.min_snap_motion
                and frame.coordinated_motion_score >= active.min_coordinated_motion
                and not frame.replay_indicator
            ):
                distance = abs(frame.timestamp - (candidate_timestamp or frame.timestamp))
                onset_options.append((distance, index, run))
                break
    if onset_options:
        _, onset_index, pre_run = min(
            onset_options,
            key=lambda item: (item[0], item[1], item[2][0].timestamp),
        )
        onset = ordered[onset_index]
        live_start = pre_run[0].timestamp
        reasons.append(f"coordinated motion/snap onset detected at {onset.timestamp:.2f}s")
        evidence["snap_onset"] = _bounded(
            0.5 * onset.motion_score + 0.5 * onset.coordinated_motion_score
        )
    else:
        onset_index = -1
        onset = None
        live_start = None
        reasons.append("rejected: no coordinated motion/snap onset after formation")
        evidence["snap_onset"] = 0.0

    if onset is not None:
        live_end, end_reason = _find_end(ordered, onset_index, active)
        if live_end <= live_start:
            live_end = min(ordered[-1].timestamp, onset.timestamp + active.min_live_seconds)
        reasons.append(end_reason)
        duration = live_end - live_start
    else:
        live_end = None
        duration = 0.0

    if live_start is not None and live_end is not None:
        coverage, missing_duration = _coverage(ordered, readings, live_start, live_end, active)
    else:
        coverage, missing_duration = 0.0, math.inf
    evidence["scorebug_coverage"] = _bounded(coverage)
    if (
        coverage >= active.min_scorebug_coverage
        and missing_duration <= active.max_missing_scorebug_seconds
    ):
        reasons.append(f"scorebug present across {coverage:.0%} of the live interval")
    else:
        reasons.append(
            f"rejected: scorebug coverage was {coverage:.0%} with {missing_duration:.2f}s missing"
        )

    continuity = False
    if onset is not None and live_end is not None:
        interval = [frame for frame in ordered if live_start <= frame.timestamp <= live_end]
        continuity = bool(interval) and all(
            not frame.replay_indicator and frame.shot_change_score <= active.max_shot_change_score
            for frame in interval
        )
    evidence["continuous_live_shot"] = 1.0 if continuity else 0.0
    if continuity:
        reasons.append("continuous live shot confirmed")
    else:
        reasons.append("rejected: replay, commentary cut, or shot discontinuity detected")

    clock_ok = False
    clock_score = 0.0
    clock_reason = "game clock could not be evaluated"
    if live_start is not None and live_end is not None:
        clock_ok, clock_score, clock_reason = _clock_progress(
            readings, live_start, live_end, active
        )
    evidence["game_clock_movement"] = clock_score
    if clock_ok:
        reasons.append(clock_reason)
    else:
        reasons.append(f"rejected: {clock_reason}")

    duration_ok = (
        onset is not None and active.min_live_seconds <= duration <= active.max_live_seconds
    )
    evidence["duration"] = (
        _bounded(duration / max(active.min_live_seconds, 1.0)) if duration else 0.0
    )
    if not duration_ok:
        reasons.append("rejected: live interval duration is outside the configured bounds")

    hard_gates = (
        bool(valid_pre_runs),
        onset is not None,
        coverage >= active.min_scorebug_coverage
        and missing_duration <= active.max_missing_scorebug_seconds,
        continuity,
        clock_ok,
        duration_ok,
    )
    accepted = all(hard_gates)
    confidence = _bounded(
        0.20 * evidence["pre_snap_formation"]
        + 0.20 * evidence["snap_onset"]
        + 0.20 * evidence["scorebug_coverage"]
        + 0.15 * evidence["continuous_live_shot"]
        + 0.20 * evidence["game_clock_movement"]
        + 0.05 * (1.0 if duration_ok else 0.0)
    )
    if accepted:
        reasons.insert(0, "accepted: all strict live-play gates passed")
    return LivePlayResult(
        accepted=accepted,
        confidence=confidence,
        live_start=live_start,
        live_end=live_end,
        snap_timestamp=onset.timestamp if onset is not None else None,
        reasons=tuple(reasons),
        evidence=tuple(sorted(evidence.items())),
    )


OCRCallback = Callable[[Any, float], ScorebugReading | Mapping[str, Any] | Sequence[Any] | None]


def _crop(frame: Any, region: tuple[float, float, float, float]) -> Any:
    height, width = frame.shape[:2]
    x, y, crop_width, crop_height = region
    left = max(0, min(width - 1, round(x * width)))
    top = max(0, min(height - 1, round(y * height)))
    right = max(left + 1, min(width, round((x + crop_width) * width)))
    bottom = max(top + 1, min(height, round((y + crop_height) * height)))
    return frame[top:bottom, left:right]


def _call_ocr(ocr: OCRCallback, crop: Any, timestamp: float) -> ScorebugReading | None:
    try:
        value = ocr(crop, timestamp)
    except TypeError:
        value = ocr(crop)  # type: ignore[call-arg]
    return _coerce_ocr(value, timestamp)


def sample_video_around_candidate(
    source: str | Path,
    candidate_timestamp: float,
    *,
    window_before: float = 5.0,
    window_after: float = 12.0,
    sample_fps: float = 4.0,
    scorebug_ocr: OCRCallback | None = None,
    scorebug_region: tuple[float, float, float, float] = (0.25, 0.75, 0.50, 0.25),
) -> SampledVideo:
    """Sample a local OpenCV window and optionally run local scorebug OCR."""

    candidate_timestamp = _finite(candidate_timestamp, "candidate_timestamp")
    if candidate_timestamp < 0:
        raise ValueError("candidate_timestamp must be non-negative")
    if window_before < 0 or window_after <= 0 or sample_fps <= 0:
        raise ValueError("window_before must be non-negative; window_after and sample_fps positive")
    if len(scorebug_region) != 4 or any(value < 0 for value in scorebug_region):
        raise ValueError("scorebug_region must contain four non-negative fractions")
    if scorebug_region[0] + scorebug_region[2] > 1 or scorebug_region[1] + scorebug_region[3] > 1:
        raise ValueError("scorebug_region must fit inside the frame")
    if cv2 is None or np is None:  # pragma: no cover
        raise RuntimeError("OpenCV and NumPy are required for local video sampling")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f"could not open video: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = frame_count / fps if frame_count else 0.0
        start = max(0.0, candidate_timestamp - window_before)
        end = candidate_timestamp + window_after
        capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        step = max(1, round(fps / sample_fps))
        frame_index = max(0, round(start * fps))
        previous_gray: Any = None
        frames: list[FrameSignal] = []
        readings: list[ScorebugReading] = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_index / fps
            frame_index += 1
            if timestamp > end:
                break
            if (frame_index - round(start * fps) - 1) % step:
                continue
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, (30, 35, 25), (95, 255, 255))
            green_ratio = float(np.count_nonzero(green_mask)) / green_mask.size
            motion = 0.0
            coordinated = 0.0
            scene_change = 0.0
            if previous_gray is not None:
                difference = cv2.absdiff(gray, previous_gray)
                raw = float(np.mean(difference)) / 255.0
                motion = _bounded(raw * 4.0)
                scene_change = _bounded(raw * 2.5)
                motion_mask = (difference >= 18).astype(np.uint8)
                active_cells = 0
                for row in range(3):
                    for col in range(3):
                        cell = motion_mask[row * 30 : (row + 1) * 30, col * 53 : (col + 1) * 53]
                        if float(np.mean(cell)) >= 0.025:
                            active_cells += 1
                coordinated = _bounded(
                    0.55 * min(1.0, float(np.mean(motion_mask)) * 10.0)
                    + 0.45 * (active_cells / 9.0)
                )
            reading = None
            scorebug_visible: bool | None = None
            if scorebug_ocr is not None:
                reading = _call_ocr(scorebug_ocr, _crop(frame, scorebug_region), timestamp)
                if reading is not None:
                    readings.append(reading)
                    scorebug_visible = reading.visible
                else:
                    scorebug_visible = False
            frames.append(
                FrameSignal(
                    timestamp=timestamp,
                    green_field_ratio=_bounded(green_ratio),
                    motion_score=motion,
                    coordinated_motion_score=coordinated,
                    shot_change_score=scene_change,
                    scorebug_visible=scorebug_visible,
                    quarter=reading.quarter if reading else None,
                    clock_seconds=reading.clock_seconds if reading else None,
                )
            )
            previous_gray = gray
        if frames:
            duration = max(duration, frames[-1].timestamp)
        return SampledVideo(tuple(frames), tuple(readings), fps, width, height, duration)
    finally:
        capture.release()


sample_candidate_video = sample_video_around_candidate


def validate_video_candidate(
    source: str | Path,
    candidate_timestamp: float,
    *,
    scorebug_ocr: OCRCallback | None = None,
    config: LivePlayConfig | None = None,
    **sampling_options: Any,
) -> LivePlayResult:
    """Convenience wrapper: locally sample a candidate, then validate it."""

    sampled = sample_video_around_candidate(
        source,
        candidate_timestamp,
        scorebug_ocr=scorebug_ocr,
        **sampling_options,
    )
    return validate_live_play(
        sampled.frames,
        sampled.scorebug_readings,
        candidate_timestamp=candidate_timestamp,
        config=config,
    )


validate_live_play_video = validate_video_candidate
