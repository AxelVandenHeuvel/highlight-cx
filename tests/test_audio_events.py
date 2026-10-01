import io
import subprocess
import wave

import numpy as np
import pytest

from football_highlights.audio_events import (
    AudioEventConfig,
    AudioExtractionError,
    build_ffmpeg_audio_command,
    compute_audio_features,
    decode_wav,
    detect_audio_events,
    detect_audio_events_from_video,
    extract_audio_ffmpeg,
)


def _wav_bytes(samples: np.ndarray, sample_rate: int = 1000, channels: int = 1) -> bytes:
    values = np.asarray(samples, dtype=np.float32)
    if channels > 1:
        values = np.repeat(values[:, None], channels, axis=1).reshape(-1)
    pcm = np.clip(values * 32767, -32768, 32767).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


def test_build_ffmpeg_command_is_local_mono_wav_extraction() -> None:
    command = build_ffmpeg_audio_command("game.mkv", sample_rate=8000, ffmpeg_bin="local-ffmpeg")

    assert command == [
        "local-ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "game.mkv",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "8000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-",
    ]
    assert "http" not in " ".join(command)


def test_decode_wav_downmixes_and_normalizes_pcm() -> None:
    samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    decoded, sample_rate = decode_wav(_wav_bytes(samples, sample_rate=2000, channels=2))

    assert sample_rate == 2000
    np.testing.assert_allclose(decoded, samples, atol=2 / 32767)


def test_extract_audio_wraps_local_ffmpeg_errors() -> None:
    def runner(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr=b"bad input")

    with pytest.raises(AudioExtractionError, match="ffmpeg audio extraction failed"):
        extract_audio_ffmpeg("game.mkv", runner=runner)


def test_extract_audio_uses_stdout_and_returns_decoded_samples() -> None:
    wav = _wav_bytes(np.array([0.0, 0.25, -0.25], dtype=np.float32), sample_rate=1000)
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=wav, stderr=b"")

    samples, sample_rate = extract_audio_ffmpeg(
        "game.mkv", ffmpeg_bin="ffmpeg-local", runner=runner
    )

    assert sample_rate == 1000
    assert calls[0][1] == {"check": True, "capture_output": True}
    np.testing.assert_allclose(samples, [0.0, 0.25, -0.25], atol=2 / 32767)


def test_features_are_finite_and_have_aligned_scores() -> None:
    samples = np.zeros(3000, dtype=np.float32)
    samples[1000:1300] = 0.8
    features = compute_audio_features(
        samples,
        1000,
        config=AudioEventConfig(sample_rate=1000, frame_seconds=0.2, hop_seconds=0.1),
    )

    assert len(features.times) == len(features.excitement_score)
    assert np.isfinite(features.excitement_score).all()
    assert np.all((features.excitement_score >= 0) & (features.excitement_score <= 1))
    assert float(np.max(features.excitement_score)) > 0.6


def test_detection_is_deterministic_and_clusters_nearby_bursts() -> None:
    sample_rate = 1000
    samples = np.zeros(8000, dtype=np.float32)
    samples[2000:2300] = 0.95
    samples[4200:4500] = -0.95
    settings = AudioEventConfig(
        sample_rate=sample_rate,
        frame_seconds=0.2,
        hop_seconds=0.1,
        activation_threshold=0.58,
        merge_gap_seconds=0.8,
        pre_roll_seconds=0.2,
        post_roll_seconds=0.4,
    )

    first = detect_audio_events(samples, sample_rate, config=settings)
    second = detect_audio_events(samples, sample_rate, config=settings)

    assert first == second
    assert len(first) == 2
    assert first[0].activity_start < first[0].peak_time < first[0].activity_end
    assert first[0].start < first[0].activity_start
    assert first[0].end > first[0].activity_end
    assert first[0].confidence > 0
    assert first[0].reasons


def test_nearby_active_frames_merge_into_one_candidate() -> None:
    sample_rate = 1000
    samples = np.zeros(4000, dtype=np.float32)
    samples[1000:1200] = 0.9
    samples[1450:1650] = 0.9
    settings = AudioEventConfig(
        sample_rate=sample_rate,
        frame_seconds=0.2,
        hop_seconds=0.1,
        activation_threshold=0.55,
        merge_gap_seconds=0.6,
        pre_roll_seconds=0,
        post_roll_seconds=0,
    )

    events = detect_audio_events(samples, sample_rate, config=settings)

    assert len(events) == 1
    assert events[0].activity_start <= 1.0
    assert events[0].activity_end >= 1.6


def test_video_detector_extracts_then_analyzes_without_network() -> None:
    sample_rate = 1000
    samples = np.zeros(3000, dtype=np.float32)
    samples[1000:1300] = 0.95
    wav = _wav_bytes(samples, sample_rate=sample_rate)

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=wav, stderr=b"")

    events = detect_audio_events_from_video(
        "local-game.mkv",
        config=AudioEventConfig(
            sample_rate=sample_rate,
            frame_seconds=0.2,
            hop_seconds=0.1,
            activation_threshold=0.55,
        ),
        runner=runner,
    )

    assert events
    assert all(0 <= event.start < event.end <= 3 for event in events)
