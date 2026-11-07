"""Streaming, encoding and external audio-tool regression tests."""

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stem_local.audio import (
    CHANNELS,
    SAMPLE_RATE,
    StreamWindow,
    chunk_starts,
    decode_audio,
    export_stems,
    run_audio_tool,
    stream_stem,
)


@pytest.mark.parametrize("frames", [1, 2, 19, 20, 21, 36, 37, 38, 39, 100, 101])
def test_streaming_identity_preserves_length_and_boundaries(
    frames: int,
    audio_factory: Callable[..., tuple[Path, np.ndarray]],
    tmp_path: Path,
) -> None:
    source, expected = audio_factory(frames=frames, sample_rate=10)
    destination = tmp_path / "vocals.wav"
    updates: list[tuple[int, int]] = []

    stream_stem(
        source=source,
        destination=destination,
        infer=lambda audio, **_kwargs: audio.copy(),
        progress=lambda done, total: updates.append((done, total)),
        window=StreamWindow(duration_seconds=2, overlap_seconds=0.2),
    )

    actual, _ = sf.read(destination, dtype="float32", always_2d=True)
    np.testing.assert_allclose(actual, expected, atol=2e-8)
    assert updates[-1] == (frames, frames)
    assert all(first[0] < second[0] for first, second in pairwise(updates))


def test_schedule_never_adds_redundant_tail() -> None:
    assert list(chunk_starts(frames=38, window=20, overlap=2)) == [0, 18]


def test_stream_reports_inner_model_steps_before_the_audio_window_finishes(
    audio_factory: Callable[..., tuple[Path, np.ndarray]],
    tmp_path: Path,
) -> None:
    source, _ = audio_factory(frames=20, sample_rate=10)
    updates: list[tuple[int, int]] = []

    def infer(audio: np.ndarray, *, progress: Callable[[int, int], None]) -> np.ndarray:
        for done in range(1, 5):
            progress(done, 4)
            assert updates[-1] == (done * 5, 20)
        return audio.copy()

    stream_stem(
        source=source,
        destination=tmp_path / "vocals.wav",
        infer=infer,
        progress=lambda done, total: updates.append((done, total)),
        window=StreamWindow(duration_seconds=2, overlap_seconds=0.2),
    )

    assert updates[:4] == [(5, 20), (10, 20), (15, 20), (20, 20)]
    assert updates[-1] == (20, 20)


@pytest.mark.parametrize("overlap", [-1, 10, 20])
def test_invalid_schedule_rejected(overlap: int) -> None:
    with pytest.raises(ValueError, match="Invalid chunk schedule"):
        list(chunk_starts(frames=100, window=20, overlap=overlap))


def test_export_only_requested_stems_with_shared_gain(
    audio_factory: Callable[..., tuple[Path, np.ndarray]],
    tmp_path: Path,
) -> None:
    _, original = audio_factory(frames=5000, sample_rate=44100, amplitude=0.4)
    voice = original * 4
    vocal_path = tmp_path / "vocal-float.wav"
    sf.write(vocal_path, voice, 44100, subtype="FLOAT")
    melody_path = tmp_path / "melody-float.wav"
    sf.write(melody_path, original, 44100, subtype="FLOAT")

    result = export_stems(
        stems={"vocals": vocal_path, "melody": melody_path},
        output_dir=tmp_path,
        output_format="WAV",
    )

    vocals, rate = sf.read(str(result["vocals"]))
    melody, _ = sf.read(str(result["melody"]))
    gain = float(str(result["shared_gain"]))
    assert 0 < gain < 1
    assert rate == SAMPLE_RATE
    assert sf.info(str(result["vocals"])).subtype == "PCM_24"
    np.testing.assert_allclose(vocals, voice * gain, atol=2e-7)
    np.testing.assert_allclose(melody, original * gain, atol=2e-7)
    assert set(result) == {"vocals", "melody", "shared_gain"}
    maximum_peak = 0.999001
    assert np.max(np.abs(vocals)) <= maximum_peak


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_decode_trims_and_resamples(
    audio_factory: Callable[..., tuple[Path, np.ndarray]],
    tmp_path: Path,
) -> None:
    source, _ = audio_factory(frames=16000, sample_rate=16000)
    destination = tmp_path / "decoded.wav"

    duration = 0.3
    decode_audio(source=source, destination=destination, start=0.2, duration=duration)

    info = sf.info(destination)
    assert info.samplerate == SAMPLE_RATE
    assert info.channels == CHANNELS
    assert info.frames == round(duration * SAMPLE_RATE)


def test_invalid_model_output_is_rejected(
    audio_factory: Callable[..., tuple[Path, np.ndarray]],
    tmp_path: Path,
) -> None:
    source, _ = audio_factory()

    with pytest.raises(ValueError, match="invalid samples"):
        stream_stem(
            source=source,
            destination=tmp_path / "invalid.wav",
            infer=lambda audio, **_kwargs: np.full_like(audio, np.nan),
            progress=lambda _done, _total: None,
        )


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="FFmpeg is not installed")
def test_mp3_export_is_320k_and_stereo(
    audio_factory: Callable[..., tuple[Path, np.ndarray]],
    tmp_path: Path,
) -> None:
    path, _ = audio_factory(frames=44100, sample_rate=44100)

    result = export_stems(stems={"vocals": path}, output_dir=tmp_path, output_format="MP3")

    executable = shutil.which("ffprobe")
    assert executable is not None
    probe = subprocess.run(  # noqa: S603 - fixed probe arguments and a test-created local file; no shell.
        [executable, "-v", "error", "-show_streams", "-of", "json", str(result["vocals"])],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "mp3"
    assert stream["bit_rate"] == "320000"
    assert stream["channels"] == CHANNELS
    assert "instrumental" not in result


def test_vocals_only_does_not_generate_other_stems(
    audio_factory: Callable[..., tuple[Path, np.ndarray]],
    tmp_path: Path,
) -> None:
    path, _ = audio_factory(frames=4410, sample_rate=44100)

    result = export_stems(stems={"vocals": path}, output_dir=tmp_path, output_format="WAV")

    assert set(result) == {"vocals", "shared_gain"}
    assert not (tmp_path / "instrumental.wav").exists()
    assert not (tmp_path / "mix.wav").exists()


def test_external_audio_process_has_a_bounded_timeout() -> None:
    allowed_seconds = 3
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="остановлен"):
        run_audio_tool([sys.executable, "-c", "import time; time.sleep(60)"], timeout_seconds=0.1)

    assert time.monotonic() - started < allowed_seconds


def test_external_audio_tool_output_is_decoded_as_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("locale.getencoding", lambda: "cp1251")
    code = "import sys; sys.stdout.buffer.write('Музыка 🎵'.encode('utf-8'))"

    result = run_audio_tool([sys.executable, "-c", code], timeout_seconds=10)

    assert result.stdout == "Музыка 🎵"
