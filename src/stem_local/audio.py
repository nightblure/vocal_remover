"""Bounded-memory audio streaming with overlapping model windows."""

import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

Audio = npt.NDArray[np.float32]
SAMPLE_RATE = 44100
CHANNELS = 2
TOOL_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class StreamWindow:
    duration_seconds: float = 20.0
    overlap_seconds: float = 2.0


DEFAULT_WINDOW = StreamWindow()


def run_audio_tool(command: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    """Run a shell-free local audio command with bounded execution time."""
    try:
        return subprocess.run(  # noqa: S603 - argv only; callers select the executable, never a shell.
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        message = f"FFmpeg превысил лимит {timeout_seconds:g} секунд и был остановлен."
        raise TimeoutError(message) from error


def decode_audio(*, source: Path, destination: Path, start: float, duration: float | None) -> None:
    """Decode the requested excerpt to a stereo float32 WAV on disk."""
    if not source.is_file():
        message = f"Входной файл отсутствует: {source}"
        raise FileNotFoundError(message)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-protocol_whitelist",
        "file,pipe",
        "-ss",
        str(start),
        "-i",
        str(source.resolve()),
    ]
    if duration is not None:
        command.extend(["-t", str(duration)])
    command.extend(["-vn", "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(destination)])
    result = run_audio_tool(command, timeout_seconds=TOOL_TIMEOUT_SECONDS)
    if result.returncode:
        message = f"FFmpeg: {result.stderr.strip()}"
        raise RuntimeError(message)
    if sf.info(destination).frames == 0:
        message = "Выбранный фрагмент пуст. Проверьте начало и длительность."
        raise ValueError(message)


def chunk_starts(*, frames: int, window: int, overlap: int) -> Iterator[int]:
    """Yield window offsets without appending a redundant overlapping tail."""
    if frames <= 0 or window <= 0 or overlap < 0 or overlap * 2 >= window:
        message = "Invalid chunk schedule"
        raise ValueError(message)
    start = 0
    while True:
        yield start
        if start + window >= frames:
            return
        start += window - overlap


def stream_stem(
    *,
    source: Path,
    destination: Path,
    infer: Callable[[Audio], Audio],
    progress: Callable[[int, int], None],
    window: StreamWindow = DEFAULT_WINDOW,
) -> None:
    """Separate bounded windows and crossfade their overlaps into one stem."""
    with (
        sf.SoundFile(source) as reader,
        sf.SoundFile(
            destination,
            mode="w",
            samplerate=reader.samplerate,
            channels=reader.channels,
            subtype="FLOAT",
        ) as writer,
    ):
        window_frames = round(window.duration_seconds * reader.samplerate)
        overlap = round(window.overlap_seconds * reader.samplerate)
        pending: Audio | None = None
        for start in chunk_starts(frames=len(reader), window=window_frames, overlap=overlap):
            reader.seek(start)
            block = reader.read(window_frames, dtype="float32", always_2d=True)
            estimate = infer(block)
            if estimate.shape != block.shape or not np.isfinite(estimate).all():
                message = "Model returned invalid samples"
                raise ValueError(message)
            if pending is not None:
                count = len(pending)
                ramp = np.linspace(0, 1, count, dtype=np.float32)[:, None]
                estimate[:count] = pending * (1 - ramp) + estimate[:count] * ramp
            final = start + len(block) >= len(reader)
            keep = 0 if final else overlap
            stop = len(estimate) - keep
            writer.write(estimate[:stop])
            pending = estimate[stop:].copy() if keep else None
            progress(min(start + stop, len(reader)), len(reader))


def _measure_peak(stems: dict[str, Path]) -> float:
    peak = 0.0
    frame_counts = set()
    for path in stems.values():
        with sf.SoundFile(path) as reader:
            if reader.samplerate != SAMPLE_RATE or reader.channels != CHANNELS:
                message = "Export expects stereo 44100 Hz audio"
                raise ValueError(message)
            frame_counts.add(len(reader))
            for block in reader.blocks(blocksize=SAMPLE_RATE, dtype="float32", always_2d=True):
                if not np.isfinite(block).all():
                    message = "Cannot export non-finite samples"
                    raise ValueError(message)
                peak = max(peak, float(np.max(np.abs(block))))
    if len(frame_counts) != 1 or 0 in frame_counts:
        message = "Stems must have equal, non-zero duration"
        raise ValueError(message)
    return peak


def export_stems(*, stems: dict[str, Path], output_dir: Path, output_format: str) -> dict[str, object]:
    """Export only requested stems, preserving their relative level with one global gain."""
    if not stems or output_format not in {"WAV", "MP3"}:
        message = "Expected at least one stem and WAV or MP3 output"
        raise ValueError(message)
    peak = _measure_peak(stems)

    target = 0.97 if output_format == "MP3" else 0.999
    gain = min(1.0, target / peak) if peak > 0 else 1.0
    extension = output_format.lower()
    outputs: dict[str, object] = {"shared_gain": gain}
    for name, path in stems.items():
        destination = output_dir / f"{name}.{extension}"
        if destination.resolve() == path.resolve():
            message = "Export must not overwrite its source"
            raise ValueError(message)
        if output_format == "MP3":
            _export_mp3(source=path, destination=destination, gain=gain)
        else:
            _export_wav(source=path, destination=destination, gain=gain)
        outputs[name] = str(destination)
    return outputs


def _export_mp3(*, source: Path, destination: Path, gain: float) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(source),
        "-af",
        f"volume={gain:.12f}",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "320k",
        str(destination),
    ]
    result = run_audio_tool(command, timeout_seconds=TOOL_TIMEOUT_SECONDS)
    if result.returncode:
        message = f"MP3 export: {result.stderr.strip()}"
        raise RuntimeError(message)


def _export_wav(*, source: Path, destination: Path, gain: float) -> None:
    with (
        sf.SoundFile(source) as reader,
        sf.SoundFile(
            destination,
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=2,
            subtype="PCM_24",
        ) as writer,
    ):
        for block in reader.blocks(blocksize=SAMPLE_RATE, dtype="float32", always_2d=True):
            writer.write(block * gain)
