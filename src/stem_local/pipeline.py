"""Disk-backed separation stages, imported only after worker input validation."""

import argparse
import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from stem_local import engine
from stem_local.audio import SAMPLE_RATE, decode_audio, export_stems, stream_stem
from stem_local.models import MODELS, EventCallback, ModelSpec


@dataclass(frozen=True)
class StageMetrics:
    load_seconds: float
    inference_seconds: float
    audit: dict[str, object]


@dataclass(frozen=True)
class PipelineResult:
    audio_seconds: float
    vocals: StageMetrics
    melody: StageMetrics | None
    outputs: dict[str, object]


def _infer_stem(
    *,
    args: argparse.Namespace,
    spec: ModelSpec,
    source: Path,
    destination: Path,
    progress: Callable[[int, int], None],
) -> StageMetrics:
    loading = time.perf_counter()
    model = engine.create_engine(
        spec=spec,
        cache=args.cache,
        threads=args.threads,
        overlap=args.overlap,
        keep_bass=args.keep_bass,
    )
    load_seconds = time.perf_counter() - loading
    started = time.perf_counter()
    stream_stem(source=source, destination=destination, infer=model.separate, progress=progress)
    return StageMetrics(load_seconds, time.perf_counter() - started, model.audit())


def run_pipeline(*, args: argparse.Namespace, work: Path, emit: EventCallback) -> PipelineResult:
    """Separate requested stems sequentially, releasing each model before the next."""
    mix = work / "mix.wav"
    vocals = work / "vocals-float.wav"
    emit({"event": "status", "message": "Декодирование аудио · CPU"})
    decode_audio(
        source=args.input,
        destination=mix,
        start=args.start,
        duration=None if args.full or args.duration is None else args.duration,
    )
    audio_seconds = sf.info(mix).frames / SAMPLE_RATE
    stages = 2 if args.mode == "vocals-melody" else 1
    emit({"event": "status", "message": "Проверка SHA-256, загрузка и выделение вокала · CPU / FP32"})
    vocal_metrics = _infer_stem(
        args=args,
        spec=MODELS[args.model],
        source=mix,
        destination=vocals,
        progress=lambda done, total: emit({"event": "progress", "done": done, "total": total * stages}),
    )
    gc.collect()
    stems = {"vocals": vocals}
    melody_metrics = None
    if args.mode == "vocals-melody":
        emit({"event": "status", "message": "Загрузка Demucs и выделение мелодий · первая модель освобождена"})
        melody = work / "melody-float.wav"
        melody_metrics = _infer_stem(
            args=args,
            spec=MODELS["htdemucs"],
            source=mix,
            destination=melody,
            progress=lambda done, total: emit({"event": "progress", "done": total + done, "total": total * stages}),
        )
        gc.collect()
        stems["melody"] = melody
    emit({"event": "status", "message": f"Экспорт {args.format} · только выбранные дорожки"})
    exported = export_stems(stems=stems, output_dir=work, output_format=args.format)
    for stem in stems:
        source = Path(str(exported[stem]))
        destination = work.parent / source.name
        source.replace(destination)
        exported[stem] = str(destination)
    return PipelineResult(audio_seconds, vocal_metrics, melody_metrics, exported)
