"""Backend-neutral contract and the single place that selects an inference adapter.

An engine receives stereo float32 samples (frames, channels) at 44100 Hz and
returns the requested stem with the same shape, finite samples, and no gain
normalization. Models must run on CPU and cannot fetch files during inference.
The caller owns chunking, sequencing, progress, cancellation and export.
"""

from pathlib import Path
from typing import Protocol

from stem_local.audio import Audio
from stem_local.models import ModelSpec


class StemEngine(Protocol):
    def separate(self, audio: Audio) -> Audio: ...

    def audit(self) -> dict[str, object]: ...


def create_engine(
    *,
    spec: ModelSpec,
    cache: Path,
    threads: int,
    overlap: int = 2,
    keep_bass: bool = False,
) -> StemEngine:
    """Construct the pinned adapter only when inference is actually requested."""
    # One real backend today. Add another explicit adapter when it exists, not a
    # plugin system that imports arbitrary code/checkpoints from a remote catalog.
    if spec.backend != "audio-separator":
        message = f"Unsupported backend: {spec.backend}"
        raise ValueError(message)

    from stem_local.backends.audio_separator import MelodyEngine, VocalEngine  # noqa: PLC0415 - keep torch out of GUI.

    if spec.role == "vocals":
        return VocalEngine(spec=spec, cache=cache, threads=threads, overlap=overlap)
    if spec.role == "melody":
        return MelodyEngine(spec=spec, cache=cache, threads=threads, keep_bass=keep_bass)
    message = f"Unsupported model role: {spec.role}"
    raise ValueError(message)
