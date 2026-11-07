"""CPU policy and model-adapter regression tests."""

import logging
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from audio_separator.separator.architectures.demucs_separator import DemucsSeparator
from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.uvr_lib_v5.demucs.apply import BagOfModels
from ml_collections import ConfigDict

from stem_local.backends.audio_separator import CPUAudit, CPUSeparator, MelodyEngine, VocalEngine


def test_cpu_selection_does_not_use_available_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    separator = object.__new__(CPUSeparator)
    separator.logger = logging.getLogger("test-cpu-policy")

    separator.setup_accelerated_inferencing_device()

    assert separator.torch_device.type == "cpu"
    assert separator.torch_device_mps is None
    assert separator.onnx_execution_provider == ["CPUExecutionProvider"]


def test_cpu_audit_checks_actual_forward_calls() -> None:
    model = torch.nn.Linear(2, 2).float().cpu()
    audit = CPUAudit(model)

    model(torch.ones(1, 2, dtype=torch.float32))

    assert audit.report()["device"] == "cpu"
    assert audit.report()["verified_forward_calls"] == 1


def test_cpu_audit_rejects_non_cpu_model() -> None:
    model = torch.nn.Linear(2, 2, device="meta")

    with pytest.raises(RuntimeError, match="CPU policy"):
        CPUAudit(model)


def test_cpu_audit_rejects_non_cpu_input() -> None:
    model = torch.nn.Identity()
    CPUAudit(model)

    with pytest.raises(RuntimeError, match="policy violation"):
        model(torch.ones(1, 2, device="meta"))


@pytest.mark.parametrize(("keep_bass", "expected"), [(False, 3), (True, 4)])
def test_melody_contains_other_and_optional_bass_not_drums(*, keep_bass: bool, expected: int) -> None:
    # Wrapper order is bass, drums, other, vocals — not the model's native order.
    sources = np.stack([np.full((2, 50), value, dtype=np.float32) for value in (1, 100, 3, 1000)])
    engine = object.__new__(MelodyEngine)
    engine.keep_bass = keep_bass
    engine.model = torch.nn.Identity()
    engine.instance = SimpleNamespace(demix_demucs=lambda _audio: sources)

    result = engine.separate(np.zeros((50, 2), dtype=np.float32))

    np.testing.assert_array_equal(result, np.full((50, 2), expected, dtype=np.float32))


class TinyVocalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return audio * self.scale


class TinyDemucsModel(TinyVocalModel):
    samplerate = 10
    segment = 0.8
    audio_channels = 2
    sources = ("drums", "bass", "other", "vocals")

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.stack([audio * self.scale * value for value in range(1, 5)], dim=1)


@pytest.fixture
def vocal_engine() -> VocalEngine:
    instance = object.__new__(MDXCSeparator)
    instance.logger = logging.getLogger("test-vocal-progress")
    instance.is_roformer = True
    instance.is_torch_compiled = False
    instance.is_primary_stem_main_target = True
    instance.primary_stem_name = "vocals"
    instance.secondary_stem_name = "instrumental"
    instance.pitch_shift = 0
    instance.segment_size = 9
    instance.overlap = 2
    instance.model_data_cfgdict = ConfigDict(
        {
            "model": {"stft_hop_length": 1},
            "audio": {"hop_length": 1},
            "training": {"target_instrument": "vocals", "instruments": ["vocals", "other"]},
        }
    )
    instance.model_run = TinyVocalModel()
    engine = object.__new__(VocalEngine)
    engine.instance = instance
    engine.model = instance.model_run
    engine.cpu_audit = CPUAudit(engine.model)
    return engine


@pytest.fixture
def melody_engine_factory() -> Callable[..., MelodyEngine]:
    def create(*, members: int = 1) -> MelodyEngine:
        instance = object.__new__(DemucsSeparator)
        instance.logger = logging.getLogger("test-melody-progress")
        instance.torch_device = torch.device("cpu")
        instance.shifts = 0
        instance.overlap = 0.25
        instance.segments_enabled = True
        model = BagOfModels([TinyDemucsModel() for _ in range(members)])
        instance.demucs_model_instance = model
        engine = object.__new__(MelodyEngine)
        engine.instance = instance
        engine.model = model
        engine.cpu_audit = CPUAudit(model)
        engine.keep_bass = False
        return engine

    return create


@pytest.mark.parametrize(("frames", "steps"), [(1, 1), (8, 1), (9, 2), (20, 4), (21, 5)])
def test_vocal_progress_uses_the_pinned_demixer_schedule(vocal_engine: VocalEngine, frames: int, steps: int) -> None:
    audio = np.ones((frames, 2), dtype=np.float32)
    updates: list[tuple[int, int]] = []

    result = vocal_engine.separate(audio, progress=lambda done, total: updates.append((done, total)))

    assert updates == [(done, steps) for done in range(1, steps + 1)]
    assert vocal_engine.cpu_audit.forward_checks == steps
    np.testing.assert_allclose(result, audio)
    vocal_engine.model(torch.ones(1, 2, frames))
    assert len(updates) == steps  # Task progress hooks must not survive a completed call.


@pytest.mark.parametrize("members", [1, 2])
@pytest.mark.parametrize("frames", [1, 8, 20, 21])
def test_melody_progress_counts_real_forwards_including_bag_members(
    *,
    melody_engine_factory: Callable[..., MelodyEngine],
    members: int,
    frames: int,
) -> None:
    engine = melody_engine_factory(members=members)
    audio = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2)
    updates: list[tuple[int, int]] = []
    expected_steps = len(range(0, frames, 6)) * members

    result = engine.separate(audio, progress=lambda done, total: updates.append((done, total)))

    assert updates == [(done, expected_steps) for done in range(1, expected_steps + 1)]
    assert engine.cpu_audit.forward_checks == expected_steps
    assert result.shape == audio.shape
    assert np.isfinite(result).all()


def test_progress_hooks_are_removed_after_cancellation(vocal_engine: VocalEngine) -> None:
    audio = np.ones((20, 2), dtype=np.float32)

    def cancel(_done: int, _total: int) -> None:
        raise InterruptedError

    with pytest.raises(InterruptedError):
        vocal_engine.separate(audio, progress=cancel)

    updates: list[tuple[int, int]] = []
    vocal_engine.separate(audio, progress=lambda done, total: updates.append((done, total)))
    assert updates == [(1, 4), (2, 4), (3, 4), (4, 4)]
