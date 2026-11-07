"""CPU policy and model-adapter regression tests."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stem_local.backends.audio_separator import CPUAudit, CPUSeparator, MelodyEngine


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
    engine.instance = SimpleNamespace(demix_demucs=lambda _audio: sources)

    result = engine.separate(np.zeros((50, 2), dtype=np.float32))

    np.testing.assert_array_equal(result, np.full((50, 2), expected, dtype=np.float32))
