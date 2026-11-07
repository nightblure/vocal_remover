"""Shared bounded audio fixtures and headless Qt configuration."""

import os
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def audio_factory(tmp_path: Path) -> Callable[..., tuple[Path, np.ndarray]]:
    def create(*, frames: int = 100, sample_rate: int = 100, amplitude: float = 0.2) -> tuple[Path, np.ndarray]:
        time = np.arange(frames, dtype=np.float32) / sample_rate
        wave = amplitude * np.sin(2 * np.pi * 7 * time)
        audio = np.column_stack((wave, wave * 0.5)).astype(np.float32)
        path = tmp_path / f"input-{frames}-{sample_rate}.wav"
        sf.write(path, audio, sample_rate, subtype="FLOAT")
        return path, audio

    return create
