"""The pinned PyTorch/audio-separator adapter. Never imported by the GUI."""

import os
import platform
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from audio_separator.separator import Separator
from audio_separator.separator.uvr_lib_v5.demucs.pretrained import get_model

from stem_local.models import ModelSpec, verify_artifact

Audio = npt.NDArray[np.float32]


class CPUSeparator(Separator):
    local_spec: ModelSpec

    def setup_accelerated_inferencing_device(self) -> None:
        # Override BEFORE load_model; never run upstream's CUDA/MPS auto-selection.
        self.torch_device = torch.device("cpu")
        self.torch_device_cpu = torch.device("cpu")
        self.torch_device_mps = None
        self.onnx_execution_provider = ["CPUExecutionProvider"]
        self.logger.info("Stem Local: forced CPU, FP32; CUDA/MPS/CoreML/DirectML disabled")

    def download_model_files(self, model_filename: str) -> tuple[str, str, str, str, str]:
        # No network catalog, auto-downloads, or hash-based online configuration lookup.
        expected = (
            self.local_spec.config.filename
            if self.local_spec.architecture == "Demucs"
            else self.local_spec.weights.filename
        )
        if model_filename != expected:
            message = "Unexpected model requested"
            raise ValueError(message)
        return (
            model_filename,
            self.local_spec.architecture,
            self.local_spec.title,
            str(Path(self.model_file_dir) / model_filename),
            str(Path(self.model_file_dir) / self.local_spec.config.filename),
        )


def configure_cpu(threads: int) -> None:
    """Apply CPU/FP32 settings before any model is loaded."""
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    os.environ.pop("AUDIO_SEPARATOR_MODEL_DIR", None)


class CPUAudit:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.forward_checks = 0
        self.verify()
        # Demucs bags dispatch to their child models rather than calling the bag's forward.
        for actual_model in getattr(model, "models", [model]):
            actual_model.register_forward_pre_hook(self._check_input)

    def verify(self) -> None:
        for tensor in (*self.model.parameters(), *self.model.buffers()):
            if tensor.device.type != "cpu":
                message = f"CPU policy violation: model tensor on {tensor.device}"
                raise RuntimeError(message)
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                message = f"FP32 policy violation: {tensor.dtype}"
                raise RuntimeError(message)

    def _check_input(self, _module: torch.nn.Module, arguments: tuple[object, ...]) -> None:
        for value in arguments:
            if isinstance(value, torch.Tensor) and (value.device.type != "cpu" or value.dtype != torch.float32):
                message = "CPU/FP32 policy violation in forward input"
                raise RuntimeError(message)
        self.forward_checks += 1

    def report(self) -> dict[str, object]:
        self.verify()
        if self.forward_checks == 0:
            message = "No verified model forward calls"
            raise RuntimeError(message)
        return {
            "device": "cpu",
            "precision": "float32",
            "verified_forward_calls": self.forward_checks,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        }


class VocalEngine:
    def __init__(self, *, spec: ModelSpec, cache: Path, threads: int, overlap: int) -> None:
        for artifact in (spec.config, spec.weights):
            verify_artifact(cache / spec.id / artifact.filename, artifact)
        configure_cpu(threads)
        self.separator = CPUSeparator(
            model_file_dir=str(cache / spec.id),
            output_dir=str(cache / spec.id),
            use_autocast=False,
            use_native_fp16=False,
            use_torch_compile=False,
            mdxc_params={
                "segment_size": 801,  # 441 * (801 - 1) = 8 s, author's training context.
                "override_model_segment_size": True,
                "batch_size": 1,
                "overlap": overlap,
                "pitch_shift": 0,
            },
        )
        self.separator.local_spec = spec
        self.separator.load_model(model_filename=spec.weights.filename)
        self.instance = self.separator.model_instance
        self.model = self.instance.model_run
        self.model.eval()
        self.cpu_audit = CPUAudit(self.model)
        if self.separator.torch_device.type != "cpu" or self.instance.torch_device.type != "cpu":
            message = "CPU policy violation in separator"
            raise RuntimeError(message)

    def separate(self, audio: Audio) -> Audio:
        # Direct demix avoids per-block normalization or lossy intermediate encoding.
        with torch.inference_mode():
            sources = self.instance.demix(mix=audio.T.copy(), override_model_segment_size=True)
        if not isinstance(sources, dict):
            message = "Expected a stem dictionary from the pinned engine"
            raise TypeError(message)
        vocal_key = next((key for key in sources if key.casefold() == "vocals"), None)
        if vocal_key is None:
            message = f"Vocal stem is missing: {list(sources)}"
            raise RuntimeError(message)
        result = np.asarray(sources[vocal_key].T, dtype=np.float32)
        if result.shape != audio.shape or not np.isfinite(result).all():
            message = f"Invalid model output: {result.shape}, expected {audio.shape}"
            raise RuntimeError(message)
        return result

    def audit(self) -> dict[str, object]:
        return self.cpu_audit.report()


class MelodyEngine:
    def __init__(self, *, spec: ModelSpec, cache: Path, threads: int, keep_bass: bool) -> None:
        for artifact in (spec.config, spec.weights):
            verify_artifact(cache / spec.id / artifact.filename, artifact)
        configure_cpu(threads)
        self.keep_bass = keep_bass
        separator = CPUSeparator(
            model_file_dir=str(cache / spec.id),
            output_dir=str(cache / spec.id),
            demucs_params={"segment_size": "Default", "shifts": 0, "overlap": 0.25, "segments_enabled": True},
        )
        separator.local_spec = spec
        separator.load_model(model_filename=spec.config.filename)
        self.instance = separator.model_instance
        self.model = get_model(name=spec.config.filename.removesuffix(".yaml"), repo=cache / spec.id)
        self.model.to(torch.device("cpu")).eval()
        self.cpu_audit = CPUAudit(self.model)
        self.instance.demucs_model_instance = self.model

    def separate(self, audio: Audio) -> Audio:
        with torch.inference_mode():
            # The wrapper reorders bass/drums, but leaves index 2=other untouched.
            sources = self.instance.demix_demucs(audio.T.copy())
        melody = sources[2].copy()
        if self.keep_bass:
            melody += sources[0]
        return np.asarray(melody.T, dtype=np.float32)

    def audit(self) -> dict[str, object]:
        return self.cpu_audit.report()
