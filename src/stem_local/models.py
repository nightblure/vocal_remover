"""Pinned model provenance. Inference never consults the engine's online catalog."""

import hashlib
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

EventCallback = Callable[[dict[str, object]], None]
DEFAULT_CACHE = Path.home() / ".cache" / "stem-local" / "models"
MSST_REVISION = "f0b527736e7da24eae5b561a2bb1efd0569ec757"
MSST_CONFIGS = f"https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/{MSST_REVISION}/configs"


@dataclass(frozen=True)
class Artifact:
    filename: str
    url: str
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    id: str
    title: str
    weights: Artifact
    config: Artifact
    license: str
    license_source: str
    license_verified: bool
    architecture: str = "MDXC"
    role: str = "vocals"
    backend: str = "audio-separator"

    def provenance(self) -> dict[str, object]:
        return asdict(self)


MODELS = {
    "kimberley": ModelSpec(
        id="kimberley",
        title="Mel-Band RoFormer · Kimberley Jensen",
        weights=Artifact(
            filename="MelBandRoformer.ckpt",
            url="https://huggingface.co/KimberleyJSN/melbandroformer/resolve/"
            "ac9b0614ab3cd7f77219e18ba494dfd93956c348/MelBandRoformer.ckpt",
            sha256="87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e",
        ),
        config=Artifact(
            filename="config_vocals_mel_band_roformer_kj.yaml",
            url=f"{MSST_CONFIGS}/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
            sha256="f63f38eb1e6e40a7db0dade714a5ae257555dd8748f4e774eae8679275a81926",
        ),
        license="MIT (model card declaration)",
        license_source="https://huggingface.co/KimberleyJSN/melbandroformer/blob/"
        "ac9b0614ab3cd7f77219e18ba494dfd93956c348/README.md",
        license_verified=True,
    ),
    "viperx": ModelSpec(
        id="viperx",
        title="BS-RoFormer · ViperX (лицензия весов не подтверждена)",
        weights=Artifact(
            filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            sha256="5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
        ),
        config=Artifact(
            filename="model_bs_roformer_ep_317_sdr_12.9755.yaml",
            url=f"{MSST_CONFIGS}/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
            sha256="42e5635ceab7287b83d9591c9880966f0b61cff26a0a40a29d236bacb32e5f2c",
        ),
        license="UNVERIFIED — not cleared for redistribution",
        license_source="https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/"
        f"{MSST_REVISION}/docs/pretrained_models.md",
        license_verified=False,
    ),
    "htdemucs": ModelSpec(
        id="htdemucs",
        title="HTDemucs · мелодии (other / other + bass)",
        weights=Artifact(
            filename="955717e8-8726e21a.th",
            url="https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
            sha256="8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4",
        ),
        config=Artifact(
            filename="htdemucs.yaml",
            url="https://raw.githubusercontent.com/facebookresearch/demucs/"
            "e976d93ecc3865e5757426930257e200846a520a/demucs/remote/htdemucs.yaml",
            sha256="239c445d0b14454d541ad8bd9bb271c9e536d267e8a4625208744cbb2e7bb66c",
        ),
        license="MIT (upstream Demucs repository license)",
        license_source="https://github.com/facebookresearch/demucs/blob/"
        "e976d93ecc3865e5757426930257e200846a520a/LICENSE",
        license_verified=True,
        architecture="Demucs",
        role="melody",
    ),
}


def sha256_file(path: Path) -> str:
    """Hash a file without loading the model weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(path: Path, artifact: Artifact) -> None:
    """Reject missing or corrupt cached model files before loading them."""
    if not path.is_file():
        message = f"Отсутствует файл {path}. Сначала скачайте выбранную модель."
        raise FileNotFoundError(message)
    if sha256_file(path) != artifact.sha256:
        message = f"SHA-256 не совпадает: {path}. Нажмите «Проверить модели», чтобы восстановить файл."
        raise ValueError(message)


def check_license(spec: ModelSpec, *, accept_unverified: bool) -> None:
    """Require explicit consent for models with unverified weight licenses."""
    if not spec.license_verified and not accept_unverified:
        message = "Лицензия весов ViperX не подтверждена. Требуется явное согласие пользователя."
        raise ValueError(message)


def download_artifact(*, artifact: Artifact, directory: Path, emit: EventCallback, repair: bool = False) -> Path:
    """Verify cached weights or download and atomically publish a verified artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / artifact.filename
    if destination.exists():
        emit({"event": "status", "message": f"Проверка SHA-256: {artifact.filename}"})
        try:
            verify_artifact(destination, artifact)
        except ValueError:
            if not repair:
                raise
            emit({"event": "status", "message": f"Восстановление повреждённого файла: {artifact.filename}"})
        else:
            emit({"event": "status", "message": f"Уже скачан и проверен: {artifact.filename}"})
            return destination

    if urlsplit(artifact.url).scheme != "https":
        message = "Model downloads require HTTPS"
        raise ValueError(message)
    request = urllib.request.Request(  # noqa: S310 - the scheme is validated above.
        artifact.url,
        headers={"User-Agent": "StemLocal/0.1"},
    )
    temporary: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - HTTPS-only request above.
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            last_report = -1
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                while block := response.read(1024 * 1024):
                    handle.write(block)
                    downloaded += len(block)
                    megabytes = downloaded // (8 * 1024 * 1024)
                    if megabytes != last_report:
                        emit(
                            {
                                "event": "download",
                                "file": artifact.filename,
                                "bytes": downloaded,
                                "total": total,
                            },
                        )
                        last_report = megabytes
        verify_artifact(temporary, artifact)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def prepare_model(
    *,
    spec: ModelSpec,
    cache: Path,
    emit: EventCallback,
    accept_unverified: bool,
    repair: bool = False,
) -> None:
    """Download a model's pinned configuration and weights after checking its license."""
    check_license(spec, accept_unverified=accept_unverified)
    for artifact in (spec.config, spec.weights):
        download_artifact(artifact=artifact, directory=cache / spec.id, emit=emit, repair=repair)
    emit({"event": "ready", "model": spec.id})
