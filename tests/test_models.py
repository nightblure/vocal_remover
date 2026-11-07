"""Model-cache integrity, download and license checks."""

import hashlib
import io
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest

from stem_local.models import MODELS, Artifact, check_license, download_artifact, verify_artifact


@pytest.fixture
def artifact() -> Artifact:
    return Artifact(
        filename="model.ckpt",
        url="https://example.invalid/model",
        sha256=hashlib.sha256(b"weights").hexdigest(),
    )


def test_cached_model_is_verified_without_network(
    artifact: Artifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / artifact.filename
    target.write_bytes(b"weights")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("Unexpected network"))

    result = download_artifact(artifact=artifact, directory=tmp_path, emit=lambda _event: None)

    assert result == target


def test_corrupt_cache_is_not_loaded_or_silently_replaced(artifact: Artifact, tmp_path: Path) -> None:
    target = tmp_path / artifact.filename
    target.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="SHA-256"):
        download_artifact(artifact=artifact, directory=tmp_path, emit=lambda _event: None)

    assert target.read_bytes() == b"corrupt"


def test_download_is_atomic_and_checksum_verified(
    artifact: Artifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response(io.BytesIO):
        headers: ClassVar[dict[str, str]] = {"Content-Length": "7"}

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response(b"weights"))
    events: list[dict[str, object]] = []

    result = download_artifact(artifact=artifact, directory=tmp_path, emit=events.append)

    verify_artifact(result, artifact)
    assert not list(tmp_path.glob("*.part"))
    assert events[0]["bytes"] == len(b"weights")


def test_bad_download_is_not_published(
    artifact: Artifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response(io.BytesIO):
        headers: ClassVar[dict[str, str]] = {}

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response(b"bad"))

    with pytest.raises(ValueError, match="SHA-256"):
        download_artifact(artifact=artifact, directory=tmp_path, emit=lambda _event: None)

    assert not list(tmp_path.iterdir())


def test_unverified_weights_require_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="ViperX"):
        check_license(MODELS["viperx"], accept_unverified=False)

    check_license(MODELS["kimberley"], accept_unverified=False)
    check_license(MODELS["viperx"], accept_unverified=True)


@pytest.mark.parametrize("scheme", ["http", "file", "ftp"])
def test_download_rejects_non_https_urls_before_network_access(
    artifact: Artifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
) -> None:
    unsafe = replace(artifact, url=f"{scheme}://example.invalid/model")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("Unexpected network"))

    with pytest.raises(ValueError, match="HTTPS"):
        download_artifact(artifact=unsafe, directory=tmp_path, emit=lambda _event: None)

    assert not list(tmp_path.iterdir())
