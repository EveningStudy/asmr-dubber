"""One-click install of anime-whisper / Qwen3-ASR and the default-recognizer fallback."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from asmr_dubber.constants import (
    ASMR_VAD_MODEL,
    DEFAULT_ALIGNER_MODEL,
    FALLBACK_ASR_BACKEND,
    FALLBACK_ASR_MODEL,
    OPTIONAL_ASR_MODEL_REVISIONS,
    PROJECT_ROOT,
    RECOMMENDED_ASR_BACKEND,
    RECOMMENDED_ASR_MODEL,
)
from asmr_dubber.environment import _complete_model_snapshot
from asmr_dubber.errors import EnvironmentError as AppEnvironmentError
from asmr_dubber.model_registry import ASR_BACKENDS
from asmr_dubber.models import ProjectSettings, settings_for_source_language
from asmr_dubber.platforms import PlatformInfo
from asmr_dubber.runtime_manager import (
    _BACKEND_MODEL_REPOS,
    BackendStatus,
    asr_default_fallback,
    backend_model_pack_ids,
    backend_status,
    download_backend_models,
    install_backend,
    ordered_backends,
)

NEW_BACKENDS = ("anime_whisper", "qwen3_asr")
FINE_TUNE = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"


@pytest.fixture(autouse=True)
def _no_network_or_packs(monkeypatch, tmp_path: Path) -> None:
    # install_backend locks the portable home; never contend with a real install.
    monkeypatch.setattr("asmr_dubber.runtime_manager.portable_home", lambda: tmp_path)
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.prepare_remote_model_pack",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.snapshot_download_with_fallback",
        lambda **_kwargs: pytest.fail("test must not download"),
    )


def _windows() -> PlatformInfo:
    return PlatformInfo("Windows", "AMD64", True, False, False)


def _linux() -> PlatformInfo:
    return PlatformInfo("Linux", "x86_64", False, False, False)


# --- registry and pins ------------------------------------------------------


@pytest.mark.parametrize("backend_id", NEW_BACKENDS)
def test_new_backends_are_installable_with_existing_python_extras(backend_id: str) -> None:
    spec = ASR_BACKENDS[backend_id]
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert spec.installer == "python-extra"
    assert spec.python_extra
    for extra in spec.python_extra.split(","):
        assert extra in extras
    assert backend_id in _BACKEND_MODEL_REPOS
    assert set(spec.models) <= set(_BACKEND_MODEL_REPOS[backend_id])


@pytest.mark.parametrize("backend_id", NEW_BACKENDS)
def test_every_downloaded_model_is_pinned_to_a_revision(backend_id: str) -> None:
    for repository in _BACKEND_MODEL_REPOS[backend_id]:
        assert len(OPTIONAL_ASR_MODEL_REVISIONS[repository]) == 40
    # Both recognizers need the ASMR VAD and the Qwen3 aligner for Japanese.
    assert ASMR_VAD_MODEL in _BACKEND_MODEL_REPOS[backend_id]
    assert DEFAULT_ALIGNER_MODEL in _BACKEND_MODEL_REPOS[backend_id]


def test_anime_whisper_is_the_recommended_default_and_qwen_stays_experimental() -> None:
    assert (RECOMMENDED_ASR_BACKEND, RECOMMENDED_ASR_MODEL) == (
        "anime_whisper",
        "litagin/anime-whisper",
    )
    assert ASR_BACKENDS["anime_whisper"].tested_default is True
    assert ASR_BACKENDS["anime_whisper"].support_level == "supported"
    assert ASR_BACKENDS["qwen3_asr"].tested_default is False
    assert ASR_BACKENDS["qwen3_asr"].support_level == "experimental"
    assert ProjectSettings().asr_backend == "anime_whisper"
    assert ProjectSettings().asr_model == "litagin/anime-whisper"


def test_offline_pack_ids_cover_vad_and_aligner_for_both_backends() -> None:
    for backend_id in NEW_BACKENDS:
        packs = backend_model_pack_ids(backend_id)
        assert {"whisper-vad-asmr-onnx", "qwen3-forced-aligner"} <= packs


# --- status -----------------------------------------------------------------


@pytest.mark.parametrize("backend_id", NEW_BACKENDS)
def test_status_needs_every_model_not_only_the_runtime(monkeypatch, backend_id: str) -> None:
    monkeypatch.setattr("asmr_dubber.runtime_manager.current_platform", _linux)
    monkeypatch.setattr("asmr_dubber.runtime_manager._imports_available", lambda _backend: True)
    present = {_BACKEND_MODEL_REPOS[backend_id][0]}
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.cached_model_path",
        lambda model_id: Path("snapshot") if model_id in present else None,
    )

    partial = backend_status(ASR_BACKENDS[backend_id])
    present.update(_BACKEND_MODEL_REPOS[backend_id])
    ready = backend_status(ASR_BACKENDS[backend_id])

    assert partial.state == "partial"
    assert ASMR_VAD_MODEL in partial.detail
    assert ready.state == "ready"


def test_status_is_missing_without_the_python_runtime(monkeypatch) -> None:
    monkeypatch.setattr("asmr_dubber.runtime_manager.current_platform", _linux)
    monkeypatch.setattr("asmr_dubber.runtime_manager._imports_available", lambda _backend: False)

    assert backend_status(ASR_BACKENDS["anime_whisper"]).state == "missing"


# --- install flow -----------------------------------------------------------


@pytest.mark.parametrize("backend_id", NEW_BACKENDS)
def test_windows_install_uses_the_script_then_imports_packs(
    monkeypatch, tmp_path: Path, backend_id: str
) -> None:
    uv = tmp_path / "uv.exe"
    uv.touch()
    order: list[str] = []
    monkeypatch.setattr("asmr_dubber.runtime_manager._uv_executable", lambda: uv)
    monkeypatch.setattr("asmr_dubber.runtime_manager.current_platform", _windows)
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager._install_windows_backend",
        lambda backend, **_kwargs: (
            order.append(f"dependencies:{backend}")
            or CompletedProcess(["pwsh"], 0, stdout="runtime ready", stderr="")
        ),
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.import_backend_model_packs",
        lambda backend, **_kwargs: order.append(f"packs:{backend}") or 0,
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.download_backend_models",
        lambda backend, **_kwargs: order.append(f"models:{backend}") or "models ready",
    )

    result = install_backend(backend_id, force=True)

    assert order == [
        f"dependencies:{backend_id}",
        f"packs:{backend_id}",
        f"models:{backend_id}",
    ]
    assert "安装完成" in result and "models ready" in result


def test_windows_install_script_accepts_the_new_backends() -> None:
    script = (PROJECT_ROOT / "scripts" / "windows" / "install-backend.ps1").read_text(
        encoding="utf-8-sig"
    )

    for backend_id in NEW_BACKENDS:
        assert f'"{backend_id}"' in script
        for extra in ASR_BACKENDS[backend_id].python_extra.split(","):  # type: ignore[union-attr]
            assert extra in script


@pytest.mark.parametrize("backend_id", NEW_BACKENDS)
def test_linux_install_requests_all_required_extras(
    monkeypatch, tmp_path: Path, backend_id: str
) -> None:
    uv = tmp_path / "uv"
    uv.touch()
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return CompletedProcess(command, 0, stdout="installed", stderr="")

    monkeypatch.setattr("asmr_dubber.runtime_manager._uv_executable", lambda: uv)
    monkeypatch.setattr("asmr_dubber.runtime_manager.current_platform", _linux)
    monkeypatch.setattr("asmr_dubber.runtime_manager._run_streaming_process", fake_run)
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.download_candidates",
        lambda *_args, **_kwargs: ("https://index.invalid/simple",),
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.import_backend_model_packs",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.download_backend_models",
        lambda *_args, **_kwargs: "models ready",
    )

    install_backend(backend_id, force=True)

    target = next(arg for arg in commands[0] if "[" in arg)
    assert target.endswith(f"[{ASR_BACKENDS[backend_id].python_extra}]")


# --- model download ---------------------------------------------------------


def _record_downloads(monkeypatch, tmp_path: Path, *, present: set[str] | None = None):
    downloaded: list[dict] = []
    available = set(present or ())

    def fake_download(**kwargs):
        downloaded.append(kwargs)
        available.add(kwargs["repo_id"])
        return str(tmp_path / kwargs["repo_id"].replace("/", "--"))

    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.snapshot_download_with_fallback", fake_download
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.cached_model_path",
        lambda model_id: tmp_path / model_id.replace("/", "--") if model_id in available else None,
    )
    monkeypatch.setattr("asmr_dubber.user_settings.load_user_settings", lambda: UserStub())
    return downloaded


class UserStub:
    huggingface_endpoint = ""


@pytest.mark.parametrize("backend_id", NEW_BACKENDS)
def test_download_fetches_each_model_at_its_pinned_revision(
    monkeypatch, tmp_path: Path, backend_id: str
) -> None:
    downloaded = _record_downloads(monkeypatch, tmp_path)

    result = download_backend_models(backend_id)

    assert [item["repo_id"] for item in downloaded] == list(_BACKEND_MODEL_REPOS[backend_id])
    for item in downloaded:
        assert item["revision"] == OPTIONAL_ASR_MODEL_REVISIONS[item["repo_id"]]
    assert "模型准备完成" in result


def test_fine_tune_download_excludes_training_state(monkeypatch, tmp_path: Path) -> None:
    downloaded = _record_downloads(monkeypatch, tmp_path)

    download_backend_models("qwen3_asr")

    ignore = next(item["ignore_patterns"] for item in downloaded if item["repo_id"] == FINE_TUNE)
    assert {"optimizer.pt", "scheduler.pt", "training_args.bin"} <= set(ignore)
    # The ordinary models are fetched whole.
    others = [item for item in downloaded if item["repo_id"] != FINE_TUNE]
    assert all("optimizer.pt" not in item.get("ignore_patterns", ()) for item in others)


def test_download_reuses_models_that_are_already_complete(monkeypatch, tmp_path: Path) -> None:
    present = set(_BACKEND_MODEL_REPOS["anime_whisper"]) - {"litagin/anime-whisper"}
    downloaded = _record_downloads(monkeypatch, tmp_path, present=present)
    logs: list[str] = []

    download_backend_models("anime_whisper", log_callback=logs.append)

    assert [item["repo_id"] for item in downloaded] == ["litagin/anime-whisper"]
    assert sum("复用已导入或已缓存的模型" in message for message in logs) == 2


def test_missing_mirror_error_names_the_download_source_setting(
    monkeypatch, tmp_path: Path
) -> None:
    present = {"Qwen/Qwen3-ASR-1.7B", DEFAULT_ALIGNER_MODEL, ASMR_VAD_MODEL}
    _record_downloads(monkeypatch, tmp_path, present=present)

    def refuse(**_kwargs):
        raise RuntimeError("未启用海外模型源")

    monkeypatch.setattr("asmr_dubber.runtime_manager.snapshot_download_with_fallback", refuse)
    monkeypatch.setattr("asmr_dubber.runtime_manager.external_downloads_allowed", lambda: False)

    with pytest.raises(AppEnvironmentError, match="海外源"):
        download_backend_models("qwen3_asr")


def test_models_with_a_modelscope_pack_do_not_get_the_mirror_hint(
    monkeypatch, tmp_path: Path
) -> None:
    present = {DEFAULT_ALIGNER_MODEL, ASMR_VAD_MODEL}
    _record_downloads(monkeypatch, tmp_path, present=present)

    def refuse(**_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("asmr_dubber.runtime_manager.snapshot_download_with_fallback", refuse)
    monkeypatch.setattr("asmr_dubber.runtime_manager.external_downloads_allowed", lambda: False)

    with pytest.raises(AppEnvironmentError) as caught:
        download_backend_models("anime_whisper")

    assert "litagin/anime-whisper" in str(caught.value)
    assert "国内镜像" not in str(caught.value)


def test_sharded_checkpoint_is_incomplete_until_every_shard_exists(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"x")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    assert _complete_model_snapshot("Qwen/Qwen3-ASR-1.7B", snapshot) is None
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"x")
    assert _complete_model_snapshot("Qwen/Qwen3-ASR-1.7B", snapshot) == snapshot.resolve()


# --- default recognizer fallback --------------------------------------------


def _recommended_installed(monkeypatch, installed: bool) -> None:
    monkeypatch.setattr("asmr_dubber.runtime_manager.current_platform", _linux)
    monkeypatch.setattr("asmr_dubber.runtime_manager._imports_available", lambda _backend: True)
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.cached_model_path",
        lambda _model_id: Path("snapshot") if installed else None,
    )
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager._local_huggingface_snapshot",
        lambda _model_id: Path("snapshot") if installed else None,
    )


def test_uninstalled_default_falls_back_to_parakeet(monkeypatch) -> None:
    _recommended_installed(monkeypatch, False)

    assert asr_default_fallback(RECOMMENDED_ASR_BACKEND, RECOMMENDED_ASR_MODEL) == (
        FALLBACK_ASR_BACKEND,
        FALLBACK_ASR_MODEL,
    )
    assert FALLBACK_ASR_BACKEND == "parakeet_nemo"


def test_installed_default_is_kept(monkeypatch) -> None:
    _recommended_installed(monkeypatch, True)

    assert asr_default_fallback(RECOMMENDED_ASR_BACKEND, RECOMMENDED_ASR_MODEL) is None


def test_explicit_choices_are_never_replaced(monkeypatch) -> None:
    _recommended_installed(monkeypatch, False)

    assert asr_default_fallback(FALLBACK_ASR_BACKEND, FALLBACK_ASR_MODEL) is None
    assert asr_default_fallback("faster_whisper", "large-v2") is None
    assert asr_default_fallback("anime_whisper", "/custom/anime-whisper") is None


def test_saved_parakeet_settings_and_projects_are_not_migrated() -> None:
    payload = {"asr_backend": "parakeet_nemo", "asr_model": FALLBACK_ASR_MODEL}

    assert ProjectSettings.model_validate(payload).asr_backend == "parakeet_nemo"
    assert ProjectSettings.model_validate(payload).asr_model == FALLBACK_ASR_MODEL


def test_english_source_never_uses_the_japanese_only_default() -> None:
    settings = settings_for_source_language(ProjectSettings(), "en")

    assert settings.asr_backend == "faster_whisper"


def test_ready_default_leads_the_backend_list(monkeypatch) -> None:
    states = {
        backend_id: BackendStatus("ready", "可用")
        if backend_id in {"parakeet_nemo", "anime_whisper"}
        else BackendStatus("missing", "未安装")
        for backend_id in ASR_BACKENDS
    }
    states["generic_asr_api"] = BackendStatus("external", "外部服务")
    monkeypatch.setattr(
        "asmr_dubber.runtime_manager.backend_status",
        lambda backend, settings=None: states[backend.id],
    )

    ordered = [backend.id for backend in ordered_backends(ASR_BACKENDS)]

    assert ordered[:2] == ["anime_whisper", "parakeet_nemo"]


def test_fresh_settings_use_parakeet_until_anime_whisper_is_installed(
    tmp_path, monkeypatch
) -> None:
    from asmr_dubber.user_settings import load_user_settings

    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(tmp_path / "config"))
    _recommended_installed(monkeypatch, False)
    assert load_user_settings().asr_backend == FALLBACK_ASR_BACKEND
    _recommended_installed(monkeypatch, True)
    assert load_user_settings().asr_backend == RECOMMENDED_ASR_BACKEND


def test_saved_choice_is_kept_even_when_not_installed(tmp_path, monkeypatch) -> None:
    import json

    from asmr_dubber.user_settings import load_user_settings

    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.json").write_text(
        json.dumps({"asr_backend": RECOMMENDED_ASR_BACKEND, "asr_model": RECOMMENDED_ASR_MODEL}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ASMR_DUBBER_CONFIG_DIR", str(config))
    _recommended_installed(monkeypatch, False)
    assert load_user_settings().asr_backend == RECOMMENDED_ASR_BACKEND
