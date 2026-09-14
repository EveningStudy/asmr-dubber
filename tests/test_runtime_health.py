import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from asmr_dubber import runtime_health as health


@pytest.mark.parametrize("code", ["3221225781", "-1073741515", "0xC0000135"])
def test_missing_dll_explained_without_guessing(code):
    result = health.explain_runtime_error(f"退出码 {code}")
    assert "找不到所需 DLL" in result
    assert "不能确定具体 DLL" in result
    assert "不会自动安装" in result


def test_unknown_error_preserved():
    assert health.explain_runtime_error("网络请求失败") == "网络请求失败"
    assert "不能仅据此判定显存不足" in health.explain_runtime_error("3221225477")
    filename_error = "文件 3221225781.wav 不存在"
    assert health.explain_runtime_error(filename_error) == filename_error
    explained = health.explain_runtime_error("退出码 3221225781")
    assert health.explain_runtime_error(explained) == explained


def test_missing_backend_does_not_install_or_touch_projects(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "portable_home", lambda: tmp_path)
    project = tmp_path / "projects" / "project.json"
    project.parent.mkdir()
    project.write_text("preserve")
    result = health.check_runtime_health("parakeet_nemo")
    assert "未安装" in result
    assert project.read_text() == "preserve"
    assert not (tmp_path / "models").exists()


def test_probe_success_and_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "isolated_runtime_environment", lambda _: {})
    monkeypatch.setattr(
        health.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=3221225781, stdout="", stderr=""),
    )
    assert "0xC0000135" in health._probe(["test"], cwd=tmp_path, runtime="test")
    monkeypatch.setattr(
        health.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="OK", stderr=""),
    )
    assert "不代表模型推理" in health._probe(["test"], cwd=tmp_path, runtime="test")


def test_probe_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "isolated_runtime_environment", lambda _: {})

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("test", 60)

    monkeypatch.setattr(health.subprocess, "run", timeout)
    assert "超过 60 秒" in health._probe(["test"], cwd=tmp_path, runtime="test")


def test_repair_requires_confirmation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not execute")

    monkeypatch.setattr(health.subprocess, "run", forbidden)
    assert "尚未执行" in health.repair_runtime_health("parakeet_nemo", "vc", False)
    with pytest.raises(ValueError):
        health.repair_runtime_health("unknown", "backend", True)


def test_backend_repair_rechecks(monkeypatch):
    from asmr_dubber import runtime_manager

    calls = []
    monkeypatch.setattr(
        runtime_manager, "install_backend", lambda b, **k: calls.append((b, k)) or "installed"
    )
    monkeypatch.setattr(health, "check_runtime_health", lambda b: "复检未通过")
    result = health.repair_runtime_health("parakeet_nemo", "backend", True)
    assert "复检未通过" in result
    assert calls == [("parakeet_nemo", {"force": True})]


def test_official_installer_requires_signature_before_execution():
    script = Path(__file__).parents[1] / "scripts/windows/repair-vc-runtime.ps1"
    text = script.read_text(encoding="utf-8-sig")
    assert text.index("Get-AuthenticodeSignature") < text.index("Start-Process")
    assert "Microsoft Corporation" in text
    assert "--proto-redir '=https'" in text
    assert "/norestart" in text
    assert "Remove-Item" not in text


@pytest.mark.skipif(health.os.name != "nt", reason="Windows installer flow")
@pytest.mark.parametrize("exit_code", [0, 1])
def test_vc_installer_result_and_recheck(tmp_path, monkeypatch, exit_code):
    monkeypatch.setattr(health, "portable_home", lambda: tmp_path)
    calls = []

    def run(command, **kwargs):
        assert "repair-vc-runtime.ps1" in command[-1]
        calls.append(command)
        return SimpleNamespace(returncode=exit_code, stdout="installer result", stderr="")

    monkeypatch.setattr(health.subprocess, "run", run)
    monkeypatch.setattr(health, "check_runtime_health", lambda b: "checked backend")
    result = health.repair_runtime_health("parakeet_nemo", "vc", True)
    assert len(calls) == 1
    assert ("checked backend" in result) == (exit_code == 0)
    assert ("修复未完成" in result) == (exit_code != 0)
