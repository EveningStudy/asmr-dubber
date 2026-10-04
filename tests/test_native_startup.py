from pathlib import Path

from asmr_dubber import ui

ROOT = Path(__file__).resolve().parents[1]


def test_native_startup_binds_requested_address_and_closes_server(monkeypatch):
    events = []

    class Server:
        def __init__(self, address):
            events.append(address)

        def __enter__(self):
            return self

        def serve_forever(self):
            events.append("serving")
            raise KeyboardInterrupt

        def __exit__(self, *args):
            events.append("closed")

    monkeypatch.setattr(ui, "Server", Server)
    monkeypatch.setattr(ui, "configure_logging", lambda: None)
    monkeypatch.setattr(ui, "require_supported_platform", lambda: None)
    ui.launch("127.0.0.1", 7870)
    assert events == [("127.0.0.1", 7870), "serving", "closed"]


def test_launcher_prepares_only_core_and_portable_builder_excludes_models():
    source = (ROOT / "launcher/windows/ASMRDubberLauncher.cs").read_text(encoding="utf-8")
    assert '"-Profile Core"' in source
    assert "import asmr_dubber.http_server, av, soundfile" in source
    assert "gradio" not in source
    assert "open_browser" in source
    builder = (ROOT / "scripts/windows/create-portable.ps1").read_text(encoding="utf-8-sig")
    assert "-Profile Core" in builder
    assert "frontend/index.html" in builder
    assert "基础包不应包含模型权重" in builder
    assert "secrets.json" not in builder
    for name in ("ASMRDubberLauncher.cs", "LauncherProcess.cs"):
        source = (ROOT / "launcher/windows" / name).read_text(encoding="utf-8")
        assert len(source.splitlines()) < 500


def test_released_frontend_assets_are_available_without_optional_interface_packages():
    import tomllib
    from importlib.resources import files

    assets = files("asmr_dubber").joinpath("frontend")
    for name in ("index.html", "styles.css", "app.js", "forms.js", "projects.js", "batch.js"):
        assert assets.joinpath(name).is_file()
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "ui" not in metadata["project"]["optional-dependencies"]
