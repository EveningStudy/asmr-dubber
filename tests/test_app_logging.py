import logging
from pathlib import Path

from asmr_dubber import app_logging


def test_application_log_is_portable_rotating_and_redacted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_logging, "portable_home", lambda: tmp_path)
    monkeypatch.setattr(app_logging, "_CONFIGURED_PATH", None)
    path = app_logging.configure_logging()

    logging.getLogger("asmr_dubber.test").error(
        "api_key=private-value Authorization: Bearer private-bearer "
        "ms-11111111-2222-3333-4444-555555555555 sk-private0123456789"
    )
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = path.read_text(encoding="utf-8")
    assert path == tmp_path / "logs" / "asmr-dubber.log"
    assert "private-value" not in content
    assert "private-bearer" not in content
    assert "ms-11111111" not in content
    assert "sk-private" not in content
    assert content.count("[已隐藏]") >= 4
