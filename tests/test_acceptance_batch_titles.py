from types import SimpleNamespace

from asmr_dubber.autoflow import engine
from asmr_dubber.user_settings import UserSettings


def test_empty_spoken_translation_of_latin_filename_preserves_title(tmp_path, monkeypatch):
    monkeypatch.setattr("asmr_dubber.user_settings.load_user_settings", UserSettings)
    monkeypatch.setattr("asmr_dubber.user_settings.resolve_api_key", lambda _: "test-key")

    def translate(sentences, **_):
        sentences[0].zh_text = "作品"
        sentences[1].zh_text = ""

    monkeypatch.setattr("asmr_dubber.translation.translate_sentences", translate)
    state = {
        "source_folder": str(tmp_path / "work"),
        "timeline": [
            {"filename": "01-original.wav", "title_ja": "original", "source_language": "ja"}
        ],
    }
    assert engine.translate_titles(state, SimpleNamespace()) == {"01-original.wav": "original"}
    assert state["title_translations"] == {"01-original.wav": "original"}
