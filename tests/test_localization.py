import re
from pathlib import Path

from asmr_dubber.localization import catalog
from asmr_dubber.services import parameters

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_has_all_static_primary_control_labels():
    translations = catalog()
    html = (ROOT / "src/asmr_dubber/frontend/index.html").read_text(encoding="utf-8")
    labels = re.findall(r"data-i18n>([^<]+)</", html)
    labels += [item["label"] for item in parameters.catalog()]
    assert all(
        label in translations for label in labels if any("\u4e00" <= c <= "\u9fff" for c in label)
    )
    assert all(isinstance(text, str) and text.strip() for text in translations.values())


def test_language_script_has_local_default_and_no_network_translation():
    script = (ROOT / "src/asmr_dubber/frontend/session.js").read_text(encoding="utf-8")
    assert "state.boot?.locales" in script
    assert "state.boot?.labels" in script
    assert "document.documentElement.lang" in script
    assert "translate.googleapis" not in script
    assert "querySelectorAll" in script
    assert "data-i18n" in script


def test_all_public_markdown_pages_have_language_pairs():
    for source in [*ROOT.glob("*.md"), *ROOT.joinpath("docs").glob("*.md")]:
        if source.name.endswith(".en.md"):
            continue
        translated = (
            source.with_name(source.stem + ".en.md")
            if source.parent == ROOT
            else source.parent / "en" / source.name
        )
        assert translated.is_file(), source
        assert "English" in source.read_text(encoding="utf-8").splitlines()[0]
        assert "中文" in translated.read_text(encoding="utf-8").splitlines()[0]


def test_prompts_are_not_documentation_navigation():
    for prompt in ROOT.joinpath("src/asmr_dubber/prompts").glob("*.md"):
        assert "[English]" not in prompt.read_text(encoding="utf-8")
