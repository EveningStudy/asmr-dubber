import ast
import json
from pathlib import Path

from asmr_dubber.localization import language_script

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_has_all_static_primary_control_labels():
    catalog = json.loads((ROOT / "src/asmr_dubber/locales/en.json").read_text(encoding="utf-8"))
    missing = set()
    for filename in ("ui.py", "separation_ui.py"):
        tree = ast.parse((ROOT / "src/asmr_dubber" / filename).read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            values = [k.value for k in call.keywords if k.arg in {"label", "placeholder"}]
            if isinstance(call.func, ast.Attribute) and call.func.attr in {
                "Button",
                "Tab",
                "Accordion",
                "Checkbox",
            }:
                values += call.args[:1]
            if isinstance(call.func, ast.Name) and call.func.id in {
                "number",
                "text",
                "radio",
                "checkbox",
            }:
                values += call.args[1:2]
            for value in values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    text = value.value
                    if any("\u4e00" <= char <= "\u9fff" for char in text) and text not in catalog:
                        missing.add(text)
    assert not missing
    assert all(isinstance(text, str) and text.strip() for text in catalog.values())


def test_language_script_has_local_default_and_no_network_translation():
    script = language_script()
    assert "__CATALOG__" not in script
    assert "let language = 'zh'" in script
    assert "localStorage" in script
    assert "fetch(" not in script
    assert "contenteditable" in script
    assert "data-no-i18n" in script


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
