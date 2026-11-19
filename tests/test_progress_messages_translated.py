"""Every Chinese progress message has an English entry or template for the English UI."""

from __future__ import annotations

import ast
import json
import re
from importlib.resources import files

CJK = re.compile(r"[一-鿿]")
CALLS = {"progress", "report", "_progress", "emit_progress"}


def _catalog() -> tuple[dict[str, str], list[re.Pattern[str]]]:
    locales = files("asmr_dubber").joinpath("locales")
    catalog: dict[str, str] = {}
    for name in ("controls.en.json", "en.json", "native.en.json"):
        catalog.update(json.loads(locales.joinpath(name).read_text(encoding="utf-8")))
    # Same matching as the page: each {name} matches any text, the rest literally.
    templates = [
        re.compile("^" + re.sub(r"\\\{\w+\\\}", "(.+?)", re.escape(key)) + "$")
        for key in catalog
        if re.search(r"\{\w+\}", key)
    ]
    return catalog, templates


def _probe(node: ast.expr) -> str | None:
    """The literal text, with every f-string field replaced by a sample value."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(part.value) if isinstance(part, ast.Constant) else "X" for part in node.values
        )
    return None


def test_every_chinese_progress_message_has_an_english_entry() -> None:
    catalog, templates = _catalog()
    missing = []
    for path in files("asmr_dubber").iterdir():
        stack = [path]
        while stack:
            item = stack.pop()
            if item.is_dir():
                stack.extend(item.iterdir())
                continue
            if not item.name.endswith(".py"):
                continue
            tree = ast.parse(item.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", "")
                if name not in CALLS:
                    continue
                text = _probe(node.args[0])
                if not text or not CJK.search(text):
                    continue
                if text in catalog or any(template.match(text) for template in templates):
                    continue
                missing.append(f"{item.name}:{node.lineno}: {text}")
    assert not missing, "\n".join(sorted(missing))
