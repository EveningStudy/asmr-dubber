"""Browser-local UI language. Processing values and user content remain untouched."""

import json
from importlib.resources import files


def catalog(language: str = "en") -> dict[str, str]:
    root = files("asmr_dubber").joinpath("locales")
    if language == "zh":
        return json.loads(root.joinpath("native.zh.json").read_text(encoding="utf-8"))
    result = json.loads(root.joinpath("en.json").read_text(encoding="utf-8"))
    result.update(json.loads(root.joinpath("native.en.json").read_text(encoding="utf-8")))
    return result


def language_script() -> str:
    root = files("asmr_dubber").joinpath("locales")
    translations = catalog()
    script = root.joinpath("switch.js").read_text(encoding="utf-8")
    return script.replace("__CATALOG__", json.dumps(translations, ensure_ascii=True))
