"""Browser-local UI language. Processing values and user content remain untouched."""

import json
from importlib.resources import files


def language_script() -> str:
    root = files("asmr_dubber").joinpath("locales")
    catalog = json.loads(root.joinpath("en.json").read_text(encoding="utf-8"))
    script = root.joinpath("switch.js").read_text(encoding="utf-8")
    return script.replace("__CATALOG__", json.dumps(catalog, ensure_ascii=True))
