import json
import re
from importlib.resources import files

from asmr_dubber.localization import catalog
from asmr_dubber.services.parameters import catalog as parameters


def test_frontend_pages_and_generated_parameter_panels_cover_every_setting():
    root = files("asmr_dubber").joinpath("frontend")
    html = root.joinpath("index.html").read_text(encoding="utf-8")
    for page in ("home", "project", "batch", "models", "settings"):
        assert f'id="{page}"' in html
    panels = set()
    for value in re.findall(r'data-panels="([^"]+)"', html):
        panels.update(value.split(","))
    assert all(parameter["panel"] in panels for parameter in parameters())
    assert '__SESSION_TOKEN__' in html
    assert '<script type="module" src="app.js">' in html
    assert "gradio" not in html.lower()


def test_native_static_and_parameter_labels_have_english_translations():
    root = files("asmr_dubber")
    html = root.joinpath("frontend/index.html").read_text(encoding="utf-8")
    translations = catalog()
    labels = re.findall(r'data-i18n>([^<]+)</', html)
    labels.extend(parameter["label"] for parameter in parameters())
    labels.extend(parameter["group"] for parameter in parameters())
    missing = {label for label in labels if re.search(r'[\u4e00-\u9fff]', label)
               and label not in translations}
    assert not missing
    chinese = catalog("zh")
    assert chinese["running"] == "正在处理…"
    assert chinese["ja"] == "日语"


def test_frontend_modules_and_locale_resources_stay_below_file_limit():
    root = files("asmr_dubber")
    for folder in ("frontend", "services"):
        for path in root.joinpath(folder).iterdir():
            if path.is_file():
                assert len(path.read_text(encoding="utf-8").splitlines()) <= 500, path
    for name in ("native.en.json", "native.zh.json"):
        resource = root.joinpath("locales", name)
        assert len(resource.read_text(encoding="utf-8").splitlines()) <= 500
        assert isinstance(json.loads(resource.read_text(encoding="utf-8")), dict)
