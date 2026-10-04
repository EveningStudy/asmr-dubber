import json
import os

from asmr_dubber.cache_cleanup import clean_caches, scan_caches
from tests.test_project_cache_cleanup import fixture_project


def test_browser_json_roundtrip_keeps_cleanup_identity_and_preserves_results(tmp_path):
    fixture_project(tmp_path)
    cached = tmp_path / "analysis/spatial-reference.wav"
    os.utime(cached, ns=(1791140000000000100, 1791140000000000100))
    before = {
        name: (tmp_path / name).read_bytes() for name in ("project.json", "voice.wav", "source.wav")
    }
    plan = scan_caches(str(tmp_path), ["rebuild"])
    browser_plan = json.loads(json.dumps(plan), parse_int=lambda value: int(float(value)))
    removed, skipped = clean_caches(browser_plan, [str(tmp_path)], True)
    assert removed == len(b"spatial")
    assert not skipped
    assert not cached.exists()
    assert all((tmp_path / name).read_bytes() == value for name, value in before.items())
