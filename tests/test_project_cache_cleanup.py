import json
import threading

import pytest

from asmr_dubber.audio import sha256_file
from asmr_dubber.cache_cleanup import clean_caches, scan_caches
from asmr_dubber.errors import ProjectError
from asmr_dubber.models import AudioInfo, DubProject, Sentence, save_project
from asmr_dubber.storage import exclusive_file_lock


def fixture_project(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "source.wav").write_bytes(b"original")
    (root / "voice.wav").write_bytes(b"voice")
    project = DubProject(
        source=AudioInfo(
            path="source.wav", sha256="source", duration_seconds=1, sample_rate=16000, channels=2
        ),
        sentences=[
            Sentence(
                id="s1",
                start_seconds=0,
                end_seconds=1,
                source_text="original",
                zh_text="translated",
                tts_file="voice.wav",
            )
        ],
    )
    save_project(project, root)
    cache = root / "analysis/separation/abc"
    cache.mkdir(parents=True)
    for name in (
        "source.wav",
        "chunk-000000.wav",
        "chunk-000000.sha256",
        "vocals.wav",
        "background.wav",
    ):
        (cache / name).write_bytes(name.encode())
    (cache / "complete.json").write_text(
        json.dumps({name: sha256_file(cache / name) for name in ("vocals.wav", "background.wav")})
    )
    (root / "analysis/spatial-reference.wav").write_bytes(b"spatial")
    return project, cache


def test_safe_scan_and_cleanup_preserve_user_results(tmp_path):
    _, cache = fixture_project(tmp_path / "p")
    root = cache.parents[2]
    before = (root / "project.json").read_bytes()
    plan = scan_caches(str(tmp_path), ["safe"])
    assert (cache / "chunk-000000.wav").exists()
    assert len(plan["projects"][0]["files"]) == 3
    with pytest.raises(ProjectError, match="确认"):
        clean_caches(plan, [str(root)], False)
    size, skipped = clean_caches(plan, [str(root)], True)
    assert size > 0 and not skipped
    assert not (cache / "source.wav").exists()
    assert (cache / "vocals.wav").exists()
    assert (root / "analysis/spatial-reference.wav").exists()
    assert (root / "source.wav").read_bytes() == b"original"
    assert (root / "voice.wav").read_bytes() == b"voice"
    assert (root / "project.json").read_bytes() == before


def test_failed_and_corrupt_separation_keep_checkpoints(tmp_path):
    _, cache = fixture_project(tmp_path)
    (cache / "vocals.wav").write_bytes(b"corrupt")
    assert not scan_caches(str(tmp_path), ["safe", "separation"])["projects"][0]["files"]
    (cache / "complete.json").unlink()
    assert not scan_caches(str(tmp_path), ["safe"])["projects"][0]["files"]


def test_changed_files_and_new_references_are_protected(tmp_path):
    project, cache = fixture_project(tmp_path)
    plan = scan_caches(str(tmp_path), ["safe", "rebuild"])
    (cache / "source.wav").write_bytes(b"changed since scan")
    project.settings.tts_external_reference_audio = "analysis/spatial-reference.wav"
    save_project(project, tmp_path)
    _, skipped = clean_caches(plan, [str(tmp_path)], True)
    assert skipped
    assert (cache / "source.wav").exists()
    assert (tmp_path / "analysis/spatial-reference.wav").exists()


def test_rebuild_and_separation_are_explicit(tmp_path):
    _, cache = fixture_project(tmp_path)
    plan = scan_caches(str(tmp_path), ["rebuild", "separation"])
    size, skipped = clean_caches(plan, [str(tmp_path)], True)
    assert size and not skipped
    assert not (cache / "vocals.wav").exists()
    assert not (cache / "complete.json").exists()
    assert not (tmp_path / "analysis/spatial-reference.wav").exists()


def test_active_project_skipped_at_scan_and_delete(tmp_path):
    fixture_project(tmp_path)
    plan = scan_caches(str(tmp_path), ["safe"])
    started, stop = threading.Event(), threading.Event()

    def lock():
        with exclusive_file_lock(tmp_path / ".project.lock"):
            started.set()
            stop.wait(5)

    worker = threading.Thread(target=lock)
    worker.start()
    assert started.wait(3)
    try:
        assert scan_caches(str(tmp_path), ["safe"])["skipped"]
        size, skipped = clean_caches(plan, [str(tmp_path)], True)
        assert size == 0 and skipped
    finally:
        stop.set()
        worker.join()


def test_symlink_cache_is_never_followed(tmp_path):
    _, cache = fixture_project(tmp_path / "project")
    victim = tmp_path / "outside.wav"
    victim.write_bytes(b"preserve")
    target = cache / "source.wav"
    target.unlink()
    try:
        target.symlink_to(victim)
    except OSError:
        pytest.skip("symlink privileges unavailable")
    plan = scan_caches(str(tmp_path), ["safe"])
    clean_caches(plan, [str(tmp_path / "project")], True)
    assert victim.read_bytes() == b"preserve"
    assert target.is_symlink()
