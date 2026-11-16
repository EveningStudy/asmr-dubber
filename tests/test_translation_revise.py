"""Revising existing translations on request: proposals only, never a silent overwrite."""

import json
import shutil
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from asmr_dubber import translation as translation_module
from asmr_dubber.api import API
from asmr_dubber.api_contract import TaskRequest
from asmr_dubber.errors import ProjectError, TranslationError
from asmr_dubber.models import Sentence, load_project, save_project
from asmr_dubber.services.application import Application
from asmr_dubber.services.tasks import ACTIVE
from asmr_dubber.translation import (
    revise_translations,
    revision_candidates,
    validate_revisions,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(translation_module.time, "sleep", lambda _seconds: None)


def sentences(count=3):
    return [
        Sentence(
            id=f"s{index}",
            start_seconds=float(index),
            end_seconds=index + 0.9,
            source_text=f"原文{index}",
            zh_text=f"旧译文{index}",
            status="translated",
        )
        for index in range(1, count + 1)
    ]


def chat(content):
    return httpx.Response(
        200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    )


def answer(*revisions):
    return json.dumps({"revisions": list(revisions)}, ensure_ascii=False)


class Server:
    """Records every request and answers with ``reply(payload)``."""

    def __init__(self, reply):
        self.reply = reply
        self.payloads = []

    def __call__(self, request):
        payload = json.loads(request.content)
        self.payloads.append(payload)
        return chat(self.reply(payload))

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self))


def revise(rows, server, instruction="语气更温柔一点", **kwargs):
    return revise_translations(
        rows,
        instruction,
        "test-key",
        "test-model",
        client=server.client(),
        max_retries=1,
        **kwargs,
    )


def target_ids(payload):
    prompt = payload["messages"][-1]["content"]
    target = prompt.split("目标项（`current_zh` 是现有译文）：\n")[1].split("\n\n")[0]
    return [item["id"] for item in json.loads(target)]


def test_only_changed_sentences_come_back_and_nothing_is_modified():
    rows = sentences(3)
    server = Server(lambda _payload: answer({"id": "s2", "zh": "温柔的新译文2", "kind": "speech"}))
    report = revise(rows, server)
    assert [(p.id, p.old_zh, p.new_zh, p.kind) for p in report.proposals] == [
        ("s2", "旧译文2", "温柔的新译文2", "speech")
    ]
    assert report.candidates == 3 and report.unchanged == 2 and not report.failed
    assert [row.zh_text for row in rows] == ["旧译文1", "旧译文2", "旧译文3"]
    assert all(row.review_locked is False for row in rows)


def test_a_revision_identical_to_the_current_text_is_not_a_proposal():
    server = Server(lambda _payload: answer({"id": "s1", "zh": "旧译文1"}))
    report = revise(sentences(1), server)
    assert report.proposals == [] and report.unchanged == 1


def test_request_carries_instruction_sources_current_text_and_neighbours():
    rows = sentences(5)
    server = Server(lambda _payload: answer())
    report = revise_translations(
        rows,
        "不要出现“亲爱的”",
        "test-key",
        "test-model",
        sentence_ids=["s3"],
        client=server.client(),
        max_retries=1,
        context_sentences=5,
    )
    assert report.candidates == 1 and report.proposals == []
    payload = server.payloads[0]
    system = payload["messages"][0]["content"]
    assert "按要求改译" in system and "revisions" in system
    texts = "\n".join(message["content"] for message in payload["messages"][1:])
    assert "不要出现“亲爱的”" in texts
    assert '"id":"s3","source":"原文3","current_zh":"旧译文3"' in texts
    assert "原文2" in texts and "原文4" in texts  # neighbours as context
    assert target_ids(payload) == ["s3"]


def test_batches_hold_at_most_sixteen_sentences():
    rows = sentences(20)
    server = Server(lambda _payload: answer())
    report = revise(rows, server)
    assert [len(target_ids(payload)) for payload in server.payloads] == [16, 4]
    assert report.candidates == 20 and report.unchanged == 20


def test_ids_outside_the_batch_fail_validation():
    with pytest.raises(TranslationError, match="不属于本批"):
        validate_revisions(answer({"id": "x9", "zh": "好"}), ["s1", "s2"])
    with pytest.raises(TranslationError, match="重复"):
        validate_revisions(answer({"id": "s1", "zh": "好"}, {"id": "s1", "zh": "行"}), ["s1"])
    with pytest.raises(TranslationError, match="校验失败"):
        validate_revisions(answer({"id": "s1", "zh": "好", "extra": 1}), ["s1"])
    with pytest.raises(TranslationError, match="校验失败"):
        validate_revisions(answer({"id": "s1", "zh": "好", "kind": "other"}), ["s1"])
    with pytest.raises(TranslationError, match="校验失败"):
        validate_revisions(answer({"id": "s1", "zh": "", "kind": "speech"}), ["s1"])
    assert validate_revisions(answer(), ["s1"]) == {}


def test_a_model_answer_with_foreign_ids_becomes_a_recorded_failure():
    server = Server(lambda _payload: answer({"id": "x9", "zh": "好"}))
    report = revise(sentences(2), server)
    assert report.proposals == []
    assert sorted(item.sentence_id for item in report.failed) == ["s1", "s2"]
    assert "不属于本批" in report.failure_message()


def test_hand_edited_sentences_are_skipped_unless_requested():
    rows = sentences(3)
    rows[1].review_locked = True
    rows[2].enabled = False
    eligible, skipped = revision_candidates(rows)
    assert [row.id for row in eligible] == ["s1"] and skipped == 1
    eligible, skipped = revision_candidates(rows, include_edited=True)
    assert [row.id for row in eligible] == ["s1", "s2"] and skipped == 0

    server = Server(
        lambda payload: answer(*({"id": i, "zh": f"新{i}"} for i in target_ids(payload)))
    )
    default = revise(rows, server)
    assert sorted(p.id for p in default.proposals) == ["s1"] and default.skipped_edited == 1
    included = revise(rows, server, include_edited=True)
    assert sorted(p.id for p in included.proposals) == ["s1", "s2"]


def test_selected_ids_limit_the_run_and_untranslated_rows_are_never_sent():
    rows = sentences(3)
    rows[0].zh_text = ""
    server = Server(
        lambda payload: answer(*({"id": i, "zh": f"新{i}"} for i in target_ids(payload)))
    )
    report = revise(rows, server, sentence_ids=["s1", "s2"])
    assert [p.id for p in report.proposals] == ["s2"]
    assert target_ids(server.payloads[0]) == ["s2"]


def test_sound_proposal_keeps_kind_and_empty_text():
    rows = sentences(2)
    server = Server(lambda _payload: answer({"id": "s1", "zh": "", "kind": "sound"}))
    report = revise(rows, server)
    assert [(p.id, p.kind, p.new_zh) for p in report.proposals] == [("s1", "sound", "")]
    assert rows[0].zh_text == "旧译文1"


def test_failing_batch_is_split_so_good_sentences_still_get_proposals():
    def reply(payload):
        ids = target_ids(payload)
        if "s2" in ids and len(ids) > 1:
            return "not json"
        if ids == ["s2"]:
            return answer({"id": "s9", "zh": "x"})
        return answer(*({"id": i, "zh": f"新{i}"} for i in ids))

    report = revise(sentences(4), Server(reply))
    assert sorted(p.id for p in report.proposals) == ["s1", "s3", "s4"]
    assert [item.sentence_id for item in report.failed] == ["s2"]


def test_machine_translation_providers_are_refused_clearly():
    server = Server(lambda _payload: answer())
    for provider in ("deepl", "google_translate", "microsoft_translate"):
        with pytest.raises(TranslationError, match="不能按要求改译"):
            revise(sentences(1), server, provider=provider)
    assert server.payloads == []


def test_blank_instruction_is_refused():
    with pytest.raises(TranslationError, match="先写下改译要求"):
        revise(sentences(1), Server(lambda _payload: answer()), instruction="  ")


def test_credential_errors_stop_the_run():
    def reject(_request):
        return httpx.Response(401, text="bad key")

    client = httpx.Client(transport=httpx.MockTransport(reject))
    with pytest.raises(TranslationError, match="401"):
        revise_translations(sentences(2), "改", "k", "m", client=client, max_retries=1)


def test_task_request_requires_instruction_and_ids():
    base = {"kind": "revise_translations", "project": "p.json", "revision": 0}
    TaskRequest.model_validate({**base, "sentence_ids": ["s1"], "instruction": "改"})
    with pytest.raises(ValueError):
        TaskRequest.model_validate({**base, "sentence_ids": ["s1"], "instruction": " "})
    with pytest.raises(ValueError):
        TaskRequest.model_validate({**base, "sentence_ids": [], "instruction": "改"})


# --- project level: proposals are stored, then accepted or rejected through the API ---


@pytest.fixture
def application(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMR_DUBBER_HOME", str(tmp_path / "home"))
    return Application(tmp_path / "tasks.json")


def completed(application, identifier):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        task = application.tasks.get(identifier)
        if task["status"] not in ACTIVE:
            return task
        time.sleep(0.01)
    pytest.fail("Task failed to finish")


@pytest.fixture(scope="module")
def template_project(tmp_path_factory):
    """One real project, created once; every test works on its own copy."""

    root = tmp_path_factory.mktemp("template")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ASMR_DUBBER_HOME", str(root / "home"))
        application = Application(root / "tasks.json")
        audio = root / "source.wav"
        sf.write(audio, np.zeros(16000 * 10), 16000)
        task = application.start({"kind": "create", "source": str(audio), "source_language": "ja"})
        manifest = completed(application, task["id"])["result"]["manifest"]
    return Path(manifest).parent


def project_with_translations(template_project, tmp_path, count=3):
    directory = tmp_path / "project"
    shutil.copytree(template_project, directory)
    project, _ = load_project(directory / "project.json")
    project.sentences = sentences(count)
    for sentence in project.sentences:
        sentence.tts_file = f"tts/{sentence.id}.wav"
        sentence.tts_cache_key = "old"
        sentence.tts_duration_seconds = 1.0
    project.settings.translation_provider = "deepseek"
    save_project(project, directory)
    return str(directory / "project.json")


def start_revision(application, manifest, monkeypatch, reply, **extra):
    server = Server(reply)
    monkeypatch.setattr(translation_module, "logged_http_client", lambda *_a, **_k: server.client())
    monkeypatch.setattr(
        "asmr_dubber.translation_revision.resolve_api_key", lambda _provider, _supplied=None: "k"
    )
    project, _ = load_project(manifest)
    request = {
        "kind": "revise_translations",
        "project": manifest,
        "revision": project.revision,
        "sentence_ids": [s.id for s in project.sentences],
        "instruction": "语气更温柔一点",
        **extra,
    }
    task = application.start(request)
    return completed(application, task["id"]), server


def test_task_stores_proposals_and_leaves_translations_alone(
    template_project, application, tmp_path, monkeypatch
):
    manifest = project_with_translations(template_project, tmp_path)
    task, _server = start_revision(
        application,
        manifest,
        monkeypatch,
        lambda _payload: answer(
            {"id": "s1", "zh": "新译文1", "kind": "speech"}, {"id": "s3", "zh": "", "kind": "sound"}
        ),
    )
    assert task["status"] == "completed", task
    assert "已提出 2 条改译提议" in task["result"]["message"]
    project, _ = load_project(manifest)
    assert [s.zh_text for s in project.sentences] == ["旧译文1", "旧译文2", "旧译文3"]
    assert [s.tts_file for s in project.sentences] == [f"tts/s{i}.wav" for i in (1, 2, 3)]
    proposals = project.translation_proposals
    assert [(p["id"], p["old_zh"], p["new_zh"], p["kind"]) for p in proposals] == [
        ("s1", "旧译文1", "新译文1", "speech"),
        ("s3", "旧译文3", "", "sound"),
    ]
    assert all(p["instruction"] == "语气更温柔一点" and p["created_at"] for p in proposals)
    assert application.projects.get(manifest)["translation_proposals"] == proposals


def test_task_with_hand_edited_sentences_follows_the_checkbox(
    template_project, application, tmp_path, monkeypatch
):
    manifest = project_with_translations(template_project, tmp_path)
    project, directory = load_project(manifest)
    project.sentences[1].review_locked = True
    save_project(project, directory)

    def reply(payload):
        return answer(*({"id": i, "zh": f"新{i}"} for i in target_ids(payload)))

    task, _ = start_revision(application, manifest, monkeypatch, reply)
    assert task["status"] == "completed"
    ids = [p["id"] for p in load_project(manifest)[0].translation_proposals]
    assert ids == ["s1", "s3"]
    task, _ = start_revision(application, manifest, monkeypatch, reply, include_edited=True)
    assert task["status"] == "completed"
    ids = [p["id"] for p in load_project(manifest)[0].translation_proposals]
    assert sorted(ids) == ["s1", "s2", "s3"]  # newer proposals replace older ones


def test_task_fails_clearly_when_nothing_can_be_revised(
    template_project, application, tmp_path, monkeypatch
):
    manifest = project_with_translations(template_project, tmp_path, count=1)
    project, directory = load_project(manifest)
    project.sentences[0].review_locked = True
    save_project(project, directory)
    task, server = start_revision(application, manifest, monkeypatch, lambda _p: answer())
    assert task["status"] == "failed" and "没有可以改译的句子" in task["error"]
    assert server.payloads == []


def test_task_with_machine_translation_provider_fails_with_chinese_message(
    template_project, application, tmp_path, monkeypatch
):
    manifest = project_with_translations(template_project, tmp_path)
    project, directory = load_project(manifest)
    project.settings.translation_provider = "deepl"
    save_project(project, directory)
    task, server = start_revision(application, manifest, monkeypatch, lambda _p: answer())
    assert task["status"] == "failed" and "不能按要求改译" in task["error"]
    assert server.payloads == []


def decide(application, manifest, accept=(), reject=()):
    revision = load_project(manifest)[0].revision
    return API(application).call(
        "projects/proposals",
        {"project": manifest, "revision": revision, "accept": list(accept), "reject": list(reject)},
    )


def stored_proposals(template_project, application, tmp_path, monkeypatch):
    manifest = project_with_translations(template_project, tmp_path)
    task, _ = start_revision(
        application,
        manifest,
        monkeypatch,
        lambda _payload: answer(
            {"id": "s1", "zh": "新译文1", "kind": "speech"},
            {"id": "s2", "zh": "新译文2", "kind": "speech"},
            {"id": "s3", "zh": "", "kind": "sound"},
        ),
    )
    assert task["status"] == "completed"
    return manifest


def test_accepting_writes_text_locks_the_row_and_invalidates_dubbing(
    template_project, application, tmp_path, monkeypatch
):
    manifest = stored_proposals(template_project, application, tmp_path, monkeypatch)
    result = decide(application, manifest, accept=["s1"])
    assert result["proposal_result"] == {"accepted": 1, "rejected": 0, "stale": 0}
    assert [p["id"] for p in result["translation_proposals"]] == ["s2", "s3"]
    project, _ = load_project(manifest)
    first = project.sentences[0]
    assert first.zh_text == "新译文1" and first.review_locked is True
    assert first.tts_file is None and first.tts_cache_key is None
    assert first.tts_duration_seconds is None and first.status == "translated"
    untouched = project.sentences[1]
    assert untouched.zh_text == "旧译文2" and untouched.tts_file == "tts/s2.wav"
    assert untouched.review_locked is False
    assert [p["id"] for p in project.translation_proposals] == ["s2", "s3"]
    assert result["revision"] == project.revision


def test_rejecting_removes_the_proposal_and_keeps_the_translation(
    template_project, application, tmp_path, monkeypatch
):
    manifest = stored_proposals(template_project, application, tmp_path, monkeypatch)
    result = decide(application, manifest, reject=["s1", "s2"])
    assert result["proposal_result"]["rejected"] == 2
    project, _ = load_project(manifest)
    assert [s.zh_text for s in project.sentences] == ["旧译文1", "旧译文2", "旧译文3"]
    assert all(s.review_locked is False and s.tts_file for s in project.sentences)
    assert [p["id"] for p in project.translation_proposals] == ["s3"]


def test_accepting_a_sound_proposal_marks_the_sentence_as_not_dubbed(
    template_project, application, tmp_path, monkeypatch
):
    manifest = stored_proposals(template_project, application, tmp_path, monkeypatch)
    decide(application, manifest, accept=["s3"])
    sentence = load_project(manifest)[0].sentences[2]
    assert sentence.status == "skipped_sound" and sentence.enabled is False
    assert sentence.zh_text == "" and sentence.tts_file is None
    assert sentence.review_locked is True


def test_accept_all_and_reject_all(template_project, application, tmp_path, monkeypatch):
    manifest = stored_proposals(template_project, application, tmp_path, monkeypatch)
    result = decide(application, manifest, accept=["s1", "s2", "s3"])
    assert result["translation_proposals"] == []
    assert result["proposal_result"]["accepted"] == 3
    assert load_project(manifest)[0].translation_proposals == []


def test_proposal_for_a_sentence_edited_since_is_dropped_not_applied(
    template_project, application, tmp_path, monkeypatch
):
    manifest = stored_proposals(template_project, application, tmp_path, monkeypatch)
    project, directory = load_project(manifest)
    project.sentences[0].zh_text = "我自己改的"
    save_project(project, directory)
    result = decide(application, manifest, accept=["s1"])
    assert result["proposal_result"] == {"accepted": 0, "rejected": 0, "stale": 1}
    project, _ = load_project(manifest)
    assert project.sentences[0].zh_text == "我自己改的"
    assert [p["id"] for p in project.translation_proposals] == ["s2", "s3"]


def test_deciding_with_an_outdated_revision_is_refused(
    template_project, application, tmp_path, monkeypatch
):
    manifest = stored_proposals(template_project, application, tmp_path, monkeypatch)
    revision = load_project(manifest)[0].revision
    with pytest.raises(ProjectError):
        API(application).call(
            "projects/proposals",
            {"project": manifest, "revision": revision - 1, "accept": ["s1"], "reject": []},
        )
    assert len(load_project(manifest)[0].translation_proposals) == 3
