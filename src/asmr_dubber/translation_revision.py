"""Project level handling of "revise on request" proposals.

A revision run never edits a sentence.  It stores proposals on the project
(``DubProject.translation_proposals``), and the user then accepts or rejects them.
Each stored proposal is a plain dict::

    {"id": "s000001", "old_zh": "...", "new_zh": "...", "kind": "speech" | "sound",
     "instruction": "<the user's wording>", "created_at": "<ISO timestamp>"}
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .errors import ProjectError, TranslationError
from .lifecycle import invalidate_outputs, project_operation
from .models import DubProject, save_project
from .task_control import CancellationSignal, check_cancelled
from .translation import (
    LLM_REVISION_PROVIDERS,
    MACHINE_REVISION_ERROR,
    Progress,
    RevisionReport,
    _mark_sound,
    _mark_translated,
    revise_translations,
)
from .user_settings import PROVIDER_PRESETS, resolve_api_key

MAX_INSTRUCTION_CHARS = 2_000


def _proposal_records(report: RevisionReport, instruction: str) -> list[dict[str, Any]]:
    created_at = datetime.now(UTC).isoformat()
    return [
        {
            "id": item.id,
            "old_zh": item.old_zh,
            "new_zh": item.new_zh,
            "kind": item.kind,
            "instruction": instruction,
            "created_at": created_at,
        }
        for item in report.proposals
    ]


def merge_proposals(
    project: DubProject, new_records: list[dict[str, Any]], decided_ids: Iterable[str] = ()
) -> None:
    """A newer proposal replaces the older pending one for the same sentence."""

    replaced = {record["id"] for record in new_records} | set(decided_ids)
    project.translation_proposals = [
        record for record in project.translation_proposals if record.get("id") not in replaced
    ] + new_records


@project_operation
def revise_project_translations(
    project: DubProject,
    project_dir: Path,
    sentence_ids: list[str],
    instruction: str,
    include_edited: bool = False,
    api_key: str | None = None,
    progress: Progress | None = None,
    cancel_event: CancellationSignal | None = None,
    client: httpx.Client | None = None,
) -> RevisionReport:
    check_cancelled(cancel_event)
    instruction = instruction.strip()
    if not instruction:
        raise ProjectError("请先写下改译要求。")
    if len(instruction) > MAX_INSTRUCTION_CHARS:
        raise ProjectError(f"改译要求太长，请控制在 {MAX_INSTRUCTION_CHARS} 字以内。")
    settings = project.settings
    provider = settings.translation_provider
    if provider not in LLM_REVISION_PROVIDERS:
        raise ProjectError(MACHINE_REVISION_ERROR)
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ProjectError(f"未知翻译服务：{provider}")
    if project.translation_language != settings.tts_target_language:
        raise ProjectError("现有译文的语言与当前目标语言不一致，请先重新翻译。")
    key = resolve_api_key(provider, api_key)

    def checkpoint(report: RevisionReport) -> None:
        # Called after every batch: proposals found so far survive a later failure.
        records = _proposal_records(report, instruction)
        if records:
            merge_proposals(project, records)
            save_project(project, project_dir)

    report = revise_translations(
        project.sentences,
        instruction,
        key,
        settings.translation_model,
        sentence_ids=sentence_ids,
        include_edited=include_edited,
        base_url=settings.translation_base_url.strip() or str(preset["base_url"]),
        provider=provider,
        source_language=project.source_language,
        target_language=settings.tts_target_language,
        system_prompt=settings.translation_prompt,
        temperature=settings.translation_temperature,
        top_p=settings.translation_top_p,
        max_output_tokens=settings.translation_max_output_tokens,
        send_context=settings.translation_send_context,
        context_sentences=settings.translation_context_sentences,
        job_id=f"asmr_{project.source.sha256[:24]}",
        progress=progress,
        on_batch=checkpoint,
        client=client,
        cancel_event=cancel_event,
        extra_body=settings.translation_extra_body,
        sound_policy=settings.translation_sound_policy,
    )
    if report.candidates == 0:
        raise ProjectError(
            "所选句子里没有可以改译的句子：需要已启用、已有译文；"
            "手动改过的句子要勾选“也包括我手动改过的句子”。"
        )
    if report.failed:
        raise TranslationError(report.failure_message())
    return report


def resolve_proposals(
    project: DubProject,
    accept_ids: Iterable[str] = (),
    reject_ids: Iterable[str] = (),
) -> dict[str, int]:
    """Apply accepted proposals to the sentences and drop every decided proposal.

    A proposal whose sentence changed after it was made (or disappeared) is stale:
    it is dropped without touching the sentence.  The caller saves the project.
    """

    accept = set(accept_ids)
    reject = set(reject_ids) - accept
    by_id = {sentence.id: sentence for sentence in project.sentences}
    policy = project.settings.translation_sound_policy
    accepted = stale = rejected = 0
    remaining: list[dict[str, Any]] = []
    for record in project.translation_proposals:
        sentence_id = record.get("id")
        if sentence_id in reject:
            rejected += 1
            continue
        if sentence_id not in accept:
            remaining.append(record)
            continue
        sentence = by_id.get(sentence_id)
        if sentence is None or sentence.zh_text != record.get("old_zh"):
            stale += 1
            continue
        text = str(record.get("new_zh", ""))
        if record.get("kind") == "sound":
            _mark_sound(sentence, text, policy)
        else:
            _mark_translated(sentence, text)
        sentence.review_locked = True
        sentence.reference_file = None
        accepted += 1
    project.translation_proposals = remaining
    if accepted:
        invalidate_outputs(project)
    return {"accepted": accepted, "rejected": rejected, "stale": stale}
