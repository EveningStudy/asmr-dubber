"""Transcript language choices for a plan draft."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .catalog import (
    ScanResult,
)
from .domain import (
    TRANSCRIPT_MODE_DIRECT,
    AudioSource,
    SmartPlanDraft,
)
from .errors import VideoPreparerError

TRANSCRIPT_LANGUAGE_LABELS = {
    "zh": "中文台本/字幕",
    "ja": "日文台本/字幕",
    "en": "英文台本/字幕",
    "ignore": "不使用这份台本",
    "auto": "按文件名自动判断",
}


def transcript_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def default_transcript_language(path: Path) -> str:
    if path.suffix.casefold() == ".pdf":
        return "ignore"
    return "zh"


def default_transcript_choices(scan: ScanResult) -> dict[str, str]:
    return {
        transcript_key(item.path): default_transcript_language(item.path)
        for item in scan.transcripts
    }


def effective_sources_for_draft(draft: SmartPlanDraft) -> list[AudioSource]:
    effective: list[AudioSource] = []
    for source in draft.sources:
        if source.transcript_path is None:
            effective.append(source)
            continue
        choice = draft.transcript_choices.get(
            transcript_key(source.transcript_path),
            default_transcript_language(source.transcript_path),
        )
        if choice == "ignore":
            effective.append(
                replace(
                    source,
                    transcript_path=None,
                    transcript_language=None,
                    transcript_timed=False,
                    transcript_mode=TRANSCRIPT_MODE_DIRECT,
                )
            )
        elif choice == "auto":
            effective.append(source)
        elif choice in {"zh", "ja", "en"}:
            effective.append(replace(source, transcript_language=choice))
        else:
            raise VideoPreparerError(f"未知台本语言选择：{choice}")
    return effective


def transcript_rows_for_draft(
    draft: SmartPlanDraft,
) -> list[tuple[Any, list[str]]]:
    matched: dict[str, list[str]] = {}
    for source in draft.sources:
        if source.transcript_path is None:
            continue
        matched.setdefault(transcript_key(source.transcript_path), []).append(
            source.relative_path or source.path.name
        )
    return [
        (transcript, matched.get(transcript_key(transcript.path), []))
        for transcript in draft.scan.transcripts
    ]


def transcript_choice_label(
    draft: SmartPlanDraft,
    transcript: Any,
) -> str:
    choice = draft.transcript_choices.get(
        transcript_key(transcript.path),
        default_transcript_language(transcript.path),
    )
    if choice == "auto":
        detected = str(transcript.language or "ja").casefold()
        detected_label = TRANSCRIPT_LANGUAGE_LABELS.get(detected, detected)
        return f"{TRANSCRIPT_LANGUAGE_LABELS['auto']}（当前{detected_label}）"
    return TRANSCRIPT_LANGUAGE_LABELS.get(choice, choice)
