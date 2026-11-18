"""Importing and reconciling source transcripts in projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from . import runlog
from .constants import SAMPLE_RATE, TRANSCRIPT_MODE_ASR_RECONCILE
from .domain import (
    TRANSCRIPT_MODE_DIRECT,
    ToolPaths,
)
from .errors import VideoPreparerError


def _qwen_script_alignment_available(paths: ToolPaths) -> bool:
    model_root = paths.asmr_home / "models"
    if not model_root.is_dir():
        return False
    return any(
        "qwen3-forcedaligner" in path.as_posix().casefold()
        for path in model_root.rglob("config.json")
    )


def _load_transcript_sentences(
    path: Path,
    *,
    language: Literal["ja", "en", "zh"],
    duration_seconds: float,
) -> Any:
    try:
        from asmr_dubber.transcript_import import parse_transcript

        return parse_transcript(
            duration_seconds=duration_seconds,
            path=path,
            language=language,
        )
    except Exception as exc:
        raise VideoPreparerError(f"无法解析台本/字幕 {path.name}：{exc}") from exc


def _replace_project_with_timed_transcripts(
    project_json: Path,
    timeline: list[dict[str, Any]],
    *,
    language: Literal["ja", "en", "zh"],
) -> int:
    """Build one project timeline directly from per-track timed subtitles."""

    try:
        from asmr_dubber.languages import source_language_label
        from asmr_dubber.models import load_project, save_project, settings_for_source_language
        from asmr_dubber.pipeline import export_transcript

        project, project_dir = load_project(project_json)
        combined = []
        for item in timeline:
            duration = int(item["duration_samples"]) / SAMPLE_RATE
            parsed = _load_transcript_sentences(
                Path(str(item["transcript"])),
                language=cast(
                    Literal["ja", "en", "zh"], item.get("transcript_language") or language
                ),
                duration_seconds=duration,
            )
            if not parsed.timed:
                raise VideoPreparerError("所选文件没有有效时间戳；停止导入，不自动改用 ASR。")
            offset = int(item["start_samples"]) / SAMPLE_RATE
            for sentence in parsed.sentences:
                copied = sentence.model_copy(deep=True)
                copied.id = f"s{len(combined) + 1:06d}"
                copied.start_seconds += offset
                copied.end_seconds += offset
                combined.append(copied)
        if not combined:
            raise VideoPreparerError("已有字幕中没有可导入的有效台词。")

        project.sentences = combined
        project.translation_language = project.settings.tts_target_language
        if project.settings.tts_target_language == "en" and language == "zh":
            for sentence in project.sentences:
                sentence.zh_text = ""
        project.source_language = language
        project.settings = settings_for_source_language(project.settings, language)
        project.asr_language = f"自动导入的分轨{source_language_label(language)}时间轴字幕"
        project.asr_settings_dirty = False
        project.settings.tts_reference_sentence_id = None
        project.chinese_stem_file = None
        project.output_file = None
        project.output_video_file = None
        project.subtitle_srt_file = None
        project.subtitle_lrc_file = None
        project.subtitle_video_file = None
        save_project(project, project_dir)
        export_transcript(project, project_dir)
        return len(combined)
    except VideoPreparerError:
        raise
    except Exception as exc:
        raise VideoPreparerError(f"无法合并分轨字幕：{exc}") from exc


def import_available_source_transcript(
    paths: ToolPaths,
    project_json: Path,
    timeline: list[dict[str, Any]],
) -> dict[str, Any] | None:
    usable = [
        item
        for item in timeline
        if item.get("transcript")
        and str(item.get("transcript_language") or "") in {"ja", "en", "zh"}
        and Path(str(item["transcript"])).suffix.casefold() != ".pdf"
    ]
    if not usable:
        return None

    # 单轨可以直接使用 ASMR Dubber 的完整台本导入流程，包括纯文本对齐。
    if len(timeline) == 1 and len(usable) == 1:
        item = usable[0]
        transcript = Path(str(item["transcript"])).resolve()
        language = str(item["transcript_language"])
        transcript_mode = str(item.get("transcript_mode") or TRANSCRIPT_MODE_DIRECT)
        use_asr_timing = transcript_mode == TRANSCRIPT_MODE_ASR_RECONCILE or not bool(
            item.get("transcript_timed")
        )
        plain_timing = "script_review" if use_asr_timing else "estimate"
        try:
            from asmr_dubber.models import load_project
            from asmr_dubber.pipeline import import_project_transcript

            project, project_dir = load_project(project_json)
            result = import_project_transcript(
                project,
                project_dir,
                transcript_path=transcript,
                plain_timing=plain_timing,
                script_language=language,
                use_embedded_timing=not use_asr_timing,
                progress=lambda message, current, total: runlog.emit(f"  {message}"),
            )
        except Exception as exc:
            raise VideoPreparerError(f"自动导入台本/字幕失败：{exc}") from exc
        return {
            "kind": "direct",
            "language": language,
            "path": str(transcript),
            **result,
        }

    if any(
        str(item.get("transcript_mode") or TRANSCRIPT_MODE_DIRECT) == TRANSCRIPT_MODE_ASR_RECONCILE
        or not bool(item.get("transcript_timed"))
        for item in usable
    ):
        return {"kind": "reconcile_after_asr", "count": len(usable)}

    # Coverage is per track, not the fraction of silence spanned by cues or
    # agreement of language labels. Chinese rows already have zh_text and
    # translation will process only untranslated source-language rows.
    if len(usable) != len(timeline):
        if any(
            str(item.get("transcript_language") or "") == "zh"
            and bool(item.get("transcript_timed"))
            and str(item.get("transcript_mode") or TRANSCRIPT_MODE_DIRECT) == TRANSCRIPT_MODE_DIRECT
            for item in usable
        ):
            return {"kind": "zh_overlay_partial", "count": len(usable)}
        return {"kind": "partial", "count": len(usable)}
    languages = {str(item["transcript_language"]) for item in usable}
    untranslated_languages = languages - {"zh"}
    if len(untranslated_languages) > 1:
        raise VideoPreparerError(
            "所有音轨已有字幕，但日文和英文字幕混合。请统一字幕语言标记或分轨处理；"
            "不会丢弃字幕重新识别。中文字幕可与一种原文语言混合。"
        )
    language = cast(Literal["ja", "en", "zh"], next(iter(untranslated_languages), "zh"))
    sentence_count = _replace_project_with_timed_transcripts(
        project_json,
        timeline,
        language=language,
    )
    return {"kind": "direct", "language": language, "sentences": sentence_count}


def reconcile_timeline_scripts(
    project_json: Path,
    timeline: list[dict[str, Any]],
) -> list[dict[str, object]]:
    """Use ASR timing for the per-track scripts selected in an AutoFlow plan."""

    try:
        from asmr_dubber.models import load_project
        from asmr_dubber.pipeline import reconcile_analyzed_project_script

        results: list[dict[str, object]] = []
        for item in timeline:
            transcript_text = str(item.get("transcript") or "").strip()
            mode = str(item.get("transcript_mode") or TRANSCRIPT_MODE_DIRECT)
            language = str(item.get("transcript_language") or "").casefold()
            if (
                not transcript_text
                or mode != TRANSCRIPT_MODE_ASR_RECONCILE
                or language not in {"ja", "en", "zh"}
            ):
                continue
            project, project_dir = load_project(project_json)
            start = int(item["start_samples"]) / SAMPLE_RATE
            end = start + int(item["duration_samples"]) / SAMPLE_RATE
            result = reconcile_analyzed_project_script(
                project,
                project_dir,
                transcript_path=Path(transcript_text),
                script_language=cast(Literal["ja", "en", "zh"], language),
                start_seconds=start,
                end_seconds=end,
                progress=lambda message, current, total: runlog.emit(f"  {message}"),
            )
            results.append(result)
        return results
    except VideoPreparerError:
        raise
    except Exception as exc:
        raise VideoPreparerError(f"使用 ASR 时间轴校对台本失败：{exc}") from exc


def overlay_timed_chinese_transcripts(
    project_json: Path,
    timeline: list[dict[str, Any]],
) -> int:
    chinese_cues: list[Any] = []
    for item in timeline:
        if (
            str(item.get("transcript_language") or "") != "zh"
            or not item.get("transcript")
            or not bool(item.get("transcript_timed"))
            or str(item.get("transcript_mode") or TRANSCRIPT_MODE_DIRECT) != TRANSCRIPT_MODE_DIRECT
        ):
            continue
        duration = int(item["duration_samples"]) / SAMPLE_RATE
        parsed = _load_transcript_sentences(
            Path(str(item["transcript"])),
            language="zh",
            duration_seconds=duration,
        )
        offset = int(item["start_samples"]) / SAMPLE_RATE
        for cue in parsed.sentences:
            copied = cue.model_copy(deep=True)
            copied.start_seconds += offset
            copied.end_seconds += offset
            chinese_cues.append(copied)
    if not chinese_cues:
        return 0

    try:
        from asmr_dubber.models import load_project, save_project
        from asmr_dubber.pipeline import export_transcript

        project, project_dir = load_project(project_json)
        assignments: dict[int, list[str]] = {}
        for cue in chinese_cues:
            best_index = -1
            best_overlap = 0.0
            cue_midpoint = (cue.start_seconds + cue.end_seconds) / 2
            best_distance = float("inf")
            for index, sentence in enumerate(project.sentences):
                overlap = max(
                    0.0,
                    min(cue.end_seconds, sentence.end_seconds)
                    - max(cue.start_seconds, sentence.start_seconds),
                )
                distance = abs(cue_midpoint - (sentence.start_seconds + sentence.end_seconds) / 2)
                if overlap > best_overlap or (
                    overlap == best_overlap and overlap > 0 and distance < best_distance
                ):
                    best_index = index
                    best_overlap = overlap
                    best_distance = distance
            if best_index < 0:
                nearest = min(
                    range(len(project.sentences)),
                    key=lambda index: abs(
                        cue_midpoint
                        - (
                            project.sentences[index].start_seconds
                            + project.sentences[index].end_seconds
                        )
                        / 2
                    ),
                    default=-1,
                )
                if nearest >= 0:
                    distance = abs(
                        cue_midpoint
                        - (
                            project.sentences[nearest].start_seconds
                            + project.sentences[nearest].end_seconds
                        )
                        / 2
                    )
                    if distance <= 3.0:
                        best_index = nearest
            if best_index >= 0:
                assignments.setdefault(best_index, []).append(cue.zh_text)

        applied = 0
        for index, texts in assignments.items():
            text = " ".join(part.strip() for part in texts if part.strip()).strip()
            if text:
                project.sentences[index].zh_text = text
                project.sentences[index].status = "translated"
                applied += 1
        if applied:
            save_project(project, project_dir)
            export_transcript(project, project_dir)
        return applied
    except Exception as exc:
        raise VideoPreparerError(f"无法合并已有中文字幕：{exc}") from exc
