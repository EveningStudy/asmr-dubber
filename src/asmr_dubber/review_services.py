"""Review-panel operations. Project revision protects edits; report is evidence."""

from __future__ import annotations

import html
import json
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .asr_review import REVIEW_VERSION, replace_window, rows_fingerprint, run_audio_review
from .audio import _run_ffmpeg, make_analysis_copy, verify_source
from .errors import OperationCancelledError, ProjectError
from .lifecycle import invalidate_outputs
from .models import Sentence, load_project, save_project
from .platforms import portable_home
from .storage import atomic_write_text, exclusive_file_lock
from .task_control import CancellationSignal
from .ui_services import ProjectView, apply_table, ui_stage_directory, view


def _report(directory: Path) -> dict[str, Any]:
    path = directory / "analysis/asr_review.json"
    if not path.is_file():
        raise ProjectError("尚无音频片段复核结果。请启用多模型复核后运行 ASR，或点击重试复核。")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != REVIEW_VERSION:
        raise ProjectError("这是旧版交叉校对报告，请运行新的音频片段复核。")
    return data


def _save_report(directory: Path, report: dict[str, Any]) -> None:
    atomic_write_text(
        directory / "analysis/asr_review.json", json.dumps(report, ensure_ascii=False, indent=2)
    )


def review_overview(manifest: str) -> tuple[list[list[Any]], list[tuple[str, str]], str]:
    if not manifest:
        return [], [], "请先打开项目。"
    _, directory = load_project(manifest)
    try:
        report = _report(directory)
    except ProjectError as exc:
        return [], [], str(exc)
    rows, choices = [], []
    status_labels = {
        "agreement": "文字一致",
        "disagreement": "存在文字差异",
        "single_family": "证据不足",
        "boundary_review": "边界需人工确认",
        "unavailable": "暂无有效候选",
    }
    action_labels = {
        "keep_baseline": "保留主稿",
        "accepted_by_user": "已采纳",
        "kept_by_user": "已确认主稿",
        "auto_applied": "自动采纳（可撤销）",
    }
    for result in report["results"]:
        window = f"{result['start']:.1f}–{result['end']:.1f}s"
        rows.append(
            [
                result["id"],
                window,
                status_labels.get(result["status"], result["status"]),
                action_labels.get(result["action"], result["action"]),
                result["baseline_text"],
                " | ".join(candidate["text"] for candidate in result["candidates"]),
            ]
        )
        choices.append(
            (
                f"{result['id']} · {window} · "
                f"{status_labels.get(result['status'], result['status'])}",
                result["id"],
            )
        )
    return (
        rows,
        choices,
        (
            f"状态：{report['state']}；{len(rows)} 个统一音频片段；"
            f"{sum(bool(item['needs_review']) for item in report['results'])} 段待查看；"
            f"{len(report['warnings'])} 项运行警告。主模型原稿已独立保存。"
            + (
                "\n"
                + "\n".join(str(item.get("error", ""))[:300] for item in report["warnings"][:5])
                if report["warnings"]
                else ""
            )
        ),
    )


def _find(report: dict[str, Any], identifier: str) -> dict[str, Any]:
    result = next((item for item in report["results"] if item["id"] == identifier), None)
    if result is None:
        raise ProjectError("请先选择复核音频片段。")
    return result


def candidate_details(
    manifest: str, identifier: str, candidate_id: str = ""
) -> tuple[list[tuple[str, str]], str, str | None]:
    project, directory = load_project(manifest)
    result = _find(_report(directory), identifier)
    choices = [
        (f"{len(item['families'])} 个家族 · {item['text'][:100]}", item["id"])
        for item in result["candidates"]
    ]
    selected = next((item for item in result["candidates"] if item["id"] == candidate_id), None)
    before = result["baseline_text"]
    if selected is None:
        body = f"<p>主稿：{html.escape(before)}</p><p>{html.escape(result['reason'])}</p>"
    else:
        after = selected["text"]
        pieces = []
        for tag, a, b, c, d in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
            if tag == "equal":
                pieces.append(html.escape(before[a:b]))
            else:
                if b > a:
                    pieces.append(
                        f"<del style='background:#ffe0e0'>{html.escape(before[a:b])}</del>"
                    )
                if d > c:
                    pieces.append(
                        f"<ins style='background:#dcfce7'>{html.escape(after[c:d])}</ins>"
                    )
        body = "<p>差异：" + "".join(pieces) + "</p>"
        body += "<p>来源：" + html.escape("、".join(selected["sources"])) + "</p>"
        if not selected["applicable"]:
            body += "<p>跨边界或人工锁定，不能一键替换；请试听后手动校对。</p>"
        if selected.get("timing_warning"):
            body += "<p>该模型时间戳越过音频边界，文字已保留；不能用截断时间作为自动采纳依据。</p>"
        if selected["protected_change"]:
            body += "<p>涉及数字或否定等高风险变化，不会自动应用；请确认原音频。</p>"
    source = verify_source(directory, project.source)
    start, end = float(result["audio_start"]), float(result["audio_end"])
    if not 0 <= start < end <= project.source.duration_seconds + 0.25:
        raise ProjectError("复核音频范围无效。")
    preview = ui_stage_directory() / f"asr-review-{uuid.uuid4().hex}.wav"
    _run_ffmpeg(
        [
            "-y",
            "-ss",
            str(start),
            "-i",
            str(source),
            "-t",
            str(end - start),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(preview),
        ]
    )
    return choices, body, str(preview)


def apply_review(
    manifest: str, identifier: str, candidate_id: str, keep: bool = False, table: Any = None
) -> ProjectView:
    project, directory = load_project(manifest)
    with exclusive_file_lock(directory / ".project.lock"):
        if table is not None:
            apply_table(project, table)
        report = _report(directory)
        result = _find(report, identifier)
        before = [row for row in project.sentences if row.id in result["baseline_ids"]]
        if rows_fingerprint(before) != result["baseline_fingerprint"]:
            raise ProjectError("主稿已改变，旧提案不能覆盖当前内容。请重试复核。")
        if keep:
            after = [row.model_copy(update={"review_locked": True}) for row in before]
            project.sentences = [
                row for row in project.sentences if row.id not in result["baseline_ids"]
            ] + after
            result["action"] = "kept_by_user"
        else:
            candidate = next(
                (item for item in result["candidates"] if item["id"] == candidate_id), None
            )
            if candidate is None:
                raise ProjectError("请先选择候选文字。")
            project.sentences = replace_window(
                project.sentences, result, candidate, project.settings, project.source_language
            )
            after = [
                row
                for row in project.sentences
                if row.id.startswith(f"{identifier}-{candidate_id}-")
            ]
            result["action"] = "accepted_by_user"
            result["selected_candidate"] = candidate_id
            invalidate_outputs(project)
        project.sentences.sort(key=lambda row: (row.start_seconds, row.end_seconds, row.id))
        report["undo"].append(
            {
                "window": identifier,
                "before": [row.model_dump() for row in before],
                "after_ids": [row.id for row in after],
                "after_fingerprint": rows_fingerprint(after),
            }
        )
        save_project(project, directory)
        _save_report(directory, report)
        from .pipeline import export_transcript

        export_transcript(project, directory)
    return view(
        project,
        directory,
        "已保留并锁定原稿。" if keep else "已采纳候选并统一分句；受影响的译文和配音需重新生成。",
    )


def undo_review(manifest: str, table: Any = None) -> ProjectView:
    project, directory = load_project(manifest)
    with exclusive_file_lock(directory / ".project.lock"):
        if table is not None:
            apply_table(project, table)
        report = _report(directory)
        if not report["undo"]:
            raise ProjectError("没有可撤销的复核操作。")
        action = report["undo"][-1]
        after = [row for row in project.sentences if row.id in action["after_ids"]]
        if rows_fingerprint(after) != action["after_fingerprint"]:
            raise ProjectError("采纳后的句子又被编辑过，不能安全撤销；请在表格中校对。")
        project.sentences = [
            row for row in project.sentences if row.id not in action["after_ids"]
        ] + [Sentence.model_validate(row) for row in action["before"]]
        project.sentences.sort(key=lambda row: (row.start_seconds, row.end_seconds, row.id))
        report["undo"].pop()
        _find(report, action["window"])["action"] = "keep_baseline"
        invalidate_outputs(project)
        save_project(project, directory)
        _save_report(directory, report)
        from .pipeline import export_transcript

        export_transcript(project, directory)
    return view(project, directory, "已撤销上一次复核操作。")


def unlock_review(manifest: str, table: Any = None) -> ProjectView:
    project, directory = load_project(manifest)
    if table is not None:
        apply_table(project, table)
    for row in project.sentences:
        row.review_locked = False
    save_project(project, directory)
    from .pipeline import export_transcript

    export_transcript(project, directory)
    return view(project, directory, "已解除人工确认锁定；文字未改变，可以重新识别或复核。")


def retry_review(
    manifest: str, table: Any, progress: Any = None, cancel_event: CancellationSignal | None = None
) -> ProjectView:
    project, directory = load_project(manifest)
    if not project.sentences:
        raise ProjectError("请先完成主模型识别，再进行复核。")
    with (
        exclusive_file_lock(directory / ".project.lock"),
        exclusive_file_lock(portable_home() / ".runtime-install.lock", timeout_seconds=30),
    ):
        apply_table(project, table)
        save_project(project, directory)
        from .pipeline import export_transcript

        export_transcript(project, directory)
        previous = rows_fingerprint(project.sentences)
        source = verify_source(directory, project.source)
        if project.settings.separation_enabled:
            from .separation import ensure_separation

            source, _ = ensure_separation(project, directory, source, progress)
            audio = make_analysis_copy(source, source.parent / "asr_16k_mono.wav")
        else:
            audio = make_analysis_copy(source, directory / "analysis/asr_16k_mono.wav")
        try:
            project.sentences = run_audio_review(
                audio,
                project.sentences,
                project.settings,
                directory / "analysis/asr_review.json",
                source_language=project.source_language,
                progress=progress,
                cancel_event=cancel_event,
            )
        except OperationCancelledError:
            raise
        if rows_fingerprint(project.sentences) != previous:
            invalidate_outputs(project)
        save_project(project, directory)
        export_transcript(project, directory)
    return view(project, directory, "音频复核完成。成功窗口已缓存，主稿与未确定内容保留。")


def align_review(
    manifest: str, table: Any, progress: Any = None, cancel_event: CancellationSignal | None = None
) -> ProjectView:
    """Timestamp alignment is explicit and cannot serve as text-correctness evidence."""
    from .forced_alignment import align_sentences_with_qwen

    project, directory = load_project(manifest)
    with (
        exclusive_file_lock(directory / ".project.lock"),
        exclusive_file_lock(portable_home() / ".runtime-install.lock", timeout_seconds=30),
    ):
        apply_table(project, table)
        save_project(project, directory)
        source = verify_source(directory, project.source)
        if project.settings.separation_enabled:
            from .separation import ensure_separation

            source, _ = ensure_separation(project, directory, source, progress)
            audio = make_analysis_copy(source, source.parent / "asr_16k_mono.wav")
        else:
            audio = make_analysis_copy(source, directory / "analysis/asr_16k_mono.wav")
        original = {row.id: row for row in project.sentences}
        proposed = [row.model_copy(deep=True) for row in project.sentences]
        identities = {id(row): row.id for row in proposed}
        records = align_sentences_with_qwen(
            audio,
            proposed,
            project.settings,
            progress=progress,
            cancel_event=cancel_event,
            source_language=project.source_language,
        )
        accepted = 0
        for row in proposed:
            row.id = identities[id(row)]
            before = original[row.id]
            if (
                max(
                    abs(row.start_seconds - before.start_seconds),
                    abs(row.end_seconds - before.end_seconds),
                )
                > project.settings.asr_review_max_drift_seconds
            ):
                row.start_seconds, row.end_seconds = before.start_seconds, before.end_seconds
            elif (row.start_seconds, row.end_seconds) != (before.start_seconds, before.end_seconds):
                row.tts_file = row.tts_cache_key = row.reference_file = None
                row.tts_duration_seconds = None
                accepted += 1
        project.sentences = sorted(
            proposed, key=lambda row: (row.start_seconds, row.end_seconds, row.id)
        )
        if accepted:
            invalidate_outputs(project)
        atomic_write_text(
            directory / "analysis/review_alignment.json",
            json.dumps(
                {"results": records, "accepted_changes": accepted}, ensure_ascii=False, indent=2
            ),
        )
        save_project(project, directory)
        from .pipeline import export_transcript

        export_transcript(project, directory)
    return view(
        project,
        directory,
        f"独立时间对齐完成，更新 {accepted} 句；文字未改变，超出允许漂移的边界已保留原值。",
    )
