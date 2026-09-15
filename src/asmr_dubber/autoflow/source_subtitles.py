"""Subtitle file publication with optional translation, without audio/video output."""

from __future__ import annotations

from pathlib import Path

from ..models import load_project, save_project
from ..pipeline import import_project_transcript
from ..task_control import check_cancelled
from . import engine as e
from .domain import AppConfig, SmartTaskPlan, ToolPaths


def execute_source_subtitles(paths: ToolPaths, config: AppConfig, plan: SmartTaskPlan) -> None:
    """Process tracks independently and concatenate subtitle clocks without media encoding."""
    check_cancelled()
    language = plan.subtitle_language
    if language not in {"source", "zh", "bilingual"}:
        raise e.VideoPreparerError("字幕内容设置无效。")
    labels = {"source": "原文字幕", "zh": "译文字幕", "bilingual": "双语字幕"}

    def output_name(original: str) -> str:
        selected = (
            original
            if plan.subtitle_naming == "original"
            else (
                plan.subtitle_custom_name if plan.subtitle_naming == "custom" else labels[language]
            )
        )
        return e.safe_filename_component(selected, fallback=labels[language], limit=180)

    root = plan.output_root
    e.prepare_smart_output_root(root, plan_id=plan.plan_id, force=plan.force, rebuild=plan.rebuild)
    e.write_plan_manifest(
        root / "处理清单.json",
        source_folder=plan.folder,
        output_folder=root,
        mode=e.MODE_AUDIO,
        layout=plan.layout,
        edition=plan.edition,
        sources=list(plan.sources),
        background=None,
        embed_subtitles=False,
        subtitles_only=True,
        plan_id=plan.plan_id,
    )
    signature = e.fingerprint(list(plan.sources), None)
    states = []
    descriptors = []
    offset_samples = 0
    srt_entries = []
    lrc_entries = []
    for index, source in enumerate(plan.sources, start=1):
        check_cancelled()
        state_path = e.planned_state_path(plan.folder, plan.plan_id, f"source-{index:04d}")
        state = None if plan.rebuild else e.load_state(state_path)
        if state and state.get("fingerprint") != signature:
            raise e.VideoPreparerError("音频或字幕已改变，请选择重做；既有字幕未被覆盖。")
        state = state or {"schema": 1, "fingerprint": signature}
        project_json = Path(str(state.get("project_json") or ""))
        if not project_json.is_file():
            project_json = e.create_asmr_project(
                paths, source.path, source_language=source.source_language
            )
            state["project_json"] = str(project_json)
            state["analyzed"] = False
            e.save_state(state_path, state)
        if not state.get("analyzed"):
            # Only source-language timed subtitles bypass ASR. Translation, if
            # requested, runs after source text is ready; inputs remain untouched.
            use_transcript = (
                source.transcript_path is not None
                and source.transcript_timed
                and source.transcript_language == source.source_language
                and source.transcript_mode == e.TRANSCRIPT_MODE_DIRECT
            )
            if use_transcript:
                project, directory = load_project(project_json)
                import_project_transcript(
                    project,
                    directory,
                    transcript_path=source.transcript_path,
                    script_language=source.source_language,
                    use_embedded_timing=True,
                )
            else:
                if source.transcript_path:
                    print("仅字幕文件：所选台本非原文时间轴字幕，本模式使用 ASR 后按所选语言输出。")
                e.run_asmr_cli(paths, "analyze", str(project_json))
            state["analyzed"] = True
            e.save_state(state_path, state)
        check_cancelled()
        project_data, directory = load_project(project_json)
        if project_data.source.media_type != "audio":
            raise e.VideoPreparerError("仅字幕文件流程只接受扫描到的音轨，不进入视频编码。")
        if project_data.settings.subtitle_timeline != "source":
            project_data.settings.subtitle_timeline = "source"
            save_project(project_data, directory)
        if language != "source":
            e.run_asmr_cli(paths, "translate", str(project_json))
        e.run_asmr_cli(paths, "subtitles", str(project_json), "--language", language)
        project = e.read_project(project_json)
        srt = e.project_asset(project_json, project.get("subtitle_srt_file"))
        lrc = e.project_asset(project_json, project.get("subtitle_lrc_file"))
        if srt is None or lrc is None:
            raise e.VideoPreparerError("字幕文件未生成完整；保留项目，下次重试可复用 ASR。")
        duration = e.audio_duration_samples(paths, source.path)
        timeline = [
            {
                "filename": source.path.name,
                "relative_path": source.relative_path or source.path.name,
                "title_ja": source.title_ja,
                "start_samples": 0,
                "duration_samples": duration,
            }
        ]
        state.update(
            {
                "timeline": timeline,
                "mode": e.MODE_AUDIO,
                "outputs": {"srt": str(srt), "lrc": str(lrc)},
            }
        )
        srt_entries.append(
            (
                srt,
                offset_samples * 1000 // e.SAMPLE_RATE,
                (offset_samples + duration) * 1000 // e.SAMPLE_RATE,
            )
        )
        lrc_entries.append((lrc, offset_samples * 1000 // e.SAMPLE_RATE))
        offset_samples += duration
        if plan.layout in {e.LAYOUT_SEPARATE, e.LAYOUT_BOTH}:
            track_folder = root / "分轨" / f"{index:03d}"
            track_folder.mkdir(parents=True, exist_ok=True)
            for original, suffix in ((srt, "srt"), (lrc, "lrc")):
                e.atomic_copy(original, track_folder / f"{output_name(source.path.stem)}.{suffix}")
        state["status"] = "completed"
        e.save_state(state_path, state)
        states.append(state)
        descriptors.append(
            {
                "job_id": f"track-{index:04d}",
                "index": index,
                "kind": "track",
                "status": "completed",
                "project_json": str(project_json),
                "outputs": state["outputs"],
                "output": str(root / "分轨" / f"{index:03d}"),
            }
        )
        print(f"{labels[language]} {index}/{len(plan.sources)} 完成：{source.path.name}")
    if plan.layout in {e.LAYOUT_MERGED, e.LAYOUT_BOTH}:
        merged = root / "合并版"
        merged.mkdir(parents=True, exist_ok=True)
        name = output_name(
            plan.sources[0].path.stem if len(plan.sources) == 1 else plan.folder.name
        )
        e.combine_srt_files(srt_entries, merged / f"{name}.srt", final_offset_ms=0)
        e.combine_lrc_files(lrc_entries, merged / f"{name}.lrc", final_offset_ms=0)
    titles = {source.relative_path or source.path.name: source.title_ja for source in plan.sources}
    e.write_smart_summary(
        root,
        source_folder=plan.folder,
        mode=e.MODE_AUDIO,
        layout=plan.layout,
        sources=plan.sources,
        states=states,
        descriptors=descriptors,
        folder_translation=plan.folder.name,
        title_translations=titles,
        footer=config.timestamp_footer,
        footer_position=config.timestamp_footer_position,
        harmonized_delay_seconds=0,
    )
    manifest = e.write_plan_manifest(
        root / "处理清单.json",
        source_folder=plan.folder,
        output_folder=root,
        mode=e.MODE_AUDIO,
        layout=plan.layout,
        edition=plan.edition,
        sources=list(plan.sources),
        background=None,
        embed_subtitles=False,
        subtitles_only=True,
        plan_id=plan.plan_id,
        jobs=descriptors,
    )
    manifest["source_subtitles_only"] = True
    manifest["subtitle_language"] = language
    manifest["subtitle_naming"] = plan.subtitle_naming
    manifest["subtitle_custom_name"] = plan.subtitle_custom_name
    e.save_state(root / "处理清单.json", manifest)
    print(f"仅字幕文件完成：{root}（{labels[language]}，未配音、未制作视频）")
