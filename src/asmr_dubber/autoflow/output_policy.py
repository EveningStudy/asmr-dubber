"""Batch output policy. Reuse synthesis; render each requested mix exactly once."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import ProjectSettings, load_project, save_project


def policy_for_settings(settings: ProjectSettings, content: str, subtitle: str) -> dict[str, Any]:
    if content not in {"dubbing", "replacement", "both", "subtitles", "source_subtitles"}:
        raise ValueError("未知处理目标。")
    variants = {
        "dubbing": ["bilingual"],
        "replacement": ["replace"],
        "both": ["bilingual", "replace"],
    }.get(content, [])
    if "replace" in variants and not settings.separation_enabled:
        raise ValueError("替换配音需要人声分离：请在设置中开启人声分离并保存，再加入队列。")
    # Saved with the plan, hashed by plan_identity, and reused after restart.
    values = settings.model_dump(mode="json")
    selected = {
        key: value
        for key, value in values.items()
        if key.startswith(("tts_", "separation_", "spatial_", "mix_", "chinese_"))
        or key in {"tts_target_language", "normalize_chinese_loudness", "match_source_loudness"}
    }
    # A sentence ID belongs to a single project, never to newly created tracks.
    selected.pop("tts_reference_sentence_id", None)
    return {"content": content, "variants": variants, "subtitle": subtitle, "settings": selected}


def configure_project(
    project_json: Path, policy: dict[str, Any], *, subtitles_only: bool = False
) -> None:
    if not policy:
        return
    project, directory = load_project(project_json)
    values = project.settings.model_dump()
    values.update(policy["settings"])
    if subtitles_only:
        values["separation_enabled"] = False
        values["separation_mix_mode"] = "bilingual"
        values["spatial_rtf_enabled"] = False
    else:
        values["separation_mix_mode"] = policy["variants"][0]
        values["mix_output_mode"] = "mixed"
    project.settings = ProjectSettings.model_validate(values)
    if not project.sentences:
        project.translation_language = project.settings.tts_target_language
    save_project(project, directory)


def render_variants(paths: Any, project_json: Path, folder: Path, state: dict[str, Any]) -> None:
    from . import engine as e

    policy = state["output_policy"]
    variants = policy["variants"]
    project, directory = load_project(project_json)
    original_settings = project.settings.model_copy(deep=True)
    records = state.setdefault("mix_variants", {})
    first_project = None
    try:
        for variant in variants:
            project, directory = load_project(project_json)
            project.settings.separation_mix_mode = variant
            project.settings.mix_output_mode = "mixed"
            save_project(project, directory)
            e.run_asmr_cli(paths, "mix", str(project_json), "--output-variant", variant)
            e.run_asmr_cli(
                paths,
                "subtitles",
                str(project_json),
                "--language",
                policy["subtitle"],
                "--output-variant",
                variant,
            )
            project, directory = load_project(project_json)
            audio = e.project_asset(project_json, project.output_file)
            if audio is None:
                raise e.VideoPreparerError("混音未生成音频。")
            cached = audio
            destination = folder / ("双语版" if variant == "bilingual" else "替换配音版")
            destination.mkdir(parents=True, exist_ok=True)
            outputs = e.copy_final_outputs(
                paths,
                project_json,
                destination,
                str(state["mode"]),
                harmonized_delay_seconds=int(state["harmonized_delay_seconds"]),
                harmonized_volume_db=float(state["harmonized_volume_db"]),
                embed_subtitles=bool(state.get("embed_subtitles", True)),
                output_stem="双语版" if variant == "bilingual" else "配音版",
            )
            records[variant] = {"audio": str(cached), "outputs": outputs}
            if first_project is None:
                first_project = project.model_copy(deep=True)
    finally:
        project, directory = load_project(project_json)
        if first_project is not None:
            for field in (
                "output_video_file",
                "subtitle_video_file",
                "subtitle_srt_file",
                "subtitle_lrc_file",
            ):
                setattr(project, field, getattr(first_project, field))
        project.settings = original_settings
        if records.get(variants[0]):
            project.output_file = records[variants[0]]["audio"]
        save_project(project, directory)
    first = records[variants[0]]
    state["project_mixed_audio"] = first["audio"]
    state["variant_outputs"] = dict(first["outputs"])
    for variant, record in records.items():
        state["variant_outputs"].update(
            {f"{variant}_{key}": value for key, value in record["outputs"].items()}
        )
