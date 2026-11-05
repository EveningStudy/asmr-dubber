"""Upgrade queued subtitle intent without reusing the old execution checkpoints."""

from copy import deepcopy
from pathlib import Path

from ..autoflow import engine as e


def normalize_subtitle_plan(payload):
    policy = payload.get("edition", {}).get("output_policy", {})
    content = policy.get("content")
    if content not in {"subtitles", "source_subtitles"} or policy.get("content_version") == 2:
        return payload
    result = deepcopy(payload)
    result["edition"]["output_policy"]["content_version"] = 2
    files_only = content == "subtitles"
    result.update(subtitles_only=True, source_subtitles_only=files_only)
    if files_only:
        result.update(mode=e.MODE_AUDIO, background=None, embed_subtitles=False)
    # A new identity prevents old media exports/checkpoints from being mistaken
    # for corrected results. Existing outputs still require explicit rebuild.
    result["plan_id"] = e.plan_identity(
        Path(result["folder"]),
        mode=result["mode"],
        layout=result["layout"],
        edition=result["edition"],
        sources=[e.deserialize_audio_source(s) for s in result["sources"]],
        output_root=Path(result["output_root"]),
        background=Path(result["background"]) if result.get("background") else None,
        embed_subtitles=result.get("embed_subtitles", True),
        translate_work_title=result.get("translate_work_title", True),
        translate_track_titles=result.get("translate_track_titles", True),
        subtitles_only=True,
        source_subtitles_only=files_only,
        subtitle_language=result.get("subtitle_language", "source"),
        subtitle_naming=result.get("subtitle_naming", "standard"),
        subtitle_custom_name=result.get("subtitle_custom_name", "字幕"),
    )
    return result
