"""Self-test checks for audio mode outputs, video rendering and subtitle files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import media_video, output_files, processes
from .constants import (
    DEFAULT_HARMONIZED_VOLUME_REDUCTION_DB,
    KEYFRAME_INTERVAL_SECONDS,
    MODE_AUDIO,
    MODE_VIDEO_NORMAL,
)
from .domain import (
    AppConfig,
    AudioSource,
    ToolPaths,
)
from .errors import VideoPreparerError
from .media_video import remux_video_without_subtitles, render_delayed_existing_video
from .processes import (
    audio_duration_samples,
    media_duration_seconds,
    media_stream_types,
    video_keyframe_times,
)
from .subtitle_files import combine_lrc_files, combine_srt_files, shift_lrc_text, shift_srt_text
from .task_media import prepare_media_phase
from .work_state import create_initial_state

_TEST_HARMONIZED_VOLUME_DB = -DEFAULT_HARMONIZED_VOLUME_REDUCTION_DB


def _check_audio_mode(
    paths: ToolPaths, root: Path, discovered: list[AudioSource], total_samples: int
) -> None:
    audio_output_folder = root / "audio-mode"
    audio_output_folder.mkdir()
    audio_state = create_initial_state(
        audio_output_folder,
        MODE_AUDIO,
        discovered,
        None,
        AppConfig(
            asmr_root=None,
            harmonized_volume_db=-10.0,
            harmonized_delay_seconds=1_200,
            timestamp_footer="",
        ),
    )
    audio_state["workspace"] = str(root / "audio-mode-work")
    prepare_media_phase(paths, audio_output_folder, audio_state, discovered)
    audio_original = audio_output_folder / "原声.flac"
    if (
        audio_state["status"] != "media_ready"
        or not audio_original.is_file()
        or audio_duration_samples(paths, audio_original) != total_samples
    ):
        raise VideoPreparerError("自检失败：纯音频模式输出错误。")

    fake_project = root / "audio-project"
    (fake_project / "output").mkdir(parents=True)
    (fake_project / "subtitles").mkdir()
    mixed_wav = fake_project / "output" / "mixed.wav"
    processes.run_ffmpeg(
        paths,
        [
            "-loglevel",
            "error",
            "-i",
            str(audio_original),
            "-c:a",
            "pcm_s16le",
            str(mixed_wav),
        ],
    )
    (fake_project / "subtitles" / "bilingual.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n测试\n",
        encoding="utf-8",
    )
    (fake_project / "subtitles" / "bilingual.lrc").write_text(
        "[00:00.00]测试\n",
        encoding="utf-8",
    )
    fake_project_json = fake_project / "project.json"
    fake_project_json.write_text(
        json.dumps(
            {
                "output_file": "output/mixed.wav",
                "subtitle_srt_file": "subtitles/bilingual.srt",
                "subtitle_lrc_file": "subtitles/bilingual.lrc",
            }
        ),
        encoding="utf-8",
    )
    audio_final_folder = root / "audio-final"
    audio_outputs = output_files.copy_final_outputs(
        paths,
        fake_project_json,
        audio_final_folder,
        MODE_AUDIO,
        harmonized_delay_seconds=1_200,
        harmonized_volume_db=-10.0,
        embed_subtitles=False,
    )
    if not all(Path(path).is_file() for path in audio_outputs.values()):
        raise VideoPreparerError("自检失败：纯音频模式最终文件回写错误。")


def _check_video_renders(
    paths: ToolPaths, root: Path, master: Path, background: Path
) -> tuple[Path, float]:
    normal = root / "normal.mp4"
    black = root / "black.mp4"
    harmony = root / "harmony.mp4"
    media_video.render_static_video(paths, master, background, normal)
    media_video.render_static_video(paths, master, None, black)
    media_video.render_static_video(
        paths,
        master,
        background,
        harmony,
        lead_seconds=KEYFRAME_INTERVAL_SECONDS,
        volume_db=_TEST_HARMONIZED_VOLUME_DB,
    )
    normal_duration = media_duration_seconds(paths, normal)
    black_duration = media_duration_seconds(paths, black)
    harmony_duration = media_duration_seconds(paths, harmony)
    if not (1.9 <= normal_duration <= 2.3):
        raise VideoPreparerError(f"自检失败：普通视频时长异常 {normal_duration:.3f}s")
    if not (1.9 <= black_duration <= 2.3):
        raise VideoPreparerError(f"自检失败：黑色背景视频时长异常 {black_duration:.3f}s")
    if not (
        normal_duration + KEYFRAME_INTERVAL_SECONDS - 0.2
        <= harmony_duration
        <= normal_duration + KEYFRAME_INTERVAL_SECONDS + 0.2
    ):
        raise VideoPreparerError(
            f"自检失败：前置静音时长异常 {harmony_duration - normal_duration:.3f}s"
        )
    keyframes = video_keyframe_times(paths, harmony)
    if len(keyframes) < 2 or not any(
        abs(timestamp - KEYFRAME_INTERVAL_SECONDS) <= 0.1 for timestamp in keyframes
    ):
        raise VideoPreparerError(f"自检失败：没有按 10 秒间隔写入关键帧：{keyframes}")
    return normal, normal_duration


def _check_subtitle_files_and_video(
    paths: ToolPaths, root: Path, master: Path, normal: Path, normal_duration: float
) -> None:
    sample_srt = "1\n00:00:01,250 --> 00:00:02,500\n测试\n"
    shifted = shift_srt_text(sample_srt, 1_200_000)
    if "00:20:01,250 --> 00:20:02,500" not in shifted:
        raise VideoPreparerError("自检失败：SRT 偏移错误。")
    shifted_file = root / "shifted.srt"
    shifted_file.write_text(shifted, encoding="utf-8")
    second_srt = root / "second.srt"
    second_srt.write_text(
        "99\n00:00:00,100 --> 00:00:00,300\n第二段\n",
        encoding="utf-8",
    )
    combined_srt = combine_srt_files(
        ((shifted_file, -1_200_000, 2_000), (second_srt, 2_000, 3_000)),
        root / "combined.srt",
    ).read_text(encoding="utf-8-sig")
    if (
        "1\n00:00:01,250 --> 00:00:02,000" not in combined_srt
        or "2\n00:00:02,100" not in combined_srt
    ):
        raise VideoPreparerError("自检失败：多文件 SRT 合并或重新编号错误。")
    delayed = root / "delayed.mp4"
    render_delayed_existing_video(
        paths,
        normal,
        delayed,
        lead_seconds=2,
        subtitle_file=shifted_file,
        volume_db=_TEST_HARMONIZED_VOLUME_DB,
        audio_source=master,
    )
    delayed_duration = media_duration_seconds(paths, delayed)
    if not (normal_duration + 1.8 <= delayed_duration <= normal_duration + 2.2):
        raise VideoPreparerError(
            f"自检失败：双语视频前导时长异常 {delayed_duration - normal_duration:.3f}s"
        )
    if "subtitle" not in media_stream_types(paths, delayed):
        raise VideoPreparerError("自检失败：开启字幕后视频中没有字幕流。")
    delayed_without_subtitle = root / "delayed-without-subtitle.mp4"
    render_delayed_existing_video(
        paths,
        normal,
        delayed_without_subtitle,
        lead_seconds=2,
        subtitle_file=None,
        volume_db=_TEST_HARMONIZED_VOLUME_DB,
        audio_source=master,
    )
    if "subtitle" in media_stream_types(paths, delayed_without_subtitle):
        raise VideoPreparerError("自检失败：关闭字幕后视频仍包含字幕流。")
    remuxed_without_subtitle = root / "remuxed-without-subtitle.mp4"
    remux_video_without_subtitles(paths, delayed, remuxed_without_subtitle)
    remuxed_streams = media_stream_types(paths, remuxed_without_subtitle)
    if "subtitle" in remuxed_streams or not {"video", "audio"}.issubset(remuxed_streams):
        raise VideoPreparerError("自检失败：无字幕重封装的媒体流错误。")
    sample_lrc = "[00:01.25]测试\n"
    if "[20:01.25]" not in shift_lrc_text(sample_lrc, 1_200_000):
        raise VideoPreparerError("自检失败：LRC 偏移错误。")
    first_lrc = root / "first.lrc"
    second_lrc = root / "second.lrc"
    first_lrc.write_text("[ar:测试]\n[00:01.25]第一段\n", encoding="utf-8")
    second_lrc.write_text("[ar:忽略]\n[00:00.10]第二段\n", encoding="utf-8")
    combined_lrc = combine_lrc_files(
        ((first_lrc, 0), (second_lrc, 2_000)),
        root / "combined.lrc",
    ).read_text(encoding="utf-8-sig")
    if "[ar:测试]" not in combined_lrc or "[00:02.10]第二段" not in combined_lrc:
        raise VideoPreparerError("自检失败：多文件 LRC 合并错误。")


def _check_timestamp_document(root: Path, timeline: list[dict[str, Any]]) -> None:
    timestamp_state = {
        "source_folder": str(root / "日本語作品"),
        "mode": MODE_VIDEO_NORMAL,
        "timeline": timeline,
        "title_translations": {
            str(item["filename"]): f"中文曲目{index}"
            for index, item in enumerate(timeline, start=1)
        },
        "folder_name_original": "日本語作品",
        "folder_name_translation": "中文作品",
    }
    timestamp_text = output_files.write_timestamp_document(timestamp_state, root).read_text(
        encoding="utf-8-sig"
    )
    if "中文名称：中文作品" not in timestamp_text or "原始名称：日本語作品" not in timestamp_text:
        raise VideoPreparerError("自检失败：时间戳文档缺少文件夹中英文名称。")
