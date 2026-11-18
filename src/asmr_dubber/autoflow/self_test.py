"""Built-in self test."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from . import media_audio, processes, runlog, smart_queue
from .catalog import (
    scan_work,
)
from .constants import (
    LAYOUT_BOTH,
    LAYOUT_MERGED,
    LAYOUT_SEPARATE,
    MODE_AUDIO,
)
from .domain import (
    AppConfig,
    AudioSource,
    SmartTaskPlan,
    ToolPaths,
)
from .errors import VideoPreparerError
from .failed_tasks import read_failed_task_records, write_failed_task_records
from .media_audio import concat_audio_files
from .modes import parse_embed_subtitles_argument
from .plan_files import smart_job_descriptors
from .processes import (
    audio_duration_samples,
)
from .prompts import parse_yes_no
from .self_test_media import (
    _check_audio_mode,
    _check_subtitle_files_and_video,
    _check_timestamp_document,
    _check_video_renders,
)
from .self_test_plans import (
    _check_catalogue,
    _check_draft_inheritance,
    _check_settings_and_defaults,
)
from .source_scan import (
    _background_from_argument,
    discover_audio,
    discover_background,
    source_from_candidate,
)
from .title_translation import identifier_only_name


def _generate_media(paths: ToolPaths, root: Path) -> None:
    sources = (
        ("1 开场.wav", "sine=frequency=440:duration=0.60", "44100", "1"),
        ("2 囁き.flac", "sine=frequency=660:duration=0.70", "48000", "2"),
        ("10 終了.m4a", "sine=frequency=880:duration=0.80", "32000", "1"),
    )
    for name, generator, rate, channels in sources:
        processes.run_ffmpeg(
            paths,
            [
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                generator,
                "-ar",
                rate,
                "-ac",
                channels,
                str(root / name),
            ],
        )
    processes.run_ffmpeg(
        paths,
        [
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:s=641x479",
            "-frames:v",
            "1",
            str(root / "null.png"),
        ],
    )


def _check_scan_and_parsing(root: Path) -> tuple[list[AudioSource], Path, list[AudioSource]]:
    discovered = discover_audio(root)
    if [item.order for item in discovered] != [1, 2, 10]:
        raise VideoPreparerError("自检失败：数字排序错误。")
    background = discover_background(root)
    if background is None:
        raise VideoPreparerError("自检失败：没有识别 null 图片。")
    smart_scan = scan_work(root, excluded_directories=("AutoFlow输出",))
    if smart_scan.audio_count != 3 or not smart_scan.editions:
        raise VideoPreparerError("自检失败：智能扫描没有识别传统数字音轨。")
    if (
        not identifier_only_name("RJ01563553")
        or not identifier_only_name("VJ_00123")
        or identifier_only_name("RJ01563553 作品标题")
    ):
        raise VideoPreparerError("自检失败：作品编号文件夹识别错误。")
    if (
        _background_from_argument(smart_scan, "auto") != smart_scan.images[0]
        or _background_from_argument(smart_scan, "1") != smart_scan.images[0]
        or _background_from_argument(smart_scan, "0") is not None
    ):
        raise VideoPreparerError("自检失败：推荐背景、编号选择或黑色背景错误。")
    if (
        parse_yes_no("Y") is not True
        or parse_yes_no("n") is not False
        or parse_yes_no("") is not None
        or parse_embed_subtitles_argument("yes") is not True
        or parse_embed_subtitles_argument("no") is not False
    ):
        raise VideoPreparerError("自检失败：Y/N 或字幕选项解析错误。")
    smart_sources = [
        source_from_candidate(index, item)
        for index, item in enumerate(smart_scan.editions[0].tracks, start=1)
    ]
    if [item.order for item in smart_sources] != [1, 2, 3]:
        raise VideoPreparerError("自检失败：智能扫描输出顺序错误。")
    if (
        len(smart_job_descriptors(root / "out", LAYOUT_MERGED, smart_sources)) != 1
        or len(smart_job_descriptors(root / "out", LAYOUT_SEPARATE, smart_sources)) != 3
        or len(smart_job_descriptors(root / "out", LAYOUT_BOTH, smart_sources)) != 4
    ):
        raise VideoPreparerError("自检失败：三种输出布局的任务规划错误。")
    return discovered, background, smart_sources


def _check_queue_continues(paths: ToolPaths, root: Path, smart_sources: list[AudioSource]) -> None:
    queue_plan_base = {
        "output_root": root / "queue-output",
        "edition_label": "测试版本",
        "sources": tuple(smart_sources),
        "edition": {},
        "mode": MODE_AUDIO,
        "layout": LAYOUT_MERGED,
        "background": None,
        "embed_subtitles": False,
        "rebuild": False,
        "force": False,
    }
    queue_plans = [
        SmartTaskPlan(
            folder=root / "queue-first",
            plan_id="queue-first",
            **queue_plan_base,
        ),
        SmartTaskPlan(
            folder=root / "queue-second",
            plan_id="queue-second",
            **{**queue_plan_base, "output_root": root / "queue-output-2"},
        ),
    ]
    queue_calls: list[str] = []

    def fake_queue_executor(
        _paths: ToolPaths,
        _config: AppConfig,
        task: SmartTaskPlan,
    ) -> None:
        queue_calls.append(task.plan_id)
        if task.plan_id == "queue-first":
            raise VideoPreparerError("预期的队列测试失败")

    existing_failed_records = read_failed_task_records()
    try:
        with runlog.bind_autoflow_callbacks(on_message=None, on_error=None):
            queue_result = smart_queue.execute_smart_queue(
                paths,
                AppConfig(
                    asmr_root=None,
                    harmonized_volume_db=-10.0,
                    harmonized_delay_seconds=1_200,
                    timestamp_footer="",
                ),
                queue_plans,
                executor=fake_queue_executor,
            )
    finally:
        # The queue test deliberately fails one item. Do not leave its
        # temporary paths in the user's real retry list after self-test.
        write_failed_task_records(existing_failed_records)
    if queue_result != 1 or queue_calls != ["queue-first", "queue-second"]:
        raise VideoPreparerError("自检失败：队列没有在单项失败后继续处理。")


def _check_master_audio(
    paths: ToolPaths, root: Path, discovered: list[AudioSource]
) -> tuple[Path, list[dict[str, Any]], int]:
    workspace = root / "work"
    workspace.mkdir()
    master, timeline = media_audio.normalize_and_concat(paths, discovered, workspace)
    total_samples = sum(int(item["duration_samples"]) for item in timeline)
    if audio_duration_samples(paths, master) != total_samples:
        raise VideoPreparerError("自检失败：母带采样数错误。")
    rebuilt_master, rebuilt_lengths = concat_audio_files(
        paths,
        [Path(str(item["normalized"])) for item in timeline],
        root / "exact-concat",
        expected_samples=[int(item["duration_samples"]) for item in timeline],
        name="self-test",
    )
    if rebuilt_lengths != [int(item["duration_samples"]) for item in timeline] or (
        audio_duration_samples(paths, rebuilt_master) != total_samples
    ):
        raise VideoPreparerError("自检失败：分轨成品的精确二次合并错误。")
    return master, timeline, total_samples


def self_test(paths: ToolPaths) -> None:
    runlog.emit("运行轻量自检；不会调用 ASR、TTS 或网络……")
    with tempfile.TemporaryDirectory(prefix="video-preparer-test-") as temporary:
        root = Path(temporary)
        _generate_media(paths, root)
        discovered, background, smart_sources = _check_scan_and_parsing(root)
        _check_queue_continues(paths, root, smart_sources)
        master, timeline, total_samples = _check_master_audio(paths, root, discovered)
        parsed_config, remembered_defaults = _check_settings_and_defaults(root)
        catalogue_root = _check_catalogue(root, parsed_config, remembered_defaults)
        _check_draft_inheritance(parsed_config, remembered_defaults, catalogue_root)
        _check_audio_mode(paths, root, discovered, total_samples)
        normal, normal_duration = _check_video_renders(paths, root, master, background)
        _check_subtitle_files_and_video(paths, root, master, normal, normal_duration)
        _check_timestamp_document(root, timeline)
    runlog.emit("自检通过。")
