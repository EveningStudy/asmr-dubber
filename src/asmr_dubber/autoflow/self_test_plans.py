"""Self-test checks for settings, catalogue scanning and plan drafts."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from .app_settings import load_app_config, load_smart_plan_defaults, save_smart_plan_defaults
from .catalog import (
    scan_work,
)
from .configure import configure_smart_plan_draft
from .constants import (
    LAYOUT_BOTH,
    LAYOUT_MERGED,
    MODE_AUDIO,
    MODE_VIDEO_HARMONIZED,
    MODE_VIDEO_NORMAL,
)
from .domain import (
    AppConfig,
    SmartPlanDefaults,
)
from .drafts import prepare_smart_plan_draft
from .errors import VideoPreparerError
from .source_scan import (
    default_edition_for_work,
)
from .track_prompts import choose_tracks


def _check_settings_and_defaults(root: Path) -> tuple[AppConfig, SmartPlanDefaults]:
    test_settings = root / "settings.txt"
    test_settings.write_text(
        "asmr_dubber_path=.\n"
        "harmonized_volume_reduction_db=12.5\n"
        "harmonized_delay_minutes=7.5\n"
        "timestamp_footer_line_1=测试页脚\n",
        encoding="utf-8",
    )
    parsed_config = load_app_config(test_settings)
    if (
        parsed_config.asmr_root != root.resolve()
        or parsed_config.harmonized_volume_db != -12.5
        or parsed_config.harmonized_delay_seconds != 450
        or parsed_config.timestamp_footer != "测试页脚"
    ):
        raise VideoPreparerError("自检失败：settings.txt 解析错误。")
    remembered_defaults = SmartPlanDefaults(
        mode=MODE_AUDIO,
        video_mode=MODE_VIDEO_HARMONIZED,
        layout=LAYOUT_BOTH,
        include_bonus=True,
        background_choice="relative",
        background_relative="cover.jpg",
        embed_subtitles=False,
        edition_extension=".wav",
        edition_language="ja",
        edition_mix_variant="standard",
        edition_orientation="standard",
    )
    remembered_path = root / "last-options.json"
    save_smart_plan_defaults(remembered_defaults, remembered_path)
    if load_smart_plan_defaults(parsed_config, remembered_path) != remembered_defaults:
        raise VideoPreparerError("自检失败：最后一次选项没有正确保存或恢复。")
    return parsed_config, remembered_defaults


def _check_catalogue(
    root: Path, parsed_config: AppConfig, remembered_defaults: SmartPlanDefaults
) -> Path:
    catalogue_root = root / "catalogue"
    wav_dir = catalogue_root / "WAV"
    mp3_dir = catalogue_root / "MP3"
    bonus_dir = catalogue_root / "特典"
    wav_dir.mkdir(parents=True)
    mp3_dir.mkdir(parents=True)
    bonus_dir.mkdir(parents=True)
    shutil.copy2(root / "1 开场.wav", wav_dir / "Track０１ 開場.wav")
    shutil.copy2(root / "1 开场.wav", mp3_dir / "Track01 開場.mp3")
    shutil.copy2(root / "2 囁き.flac", bonus_dir / "EX01 おまけ.wav")
    (wav_dir / "Track０１ 開場.wav.vtt").write_text(
        "WEBVTT\n\n00:00.000 --> 00:00.500\n開場\n",
        encoding="utf-8",
    )
    plain_lrc = catalogue_root / "字幕" / "01.lrc"
    plain_lrc.parent.mkdir(parents=True)
    plain_lrc.write_text("[00:00.00]中文台词\n", encoding="utf-8")
    japanese_lrc = catalogue_root / "日文" / "line.lrc"
    japanese_lrc.parent.mkdir(parents=True)
    japanese_lrc.write_text("[00:00.00]日本語\n", encoding="utf-8")
    catalogue = scan_work(catalogue_root)
    transcript_languages = {item.path.resolve(): item.language for item in catalogue.transcripts}
    if (
        transcript_languages.get(plain_lrc.resolve()) != "zh"
        or transcript_languages.get(japanese_lrc.resolve()) != "ja"
    ):
        raise VideoPreparerError("自检失败：普通 LRC 或明确日文目录的语言判断错误。")
    wav_edition = next(
        (
            item
            for item in catalogue.editions
            if item.extension == ".wav" and item.directory == "WAV"
        ),
        None,
    )
    if (
        wav_edition is None
        or wav_edition.tracks[0].order_key[1] != 1
        or wav_edition.tracks[0].transcript is None
    ):
        raise VideoPreparerError("自检失败：全角 Track 编号或 VTT 自动匹配错误。")
    _, selected_with_bonus, _ = choose_tracks(
        catalogue,
        parsed_config,
        edition_argument=wav_edition.id,
        include_bonus=True,
    )
    if len(selected_with_bonus) != 2 or selected_with_bonus[-1].category != "bonus":
        raise VideoPreparerError("自检失败：独立特典目录没有加入所选版本。")
    inherited_edition = default_edition_for_work(
        catalogue,
        parsed_config,
        remembered_defaults,
    )
    if inherited_edition.extension != ".wav":
        raise VideoPreparerError("自检失败：下一个作品没有沿用上次的音频格式。")
    return catalogue_root


def _check_draft_inheritance(
    parsed_config: AppConfig, remembered_defaults: SmartPlanDefaults, catalogue_root: Path
) -> None:
    queue_args = argparse.Namespace(
        mode=None,
        layout=None,
        edition=None,
        include_bonus=False,
        output_root=None,
        background=None,
        embed_subtitles=None,
        rebuild=False,
        force=False,
    )
    inherited_draft = prepare_smart_plan_draft(
        parsed_config,
        catalogue_root,
        remembered_defaults,
        queue_args,
    )
    if (
        inherited_draft.mode != MODE_AUDIO
        or inherited_draft.video_mode != MODE_VIDEO_HARMONIZED
        or inherited_draft.layout != LAYOUT_BOTH
        or not inherited_draft.include_bonus
        or inherited_draft.embed_subtitles
    ):
        raise VideoPreparerError("自检失败：作品选项没有从上一个项目继承。")
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as sink,
        redirect_stdout(sink),
        redirect_stderr(sink),
        patch(
            "builtins.input",
            side_effect=("3", "2", "4", "1", "5", "1", "0"),
        ),
    ):
        inherited_plan, next_defaults, start_now = configure_smart_plan_draft(
            inherited_draft,
            parsed_config,
            index=1,
            rebuild=False,
            force=False,
        )
    if (
        not start_now
        or inherited_plan.mode != MODE_VIDEO_NORMAL
        or inherited_plan.layout != LAYOUT_MERGED
        or next_defaults.mode != MODE_VIDEO_NORMAL
        or next_defaults.video_mode != MODE_VIDEO_NORMAL
    ):
        raise VideoPreparerError("自检失败：默认选项面板没有正确修改或开始任务。")
