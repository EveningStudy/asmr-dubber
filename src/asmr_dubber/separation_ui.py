"""Experimental separation/RTF controls, isolated from ordinary dubbing settings."""

from __future__ import annotations

import json
from typing import Any

from .separation import model_directory, prepare_local_model, verify_local_model
from .user_settings import clear_service_key, save_service_key


def build_controls(gr, stored, components):
    gr.Markdown(
        "## 人声分离\n**实验性，不推荐。普通用户请保持关闭。** "
        "分离可能损伤耳语、呼吸和口腔声；背景可能残留日语。RTF（原声空间线索迁移）与混音方式请到“混音与字幕”设置。"
    )

    def checkbox(name, label, info=""):
        components[name] = gr.Checkbox(label=label, value=getattr(stored, name), info=info)
        return components[name]

    def number(name, label, lo, hi, step: float = 1, info=""):
        components[name] = gr.Slider(
            label=label, value=getattr(stored, name), minimum=lo, maximum=hi, step=step, info=info
        )

    def text(name, label, info="", lines=1):
        components[name] = gr.Textbox(
            label=label, value=getattr(stored, name), info=info, lines=lines
        )

    def radio(name, label, choices):
        components[name] = gr.Radio(label=label, choices=choices, value=getattr(stored, name))
        return components[name]

    sep_toggle = checkbox(
        "separation_enabled",
        "启用人声分离（实验性，不推荐）",
        "开启后，ASR 使用分离人声；原录音保留作为 RTF 参考和双语混音源。",
    )
    # Keep configuration available while disabled; only the explicit enable flag
    # controls execution. Nested visibility changes can race during page hydration.
    with gr.Group() as sep_group:
        gr.Markdown("下面参数可预先配置；未勾选启用时不会执行分离。")
        backend = radio(
            "separation_backend",
            "分离后端",
            [
                ("本地 audio-separator", "local"),
                ("Replicate 云端", "replicate"),
                ("自建 HTTP 兼容接口", "http"),
            ],
        )
        with gr.Group(visible=stored.separation_backend == "local") as local_group:
            gr.Markdown(
                "支持 audio-separator 的 Mel/BS-RoFormer、MDX/MDX23C、VR、Demucs 模型接口。"
                "本机优先试用 Kimberley Jensen Mel-Band RoFormer；这不是 ASMR 质量保证。"
                "Bandit/SAM 未直接适配，不会伪装成已支持的本地模型。"
            )
            choices = [("Mel-Band RoFormer · Kimberley Jensen", "vocals_mel_band_roformer.ckpt")]
            catalog_path = model_directory() / "catalog.json"
            if catalog_path.is_file():
                try:
                    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
                    choices = [
                        (f"{family} · {label}", data["filename"])
                        for family, entries in catalog.items()
                        for label, data in entries.items()
                    ]
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            components["separation_model"] = gr.Dropdown(
                label="分离模型（本地目录清单，不代表已下载）",
                choices=choices,
                value=stored.separation_model,
                allow_custom_value=True,
                info="切换模型后必须显式下载。清单在成功下载时更新，重启页面服务器载入。",
            )
            radio("separation_device", "分离设备", [("CUDA", "cuda"), ("CPU（较慢）", "cpu")])
            checkbox(
                "separation_use_autocast",
                "自动混合精度",
                "降低显存；不稳定时关闭。ONNX 架构是否使用 GPU 取决于独立环境的执行提供程序。",
            )
            number(
                "separation_normalization",
                "分离归一化峰值",
                0.1,
                1,
                0.01,
                "默认 1；过度归一化可能改变残差听感。",
            )
            text(
                "separation_vocal_stem",
                "作为人声的输出轨名称",
                "通常 Vocals。切换多轨模型时必须确认此名称；其余原声通过互补残差保留。",
            )
            with gr.Accordion("模型高级参数（仅所选架构生效）", open=False):
                text(
                    "separation_common_params",
                    "通用高级参数（JSON）",
                    "invert_using_spec 频谱反相；use_torch_compile 编译；"
                    "use_native_fp16 原生半精度（与混合精度互斥）；"
                    "amplification_threshold 最小放大电平。",
                    3,
                )
                gr.Markdown(
                    "只开放推理参数，不修改训练结构。JSON 参数完整透传到固定版本 audio-separator；"
                    "不支持的字段会报错。不要任意增大批大小。"
                )
                text(
                    "separation_mdxc_params",
                    "Mel/BS-RoFormer 与 MDX23C 参数",
                    "segment_size 分段；override_model_segment_size 是否覆盖；"
                    "batch_size 批大小；overlap 重叠次数；pitch_shift 半音偏移。",
                    4,
                )
                text(
                    "separation_mdx_params",
                    "MDX 参数",
                    "hop_length、segment_size、overlap(0–1)、batch_size、enable_denoise。",
                    3,
                )
                text(
                    "separation_vr_params",
                    "VR 参数",
                    "batch_size、window_size、aggression、enable_tta、enable_post_process、post_process_threshold、high_end_process。",
                    4,
                )
                text(
                    "separation_demucs_params",
                    "Demucs 参数",
                    "segment_size、shifts、overlap(0–1)、segments_enabled。",
                    3,
                )
            consent = gr.Checkbox(
                label=(
                    "允许本次从 PyPI/PyTorch/GitHub/Hugging Face 下载"
                    "（可能数 GB；权重遵循各自许可）"
                ),
                value=False,
            )
            with gr.Row():
                install = gr.Button("安装独立环境并下载模型")
                download = gr.Button("仅下载/校验模型")
                check = gr.Button("检查本地模型完整性")
            install_status = gr.Textbox(label="实验分离安装状态", lines=5, interactive=False)
        with gr.Group(visible=stored.separation_backend != "local") as cloud_group:
            gr.Markdown(
                "**会向你选择的服务商发送音频，可能产生费用。** 不自动部署云服务。"
                "Replicate 填固定版本 ID；不同模型的输入/输出字段需按供应商文档填写。"
                "自建协议：multipart 音频 + model + parameters，返回含人声 HTTPS URL 的 JSON。"
            )
            checkbox("separation_cloud_consent", "同意上传音频到所选分离服务（实验性，不推荐）")
            text(
                "separation_api_url",
                "自建 API 完整地址",
                "仅自建 HTTP 使用；外网必须 HTTPS。本机可 http://127.0.0.1。",
            )
            text(
                "separation_api_model",
                "云端模型／固定版本 ID",
                "Replicate 必须为 64 位十六进制版本 ID；自建接口填写其模型 ID。",
            )
            text(
                "separation_api_audio_field",
                "输入音频字段名",
                "常见 audio 或 audio_file；不是下载地址。",
            )
            text(
                "separation_api_output_field",
                "人声 URL 的 JSON 字段路径",
                "Replicate 相对 output；如 vocals、0；空值表示输出本身为 URL。",
            )
            text(
                "separation_api_params",
                "云模型其余参数（JSON）",
                "服务商参数完整透传；不能覆盖音频字段。请求失败不自动重发计费任务。",
                4,
            )
            api_key = gr.Textbox(label="分离 API Key", type="password")
            with gr.Row():
                save_key = gr.Button("保存分离 Key")
                clear_key = gr.Button("清除分离 Key")
            key_status = gr.Textbox(label="分离密钥状态", interactive=False)
        number(
            "separation_chunk_seconds",
            "外层分块秒数",
            5,
            300,
            1,
            "默认 30，长音频逐块处理。云端建议 ≤30，避免请求体过大。",
        )
        number(
            "separation_overlap_seconds",
            "分块上下文秒数",
            0,
            3,
            0.1,
            "默认 1；边缘上下文裁掉后放回原时间轴。不是保证无接缝。",
        )
        number("separation_timeout_seconds", "每块超时秒数", 30, 14400, 30)

    backend.change(
        lambda value: (gr.update(visible=value == "local"), gr.update(visible=value != "local")),
        inputs=[backend],
        outputs=[local_group, cloud_group],
        queue=False,
        api_name=False,
    )

    def install_model(model, confirmed, environment=False):
        if not confirmed:
            yield "未执行，请先明确同意下载来源和可能的磁盘占用。"
            return
        yield "正在安装/下载实验分离资源；不修改项目、模型选择或功能开关。"
        try:
            yield prepare_local_model(model, install=environment)
        except Exception as exc:
            yield f"安装未完成：{exc}"

    def install_environment(model, confirmed):
        yield from install_model(model, confirmed, True)

    install.click(
        install_environment,
        inputs=[components["separation_model"], consent],
        outputs=[install_status],
        concurrency_id="runtime_mutation",
        concurrency_limit=1,
        api_name=False,
    )
    download.click(
        install_model,
        inputs=[components["separation_model"], consent],
        outputs=[install_status],
        concurrency_id="runtime_mutation",
        concurrency_limit=1,
        api_name=False,
    )

    def check_model(model):
        try:
            return "本地文件校验通过：" + verify_local_model(model) + "；不代表分离听感合格。"
        except Exception as exc:
            return str(exc)

    check.click(
        check_model,
        inputs=[components["separation_model"]],
        outputs=[install_status],
        concurrency_id="runtime_mutation",
        api_name=False,
    )

    def persist_key(provider, value):
        if provider == "local":
            return "本地后端不需要 Key。", ""
        save_service_key("separation:" + provider, value)
        return "密钥已保存（只存本机，不写入项目）。", ""

    save_key.click(
        persist_key, inputs=[backend, api_key], outputs=[key_status, api_key], api_name=False
    )
    clear_key.click(
        lambda provider: clear_service_key("separation:" + provider) or "密钥已清除。",
        inputs=[backend],
        outputs=[key_status],
        api_name=False,
    )
    return {
        "fn": lambda enabled, provider: (
            gr.update(visible=True),
            gr.update(visible=provider == "local"),
            gr.update(visible=provider != "local"),
        ),
        "inputs": [sep_toggle, backend],
        "outputs": [sep_group, local_group, cloud_group],
        "queue": False,
        "api_name": False,
    }


def build_mix_controls(gr, stored, components) -> dict[str, Any]:
    def checkbox(name, label, info=""):
        components[name] = gr.Checkbox(label=label, value=getattr(stored, name), info=info)
        return components[name]

    def number(name, label, lo, hi, step: float = 1, info=""):
        components[name] = gr.Slider(
            label=label, value=getattr(stored, name), minimum=lo, maximum=hi, step=step, info=info
        )

    def text(name, label, info="", lines=1):
        components[name] = gr.Textbox(
            label=label, value=getattr(stored, name), info=info, lines=lines
        )

    def radio(name, label, choices):
        components[name] = gr.Radio(label=label, choices=choices, value=getattr(stored, name))
        return components[name]

    gr.Markdown(
        "### 混音方式\n中文替换为实验性，不推荐。普通双语保留完整原声；RTF 独立控制中文空间处理。"
    )
    mode = radio(
        "separation_mix_mode",
        "混音方式",
        [
            ("普通双语：原音 + 中文", "bilingual"),
            ("中文替换：分离背景 + 中文（必须开启分离，实验性，不推荐）", "replace"),
        ],
    )
    rtf_toggle = checkbox(
        "spatial_rtf_enabled",
        "启用 RTF（原声空间线索迁移）",
        "不依赖人声分离，双语和替换模式均可使用。仅处理中文，不混入原日语波形。",
    )
    with gr.Accordion("RTF（原声空间线索迁移）参数（未启用时不生效）", open=False):
        number("spatial_rtf_strength", "RTF 干湿比例", 0, 1, 0.05)
        number("spatial_rtf_level_strength", "相对远近电平变化强度", 0, 1, 0.05)
        number("spatial_rtf_color_strength", "相对频谱染色强度", 0, 1, 0.05)
        radio(
            "spatial_rtf_fft_size",
            "FFT 窗口",
            [("自动", 0), ("1024", 1024), ("2048", 2048), ("4096", 4096), ("8192", 8192)],
        )
        radio("spatial_rtf_hop_divisor", "FFT 步长除数", [("4", 4), ("8（默认）", 8), ("16", 16)])
        number("spatial_rtf_block_seconds", "RTF 内存分块秒数", 1, 30, 1)

    with gr.Group(visible=stored.separation_mix_mode == "replace") as keep_group:
        keep = radio(
            "separation_keep_original",
            "替换模式的原声回填",
            [
                ("不回填", "none"),
                ("自动：回填未配音句子的分离人声", "unvoiced"),
                ("手动：指定句子 ID", "manual"),
            ],
        )
        text(
            "separation_keep_ids",
            "手动保留原声的句子 ID",
            "逗号或空格分隔，如 s000006,s000010。不是回填完整背景。"
            "自动模式不保证这些句子是非语言发声。",
        )
        number("separation_keep_padding_ms", "回填前后保留毫秒", 0, 500, 10)
    gr.Markdown(
        "修改后在页面底部保存到当前项目或两者。RTF 只需重新混音；"
        "分离设置会标记 ASR 待更新，需重跑相关阶段。"
    )

    dependency = gr.Markdown()
    summary = gr.Markdown()
    sep = components["separation_enabled"]

    def refresh(enabled, selected, rtf, policy):
        conflict = selected == "replace" and not enabled
        note = (
            "**设置冲突：中文替换需要开启人声分离。请改回双语或开启分离后保存。**"
            if conflict
            else "需要先在“人声分离”页开启分离才能选择中文替换；不会自动开启或下载模型。"
            if not enabled
            else "中文替换可用；已有音频尚未分离时，混音阶段会先执行分离。"
        )
        chinese = "RTF 中文配音" if rtf else "中文配音"
        result = ("分离背景" if selected == "replace" else "完整原音") + "＋" + chinese
        if selected == "replace" and policy != "none":
            result += "＋" + ("未配音句子的原声" if policy == "unvoiced" else "手动指定原声")
        return (
            gr.update(interactive=bool(enabled or conflict)),
            gr.update(visible=selected == "replace"),
            note,
            "**本次混音组成：** " + result + ("（设置冲突，不能保存）" if conflict else ""),
        )

    options = dict(
        fn=refresh,
        inputs=[sep, mode, rtf_toggle, keep],
        outputs=[mode, keep_group, dependency, summary],
        queue=False,
        api_name=False,
    )
    for control in (sep, mode, rtf_toggle, keep):
        control.change(**options)
    return options
