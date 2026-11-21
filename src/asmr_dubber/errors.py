from __future__ import annotations


class AsmrDubberError(RuntimeError):
    """Base error shown to CLI and UI users, carrying a stable code and recovery hint."""

    default_code: str = "ASMR_ERROR"
    default_hint: str = "请查看界面日志或导出诊断包以定位原因。"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code: str = code or self.default_code
        self.hint: str = hint if hint is not None else self.default_hint


class EnvironmentError(AsmrDubberError):
    """The local operating system or model runtime is unsuitable."""

    default_code = "ENVIRONMENT_UNSUITABLE"
    default_hint = "请在“设备与模型”页面检查运行环境、显卡驱动与模型安装状态。"


class InstallPausedError(EnvironmentError):
    """A resumable dependency or model download was paused by the user."""

    default_code = "INSTALL_PAUSED"
    default_hint = "下载进度已保留，随时点击继续即可恢复。"


class OperationCancelledError(AsmrDubberError):
    """The user cancelled an in-progress project operation."""

    default_code = "OPERATION_CANCELLED"
    default_hint = "已完成的进度已自动保存，需要时可以直接继续。"


class ProjectError(AsmrDubberError):
    """A project manifest is missing or inconsistent."""

    default_code = "PROJECT_INVALID"
    default_hint = "请确认项目文件与原始音视频未被移动，或重新打开项目。"


class ProjectConflictError(ProjectError):
    """Another process saved the same project after it was loaded."""

    default_code = "PROJECT_CONFLICT"
    default_hint = "请刷新或重新加载项目后再保存修改。"


class TranslationError(AsmrDubberError):
    """The selected translation service did not return usable translations."""

    default_code = "TRANSLATION_FAILED"
    default_hint = "请检查翻译服务的 API Key、网络连接或额度，必要时降低并发重试。"


class SynthesisError(AsmrDubberError):
    """Voice cloning failed."""

    default_code = "SYNTHESIS_FAILED"
    default_hint = "请检查配音模型运行环境、显存余量，或在句子表中微调失败句的时间和文本后重试。"


def classify_error(exc: BaseException) -> tuple[str, str]:
    """Return ``(error_code, recovery_hint)`` for any pipeline or task exception."""

    text = str(exc)
    lowered = text.casefold()
    if isinstance(exc, OperationCancelledError):
        return exc.code, exc.hint
    if isinstance(exc, InstallPausedError):
        return exc.code, exc.hint
    if isinstance(exc, ProjectConflictError):
        return exc.code, exc.hint
    if "尚未保存 api key" in lowered or (
        "api key" in lowered and ("missing" in lowered or "未配置" in text)
    ):
        return "API_KEY_MISSING", "请前往“设置 → 翻译服务”填写并保存有效的 API Key。"
    if (
        "401" in lowered
        or "unauthorized" in lowered
        or "invalid api key" in lowered
        or "密钥无效" in text
    ):
        return "API_KEY_INVALID", "API Key 无效或已过期，请在“设置 → 翻译服务”中更新密钥。"
    if (
        "402" in lowered
        or "insufficient" in lowered
        or "quota" in lowered
        or "余额" in text
        or "额度" in text
    ):
        return "TRANSLATION_QUOTA", "翻译服务账户余额或配额不足，请充值或切换其他翻译服务后继续。"
    if "429" in lowered or "rate limit" in lowered or "频率" in text or "限流" in text:
        return (
            "TRANSLATION_RATE_LIMIT",
            "请求触发服务商限流；已完成批次已保存，稍等片刻后点击继续即可。",
        )
    if "out of memory" in lowered or "cuda error" in lowered or "显存不足" in text:
        return (
            "CUDA_OOM",
            "显存或虚拟内存不足：请先重启程序释放上一步识别占用的显存、"
            "关闭浏览器视频/壁纸/游戏等占显存程序、确认配音已开启 FP16；"
            "Windows 若报错仍显示有空闲显存，"
            "请检查系统虚拟内存（页面文件）是否已设为自动管理且 C 盘空间充足。",
        )
    if "找不到" in text and ("运行环境" in text or "模型" in text):
        return "RUNTIME_MISSING", "请在“设备与模型”页面安装或修复对应的本地模型运行环境。"
    if "句未取得译文：" in text:
        return "PARTIAL_TRANSLATION", "其余译文已保存；可在句子表中筛选未翻译项单独重译。"
    if "部分句子的逐句克隆失败" in text:
        return "PARTIAL_SYNTHESIS", "成功句子的配音缓存已保留；请检查报错句后单独重新生成。"
    if isinstance(exc, AsmrDubberError):
        return exc.code, exc.hint
    return "INTERNAL_ERROR", "操作意外失败；已完成部分已保存，可导出诊断包排查具体原因。"
