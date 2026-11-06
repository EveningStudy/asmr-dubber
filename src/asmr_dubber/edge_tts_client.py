"""Shared compatibility, retry and cancellation contract for Edge TTS."""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .errors import OperationCancelledError, SynthesisError
from .task_control import check_cancelled

MINIMUM_VERSION = "7.3.1"
_RETRY_DELAYS = (1.0, 2.0, 4.0)
logger = logging.getLogger(__name__)


def compatibility_problem() -> str | None:
    try:
        installed = version("edge-tts") or "元数据不完整"
    except PackageNotFoundError:
        installed = "未安装"
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?", installed)
    if match and (7, 3, 1) <= tuple(map(int, match.groups())) < (8, 0, 0):
        return None
    return (
        f"Edge TTS 客户端版本不兼容（当前 {installed}，需要 >={MINIMUM_VERSION},<8）。"
        "请关闭程序并用新版启动器重新启动，或运行 setup 更新基础依赖。"
    )


def require_compatible_client() -> None:
    if problem := compatibility_problem():
        raise SynthesisError(problem)


def _failure_message(exc: Exception) -> str:
    status = getattr(exc, "status", None)
    if status == 403:
        return (
            "Microsoft Edge TTS 拒绝鉴权（HTTP 403）。请确认已使用新版程序并更新基础依赖；"
            "若仍失败，请稍后重试或检查服务访问限制。成功缓存已保留。"
        )
    if status == 429:
        return "Microsoft Edge TTS 请求过于频繁（HTTP 429）。请降低配音并发并稍后重试。"
    if type(exc).__name__ == "NoAudioReceived":
        return (
            "Edge TTS 已连接 Microsoft 服务，但连续 4 次未收到音频。"
            "可能是服务或音色临时异常；请稍后只重试失败句，或更换音色。"
        )
    message = str(exc)
    lowered = message.casefold()
    if "getaddrinfo" in lowered or "name or service not known" in lowered:
        return (
            "无法解析 Microsoft Edge TTS 服务地址。请检查 DNS、代理或 TUN 后重试，成功缓存已保留。"
        )
    if isinstance(exc, TimeoutError) or "timed out" in lowered or "timeout" in lowered:
        return "Microsoft Edge TTS 请求超时，已停止等待。请稍后重试，成功缓存已保留。"
    return f"Edge TTS 请求失败，成功缓存已保留。请检查网络或稍后重试。原始错误：{message}"


async def _retry_delay(seconds: float) -> None:
    deadline = asyncio.get_running_loop().time() + seconds
    while (remaining := deadline - asyncio.get_running_loop().time()) > 0:
        check_cancelled()
        await asyncio.sleep(min(remaining, 0.1))
    check_cancelled()


async def _save_once(communicator, destination: Path) -> None:
    # aiohttp can override edge-tts's socket read timeout. Poll cancellation and
    # enforce our own outer deadline even when a WebSocket stays silently open.
    check_cancelled()
    request = asyncio.create_task(communicator.save(str(destination)))
    try:
        while True:
            check_cancelled()
            done, _ = await asyncio.wait({request}, timeout=0.1)
            if done:
                request.result()
                check_cancelled()
                return
    finally:
        if not request.done():
            request.cancel()
        with suppress(asyncio.CancelledError):
            await request


def save_audio(
    text: str,
    voice: str,
    destination: Path,
    *,
    rate: str | None = None,
    timeout_seconds: float = 30.0,
) -> None:
    """Save to a caller-owned temporary path; never overwrite a cached success."""
    require_compatible_client()
    try:
        import aiohttp
        import edge_tts
    except ImportError as exc:
        raise SynthesisError("Edge TTS 运行依赖缺失；请重新运行 setup 修复基础依赖。") from exc

    edge_errors = getattr(edge_tts, "exceptions", None)
    retryable: tuple[type[Exception], ...] = (
        OSError,
        aiohttp.ClientError,
        *(
            error
            for name in (
                "NoAudioReceived",
                "UnknownResponse",
                "WebSocketError",
                "SkewAdjustmentError",
            )
            if isinstance((error := getattr(edge_errors, name, None)), type)
            and issubclass(error, Exception)
        ),
    )

    async def attempts() -> None:
        for attempt in range(4):
            check_cancelled()
            destination.unlink(missing_ok=True)
            try:
                communicator = (
                    edge_tts.Communicate(text, voice=voice, rate=rate)
                    if rate is not None
                    else edge_tts.Communicate(text, voice=voice)
                )
                # Bound each attempt as well as the entire sentence, including
                # all retries, with the configured single-sentence timeout.
                await asyncio.wait_for(
                    _save_once(communicator, destination), min(60, timeout_seconds)
                )
                if not destination.is_file() or destination.stat().st_size == 0:
                    raise SynthesisError("Edge TTS 没有生成音频。")
                return
            except OperationCancelledError:
                raise
            except retryable as exc:
                if getattr(exc, "status", None) == 403 or attempt == 3:
                    raise
                logger.warning("Edge TTS 请求失败，准备重试：尝试=%d/4 错误=%s", attempt + 1, exc)
                await _retry_delay(_RETRY_DELAYS[attempt])

    async def bounded() -> None:
        await asyncio.wait_for(attempts(), timeout_seconds)

    try:
        asyncio.run(bounded())
    except (OperationCancelledError, SynthesisError):
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise SynthesisError(_failure_message(exc)) from exc
