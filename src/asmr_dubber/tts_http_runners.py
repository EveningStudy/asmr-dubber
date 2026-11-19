from __future__ import annotations

import base64
import json
import mimetypes
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .api_contracts import (
    APIContractError,
    bearer_headers,
    endpoint_url,
    merge_extra_body,
    write_audio_response,
)
from .errors import OperationCancelledError, SynthesisError
from .model_registry import TTS_BACKENDS
from .models import DubProject, Sentence
from .task_control import check_cancelled
from .tts import attempt_seed, spoken_text
from .voice_reference import VoiceReference

_RETRYABLE_HTTP_STATUS = frozenset({429, 502, 503, 504})


def _post_with_retry(
    request_fn: Callable[[], httpx.Response],
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> httpx.Response:
    """Retry transient HTTP 429/502/503/504 or transport errors with bounded backoff."""
    for attempt in range(max_retries + 1):
        check_cancelled()
        try:
            response = request_fn()
        except (httpx.RequestError, OSError, TimeoutError):
            if attempt >= max_retries:
                raise
            delay = min(8.0, base_delay * (2**attempt))
            check_cancelled()
            time.sleep(delay)
            continue
        if response.status_code not in _RETRYABLE_HTTP_STATUS or attempt >= max_retries:
            return response
        delay = min(8.0, base_delay * (2**attempt))
        check_cancelled()
        time.sleep(delay)
    raise AssertionError("unreachable")


def _gpt_sovits_runner(
    project: DubProject,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    from . import tts_backends

    url = f"{project.settings.tts_api_base_url.rstrip('/')}/tts"
    client = tts_backends.logged_http_client(
        "GPT-SoVITS API",
        timeout=project.settings.tts_timeout_seconds,
    )

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        payload = {
            "text": spoken_text(project.settings, sentence.zh_text),
            "text_lang": project.settings.tts_target_language,
            "ref_audio_path": str(reference.path),
            "prompt_lang": reference.language,
            "prompt_text": reference.text,
            "top_k": project.settings.tts_gpt_top_k,
            "top_p": project.settings.tts_top_p,
            "temperature": project.settings.tts_temperature,
            "text_split_method": project.settings.tts_gpt_text_split_method,
            "speed_factor": project.settings.tts_speed,
            "seed": attempt_seed(project.settings.random_seed, sentence),
            "media_type": "wav",
            "streaming_mode": False,
            "sample_steps": project.settings.tts_gpt_sample_steps,
        }
        response = _post_with_retry(lambda: client.post(url, json=payload))
        if response.is_error:
            raise SynthesisError(f"GPT-SoVITS API {response.status_code}: {response.text[:500]}")
        output.write_bytes(response.content)

    return run, client.close


def _cosyvoice_runner(
    project: DubProject,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    from . import tts_backends

    mode = project.settings.tts_cosyvoice_mode
    endpoint = "inference_zero_shot" if mode == "zero_shot" else "inference_cross_lingual"
    url = f"{project.settings.tts_api_base_url.rstrip('/')}/{endpoint}"
    client = tts_backends.logged_http_client(
        "CosyVoice API",
        timeout=project.settings.tts_timeout_seconds,
    )

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        data = {"tts_text": spoken_text(project.settings, sentence.zh_text)}
        if mode == "zero_shot":
            data["prompt_text"] = reference.text

        def do_post() -> httpx.Response:
            with reference.path.open("rb") as handle:
                return client.post(
                    url,
                    data=data,
                    files={"prompt_wav": (reference.path.name, handle, "audio/wav")},
                )

        response = _post_with_retry(do_post)
        if response.is_error:
            raise SynthesisError(f"CosyVoice API {response.status_code}: {response.text[:500]}")
        output.write_bytes(response.content)

    return run, client.close


def _fish_runner(
    project: DubProject,
    api_key: str | None = None,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    from . import tts_backends

    base = project.settings.tts_api_base_url.rstrip("/")
    url = base if base.endswith("/v1/tts") else f"{base}/v1/tts"
    key = (api_key or "").strip() or tts_backends.saved_service_key(
        f"tts:{project.settings.tts_backend}"
    )
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    client = tts_backends.logged_http_client(
        "Fish Speech / Fish Audio API",
        headers=headers,
        timeout=project.settings.tts_timeout_seconds,
    )
    reference_payloads: dict[tuple[str, str], str] = {}
    payload_lock = threading.Lock()

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        reference_key = (str(reference.path.resolve()), reference.text)
        with payload_lock:
            encoded_reference = reference_payloads.get(reference_key)
            if encoded_reference is None:
                encoded_reference = base64.b64encode(reference.path.read_bytes()).decode("ascii")
                if len(reference_payloads) >= 8:
                    reference_payloads.pop(next(iter(reference_payloads)))
                reference_payloads[reference_key] = encoded_reference
        payload = {
            "text": spoken_text(project.settings, sentence.zh_text),
            "format": "wav",
            "normalize": True,
            "references": [
                {
                    "audio": encoded_reference,
                    "text": reference.text,
                }
            ],
        }
        response = _post_with_retry(lambda: client.post(url, json=payload))
        if response.is_error:
            raise SynthesisError(f"Fish Speech API {response.status_code}: {response.text[:500]}")
        output.write_bytes(response.content)

    def cleanup() -> None:
        reference_payloads.clear()
        client.close()

    return run, cleanup


def _indextts_api_runner(
    project: DubProject,
    api_key: str | None = None,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    from . import tts_backends

    base = project.settings.tts_api_base_url
    key = (api_key or "").strip() or tts_backends.saved_service_key("tts:indextts2_api")
    client = tts_backends.logged_http_client(
        "IndexTTS2 API",
        headers=bearer_headers(key),
        timeout=project.settings.tts_timeout_seconds,
    )
    try:
        url = endpoint_url(base, "/v1/tts")
    except APIContractError as exc:
        client.close()
        raise SynthesisError(str(exc)) from exc

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        try:
            payload: dict[str, object] = {
                "model": project.settings.tts_model or "IndexTTS2",
                "text": spoken_text(project.settings, sentence.zh_text),
                "emotion_alpha": project.settings.tts_index_emo_alpha,
                "temperature": project.settings.tts_temperature,
                "top_p": project.settings.tts_top_p,
                "speed": project.settings.tts_speed,
            }
            payload = merge_extra_body(
                payload,
                project.settings.tts_api_extra_body,
                label="IndexTTS2 API 附加请求参数",
                reserved={"text"},
            )
        except (OSError, APIContractError) as exc:
            raise SynthesisError(str(exc)) from exc
        data = {
            key: json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else str(value)
            for key, value in payload.items()
        }

        def do_post() -> httpx.Response:
            files: dict[str, tuple[str, Any, str]] = {
                "voice": (reference.path.name, reference.path.open("rb"), "audio/wav")
            }
            if reference.emotion_path and reference.emotion_path.is_file():
                files["emotion_audio"] = (
                    reference.emotion_path.name,
                    reference.emotion_path.open("rb"),
                    "audio/wav",
                )
            try:
                return client.post(url, data=data, files=files)
            finally:
                for item in files.values():
                    item[1].close()

        response = _post_with_retry(do_post)
        try:
            write_audio_response(response, output, client=client)
        except APIContractError as exc:
            raise SynthesisError(f"IndexTTS2 API 返回无效结果：{exc}") from exc

    return run, client.close


def _generic_tts_api_runner(
    project: DubProject,
    api_key: str | None = None,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    from . import tts_backends

    key = (api_key or "").strip() or tts_backends.saved_service_key("tts:generic_tts_api")
    client = tts_backends.logged_http_client(
        "通用 TTS API",
        headers=bearer_headers(key),
        timeout=project.settings.tts_timeout_seconds,
    )
    try:
        url = endpoint_url(project.settings.tts_api_base_url, "/audio/speech")
    except APIContractError as exc:
        client.close()
        raise SynthesisError(str(exc)) from exc

    def run(sentence: Sentence, _reference: VoiceReference, output: Path) -> None:
        try:
            payload = merge_extra_body(
                {
                    "model": project.settings.tts_model or "tts-1",
                    "input": spoken_text(project.settings, sentence.zh_text),
                    "voice": project.settings.tts_voice or "alloy",
                    "response_format": "wav",
                    "speed": project.settings.tts_speed,
                },
                project.settings.tts_api_extra_body,
                label="TTS API 附加请求参数",
                reserved={"model", "input"},
            )
        except APIContractError as exc:
            raise SynthesisError(str(exc)) from exc
        response = _post_with_retry(lambda: client.post(url, json=payload))
        try:
            write_audio_response(response, output, client=client)
        except APIContractError as exc:
            raise SynthesisError(f"通用 TTS API 返回无效结果：{exc}") from exc

    return run, client.close


def _edge_tts_runner(
    project: DubProject,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    from . import tts_backends
    from .audio import _run_ffmpeg

    voice = project.settings.tts_voice.strip() or TTS_BACKENDS["edge_tts"].default_voice
    if project.settings.tts_target_language == "en" and not voice.startswith("en-"):
        voice = "en-US-AriaNeural"
    elif project.settings.tts_target_language == "zh" and voice.startswith("en-"):
        voice = TTS_BACKENDS["edge_tts"].default_voice
    speed_percent = round((project.settings.tts_speed - 1.0) * 100)
    rate = f"{speed_percent:+d}%"

    def run(sentence: Sentence, _reference: VoiceReference, output: Path) -> None:
        encoded = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.edge.mp3")

        started = time.perf_counter()
        tts_backends.logger.info("API 请求开始：服务=Edge TTS 地址=Microsoft 在线语音服务")
        try:
            from .edge_tts_client import save_audio

            save_audio(
                spoken_text(project.settings, sentence.zh_text),
                voice,
                encoded,
                rate=rate,
                timeout_seconds=project.settings.tts_timeout_seconds,
            )
            _run_ffmpeg(
                [
                    "-y",
                    "-i",
                    str(encoded),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_f32le",
                    str(output),
                ]
            )
        except (SynthesisError, OperationCancelledError):
            raise
        except Exception as exc:
            raise SynthesisError(f"Edge TTS 生成失败：{exc}") from exc
        finally:
            encoded.unlink(missing_ok=True)
            tts_backends.logger.info(
                "API 请求结束：服务=Edge TTS 耗时=%d ms",
                round((time.perf_counter() - started) * 1000),
            )

    return run, lambda: None


def _mimo_runner(
    project: DubProject,
    api_key: str | None = None,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    from . import tts_backends

    base = project.settings.tts_api_base_url.rstrip("/")
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    key = (api_key or "").strip() or tts_backends.saved_service_key("tts:mimo_tts")
    if not key:
        raise SynthesisError("小米 MiMo TTS 尚未保存 API Key。")
    client = tts_backends.logged_http_client(
        "小米 MiMo TTS API",
        headers={"api-key": key, "Content-Type": "application/json"},
        timeout=project.settings.tts_timeout_seconds,
    )
    encoded_references: dict[str, str] = {}
    payload_lock = threading.Lock()

    def reference_data_uri(reference: VoiceReference) -> str:
        resolved = str(reference.path.resolve())
        with payload_lock:
            cached = encoded_references.get(resolved)
            if cached is not None:
                return cached
            mime = mimetypes.guess_type(reference.path.name)[0] or "audio/wav"
            if mime in {"audio/x-wav", "audio/vnd.wave"}:
                mime = "audio/wav"
            encoded = base64.b64encode(reference.path.read_bytes()).decode("ascii")
            value = f"data:{mime};base64,{encoded}"
            if len(encoded_references) >= 8:
                encoded_references.pop(next(iter(encoded_references)))
            encoded_references[resolved] = value
            return value

    def run(sentence: Sentence, reference: VoiceReference, output: Path) -> None:
        model = project.settings.tts_model.strip() or TTS_BACKENDS["mimo_tts"].default_model
        style = project.settings.tts_style_prompt.strip()
        messages = [
            {"role": "user", "content": style},
            {"role": "assistant", "content": spoken_text(project.settings, sentence.zh_text)},
        ]
        audio: dict[str, object] = {"format": "wav"}
        if model == "mimo-v2.5-tts-voiceclone":
            audio["voice"] = reference_data_uri(reference)
        elif model == "mimo-v2.5-tts-voicedesign":
            if not style:
                messages[0]["content"] = (
                    "A gentle, natural English female voice at a steady pace."
                    if project.settings.tts_target_language == "en"
                    else "温柔自然的中文女声，语速平稳。"
                )
            audio["optimize_text_preview"] = True
        else:
            audio["voice"] = (
                project.settings.tts_voice.strip() or TTS_BACKENDS["mimo_tts"].default_voice
            )
        response = _post_with_retry(
            lambda: client.post(
                url,
                json={"model": model, "messages": messages, "audio": audio, "stream": False},
            )
        )
        if response.is_error:
            raise SynthesisError(f"小米 MiMo TTS API {response.status_code}: {response.text[:500]}")
        try:
            encoded_audio = response.json()["choices"][0]["message"]["audio"]["data"]
            audio_bytes = base64.b64decode(encoded_audio, validate=True)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SynthesisError("小米 MiMo TTS 返回格式无效，未找到可解码的音频。") from exc
        if not audio_bytes:
            raise SynthesisError("小米 MiMo TTS 返回了空音频。")
        output.write_bytes(audio_bytes)

    def cleanup() -> None:
        encoded_references.clear()
        client.close()

    return run, cleanup


def _minimax_runner(
    project: DubProject,
    api_key: str | None = None,
) -> tuple[Callable[[Sentence, VoiceReference, Path], None], Callable[[], None]]:
    from . import tts_backends

    base = project.settings.tts_api_base_url.rstrip("/")
    url = base if base.endswith("/v1/t2a_v2") else f"{base}/v1/t2a_v2"
    key = (api_key or "").strip() or tts_backends.saved_service_key("tts:minimax")
    if not key:
        raise SynthesisError("MiniMax TTS 尚未保存 API Key。")
    client = tts_backends.logged_http_client(
        "MiniMax TTS API",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=project.settings.tts_timeout_seconds,
    )

    def run(sentence: Sentence, _reference: VoiceReference, output: Path) -> None:
        voice_setting: dict[str, object] = {
            "voice_id": project.settings.tts_voice.strip() or TTS_BACKENDS["minimax"].default_voice,
            "speed": project.settings.tts_speed,
            "vol": project.settings.tts_volume,
            "pitch": project.settings.tts_pitch,
        }
        emotion = project.settings.tts_emotion.strip()
        if emotion and emotion != "auto":
            voice_setting["emotion"] = emotion
        payload = {
            "model": project.settings.tts_model or TTS_BACKENDS["minimax"].default_model,
            "text": spoken_text(project.settings, sentence.zh_text),
            "stream": False,
            "voice_setting": voice_setting,
            "audio_setting": {
                "sample_rate": 32_000,
                "bitrate": 128_000,
                "format": "wav",
                "channel": 1,
            },
            "language_boost": "English"
            if project.settings.tts_target_language == "en"
            else "Chinese",
            "output_format": "hex",
            "subtitle_enable": False,
        }
        response = _post_with_retry(lambda: client.post(url, json=payload))
        if response.is_error:
            raise SynthesisError(f"MiniMax TTS API {response.status_code}: {response.text[:500]}")
        try:
            data = response.json()
            base_response = data.get("base_resp") or {}
            if base_response.get("status_code", 0) != 0:
                raise SynthesisError(
                    "MiniMax TTS 拒绝请求：" + str(base_response.get("status_msg", "未知错误"))
                )
            audio_bytes = bytes.fromhex(data["data"]["audio"])
        except SynthesisError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SynthesisError("MiniMax TTS 返回格式无效，未找到可解码的音频。") from exc
        if not audio_bytes:
            raise SynthesisError("MiniMax TTS 返回了空音频。")
        output.write_bytes(audio_bytes)

    return run, client.close
