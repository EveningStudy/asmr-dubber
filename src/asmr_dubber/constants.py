from pathlib import Path

from .platforms import user_data_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_SCHEMA_VERSION = 3
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
ASMR_VAD_MODEL = "TransWithAI/Whisper-Vad-EncDec-ASMR-onnx"
RECOMMENDED_ASR_BACKEND = "parakeet_nemo"
RECOMMENDED_ASR_MODEL = "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo"
DEFAULT_ASR_REVIEW_TEXT_PRIORITY = (
    "parakeet_nemo|grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo"
)
DEFAULT_ASR_REVIEW_TIMESTAMP_PRIORITY = f"qwen_forced_aligner|{DEFAULT_ALIGNER_MODEL}"
DEFAULT_ASR_REVIEW_MODELS = (
    DEFAULT_ASR_REVIEW_TEXT_PRIORITY,
    "kotoba_whisper|kotoba-tech/kotoba-whisper-v2.2",
)
DEFAULT_TRANSLATION_MODEL = "deepseek-v4-pro"
RECOMMENDED_TTS_BACKEND = "indextts2"
RECOMMENDED_TTS_MODEL = "IndexTTS2"
DEFAULT_INDEXTTS_EMOTION_WEIGHT = 0.5
DEFAULT_CHINESE_DUBBING_OFFSET_MS = 500
DEFAULT_CHINESE_MAX_AUTO_SPEED = 1.8
DEFAULT_CHINESE_RELATIVE_LOUDNESS_DB = -8.0
MAX_CHINESE_AUTO_SPEED = 4.0
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_ASR_REVIEW_PROMPT = """你是语音识别校对专家。你会收到按时间窗口组织的多个 ASR 候选。

任务：
1. 结合相邻窗口、用户提供的作品/人物/场景背景，恢复每个窗口最可信的源语言原话。
2. 检查 ASR 常见问题：同音词、专名、口语省略、耳语漏字、错误断句、
   Whisper/语言模型重复循环、静音处凭空生成内容，以及分段时间戳整体漂移。
3. 只能依据候选证据纠错，不得补写所有候选都没有支持的剧情或台词。
4. 笑声、喘息、呻吟、亲吻声、拉长音等纯非语言内容输出空 source；混有实义台词时保留实义部分。
5. 每个输入 window_id 恰好输出一项并保持顺序。evidence_ids 只能引用该窗口给出的候选 id；
   它们用于程序从真实 ASR 时间戳计算边界，不要自行输出或猜测时间。
6. 若证据互相冲突且无法可靠判断，优先保留多个模型一致且符合上下文的部分，并降低 confidence；
   若基本确定是幻觉，source 置空。
7. 输入会标记 text_priority。它是文字判断的优先来源，但不得压过多个模型一致的反证或明显幻觉。
   timestamp_priority 只由程序计算时间边界，不要为了迁就它而改写 source。
8. 只输出严格 JSON：
{"results":[{"window_id":"w000001","source":"校对后的源文","evidence_ids":["w000001-c01"],"confidence":0.95}]}
"""
DEFAULT_PROJECTS_DIR = user_data_dir() / "projects"
DEFAULT_RUNTIMES_DIR = user_data_dir() / "runtimes"
DEFAULT_MODELS_DIR = user_data_dir() / "models"
DEFAULT_INDEXTTS_ROOT = DEFAULT_RUNTIMES_DIR / "index-tts"
DEFAULT_INDEXTTS_MODEL_DIR = DEFAULT_INDEXTTS_ROOT / "checkpoints"
DEFAULT_INDEXTTS_CONFIG = DEFAULT_INDEXTTS_MODEL_DIR / "config.yaml"
INDEXTTS_REQUIRED_FILES = frozenset(
    {
        "config.yaml",
        "bpe.model",
        "gpt.pth",
        "s2mel.pth",
        "wav2vec2bert_stats.pt",
        "feat1.pt",
        "feat2.pt",
        "hf_cache/semantic_codec_model.safetensors",
        "hf_cache/campplus_cn_common.bin",
        "hf_cache/bigvgan/config.json",
        "hf_cache/bigvgan/bigvgan_generator.pt",
    }
)
INDEXTTS_REQUIRED_DIRS = frozenset({"qwen0.6bemo4-merge", "hf_cache/w2v-bert-2.0"})

# Pin the exact model snapshots validated for this release. Project manifests keep the
# human-readable repository ids, while loaders resolve these revisions from the
# local Hugging Face cache whenever they are available.
MODEL_REVISIONS: dict[str, str] = {}

# Optional ASR snapshots in the recommended profile are also pinned so an update upstream
# cannot silently change transcription behaviour between runs.  Their Hub
# manifests provide per-file integrity metadata; unlike the bundled defaults,
# these snapshots are allowed to evolve without duplicating every file size
# in this repository.
OPTIONAL_ASR_MODEL_REVISIONS = {
    ASMR_VAD_MODEL: "6ac29e2cbf2f4f8e9b639861766a8639dd666e9c",
    DEFAULT_ALIGNER_MODEL: "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
    "kotoba-tech/kotoba-whisper-v2.2": "9d33482a0eb9b57f1ad80708e8ac5538246d8355",
    "kotoba-tech/kotoba-whisper-v2.1": "57a9d8ab771a0124706b67d22509bedd07c36187",
    "kotoba-tech/kotoba-whisper-v2.0": "7eb575277d18909a4af8a24e3ae8cce2e99794ae",
    "Systran/faster-whisper-large-v2": "f0fe81560cb8b68660e564f55dd99207059c092e",
    "kotoba-tech/kotoba-whisper-v2.0-faster": ("f44edd35eaeb2274e85ac7b31fb2c6f59ff1c4bc"),
}

# Exact runtime files and byte sizes from those three repository revisions.
# This prevents a partially downloaded snapshot directory from being mistaken
# for a usable offline model.
MODEL_REQUIRED_FILES: dict[str, dict[str, int]] = {
    ASMR_VAD_MODEL: {
        "inference.py": 25880,
        "model.onnx": 119137398,
        "model_metadata.json": 370,
        "requirements.txt": 233,
    },
    DEFAULT_ALIGNER_MODEL: {
        "chat_template.json": 1161,
        "config.json": 5982,
        "generation_config.json": 115,
        "merges.txt": 1671853,
        "model.safetensors": 1835544544,
        "preprocessor_config.json": 330,
        "tokenizer_config.json": 12666,
        "vocab.json": 2776833,
    },
}
MODEL_LFS_SHA256: dict[str, dict[str, str]] = {
    ASMR_VAD_MODEL: {
        "model.onnx": "cd47513515766d57f740e3094440dbbca9ab87e026b9cf21540d7ad588c0e047",
    },
    DEFAULT_ALIGNER_MODEL: {
        "model.safetensors": "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7",
    },
}
