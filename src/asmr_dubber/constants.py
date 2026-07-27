from pathlib import Path

from .platforms import user_data_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_SCHEMA_VERSION = 1
DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
RECOMMENDED_ASR_BACKEND = "parakeet_nemo"
RECOMMENDED_ASR_MODEL = "grider-transwithai/parakeet-ctc-1.1b-ja::parakeet-ja-gal.nemo"
DEFAULT_TRANSLATION_MODEL = "deepseek-v4-pro"
# Tencent Hunyuan Hy-MT2 is a built-in local translation model. It loads via
# Transformers like the bundled ASR/TTS models and does not need an HTTP API.
DEFAULT_HUNYUAN_MT_MODEL = "tencent/Hy-MT2-1.8B"
# VoxCPM2 remains a verified compatibility backend and part of the bundled
# Qwen runtime. New projects default to the separately installed IndexTTS2
# backend because it is the recommended voice-cloning path.
DEFAULT_TTS_MODEL = "openbmb/VoxCPM2"
RECOMMENDED_TTS_BACKEND = "indextts2"
RECOMMENDED_TTS_MODEL = "IndexTTS2"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_ASR_REVIEW_PROMPT = """你是日语语音识别校对专家。你会收到按时间窗口组织的多个 ASR 候选。

任务：
1. 结合相邻窗口、用户提供的作品/人物/场景背景，恢复每个窗口最可信的日文原话。
2. 特别检查日语 ASR 常见问题：同音字、助词、专名、口语省略、耳语漏字、错误断句、
   Whisper/语言模型重复循环、静音处凭空生成内容，以及分段时间戳整体漂移。
3. 只能依据候选证据纠错，不得补写所有候选都没有支持的剧情或台词。
4. 笑声、喘息、呻吟、亲吻声、拉长音等纯非语言内容输出空 ja；混有实义台词时保留实义部分。
5. 每个输入 window_id 恰好输出一项并保持顺序。evidence_ids 只能引用该窗口给出的候选 id；
   它们用于程序从真实 ASR 时间戳计算边界，不要自行输出或猜测时间。
6. 若证据互相冲突且无法可靠判断，优先保留日语专用模型之间一致的部分，并降低 confidence；
   若基本确定是幻觉，ja 置空。
7. 只输出严格 JSON：
{"results":[{"window_id":"w000001","ja":"校对后的日文","evidence_ids":["w000001-c01"],"confidence":0.95}]}
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
MODEL_REVISIONS = {
    DEFAULT_ASR_MODEL: "7278e1e70fe206f11671096ffdd38061171dd6e5",
    DEFAULT_ALIGNER_MODEL: "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
    DEFAULT_TTS_MODEL: "bffb3df5a29440629464e5e839f4d214c8714c3d",
}

# Recommended optional ASR snapshots are also pinned so an update upstream
# cannot silently change transcription behaviour between runs.  Their Hub
# manifests provide per-file integrity metadata; unlike the bundled defaults,
# these snapshots are allowed to evolve without duplicating every file size
# in this repository.
OPTIONAL_ASR_MODEL_REVISIONS = {
    "kotoba-tech/kotoba-whisper-v2.2": "9d33482a0eb9b57f1ad80708e8ac5538246d8355",
    "kotoba-tech/kotoba-whisper-v2.1": "57a9d8ab771a0124706b67d22509bedd07c36187",
    "kotoba-tech/kotoba-whisper-v2.0": "7eb575277d18909a4af8a24e3ae8cce2e99794ae",
    "Qwen/Qwen3-ASR-0.6B": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    "Systran/faster-whisper-large-v2": "f0fe81560cb8b68660e564f55dd99207059c092e",
    "kotoba-tech/kotoba-whisper-v2.0-faster": ("f44edd35eaeb2274e85ac7b31fb2c6f59ff1c4bc"),
}

# Optional translation model snapshots.  Hunyuan Hy-MT2 is pinned to a reviewed
# revision so the local Transformers loader cannot silently pick up an upstream
# change.  The Hub manifest provides per-file integrity metadata; we only need
# the revision here so resolve_transformers_model_source accepts the model id.
OPTIONAL_TRANSLATION_MODEL_REVISIONS = {
    DEFAULT_HUNYUAN_MT_MODEL: "main",
}

# Exact runtime files and byte sizes from those three repository revisions.
# This prevents a partially downloaded snapshot directory from being mistaken
# for a usable offline model.
MODEL_REQUIRED_FILES = {
    DEFAULT_ASR_MODEL: {
        "chat_template.json": 1161,
        "config.json": 6194,
        "generation_config.json": 142,
        "merges.txt": 1671853,
        "model-00001-of-00002.safetensors": 4220320824,
        "model-00002-of-00002.safetensors": 478200688,
        "model.safetensors.index.json": 64821,
        "preprocessor_config.json": 330,
        "tokenizer_config.json": 12487,
        "vocab.json": 2776833,
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
    DEFAULT_TTS_MODEL: {
        "audiovae.pth": 376951122,
        "config.json": 4336,
        "model.safetensors": 4580080592,
        "special_tokens_map.json": 1632,
        "tokenization_voxcpm2.py": 2895,
        "tokenizer.json": 3676772,
        "tokenizer_config.json": 5059,
    },
}

MODEL_LFS_SHA256 = {
    DEFAULT_ASR_MODEL: {
        "model-00001-of-00002.safetensors": (
            "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6"
        ),
        "model-00002-of-00002.safetensors": (
            "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc"
        ),
    },
    DEFAULT_ALIGNER_MODEL: {
        "model.safetensors": ("47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7"),
    },
    DEFAULT_TTS_MODEL: {
        "audiovae.pth": "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1",
        "model.safetensors": ("f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d"),
    },
}
