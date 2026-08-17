from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".mka",
    ".m4b",
    ".ape",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TIMED_TRANSCRIPT_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa", ".lrc"}
TRANSCRIPT_EXTENSIONS = TIMED_TRANSCRIPT_EXTENSIONS

_LOSSLESS = {".wav", ".flac", ".ape"}
_FORMAT_SCORE = {
    ".wav": 60,
    ".flac": 58,
    ".ape": 54,
    ".m4a": 43,
    ".mka": 42,
    ".mp3": 40,
    ".opus": 39,
    ".ogg": 38,
    ".aac": 37,
    ".m4b": 36,
    ".wma": 30,
}

_SAMPLE_RE = re.compile(r"(?:サンプル|sample|试听|試聴|demo|\bcm\b)", re.IGNORECASE)
_ALARM_RE = re.compile(r"(?:アラーム|alarm|着信|通知音)", re.IGNORECASE)
_FREETALK_RE = re.compile(r"(?:フリー[ _-]*トーク|free[ _-]*talk|after[ _-]*talk)", re.IGNORECASE)
_BONUS_RE = re.compile(
    r"(?:特典|おまけ|オマケ|bonus|extra|extrack|赠品|贈品|早期特典)",
    re.IGNORECASE,
)
_NO_SE_RE = re.compile(
    r"(?:se|効果音|环境音|環境音)[ _\-]*(?:なし|無し|无|無|off)|(?:音声|声|voice)[ _\-]*のみ",
    re.IGNORECASE,
)
_WITH_SE_RE = re.compile(r"(?:se|効果音)[ _\-]*(?:あり|有|on)", re.IGNORECASE)
_VOICE_ONLY_RE = re.compile(r"(?:音声|声|voice)[ _\-]*のみ", re.IGNORECASE)
_REVERSED_RE = re.compile(r"(?:左右反転|反転版|左右逆|reverse|reversed)", re.IGNORECASE)
_ZH_RE = re.compile(
    r"(?:简体|簡體|繁体|繁體|中文|中国語|zh[-_ ]?(?:cn|hans|hant)?|chinese)", re.IGNORECASE
)
_EN_RE = re.compile(r"(?:英語|英文|english)", re.IGNORECASE)
_JA_RE = re.compile(r"(?:日文|日本語|japanese)", re.IGNORECASE)
_GENERIC_DIRECTORY_RE = re.compile(
    r"^(?:\d+[. _-]*)?(?:本編|本篇|main|audio|音声|音聲|wav|flac|mp3|m4a|aac|ogg|opus)$",
    re.IGNORECASE,
)

_ORDER_PATTERNS = (
    re.compile(
        r"^\s*(?P<prefix>ex|extra|bonus)?\s*[#№]?\s*(?P<number>\d+)(?P<suffix>[a-z])?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:track|トラック|章节|章節|(?:mp3|wav|flac|audio)?tr)"
        r"\s*[#№]?\s*(?P<number>\d+)(?P<suffix>[a-z])?",
        re.IGNORECASE,
    ),
    re.compile(r"[#№]\s*(?P<number>\d+)(?P<suffix>[a-z])?", re.IGNORECASE),
)


def normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def natural_key(value: str) -> tuple[tuple[int, object], ...]:
    """返回可比较的自然排序键，同时兼容全角数字。"""

    text = normalized_text(value)
    parts = re.split(r"(\d+)", text)
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts if part)


def track_order(value: str) -> tuple[int, int, str, tuple[tuple[int, object], ...]]:
    text = normalized_text(value)
    for pattern in _ORDER_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        prefix = str(match.groupdict().get("prefix") or "")
        number = int(match.group("number"))
        suffix = str(match.groupdict().get("suffix") or "")
        section = 1 if prefix else 0
        return section, number, suffix, natural_key(text)
    if re.search(r"(?:おまけ|特典|bonus|extra|ex)", text, re.IGNORECASE):
        return 1, 10**9, "", natural_key(text)
    return 2, 10**9, "", natural_key(text)


def clean_track_title(stem: str) -> str:
    text = unicodedata.normalize("NFKC", stem).strip()
    patterns = (
        r"^\s*(?:track|トラック|章节|章節)\s*[#№]?\s*\d+[a-z]?\s*[._\-、：: ]*",
        r"^\s*(?:(?:mp3|wav|flac|audio)?tr)\s*[#№]?\s*\d+[a-z]?\s*[._\-、：: ]*",
        r"^\s*(?:ex|extra|bonus)?\s*[#№]?\s*\d+[a-z]?\s*[._\-、：: ]*",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
        if cleaned and cleaned != text:
            return cleaned
    return text


def _language(path_text: str) -> str:
    if _ZH_RE.search(path_text):
        return "zh"
    if _EN_RE.search(path_text):
        return "en"
    # Do not classify arbitrary words such as ``Ci-en`` as English.  A bare
    # ``en`` is accepted only as a complete directory/path segment.
    if re.search(r"(?:^|/)en(?:[-_]?(?:us|gb))?(?:/|$)", normalized_text(path_text)):
        return "en"
    return "ja"


def _category(path_text: str, stem_text: str) -> str:
    combined = f"{path_text} / {stem_text}"
    if _SAMPLE_RE.search(combined):
        return "sample"
    if _ALARM_RE.search(combined):
        return "alarm"
    if _FREETALK_RE.search(combined):
        return "freetalk"
    if _BONUS_RE.search(combined) or re.match(
        r"^\s*(?:ex|extra|bonus)\s*\d+", stem_text, re.IGNORECASE
    ):
        return "bonus"
    return "main"


def _mix_variant(path_text: str) -> str:
    if _VOICE_ONLY_RE.search(path_text):
        return "voice_only"
    if _NO_SE_RE.search(path_text):
        return "no_se"
    if _WITH_SE_RE.search(path_text):
        return "with_se"
    return "standard"


def _orientation(path_text: str) -> str:
    return "reversed" if _REVERSED_RE.search(path_text) else "normal"


def _subtitle_sample(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        payload = path.read_bytes()[:256_000]
    except OSError:
        return ""
    for encoding in ("utf-8-sig", "utf-16", "cp932", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeError:
            continue
    return payload.decode("utf-8", errors="ignore")


def _subtitle_content_language(path: Path | None) -> str | None:
    text = _subtitle_sample(path)
    if not text:
        return None
    kana = len(re.findall(r"[\u3040-\u30ff]", text))
    han = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if kana:
        return "ja"
    if han >= 2:
        return "zh"
    if latin >= 8:
        return "en"
    return None


def _transcript_language(path: Path, *, content_path: Path | None = None) -> str:
    path_text = path.as_posix()
    if _ZH_RE.search(path_text):
        return "zh"
    if _EN_RE.search(path_text) or re.search(
        r"(?:^|/)en(?:[-_]?(?:us|gb))?(?:/|$)",
        normalized_text(path_text),
    ):
        return "en"
    if _JA_RE.search(path_text) or re.search(
        r"(?:^|/)ja(?:[-_]?jp)?(?:/|$)",
        normalized_text(path_text),
    ):
        return "ja"
    content_language = _subtitle_content_language(content_path)
    if content_language is not None:
        return content_language
    # DLsite 同捆的 LRC 多数是中文翻译字幕；内容也无法判断时按中文。
    if path.suffix.casefold() == ".lrc":
        return "zh"
    return "ja"


def _transcript_stem(path: Path) -> str:
    stem = path.stem
    # 兼容 01.mp3.vtt 这种双扩展名字幕。
    if Path(stem).suffix.casefold() in AUDIO_EXTENSIONS:
        stem = Path(stem).stem
    return normalized_text(stem)


@dataclass(frozen=True)
class TranscriptCandidate:
    path: Path
    relative_path: str
    language: str
    timed: bool


@dataclass(frozen=True)
class TrackCandidate:
    path: Path
    relative_path: str
    title: str
    extension: str
    category: str
    language: str
    mix_variant: str
    orientation: str
    order_key: tuple[int, int, str, tuple[tuple[int, object], ...]]
    transcript: TranscriptCandidate | None = None

    @property
    def is_optional(self) -> bool:
        return self.category != "main"


@dataclass(frozen=True)
class Edition:
    id: str
    label: str
    directory: str
    extension: str
    language: str
    mix_variant: str
    orientation: str
    tracks: tuple[TrackCandidate, ...]
    optional_tracks: tuple[TrackCandidate, ...]
    score: int
    legacy_compatible: bool = False

    @property
    def all_tracks(self) -> tuple[TrackCandidate, ...]:
        return (*self.tracks, *self.optional_tracks)


@dataclass(frozen=True)
class ScanResult:
    root: Path
    editions: tuple[Edition, ...]
    images: tuple[Path, ...]
    transcripts: tuple[TranscriptCandidate, ...]
    documents: tuple[Path, ...]
    audio_count: int


def _pair_transcript(
    audio: Path,
    relative_audio: Path,
    transcripts: Iterable[TranscriptCandidate],
) -> TranscriptCandidate | None:
    candidates = list(transcripts)
    if not candidates:
        return None
    audio_stem = normalized_text(audio.stem)
    audio_name = normalized_text(audio.name)
    parent = normalized_text(relative_audio.parent.as_posix())

    ranked: list[tuple[float, TranscriptCandidate]] = []
    for transcript in candidates:
        transcript_rel = Path(transcript.relative_path)
        transcript_parent = normalized_text(transcript_rel.parent.as_posix())
        transcript_stem = _transcript_stem(transcript.path)
        score = 0.0
        raw_name = normalized_text(transcript.path.name)
        if raw_name.startswith(audio_name + "."):
            score += 100.0
        if transcript_stem == audio_stem:
            score += 80.0
        if transcript_parent == parent:
            score += 20.0
        similarity = SequenceMatcher(None, transcript_stem, audio_stem).ratio()
        score += similarity * 20.0
        if score >= 45.0:
            ranked.append((score, transcript))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], natural_key(item[1].relative_path)))
    return ranked[0][1]


def _background_score(path: Path, root: Path) -> tuple[int, int, tuple[tuple[int, object], ...]]:
    rel = path.relative_to(root)
    text = normalized_text(rel.as_posix())
    score = 0
    if normalized_text(path.stem) == "null":
        score += 1000
    if re.search(r"(?:cover|jacket|ジャケット|表紙|封面|メインビジュアル)", text):
        score += 500
    if len(rel.parts) == 1:
        score += 100
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return -score, -size, natural_key(rel.as_posix())


def _edition_label(
    directory: str,
    extension: str,
    language: str,
    mix_variant: str,
    orientation: str,
) -> str:
    parts: list[str] = []
    if directory and directory != ".":
        parts.append(directory)
    parts.append(extension.removeprefix(".").upper())
    if language == "zh":
        parts.append("中文目录")
    elif language == "en":
        parts.append("英文目录")
    if mix_variant == "with_se":
        parts.append("SE 有")
    elif mix_variant == "no_se":
        parts.append("SE 无")
    elif mix_variant == "voice_only":
        parts.append("纯人声")
    if orientation == "reversed":
        parts.append("左右反转")
    return " / ".join(parts)


def _edition_score(
    extension: str,
    language: str,
    mix_variant: str,
    orientation: str,
    tracks: list[TrackCandidate],
    directory: str,
) -> int:
    score = _FORMAT_SCORE.get(extension, 0)
    score += 25 if language == "ja" else -10
    score += 8 if orientation == "normal" else -15
    score += {"with_se": 7, "standard": 5, "no_se": 2, "voice_only": -2}.get(mix_variant, 0)
    score += min(30, len(tracks) * 3)
    if _SAMPLE_RE.search(directory):
        score -= 100
    if tracks and all(item.category != "main" for item in tracks):
        score -= 100
    return score


def _legacy_edition(tracks: list[TrackCandidate]) -> bool:
    if not tracks or any(Path(item.relative_path).parent != Path(".") for item in tracks):
        return False
    numbered = [
        item.order_key[1] for item in tracks if item.order_key[0] == 0 and item.order_key[1] < 10**9
    ]
    return len(numbered) == len(tracks) and len(numbered) == len(set(numbered))


def scan_work(
    root: Path,
    *,
    excluded_directories: Iterable[str] | None = None,
) -> ScanResult:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"作品文件夹不存在：{root}")

    audio_paths: list[Path] = []
    transcript_paths: list[Path] = []
    images: list[Path] = []
    documents: list[Path] = []
    excluded = {
        "autoflow输出",
        "autoflow output",
        ".git",
        ".state",
        ".work",
    }
    excluded.update(normalized_text(item) for item in (excluded_directories or ()))

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(normalized_text(part) in excluded for part in relative.parts[:-1]):
            continue
        suffix = path.suffix.casefold()
        if suffix in AUDIO_EXTENSIONS:
            # Do not feed previous AutoFlow products or common temporary
            # renders back into a later scan of the same source folder.
            stem = normalized_text(path.stem)
            if stem in {"原声", "双语版", "mixed", "chinese", "shared-reference"}:
                continue
            if stem.endswith("_batch") or stem.startswith(".autoflow"):
                continue
            audio_paths.append(path.resolve())
        elif suffix in TRANSCRIPT_EXTENSIONS:
            transcript_paths.append(path.resolve())
            documents.append(path.resolve())
        elif suffix in IMAGE_EXTENSIONS:
            images.append(path.resolve())

    transcripts = tuple(
        TranscriptCandidate(
            path=path,
            relative_path=path.relative_to(root).as_posix(),
            language=_transcript_language(
                path.relative_to(root),
                content_path=path,
            ),
            timed=path.suffix.casefold() in TIMED_TRANSCRIPT_EXTENSIONS,
        )
        for path in sorted(
            transcript_paths, key=lambda item: natural_key(item.relative_to(root).as_posix())
        )
    )

    tracks: list[TrackCandidate] = []
    for path in audio_paths:
        relative = path.relative_to(root)
        path_text = normalized_text(relative.as_posix())
        stem_text = normalized_text(path.stem)
        tracks.append(
            TrackCandidate(
                path=path,
                relative_path=relative.as_posix(),
                title=clean_track_title(path.stem),
                extension=path.suffix.casefold(),
                category=_category(path_text, stem_text),
                language=_language(path_text),
                mix_variant=_mix_variant(path_text),
                orientation=_orientation(path_text),
                order_key=track_order(path.stem),
                transcript=_pair_transcript(path, relative, transcripts),
            )
        )

    # 一个干净的根目录数字音轨集合继续视为一个版本，即使格式混用。
    main_tracks = [item for item in tracks if item.category == "main"]
    groups: dict[tuple[str, str, str, str, str], list[TrackCandidate]] = {}
    if _legacy_edition(main_tracks):
        key = (".", "mixed", "ja", "standard", "normal")
        groups[key] = tracks
    else:
        for track in tracks:
            parent = Path(track.relative_path).parent.as_posix()
            key = (
                parent,
                track.extension,
                track.language,
                track.mix_variant,
                track.orientation,
            )
            groups.setdefault(key, []).append(track)

    editions: list[Edition] = []
    for (directory, extension, language, mix_variant, orientation), items in groups.items():
        ordered = sorted(items, key=lambda item: (item.order_key, natural_key(item.relative_path)))
        primary = [item for item in ordered if item.category == "main"]
        optional = [item for item in ordered if item.category != "main"]
        if not primary:
            # 纯特典目录仍可作为单独版本选择。
            primary, optional = optional, []
        digest = hashlib.sha256(
            "\n".join(item.relative_path for item in ordered).encode("utf-8")
        ).hexdigest()[:10]
        display_extension = extension if extension != "mixed" else ".mixed"
        label = _edition_label(directory, display_extension, language, mix_variant, orientation)
        legacy = directory == "." and extension == "mixed"
        editions.append(
            Edition(
                id=digest,
                label=label,
                directory=directory,
                extension=display_extension,
                language=language,
                mix_variant=mix_variant,
                orientation=orientation,
                tracks=tuple(primary),
                optional_tracks=tuple(optional),
                score=_edition_score(
                    display_extension if display_extension != ".mixed" else primary[0].extension,
                    language,
                    mix_variant,
                    orientation,
                    primary,
                    directory,
                ),
                legacy_compatible=legacy,
            )
        )

    editions.sort(key=lambda item: (-item.score, natural_key(item.label)))
    images.sort(key=lambda path: _background_score(path, root))
    return ScanResult(
        root=root,
        editions=tuple(editions),
        images=tuple(images),
        transcripts=transcripts,
        documents=tuple(
            sorted(
                set(documents),
                key=lambda item: natural_key(item.relative_to(root).as_posix()),
            )
        ),
        audio_count=len(audio_paths),
    )


def with_transcript(
    track: TrackCandidate, transcript: TranscriptCandidate | None
) -> TrackCandidate:
    """供手动选择流程覆盖自动匹配结果。"""

    return replace(track, transcript=transcript)
