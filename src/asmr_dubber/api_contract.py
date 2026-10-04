"""Validation for the native JSON interface; no application policy lives here."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Empty(Request):
    pass


class FilePath(Request):
    path: str


class ProjectRequest(Request):
    project: str = Field(min_length=1)


class TableRequest(ProjectRequest):
    revision: int = Field(ge=0)
    rows: list[list[Any]]


class SentenceRequest(ProjectRequest):
    revision: int = Field(ge=0)
    row: list[Any] = Field(min_length=10, max_length=10)


class SettingsRequest(Request):
    changes: dict[str, Any]
    project: str | None = None
    revision: int | None = Field(default=None, ge=0)


class KeyRequest(Request):
    kind: Literal["translation", "service"]
    name: str
    value: str = ""
    clear: bool = False


class ModelRequest(Request):
    model: str


class TaskID(Request):
    identifier: str


class TaskRequest(Request):
    kind: Literal[
        "create",
        "analyze",
        "translate",
        "synthesize",
        "mix",
        "export",
        "subtitles",
        "import",
        "accept_review",
        "keep_review",
        "undo_review",
        "unlock_review",
        "retry_review",
        "align_review",
        "preview_reference",
        "preview_edge",
        "download",
        "import_models",
        "batch",
        "health",
        "repair",
        "diagnostic",
    ]
    project: str | None = None
    revision: int | None = Field(default=None, ge=0)
    source: str | None = None
    source_language: Literal["ja", "en", "zh"] = "ja"
    target_language: Literal["zh", "en"] | None = None
    force: bool = False
    audio: bool = True
    language: Literal["bilingual", "source", "zh", "none"] = "bilingual"
    file: str | None = None
    text: str = ""
    timing: Literal["estimate", "qwen", "script_review"] = "estimate"
    script_kind: Literal["source", "zh"] = "source"
    window: str = ""
    candidate: str = ""
    sentence: str = ""
    voice: str = ""
    model: str = ""
    plan: str = ""
    backend: str = ""
    action: str = ""
    service: Literal["asr", "translation", "tts"] = "translation"

    @model_validator(mode="after")
    def required_fields(self):
        if self.kind == "create" and not self.source:
            raise ValueError("source is required")
        project_actions = {
            "analyze",
            "translate",
            "synthesize",
            "mix",
            "export",
            "subtitles",
            "import",
            "accept_review",
            "keep_review",
            "undo_review",
            "unlock_review",
            "retry_review",
            "align_review",
            "preview_reference",
        }
        if self.kind in project_actions and (not self.project or self.revision is None):
            raise ValueError("project and revision are required")
        if self.kind == "download" and not self.model:
            raise ValueError("model is required")
        return self


class ReferenceRequest(ProjectRequest):
    sentence: str = ""
    external: str = ""
    text: str = ""
    language: Literal["auto", "ja", "en", "zh"] = "auto"
    start: float | None = None
    end: float | None = None
    revision: int = Field(ge=0)


class ReviewRequest(ProjectRequest):
    window: str = ""
    candidate: str = ""


class OpenRequest(ProjectRequest):
    output: bool = False


class ScanRequest(Request):
    folder: str
    include_bonus: bool | None = None


class EditionRequest(ScanRequest):
    edition: str
    include_bonus: bool = False


class TracksRequest(Request):
    folder: str
    sources: list[dict[str, Any]]
    order: list[str]


class SubtitleRequest(Request):
    folder: str
    sources: list[dict[str, Any]]
    track: str
    file: str
    language: str
    mode: str = "direct"


class PlanRequest(Request):
    folder: str
    edition: str
    sources: list[dict[str, Any]]
    mode: str
    layout: str
    background: str = "black"
    embed_subtitles: bool = False
    rebuild: bool = False
    content: Literal["dubbing", "replacement", "both", "subtitles", "source_subtitles"] = "dubbing"
    subtitle_language: Literal["bilingual", "source", "zh"] = "bilingual"
    editing: str = ""


class OrderRequest(Request):
    order: list[str]


class StorageRequest(Request):
    root: str
    categories: list[str]


class CleanRequest(Request):
    plan: dict[str, Any]
    selected: list[str]
    confirmed: bool
