"""AutoFlow exception types."""

from __future__ import annotations

from typing import Any

from ..errors import ProjectError


class VideoPreparerError(ProjectError):
    pass


class StageFailedError(VideoPreparerError):
    """A pipeline stage failed; ``result`` is the stage's own JSON report, if any."""

    def __init__(self, stage: str, message: str, result: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.result = result
