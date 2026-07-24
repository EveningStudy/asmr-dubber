class AsmrDubberError(RuntimeError):
    """Base error shown to CLI and UI users."""


class EnvironmentError(AsmrDubberError):
    """The local operating system or model runtime is unsuitable."""


class InstallPausedError(EnvironmentError):
    """A resumable dependency or model download was paused by the user."""


class ProjectError(AsmrDubberError):
    """A project manifest is missing or inconsistent."""


class TranslationError(AsmrDubberError):
    """The selected translation service did not return usable translations."""


class SynthesisError(AsmrDubberError):
    """Voice cloning failed."""
