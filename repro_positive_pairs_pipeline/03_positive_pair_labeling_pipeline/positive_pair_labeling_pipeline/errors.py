"""Typed pipeline failures used to fail closed on stale or partial runs."""


class PipelineError(RuntimeError):
    """Base class for an expected pipeline failure."""


class ConfigurationError(PipelineError):
    """The immutable run configuration is invalid or changed."""


class ArtifactIntegrityError(PipelineError):
    """An artifact is absent, stale, or does not match its recorded digest."""


class StageOrderError(PipelineError):
    """A stage was requested before its upstream dependency completed."""


class BatchStateError(PipelineError):
    """A remote batch is incomplete, failed, ambiguous, or inconsistent."""


class ResponseValidationError(PipelineError):
    """Batch output is incomplete, malformed, or cannot be aligned safely."""
