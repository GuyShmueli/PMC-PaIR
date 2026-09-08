"""Typed pipeline failures used to fail closed on stale or partial runs."""


class PipelineError(RuntimeError):
    """Base class for an expected pipeline failure."""


class ConfigurationError(PipelineError):
    """The immutable run configuration is invalid or changed."""


class ArtifactIntegrityError(PipelineError):
    """An artifact is absent, stale, or does not match its recorded digest."""


class DiagramFilteringError(PipelineError):
    """An image could not be classified by the diagram filter."""
