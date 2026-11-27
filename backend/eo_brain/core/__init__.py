"""Core data access layer.

The data cube names are imported on first use (PEP 562), so the pure modules
(core.scenes, core.coverage) import without the STAC stack: the offline CI
job tests them with requirements-ci.txt only.
"""
__all__ = ["DataCube", "LoadResult", "LoadConfig", "BBox", "TimeRange", "SearchParams", "get_provider", "PROVIDERS"]


def __getattr__(name):
    if name in __all__:
        from eo_brain.core import datacube
        return getattr(datacube, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
