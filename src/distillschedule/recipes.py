"""Locate the installed, curated recipe resources without importing models.

Recipes are data, not Python packages. Helpers run at their installed paths so
their sibling templates remain available; native recipes still require their
explicit external source/assets. There is deliberately no cwd/source fallback.
"""
from importlib import metadata
from pathlib import Path


def directory() -> Path:
    """Return this distribution's recipe directory, or fail on an incomplete install."""
    distribution = metadata.distribution('distillschedule')
    suffix = 'share/distillschedule/recipes/README.md'
    anchors = [entry for entry in distribution.files or () if str(entry).endswith(suffix)]
    if len(anchors) != 1:
        raise RuntimeError('installed recipe resources are missing; install the complete framework wheel')
    anchor = Path(distribution.locate_file(anchors[0])).resolve()
    if not anchor.is_file():
        raise RuntimeError('installed recipe resources are missing: ' + str(anchor))
    return anchor.parent
