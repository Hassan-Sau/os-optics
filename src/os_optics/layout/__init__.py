"""Layout checks (2D/2.5D). Requires Shapely; importing this package without it raises ImportError, which the
pipeline converts into a DEPENDENCY_UNAVAILABLE / UNKNOWN result instead of crashing."""

from .checks import moving_envelope, run_layout_checks

__all__ = ["run_layout_checks", "moving_envelope"]
