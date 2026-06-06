"""
parameters.py — DEPRECATED: Thin re-exporter for the ``config`` package.

All parameters have been moved to the ``config/`` package for modularity.
This file re-exports everything from ``config`` for backward compatibility.

New code should import directly from ``config``:
    from config import NB_CLASSES, BATCH_SIZE, ...
    from config.validation import validate_full_config

Legacy imports (``import parameters as params``) continue to work.
"""

import warnings as _warnings

# Emit deprecation warning when someone imports parameters.py directly
_warnings.warn(
    "parameters.py is deprecated. Use `from config import ...` instead. "
    "See config/__init__.py for the new API.",
    PendingDeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the config package
from config import *  # noqa: F401, F403, E402