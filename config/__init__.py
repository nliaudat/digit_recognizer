# config/__init__.py
"""
Configuration package for Digit Recognizer.

This package provides a modular replacement for the monolithic ``parameters.py``.
For backward compatibility, ``parameters.py`` re-exports everything from this
package so existing imports continue to work.

Usage (new code):
    from config import NB_CLASSES, BATCH_SIZE, ...
    from config.validation import validate_full_config

Usage (legacy, still works):
    import parameters as params
    params.NB_CLASSES  # still works via re-export
"""
import warnings
import sys

# --------------------------------------------------------------------------- #
#  Re-export everything from parameters.py for backward compatibility
# --------------------------------------------------------------------------- #
# This allows `from config import NB_CLASSES` to work transparently.
# The actual parameter definitions remain in parameters.py during the
# migration period.

# Import all public names from parameters.py
import parameters as _params

# Collect all public names (non-underscore) from parameters
_public_names = [name for name in dir(_params) if not name.startswith('_')]

# Re-export them all
for _name in _public_names:
    globals()[_name] = getattr(_params, _name)

# Emit deprecation warning when someone imports from parameters.py directly
# (handled by the re-exporter in parameters.py itself)

# Make submodules available
from . import validation

__all__ = _public_names + ['validation']
