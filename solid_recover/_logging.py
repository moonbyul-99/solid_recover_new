"""Centralised logger factory for solid_recover.

Replaces scattered ``print('✅ ...')`` / ``print('🚀 ...')`` calls with proper
Python logging. Users can tune verbosity via the ``SR_LOGLEVEL`` env var
(``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_DEFAULT_FMT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("SR_LOGLEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("solid_recover")
    root.setLevel(level)
    # Avoid double handlers when imported multiple times (e.g. in notebooks).
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT))
        root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-scoped logger under the ``solid_recover`` namespace."""
    _configure_root()
    if name is None or name == "solid_recover":
        return logging.getLogger("solid_recover")
    if name.startswith("solid_recover"):
        return logging.getLogger(name)
    return logging.getLogger(f"solid_recover.{name}")
