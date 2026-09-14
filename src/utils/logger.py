"""Project-wide logging configuration."""

from __future__ import annotations

import logging
import sys

from src.config import LOG_FORMAT, LOG_LEVEL

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance.

    Ensures handlers are only attached once per process, so repeated
    calls (e.g. from notebooks re-running cells) don't duplicate log lines.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)

    if not _CONFIGURED:
        root = logging.getLogger()
        root.setLevel(LOG_LEVEL)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        if not root.handlers:
            root.addHandler(handler)
        _CONFIGURED = True

    return logger
