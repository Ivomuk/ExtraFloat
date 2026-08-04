"""
Centralised logging setup for the PD model pipeline.

Usage in any module::

    from pd_model.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("...")
"""

from __future__ import annotations

import logging
import re
import sys

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Uganda MSISDN formats: international (256xxxxxxxxx, optional leading +)
# and local mobile (07xxxxxxxx).
_MSISDN_RE = re.compile(r"\b(?:\+?256\d{9}|07\d{8})\b")


class PIIRedactingFilter(logging.Filter):
    """Scrub MSISDN-like values from log records before emission.

    Enforces as a structural guarantee what is currently informal practice:
    no individual phone numbers appear in log output, even as the codebase evolves.

    Covered formats (Uganda):
    - International: 256xxxxxxxxx or +256xxxxxxxxx
    - Local 0-prefix: 07xxxxxxxx
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Format eagerly so dict/list/non-string args embedded via %s are also scrubbed.
        record.msg = _MSISDN_RE.sub("[MSISDN]", record.getMessage())
        record.args = ()  # already baked into msg; prevents double-formatting
        return True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with a StreamHandler attached to stdout.

    Idempotent: calling this multiple times with the same *name* will not
    attach duplicate handlers.

    Args:
        name:  Logger name; pass ``__name__`` from the calling module.
        level: Initial log level (default INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        handler.addFilter(PIIRedactingFilter())
        logger.addHandler(handler)

    logger.setLevel(level)
    # Prevent double-logging when the root logger also has handlers
    logger.propagate = False

    return logger


def install_pii_filter() -> None:
    """Install PIIRedactingFilter on all current root logger handlers.

    Call once at application startup after ``logging.basicConfig()`` to protect
    loggers that propagate to root — extrafloat modules, run_* entry-point scripts,
    and any third-party library logger.

    Idempotent: a second call will not add a duplicate filter to a handler that
    already has one.
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, PIIRedactingFilter) for f in handler.filters):
            handler.addFilter(PIIRedactingFilter())


def configure_root_level(level: int = logging.INFO) -> None:
    """
    Convenience function to adjust the level of every pd_model logger at once.

    Typically called once from ``run_pipeline.py`` based on the ``--log-level``
    CLI argument.
    """
    logging.getLogger("pd_model").setLevel(level)
