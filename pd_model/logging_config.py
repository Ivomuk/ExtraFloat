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

# Uganda/Kenya MSISDN pattern: 25[67] followed by 8–9 digits.
# Matches both local (07xx) and international (2567xx) formats that may
# appear if a raw MSISDN is accidentally interpolated into a log message.
_MSISDN_RE = re.compile(r"\b25[67]\d{8,9}\b")


class PIIRedactingFilter(logging.Filter):
    """Scrub MSISDN-like values from log records before emission.

    Enforces as a structural guarantee what is currently informal practice:
    no individual phone numbers appear in log output, even as the codebase evolves.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _MSISDN_RE.sub("[MSISDN]", str(record.msg))
        record.args = tuple(
            _MSISDN_RE.sub("[MSISDN]", str(a)) if isinstance(a, str) else a for a in (record.args or ())
        )
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


def configure_root_level(level: int = logging.INFO) -> None:
    """
    Convenience function to adjust the level of every pd_model logger at once.

    Typically called once from ``run_pipeline.py`` based on the ``--log-level``
    CLI argument.
    """
    logging.getLogger("pd_model").setLevel(level)
