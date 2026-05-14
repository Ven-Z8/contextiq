"""Structured logging setup."""

from __future__ import annotations

import logging

import structlog


def configure_logging() -> None:
    """Configure structlog for local CLI and API use."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
