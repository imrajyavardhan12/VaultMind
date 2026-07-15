"""Structlog configuration.

User-facing output goes to stdout via Rich.
Structured logs go to file. Debug mode sends human-readable logs to stderr.
"""

from __future__ import annotations

import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, NoReturn

import structlog


class BufferedLogs:
    """Capture preflight events in memory until a command confirms real work."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def capture(
        self,
        _logger: Any,
        method_name: str,
        event_dict: MutableMapping[str, Any],
    ) -> NoReturn:
        self.events.append((method_name, dict(event_dict)))
        raise structlog.DropEvent

    def replay(self) -> None:
        """Replay captured events through the currently configured logger."""
        events, self.events = self.events, []
        logger = structlog.get_logger()
        for method_name, event_dict in events:
            event = event_dict.pop("event", "")
            method = getattr(logger, method_name, logger.info)
            method(event, **event_dict)


def setup_buffered_logging() -> BufferedLogs:
    """Replace any prior logger with a write-free in-memory preflight buffer."""
    buffered = BufferedLogs()
    structlog.configure(
        processors=[buffered.capture],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        logger_factory=structlog.ReturnLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    return buffered


def setup_logging(verbose: bool = False) -> None:
    """Configure structlog. Logs go to file; verbose mode adds stderr output."""
    log_dir = Path.home() / ".local" / "share" / "vaultmind"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vaultmind.log"

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if verbose:
        structlog.configure(
            processors=[
                *processors,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(0),
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        )
    else:
        structlog.configure(
            processors=[
                *processors,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(20),
            logger_factory=structlog.PrintLoggerFactory(file=open(log_file, "a")),  # noqa: SIM115
        )
