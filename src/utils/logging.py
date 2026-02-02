"""Structured logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    """Configure structured logging with rich console output.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path for JSON output
        verbose: Enable verbose debug output
    """
    # Determine log level
    level = logging.DEBUG if verbose else getattr(logging, log_level.upper())

    # Configure structlog processors
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Add format_exc_info for better exception handling
    if verbose:
        processors.append(structlog.processors.format_exc_info)

    # Console output with Rich
    console = Console(stderr=True)
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=verbose,
        markup=True,
    )
    console_handler.setLevel(level)

    handlers: list[logging.Handler] = [console_handler]

    # File output with JSON formatting
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=processors,
            )
        )
        handlers.append(file_handler)

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=handlers,
        force=True,
    )

    # Configure structlog
    structlog.configure(
        processors=processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Suppress noisy libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)
