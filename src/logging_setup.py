"""Centralised logging for Agentic RAG.

One logger tree under "agentrag" — node decisions flow through it identically
whether the graph runs from the CLI (`python -m src.main`) or the web app
(`python -m web.app`). Call `setup_logging()` once at each entry point.
"""

import logging
import sys

LOGGER_NAME = "agentrag"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a stderr handler to the "agentrag" logger. Idempotent.

    Safe to call multiple times (web/uvicorn may re-import) — a second call
    with handlers already present is a no-op.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # don't double-log through the root/uvicorn logger
    return logger
