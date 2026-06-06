"""
utils/logging_setup.py
-----------------------
Centralised logging configuration for the whole project.
Import get_logger() in any module instead of using bare print() calls.

Usage:
    from utils.logging_setup import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

import logging
import sys
from utils.config import LOG_LEVEL, LOG_FILE


def get_logger(name: str = "pneumonia_vit") -> logging.Logger:
    """
    Return a named logger configured with:
      - StreamHandler → stdout (always)
      - FileHandler   → LOG_FILE (if set in config)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when module is re-imported
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if LOG_FILE:
        try:
            file_handler = logging.FileHandler(LOG_FILE)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            logger.warning(f"Could not open log file at '{LOG_FILE}'. File logging disabled.")

    return logger
