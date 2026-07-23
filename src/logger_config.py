"""
Root-logger configuration shared across the project: an always-on console handler plus an
optional rotating file handler.

The module "logger_config" adds:
    - A single "setup_logger" entry point configuring the root logger once per process, so that every
      module-level logger created elsewhere in the codebase (e.g. "logging.getLogger(__name__)")
      propagates through the same handlers.
    - An optional rotating log file (max 1 MiB per file, 3 backups kept) alongside a console handler
      that is always added.

The typical workflow is:
    >>> from src.logger_config import setup_logger
    >>> setup_logger(output_path=pipeline_settings.output_path)

    * elsewhere in the codebase:
    >>> logger = logging.getLogger(__name__)
"""


import datetime as dt
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logger(files_path: Path | None = None) -> None:
    """
    Configure the root logger with a console handler and, optionally, a rotating file handler.

    The root logger's level is set to DEBUG so that each handler's own level (DEBUG for the file
    handler, INFO for the console handler) is what actually determines what gets emitted where. If
    the root logger already has handlers attached — i.e. this function has already run once in the
    current process — it returns immediately without adding duplicate handlers.

    Arguments:
        - files_path: When provided, a "RotatingFileHandler" writing to
          "{files_path}/logs/log_{YYYYMMDD_HHMMSS}.txt" (max 1 MiB per file, 3 backups kept) is
          added at DEBUG level. When None, only the console handler (INFO level) is configured.
    """

    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if files_path is not None:
        logs_path: Path = files_path / 'logs'
        Path.mkdir(logs_path, exist_ok=True, parents=True)

        file_handler = RotatingFileHandler(
            os.path.join(
                logs_path, f'log_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            ),
            maxBytes=1024**2,
            backupCount=3
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)