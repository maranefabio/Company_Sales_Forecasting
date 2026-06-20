import datetime as dt
import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(output_path: str | None = None) -> None:
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if output_path is not None:
        os.makedirs(os.path.join(output_path, 'logs'), exist_ok=True)
        logs_path: str = os.path.join(output_path, 'logs')
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

