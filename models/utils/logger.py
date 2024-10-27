import logging
import sys


def config():
    """
    Config custom logging
    """
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Create a handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)

    # Create a formatter and add it to a handler
    logger_format = logging.Formatter('[%(levelname)s][%(asctime)s][TwoTowerLog] - %(message)s')
    c_handler.setFormatter(logger_format)

    # Add handler to the logger
    logger.addHandler(c_handler)

    logger.setLevel(logging.INFO)

    return logger


def get_logger() -> logging.Logger:
    """
    Get the global logger
    """
    global LOGGER
    return LOGGER


LOGGER = config()