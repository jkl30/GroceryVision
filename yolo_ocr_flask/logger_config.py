# logger_config.py

import logging

def configure_logging():
    """
    Configure logging for the application.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('object_detection.log'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)
