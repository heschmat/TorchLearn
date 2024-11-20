# logger

import os
import logging

# Set up the logger configuration only once
LOG_FILE = "logs/pipeline.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logger = logging.getLogger("pipeline_logger")
logger.setLevel(logging.INFO)

# Only add handlers if they don't already exist (avoids duplicate handlers)
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
