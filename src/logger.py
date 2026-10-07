"""
Configure logging to record project progress, warnings, and errors.
Include useful details such as timestamps, severity levels, and messages.
Send logs to a file or the console to help monitor and debug execution.
"""

import logging
import os
from datetime import datetime

# Create a timestamped filename.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Build and create only the logs directory.
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Build the file path inside that directory.
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging to write INFO and higher-severity messages to this file.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(linenno)d - %(levelname)s - %(message)s",
)

# Write a message to the log file.
logging.info("Logging has started")