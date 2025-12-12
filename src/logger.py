# logger.py

# Central logging configuration using RotatingFileHandler
# This logger writes logs to a single file (project.log) and automatically rotates the file when it reaches a certain size. 
# It also logs filename, line number and function name for easier debugging.

import logging
import os
from logging.handlers import RotatingFileHandler

# ---------------------------------------------------------
# 1. Create a directory to store log files
# ---------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. Define the main log file path
# ---------------------------------------------------------
LOG_FILE = os.path.join(LOG_DIR, "project.log")

# ---------------------------------------------------------
# 3. Configure RotatingFileHandler
#    - maxBytes: rotate when file reaches 5 MB
#    - backupCount: keep last 5 rotated log files
# ---------------------------------------------------------
handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,   # 5 MB
    backupCount=5,              # keep last 5 logs
    encoding="utf-8"
)

# ---------------------------------------------------------
# 4. Define the logging format
#    %(filename)s  -> file where log happened
#    %(lineno)d    -> line number
#    %(funcName)s  -> function name
# ---------------------------------------------------------
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s "
    "[%(filename)s:%(lineno)d - %(funcName)s()] "
    "- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

handler.setFormatter(formatter)

# ---------------------------------------------------------
# 5. Configure the root logger
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler]
)

# ---------------------------------------------------------
# 6. Test log (only runs when executing this file directly)
# ---------------------------------------------------------
if __name__ == '__main__':
    logging.info("Logging has been initiated...")
    