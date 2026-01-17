import logging
import sys

# 1. Define log format
logging_str = "[%(asctime)s] [%(levelname)s] %(message)s"

# 2. Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 3. Create the logger object (This is what was missing!)
logger = logging.getLogger("MultimodalRAG")