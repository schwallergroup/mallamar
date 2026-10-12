# Entry point for `python -m agent_lg`.
import logging
from agent_lg.cli import main

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
