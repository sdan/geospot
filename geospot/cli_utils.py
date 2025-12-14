"""
CLI utilities for handling log directories.
"""

import logging
import os
import shutil
from typing import Literal

logger = logging.getLogger(__name__)

LogdirBehavior = Literal["delete", "resume", "ask", "raise"]


def check_log_dir(log_dir: str, behavior_if_exists: LogdirBehavior):
    """
    Call this at the beginning of CLI entrypoint to training scripts.

    Args:
        log_dir: The directory to check.
        behavior_if_exists: What to do if the log directory already exists.
            "ask": Ask user if they want to delete the log directory.
            "resume": Continue to the training loop (try to resume from checkpoint).
            "delete": Delete the log directory and start fresh.
            "raise": Raise an error if the log directory already exists.
    """
    if os.path.exists(log_dir):
        if behavior_if_exists == "delete":
            logger.info(
                f"Log directory {log_dir} already exists. Deleting and starting fresh."
            )
            shutil.rmtree(log_dir)
        elif behavior_if_exists == "ask":
            while True:
                user_input = input(
                    f"Log directory {log_dir} already exists. [delete/resume/exit]: "
                )
                if user_input == "delete":
                    shutil.rmtree(log_dir)
                    return
                elif user_input == "resume":
                    return
                elif user_input == "exit":
                    exit(0)
                else:
                    logger.warning(f"Invalid input: {user_input}")
        elif behavior_if_exists == "resume":
            return
        elif behavior_if_exists == "raise":
            raise ValueError(f"Log directory {log_dir} already exists.")
        else:
            raise AssertionError(f"Invalid behavior_if_exists: {behavior_if_exists}")
    else:
        logger.info(f"Log directory {log_dir} does not exist. Will create it.")
