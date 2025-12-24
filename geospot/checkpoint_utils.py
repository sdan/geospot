"""Checkpoint utilities for resumable training."""

import json
import logging
import os
from typing import Any

import tinker

logger = logging.getLogger(__name__)

CHECKPOINTS_FILE = "checkpoints.jsonl"


def get_last_checkpoint(log_dir: str, required_key: str = "state_path") -> dict[str, Any] | None:
    """Get the last checkpoint from checkpoints.jsonl in log directory."""
    checkpoint_path = os.path.join(log_dir, CHECKPOINTS_FILE)
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoints found at {checkpoint_path}")
        return None

    checkpoints = []
    with open(checkpoint_path, "r") as f:
        for line in f:
            if line.strip():
                checkpoints.append(json.loads(line))

    checkpoints_with_key = [c for c in checkpoints if required_key in c]
    if checkpoints_with_key:
        logger.info(f"Found {len(checkpoints_with_key)} checkpoints in {log_dir}")
        logger.info(f"Resuming from: {checkpoints_with_key[-1]}")
        return checkpoints_with_key[-1]
    else:
        logger.info(f"No checkpoints with key '{required_key}' in {log_dir}")
        return None


async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    step: int,
) -> str:
    """Save checkpoint and record in checkpoints.jsonl."""
    result = await (await training_client.save_state_async(name)).result_async()
    state_path = result.path

    checkpoint_info = {
        "name": name,
        "step": step,
        "state_path": state_path,
    }

    with open(os.path.join(log_path, CHECKPOINTS_FILE), "a") as f:
        f.write(json.dumps(checkpoint_info) + "\n")

    logger.info(f"Saved checkpoint: {name} -> {state_path}")
    return state_path


def save_checkpoint(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    step: int,
) -> str:
    """Sync version of save_checkpoint_async."""
    import asyncio
    return asyncio.run(save_checkpoint_async(training_client, name, log_path, step))
