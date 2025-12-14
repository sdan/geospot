"""HuggingFace Hub streaming utilities."""

import os
import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def get_hf_shard_urls(
    repo: str,
    token: str | None = None,
    prefix: str | None = None,
    exclude_prefix: str | None = "eval/",
) -> list[str]:
    """Fetch tar shard URLs from HuggingFace Hub for WebDataset streaming."""
    from huggingface_hub import list_repo_files

    if token is None:
        token = os.environ.get("HF_TOKEN")

    logger.info(f"Fetching shard list from HuggingFace: {repo}")
    files = list(list_repo_files(repo, repo_type="dataset", token=token))
    tar_files = [f for f in files if f.endswith(".tar")]

    if prefix:
        prefixes = [p.strip() for p in prefix.split(",")]
        tar_files = [f for f in tar_files if any(f.startswith(p) for p in prefixes)]
        if exclude_prefix and not any(p.startswith("eval") for p in prefixes):
            tar_files = [f for f in tar_files if not f.startswith(exclude_prefix)]
    elif exclude_prefix:
        tar_files = [f for f in tar_files if not f.startswith(exclude_prefix)]

    tar_files = sorted(tar_files)
    base_url = f"https://huggingface.co/datasets/{repo}/resolve/main"
    urls = [f"{base_url}/{f}" for f in tar_files]

    logger.info(f"Found {len(urls):,} shards")
    return urls


def make_authenticated_urls(urls: Sequence[str], token: str) -> list[str]:
    """Convert URLs to authenticated pipe commands for private repos."""
    return [
        f"pipe:curl -s -L -H 'Authorization: Bearer {token}' '{url}'"
        for url in urls
    ]
