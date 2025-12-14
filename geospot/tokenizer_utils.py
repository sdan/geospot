"""
Utilities for working with tokenizers.

Avoid importing AutoTokenizer and PreTrainedTokenizer until runtime, because they're slow imports.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    Tokenizer: TypeAlias = PreTrainedTokenizer
else:
    Tokenizer: TypeAlias = Any


@cache
def get_tokenizer(model_name: str) -> Tokenizer:
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    kwargs: dict[str, Any] = {}

    # Qwen models may need trust_remote_code
    if "qwen" in model_name.lower():
        kwargs["trust_remote_code"] = True

    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
