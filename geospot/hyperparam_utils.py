"""
Utilities for guessing good hyperparameters for VLM fine-tuning.
"""


def get_lr(model_name: str, is_lora: bool = True) -> float:
    """Get recommended learning rate for a model."""
    base_lr = 5e-05
    lora_multiplier = 10.0

    lr = base_lr * lora_multiplier if is_lora else base_lr

    # Adjust based on model family
    if "qwen" in model_name.lower():
        # Qwen VL models tend to work well with slightly lower LR
        lr = lr * 0.5

    return lr


def get_lora_lr_over_full_finetune_lr(model_name: str, lora_alpha: int = 32) -> float:
    """
    Return the factor to scale full fine-tuning LR by for LoRA.
    """
    return 10.0
