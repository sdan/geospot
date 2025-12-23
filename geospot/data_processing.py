"""
Data processing functions for GRPO training.

Converts trajectories to training data with:
- target_tokens: next-token prediction targets
- logprobs: per-token log probabilities from sampling
- advantages: group-centered rewards (GRPO)
- mask: 1.0 for action tokens, 0.0 for observation tokens

Reference: tinker_cookbook/rl/data_processing.py
"""

import logging
from typing import List

import tinker
import torch
from tinker import TensorData

from geospot.types import Trajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


def create_rightshifted_model_input_and_leftshifted_targets(
    chunks: list[tinker.ModelInputChunk],
) -> tuple[tinker.ModelInput, list[int]]:
    """
    Given a full sequence of model input chunks, create:
    - "inputs" (with last token removed)
    - "targets" (with first token removed)

    Reference: tinker_cookbook/supervised/common.py
    """
    assert len(chunks) >= 1, "must have at least one chunk"

    last_chunk = chunks[-1]
    if not isinstance(last_chunk, tinker.types.EncodedTextChunk):
        raise ValueError("The last chunk must be a text chunk")

    total_length = sum(c.length for c in chunks)
    if total_length < 2:
        raise ValueError("need at least 2 tokens for input/target split")

    # Build input chunks: all but last, then append truncated last chunk
    input_chunks: list[tinker.ModelInputChunk] = list(chunks[:-1])
    if last_chunk.length > 1:
        input_chunks.append(tinker.types.EncodedTextChunk(tokens=last_chunk.tokens[:-1]))

    # Build target tokens: collect all tokens, then slice off first
    all_tokens: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            all_tokens.extend(chunk.tokens)
        else:
            all_tokens.extend([0] * chunk.length)
    target_tokens = all_tokens[1:]

    return tinker.ModelInput(chunks=input_chunks), target_tokens


def compute_advantages(trajectory_groups_P: List[TrajectoryGroup]) -> List[torch.Tensor]:
    """
    Compute GRPO advantages: center rewards within each group.

    GRPO = Group Relative Policy Optimization
    advantage_i = reward_i - mean(rewards in group)
    """
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        advantages_G = rewards_G - rewards_G.mean()
        advantages_P.append(advantages_G)

    return advantages_P


# Type aliases for flattened observation representation
FlatObElem = int | tinker.ModelInputChunk
FlatOb = list[FlatObElem]


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """Check if seq1 is a prefix of seq2."""
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    """Count total token length of flattened observation."""
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _flat_ob_to_model_input(flat_ob: FlatOb) -> tinker.ModelInput:
    """Convert flattened observation back to ModelInput."""
    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=current_text_chunk))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return tinker.ModelInput(chunks=out)


def _flatten_chunks(chunks: list[tinker.ModelInputChunk]) -> FlatOb:
    """Flatten ModelInput chunks to list of tokens/chunks."""
    out: FlatOb = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def trajectory_to_data(traj: Trajectory, traj_advantage: float) -> list[tinker.Datum]:
    """
    Convert a trajectory to training Datum objects.

    If the sequence grows by appending (each observation contains previous
    observation+action as prefix), returns a single Datum. Otherwise, returns
    multiple Datums for non-contiguous sequences.

    Reference: tinker_cookbook/rl/data_processing.py
    """

    class SequenceAccumulator:
        full_sequence: list[FlatObElem] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
            list(all_tokens_T.chunks)
        )
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        return tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
                "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
                "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
                "mask": TensorData.from_torch(torch.tensor(mask_T)),
            },
        )

    data: list[tinker.Datum] = []
    for transition in traj.transitions:
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac

        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat

        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)

        # Observation tokens: logprobs=0, advantages=0, mask=0
        # Action tokens: logprobs from sampling, advantages from GRPO, mask=1
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        SequenceAccumulator.advantages.extend(
            [0.0] * delta_ob_len + [traj_advantage] * len(ac_with_logprobs.tokens)
        )
        SequenceAccumulator.mask.extend(
            [0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens)
        )

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def assemble_training_data(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P: List[torch.Tensor],
) -> tuple[list[tinker.Datum], list[dict[str, int]]]:
    """
    Convert trajectory groups to training data.

    Returns:
        data_D: List of Datum objects for training
        metadata_D: List of metadata dicts with group_idx and traj_idx
    """
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G) in enumerate(
        zip(trajectory_groups_P, advantages_P, strict=True)
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            zip(traj_group.trajectories_G, advantages_G, strict=True)
        ):
            new_data = trajectory_to_data(traj, float(traj_advantage))
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

    return data_D, metadata_D


def remove_constant_reward_groups(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[TrajectoryGroup]:
    """
    Remove groups where all trajectories have identical rewards.

    These groups have zero gradient (all advantages = 0 after centering).
    """
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        rewards = group.get_total_rewards()
        if len(set(rewards)) > 1:  # More than one unique reward
            new_groups.append(group)

    if not new_groups:
        logger.warning("All reward groups are uniform - no gradient signal")
        return trajectory_groups_P[0:1]  # Return one to avoid empty list issues

    return new_groups
