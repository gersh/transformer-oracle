"""
Layer 1: Instruction Fetch via Attention.

Given the current PC value in the state tensor, fetch the instruction
at that address from instruction memory.

Supports two modes:
1. State-tensor mode: Instructions are columns in the state matrix.
   Attention Q (from PC column) matches against K (instruction column positions).
   The instruction with position == PC is selected.

2. MLP-baked mode: Instructions are encoded in MLP weights as a lookup table.
   One hidden neuron per instruction slot activates for exact PC match.

In both modes, the fetched instruction's encoding is copied into the
command buffer rows of the PC column (or a dedicated scratch column).
"""

import torch
import numpy as np
from ..core.state import (
    StateTensor, StateConfig, VALUE_START, VALUE_END,
    POSITION_START, POSITION_END, POSITION_BITS,
    int_to_bipolar, bipolar_to_int
)


def fetch_instruction_direct(state: StateTensor) -> int:
    """Direct (non-transformer) instruction fetch for testing.

    Reads the PC, computes the instruction index, and returns the
    packed instruction word from the correct instruction column.

    Returns the instruction index (not the column index).
    """
    pc = state.get_pc()
    # Each instruction is one word, PC increments by 1 per instruction
    # (In RISC-V, PC increments by 4 because instructions are 4 bytes,
    #  but in NISA we use word-addressed instruction memory)
    instr_idx = pc
    return instr_idx


def compute_fetch_attention_scores(state: StateTensor) -> torch.Tensor:
    """Compute attention scores for instruction fetch (state-tensor mode).

    The query is derived from the PC's value bits.
    The keys are the position encodings of instruction columns.

    Returns scores of shape (n_instr_slots,) where the instruction
    matching the current PC has the highest score.
    """
    config = state.config
    pc_value = state.get_pc()

    # Query: bipolar encoding of the PC value (used as instruction index)
    # We match against instruction column positions relative to instr_start
    pc_bipolar = int_to_bipolar(pc_value, POSITION_BITS)

    # Keys: position encodings of instruction columns (relative index)
    n_instr = config.n_instr_slots
    scores = torch.zeros(n_instr, dtype=torch.float64)

    for i in range(n_instr):
        # The position encoding in the column stores the absolute column index
        # We need to match PC (instruction index) against relative position
        key_bipolar = int_to_bipolar(i, POSITION_BITS)
        # Dot product of bipolar vectors: equals n_bits when perfect match,
        # less when bits differ. Each matching bit contributes +1, each
        # mismatching bit contributes -1.
        scores[i] = (pc_bipolar * key_bipolar).sum()

    return scores


def hard_attention_fetch(state: StateTensor) -> torch.Tensor:
    """Fetch instruction using hard attention (argmax).

    Returns the value bits of the selected instruction column.
    Since we're constructing weights analytically (no training),
    we can use argmax instead of softmax.
    """
    scores = compute_fetch_attention_scores(state)
    selected_idx = scores.argmax().item()

    # Read the value bits from the selected instruction column
    col = state.config.instr_start + selected_idx
    return state.data[VALUE_START:VALUE_END, col].clone()
