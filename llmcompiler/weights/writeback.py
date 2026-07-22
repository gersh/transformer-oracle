"""
Layer 8: Register Writeback.

Writes the ALU result (or loaded memory value) to the destination register.
Uses the decoded rd field to select which register column to update.

The writeback is a demux operation: for each register column, either
pass through unchanged (if not rd) or replace with the new value (if rd).

x0 (register 0) is never written - hardwired to zero.
"""

import torch
from ..core.state import StateTensor, VALUE_START, VALUE_END, int_to_bipolar


def writeback_register(state: StateTensor, rd: int, value_bipolar: torch.Tensor):
    """Write a bipolar value to register rd in the state tensor.

    Args:
        state: the state tensor to modify
        rd: destination register index (0-31)
        value_bipolar: bipolar-encoded 32-bit result to write
    """
    if rd == 0:
        return  # x0 is hardwired to zero
    assert 0 < rd < state.config.n_regs
    state.data[VALUE_START:VALUE_END, rd] = value_bipolar


def update_pc(state: StateTensor, new_pc: int):
    """Update the program counter."""
    state.set_pc(new_pc)
