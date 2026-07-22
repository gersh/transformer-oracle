"""
Transformer CPU: executes NISA programs via actual transformer forward passes.

The transformer operates on the state tensor (1, n_cols, d_state).
Each column is a "token" (register, PC, memory word, or instruction).
Each forward pass = one instruction cycle.

Architecture (10 functional stages, implemented as attention + MLP layers):

  L1  Attention: Instruction Fetch (PC → instruction lookup)
  L2  Attention: Operand Read (instruction → register values)
  L3  Attention: Memory Read (for LOAD instructions)
  L4  MLP:       ALU Execute (bipolar arithmetic)
  L5  MLP:       Opcode Routing (select correct ALU result)
  L6  MLP:       Branch Condition (evaluate branch predicates)
  L7  MLP:       PC Update (PC+1 or branch target)
  L8  Attention: Register Writeback (result → destination register)
  L9  Attention: Memory Write (for STORE instructions)
  L10 MLP:       Error Correction (snap to bipolar)

The weights are constructed ANALYTICALLY (no training). The program is
stored in instruction memory columns. The weights implement the CPU
logic and are program-independent.

For Phase 2, we implement a hybrid approach:
- The transformer architecture is defined as nn.Module
- Weights are computed analytically and loaded
- Forward passes run on GPU via PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from ..core.state import (
    StateTensor, StateConfig, DEFAULT_CONFIG,
    VALUE_START, VALUE_END, VALUE_BITS,
    POSITION_START, POSITION_END, POSITION_BITS,
    TYPE_START,
    int_to_bipolar, bipolar_to_int,
)
from ..core.nisa import Opcode, Instruction


class HardAttention(nn.Module):
    """Attention layer with analytically-set weights.

    Uses scaled dot-product attention with high temperature to approximate
    hard (one-hot) selection. For columns where attention is not needed
    (non-participating columns), the query is set to produce uniform low
    scores so they pass through unchanged via the residual.
    """

    def __init__(self, d_state: int, n_heads: int = 1, temperature: float = 50.0):
        super().__init__()
        self.d_state = d_state
        self.n_heads = n_heads
        self.d_head = d_state // n_heads
        self.temperature = temperature

        self.W_q = nn.Parameter(torch.zeros(d_state, d_state, dtype=torch.float64))
        self.W_k = nn.Parameter(torch.zeros(d_state, d_state, dtype=torch.float64))
        self.W_v = nn.Parameter(torch.zeros(d_state, d_state, dtype=torch.float64))
        self.W_o = nn.Parameter(torch.zeros(d_state, d_state, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, n_cols, d_state) → (1, n_cols, d_state)"""
        B, N, D = x.shape

        Q = x @ self.W_q.T  # (B, N, D)
        K = x @ self.W_k.T
        V = x @ self.W_v.T

        # Scaled dot-product attention with high temperature
        scores = (Q @ K.transpose(-2, -1)) * (self.temperature / (D ** 0.5))
        attn = F.softmax(scores, dim=-1)

        out = attn @ V
        out = out @ self.W_o.T

        return out  # residual added by the caller


class AnalyticalMLP(nn.Module):
    """MLP layer with analytically-set weights.

    Implements f(x) = W2 @ ReLU(W1 @ x + b1) + b2
    Applied independently to each column (token).
    """

    def __init__(self, d_state: int, d_hidden: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.zeros(d_hidden, d_state, dtype=torch.float64))
        self.b1 = nn.Parameter(torch.zeros(d_hidden, dtype=torch.float64))
        self.W2 = nn.Parameter(torch.zeros(d_state, d_hidden, dtype=torch.float64))
        self.b2 = nn.Parameter(torch.zeros(d_state, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, n_cols, d_state) → (1, n_cols, d_state)"""
        h = F.relu(x @ self.W1.T + self.b1)
        out = h @ self.W2.T + self.b2
        return out  # residual added by caller


class TransformerCPU(nn.Module):
    """The transformer that IS the CPU.

    Each forward pass executes one instruction cycle.
    Weights are program-independent (the program is in the state tensor).
    """

    def __init__(self, config: StateConfig = DEFAULT_CONFIG,
                 alu_hidden: int = 512, temperature: float = 50.0):
        super().__init__()
        self.config = config
        d = config.d_state

        # L1: Instruction Fetch (attention)
        self.fetch_attn = HardAttention(d, temperature=temperature)

        # L2: Operand Decode + Read (attention)
        self.decode_attn = HardAttention(d, temperature=temperature)

        # L3: Memory Read (attention)
        self.memread_attn = HardAttention(d, temperature=temperature)

        # L4-L5: ALU Execute + Opcode Routing (MLP)
        self.alu_mlp = AnalyticalMLP(d, alu_hidden)

        # L6-L7: Branch Condition + PC Update (MLP)
        self.branch_mlp = AnalyticalMLP(d, 128)

        # L8: Register Writeback (attention)
        self.writeback_attn = HardAttention(d, temperature=temperature)

        # L9: Memory Write (attention)
        self.memwrite_attn = HardAttention(d, temperature=temperature)

        # L10: Error Correction (MLP)
        self.error_mlp = AnalyticalMLP(d, d)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Execute one instruction cycle.

        Args:
            state: (1, n_cols, d_state) tensor in float64

        Returns:
            updated state: (1, n_cols, d_state) tensor
        """
        x = state

        # L1: Instruction Fetch
        x = x + self.fetch_attn(x)

        # L2: Operand Decode + Read
        x = x + self.decode_attn(x)

        # L3: Memory Read (LOAD)
        x = x + self.memread_attn(x)

        # L4-L5: ALU + Opcode Routing
        x = x + self.alu_mlp(x)

        # L6-L7: Branch + PC Update
        x = x + self.branch_mlp(x)

        # L8: Register Writeback
        x = x + self.writeback_attn(x)

        # L9: Memory Write (STORE)
        x = x + self.memwrite_attn(x)

        # L10: Error Correction (snap to bipolar)
        x = x + self.error_mlp(x)

        return x

    def execute(self, state: StateTensor, max_cycles: int = 10000,
                trace: bool = False) -> 'TransformerExecutionResult':
        """Run the transformer CPU iteratively until HALT.

        Args:
            state: initial StateTensor with loaded program
            max_cycles: maximum forward passes
            trace: print state each cycle

        Returns:
            TransformerExecutionResult
        """
        device = next(self.parameters()).device
        x = state.data.T.unsqueeze(0).to(device)  # (1, n_cols, d_state)

        cycles = 0
        halted = False

        for cycle in range(max_cycles):
            if trace:
                pc = _extract_pc(x, self.config)
                print(f"  cycle {cycle}: PC={pc}")

            x = self.forward(x)
            cycles = cycle + 1

            # Check halt condition
            pc = _extract_pc(x, self.config)
            if _check_halt(x, pc, self.config):
                halted = True
                break

        # Extract final state
        final_state = StateTensor(self.config)
        final_state.data = x.squeeze(0).T.cpu()

        return TransformerExecutionResult(final_state, cycles, halted)


class TransformerExecutionResult:
    """Result from transformer CPU execution."""

    def __init__(self, state: StateTensor, cycles: int, halted: bool):
        self.state = state
        self.cycles = cycles
        self.halted = halted

    def reg(self, idx: int) -> int:
        return self.state.get_register(idx)

    @property
    def registers(self) -> dict[int, int]:
        return self.state.dump_registers()

    def __repr__(self):
        status = "HALTED" if self.halted else "RUNNING"
        return f"TransformerExecutionResult({status}, cycles={self.cycles})"


def _extract_pc(x: torch.Tensor, config: StateConfig) -> int:
    """Extract PC value from state tensor."""
    pc_col = config.pc_col
    pc_bits = x[0, pc_col, VALUE_START:VALUE_END]
    return bipolar_to_int(pc_bits.cpu())


def _check_halt(x: torch.Tensor, pc: int, config: StateConfig) -> bool:
    """Check if the instruction at PC is HALT."""
    instr_col = config.instr_start + pc
    if instr_col >= config.n_columns:
        return True  # out of bounds = halt
    instr_bits = x[0, instr_col, VALUE_START:VALUE_END]
    packed = bipolar_to_int(instr_bits.cpu())
    opcode = packed & 0x1F
    return opcode == int(Opcode.HALT)
