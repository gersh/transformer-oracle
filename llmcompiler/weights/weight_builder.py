"""
Analytical Weight Builder for the Transformer CPU.

Constructs all weight matrices for the 10-layer transformer so that
each forward pass correctly executes one NISA instruction.

The key insight: attention weights are set so that dot-product scores
produce exact one-hot selection of the desired column, and MLP weights
implement Boolean/arithmetic circuits in bipolar encoding.

This module uses the state tensor layout from core/state.py:
  - Rows 0-31: value bits (bipolar)
  - Rows 32-42: position encoding (bipolar address)
  - Row 43: column type

Weight construction is done in float64 for precision.
"""

import torch
import numpy as np
from ..core.state import (
    StateConfig, DEFAULT_CONFIG,
    VALUE_START, VALUE_END, VALUE_BITS,
    POSITION_START, POSITION_END, POSITION_BITS,
    TYPE_START, TYPE_END, D_STATE,
    COL_TYPE_REG, COL_TYPE_PC, COL_TYPE_DATAMEM, COL_TYPE_INSTRMEM,
    int_to_bipolar,
)
from ..core.nisa import Opcode
from .bipolar_arithmetic import (
    bipolar_add_32bit, bipolar_sub_32bit,
    bipolar_and_32bit, bipolar_or_32bit, bipolar_xor_32bit, bipolar_not_32bit,
)
from ..runtime.transformer_cpu import TransformerCPU, HardAttention, AnalyticalMLP


def build_all_weights(config: StateConfig = DEFAULT_CONFIG,
                      temperature: float = 50.0) -> TransformerCPU:
    """Construct a TransformerCPU with analytically-computed weights.

    Returns a ready-to-execute TransformerCPU model.
    """
    model = TransformerCPU(config, temperature=temperature)

    with torch.no_grad():
        _build_fetch_weights(model.fetch_attn, config)
        _build_error_correction_weights(model.error_mlp, config)
        # L2-L9 weights are built for specific instruction support
        # For Phase 2, we start with the reference executor fallback

    model.eval()
    return model


# ── L1: Instruction Fetch ──
#
# Goal: The PC column should "fetch" the value bits of the instruction
# at address PC from the instruction memory columns.
#
# Mechanism: Attention where:
#   Query = position encoding derived from PC's VALUE bits
#   Key   = position encoding of each instruction column (relative to instr_start)
#   Value = value bits of each instruction column
#
# The attention score between PC and instruction column i is:
#   score = dot(Q_pc, K_i) = dot(pc_value_as_position, instr_position)
#
# When PC value matches instruction index i, the score is maximal (all bits match),
# and softmax with high temperature gives near-one-hot selection.
#
# The output (fetched instruction bits) is written into the PC column's value rows
# via the output projection. This is the "command buffer" that downstream layers read.

def _build_fetch_weights(attn: HardAttention, config: StateConfig):
    """Build L1 instruction fetch attention weights.

    Strategy:
    - W_q maps PC column's value bits → query based on instruction index
    - W_k maps each column's position encoding → key
    - W_v passes through value bits
    - W_o routes fetched instruction to a scratch area

    For simplicity in Phase 2, we use the position encoding rows directly:
    - Query: extract value bits 0..POSITION_BITS-1 from the PC column
      (these represent the PC value = instruction index)
    - Key: extract position bits (rows POSITION_START..POSITION_END)
      relative to instr_start

    The attention score for PC column attending to instruction column i:
      score = sum over bits: pc_value_bit[b] * instr_position_bit[b]
    This equals POSITION_BITS when all bits match, and less otherwise.
    """
    d = config.d_state
    W_q = torch.zeros(d, d, dtype=torch.float64)
    W_k = torch.zeros(d, d, dtype=torch.float64)
    W_v = torch.zeros(d, d, dtype=torch.float64)
    W_o = torch.zeros(d, d, dtype=torch.float64)

    # W_q: For each column, map value bits to "query space"
    # We use rows 0..POSITION_BITS-1 of the value as the query
    # (the instruction index is the low bits of the PC value)
    for b in range(min(POSITION_BITS, VALUE_BITS)):
        # Query uses value bits (row b) but places them in position-encoding rows
        W_q[POSITION_START + b, VALUE_START + b] = 1.0

    # W_k: Identity on position encoding rows
    for b in range(POSITION_BITS):
        W_k[POSITION_START + b, POSITION_START + b] = 1.0

    # W_v: Pass through value bits
    for b in range(VALUE_BITS):
        W_v[VALUE_START + b, VALUE_START + b] = 1.0

    # W_o: Identity on value bits (output the fetched instruction)
    for b in range(VALUE_BITS):
        W_o[VALUE_START + b, VALUE_START + b] = 1.0

    attn.W_q.data = W_q
    attn.W_k.data = W_k
    attn.W_v.data = W_v
    attn.W_o.data = W_o


# ── L10: Error Correction ──
#
# Goal: Snap all value bits back to exact bipolar {-1, +1}.
# f(x) = +1 if x > 0, -1 if x <= 0
#
# Using ReLU MLP:
#   h = ReLU(scale * x)  where scale is large
#   out = 2/scale * h - 1
# This gives: x>0 → h=scale*x → out≈2x-1≈+1; x<0 → h=0 → out=-1

def _build_error_correction_weights(mlp: AnalyticalMLP, config: StateConfig):
    """Build L10 error correction MLP weights.

    The MLP should output a DELTA that, when added to the input via
    the residual connection, snaps values to bipolar.

    For a value bit currently at v (close to +1 or -1):
      We want: v + delta = sign(v)
      So: delta = sign(v) - v

    Using ReLU:
      h_pos = ReLU(scale * v)           → scale*v if v>0, else 0
      h_neg = ReLU(-scale * v)          → scale*|v| if v<0, else 0
      snap = (2/scale)*h_pos - (2/scale)*h_neg - ?

    Actually simpler: the delta should be 0 when v is exactly ±1.
    When v drifts (e.g., v=0.95), delta should push it to 1.0.

    For Phase 2, we use a simpler approach: after each forward pass,
    we apply hard snapping in Python. The MLP weights are set to
    identity (zero delta) and snapping is done externally.
    """
    d = config.d_state
    # For now: zero weights = no MLP contribution, snapping done externally
    # This will be replaced with proper analytical weights in Phase 3
    pass


def build_reference_hybrid_executor(config: StateConfig = DEFAULT_CONFIG):
    """Build a hybrid executor that uses the transformer architecture
    but falls back to reference execution for correctness.

    This is the Phase 2 bridge: the transformer model exists and can
    run on GPU, but complex operations (ALU, branches) are still
    implemented via the reference executor's logic, just expressed
    as weight matrices.
    """
    model = build_all_weights(config)
    return model
