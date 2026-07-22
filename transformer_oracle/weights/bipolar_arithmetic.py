"""
Exact bipolar arithmetic using ReLU threshold networks.

All values are in bipolar encoding: {-1, +1} where +1 = binary 1, -1 = binary 0.

This module constructs MLP weight matrices that implement exact discrete
operations (ADD, SUB, AND, OR, XOR, NOT, shifts) on bipolar-encoded values.

The key insight (Loom, Proposition 1): any Boolean function on bipolar inputs
can be implemented exactly using ReLU activations with integer thresholds.

For arithmetic: we decompose 32-bit addition into byte-level operations where
float64 arithmetic is exact (float64 has 53-bit mantissa, more than enough for
8+8+1 = 17-bit intermediate results).

Weight matrices are returned as numpy arrays or torch tensors for direct
insertion into the transformer's MLP layers.
"""

import torch
import numpy as np
from typing import Tuple


# --- Bipolar Boolean Gates ---
# For bipolar inputs a, b in {-1, +1}:
#   binary(a) = (a + 1) / 2  →  {0, 1}
#   bipolar(x) = 2*x - 1     →  {-1, +1}

def bipolar_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact bipolar AND: returns +1 iff both inputs are +1.

    Implementation: ReLU(a + b - 1) gives:
      a=-1, b=-1: ReLU(-3) = 0
      a=-1, b=+1: ReLU(-1) = 0
      a=+1, b=-1: ReLU(-1) = 0
      a=+1, b=+1: ReLU(+1) = 1

    Then map {0, 1} back to bipolar: 2*result - 1
    """
    return 2.0 * torch.relu(a + b - 1.0) - 1.0


def bipolar_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact bipolar OR: returns +1 if either input is +1.

    Implementation: ReLU(a + b + 1) - ReLU(a + b - 1) gives:
      a=-1, b=-1: ReLU(-1) - ReLU(-3) = 0 - 0 = 0
      a=-1, b=+1: ReLU(+1) - ReLU(-1) = 1 - 0 = 1
      a=+1, b=-1: ReLU(+1) - ReLU(-1) = 1 - 0 = 1
      a=+1, b=+1: ReLU(+3) - ReLU(+1) = 3 - 1 = 2

    Clamp to 1, then map to bipolar.

    Simpler: OR(a,b) = NOT(AND(NOT(a), NOT(b)))
    """
    return bipolar_not(bipolar_and(bipolar_not(a), bipolar_not(b)))


def bipolar_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact bipolar XOR: returns +1 iff exactly one input is +1.

    XOR = (a OR b) AND NOT(a AND b)
    Or equivalently: XOR(a,b) = -a*b in bipolar encoding.
    """
    return -a * b


def bipolar_not(a: torch.Tensor) -> torch.Tensor:
    """Exact bipolar NOT: flip the sign."""
    return -a


# --- AND/OR/XOR as MLP weight matrices ---
# These construct W1, b1, W2, b2 for a single-hidden-layer MLP
# that computes the gate: output = W2 @ ReLU(W1 @ input + b1) + b2

def and_gate_weights(n_bits: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct MLP weights for bitwise AND of two n-bit bipolar vectors.

    Input: [a_0, ..., a_{n-1}, b_0, ..., b_{n-1}]  (2*n_bits)
    Output: [and_0, ..., and_{n-1}]  (n_bits)

    For each bit i, AND(a_i, b_i) uses 1 hidden neuron:
      h_i = ReLU(a_i + b_i - 1)  →  0 or 1
      out_i = 2 * h_i - 1        →  -1 or +1
    """
    in_dim = 2 * n_bits
    hidden_dim = n_bits

    W1 = np.zeros((hidden_dim, in_dim), dtype=np.float64)
    b1 = np.full(hidden_dim, -1.0, dtype=np.float64)
    W2 = np.zeros((n_bits, hidden_dim), dtype=np.float64)
    b2 = np.full(n_bits, -1.0, dtype=np.float64)

    for i in range(n_bits):
        W1[i, i] = 1.0          # a_i
        W1[i, n_bits + i] = 1.0  # b_i
        W2[i, i] = 2.0

    return W1, b1, W2, b2


def xor_gate_weights(n_bits: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct MLP weights for bitwise XOR.

    XOR(a,b) in bipolar = -a*b, but MLP with ReLU can't directly multiply.
    Use: XOR = OR AND NOT(AND) = (NOT(NOT_a AND NOT_b)) AND NOT(a AND b)

    Simpler 2-hidden-neuron approach per bit:
      h1_i = ReLU(a_i + b_i - 1)   →  AND(a_i, b_i)  in {0, 1}
      h2_i = ReLU(-a_i - b_i - 1)  →  NOR(a_i, b_i)  in {0, 1}
      out_i = -2*(h1_i + h2_i) + 1  →  XOR(a_i, b_i) in {-1, +1}

    Truth table verification:
      a=-1, b=-1: h1=ReLU(-3)=0, h2=ReLU(-1)=0, out=-2*0+1=+1  ✗ (should be -1)

    Hmm, let me reconsider. In bipolar:
      -1 XOR -1 = -1 (0 XOR 0 = 0)
      -1 XOR +1 = +1 (0 XOR 1 = 1)
      +1 XOR -1 = +1 (1 XOR 0 = 1)
      +1 XOR +1 = -1 (1 XOR 1 = 0)

    So XOR(a,b) = -a*b. Using ReLU:
      h1_i = ReLU(a_i - b_i)   → max(a-b, 0)
      h2_i = ReLU(-a_i + b_i)  → max(b-a, 0)
      For a=+1,b=-1: h1=2, h2=0
      For a=-1,b=+1: h1=0, h2=2
      For a=+1,b=+1: h1=0, h2=0
      For a=-1,b=-1: h1=0, h2=0
      out_i = h1_i + h2_i - 1  →  gives 1, 1, -1, -1 ✓
    """
    in_dim = 2 * n_bits
    hidden_dim = 2 * n_bits  # 2 neurons per bit

    W1 = np.zeros((hidden_dim, in_dim), dtype=np.float64)
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    W2 = np.zeros((n_bits, hidden_dim), dtype=np.float64)
    b2 = np.full(n_bits, -1.0, dtype=np.float64)

    for i in range(n_bits):
        # h1 = ReLU(a_i - b_i)
        W1[2*i, i] = 1.0
        W1[2*i, n_bits + i] = -1.0
        # h2 = ReLU(-a_i + b_i)
        W1[2*i+1, i] = -1.0
        W1[2*i+1, n_bits + i] = 1.0
        # out = h1 + h2 - 1
        W2[i, 2*i] = 1.0
        W2[i, 2*i+1] = 1.0

    return W1, b1, W2, b2


# --- Bipolar Full Adder ---

def bipolar_half_add(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Half adder on bipolar bits. Returns (sum_bit, carry_bit) both bipolar.

    sum = XOR(a, b) = -a * b
    carry = AND(a, b) = 2*ReLU(a + b - 1) - 1
    """
    s = -a * b
    c = 2.0 * torch.relu(a + b - 1.0) - 1.0
    return s, c


def bipolar_full_add(a: torch.Tensor, b: torch.Tensor,
                     cin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full adder on bipolar bits. Returns (sum_bit, carry_out) both bipolar.

    Using two half-adders:
      s1, c1 = half_add(a, b)
      sum, c2 = half_add(s1, cin)
      carry = OR(c1, c2)
    """
    s1, c1 = bipolar_half_add(a, b)
    s, c2 = bipolar_half_add(s1, cin)
    cout = bipolar_or(c1, c2)
    return s, cout


def bipolar_add_32bit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add two 32-bit bipolar-encoded integers using ripple-carry.

    Args:
        a: bipolar tensor of shape (32,) representing first operand
        b: bipolar tensor of shape (32,) representing second operand

    Returns:
        result: bipolar tensor of shape (32,) representing a + b (mod 2^32)
    """
    assert a.shape == (32,) and b.shape == (32,)
    result = torch.zeros(32, dtype=a.dtype)
    carry = torch.tensor(-1.0, dtype=a.dtype)  # -1 = bipolar 0 = no carry

    for i in range(32):
        result[i], carry = bipolar_full_add(a[i], b[i], carry)

    return result


def bipolar_negate_32bit(a: torch.Tensor) -> torch.Tensor:
    """Two's complement negation: -a = NOT(a) + 1.

    In bipolar: NOT is just sign flip, then add 1.
    """
    not_a = bipolar_not(a)
    one = torch.full((32,), -1.0, dtype=a.dtype)
    one[0] = 1.0  # bipolar encoding of integer 1
    return bipolar_add_32bit(not_a, one)


def bipolar_sub_32bit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Subtract: a - b = a + (-b)."""
    return bipolar_add_32bit(a, bipolar_negate_32bit(b))


def bipolar_and_32bit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bitwise AND of two 32-bit bipolar values."""
    return bipolar_and(a, b)


def bipolar_or_32bit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bitwise OR of two 32-bit bipolar values."""
    return bipolar_or(a, b)


def bipolar_xor_32bit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bitwise XOR of two 32-bit bipolar values."""
    return bipolar_xor(a, b)


def bipolar_not_32bit(a: torch.Tensor) -> torch.Tensor:
    """Bitwise NOT of a 32-bit bipolar value."""
    return bipolar_not(a)


def bipolar_shl_32bit(a: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical shift left by a constant amount.

    Shift bits up, fill low bits with -1 (binary 0).
    """
    shift = shift & 0x1F  # mod 32
    if shift == 0:
        return a.clone()
    result = torch.full((32,), -1.0, dtype=a.dtype)
    result[shift:] = a[:32-shift]
    return result


def bipolar_shr_32bit(a: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical shift right by a constant amount.

    Shift bits down, fill high bits with -1 (binary 0).
    """
    shift = shift & 0x1F
    if shift == 0:
        return a.clone()
    result = torch.full((32,), -1.0, dtype=a.dtype)
    result[:32-shift] = a[shift:]
    return result


def bipolar_sra_32bit(a: torch.Tensor, shift: int) -> torch.Tensor:
    """Arithmetic shift right by a constant amount.

    Shift bits down, fill high bits with the sign bit (MSB).
    """
    shift = shift & 0x1F
    if shift == 0:
        return a.clone()
    result = torch.full((32,), a[31].item(), dtype=a.dtype)  # fill with sign
    result[:32-shift] = a[shift:]
    return result


# --- Adder MLP Weight Construction ---
# Constructs the weight matrices for a 32-bit adder implemented as an MLP.

def build_adder_weights_bytewise() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct MLP weights for a 32-bit adder using byte-level decomposition.

    The adder processes 4 bytes of each operand. For each byte:
    - Convert 8 bipolar bits to an integer value (in float64, exact)
    - Add the two byte values + carry_in
    - Extract result byte and carry_out

    This is implemented as a 2-layer MLP:
    Layer 1 (hidden): Converts bipolar bits to byte values and computes sums
    Layer 2 (output): Extracts result bits and carry propagation

    Input layout: [a_0..a_31, b_0..b_31] = 64 bipolar values
    Output: [result_0..result_31] = 32 bipolar values

    For Phase 1, we use the direct ripple-carry approach in bipolar_add_32bit()
    and reserve this function for the optimized MLP path in Phase 2+.
    """
    # This is a placeholder for the optimized byte-level MLP weights.
    # Phase 1 uses the direct bipolar_add_32bit() function instead.
    raise NotImplementedError("Byte-level adder MLP weights will be implemented in Phase 2")


# --- Convenience: execute ALU operation on bipolar values ---

def execute_alu_op(opcode: int, a: torch.Tensor, b: torch.Tensor,
                   c_val: torch.Tensor = None) -> torch.Tensor:
    """Execute an ALU operation on bipolar-encoded 32-bit values.

    This is the reference implementation used for testing.
    The transformer will implement this via constructed MLP weights.

    Args:
        opcode: NISA opcode (as int)
        a: first operand (bipolar 32-bit)
        b: second operand (bipolar 32-bit), or immediate
        c_val: third operand if needed

    Returns:
        result: bipolar 32-bit result
    """
    from ..core.nisa import Opcode

    op = Opcode(opcode)
    if op == Opcode.ADD:
        return bipolar_add_32bit(a, b)
    elif op == Opcode.SUB:
        return bipolar_sub_32bit(a, b)
    elif op == Opcode.AND:
        return bipolar_and_32bit(a, b)
    elif op == Opcode.OR:
        return bipolar_or_32bit(a, b)
    elif op == Opcode.XOR:
        return bipolar_xor_32bit(a, b)
    elif op == Opcode.NOT:
        return bipolar_not_32bit(a)
    elif op == Opcode.SLT:
        # signed less than: result is 1 if a < b (signed), else 0
        from ..core.state import bipolar_to_int, int_to_bipolar
        va = bipolar_to_int(a, signed=True)
        vb = bipolar_to_int(b, signed=True)
        return int_to_bipolar(1 if va < vb else 0)
    elif op == Opcode.SLTU:
        from ..core.state import bipolar_to_int, int_to_bipolar
        va = bipolar_to_int(a, signed=False)
        vb = bipolar_to_int(b, signed=False)
        return int_to_bipolar(1 if va < vb else 0)
    else:
        raise ValueError(f"Unsupported ALU opcode: {op}")
