"""
Fast vectorized bipolar arithmetic for GPU execution.

Key insight: instead of ripple-carry addition bit-by-bit, convert
bipolar → integer, do arithmetic in integer domain (exact in float64
for 32-bit values since float64 has 53-bit mantissa), convert back.

  bipolar_to_int: dot([1, 2, 4, ...], (bits + 1) / 2)
  int_to_bipolar: 2 * ((int >> bit_positions) & 1) - 1

These conversions are linear operations (matrix multiplies), making
them fully parallelizable on GPU. All ALU operations become:
  1. Convert operands: O(32) parallel multiply-add
  2. Compute result: O(1) arithmetic
  3. Convert back: O(32) parallel bit extraction

This replaces the O(32) sequential ripple-carry in bipolar_arithmetic.py.
"""

import torch
from typing import Optional

# Pre-computed constants
_POW2 = None  # [1, 2, 4, 8, ..., 2^31]
_BITS = None  # [0, 1, 2, ..., 31]
_MOD32 = 2**32


def _ensure_constants(device: torch.device, dtype: torch.dtype = torch.float64):
    """Lazily initialize constants on the correct device."""
    global _POW2, _BITS
    if _POW2 is None or _POW2.device != device:
        _POW2 = torch.tensor([2**i for i in range(32)], dtype=dtype, device=device)
        _BITS = torch.arange(32, device=device)


def bp_to_int(bp: torch.Tensor) -> torch.Tensor:
    """Convert bipolar {-1, +1} tensor(s) to integer value(s).

    Args:
        bp: (..., 32) bipolar tensor

    Returns:
        (...,) integer tensor (as float64 for precision)
    """
    _ensure_constants(bp.device, bp.dtype)
    binary = (bp + 1.0) * 0.5  # {-1,+1} → {0,1}
    return (binary * _POW2).sum(dim=-1)


def int_to_bp(val: torch.Tensor, n_bits: int = 32) -> torch.Tensor:
    """Convert integer value(s) to bipolar {-1, +1} tensor(s).

    Args:
        val: (...,) integer tensor (as float64)
        n_bits: number of bits

    Returns:
        (..., n_bits) bipolar tensor
    """
    _ensure_constants(val.device, val.dtype)
    val_long = val.long()
    # Extract each bit: (val >> i) & 1
    bits = (val_long.unsqueeze(-1) >> _BITS[:n_bits]) & 1
    # Convert to bipolar: {0,1} → {-1,+1}
    return (2.0 * bits.to(val.dtype) - 1.0)


# ── Vectorized ALU operations ──
# All operate on bipolar tensors of shape (..., 32)
# and return bipolar tensors of shape (..., 32)

def fast_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """32-bit addition: (a + b) mod 2^32, fully vectorized."""
    a_int = bp_to_int(a)
    b_int = bp_to_int(b)
    result = (a_int + b_int) % _MOD32
    return int_to_bp(result)


def fast_sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """32-bit subtraction: (a - b) mod 2^32."""
    a_int = bp_to_int(a)
    b_int = bp_to_int(b)
    result = (a_int - b_int) % _MOD32
    return int_to_bp(result)


def fast_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """32-bit multiplication: (a * b) mod 2^32.

    The product of two 32-bit values can reach ~2^64, which exceeds float64's
    exact-integer range (2^53). Doing the multiply in float64 (as bp_to_int
    returns) silently rounds away the low bits before the mod, corrupting the
    result. Multiply in int64 instead: it wraps mod 2^64, and since 2^32 | 2^64
    the low 32 bits are preserved exactly.
    """
    a_int = bp_to_int(a).long()
    b_int = bp_to_int(b).long()
    result = (a_int * b_int) & 0xFFFFFFFF
    return int_to_bp(result.double())


def fast_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bitwise AND: exact in bipolar domain."""
    # AND(a,b) in bipolar: 2*ReLU(a+b-1) - 1
    return 2.0 * torch.relu(a + b - 1.0) - 1.0


def fast_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bitwise OR: via De Morgan = NOT(AND(NOT(a), NOT(b)))."""
    return -fast_and(-a, -b)


def fast_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bitwise XOR: in bipolar, XOR(a,b) = -a*b."""
    return -a * b


def fast_not(a: torch.Tensor) -> torch.Tensor:
    """Bitwise NOT: flip sign."""
    return -a


def fast_shl(a: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical shift left by constant amount."""
    shift = shift & 0x1F
    if shift == 0:
        return a.clone()
    result = torch.full_like(a, -1.0)
    result[..., shift:] = a[..., :32-shift]
    return result


def fast_shr(a: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical shift right by constant amount."""
    shift = shift & 0x1F
    if shift == 0:
        return a.clone()
    result = torch.full_like(a, -1.0)
    result[..., :32-shift] = a[..., shift:]
    return result


def fast_sra(a: torch.Tensor, shift: int) -> torch.Tensor:
    """Arithmetic shift right by constant (sign-extending)."""
    shift = shift & 0x1F
    if shift == 0:
        return a.clone()
    sign = a[..., 31:32]  # MSB
    result = sign.expand_as(a).clone()
    result[..., :32-shift] = a[..., shift:]
    return result


def fast_shl_var(a: torch.Tensor, shift_bp: torch.Tensor) -> torch.Tensor:
    """Shift left by variable amount (from register)."""
    shift = int(bp_to_int(shift_bp).item()) & 0x1F
    return fast_shl(a, shift)


def fast_shr_var(a: torch.Tensor, shift_bp: torch.Tensor) -> torch.Tensor:
    """Shift right by variable amount."""
    shift = int(bp_to_int(shift_bp).item()) & 0x1F
    return fast_shr(a, shift)


def fast_sra_var(a: torch.Tensor, shift_bp: torch.Tensor) -> torch.Tensor:
    """Arithmetic shift right by variable amount."""
    shift = int(bp_to_int(shift_bp).item()) & 0x1F
    return fast_sra(a, shift)


def fast_slt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Set less than (signed)."""
    a_int = bp_to_int(a)
    b_int = bp_to_int(b)
    # Convert to signed
    a_s = torch.where(a_int >= 2**31, a_int - _MOD32, a_int)
    b_s = torch.where(b_int >= 2**31, b_int - _MOD32, b_int)
    result = torch.where(a_s < b_s, torch.ones_like(a_int), torch.zeros_like(a_int))
    return int_to_bp(result)


def fast_sltu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Set less than (unsigned)."""
    a_int = bp_to_int(a)
    b_int = bp_to_int(b)
    result = torch.where(a_int < b_int, torch.ones_like(a_int), torch.zeros_like(a_int))
    return int_to_bp(result)


def snap_to_bipolar(x: torch.Tensor) -> torch.Tensor:
    """Error correction: snap values to exact {-1, +1}."""
    return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))
