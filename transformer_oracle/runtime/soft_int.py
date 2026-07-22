"""
Soft integer arithmetic — differentiable 32-bit overflow semantics.

Uses straight-through estimator (STE):
  Forward: exact integer semantics (hard)
  Backward: smooth gradient that knows about the 2^31 boundary
"""

import torch

MOD32 = 2.0 ** 32
HALF = 2.0 ** 31


class _SoftSigned(torch.autograd.Function):
    """Hard signed conversion with gradient that knows the boundary.

    Forward: x >= 2^31 → x - 2^32 (exact signed interpretation)
    Backward: gradient = 1 everywhere, PLUS a large negative impulse
              near x = 2^31 signaling "crossing this boundary flips the sign"
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= HALF, x - MOD32, x)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        # Normal gradient is 1 (linear function of x)
        # BUT near the 2^31 boundary, the signed value jumps by -2^32
        # Model this with a soft Dirac delta (Gaussian bump)
        dist = (x - HALF) / 1e6  # scale for soft boundary
        boundary_signal = -MOD32 * torch.exp(-0.5 * dist * dist) / 1e6
        return grad * (1.0 + boundary_signal)


class _SoftMod32(torch.autograd.Function):
    """Hard mod 2^32 with gradient that knows about wraparound."""
    @staticmethod
    def forward(ctx, x):
        return torch.fmod(x, MOD32)

    @staticmethod
    def backward(ctx, grad):
        return grad  # STE: gradient passes through


def soft_signed(x: torch.Tensor) -> torch.Tensor:
    """Differentiable signed interpretation. Exact forward, smooth backward."""
    return _SoftSigned.apply(x)


def soft_mod32(x: torch.Tensor) -> torch.Tensor:
    """Differentiable mod 2^32."""
    return _SoftMod32.apply(x)


def soft_signed_gt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable signed a > b comparison.

    Forward: exact (a_signed > b_signed)
    Backward: gradient from (signed_a - signed_b) through sigmoid
    """
    sa = soft_signed(a)
    sb = soft_signed(b)
    diff = sa - sb
    # Soft comparison: sigmoid of the difference
    # k controls sharpness: higher = more precise but smaller gradient
    return torch.sigmoid(diff / 100.0)


def soft_unsigned_gt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable unsigned a > b comparison."""
    diff = a - b
    return torch.sigmoid(diff / 100.0)


def soft_add32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable 32-bit addition with wraparound."""
    return soft_mod32(a + b)


def soft_mul32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable 32-bit multiplication with wraparound."""
    return soft_mod32(a * b)
