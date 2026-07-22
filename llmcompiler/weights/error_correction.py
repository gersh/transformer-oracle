"""
Layer 10: Error Correction.

After each forward pass, floating-point arithmetic may introduce small
errors that drift bipolar values away from exact {-1, +1}. This layer
snaps all value bits back to their nearest bipolar value.

In a real transformer implementation, this is an MLP layer with weights:
  output = sign(input) * 1.0
Approximated as: output = 2 * ReLU(input) - 2 * ReLU(-input) + epsilon

For the Phase 1 reference executor, we directly clamp values.
"""

import torch
from ..core.state import StateTensor, VALUE_START, VALUE_END, int_to_bipolar


def error_correct(state: StateTensor):
    """Snap all value bits back to exact bipolar {-1, +1}.

    This is the L10 error correction step. Must be applied after
    every forward pass to prevent error accumulation.
    """
    state.snap_to_bipolar()


def build_error_correction_weights(d_state: int) -> dict:
    """Construct MLP weights for error correction layer.

    The error correction MLP implements:
      f(x) = +1 if x > 0, -1 if x <= 0

    Using ReLU: f(x) = 2*ReLU(x) / (|x| + eps) - 1
    But since we only need to snap values that are already close to {-1, +1},
    a simpler approach works:
      h1 = ReLU(x)      → positive part
      h2 = ReLU(-x)     → negative part
      out = 2 * sign(h1 - h2 + eps) - 1

    For an MLP approximation:
      h = ReLU(scale * x)  where scale is large (e.g., 100)
      out = clamp(2 * h / (h + eps) - 1, -1, 1)

    For Phase 1, we use direct snapping in error_correct().
    This function returns placeholder weights for future transformer integration.
    """
    import numpy as np

    # Large scale to make ReLU act as a hard threshold
    scale = 100.0

    # W1: identity scaled up (for value bits only)
    n_value = VALUE_END - VALUE_START  # 32
    W1 = scale * np.eye(n_value, dtype=np.float64)
    b1 = np.zeros(n_value, dtype=np.float64)

    # W2: maps ReLU output back to bipolar
    # ReLU(scale * x) for x in {-1, +1}: gives 0 or scale
    # We want: 2 * (ReLU(scale*x) / scale) - 1 = 2*(x>0) - 1 = sign(x)
    W2 = (2.0 / scale) * np.eye(n_value, dtype=np.float64)
    b2 = -1.0 * np.ones(n_value, dtype=np.float64)

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
