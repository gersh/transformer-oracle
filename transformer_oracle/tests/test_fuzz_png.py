"""
Differentiable fuzzing of a PNG header validator.

PNG file format (first 33 bytes):
  Bytes 0-7:   Magic number: 89 50 4E 47 0D 0A 1A 0A
  Bytes 8-11:  First chunk length (big-endian, should be 13 for IHDR)
  Bytes 12-15: Chunk type ("IHDR" = 49 48 44 52)
  Bytes 16-19: Width (big-endian, must be > 0)
  Bytes 20-23: Height (big-endian, must be > 0)
  Byte 24:     Bit depth (1, 2, 4, 8, or 16)
  Byte 25:     Color type (0, 2, 3, 4, or 6)
  Byte 26:     Compression method (must be 0)
  Byte 27:     Filter method (must be 0)
  Byte 28:     Interlace method (0 or 1)
  Bytes 29-32: CRC32

We pass the header as 9 x 32-bit words (w0-w8) and the validator
returns a "score" counting how many checks pass.

The fuzzer uses gradients to discover that w0 must be 0x89504E47,
w1 must be 0x0D0A1A0A, etc.
"""

import torch
import pytest
from ..compiler.python_compiler import compile_python
from ..runtime.differentiable_executor import execute_differentiable, fuzz
from ..runtime.gpu_executor import gpu_execute


# ── PNG header validator (runs on our transformer CPU) ──

def png_validate(w0, w1, w2, w3, w4, w5, w6):
    """Validate a PNG header. Returns number of checks passed (0-7).

    Args (as 32-bit words, big-endian byte packing):
        w0: magic bytes 0-3  (should be 0x89504E47)
        w1: magic bytes 4-7  (should be 0x0D0A1A0A)
        w2: chunk length     (should be 0x0000000D = 13)
        w3: chunk type       (should be 0x49484452 = "IHDR")
        w4: image width      (must be > 0)
        w5: image height     (must be > 0)
        w6: packed byte field: bit_depth(8) | color_type(8) | compression(8) | filter(8)
            compression and filter must be 0, so low 16 bits must be 0
    """
    score = 0

    # Check 1: PNG magic first word
    if w0 == 0x89504E47:
        score += 1

    # Check 2: PNG magic second word
    if w1 == 0x0D0A1A0A:
        score += 1

    # Check 3: IHDR chunk length = 13
    if w2 == 13:
        score += 1

    # Check 4: Chunk type = "IHDR"
    if w3 == 0x49484452:
        score += 1

    # Check 5: Width > 0
    if w4 > 0:
        score += 1

    # Check 6: Height > 0
    if w5 > 0:
        score += 1

    # Check 7: Compression and filter methods = 0
    # Low 16 bits of w6 should be 0
    low16 = w6 & 0xFFFF
    if low16 == 0:
        score += 1

    return score


# Valid PNG header words for reference
VALID_PNG = {
    1: 0x89504E47,  # magic 0
    2: 0x0D0A1A0A,  # magic 1
    3: 13,           # chunk length
    4: 0x49484452,  # "IHDR"
    5: 100,          # width
    6: 100,          # height
    7: 0x08020000,  # bit_depth=8, color_type=2 (RGB), compress=0, filter=0
}


class TestPNGValidator:
    """Verify the PNG validator works correctly."""

    def test_valid_png(self):
        """Valid PNG header should score 7/7."""
        nisa = compile_python(png_validate)
        result = gpu_execute(nisa, initial_registers=VALID_PNG, device='cuda')
        assert result.reg(10) == 7, f"Valid PNG scored {result.reg(10)}/7"

    def test_all_zeros(self):
        """All-zero input should fail most checks."""
        nisa = compile_python(png_validate)
        zeros = {i: 0 for i in range(1, 8)}
        result = gpu_execute(nisa, initial_registers=zeros, device='cuda')
        # Only "compression/filter=0" check passes (low16 of 0 is 0)
        assert result.reg(10) <= 2, f"All zeros scored {result.reg(10)}/7"

    def test_only_magic(self):
        """Correct magic bytes but wrong rest should score 2."""
        nisa = compile_python(png_validate)
        regs = {i: 0 for i in range(1, 8)}
        regs[1] = 0x89504E47
        regs[2] = 0x0D0A1A0A
        result = gpu_execute(nisa, initial_registers=regs, device='cuda')
        # magic1 + magic2 + compress/filter=0 + chunk_len=0?
        assert result.reg(10) >= 2


class TestPNGFuzzing:
    """Fuzz the PNG validator to discover valid header values."""

    def test_fuzz_discovers_branches(self):
        """Fuzzer should discover multiple validation branches."""
        nisa = compile_python(png_validate)

        result = fuzz(
            nisa,
            n_input_regs=7,
            n_iterations=300,
            lr=50.0,
            verbose=True,
            seed=42,
        )

        print(f"\nBest coverage: {len(result['best_coverage'])} branches")
        print(f"Best inputs:")
        for reg, val in sorted(result['best_inputs'].items()):
            print(f"  w{reg-1} (r{reg}) = 0x{val:08X} ({val})")

        # Should cover at least several branch directions
        assert len(result['best_coverage']) >= 3

    def test_gradient_toward_magic(self):
        """Gradients should point toward the correct PNG magic bytes."""
        nisa = compile_python(png_validate)

        # Start with w0 close to the magic value
        w0 = torch.tensor(0x89504E40, dtype=torch.float64, requires_grad=True)  # off by 7
        inputs = {
            1: w0,
            2: torch.tensor(0.0, dtype=torch.float64),
            3: torch.tensor(0.0, dtype=torch.float64),
            4: torch.tensor(0.0, dtype=torch.float64),
            5: torch.tensor(100.0, dtype=torch.float64),
            6: torch.tensor(100.0, dtype=torch.float64),
            7: torch.tensor(0.0, dtype=torch.float64),
        }

        result = execute_differentiable(nisa, inputs)

        # Find the branch that checks w0 == 0x89504E47
        magic_branches = [b for b in result.branch_events if b.pc < 10]
        assert len(magic_branches) > 0, "No branches found for magic check"

        # The distance should be non-zero (w0 is wrong)
        dist = magic_branches[0].distance
        assert abs(dist.item()) > 0

        # Gradient should exist and point toward fixing w0
        dist.backward()
        assert w0.grad is not None
        print(f"\nw0 = 0x{int(w0.item()):08X}")
        print(f"Target = 0x89504E47")
        print(f"Branch distance = {dist.item():.1f}")
        print(f"Gradient = {w0.grad.item():.6f}")

    def test_gradient_descent_finds_magic(self):
        """Use gradient descent to discover the PNG magic number."""
        nisa = compile_python(png_validate)

        # Start from random value, optimize toward correct magic
        w0 = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([w0], lr=100.0)

        target = 0x89504E47  # correct magic
        best_distance = float('inf')

        for i in range(500):
            optimizer.zero_grad()

            inputs = {
                1: w0,
                2: torch.tensor(0.0, dtype=torch.float64),
                3: torch.tensor(0.0, dtype=torch.float64),
                4: torch.tensor(0.0, dtype=torch.float64),
                5: torch.tensor(1.0, dtype=torch.float64),
                6: torch.tensor(1.0, dtype=torch.float64),
                7: torch.tensor(0.0, dtype=torch.float64),
            }

            result = execute_differentiable(nisa, inputs)

            if result.branch_distances:
                # Minimize distance to flipping the first branch (magic check)
                loss = result.branch_distances[0] ** 2
                loss.backward()
                optimizer.step()

                dist = abs(w0.detach().item() - target)
                if dist < best_distance:
                    best_distance = dist

                if i % 100 == 0:
                    print(f"  iter {i}: w0={int(w0.item()):>12d} "
                          f"(target={target}, dist={dist:.0f})")

        final_val = int(w0.detach().item())
        print(f"\nFinal w0: {final_val} (0x{final_val & 0xFFFFFFFF:08X})")
        print(f"Target:   {target} (0x{target:08X})")
        print(f"Distance: {abs(final_val - target)}")

        # Should get significantly closer to the target
        assert abs(final_val - target) < abs(1000 - target), \
            f"Gradient descent didn't move toward target: {final_val} vs {target}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
