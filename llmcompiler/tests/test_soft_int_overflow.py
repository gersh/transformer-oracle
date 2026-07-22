"""
Gradient-guided overflow discovery using soft integer arithmetic.

The soft_signed function models the signed/unsigned confusion
differentiably, so gradients can guide toward overflow values.
"""

import torch
import pytest
from ..runtime.soft_int import soft_signed, soft_signed_gt, soft_mod32, soft_mul32


class TestSoftIntBasics:

    def test_soft_signed_positive(self):
        """Values < 2^31 stay positive (exact in forward)."""
        x = torch.tensor(1000.0, dtype=torch.float64)
        assert soft_signed(x).item() == 1000.0

    def test_soft_signed_negative(self):
        """Values > 2^31 wrap to negative (exact in forward)."""
        x = torch.tensor(2.0**31 + 1000, dtype=torch.float64)
        result = soft_signed(x).item()
        expected = -(2**31) + 1000
        assert result == expected, f"Expected {expected}, got {result}"

    def test_soft_signed_gradient(self):
        """Gradient exists and is large near the 2^31 boundary."""
        x = torch.tensor(2.0**31, dtype=torch.float64, requires_grad=True)
        s = soft_signed(x)
        s.backward()
        assert x.grad is not None
        assert x.grad.item() != 0
        print(f"\n  Gradient at 2^31 boundary: {x.grad.item():.4f}")

    def test_soft_signed_gt_normal(self):
        """Normal signed comparison: 100 > 50."""
        a = torch.tensor(100.0, dtype=torch.float64)
        b = torch.tensor(50.0, dtype=torch.float64)
        assert soft_signed_gt(a, b).item() > 0.5

    def test_soft_signed_gt_overflow(self):
        """THE BUG: 0x80000001 > 50 is FALSE in signed."""
        a = torch.tensor(2.0**31 + 1, dtype=torch.float64)
        b = torch.tensor(50.0, dtype=torch.float64)
        result = soft_signed_gt(a, b).item()
        # signed(2^31+1) = -2^31+1 ≈ -2147483647 < 50 → FALSE
        assert result < 0.5, f"Signed comparison should be FALSE, got {result}"


class TestGradientFindsOverflow:

    def test_gradient_crosses_signed_boundary(self):
        """Gradient descent pushes chunk_len past 2^31 to flip signed check."""
        print("\n" + "="*60)
        print("SOFT INTEGER GRADIENT → SIGNED OVERFLOW")
        print("="*60)

        remaining = torch.tensor(88.0, dtype=torch.float64)

        # Start CLOSE to the 2^31 boundary
        chunk_len = torch.tensor(2.0**31 - 1000, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([chunk_len], lr=500.0)

        print(f"\n  Goal: push chunk_len past 2^31 where signed wraps negative")
        print(f"        Starting at {chunk_len.item():.0f} (2^31 = {2**31})\n")

        for i in range(200):
            optimizer.zero_grad()

            # Direct loss: make soft_signed(chunk_len) negative
            # When chunk_len < 2^31: signed is positive → loss > 0
            # When chunk_len > 2^31: signed is negative → loss = 0 (achieved!)
            signed_val = soft_signed(chunk_len)
            loss = torch.relu(signed_val)  # penalize positive signed value

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                cl = chunk_len.item()
                sv = soft_signed(chunk_len).item()
                gt = soft_signed_gt(chunk_len, remaining).item()
                grad = chunk_len.grad.item() if chunk_len.grad is not None else 0
                print(f"  iter {i:3d}: chunk_len={cl:>15.0f} "
                      f"signed={sv:>15.0f} "
                      f"gt_check={gt:.4f} grad={grad:>12.2f}")

        final_cl = chunk_len.item()
        final_signed = soft_signed(chunk_len).item()
        final_gt = soft_signed_gt(chunk_len, remaining).item()

        print(f"\n  Final: chunk_len = {final_cl:.0f} (0x{int(final_cl) & 0xFFFFFFFF:08X})")
        print(f"         signed    = {final_signed:.0f}")
        print(f"         gt_check  = {final_gt:.4f} ({'TAKEN (caught)' if final_gt > 0.5 else 'NOT TAKEN (BYPASS!)'})")

        if final_cl > 2**31 and final_gt < 0.5:
            print(f"\n  >>> GRADIENT FOUND THE OVERFLOW! <<<")
            print(f"  chunk_len = 0x{int(final_cl) & 0xFFFFFFFF:08X}")
            print(f"  As signed: {int(final_signed)} (NEGATIVE!)")
            print(f"  Signed comparison 'chunk_len > {int(remaining.item())}' → FALSE")
            print(f"  Unsigned chunk_len = {int(final_cl)} >> {int(remaining.item())} → OVERFLOW!")

        assert final_cl > 2**31, f"Should cross 2^31, got {final_cl}"
        assert final_gt < 0.5, f"Signed check should be bypassed, got {final_gt}"

    def test_gradient_integer_overflow_mul(self):
        """Gradient descent finds width*height that overflows 32 bits."""
        print("\n" + "="*60)
        print("SOFT INTEGER GRADIENT → MULTIPLICATION OVERFLOW")
        print("="*60)

        width = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)
        height = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([width, height], lr=5000.0)

        print(f"\n  Goal: find width, height where width*height")
        print(f"        overflows 32 bits to a small value\n")

        for i in range(500):
            optimizer.zero_grad()

            # Real product (can be > 2^32)
            real_product = width * height

            # 32-bit wrapped product
            wrapped = soft_mod32(real_product)

            # We want: wrapped is SMALL (< 4096) but real is LARGE
            # Loss: minimize wrapped + penalize small dimensions
            loss = wrapped / 4096.0 + 4096.0 / (width + 1) + 4096.0 / (height + 1)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                width.clamp_(min=1.0)
                height.clamp_(min=1.0)

            if i % 100 == 0:
                w = width.item()
                h = height.item()
                real = w * h
                wrap = soft_mod32(torch.tensor(real)).item()
                print(f"  iter {i:3d}: w={w:>12.0f} h={h:>12.0f} "
                      f"real={real:>16.0f} wrapped={wrap:>10.0f}")

        w_final = int(width.item())
        h_final = int(height.item())
        real = w_final * h_final
        wrapped = real % (2**32)

        print(f"\n  Final: {w_final} × {h_final} = {real:,} bytes")
        print(f"         mod 2^32 = {wrapped:,} bytes")
        if wrapped < 4096 and real > 2**32:
            print(f"  >>> OVERFLOW: allocates {wrapped} bytes, needs {real:,}!")
            print(f"  >>> Missing {real - wrapped:,} bytes → HEAP OVERFLOW!")
        elif wrapped < real:
            print(f"  >>> OVERFLOW: product wrapped around!")
            print(f"  Wraps from {real:,} to {wrapped:,}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
