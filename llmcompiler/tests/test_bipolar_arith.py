"""
Tests for bipolar arithmetic operations.

Exhaustively tests 8-bit operations, spot-checks 32-bit.
"""

import torch
import pytest
from ..core.state import int_to_bipolar, bipolar_to_int
from ..weights.bipolar_arithmetic import (
    bipolar_and, bipolar_or, bipolar_xor, bipolar_not,
    bipolar_add_32bit, bipolar_sub_32bit, bipolar_negate_32bit,
    bipolar_and_32bit, bipolar_or_32bit, bipolar_xor_32bit, bipolar_not_32bit,
    bipolar_shl_32bit, bipolar_shr_32bit, bipolar_sra_32bit,
    bipolar_half_add, bipolar_full_add,
)


class TestBipolarEncoding:
    def test_encode_decode_zero(self):
        bp = int_to_bipolar(0)
        assert bipolar_to_int(bp) == 0

    def test_encode_decode_one(self):
        bp = int_to_bipolar(1)
        assert bipolar_to_int(bp) == 1

    def test_encode_decode_max(self):
        bp = int_to_bipolar(0xFFFFFFFF)
        assert bipolar_to_int(bp) == 0xFFFFFFFF

    def test_encode_decode_roundtrip(self):
        for v in [0, 1, 2, 127, 255, 256, 1000, 0x7FFFFFFF, 0xFFFFFFFF]:
            bp = int_to_bipolar(v)
            assert bipolar_to_int(bp) == v, f"Failed for {v}"

    def test_signed_decode(self):
        bp = int_to_bipolar(0xFFFFFFFF)  # -1 in signed 32-bit
        assert bipolar_to_int(bp, signed=True) == -1

        bp = int_to_bipolar(0x80000000)  # -2^31 in signed 32-bit
        assert bipolar_to_int(bp, signed=True) == -(1 << 31)

    def test_bipolar_values_are_correct(self):
        bp = int_to_bipolar(5)  # binary: 101
        assert bp[0].item() == 1.0   # bit 0 = 1
        assert bp[1].item() == -1.0  # bit 1 = 0
        assert bp[2].item() == 1.0   # bit 2 = 1
        for i in range(3, 32):
            assert bp[i].item() == -1.0  # rest are 0


class TestBipolarGates:
    def test_and_truth_table(self):
        for a, b, expected in [(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, 1)]:
            at = torch.tensor(a, dtype=torch.float64)
            bt = torch.tensor(b, dtype=torch.float64)
            result = bipolar_and(at, bt)
            assert result.item() == expected, f"AND({a},{b}) = {result.item()}, expected {expected}"

    def test_or_truth_table(self):
        for a, b, expected in [(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)]:
            at = torch.tensor(a, dtype=torch.float64)
            bt = torch.tensor(b, dtype=torch.float64)
            result = bipolar_or(at, bt)
            assert result.item() == expected, f"OR({a},{b}) = {result.item()}, expected {expected}"

    def test_xor_truth_table(self):
        for a, b, expected in [(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]:
            at = torch.tensor(a, dtype=torch.float64)
            bt = torch.tensor(b, dtype=torch.float64)
            result = bipolar_xor(at, bt)
            assert result.item() == expected, f"XOR({a},{b}) = {result.item()}, expected {expected}"

    def test_not(self):
        assert bipolar_not(torch.tensor(1.0)).item() == -1.0
        assert bipolar_not(torch.tensor(-1.0)).item() == 1.0


class TestHalfAdder:
    def test_all_cases(self):
        cases = [
            (-1, -1, -1, -1),  # 0+0 = sum=0, carry=0
            (-1,  1,  1, -1),  # 0+1 = sum=1, carry=0
            ( 1, -1,  1, -1),  # 1+0 = sum=1, carry=0
            ( 1,  1, -1,  1),  # 1+1 = sum=0, carry=1
        ]
        for a, b, exp_s, exp_c in cases:
            at = torch.tensor(a, dtype=torch.float64)
            bt = torch.tensor(b, dtype=torch.float64)
            s, c = bipolar_half_add(at, bt)
            assert s.item() == exp_s, f"half_add({a},{b}) sum={s.item()}, expected {exp_s}"
            assert c.item() == exp_c, f"half_add({a},{b}) carry={c.item()}, expected {exp_c}"


class TestFullAdder:
    def test_all_cases(self):
        # All 8 input combinations for (a, b, cin) → (sum, cout)
        cases = [
            (-1, -1, -1, -1, -1),  # 0+0+0 = 0, carry=0
            (-1, -1,  1,  1, -1),  # 0+0+1 = 1, carry=0
            (-1,  1, -1,  1, -1),  # 0+1+0 = 1, carry=0
            (-1,  1,  1, -1,  1),  # 0+1+1 = 0, carry=1
            ( 1, -1, -1,  1, -1),  # 1+0+0 = 1, carry=0
            ( 1, -1,  1, -1,  1),  # 1+0+1 = 0, carry=1
            ( 1,  1, -1, -1,  1),  # 1+1+0 = 0, carry=1
            ( 1,  1,  1,  1,  1),  # 1+1+1 = 1, carry=1
        ]
        for a, b, cin, exp_s, exp_c in cases:
            at = torch.tensor(a, dtype=torch.float64)
            bt = torch.tensor(b, dtype=torch.float64)
            ct = torch.tensor(cin, dtype=torch.float64)
            s, cout = bipolar_full_add(at, bt, ct)
            assert s.item() == exp_s, \
                f"full_add({a},{b},{cin}) sum={s.item()}, expected {exp_s}"
            assert cout.item() == exp_c, \
                f"full_add({a},{b},{cin}) carry={cout.item()}, expected {exp_c}"


class TestAdd32:
    def test_simple_add(self):
        a = int_to_bipolar(5)
        b = int_to_bipolar(3)
        result = bipolar_add_32bit(a, b)
        assert bipolar_to_int(result) == 8

    def test_add_zero(self):
        a = int_to_bipolar(42)
        b = int_to_bipolar(0)
        result = bipolar_add_32bit(a, b)
        assert bipolar_to_int(result) == 42

    def test_add_overflow(self):
        a = int_to_bipolar(0xFFFFFFFF)
        b = int_to_bipolar(1)
        result = bipolar_add_32bit(a, b)
        assert bipolar_to_int(result) == 0  # wraps around

    def test_add_large(self):
        a = int_to_bipolar(0x12345678)
        b = int_to_bipolar(0x9ABCDEF0)
        expected = (0x12345678 + 0x9ABCDEF0) & 0xFFFFFFFF
        result = bipolar_add_32bit(a, b)
        assert bipolar_to_int(result) == expected

    def test_exhaustive_8bit(self):
        """Exhaustively test all 8-bit addition pairs."""
        for x in range(256):
            for y in range(256):
                a = int_to_bipolar(x)
                b = int_to_bipolar(y)
                result = bipolar_add_32bit(a, b)
                expected = (x + y) & 0xFFFFFFFF
                assert bipolar_to_int(result) == expected, \
                    f"ADD({x}, {y}) = {bipolar_to_int(result)}, expected {expected}"


class TestSub32:
    def test_simple_sub(self):
        a = int_to_bipolar(8)
        b = int_to_bipolar(3)
        result = bipolar_sub_32bit(a, b)
        assert bipolar_to_int(result) == 5

    def test_sub_underflow(self):
        a = int_to_bipolar(0)
        b = int_to_bipolar(1)
        result = bipolar_sub_32bit(a, b)
        assert bipolar_to_int(result) == 0xFFFFFFFF  # wraps to -1 unsigned

    def test_sub_self(self):
        a = int_to_bipolar(12345)
        result = bipolar_sub_32bit(a, a.clone())
        assert bipolar_to_int(result) == 0


class TestBitwise32:
    def test_and(self):
        a = int_to_bipolar(0xFF00FF00)
        b = int_to_bipolar(0x0F0F0F0F)
        result = bipolar_and_32bit(a, b)
        assert bipolar_to_int(result) == 0x0F000F00

    def test_or(self):
        a = int_to_bipolar(0xFF00FF00)
        b = int_to_bipolar(0x0F0F0F0F)
        result = bipolar_or_32bit(a, b)
        assert bipolar_to_int(result) == 0xFF0FFF0F

    def test_xor(self):
        a = int_to_bipolar(0xFF00FF00)
        b = int_to_bipolar(0x0F0F0F0F)
        result = bipolar_xor_32bit(a, b)
        assert bipolar_to_int(result) == (0xFF00FF00 ^ 0x0F0F0F0F)

    def test_not(self):
        a = int_to_bipolar(0xFF00FF00)
        result = bipolar_not_32bit(a)
        assert bipolar_to_int(result) == 0x00FF00FF


class TestShifts:
    def test_shl(self):
        a = int_to_bipolar(1)
        result = bipolar_shl_32bit(a, 4)
        assert bipolar_to_int(result) == 16

    def test_shr(self):
        a = int_to_bipolar(256)
        result = bipolar_shr_32bit(a, 4)
        assert bipolar_to_int(result) == 16

    def test_sra_positive(self):
        a = int_to_bipolar(256)
        result = bipolar_sra_32bit(a, 4)
        assert bipolar_to_int(result) == 16

    def test_sra_negative(self):
        a = int_to_bipolar(0xFFFFFF00)  # -256 in signed
        result = bipolar_sra_32bit(a, 4)
        expected = 0xFFFFFFF0  # -16 in signed (sign-extended)
        assert bipolar_to_int(result) == expected

    def test_shl_zero_shift(self):
        a = int_to_bipolar(42)
        result = bipolar_shl_32bit(a, 0)
        assert bipolar_to_int(result) == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
