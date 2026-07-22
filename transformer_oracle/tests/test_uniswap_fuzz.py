"""
Gradient-guided fuzzing of Uniswap V2 swap invariant.

The core invariant: after a swap with 0.3% fee,
  balance0_adjusted * balance1_adjusted >= reserve0 * reserve1 * 1000^2

Where:
  balance0_adjusted = balance0 * 1000 - amount0In * 3
  balance1_adjusted = balance1 * 1000 - amount1In * 3

If we can find swap amounts that VIOLATE this invariant
(make the product decrease), that's a drain exploit.

Also tests:
  - Mint/burn liquidity calculation precision
  - Price oracle overflow (UQ112x112 fixed-point)
  - sqrt() precision in edge cases
"""

import torch
import pytest
from ..compiler.python_compiler import compile_python
from ..runtime.differentiable_executor import execute_differentiable
from ..runtime.gpu_executor import gpu_execute


# ══════════════════════════════════════════════════
# Uniswap V2 swap math — EXACT from UniswapV2Pair.sol
# ══════════════════════════════════════════════════

def ult64(a_hi, a_lo, b_hi, b_lo):
    """1 if 64-bit a_hi:a_lo < b_hi:b_lo (unsigned), else 0. Inlined by the compiler.
    K products overflow 32 bits for realistic pools, so K comparisons use full 64-bit
    values: `*` gives the low 32 bits and `umulh` the high 32."""
    if ult(a_hi, b_hi):
        return 1
    if a_hi == b_hi:
        return ult(a_lo, b_lo)
    return 0


def uniswap_swap_check(reserve0, reserve1, amount0_in, amount1_out):
    """Simulate a swap: deposit amount0_in of token0, withdraw amount1_out of token1.

    Returns:
      0 = invariant holds (swap is valid)
      1 = insufficient liquidity
      2 = no input
      3 = INVARIANT VIOLATION — K decreased! (exploit!)
      4 = balance underflow
    """
    # Uniswap uses uint112 for reserves (max ~5.19e33)
    # We use uint32 for simplicity (max ~4.29e9)

    if amount1_out >= reserve1:
        return 1  # insufficient liquidity

    if amount0_in == 0:
        return 2  # no input

    # After swap: new balances
    balance0 = reserve0 + amount0_in
    balance1 = reserve1 - amount1_out

    # Underflow check
    if balance1 > reserve1:
        return 4  # underflow

    # K invariant with 0.3% fee (lines 180-182 of UniswapV2Pair.sol)
    # balance0_adjusted = balance0 * 1000 - amount0_in * 3
    # balance1_adjusted = balance1 * 1000 - amount1_out_fee...
    # Actually in Uniswap: amount1In = 0 (we only send token0)
    # So balance1_adjusted = balance1 * 1000
    balance0_adj = balance0 * 1000 - amount0_in * 3
    balance1_adj = balance1 * 1000

    # Old K (scaled by 1000^2), as a 64-bit value (it overflows 32 bits for real pools).
    rr = reserve0 * reserve1
    old_hi = umulh(rr, 1000000)
    old_lo = rr * 1000000
    new_hi = umulh(balance0_adj, balance1_adj)
    new_lo = balance0_adj * balance1_adj

    if ult64(new_hi, new_lo, old_hi, old_lo):
        return 3  # INVARIANT VIOLATION!

    return 0


def uniswap_optimal_out(reserve0, reserve1, amount0_in):
    """Calculate the maximum amount1_out for a given amount0_in.

    From the constant product formula with 0.3% fee:
      amount1_out = reserve1 * amount0_in * 997 / (reserve0 * 1000 + amount0_in * 997)

    Returns the computed output amount.
    """
    if amount0_in == 0:
        return 0
    if reserve0 == 0:
        return 0

    numerator = reserve1 * amount0_in * 997
    denominator = reserve0 * 1000 + amount0_in * 997

    if denominator == 0:
        return 0

    # Integer division — rounds down (favors the protocol)
    amount_out = numerator - (numerator - denominator + 1)
    # Simpler: just divide
    # But we don't have division in our subset... use repeated subtraction
    # Actually, let's compute it differently
    # amount_out = numerator / denominator via subtraction
    result = 0
    remaining = numerator
    while remaining >= denominator:
        remaining = remaining - denominator
        result = result + 1

    return result


def uniswap_check_optimal(reserve0, reserve1, amount0_in):
    """Compute optimal output and verify the swap invariant holds.

    Returns:
      0 = invariant holds for computed output
      5 = computed output violates invariant
      6 = computed output + 1 also passes (can extract more!)
    """
    if reserve0 == 0 or reserve1 == 0:
        return 1
    if amount0_in == 0:
        return 2

    # Compute optimal output
    out = uniswap_optimal_out(reserve0, reserve1, amount0_in)

    if out >= reserve1:
        return 1  # would drain pool

    # Check invariant with computed output
    balance0 = reserve0 + amount0_in
    balance1 = reserve1 - out
    b0_adj = balance0 * 1000 - amount0_in * 3
    b1_adj = balance1 * 1000
    # K overflows 32 bits (e.g. 1e4·1e4·1e6 = 1e14); compare as 64-bit values.
    rr = reserve0 * reserve1
    old_hi = umulh(rr, 1000000)
    old_lo = rr * 1000000

    if ult64(umulh(b0_adj, b1_adj), b0_adj * b1_adj, old_hi, old_lo):
        return 5  # COMPUTED OUTPUT VIOLATES INVARIANT!

    # Check: can we extract one MORE token and still pass?
    if out + 1 < reserve1:
        balance1_extra = reserve1 - out - 1
        b1_extra = balance1_extra * 1000
        # b0_adj*b1_extra >= old_k  ==  not (b0_adj*b1_extra < old_k)
        if ult64(umulh(b0_adj, b1_extra), b0_adj * b1_extra, old_hi, old_lo) == 0:
            return 6  # CAN EXTRACT MORE — rounding leaves money on table

    return 0


class TestUniswapSwap:

    def test_basic_swap(self):
        """Normal swap should pass invariant."""
        nisa = compile_python(uniswap_swap_check)
        # Pool: 1000 token0, 1000 token1
        # Swap: put in 10 token0, take out 9 token1 (< optimal)
        r = gpu_execute(nisa, initial_registers={1: 1000, 2: 1000, 3: 10, 4: 9},
                        device='cuda')
        assert r.reg(10) == 0, f"Expected valid swap, got {r.reg(10)}"

    def test_greedy_swap_fails(self):
        """Taking too much output should violate invariant."""
        nisa = compile_python(uniswap_swap_check)
        # Try to take 100 token1 for only 10 token0 input — too greedy
        r = gpu_execute(nisa, initial_registers={1: 1000, 2: 1000, 3: 10, 4: 100},
                        device='cuda')
        assert r.reg(10) == 3, f"Expected invariant violation (3), got {r.reg(10)}"

    def test_drain_fails(self):
        """Can't withdraw more than the reserve."""
        nisa = compile_python(uniswap_swap_check)
        r = gpu_execute(nisa, initial_registers={1: 1000, 2: 1000, 3: 10, 4: 1001},
                        device='cuda')
        assert r.reg(10) == 1  # insufficient liquidity

    def test_optimal_output(self):
        """Verify optimal output computation."""
        nisa = compile_python(uniswap_check_optimal)
        # Pool: 10000/10000, swap 100 in
        # Expected: ~99 out (with fee)
        r = gpu_execute(nisa, initial_registers={1: 10000, 2: 10000, 3: 100},
                        device='cuda', max_cycles=5000000)
        print(f"\n  Optimal output check: code={r.reg(10)}")
        assert r.reg(10) in (0, 6)  # 0=exact, 6=rounding leaves some


class TestGradientSwapFuzzing:

    def test_gradient_finds_boundary(self):
        """Use gradient descent to find the max extractable output."""
        nisa = compile_python(uniswap_swap_check)

        print("\n" + "="*60)
        print("GRADIENT-GUIDED UNISWAP SWAP FUZZING")
        print("="*60)

        # Pool: 10000/10000
        reserve0 = 10000.0
        reserve1 = 10000.0
        amount_in = 100.0

        # Search for max amount_out that doesn't violate K
        amount_out = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([amount_out], lr=5.0)

        print(f"\n  Pool: {int(reserve0)}/{int(reserve1)}")
        print(f"  Swap in: {int(amount_in)} token0")
        print(f"  Searching for max token1 output...\n")

        for i in range(300):
            optimizer.zero_grad()
            result = execute_differentiable(nisa, {
                1: torch.tensor(reserve0, dtype=torch.float64),
                2: torch.tensor(reserve1, dtype=torch.float64),
                3: torch.tensor(amount_in, dtype=torch.float64),
                4: amount_out,
            })

            # Loss: maximize amount_out while keeping code == 0
            output_code = result.registers[10]
            # We want code to stay 0 (valid), so penalize code != 0
            # But also maximize amount_out
            loss = -amount_out + 1000 * output_code
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                amount_out.clamp_(min=1.0, max=reserve1 - 1)

            if i % 50 == 0:
                out_int = int(amount_out.item())
                r = gpu_execute(nisa, initial_registers={
                    1: int(reserve0), 2: int(reserve1),
                    3: int(amount_in), 4: out_int
                }, device='cuda')
                grad = amount_out.grad.item() if amount_out.grad is not None else 0
                print(f"  iter {i:3d}: out={out_int:5d} code={r.reg(10)} grad={grad:.4f}")

        # Verify final amount
        final_out = int(amount_out.item())
        r_final = gpu_execute(nisa, initial_registers={
            1: int(reserve0), 2: int(reserve1),
            3: int(amount_in), 4: final_out
        }, device='cuda')
        r_plus1 = gpu_execute(nisa, initial_registers={
            1: int(reserve0), 2: int(reserve1),
            3: int(amount_in), 4: final_out + 1
        }, device='cuda')

        print(f"\n  Max extractable: {final_out} → code={r_final.reg(10)}")
        print(f"  One more ({final_out+1}):  → code={r_plus1.reg(10)}")

        # Compute what the formula gives
        # amount_out = reserve1 * amountIn * 997 / (reserve0 * 1000 + amountIn * 997)
        numerator = int(reserve1) * int(amount_in) * 997
        denominator = int(reserve0) * 1000 + int(amount_in) * 997
        formula_out = numerator // denominator
        print(f"  Formula output:  {formula_out}")
        print(f"  Gradient found:  {final_out}")

        if final_out > formula_out:
            print(f"\n  !!! GRADIENT FOUND MORE THAN FORMULA — possible exploit !!!")
        elif final_out == formula_out:
            print(f"\n  Gradient found exact optimal — swap math is tight.")
        else:
            print(f"\n  Gradient found less than optimal — conservative.")

    def test_fuzz_many_pools(self):
        """Fuzz swap invariant across many pool sizes."""
        nisa = compile_python(uniswap_swap_check)

        print("\n  Fuzzing swap invariant across pool sizes:")
        import random
        random.seed(42)

        violations = []
        for trial in range(1000):
            r0 = random.randint(100, 1000000)
            r1 = random.randint(100, 1000000)
            amt_in = random.randint(1, r0 // 2)

            # Compute optimal output
            num = r1 * amt_in * 997
            den = r0 * 1000 + amt_in * 997
            opt_out = num // den

            if opt_out <= 0 or opt_out >= r1:
                continue

            # Verify invariant holds for optimal output
            r = gpu_execute(nisa, initial_registers={1: r0, 2: r1, 3: amt_in, 4: opt_out},
                            device='cuda')
            if r.reg(10) == 3:
                violations.append((r0, r1, amt_in, opt_out))

            # Check optimal + 1 violates
            r2 = gpu_execute(nisa, initial_registers={1: r0, 2: r1, 3: amt_in, 4: opt_out + 1},
                             device='cuda')
            if r2.reg(10) == 0:
                violations.append((r0, r1, amt_in, opt_out + 1, "EXTRA"))

        print(f"    1000 random pools tested")
        print(f"    Violations found: {len(violations)}")
        if violations:
            for v in violations[:3]:
                if len(v) == 5:
                    print(f"      EXTRA: pool={v[0]}/{v[1]} in={v[2]} out={v[3]}")
                else:
                    print(f"      BUG: pool={v[0]}/{v[1]} in={v[2]} out={v[3]}")
        else:
            print(f"    Uniswap V2 swap invariant is SOLID.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
