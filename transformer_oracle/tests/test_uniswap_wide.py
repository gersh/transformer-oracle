"""
Uniswap V2 swap fuzzing with 256-bit integers.

Uses the wide_int_executor with bit_width=256, matching Solidity's uint256.
Now we can test with realistic pool sizes (millions of tokens).
"""

import random
import pytest
from ..compiler.python_compiler import compile_python
from ..runtime.wide_int_executor import wide_execute


def swap_check(reserve0, reserve1, amount_in, amount_out):
    """Uniswap V2 K invariant check — EXACT from UniswapV2Pair.sol lines 180-182."""
    if amount_out >= reserve1:
        return 1
    if amount_in == 0:
        return 2
    balance0 = reserve0 + amount_in
    balance1 = reserve1 - amount_out
    b0_adj = balance0 * 1000 - amount_in * 3
    b1_adj = balance1 * 1000
    old_k = reserve0 * reserve1 * 1000000
    new_k = b0_adj * b1_adj
    if new_k < old_k:
        return 3
    return 0


class TestUniswapWide:
    """Test with 256-bit integers matching Solidity."""

    @pytest.fixture(autouse=True)
    def compile_swap(self):
        self.nisa = compile_python(swap_check)

    def _run(self, r0, r1, ai, ao, bits=256):
        regs, _, _ = wide_execute(self.nisa,
                                  initial_registers={1: r0, 2: r1, 3: ai, 4: ao},
                                  bit_width=bits)
        return regs[10]

    def test_small_pool(self):
        """Small pool sanity check."""
        assert self._run(1000, 1000, 10, 9) == 0

    def test_realistic_pool(self):
        """Realistic DeFi pool: $10M TVL."""
        r0 = 10_000_000 * 10**18  # 10M tokens with 18 decimals
        r1 = 10_000_000 * 10**18
        amt_in = 100_000 * 10**18  # 100K swap
        opt = (r1 * amt_in * 997) // (r0 * 1000 + amt_in * 997)
        code = self._run(r0, r1, amt_in, opt)
        print(f"\n  $10M pool, $100K swap: out={opt / 10**18:.2f} tokens, code={code}")
        assert code == 0

    def test_whale_swap(self):
        """Whale swap: 10% of pool."""
        r0 = 1_000_000 * 10**18
        r1 = 1_000_000 * 10**18
        amt_in = 100_000 * 10**18
        opt = (r1 * amt_in * 997) // (r0 * 1000 + amt_in * 997)
        code = self._run(r0, r1, amt_in, opt)
        code_plus = self._run(r0, r1, amt_in, opt + 1)
        print(f"\n  Whale swap (10% of pool): out={opt/10**18:.2f}")
        print(f"  Optimal passes: {code == 0}, +1 fails: {code_plus == 3}")
        assert code == 0

    def test_tiny_swap(self):
        """Dust swap: 1 wei."""
        r0 = 1_000_000 * 10**18
        r1 = 1_000_000 * 10**18
        amt_in = 1  # 1 wei
        opt = (r1 * amt_in * 997) // (r0 * 1000 + amt_in * 997)
        code = self._run(r0, r1, amt_in, max(opt, 0))
        print(f"\n  Dust swap (1 wei): optimal_out={opt}")
        # With 1 wei input into a $1M pool, output should be 0 (rounds down)
        assert code == 0 or code == 2

    def test_max_uint112_reserves(self):
        """Maximum Uniswap reserves (uint112 max ≈ 5.19e33)."""
        max112 = (1 << 112) - 1
        r0 = max112
        r1 = max112
        amt_in = max112 // 100
        opt = (r1 * amt_in * 997) // (r0 * 1000 + amt_in * 997)
        code = self._run(r0, r1, amt_in, opt)
        print(f"\n  Max uint112 reserves: {max112:.2e}")
        print(f"  1% swap: out={opt:.2e}, code={code}")
        assert code == 0

    def test_formula_tightness(self):
        """Check: is the formula always tight (opt+1 always fails)?"""
        print("\n  Formula tightness check (various pool sizes):")
        loose_count = 0
        random.seed(42)
        for _ in range(200):
            scale = 10 ** random.randint(1, 30)  # 10 to 10^30
            r0 = random.randint(1000, 10000) * scale
            r1 = random.randint(1000, 10000) * scale
            pct = random.uniform(0.001, 0.1)  # 0.1% to 10% of pool
            amt_in = int(r0 * pct)
            if amt_in == 0:
                continue

            opt = (r1 * amt_in * 997) // (r0 * 1000 + amt_in * 997)
            if opt <= 0 or opt >= r1:
                continue

            code = self._run(r0, r1, amt_in, opt)
            code_plus = self._run(r0, r1, amt_in, opt + 1)

            if code != 0:
                print(f"    BUG: pool={r0:.2e}/{r1:.2e} in={amt_in:.2e} opt={opt:.2e} FAILS!")
            if code_plus == 0:
                loose_count += 1

        print(f"    Loose (can extract +1): {loose_count}/200")
        print(f"    → Integer division rounds down, leaving dust for LPs")

    def test_random_fuzz_256bit(self):
        """Heavy random fuzzing with 256-bit math."""
        print("\n  Random fuzzing (10K pools, 256-bit):")
        random.seed(42)
        violations = []
        tested = 0

        for trial in range(10000):
            # Random pool sizes across many orders of magnitude
            exp0 = random.randint(3, 30)
            exp1 = random.randint(3, 30)
            r0 = random.randint(1, 9999) * (10 ** exp0)
            r1 = random.randint(1, 9999) * (10 ** exp1)
            amt_in = random.randint(1, max(r0 // 10, 1))

            opt = (r1 * amt_in * 997) // (r0 * 1000 + amt_in * 997)
            if opt <= 0 or opt >= r1:
                continue
            tested += 1

            code = self._run(r0, r1, amt_in, opt)
            if code == 3:
                violations.append((r0, r1, amt_in, opt, code))

        print(f"    {tested} valid pools tested")
        print(f"    Invariant violations: {len(violations)}")
        if violations:
            for r0, r1, ai, opt, c in violations[:3]:
                print(f"      pool={r0:.2e}/{r1:.2e} in={ai:.2e} out={opt:.2e}")
        else:
            print(f"    Uniswap V2 swap math is CORRECT across all scales.")
        assert len(violations) == 0

    def test_unbalanced_pools(self):
        """Test extremely unbalanced pools (common attack vector)."""
        print("\n  Unbalanced pool tests:")
        cases = [
            (10**30, 1, 10**20, "huge/tiny"),
            (1, 10**30, 1, "tiny/huge"),
            (10**18, 10**6, 10**17, "18dec/6dec"),
            (10**6, 10**18, 10**5, "6dec/18dec"),
        ]
        for r0, r1, ai, desc in cases:
            opt = (r1 * ai * 997) // (r0 * 1000 + ai * 997) if r0 > 0 else 0
            if opt <= 0 or opt >= r1:
                print(f"    {desc}: out=0 (too small)")
                continue
            code = self._run(r0, r1, ai, opt)
            print(f"    {desc}: pool={r0:.0e}/{r1:.0e} in={ai:.0e} out={opt:.0e} code={code}")
            assert code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
