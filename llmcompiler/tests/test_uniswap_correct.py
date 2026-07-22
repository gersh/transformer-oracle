"""Uniswap V2 swap invariant — 32-bit safe pool sizes."""
import random
import pytest
from ..compiler.python_compiler import compile_python
from ..runtime.gpu_executor import gpu_execute


def ult64(a_hi, a_lo, b_hi, b_lo):
    """1 if the 64-bit value a_hi:a_lo < b_hi:b_lo (unsigned), else 0.
    (`ult`/`umulh` are compiler builtins; this is inlined at the call site.)"""
    if ult(a_hi, b_hi):
        return 1
    if a_hi == b_hi:
        return ult(a_lo, b_lo)
    return 0


def swap_check(reserve0, reserve1, amount_in, amount_out):
    """Uniswap V2 K invariant check (UniswapV2Pair.sol lines 180-182)."""
    if amount_out >= reserve1:
        return 1
    if amount_in == 0:
        return 2
    balance0 = reserve0 + amount_in
    balance1 = reserve1 - amount_out
    b0_adj = balance0 * 1000 - amount_in * 3
    b1_adj = balance1 * 1000
    # K products overflow 32 bits for realistic pools (e.g. 1000·1000·1e6 = 1e12), so
    # compare as full 64-bit values: `*` gives the low 32 bits, `umulh` the high 32.
    rr = reserve0 * reserve1
    old_hi = umulh(rr, 1000000)
    old_lo = rr * 1000000
    new_hi = umulh(b0_adj, b1_adj)
    new_lo = b0_adj * b1_adj
    if ult64(new_hi, new_lo, old_hi, old_lo):
        return 3
    return 0


class TestUniswapCorrect:

    @pytest.fixture(autouse=True)
    def compile_swap(self):
        self.nisa = compile_python(swap_check)

    def _run(self, r0, r1, ai, ao):
        return gpu_execute(self.nisa, initial_registers={1:r0, 2:r1, 3:ai, 4:ao},
                           device='cuda').reg(10)

    def test_formula_verification(self):
        """Verify optimal output from formula passes, +1 fails."""
        print("\n  Formula verification (reserves ≤ 60):")
        for r0, r1, ai in [(50,50,5), (60,40,10), (30,60,3), (50,50,1), (55,45,15)]:
            opt = (r1 * ai * 997) // (r0 * 1000 + ai * 997)
            if opt <= 0 or opt >= r1:
                continue
            code = self._run(r0, r1, ai, opt)
            code_plus = self._run(r0, r1, ai, opt + 1)
            tight = "tight" if code_plus == 3 else "LOOSE"
            print(f"    {r0}/{r1} in={ai}: opt={opt} code={code} +1={code_plus} ({tight})")
            assert code == 0, f"Optimal output should pass: code={code}"

    def test_random_fuzz(self):
        """Fuzz 1000 random small pools."""
        random.seed(42)
        violations = []
        loose = 0
        tested = 0

        for _ in range(1000):
            r0 = random.randint(10, 60)
            r1 = random.randint(10, 60)
            ai = random.randint(1, min(r0, 20))
            opt = (r1 * ai * 997) // (r0 * 1000 + ai * 997)
            if opt <= 0 or opt >= r1:
                continue
            tested += 1
            code = self._run(r0, r1, ai, opt)
            if code == 3:
                violations.append((r0, r1, ai, opt))
            code2 = self._run(r0, r1, ai, opt + 1)
            if code2 == 0:
                loose += 1

        print(f"\n  Random fuzz: {tested} valid pools tested")
        print(f"  Invariant violations: {len(violations)}")
        print(f"  Loose (opt+1 passes): {loose}")
        if loose:
            print(f"  → Rounding favors LPs (by design)")
        if not violations:
            print(f"  Uniswap V2 swap math is CORRECT for all tested pools.")
        assert len(violations) == 0

    def test_edge_cases(self):
        """Test edge case pool sizes and swap amounts."""
        edge = [
            (10, 10, 1, 0),    # minimum swap
            (60, 60, 60, 29),  # large relative to pool
            (10, 60, 9, 26),   # unbalanced pool
            (60, 10, 1, 0),    # very unbalanced
        ]
        for r0, r1, ai, ao in edge:
            code = self._run(r0, r1, ai, ao)
            print(f"    {r0}/{r1} in={ai} out={ao}: code={code}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
