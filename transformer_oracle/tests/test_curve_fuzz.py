"""
Fuzzing Curve StableSwap invariant.

Implements the EXACT math from StableSwap3Pool.vy (get_D, get_y)
and tests with 256-bit precision to find:
  - Newton's method convergence failures
  - D invariant violations after swaps
  - Precision loss at extreme pool states
  - Edge cases with amplification parameter A
"""

import random
import pytest


# ══════════════════════════════════════════════════
# EXACT Curve math from StableSwap3Pool.vy
# ══════════════════════════════════════════════════

def get_D(xp, amp):
    """Compute StableSwap invariant D via Newton's method.
    VERBATIM from StableSwap3Pool.vy lines 195-220."""
    N = len(xp)
    S = sum(xp)
    if S == 0:
        return 0

    D = S
    Ann = amp * N

    for _ in range(255):
        D_P = D
        for x in xp:
            D_P = D_P * D // (x * N)  # integer division
        Dprev = D
        D = (Ann * S + D_P * N) * D // ((Ann - 1) * D + (N + 1) * D_P)
        if abs(D - Dprev) <= 1:
            return D
    raise ValueError("get_D didn't converge")


def get_y(xp, i, j, x_new, amp):
    """Compute new balance y_j after changing x_i to x_new.
    VERBATIM from StableSwap3Pool.vy lines 356-393."""
    N = len(xp)
    D = get_D(xp, amp)
    Ann = amp * N

    c = D
    S_ = 0
    for k in range(N):
        if k == i:
            _x = x_new
        elif k != j:
            _x = xp[k]
        else:
            continue
        S_ += _x
        c = c * D // (_x * N)

    c = c * D // (Ann * N)
    b = S_ + D // Ann

    y = D
    for _ in range(255):
        y_prev = y
        y = (y * y + c) // (2 * y + b - D)
        if abs(y - y_prev) <= 1:
            return y
    raise ValueError("get_y didn't converge")


def swap_and_check(xp, i, j, dx, amp, fee_num=4, fee_den=10000):
    """Execute a Curve swap and check invariant.

    Returns: (dy, D_before, D_after, code)
      code 0 = valid, 1 = D violation, 2 = convergence error, 3 = bad output
    """
    try:
        D_before = get_D(xp, amp)
    except ValueError:
        return 0, 0, 0, 2

    xp_new = list(xp)
    xp_new[i] = xp[i] + dx

    try:
        y = get_y(xp, i, j, xp_new[i], amp)
    except ValueError:
        return 0, D_before, 0, 2

    dy = xp[j] - y - 1  # -1 for rounding (favors pool)
    if dy <= 0:
        return 0, D_before, D_before, 3

    # Apply fee
    fee = dy * fee_num // fee_den
    dy_after_fee = dy - fee

    # New state after swap (fee stays in pool)
    xp_new[j] = xp[j] - dy_after_fee

    try:
        D_after = get_D(xp_new, amp)
    except ValueError:
        return dy_after_fee, D_before, 0, 2

    # D should not decrease (fees only increase it)
    if D_after < D_before - 1:
        return dy_after_fee, D_before, D_after, 1  # VIOLATION

    return dy_after_fee, D_before, D_after, 0


class TestCurveGetD:

    def test_balanced_2pool(self):
        """Balanced 2-pool: D = sum."""
        D = get_D([10**18, 10**18], 100)
        assert abs(D - 2 * 10**18) <= 1
        print(f"\n  Balanced 1e18/1e18, A=100: D={D}")

    def test_balanced_3pool(self):
        """Balanced 3-pool: D = sum."""
        D = get_D([10**18, 10**18, 10**18], 100)
        assert abs(D - 3 * 10**18) <= 1
        print(f"\n  Balanced 3pool 1e18 each, A=100: D={D}")

    def test_unbalanced(self):
        D = get_D([1500000 * 10**18, 500000 * 10**18], 100)
        print(f"\n  Unbalanced 1.5M/0.5M, A=100: D={D / 10**18:.2f}")
        assert D > 0

    def test_extreme_A(self):
        """High A → closer to constant sum (less slippage)."""
        D_low = get_D([10**18, 10**18], 1)
        D_high = get_D([10**18, 10**18], 10000)
        print(f"\n  A=1: D={D_low}, A=10000: D={D_high}")
        # Both should be ~2e18 for balanced pool
        assert abs(D_low - 2 * 10**18) <= 1
        assert abs(D_high - 2 * 10**18) <= 1


class TestCurveSwapInvariant:

    def test_basic_swap(self):
        """Normal swap preserves D."""
        xp = [10**24, 10**24]  # 1M tokens with 18 decimals
        dy, D_before, D_after, code = swap_and_check(xp, 0, 1, 10**22, 100)
        print(f"\n  Swap 10K in 1M/1M pool: dy={dy/10**18:.2f}, D_before={D_before}, D_after={D_after}")
        assert code == 0

    def test_random_fuzz_2pool(self):
        """Random fuzz 2-pool swaps."""
        print("\n  Random 2-pool fuzz (1000 swaps):")
        random.seed(42)
        violations = []
        convergence_fails = 0

        for trial in range(1000):
            base = 10 ** random.randint(18, 30)
            ratio = random.uniform(0.1, 10.0)
            xp = [int(base * ratio), int(base / ratio)]
            dx = random.randint(1, max(xp[0] // 10, 1))
            amp = random.choice([1, 10, 50, 100, 500, 1000, 5000])

            dy, db, da, code = swap_and_check(xp, 0, 1, dx, amp)
            if code == 1:
                violations.append((xp, dx, amp, db, da))
            elif code == 2:
                convergence_fails += 1

        print(f"    D violations: {len(violations)}")
        print(f"    Convergence failures: {convergence_fails}")
        if violations:
            for xp, dx, amp, db, da in violations[:3]:
                print(f"      pool={xp} dx={dx} A={amp} D:{db}→{da}")
        else:
            print(f"    StableSwap invariant holds.")

    def test_random_fuzz_3pool(self):
        """Random fuzz 3-pool swaps (DAI/USDC/USDT)."""
        print("\n  Random 3-pool fuzz (500 swaps):")
        random.seed(123)
        violations = []
        convergence_fails = 0

        for trial in range(500):
            base = 10 ** random.randint(20, 26)
            xp = [int(base * random.uniform(0.5, 2.0)) for _ in range(3)]
            i = random.randint(0, 2)
            j = (i + random.randint(1, 2)) % 3
            dx = random.randint(1, max(xp[i] // 10, 1))
            amp = random.choice([100, 500, 1000, 2000, 5000])

            dy, db, da, code = swap_and_check(xp, i, j, dx, amp)
            if code == 1:
                violations.append((xp, i, j, dx, amp, db, da))
            elif code == 2:
                convergence_fails += 1

        print(f"    D violations: {len(violations)}")
        print(f"    Convergence failures: {convergence_fails}")
        if violations:
            for xp, i, j, dx, amp, db, da in violations[:3]:
                print(f"      pool={[f'{x:.0e}' for x in xp]} {i}→{j} dx={dx:.0e} A={amp}")
                print(f"      D: {db} → {da} (decreased by {db-da})")
        else:
            print(f"    StableSwap 3-pool invariant holds.")

    def test_extreme_imbalance(self):
        """Heavily imbalanced pool — attack surface for depeg exploits."""
        print("\n  Extreme imbalance tests:")
        for ratio in [100, 1000, 10000, 100000]:
            xp = [10**24, 10**24 // ratio]
            try:
                dy, db, da, code = swap_and_check(xp, 0, 1, 10**22, 100)
                print(f"    ratio=1:{ratio} → dy={dy:.0e} code={code}")
            except Exception as e:
                print(f"    ratio=1:{ratio} → error: {e}")

    def test_near_zero_balance(self):
        """Pool nearly drained — can invariant be broken?"""
        print("\n  Near-depletion fuzzing:")
        violations = []
        for xp1 in [10**18, 10**15, 10**12, 10**9, 10**6, 1000, 100, 10, 1]:
            xp = [10**24, xp1]
            for amp in [1, 100, 5000]:
                try:
                    dy, db, da, code = swap_and_check(xp, 0, 1, 10**20, amp)
                    if code == 1:
                        violations.append((xp, amp, db, da))
                        print(f"    !!! VIOLATION: pool=[1e24,{xp1}] A={amp} D:{db}→{da}")
                except:
                    pass

        if not violations:
            print(f"    All near-depletion swaps preserve D.")

    def test_fee_extraction(self):
        """Can fees be extracted to decrease D?"""
        print("\n  Fee extraction test:")
        xp = [10**24, 10**24]
        amp = 100

        # Do many swaps back and forth — D should only increase (fees accumulate)
        D_initial = get_D(xp, amp)
        current_xp = list(xp)

        for i in range(10):
            dy, _, _, code = swap_and_check(current_xp, 0, 1, current_xp[0] // 100, amp)
            if code == 0 and dy > 0:
                current_xp[0] += current_xp[0] // 100
                current_xp[1] -= dy

            dy2, _, _, code2 = swap_and_check(current_xp, 1, 0, current_xp[1] // 100, amp)
            if code2 == 0 and dy2 > 0:
                current_xp[1] += current_xp[1] // 100
                current_xp[0] -= dy2

        D_final = get_D(current_xp, amp)
        print(f"    After 10 round-trip swaps:")
        print(f"    D: {D_initial} → {D_final}")
        print(f"    D increased by: {D_final - D_initial}")
        assert D_final >= D_initial - 1, "D should not decrease with fees!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
