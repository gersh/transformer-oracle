"""
Forgery testing for XRP Ledger equality proof.

Models the Sigma protocol verification in scalar arithmetic
(mod curve order) and uses fuzzing to search for proof values
that pass verification WITHOUT knowing the secret witness.

The equality proof verifies:
  s*G == T1 + e*C1        (equation 1)
  s*P == T2 + e*(C2-mG)   (equation 2)

Where e = Hash(domain || C1 || C2 || pk || mG || T1 || T2 || context)

In scalar space (discrete log representation):
  If C1 = r*G, C2 = m*G + r*P, T1 = t*G, T2 = t*P
  Then: s = t + e*r satisfies both equations.

We model this by representing points as their discrete logs
(which we wouldn't know in practice, but lets us test the LOGIC).

A forgery would be: finding (T1, T2, s) that satisfies the
verification equations for ARBITRARY e without knowing r.
"""

import random
import hashlib
import pytest

# secp256k1 curve order
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def mod_n(x):
    return x % N


def scalar_hash(*args):
    """Simplified Fiat-Shamir: hash scalars to produce challenge."""
    h = hashlib.sha256()
    h.update(b"MPT_POK_PLAINTEXT_PROOF")
    for a in args:
        h.update(a.to_bytes(32, 'big'))
    return int.from_bytes(h.digest(), 'big') % N


class TestEqualityProofSoundness:
    """Test that the equality proof can't be forged."""

    def test_honest_proof_verifies(self):
        """An honest prover with the real witness can create a valid proof."""
        random.seed(42)

        # Secret witness
        r = random.randint(1, N - 1)  # randomness
        m = 1000  # amount (known to prover)

        # Public key (P = sk*G, but we just track scalars)
        sk = random.randint(1, N - 1)

        # Public values (in scalar/DL representation)
        # C1 = r*G → scalar r
        # C2 = m*G + r*P → scalar m + r*sk
        c1_scalar = r
        c2_scalar = mod_n(m + r * sk)

        # Prover commits: t random, T1 = t*G, T2 = t*P
        t = random.randint(1, N - 1)
        t1_scalar = t
        t2_scalar = mod_n(t * sk)

        # Challenge (Fiat-Shamir)
        e = scalar_hash(c1_scalar, c2_scalar, sk, m, t1_scalar, t2_scalar)

        # Response
        s = mod_n(t + e * r)

        # Verification equation 1: s*G == T1 + e*C1
        # In scalars: s == t1 + e*c1
        lhs1 = s
        rhs1 = mod_n(t1_scalar + e * c1_scalar)
        eq1 = (lhs1 == rhs1)

        # Verification equation 2: s*P == T2 + e*(C2 - m*G)
        # In scalars: s*sk == t2 + e*(c2 - m)
        lhs2 = mod_n(s * sk)
        rhs2 = mod_n(t2_scalar + e * mod_n(c2_scalar - m))
        eq2 = (lhs2 == rhs2)

        print(f"\n  Honest proof: eq1={eq1} eq2={eq2}")
        assert eq1 and eq2, "Honest proof should verify!"

    def test_random_proof_fails(self):
        """Random proof values should NOT verify."""
        random.seed(42)
        r = random.randint(1, N - 1)
        sk = random.randint(1, N - 1)
        m = 1000
        c1 = r
        c2 = mod_n(m + r * sk)

        failures = 0
        for _ in range(10000):
            # Random proof (no knowledge of r)
            t1 = random.randint(1, N - 1)
            t2 = random.randint(1, N - 1)
            s = random.randint(1, N - 1)

            e = scalar_hash(c1, c2, sk, m, t1, t2)

            eq1 = (s == mod_n(t1 + e * c1))
            eq2 = (mod_n(s * sk) == mod_n(t2 + e * mod_n(c2 - m)))

            if eq1 and eq2:
                failures += 1
                print(f"  !!! FORGERY: s={s}, t1={t1}, t2={t2} !!!")

        print(f"\n  10,000 random proofs: {failures} forgeries found")
        assert failures == 0, "Random proofs should never verify!"

    def test_forge_without_witness(self):
        """Attempt forgery: choose s first, compute T1 and T2 to match.

        The attacker's strategy:
          1. Pick arbitrary s
          2. For eq1: need T1 = s*G - e*C1, but e depends on T1 (circular!)
          3. Try: pick T1 arbitrarily, compute e, check if s matches

        This is equivalent to finding a hash preimage — should be infeasible.
        """
        random.seed(42)
        r = random.randint(1, N - 1)
        sk = random.randint(1, N - 1)
        m = 1000
        c1 = r
        c2 = mod_n(m + r * sk)

        forgeries = 0
        # Try many s values
        for trial in range(10000):
            s = random.randint(1, N - 1)
            t1 = random.randint(1, N - 1)

            # Compute what e would be
            # We can compute t2 from the first equation to satisfy eq1:
            # But we don't know what t2 to use for the hash...
            # So we pick t2 randomly and check
            t2 = random.randint(1, N - 1)
            e = scalar_hash(c1, c2, sk, m, t1, t2)

            # Check eq1: s == t1 + e*c1 (mod N)?
            if s != mod_n(t1 + e * c1):
                continue  # eq1 fails, try next

            # eq1 passes! Check eq2
            if mod_n(s * sk) == mod_n(t2 + e * mod_n(c2 - m)):
                forgeries += 1
                print(f"  !!! FORGERY at trial {trial} !!!")

        print(f"\n  10,000 forgery attempts: {forgeries} succeeded")
        assert forgeries == 0

    def test_forge_with_chosen_e(self):
        """Forgery attempt: choose e first, then compute T1, T2 to match.

        If the attacker can choose e freely (simulator's strategy):
          1. Pick e, s arbitrarily
          2. Compute T1 = s*G - e*C1 (in scalar: t1 = s - e*c1)
          3. Compute T2 = s*P - e*(C2-mG) (in scalar: t2 = s*sk - e*(c2-m))
          4. Check: does Hash(..., T1, T2, ...) == e?

        This only works if you can find a hash preimage, which is
        computationally infeasible with SHA-256.
        """
        random.seed(42)
        r = random.randint(1, N - 1)
        sk = random.randint(1, N - 1)
        m = 1000
        c1 = r
        c2 = mod_n(m + r * sk)

        forgeries = 0
        for trial in range(100000):
            # Attacker chooses e and s freely
            e_chosen = random.randint(1, N - 1)
            s = random.randint(1, N - 1)

            # Compute T1, T2 that satisfy the verification equations
            t1 = mod_n(s - e_chosen * c1)
            t2 = mod_n(s * sk - e_chosen * mod_n(c2 - m))

            # Now check: does the Fiat-Shamir hash give back e_chosen?
            e_actual = scalar_hash(c1, c2, sk, m, t1, t2)

            if e_actual == e_chosen:
                forgeries += 1
                print(f"  !!! HASH PREIMAGE FOUND at trial {trial} !!!")
                print(f"  e={e_chosen}, s={s}")

        print(f"\n  100,000 preimage attempts: {forgeries} found")
        assert forgeries == 0, "Finding a hash preimage would break SHA-256!"

    def test_replay_different_amount(self):
        """Can a proof for amount=X be replayed for amount=Y?"""
        random.seed(42)
        r = random.randint(1, N - 1)
        sk = random.randint(1, N - 1)

        # Create honest proof for amount=1000
        m1 = 1000
        c1 = r
        c2 = mod_n(m1 + r * sk)
        t = random.randint(1, N - 1)
        t1 = t
        t2 = mod_n(t * sk)
        e1 = scalar_hash(c1, c2, sk, m1, t1, t2)
        s1 = mod_n(t + e1 * r)

        # Try to use this proof for amount=2000
        m2 = 2000
        e2 = scalar_hash(c1, c2, sk, m2, t1, t2)

        # Check eq1 with new challenge
        eq1 = (s1 == mod_n(t1 + e2 * c1))
        # Check eq2 with new amount
        eq2 = (mod_n(s1 * sk) == mod_n(t2 + e2 * mod_n(c2 - m2)))

        print(f"\n  Replay attack: proof for {m1} used for {m2}")
        print(f"  eq1={eq1} eq2={eq2}")
        assert not (eq1 and eq2), "Replay for different amount should fail!"

    def test_replay_different_recipient(self):
        """Can a proof be replayed for a different recipient?"""
        random.seed(42)
        r = random.randint(1, N - 1)
        sk1 = random.randint(1, N - 1)
        sk2 = random.randint(1, N - 1)  # different recipient
        m = 1000

        # Proof for recipient 1
        c1 = r
        c2 = mod_n(m + r * sk1)
        t = random.randint(1, N - 1)
        t1 = t
        t2_1 = mod_n(t * sk1)
        e = scalar_hash(c1, c2, sk1, m, t1, t2_1)
        s = mod_n(t + e * r)

        # Try verification against recipient 2
        e2 = scalar_hash(c1, c2, sk2, m, t1, t2_1)
        eq1 = (s == mod_n(t1 + e2 * c1))
        eq2 = (mod_n(s * sk2) == mod_n(t2_1 + e2 * mod_n(c2 - m)))

        print(f"\n  Replay to different recipient: eq1={eq1} eq2={eq2}")
        assert not (eq1 and eq2), "Replay for different recipient should fail!"

    def test_zero_amount_edge_case(self):
        """Test the amount=0 edge case — does it have different security?"""
        random.seed(42)
        r = random.randint(1, N - 1)
        sk = random.randint(1, N - 1)

        # amount=0: C2 = 0*G + r*P = r*P
        m = 0
        c1 = r
        c2 = mod_n(r * sk)  # just r*P, no m*G

        # Honest proof
        t = random.randint(1, N - 1)
        t1 = t
        t2 = mod_n(t * sk)
        # For amount=0, mG is omitted from hash
        e = scalar_hash(c1, c2, sk, t1, t2)  # NO m in hash
        s = mod_n(t + e * r)

        # Verify
        eq1 = (s == mod_n(t1 + e * c1))
        eq2 = (mod_n(s * sk) == mod_n(t2 + e * c2))  # C2-0 = C2
        print(f"\n  amount=0 proof: eq1={eq1} eq2={eq2}")
        assert eq1 and eq2

        # Can a proof for amount=0 be used for amount=100?
        m_fake = 100
        e_fake = scalar_hash(c1, c2, sk, m_fake, t1, t2)
        eq1_f = (s == mod_n(t1 + e_fake * c1))
        eq2_f = (mod_n(s * sk) == mod_n(t2 + e_fake * mod_n(c2 - m_fake)))
        print(f"  Replay amount=0→100: eq1={eq1_f} eq2={eq2_f}")
        assert not (eq1_f and eq2_f), "Zero→nonzero replay should fail!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
