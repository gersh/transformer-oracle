#!/usr/bin/env python3
"""
Worked example: auditing a Lean `native_decide` verdict on the transformer.

Lean's `native_decide` proves a proposition `P` by *compiling* its `Decidable` instance to native
code, running it, and trusting that the machine returned `true`. That trust rests on the
`ofReduceBool` / `trustCompiler` axioms — the compiler + your CPU become part of the trusted base.

This tool re-runs the SAME decision procedure through an INDEPENDENT executor: the LLMCompiler
transformer (C -> RV32I -> NISA -> bipolar tensor VM). We compile a tiny replica of what Lean emits
for the proposition, execute it on the transformer, and cross-check against a pure-Python oracle.

  Lean native_decide  --claims-->  P is true   (trusts GCC + CPU)
  Transformer         --computes-> the Bool     (independent silicon path)
  Python              --computes-> the truth     (independent implementation)

Agreement across all three is the audit passing. A DISAGREEMENT is a falsifier: it means the
compiled decision procedure Lean trusted returned something the math does not support — exactly the
kind of `native_decide` soundness gap this project hunts for.

Run:  python -m llmcompiler.lean_audit.example_audit
"""
from ..compiler.compiler import compile_and_run
from . import leanrt_source


def decide_mul_eq(a: int, b: int, c: int, *, device: str = "cpu") -> int:
    """Replica of what Lean emits for `decide (a * b = c)` over `Nat`:
    compute `Nat.decEq (Nat.mul a b) c` and return the resulting Bool (1/0) in a0.

    We build the Nat operands from decimal strings (arbitrary precision), multiply with the same
    limb algorithm the runtime uses, and compare — then hand the Bool back exactly as the compiled
    `Decidable` instance would.
    """
    src = leanrt_source("nat") + f"""
int _start(void) {{
    lean_object* a = lrt_of_dec("{a}");
    lean_object* b = lrt_of_dec("{b}");
    lean_object* c = lrt_of_dec("{c}");
    lean_object* prod = lean_nat_mul(a, b);
    return (int)lean_nat_dec_eq(prod, c);   /* the Bool native_decide would trust */
}}
"""
    res = compile_and_run(src, language="c", device=device, max_cycles=200000)
    return res.reg(10)  # a0 = return value


def audit(a: int, b: int, c: int, *, device: str = "cpu") -> dict:
    """Three-way audit of the proposition `a * b = c`."""
    transformer_bool = decide_mul_eq(a, b, c, device=device)
    python_truth = 1 if (a * b == c) else 0
    return {
        "proposition": f"{a} * {b} = {c}",
        "transformer": transformer_bool,   # independent execution of the decision procedure
        "python_oracle": python_truth,     # independent implementation of the math
        "agree": transformer_bool == python_truth,
    }


def main() -> None:
    # A mix of TRUE and FALSE propositions. A correct decision procedure must return 1 for the true
    # ones and 0 for the false ones; the false cases prove the check is not vacuously passing.
    cases = [
        (123456789, 987654321, 123456789 * 987654321),          # true
        (2**64 - 59, 2**64 - 83, (2**64 - 59) * (2**64 - 83)),  # true, ~128-bit
        (99999999999, 88888888888, 1),                          # false (product is huge)
        (7, 6, 42),                                             # true, small
        (7, 6, 41),                                             # false, off by one
    ]
    print(f"{'proposition':>52s} | transformer | oracle | verdict")
    print("-" * 92)
    all_ok = True
    for a, b, c in cases:
        r = audit(a, b, c)
        verdict = "AGREE" if r["agree"] else "*** DISAGREE (falsifier!) ***"
        all_ok &= r["agree"]
        prop = r["proposition"]
        prop = prop if len(prop) <= 50 else prop[:47] + "..."
        print(f"{prop:>52s} | {r['transformer']:^11d} | {r['python_oracle']:^6d} | {verdict}")
    print("-" * 92)
    print("AUDIT PASSED: transformer and oracle agree on every case."
          if all_ok else "AUDIT FOUND A DISAGREEMENT — investigate the compiled decision procedure.")


if __name__ == "__main__":
    main()
