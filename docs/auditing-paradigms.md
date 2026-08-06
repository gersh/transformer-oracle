# Auditing Paradigms & Falsification Methodology

`Transformer-Oracle` is built as an independent execution oracle to audit compiled code, kernel implementations, and decision procedures. It provides four distinct auditing paradigms designed to catch bugs, compiler miscompilations, and semantic divergences.

---

## 1. Differential Testing

In differential testing, the same algorithm or input is executed across two completely independent toolchains:

```
                  ┌──▶ Host CPU Execution (GCC / Native x86_64)
    Input Case ───┤
                  └──▶ Transformer VM (LLVM / RISC-V NISA Tensor VM)
                                 │
                                 ▼
                     [ Cross-Check Oracle ]
                  (Pass if Agree, Fail if Diverged)
```

* **Purpose**: Catch backend compiler bugs, undefined behavior (UB), and instruction encoding errors.
* **Key Asymmetry**: Agreement across both runs raises confidence; any disagreement **falsifies** at least one implementation.

---

## 2. Metamorphic Testing

Instead of relying on a pre-computed expected output, **metamorphic testing** verifies algebraic properties or invariants that must hold true across input variations:

### Example Invariants
* **Commutativity**: $f(a, b) = f(b, a)$
* **Reversibility**: $\text{decode}(\text{encode}(x)) = x$
* **Reduction Equivalence**: $K(\text{term}_A) \equiv K(\text{term}_B)$ for alpha-equivalent terms.

```python
from transformer_oracle.compiler.compiler import compile_and_run
from transformer_oracle.lean_audit import leanrt_source

# Metamorphic check: gcd(a, b) * lcm(a, b) == a * b
def test_gcd_lcm_identity(a, b):
    src = leanrt_source("nat") + f'''
    int _start(void) {{
        lean_object* va = lrt_of_dec("{a}");
        lean_object* vb = lrt_of_dec("{b}");
        lean_object* g  = lean_nat_gcd(va, vb);
        lean_object* l  = lean_nat_lcm(va, vb);
        lean_object* lhs = lean_nat_mul(g, l);
        lean_object* rhs = lean_nat_mul(va, vb);
        return (int)lean_nat_dec_eq(lhs, rhs);
    }}'''
    res = compile_and_run(src, language="c", device="cpu")
    assert res.reg(10) == 1, f"Metamorphic invariant failed for a={a}, b={b}"
```

---

## 3. Property-Based Automated Falsification (`lean_plausible`)

The repository includes `lean_plausible_audit`, which integrates property-based testing (QuickCheck-style randomized generation):

* **Generators**: Automatically generates thousands of arbitrary inductive data structures (e.g. deeply nested ASTs, large random `Nat`/`Int` instances).
* **Fuzzing Engine**: Feeds generated instances into the Transformer VM while monitoring bounds, stack depth, and runtime assertions.

---

## 4. Auditing `native_decide` & Untrusted `@[extern]` Code

Lean 4's `native_decide` tactic trusts `@[extern]` C/Rust primitives compiled into native binaries. A memory safety bug or subtle sign-extension error in an extern primitive can cause Lean to **accept a false proof**.

### Auditing Workflow
1. Isolate the `@[extern]` function under audit.
2. Compile it against `transformer_oracle`'s mini Lean runtime.
3. Run the exact binary on the analytical Transformer VM across boundary values (0, 1, $2^{31}-1$, $2^{31}$, $2^{64}-1$).
4. Compare execution against an independent pure Python or interval-arithmetic oracle.
