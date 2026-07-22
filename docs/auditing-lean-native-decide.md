# Auditing Lean 4 `native_decide` with the Transformer Oracle

This guide explains how to use LLMCompiler to audit uses of `native_decide` in Lean code and modules:
what the trust gap is, the auditing methodology, and how to run a decision procedure on the
transformer as an independent check. For the underlying mathematics see the
[formal paper](paper/llmcompiler-lean-audit.tex).

---

## 1. What `native_decide` trusts, and why to audit it

To prove a decidable proposition `P`, Lean offers two tactics:

- `decide` — reduces the `Decidable P` instance **inside the trusted kernel**. Slow, but the only
  thing trusted is the kernel.
- `native_decide` — **compiles** the `Decidable P` instance to C, then to a native binary, runs it,
  and if it returns `isTrue` accepts `P`. Orders of magnitude faster, but it adds three things to the
  trusted computing base:
  1. the **Lean → C compiler** and the C toolchain (GCC/Clang);
  2. every `@[extern]` / `@[implemented_by]` **fast path** the procedure touches (hand-written C or
     alternative Lean that replaces the "reference" definition at runtime);
  3. the **physical CPU** that runs the binary.

This trust is discharged by the axioms `Lean.ofReduceBool` and `Lean.trustCompiler`. If any of the
three is wrong for the specific computation, `native_decide` can accept a **false** proposition.
The known-bug pattern is (2): a `*Fast`/`*Unsafe` `@[implemented_by]` replacement that disagrees with
its reference definition at an edge case (e.g. `Vector.scanr`, an off-by-one in a foldl fast path).

> **Key asymmetry (Dijkstra).** Re-running a decision procedure and getting the *same* answer proves
> nothing — it can share the same bug. Getting a *different* answer is a **falsifier**: it proves at
> least one path is wrong. Audits hunt for disagreements; a clean audit is evidence, not proof.

---

## 2. The method

We treat the compiled decision procedure as an object to be differentially tested. For a proposition
`P` with `Decidable` instance `d`, we obtain **three independent verdicts**:

| Source | What it is | What it trusts |
|--------|-----------|----------------|
| `native_decide` | Lean's compiled `d`, run on the CPU | GCC + fast paths + CPU |
| **Transformer** | the *same algorithm* re-run on LLMCompiler's bipolar tensor VM | a disjoint compiler + a different executor + (a different device) |
| **Oracle** | a pure, independent implementation of the mathematics (Python `int`, `sympy`, `mpmath`, or Lean `decide` itself) | its own small code |

Agreement across all three is the audit passing. Any disagreement is investigated: it localizes to
the compiled procedure, the transformer path, or the oracle — and the first is exactly a
`native_decide` soundness gap.

Four complementary tactics, in rough order of power:

1. **Differential** — run `d` on the transformer, compare to the oracle on many inputs, heavy on
   edge cases (0, 1, boundaries `2^31`/`2^32`/`2^63`, negatives, empty structures).
2. **Metamorphic** — check algebraic identities that must hold regardless of the specific value
   (`a*b = b*a`, `gcd(a,b)·lcm(a,b) = a·b`, `factor(x)` multiplies back to `x`). No oracle needed —
   the identity *is* the oracle.
3. **Bounded-exhaustive** — for finite domains, enumerate every case.
4. **Cross-architecture / cross-version redundancy** — run the *same* compiled binary on aarch64 and
   amd64 (and multiple Lean versions). Deterministic integer ops must agree bit-for-bit; a difference
   is a codegen or (rarely) a hardware fault. This is the only defense against silent data corruption
   ("mercurial cores").

---

## 3. How the transformer becomes the oracle

Lean emits **C** for a decision procedure (`lean --c` or the `.c` files in a build's
`build/ir/`). That C uses the Lean runtime ABI: `lean_object*` values, `lean_nat_*`, `lean_int_*`,
constructor accessors, etc. LLMCompiler ships a **freestanding mini Lean runtime**
(`llmcompiler/lean_audit/runtime/`) that implements just enough of that ABI to compile and run the
emitted C through the `C → RV32I → NISA → transformer` pipeline:

| Header | Provides | Notes |
|--------|----------|-------|
| `leanrt_nat.h` | arbitrary-precision `Nat` | tagged scalars for `Nat < 2^31`; heap bignums as base-`2^32` limbs; add/sub/mul/divmod/mod/pow/land, `dec_eq`/`dec_lt`/`dec_le`. |
| `leanrt_int.h` | arbitrary-precision `Int` | sign + `Nat` magnitude; Euclidean `ediv`/`emod` matching Lean's semantics; `natAbs`, comparisons. |
| `leanrt_ctor.h` | boxed inductives | `List`, `Option`, `Prod`, records via tagged constructor objects — unlocks structural-recursion procedures. |

The object model matches Lean's ABI as far as the emitted C observes it: a `lean_object*` is either a
tagged scalar (low bit 1) or a heap pointer into a bump arena. Only the `lean_*` functions inspect the
private heap layout, so the runtime is a drop-in for the fragment of the ABI that decision procedures
use. Correctness of every algorithm (bignum long division, Euclidean `emod`, etc.) is asserted against
the oracle as part of auditing.

---

## 4. Worked example

`llmcompiler/lean_audit/example_audit.py` audits `decide (a * b = c)` over `Nat`. Lean would compile
this to `Nat.decEq (Nat.mul a b) c : Bool`; we replicate that with the runtime, run it on the
transformer, and compare to Python.

```python
from llmcompiler.lean_audit.example_audit import audit
print(audit(2**64 - 59, 2**64 - 83, (2**64 - 59) * (2**64 - 83)))
# {'proposition': '... = ...', 'transformer': 1, 'python_oracle': 1, 'agree': True}
```

Run the whole demo (note the deliberate *false* cases — they prove the check is not vacuously
passing):

```bash
python -m llmcompiler.lean_audit.example_audit
```

Under the hood (`decide_mul_eq`):

```python
src = leanrt_nat_header + '''
int _start(void){
    lean_object* a = lrt_of_dec("...");
    lean_object* b = lrt_of_dec("...");
    lean_object* c = lrt_of_dec("...");
    return (int)lean_nat_dec_eq(lean_nat_mul(a,b), c);  // the Bool native_decide would trust
}'''
res = compile_and_run(src, language="c", device="cpu")   # runs on the transformer
verdict = res.reg(10)                                    # a0 = 1/0
```

---

## 5. Auditing your own module — recipe

1. **Locate the `native_decide` calls** and, for each, the proposition `P` and its `Decidable`
   instance. Identify which standard-library functions the instance reduces through — especially any
   with `@[extern]` or `@[implemented_by]` (that is where bugs live).
2. **Get the compiled algorithm.** Either read Lean's emitted C for the instance, or re-express the
   decision procedure directly in C against the mini runtime (as in the example). Keep it faithful to
   what Lean actually runs, not to the paper definition.
3. **Pick an oracle.** Python arbitrary-precision `int` for `Nat`/`Int`; `sympy` for number
   theory/polynomials; `mpmath` (with interval checks) for reals; or Lean's own `decide` for small
   cases. The oracle must be an *independent* implementation.
4. **Generate inputs** — random + adversarial edges + (if finite) exhaustive. For metamorphic checks,
   generate identity instances instead of input/output pairs.
5. **Run and diff.** Compile each case with `compile_and_run(..., language="c")`, read `reg(10)`,
   compare to the oracle. Log every disagreement with its inputs.
6. **For determinism-critical results, add redundancy** — run the binary under QEMU on a second
   architecture (`docker run --platform linux/amd64 ...`) and across Lean versions; require bit-exact
   agreement on integer outputs. (Floating-point transcendentals are *not* portable — treat only
   IEEE `+ − × ÷ √` as reproducible; `sin`/`exp`/`pow` depend on the platform `libm`.)

---

## 6. Limitations and honest scope

- **A clean audit is not a proof.** It raises confidence proportional to input coverage; it never
  certifies. Only `decide` (kernel reduction) or a `Prop`-level proof gives certainty.
- **Shared-bug blind spot.** If the transformer path and Lean's path happen to share the identical
  defect, differential testing cannot see it. Metamorphic and cross-implementation oracles mitigate
  this; nothing eliminates it entirely.
- **Coverage of the ABI.** The mini runtime covers `Nat`, `Int`, and common inductives. Procedures
  that lean on `Float`, `Array` mutation, `String`, or exotic externs need the corresponding runtime
  piece added (the pipeline supports it; the header just has to exist).
- **Bounded computation.** The transformer runs a fixed cycle budget and a bump arena; very heavy
  procedures must be scaled (`max_cycles`, arena size) or reduced to a representative core.

For the formal statement of what a disagreement proves — and the threat model including silent
hardware faults — see the [paper](paper/llmcompiler-lean-audit.tex), §6–7.
