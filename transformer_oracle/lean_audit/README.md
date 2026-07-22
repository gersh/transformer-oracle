# `lean_audit` — Lean `native_decide` auditing toolkit

Run Lean 4 decision procedures on the Transformer-Oracle transformer as an **independent execution oracle**,
to cross-check `native_decide` verdicts. See the top-level
[docs/auditing-lean-native-decide.md](../../docs/auditing-lean-native-decide.md) for the full guide
and [docs/paper](../../docs/paper) for the mathematics.

## Contents

- `runtime/` — a freestanding **mini Lean runtime** that implements enough of the Lean object ABI to
  compile emitted C through the `C → RV32I → NISA → transformer` pipeline:
  - `leanrt_nat.h` — arbitrary-precision `Nat` (tagged scalars + base-`2^32` limb bignums).
  - `leanrt_int.h` — arbitrary-precision `Int` (sign + `Nat` magnitude, Euclidean `ediv`/`emod`).
  - `leanrt_ctor.h` — boxed inductives (`List`, `Option`, `Prod`, records).
- `example_audit.py` — a runnable three-way audit (`native_decide`-style procedure vs. transformer
  vs. Python oracle) of `decide (a * b = c)` over `Nat`.

## Quick start

```bash
python -m transformer_oracle.lean_audit.example_audit
```

```python
from transformer_oracle.lean_audit.example_audit import audit
audit(7, 6, 42)     # -> {'transformer': 1, 'python_oracle': 1, 'agree': True}
audit(7, 6, 41)     # -> {'transformer': 0, 'python_oracle': 0, 'agree': True}  (false prop)
```

## Writing your own audit

Assemble the runtime with `leanrt_source(level)`, write a `_start()` that computes the `Bool` (or
value) the `Decidable` instance would compute, and run it on the transformer:

```python
from transformer_oracle.lean_audit import leanrt_source
from transformer_oracle.compiler.compiler import compile_and_run

# Audit `decide (2^40 % 7 = 4)` over Nat (Lean: Nat.decEq (Nat.mod (2^40) 7) 4).
src = leanrt_source("nat") + '''
int _start(void){
    lean_object* lhs = lean_nat_mod( lean_nat_pow(lrt_box(2), lrt_box(40)), lrt_box(7) );
    return (int) lean_nat_dec_eq(lhs, lrt_box(4));   // the Bool native_decide would trust
}'''
res = compile_and_run(src, language="c", device="cpu", max_cycles=300000)
assert res.reg(10) == (1 if (2**40) % 7 == 4 else 0)   # transformer == Python oracle (== 0; it's false)
```

Requires `riscv64-linux-gnu-gcc` on `PATH` and PyTorch. Use `device="cuda"` for large runs.

### Runtime coverage status

`leanrt_source("nat")` (`Nat`: add/sub/mul/div/mod/pow/land, comparisons, `dec_eq`/`dec_lt`/`dec_le`)
is validated end-to-end on the transformer. The `int` and `ctor` layers are validated against the
**host reference build** (`gcc -I runtime`), and a subset runs on the transformer, but a known
pipeline miscompilation currently affects some *register-heavy composed* `Int` routines (e.g. the
negative-dividend Euclidean `emod` branch) — the same codegen class as the two compiler bugs fixed
earlier in this project. When auditing `Int`/inductive procedures, **cross-check against the host
reference build** until that is resolved. This is itself a live demonstration of the method: the
differential audit caught a defect in the compiler pipeline.
