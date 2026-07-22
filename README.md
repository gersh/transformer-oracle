# Transformer-Oracle

**An independent execution oracle for auditing Lean 4's `native_decide` — built by compiling programs
into transformer weights.**

Lean's `native_decide` closes a proof goal by *compiling* its `Decidable` instance to native code,
running it, and trusting that the machine answered `true`. That trust pulls the C compiler, the
standard-library `@[extern]`/`@[implemented_by]` fast paths, **and the physical CPU** into the trusted
base (via the `ofReduceBool` / `trustCompiler` axioms). A miscompilation, a buggy fast path, or a
silent hardware fault can make `native_decide` "prove" something false.

Transformer-Oracle re-runs the *same decision procedure* through a completely independent executor — a
transformer whose weights are **constructed** (not trained) so that one forward pass performs one CPU
cycle, computing in a bipolar `{−1,+1}` tensor encoding on a GPU — and cross-checks it against a pure
oracle. **A disagreement is a falsifier:** proof that the compiled procedure Lean trusted returned
something the mathematics does not support.

> ### 📄 Read the paper
> **[Transformer-Oracle: An Independent Transformer Oracle for Auditing Lean 4's `native_decide`](docs/paper/transformer-oracle-lean-audit.pdf)** (PDF)
> — NISA operational semantics, the bipolar-transformer executor (one-cycle faithfulness), the
> compilation pipeline and mini Lean runtime, and the audit's soundness argument (a *sound but
> incomplete falsifier*), with threat model and case studies. Source: [`.tex`](docs/paper/transformer-oracle-lean-audit.tex).

```
   .lean ──emit C──▶ Lean runtime ─┐
                                    ├─▶  C ──gcc(rv32im)──▶ RV32I ──▶ NISA ──▶ Transformer
   (your decision procedure) ───────┘                                         (bipolar tensor VM)
                                                                                    │
                          Python / sympy / mpmath  ◀── cross-check ────────────────┘
```

## Auditing Lean — start here

The audit is three independent verdicts on a decidable proposition `P`: what `native_decide` claims,
what the transformer computes running the *same* algorithm, and what a pure oracle says. Agreement is
the audit passing; any disagreement localizes a real fault.

```bash
pip install torch                       # CPU is enough
apt install gcc-riscv64-linux-gnu       # the C front-end's cross-compiler
python -m transformer_oracle.lean_audit.example_audit
```

```
          123456789 * 987654321 = 121932631112635269 |      1      |   1    | AGREE
  18446744073709551557 * 18446744073709551533 = 3... |      1      |   1    | AGREE   (~128-bit)
                       99999999999 * 88888888888 = 1 |      0      |   0    | AGREE   (false prop)
                                          7 * 6 = 41 |      0      |   0    | AGREE   (false prop)
AUDIT PASSED: transformer and oracle agree on every case.
```

Audit `decide (a * b = c)` over `Nat` — the transformer runs `Nat.decEq (Nat.mul a b) c`, exactly what
`native_decide` would compile, while Python is the oracle:

```python
from transformer_oracle.lean_audit.example_audit import audit
audit(2**64 - 59, 2**64 - 83, (2**64 - 59) * (2**64 - 83))
# {'proposition': '...', 'transformer': 1, 'python_oracle': 1, 'agree': True}
```

To audit your own module, express the decision procedure against the bundled mini Lean runtime and run
it on the transformer:

```python
from transformer_oracle.lean_audit import leanrt_source
from transformer_oracle.compiler.compiler import compile_and_run

# decide (2^40 % 7 = 4)  ==  Nat.decEq (Nat.mod (2^40) 7) 4
src = leanrt_source("nat") + '''
int _start(void){
    lean_object* lhs = lean_nat_mod( lean_nat_pow(lrt_box(2), lrt_box(40)), lrt_box(7) );
    return (int) lean_nat_dec_eq(lhs, lrt_box(4));   // the Bool native_decide would trust
}'''
res = compile_and_run(src, language="c", device="cpu", max_cycles=300000)
assert res.reg(10) == (1 if (2**40) % 7 == 4 else 0)   # transformer == oracle
```

**Full guide:** [docs/auditing-lean-native-decide.md](docs/auditing-lean-native-decide.md) — the trust
surface, the differential/metamorphic/redundancy methodology, a recipe for your own modules, and the
honest limitations. **Toolkit reference:** [`transformer_oracle/lean_audit/`](transformer_oracle/lean_audit/).

## How the transformer executes code

The C front-end lowers C (including Lean's emitted C) to **NISA**, a 40-opcode RISC-V-derived ISA, via
`riscv64-linux-gnu-gcc` and an RV32I→NISA translator. The executor is a 10-layer transformer whose
weights are analytically constructed so that each forward pass performs exactly one NISA instruction:
register reads are attention one-hot selections, the ALU is a bipolar `{−1,+1}` arithmetic circuit,
opcode dispatch is branchless one-hot masking, and an error-correction step snaps values back to
`{−1,+1}` each cycle. The result is an *exact*, independent re-execution of the program — see the
[paper](docs/paper/transformer-oracle-lean-audit.pdf) for the operational semantics and the one-cycle
faithfulness argument.

Run any program on it directly:

```python
from transformer_oracle.compiler.compiler import compile_and_run
src = "int _start(void){ int s=0; for(int i=1;i<=100;i++) s+=i; return s; }"
print(compile_and_run(src, language="c", device="cpu").reg(10))   # 5050
```

## Install

Requires Python 3.10+ with [PyTorch](https://pytorch.org) (CPU suffices; CUDA accelerates large runs)
and a RISC-V cross-compiler (`apt install gcc-riscv64-linux-gnu`).

```bash
git clone https://github.com/gersh/transformer-oracle && cd transformer-oracle
pip install torch pytest
python -m pytest transformer_oracle/tests -q
```

The package imports as `transformer_oracle` from the repository root.

## Repository layout

| Path | Contents |
|------|----------|
| `transformer_oracle/lean_audit/` | **Lean auditing toolkit**: a freestanding mini Lean runtime + worked audit example. |
| `transformer_oracle/core/` | The NISA ISA: `Opcode` (40 ops), `Instruction`, machine `state`. |
| `transformer_oracle/compiler/` | `compile_c` (C→RV32I via gcc), RV32I parser, RV32I→NISA translator, `compile_and_run`. |
| `transformer_oracle/weights/` | Analytical weight builder — constructs transformer weights that execute NISA. |
| `transformer_oracle/runtime/` | Executors: pure-tensor / GPU / CUDA transformer VMs, plus reference interpreters. |
| `transformer_oracle/tests/` | Compiler, executor, and end-to-end program tests. |
| `docs/` | The how-to guide and the [formal paper](docs/paper/). |

## Status

The analytical transformer faithfully executes the full 40-opcode NISA ISA; the C→RV32I→NISA pipeline
compiles real programs; and the Lean-audit toolkit runs arbitrary-precision `Nat` decision procedures
end-to-end on the transformer (`Int`/inductive layers are validated against a host reference build,
with one known composed-`Int` miscompilation documented). Applying the same oracle/metamorphic method
found two compiler bugs in this project (both fixed) and, pointed externally, real defects in unrelated
mature libraries — see the paper's results section.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
