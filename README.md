# LLMCompiler

**Compiling ordinary programs into transformer weights — and using the result as an independent
execution oracle to audit Lean 4's `native_decide`.**

LLMCompiler takes C or RISC-V assembly, lowers it to a small 40-opcode ISA called **NISA**, and
executes it on a **transformer** whose weights are constructed (not trained) so that one forward pass
performs exactly one CPU cycle. Because the executor is a completely different implementation of
computation from a normal CPU — arithmetic runs in a bipolar `{−1, +1}` tensor encoding on a GPU —
it makes an excellent *independent oracle*. That is the basis of this project's headline application:
**checking that Lean's `native_decide` really computed what it claims to have computed.**

```
   C  ──gcc(rv32im)──▶  RV32I  ──translate──▶  NISA  ──build weights──▶  Transformer
  .lean ─emit C─▶ Lean runtime ─┘                                          (bipolar tensor VM)
```

## Why this matters for Lean

`native_decide` closes a goal by *compiling* its `Decidable` instance to native code, running it, and
trusting that the machine answered `true`. This is fast and powerful, but it moves the C compiler,
the standard-library `@[extern]`/`@[implemented_by]` fast paths, **and the physical CPU** into the
trusted base (via the `ofReduceBool` / `trustCompiler` axioms). A miscompilation, a buggy fast path,
or a silent hardware fault can make `native_decide` "prove" a falsehood.

This repository re-runs the *same decision procedure* through a totally independent executor and
cross-checks it against a pure-oracle. **A disagreement is a falsifier** — proof that the compiled
procedure Lean trusted returned something the mathematics does not support.

See:
- **[docs/auditing-lean-native-decide.md](docs/auditing-lean-native-decide.md)** — the practical
  guide: the trust surface, the methodology, and how to audit your own Lean modules with the tools
  here.
- **[docs/paper/llmcompiler-lean-audit.tex](docs/paper/llmcompiler-lean-audit.tex)** — the formal
  paper: NISA operational semantics, the bipolar-transformer executor, pipeline correctness, the
  mini Lean runtime, and the soundness argument for the audit method.

## Install

Requirements:
- Python 3.10+ with [PyTorch](https://pytorch.org) (CPU is enough; CUDA accelerates large runs).
- A RISC-V cross-compiler for the C front-end: `riscv64-linux-gnu-gcc`
  (`apt install gcc-riscv64-linux-gnu`).

```bash
git clone <this-repo> llmcompiler && cd llmcompiler
pip install torch pytest
python -m pytest llmcompiler/tests -q          # run the test suite
```

The package imports as `llmcompiler` from the repository root.

## Quickstart

Compile and run a C program on the transformer:

```python
from llmcompiler.compiler.compiler import compile_and_run

src = "int _start(void){ int s=0; for(int i=1;i<=100;i++) s+=i; return s; }"
res = compile_and_run(src, language="c", device="cpu")
print(res.reg(10))   # a0 = 5050
```

Audit a Lean-style decision procedure (three-way: transformer vs. oracle):

```bash
python -m llmcompiler.lean_audit.example_audit
```

```
          123456789 * 987654321 = 121932631112635269 |      1      |   1    | AGREE
  18446744073709551557 * 18446744073709551533 = 3... |      1      |   1    | AGREE
                       99999999999 * 88888888888 = 1 |      0      |   0    | AGREE
...
AUDIT PASSED: transformer and oracle agree on every case.
```

## Repository layout

| Path | Contents |
|------|----------|
| `llmcompiler/core/` | The NISA ISA: `Opcode` (40 ops), `Instruction`, machine `state`. |
| `llmcompiler/compiler/` | `compile_c` (C→RV32I via gcc), RV32I parser, RV32I→NISA translator, `compile_and_run`. |
| `llmcompiler/weights/` | Analytical weight builder — constructs transformer weights that execute NISA. |
| `llmcompiler/runtime/` | Executors: pure-tensor / GPU / CUDA transformer VMs, plus reference interpreters. |
| `llmcompiler/lean_audit/` | **Lean auditing toolkit**: a freestanding mini Lean runtime + worked audit example. |
| `llmcompiler/tests/` | Compiler, executor, and end-to-end program tests. |
| `docs/` | The how-to guide and the formal paper. |

## Status

Phase 1 complete: the analytical transformer faithfully executes the full 40-opcode NISA ISA;
the C→RV32I→NISA pipeline compiles real programs; the Lean-audit toolkit runs arbitrary-precision
`Nat`/`Int` and structural-recursion decision procedures on the transformer. Two compiler bugs found
and fixed during auditing (global read-modify-write indexing; `x30`/`x31` scratch-register
collision). Applying the same oracle/metamorphic method externally has surfaced real defects in other
mature systems (e.g. an `fmpz_sqrtmod` infinite loop in FLINT); see the paper's results section.

## License

See `LICENSE` (or add one before publishing).
