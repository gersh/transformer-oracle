# Formal paper

`llmcompiler-lean-audit.tex` — *An Independent Transformer Oracle for Auditing Lean 4's
`native_decide`*. It formalises the NISA operational semantics, the bipolar-transformer executor
(one-cycle faithfulness), the compilation pipeline and mini Lean runtime, and the audit's soundness
argument (a *sound but incomplete falsifier*), with a threat model and case studies.

## Build

Needs a TeX distribution (TeX Live / MiKTeX). Only standard packages are used.

```bash
pdflatex llmcompiler-lean-audit.tex
pdflatex llmcompiler-lean-audit.tex   # second pass resolves cross-references
```

This produces `llmcompiler-lean-audit.pdf`. On Debian/Ubuntu:
`apt install texlive-latex-base texlive-latex-extra`.
