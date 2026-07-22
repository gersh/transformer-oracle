# Formal paper

`transformer-oracle-lean-audit.tex` — *An Independent Transformer Oracle for Auditing Lean 4's
`native_decide`*. It formalises the NISA operational semantics, the bipolar-transformer executor
(one-cycle faithfulness), the compilation pipeline and mini Lean runtime, and the audit's soundness
argument (a *sound but incomplete falsifier*), with a threat model and case studies.

## Build

Needs a TeX distribution (TeX Live / MiKTeX). Only standard packages are used.

```bash
pdflatex transformer-oracle-lean-audit.tex
pdflatex transformer-oracle-lean-audit.tex   # second pass resolves cross-references
```

This produces `transformer-oracle-lean-audit.pdf`. On Debian/Ubuntu:
`apt install texlive-latex-base texlive-latex-extra`.
