"""Lean `native_decide` auditing toolkit.

Run Lean 4 decision procedures on the Transformer-Oracle transformer as an independent execution oracle,
to cross-check `native_decide` verdicts. See ``docs/auditing-lean-native-decide.md``.

The public helper here assembles the freestanding mini Lean runtime as a single C source string.
``compile_and_run`` compiles a lone source string (no include path), so the runtime headers — which
``#include`` one another — must be flattened before use. ``leanrt_source`` does exactly that:

    >>> from transformer_oracle.lean_audit import leanrt_source
    >>> src = leanrt_source("int") + 'int _start(void){ ... }'
    >>> from transformer_oracle.compiler.compiler import compile_and_run
    >>> compile_and_run(src, language="c", device="cpu").reg(10)
"""
import os
import re

_RUNTIME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime")

# Dependency order: each header pulls in the ones before it. "level" picks how much to include.
_LAYERS = {
    "nat": ["leanrt_nat.h"],
    "int": ["leanrt_nat.h", "leanrt_int.h"],
    "ctor": ["leanrt_nat.h", "leanrt_int.h", "leanrt_ctor.h"],
}

_LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*"leanrt_\w+\.h"\s*$', re.MULTILINE)


def leanrt_source(level: str = "nat") -> str:
    """Return the mini Lean runtime up to ``level`` (``"nat"`` | ``"int"`` | ``"ctor"``) as one
    self-contained C source string, with the internal ``#include "leanrt_*.h"`` lines removed so it
    compiles from a single temp file. Append your ``int _start(void){...}`` and hand it to
    ``compile_and_run(src, language="c")``.
    """
    if level not in _LAYERS:
        raise ValueError(f"level must be one of {sorted(_LAYERS)}; got {level!r}")
    parts = []
    for name in _LAYERS[level]:
        with open(os.path.join(_RUNTIME_DIR, name)) as f:
            parts.append(_LOCAL_INCLUDE.sub("", f.read()))
    return "\n".join(parts)


__all__ = ["leanrt_source"]
