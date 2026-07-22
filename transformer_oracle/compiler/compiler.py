"""
End-to-end compiler: C source → NISA → execution.

Pipeline:
  1. C source → RV32I assembly (via riscv64-linux-gnu-gcc -march=rv32i)
  2. RV32I assembly → parsed instructions (rv32i_parser)
  3. Parsed RV32I → NISA instructions (rv32i_to_nisa)
  4. NISA → execution (gpu_executor or reference executor)

For cases where the toolchain isn't available, also supports
compiling from RV32I assembly text directly.
"""

import subprocess
import tempfile
import os
from typing import Optional
from pathlib import Path

from ..core.nisa import Instruction, halt
from .rv32i_parser import parse_rv32i_assembly
from .rv32i_to_nisa import translate_program
from ..runtime.gpu_executor import gpu_execute, GPUExecutionResult
from ..runtime.executor import execute_program, ExecutionResult


def compile_c(source: str, gcc_path: str = "riscv64-linux-gnu-gcc") -> str:
    """Compile C source to RV32I assembly using GCC.

    Args:
        source: C source code string
        gcc_path: path to RISC-V GCC

    Returns:
        RV32I assembly text
    """
    with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as f:
        f.write(source)
        c_path = f.name

    asm_path = c_path.replace('.c', '.s')

    try:
        result = subprocess.run(
            [gcc_path, '-march=rv32im', '-mabi=ilp32',
             '-O1', '-S', '-o', asm_path,
             '-nostdlib', '-ffreestanding',
             '-fno-builtin', '-fno-stack-protector',
             '-fno-pic', '-fno-pie',
             '-fno-jump-tables',
             # Reserve x30/x31: the RV32I→NISA translator uses them as scratch
             # (_TMP1/_TMP2). Without this, gcc allocates t5/t6 (=x30/x31) for live
             # values in register-heavy functions and the translator clobbers them.
             '-ffixed-x30', '-ffixed-x31',
             c_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"GCC failed:\n{result.stderr}")

        with open(asm_path) as f:
            return f.read()
    finally:
        for p in [c_path, asm_path]:
            if os.path.exists(p):
                os.unlink(p)


def compile_asm(asm_source: str, add_halt: bool = True,
                stack_top: int = 252) -> list[Instruction]:
    """Compile RV32I assembly to NISA instructions.

    Args:
        asm_source: RV32I assembly text
        add_halt: append HALT instruction at end
        stack_top: initial stack pointer (word address)

    Returns:
        list of NISA instructions
    """
    nisa_instrs, _data = compile_asm_with_data(asm_source, add_halt=add_halt,
                                               stack_top=stack_top)
    return nisa_instrs


def compile_asm_with_data(asm_source: str, add_halt: bool = True,
                          stack_top: int = 252) -> tuple[list[Instruction], dict[int, int]]:
    """Like `compile_asm` but also returns the initialized-data image (byte-addr → byte)
    laid out for the program's globals (.data/.bss/.rodata)."""
    from .rv32i_parser import parse_rv32i_assembly_with_data
    rv_instrs, rv_labels, data_image = parse_rv32i_assembly_with_data(asm_source)
    nisa_instrs = translate_program(rv_instrs, rv_labels, stack_top=stack_top,
                                    data_image=data_image)

    if add_halt:
        nisa_instrs.append(halt())

    return nisa_instrs, data_image


def compile_and_run(source: str, *,
                    language: str = "asm",
                    device: str = "cuda",
                    max_cycles: int = 50000,
                    initial_memory: Optional[dict[int, int]] = None,
                    trace: bool = False,
                    gcc_path: str = "riscv64-linux-gnu-gcc",
                    stack_top: int = 252,
                    mem_size: Optional[int] = None,
                    ) -> GPUExecutionResult:
    """Compile and execute a program.

    Args:
        source: C source code or RV32I assembly text
        language: "c" for C source, "asm" for RV32I assembly
        device: "cuda" or "cpu"
        max_cycles: maximum execution cycles
        initial_memory: pre-initialized memory values
        trace: print execution trace
        gcc_path: path to RISC-V GCC

    Returns:
        GPUExecutionResult
    """
    if language == "c":
        asm = compile_c(source, gcc_path=gcc_path)
        nisa, data_image = compile_asm_with_data(asm, stack_top=stack_top)
    elif language == "asm":
        nisa, data_image = compile_asm_with_data(source, stack_top=stack_top)
    else:
        raise ValueError(f"Unknown language: {language}")

    # RV32I-compiled code is byte-addressed (SP and offsets are byte addresses). Build a
    # byte-addressed memory image: initialized globals first, then any caller-supplied
    # word-addressed initial_memory on top (back-compat semantics).
    from ..runtime.gpu_executor import _MEM_SIZE
    msize = mem_size if mem_size is not None else _MEM_SIZE
    mem = bytearray(msize)
    for addr, byte in data_image.items():
        if 0 <= addr < msize:
            mem[addr] = byte & 0xFF
    if initial_memory:
        for word_addr, val in initial_memory.items():
            byte_addr = word_addr * 4
            if 0 <= byte_addr + 3 < msize:
                val &= 0xFFFFFFFF
                mem[byte_addr] = val & 0xFF
                mem[byte_addr + 1] = (val >> 8) & 0xFF
                mem[byte_addr + 2] = (val >> 16) & 0xFF
                mem[byte_addr + 3] = (val >> 24) & 0xFF

    return gpu_execute(
        nisa,
        max_cycles=max_cycles,
        memory_bytes=mem,
        mem_size=msize,
        byte_addressed=True,
        device=device,
        trace=trace,
    )
