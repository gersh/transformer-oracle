"""
GPU Executor: runs NISA programs on GPU using vectorized bipolar operations.

This executor keeps registers on GPU as bipolar tensors and uses
fast_bipolar operations for all arithmetic. Memory is byte-addressed
(stored as a flat integer array for speed).

Supports:
- All ALU operations (ADD, SUB, MUL, AND, OR, XOR, NOT, shifts, SLT)
- Word and byte/halfword memory (LOAD, STORE, LOADB, STOREB, LOADH, STOREH)
- Direct and indirect jumps (JMP, JMPR)
- All branch types (BEQ, BNE, BLT, BGE, BLTU, BGEU)
"""

import torch
from typing import Optional

from ..core.state import (
    StateTensor, StateConfig, DEFAULT_CONFIG,
    VALUE_START, VALUE_END, VALUE_BITS,
    int_to_bipolar, bipolar_to_int,
)
from ..core.nisa import Instruction, Opcode
from ..weights.fast_bipolar import (
    bp_to_int, int_to_bp,
    fast_add, fast_sub, fast_mul,
    fast_and, fast_or, fast_xor, fast_not,
    fast_shl_var, fast_shr_var, fast_sra_var,
    fast_slt, fast_sltu,
    snap_to_bipolar,
)


class GPUExecutionResult:
    """Result from GPU execution."""

    def __init__(self, registers: dict[int, int], pc: int,
                 cycles: int, halted: bool,
                 memory: Optional[bytearray] = None):
        self._registers = registers
        self._pc = pc
        self.cycles = cycles
        self.halted = halted
        self._memory = memory or bytearray()

    def reg(self, idx: int) -> int:
        return self._registers.get(idx, 0)

    @property
    def registers(self) -> dict[int, int]:
        return self._registers

    def mem_word(self, byte_addr: int) -> int:
        """Read a 32-bit word from byte address (little-endian)."""
        m = self._memory
        if byte_addr + 3 < len(m):
            return m[byte_addr] | (m[byte_addr+1]<<8) | (m[byte_addr+2]<<16) | (m[byte_addr+3]<<24)
        return 0

    def mem_byte(self, addr: int) -> int:
        if addr < len(self._memory):
            return self._memory[addr]
        return 0

    def __repr__(self):
        status = "HALTED" if self.halted else "RUNNING"
        return f"GPUExecutionResult({status}, cycles={self.cycles}, PC={self._pc})"


_MOD32 = 2**32
_MEM_SIZE = 65536  # 64KB default byte-addressable memory


def gpu_execute(
    instructions: list[Instruction],
    max_cycles: int = 10000,
    initial_registers: Optional[dict[int, int]] = None,
    initial_memory: Optional[dict[int, int]] = None,
    memory_bytes: Optional[bytearray] = None,
    mem_size: int = _MEM_SIZE,
    device: str = 'cuda',
    trace: bool = False,
    byte_addressed: Optional[bool] = None,
) -> GPUExecutionResult:
    """Execute a NISA program on GPU.

    Args:
        instructions: NISA program
        max_cycles: max instruction cycles
        initial_registers: {reg_idx: value}
        initial_memory: {word_addr: word_value} — for backward compat, word-addressed
        memory_bytes: raw byte-addressed memory (overrides initial_memory)
        mem_size: size of byte-addressable memory
        device: 'cuda' or 'cpu'
        trace: print each cycle

    Returns:
        GPUExecutionResult
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'

    dev = torch.device(device)
    dtype = torch.float64

    # ── Initialize registers on GPU ──
    regs = torch.full((32, 32), -1.0, dtype=dtype, device=dev)
    regs[0] = int_to_bipolar(0).to(dev)

    if initial_registers:
        for idx, val in initial_registers.items():
            if idx > 0:
                regs[idx] = int_to_bipolar(val).to(dtype=dtype, device=dev)

    # ── Initialize byte-addressed memory ──
    mem = bytearray(mem_size)

    if memory_bytes is not None:
        mem[:len(memory_bytes)] = memory_bytes
    elif initial_memory:
        # Backward compat: word-addressed initial_memory → write as LE words
        for word_addr, val in initial_memory.items():
            byte_addr = word_addr * 4
            if byte_addr + 3 < mem_size:
                val = val & 0xFFFFFFFF
                mem[byte_addr] = val & 0xFF
                mem[byte_addr+1] = (val >> 8) & 0xFF
                mem[byte_addr+2] = (val >> 16) & 0xFF
                mem[byte_addr+3] = (val >> 24) & 0xFF

    # Auto-detect addressing mode: if memory_bytes is provided, use byte addressing
    # (RV32I-compiled code). Otherwise use word addressing (old NISA programs).
    if byte_addressed is None:
        byte_addressed = (memory_bytes is not None)

    # Pre-decode
    decoded = _predecode(instructions)

    pc = 0
    n_instr = len(instructions)
    cycles = 0
    halted = False

    # ── Execution loop ──
    for cycle in range(max_cycles):
        if pc >= n_instr:
            halted = True
            break

        op, a, b, c, imm = decoded[pc]

        if trace:
            print(f"  cycle {cycle:4d}: PC={pc:3d}  {instructions[pc]}")

        next_pc = pc + 1

        if op == Opcode.HALT:
            halted = True
            cycles = cycle + 1
            break

        elif op == Opcode.NOP:
            pass

        elif op == Opcode.MOVI:
            if a != 0:
                regs[a] = int_to_bp(torch.tensor(imm, dtype=dtype, device=dev))

        elif op == Opcode.MOV:
            if a != 0:
                regs[a] = regs[b].clone()

        elif op == Opcode.ADD:
            if a != 0:
                regs[a] = fast_add(regs[b], regs[c])

        elif op == Opcode.SUB:
            if a != 0:
                regs[a] = fast_sub(regs[b], regs[c])

        elif op == Opcode.MUL:
            if a != 0:
                regs[a] = fast_mul(regs[b], regs[c])

        elif op == Opcode.AND:
            if a != 0:
                regs[a] = fast_and(regs[b], regs[c])

        elif op == Opcode.OR:
            if a != 0:
                regs[a] = fast_or(regs[b], regs[c])

        elif op == Opcode.XOR:
            if a != 0:
                regs[a] = fast_xor(regs[b], regs[c])

        elif op == Opcode.NOT:
            if a != 0:
                regs[a] = fast_not(regs[b])

        elif op == Opcode.SHL:
            if a != 0:
                regs[a] = fast_shl_var(regs[b], regs[c])

        elif op == Opcode.SHR:
            if a != 0:
                regs[a] = fast_shr_var(regs[b], regs[c])

        elif op == Opcode.SRA:
            if a != 0:
                regs[a] = fast_sra_var(regs[b], regs[c])

        elif op == Opcode.SLT:
            if a != 0:
                regs[a] = fast_slt(regs[b], regs[c])

        elif op == Opcode.SLTU:
            if a != 0:
                regs[a] = fast_sltu(regs[b], regs[c])

        elif op == Opcode.DIV:
            if a != 0:
                bv = bp_to_int(regs[b]).item()
                cv = bp_to_int(regs[c]).item()
                bs = int(bv) - _MOD32 if bv >= 2**31 else int(bv)
                cs = int(cv) - _MOD32 if cv >= 2**31 else int(cv)
                res = 0xFFFFFFFF if cs == 0 else int(abs(bs) // abs(cs)) * (1 if (bs < 0) == (cs < 0) else -1)
                regs[a] = int_to_bp(torch.tensor(res & 0xFFFFFFFF, dtype=dtype, device=dev))

        elif op == Opcode.DIVU:
            if a != 0:
                bv = int(bp_to_int(regs[b]).item())
                cv = int(bp_to_int(regs[c]).item())
                res = 0xFFFFFFFF if cv == 0 else bv // cv
                regs[a] = int_to_bp(torch.tensor(res & 0xFFFFFFFF, dtype=dtype, device=dev))

        elif op == Opcode.REM:
            if a != 0:
                bv = bp_to_int(regs[b]).item()
                cv = bp_to_int(regs[c]).item()
                bs = int(bv) - _MOD32 if bv >= 2**31 else int(bv)
                cs = int(cv) - _MOD32 if cv >= 2**31 else int(cv)
                res = int(bv) if cs == 0 else (abs(bs) % abs(cs)) * (1 if bs >= 0 else -1)
                regs[a] = int_to_bp(torch.tensor(res & 0xFFFFFFFF, dtype=dtype, device=dev))

        elif op == Opcode.REMU:
            if a != 0:
                bv = int(bp_to_int(regs[b]).item())
                cv = int(bp_to_int(regs[c]).item())
                res = bv if cv == 0 else bv % cv
                regs[a] = int_to_bp(torch.tensor(res & 0xFFFFFFFF, dtype=dtype, device=dev))

        elif op == Opcode.MULHU:
            if a != 0:
                bv = int(bp_to_int(regs[b]).item())
                cv = int(bp_to_int(regs[c]).item())
                res = (bv * cv) >> 32
                regs[a] = int_to_bp(torch.tensor(res & 0xFFFFFFFF, dtype=dtype, device=dev))

        elif op == Opcode.MULH:
            if a != 0:
                bv = int(bp_to_int(regs[b]).item())
                cv = int(bp_to_int(regs[c]).item())
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                cs = cv - 0x100000000 if cv >= 0x80000000 else cv
                res = (bs * cs) >> 32
                regs[a] = int_to_bp(torch.tensor(res & 0xFFFFFFFF, dtype=dtype, device=dev))

        elif op == Opcode.MULHSU:
            if a != 0:
                bv = int(bp_to_int(regs[b]).item())
                cv = int(bp_to_int(regs[c]).item())
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                res = (bs * cv) >> 32
                regs[a] = int_to_bp(torch.tensor(res & 0xFFFFFFFF, dtype=dtype, device=dev))

        # ── Word memory ──
        elif op == Opcode.LOAD:
            if a != 0:
                base_val = int(bp_to_int(regs[b]).item())
                addr = (base_val + c) & 0xFFFFFFFF
                byte_addr = addr if byte_addressed else addr * 4
                if byte_addr + 3 < mem_size:
                    val = _mem_read_word(mem, byte_addr)
                    regs[a] = int_to_bp(torch.tensor(val, dtype=dtype, device=dev))

        elif op == Opcode.STORE:
            base_val = int(bp_to_int(regs[b]).item())
            addr = (base_val + c) & 0xFFFFFFFF
            byte_addr = addr if byte_addressed else addr * 4
            if byte_addr + 3 < mem_size:
                val = int(bp_to_int(regs[a]).item())
                _mem_write_word(mem, byte_addr, val)

        # ── Byte/halfword memory (byte-addressed) ──
        elif op == Opcode.LOADB:
            if a != 0:
                addr = (int(bp_to_int(regs[b]).item()) + c) & 0xFFFFFFFF
                if addr < mem_size:
                    val = mem[addr]  # zero-extend
                    regs[a] = int_to_bp(torch.tensor(val, dtype=dtype, device=dev))

        elif op == Opcode.LOADBS:
            if a != 0:
                addr = (int(bp_to_int(regs[b]).item()) + c) & 0xFFFFFFFF
                if addr < mem_size:
                    val = mem[addr]
                    if val >= 128:
                        val -= 256  # sign-extend
                    val = val & 0xFFFFFFFF
                    regs[a] = int_to_bp(torch.tensor(val, dtype=dtype, device=dev))

        elif op == Opcode.LOADH:
            if a != 0:
                addr = (int(bp_to_int(regs[b]).item()) + c) & 0xFFFFFFFF
                if addr + 1 < mem_size:
                    val = mem[addr] | (mem[addr+1] << 8)  # LE, zero-extend
                    regs[a] = int_to_bp(torch.tensor(val, dtype=dtype, device=dev))

        elif op == Opcode.LOADHS:
            if a != 0:
                addr = (int(bp_to_int(regs[b]).item()) + c) & 0xFFFFFFFF
                if addr + 1 < mem_size:
                    val = mem[addr] | (mem[addr+1] << 8)
                    if val >= 0x8000:
                        val -= 0x10000
                    val = val & 0xFFFFFFFF
                    regs[a] = int_to_bp(torch.tensor(val, dtype=dtype, device=dev))

        elif op == Opcode.STOREB:
            addr = (int(bp_to_int(regs[b]).item()) + c) & 0xFFFFFFFF
            if addr < mem_size:
                val = int(bp_to_int(regs[a]).item()) & 0xFF
                mem[addr] = val

        elif op == Opcode.STOREH:
            addr = (int(bp_to_int(regs[b]).item()) + c) & 0xFFFFFFFF
            if addr + 1 < mem_size:
                val = int(bp_to_int(regs[a]).item()) & 0xFFFF
                mem[addr] = val & 0xFF
                mem[addr+1] = (val >> 8) & 0xFF

        # ── Control flow ──
        elif op == Opcode.JMP:
            next_pc = a

        elif op == Opcode.JMPR:
            # Indirect jump: PC = reg[a]
            next_pc = int(bp_to_int(regs[a]).item())

        elif op in (Opcode.BEQ, Opcode.BNE, Opcode.BLT, Opcode.BGE,
                    Opcode.BLTU, Opcode.BGEU):
            va_int = bp_to_int(regs[a]).item()
            vb_int = bp_to_int(regs[b]).item()
            va_s = va_int - _MOD32 if va_int >= 2**31 else va_int
            vb_s = vb_int - _MOD32 if vb_int >= 2**31 else vb_int

            taken = False
            if op == Opcode.BEQ:
                taken = (va_int == vb_int)
            elif op == Opcode.BNE:
                taken = (va_int != vb_int)
            elif op == Opcode.BLT:
                taken = (va_s < vb_s)
            elif op == Opcode.BGE:
                taken = (va_s >= vb_s)
            elif op == Opcode.BLTU:
                taken = (va_int < vb_int)
            elif op == Opcode.BGEU:
                taken = (va_int >= vb_int)

            if taken:
                next_pc = c

        pc = next_pc
        cycles = cycle + 1

    # ── Extract results ──
    result_regs = {}
    for i in range(32):
        if i == 0:
            result_regs[i] = 0
        else:
            result_regs[i] = int(bp_to_int(regs[i]).item())

    return GPUExecutionResult(result_regs, pc, cycles, halted, mem)


def _mem_read_word(mem: bytearray, addr: int) -> int:
    """Read a 32-bit LE word from byte address."""
    return mem[addr] | (mem[addr+1]<<8) | (mem[addr+2]<<16) | (mem[addr+3]<<24)


def _mem_write_word(mem: bytearray, addr: int, val: int):
    """Write a 32-bit LE word to byte address."""
    val = val & 0xFFFFFFFF
    mem[addr] = val & 0xFF
    mem[addr+1] = (val >> 8) & 0xFF
    mem[addr+2] = (val >> 16) & 0xFF
    mem[addr+3] = (val >> 24) & 0xFF


def _predecode(instructions: list[Instruction]) -> list[tuple]:
    """Pre-decode instructions into (opcode, a, b, c, immediate) tuples."""
    decoded = []
    for instr in instructions:
        op = instr.opcode
        a, b, c = instr.a, instr.b, instr.c
        if op == Opcode.MOVI:
            imm = ((b & 0xFFFF) << 16) | (c & 0xFFFF)
        else:
            imm = 0
        decoded.append((op, a, b, c, imm))
    return decoded
