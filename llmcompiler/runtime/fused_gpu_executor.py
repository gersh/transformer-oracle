"""
Fused GPU executor — minimal CPU↔GPU sync.

Strategy: batch the opcode dispatch into a single tensor operation per cycle,
pre-compute ALL instruction metadata on CPU before the loop, and only
touch GPU for the actual register/memory operations.

The key optimization: pre-decode instructions into parallel arrays on CPU
(avoiding per-cycle GPU→CPU transfers for instruction fetch), and keep
only the register file and memory on GPU.
"""

import torch
from typing import Optional
from ..core.nisa import Instruction, Opcode

MASK32 = 0xFFFFFFFF


def fused_gpu_execute(
    instructions: list[Instruction],
    max_cycles: int = 10000,
    initial_registers: Optional[dict[int, int]] = None,
    initial_memory: Optional[dict[int, int]] = None,
    memory_bytes: Optional[bytearray] = None,
    mem_size: int = 65536,
    device: str = 'cuda',
) -> tuple[dict[int, int], int, bool]:
    """Execute with registers on GPU, instruction dispatch on CPU.

    Optimizations vs old executor:
    - No bipolar encoding (direct int64)
    - No bp_to_int/int_to_bp conversions
    - Pre-decoded instructions (no per-cycle overhead)
    - Memory as CPU bytearray (faster than GPU for random access)
    - Register file on GPU for ALU ops
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
    dev = torch.device(device)
    byte_addressed = (memory_bytes is not None)

    # ── Pre-decode all instructions on CPU ──
    n_instr = len(instructions)
    ops = [0] * n_instr
    As = [0] * n_instr
    Bs = [0] * n_instr
    Cs = [0] * n_instr
    imms = [0] * n_instr
    for i, ins in enumerate(instructions):
        ops[i] = int(ins.opcode)
        As[i] = ins.a
        Bs[i] = ins.b
        Cs[i] = ins.c
        if ins.opcode == Opcode.MOVI:
            imms[i] = ((ins.b & 0xFFFF) << 16) | (ins.c & 0xFFFF)

    # ── Register file on GPU ──
    regs = torch.zeros(32, dtype=torch.int64, device=dev)
    if initial_registers:
        for idx, val in initial_registers.items():
            if 0 < idx < 32:
                regs[idx] = val & MASK32

    # ── Memory on CPU (bytearray — faster for random byte access) ──
    mem = bytearray(mem_size)
    if memory_bytes is not None:
        mem[:len(memory_bytes)] = memory_bytes[:mem_size]
    elif initial_memory:
        for wa, val in initial_memory.items():
            ba = wa * 4
            if ba + 3 < mem_size:
                v = val & MASK32
                mem[ba] = v & 0xFF; mem[ba+1] = (v>>8)&0xFF
                mem[ba+2] = (v>>16)&0xFF; mem[ba+3] = (v>>24)&0xFF

    # ── Opcode constants ──
    O_HALT = int(Opcode.HALT)
    O_NOP = int(Opcode.NOP)
    O_MOVI = int(Opcode.MOVI)
    O_MOV = int(Opcode.MOV)
    O_ADD = int(Opcode.ADD)
    O_SUB = int(Opcode.SUB)
    O_MUL = int(Opcode.MUL)
    O_AND = int(Opcode.AND)
    O_OR = int(Opcode.OR)
    O_XOR = int(Opcode.XOR)
    O_NOT = int(Opcode.NOT)
    O_SHL = int(Opcode.SHL)
    O_SHR = int(Opcode.SHR)
    O_SRA = int(Opcode.SRA)
    O_SLT = int(Opcode.SLT)
    O_SLTU = int(Opcode.SLTU)
    O_DIV = int(Opcode.DIV)
    O_DIVU = int(Opcode.DIVU)
    O_REM = int(Opcode.REM)
    O_REMU = int(Opcode.REMU)
    O_MULH = int(Opcode.MULH)
    O_MULHU = int(Opcode.MULHU)
    O_MULHSU = int(Opcode.MULHSU)
    O_LOAD = int(Opcode.LOAD)
    O_STORE = int(Opcode.STORE)
    O_LOADB = int(Opcode.LOADB)
    O_LOADBS = int(Opcode.LOADBS)
    O_LOADH = int(Opcode.LOADH)
    O_LOADHS = int(Opcode.LOADHS)
    O_STOREB = int(Opcode.STOREB)
    O_STOREH = int(Opcode.STOREH)
    O_JMP = int(Opcode.JMP)
    O_JMPR = int(Opcode.JMPR)
    O_BEQ = int(Opcode.BEQ)
    O_BNE = int(Opcode.BNE)
    O_BLT = int(Opcode.BLT)
    O_BGE = int(Opcode.BGE)
    O_BLTU = int(Opcode.BLTU)
    O_BGEU = int(Opcode.BGEU)

    # Cache register reads — avoid GPU→CPU sync by keeping a CPU mirror
    # We sync lazily: only read from GPU when needed after GPU-computed values
    reg_cache = [0] * 32  # CPU mirror of register values
    if initial_registers:
        for idx, val in initial_registers.items():
            if 0 < idx < 32:
                reg_cache[idx] = val & MASK32
    cache_dirty = set()  # registers modified on GPU but not synced to cache

    def _reg(i):
        """Read register, syncing from GPU if needed."""
        if i == 0: return 0
        if i in cache_dirty:
            reg_cache[i] = regs[i].item()
            cache_dirty.discard(i)
        return reg_cache[i]

    def _set_reg(i, val):
        """Set register on both GPU and CPU cache."""
        if i == 0: return
        val = val & MASK32
        regs[i] = val
        reg_cache[i] = val

    def _set_reg_gpu(i, val_tensor):
        """Set register from GPU tensor result."""
        if i == 0: return
        regs[i] = val_tensor & MASK32
        cache_dirty.add(i)

    # ── Execution loop ──
    pc = 0
    cycles = 0
    halted = False

    for cycle in range(max_cycles):
        if pc >= n_instr or pc < 0:
            halted = True
            break

        op = ops[pc]
        a = As[pc]
        b = Bs[pc]
        c = Cs[pc]
        imm = imms[pc]
        next_pc = pc + 1

        if op == O_HALT:
            halted = True; cycles = cycle + 1; break
        elif op == O_NOP:
            pass
        elif op == O_MOVI:
            _set_reg(a, imm)
        elif op == O_MOV:
            _set_reg(a, _reg(b))

        # ALU — do on GPU, lazy sync
        elif op == O_ADD:
            if a: _set_reg_gpu(a, regs[b] + regs[c])
        elif op == O_SUB:
            if a: _set_reg_gpu(a, regs[b] - regs[c])
        elif op == O_MUL:
            if a: _set_reg_gpu(a, regs[b] * regs[c])
        elif op == O_AND:
            if a: _set_reg_gpu(a, regs[b] & regs[c])
        elif op == O_OR:
            if a: _set_reg_gpu(a, regs[b] | regs[c])
        elif op == O_XOR:
            if a: _set_reg_gpu(a, regs[b] ^ regs[c])
        elif op == O_NOT:
            if a: _set_reg_gpu(a, ~regs[b])
        elif op == O_SHL:
            if a:
                s = _reg(c) & 0x1F
                _set_reg(a, (_reg(b) << s) & MASK32)
        elif op == O_SHR:
            if a:
                s = _reg(c) & 0x1F
                _set_reg(a, (_reg(b) >> s) & MASK32)
        elif op == O_SRA:
            if a:
                s = _reg(c) & 0x1F
                v = _reg(b)
                if v >= 0x80000000: v -= 0x100000000
                _set_reg(a, (v >> s) & MASK32)
        elif op == O_SLT:
            if a:
                va, vb = _reg(b), _reg(c)
                vs = va - 0x100000000 if va >= 0x80000000 else va
                ws = vb - 0x100000000 if vb >= 0x80000000 else vb
                _set_reg(a, 1 if vs < ws else 0)
        elif op == O_SLTU:
            if a: _set_reg(a, 1 if _reg(b) < _reg(c) else 0)
        elif op == O_DIV:
            if a:
                bv, cv = _reg(b), _reg(c)
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                cs = cv - 0x100000000 if cv >= 0x80000000 else cv
                if cs == 0: _set_reg(a, MASK32)
                else:
                    r = int(abs(bs)//abs(cs)) * (1 if (bs<0)==(cs<0) else -1)
                    _set_reg(a, r & MASK32)
        elif op == O_DIVU:
            if a:
                bv, cv = _reg(b), _reg(c)
                _set_reg(a, MASK32 if cv == 0 else (bv // cv) & MASK32)
        elif op == O_REM:
            if a:
                bv, cv = _reg(b), _reg(c)
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                cs = cv - 0x100000000 if cv >= 0x80000000 else cv
                if cs == 0: _set_reg(a, bv)
                else:
                    r = abs(bs) % abs(cs)
                    if bs < 0: r = -r
                    _set_reg(a, r & MASK32)
        elif op == O_REMU:
            if a:
                bv, cv = _reg(b), _reg(c)
                _set_reg(a, bv if cv == 0 else (bv % cv) & MASK32)
        elif op == O_MULHU:
            if a:
                _set_reg(a, ((_reg(b) * _reg(c)) >> 32) & MASK32)
        elif op == O_MULH:
            if a:
                bs = _reg(b); cs = _reg(c)
                if bs >= 0x80000000: bs -= 0x100000000
                if cs >= 0x80000000: cs -= 0x100000000
                _set_reg(a, ((bs * cs) >> 32) & MASK32)
        elif op == O_MULHSU:
            if a:
                bs = _reg(b)
                if bs >= 0x80000000: bs -= 0x100000000
                _set_reg(a, ((bs * _reg(c)) >> 32) & MASK32)

        # Memory
        elif op == O_LOAD:
            if a:
                addr = _reg(b) + c
                if not byte_addressed: addr *= 4
                addr &= MASK32
                if addr + 3 < mem_size:
                    v = mem[addr] | (mem[addr+1]<<8) | (mem[addr+2]<<16) | (mem[addr+3]<<24)
                    _set_reg(a, v)
        elif op == O_STORE:
            addr = _reg(b) + c
            if not byte_addressed: addr *= 4
            addr &= MASK32
            if addr + 3 < mem_size:
                v = _reg(a) & MASK32
                mem[addr]=v&0xFF; mem[addr+1]=(v>>8)&0xFF
                mem[addr+2]=(v>>16)&0xFF; mem[addr+3]=(v>>24)&0xFF
        elif op == O_LOADB:
            if a:
                addr = (_reg(b) + c) & MASK32
                if addr < mem_size: _set_reg(a, mem[addr])
        elif op == O_LOADBS:
            if a:
                addr = (_reg(b) + c) & MASK32
                if addr < mem_size:
                    v = mem[addr]
                    _set_reg(a, (v-256)&MASK32 if v>=128 else v)
        elif op == O_LOADH:
            if a:
                addr = (_reg(b) + c) & MASK32
                if addr+1 < mem_size:
                    _set_reg(a, mem[addr]|(mem[addr+1]<<8))
        elif op == O_LOADHS:
            if a:
                addr = (_reg(b) + c) & MASK32
                if addr+1 < mem_size:
                    v = mem[addr]|(mem[addr+1]<<8)
                    _set_reg(a, (v-0x10000)&MASK32 if v>=0x8000 else v)
        elif op == O_STOREB:
            addr = (_reg(b) + c) & MASK32
            if addr < mem_size: mem[addr] = _reg(a) & 0xFF
        elif op == O_STOREH:
            addr = (_reg(b) + c) & MASK32
            if addr+1 < mem_size:
                v = _reg(a)
                mem[addr]=v&0xFF; mem[addr+1]=(v>>8)&0xFF

        # Control flow
        elif op == O_JMP:
            next_pc = a
        elif op == O_JMPR:
            next_pc = _reg(a)
        elif op >= O_BEQ and op <= O_BGEU:
            ra_v, rb_v = _reg(a), _reg(b)
            ra_s = ra_v - 0x100000000 if ra_v >= 0x80000000 else ra_v
            rb_s = rb_v - 0x100000000 if rb_v >= 0x80000000 else rb_v
            taken = False
            if op == O_BEQ: taken = (ra_v == rb_v)
            elif op == O_BNE: taken = (ra_v != rb_v)
            elif op == O_BLT: taken = (ra_s < rb_s)
            elif op == O_BGE: taken = (ra_s >= rb_s)
            elif op == O_BLTU: taken = (ra_v < rb_v)
            elif op == O_BGEU: taken = (ra_v >= rb_v)
            if taken: next_pc = c

        pc = next_pc
        cycles = cycle + 1

    result_regs = {i: (0 if i == 0 else _reg(i)) for i in range(32)}
    return result_regs, cycles, halted
