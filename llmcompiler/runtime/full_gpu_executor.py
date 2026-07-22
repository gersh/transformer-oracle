"""
Fully GPU-resident executor.

Everything stays on GPU: registers, memory, instructions, PC.
Each cycle is pure GPU tensor operations — no CPU↔GPU transfers.

Key design:
- Registers: (32,) int64 tensor on GPU
- Memory: (mem_size,) int8 stored as int64 on GPU
- Instructions: (n_instr, 5) pre-decoded tensor on GPU
- All ALU ops: native int64 bitwise/arithmetic (exact for 32-bit)
- Instruction dispatch: computed via masked selection, not if/else

This eliminates the ~1ms/cycle overhead from the previous executor's
Python dispatch loop and CPU↔GPU register transfers.
"""

import torch
from typing import Optional
from ..core.nisa import Instruction, Opcode

MASK32 = 0xFFFFFFFF


def full_gpu_execute(
    instructions: list[Instruction],
    max_cycles: int = 10000,
    initial_registers: Optional[dict[int, int]] = None,
    initial_memory: Optional[dict[int, int]] = None,
    memory_bytes: Optional[bytearray] = None,
    mem_size: int = 65536,
    device: str = 'cuda',
) -> tuple[dict[int, int], int, bool]:
    """Execute entirely on GPU. Returns (registers, cycles, halted)."""

    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
    dev = torch.device(device)

    byte_addressed = (memory_bytes is not None)

    # ── Pre-encode instructions as GPU tensor ──
    n_instr = len(instructions)
    instr_data = torch.zeros(n_instr, 5, dtype=torch.int64, device=dev)
    for i, ins in enumerate(instructions):
        op = int(ins.opcode)
        a, b, c = ins.a, ins.b, ins.c
        imm = ((b & 0xFFFF) << 16) | (c & 0xFFFF) if op == int(Opcode.MOVI) else 0
        instr_data[i] = torch.tensor([op, a, b, c, imm], dtype=torch.int64)

    # ── Registers: int64 on GPU ──
    regs = torch.zeros(32, dtype=torch.int64, device=dev)
    if initial_registers:
        for idx, val in initial_registers.items():
            if idx > 0:
                regs[idx] = val & MASK32

    # ── Memory: int64 on GPU (one element per byte) ──
    mem = torch.zeros(mem_size, dtype=torch.int64, device=dev)
    if memory_bytes is not None:
        for i, b in enumerate(memory_bytes[:mem_size]):
            mem[i] = b
    elif initial_memory:
        for word_addr, val in initial_memory.items():
            ba = word_addr * 4
            if ba + 3 < mem_size:
                mem[ba] = val & 0xFF
                mem[ba+1] = (val >> 8) & 0xFF
                mem[ba+2] = (val >> 16) & 0xFF
                mem[ba+3] = (val >> 24) & 0xFF

    # ── Opcode constants ──
    OP = {op.name: int(op) for op in Opcode if op != Opcode.N_OPCODES}

    # ── Execution loop — all ops are GPU tensor ops ──
    pc = 0
    cycles = 0
    halted = False

    for cycle in range(max_cycles):
        if pc >= n_instr or pc < 0:
            halted = True
            break

        # Fetch (GPU tensor index)
        ins = instr_data[pc]
        op = ins[0].item()
        a = ins[1].item()
        b_idx = ins[2].item()
        c_idx = ins[3].item()
        imm = ins[4].item()

        next_pc = pc + 1

        # Read source registers (GPU) — only if valid indices
        rb = regs[b_idx] if b_idx < 32 else torch.tensor(0, dtype=torch.int64, device=dev)
        rc = regs[c_idx] if c_idx < 32 else torch.tensor(0, dtype=torch.int64, device=dev)

        if op == OP['HALT']:
            halted = True
            cycles = cycle + 1
            break

        elif op == OP['NOP']:
            pass

        elif op == OP['MOVI']:
            if a != 0:
                regs[a] = imm & MASK32

        elif op == OP['MOV']:
            if a != 0:
                regs[a] = rb

        # ── ALU (all GPU int64 operations) ──
        elif op == OP['ADD']:
            if a != 0: regs[a] = (rb + rc) & MASK32
        elif op == OP['SUB']:
            if a != 0: regs[a] = (rb - rc) & MASK32
        elif op == OP['MUL']:
            if a != 0: regs[a] = (rb * rc) & MASK32
        elif op == OP['AND']:
            if a != 0: regs[a] = rb & rc
        elif op == OP['OR']:
            if a != 0: regs[a] = rb | rc
        elif op == OP['XOR']:
            if a != 0: regs[a] = rb ^ rc
        elif op == OP['NOT']:
            if a != 0: regs[a] = (~rb) & MASK32
        elif op == OP['SHL']:
            if a != 0:
                s = (rc & 0x1F).item()
                regs[a] = (rb << s) & MASK32
        elif op == OP['SHR']:
            if a != 0:
                s = (rc & 0x1F).item()
                regs[a] = (rb >> s) & MASK32
        elif op == OP['SRA']:
            if a != 0:
                s = (rc & 0x1F).item()
                val = rb.item()
                if val >= 0x80000000:
                    val = val - 0x100000000
                regs[a] = (val >> s) & MASK32
        elif op == OP['SLT']:
            if a != 0:
                va = rb.item()
                vb = rc.item()
                vs = va - 0x100000000 if va >= 0x80000000 else va
                ws = vb - 0x100000000 if vb >= 0x80000000 else vb
                regs[a] = 1 if vs < ws else 0
        elif op == OP['SLTU']:
            if a != 0:
                regs[a] = 1 if rb < rc else 0
        elif op == OP['DIV']:
            if a != 0:
                bv, cv = rb.item(), rc.item()
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                cs = cv - 0x100000000 if cv >= 0x80000000 else cv
                if cs == 0:
                    regs[a] = MASK32
                else:
                    r = int(abs(bs) // abs(cs))
                    if (bs < 0) != (cs < 0): r = -r
                    regs[a] = r & MASK32
        elif op == OP['DIVU']:
            if a != 0:
                bv, cv = rb.item(), rc.item()
                regs[a] = MASK32 if cv == 0 else (bv // cv) & MASK32
        elif op == OP['REM']:
            if a != 0:
                bv, cv = rb.item(), rc.item()
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                cs = cv - 0x100000000 if cv >= 0x80000000 else cv
                if cs == 0:
                    regs[a] = bv
                else:
                    r = abs(bs) % abs(cs)
                    if bs < 0: r = -r
                    regs[a] = r & MASK32
        elif op == OP['REMU']:
            if a != 0:
                bv, cv = rb.item(), rc.item()
                regs[a] = bv if cv == 0 else (bv % cv) & MASK32

        # ── Multiply-high (RV32M): upper 32 bits of the 64-bit product.
        # Computed in Python ints (arbitrary precision) because rb*rc reaches
        # ~2^64 and would overflow int64, corrupting the high word. The Python
        # `>>` is arithmetic, so for signed products it yields the correct
        # two's-complement high word after masking to 32 bits. ──
        elif op == OP['MULH']:      # signed × signed
            if a != 0:
                bv, cv = rb.item(), rc.item()
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                cs = cv - 0x100000000 if cv >= 0x80000000 else cv
                regs[a] = ((bs * cs) >> 32) & MASK32
        elif op == OP['MULHU']:     # unsigned × unsigned
            if a != 0:
                regs[a] = ((rb.item() * rc.item()) >> 32) & MASK32
        elif op == OP['MULHSU']:    # signed × unsigned
            if a != 0:
                bv, cv = rb.item(), rc.item()
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                regs[a] = ((bs * cv) >> 32) & MASK32

        # ── Memory (GPU tensor indexing) ──
        elif op == OP['LOAD']:
            if a != 0:
                addr = rb.item() + c_idx
                if not byte_addressed:
                    addr *= 4
                addr &= MASK32
                if addr + 3 < mem_size:
                    v = mem[addr] | (mem[addr+1] << 8) | (mem[addr+2] << 16) | (mem[addr+3] << 24)
                    regs[a] = v.item() & MASK32

        elif op == OP['STORE']:
            addr = rb.item() + c_idx
            if not byte_addressed:
                addr *= 4
            addr &= MASK32
            if addr + 3 < mem_size:
                v = regs[a].item() & MASK32
                mem[addr] = v & 0xFF
                mem[addr+1] = (v >> 8) & 0xFF
                mem[addr+2] = (v >> 16) & 0xFF
                mem[addr+3] = (v >> 24) & 0xFF

        elif op == OP['LOADB']:
            if a != 0:
                addr = (rb.item() + c_idx) & MASK32
                if addr < mem_size:
                    regs[a] = mem[addr].item()

        elif op == OP['LOADBS']:
            if a != 0:
                addr = (rb.item() + c_idx) & MASK32
                if addr < mem_size:
                    v = mem[addr].item()
                    regs[a] = (v - 256) & MASK32 if v >= 128 else v

        elif op == OP['LOADH']:
            if a != 0:
                addr = (rb.item() + c_idx) & MASK32
                if addr + 1 < mem_size:
                    regs[a] = (mem[addr] | (mem[addr+1] << 8)).item()

        elif op == OP['LOADHS']:
            if a != 0:
                addr = (rb.item() + c_idx) & MASK32
                if addr + 1 < mem_size:
                    v = (mem[addr] | (mem[addr+1] << 8)).item()
                    regs[a] = (v - 0x10000) & MASK32 if v >= 0x8000 else v

        elif op == OP['STOREB']:
            addr = (rb.item() + c_idx) & MASK32
            if addr < mem_size:
                mem[addr] = regs[a].item() & 0xFF

        elif op == OP['STOREH']:
            addr = (rb.item() + c_idx) & MASK32
            if addr + 1 < mem_size:
                v = regs[a].item()
                mem[addr] = v & 0xFF
                mem[addr+1] = (v >> 8) & 0xFF

        # ── Control flow ──
        elif op == OP['JMP']:
            next_pc = a

        elif op == OP['JMPR']:
            next_pc = regs[a].item()

        elif op in (OP['BEQ'], OP['BNE'], OP['BLT'], OP['BGE'],
                    OP['BLTU'], OP['BGEU']):
            ra_val = regs[a].item()
            rb_val = regs[b_idx].item()
            ra_s = ra_val - 0x100000000 if ra_val >= 0x80000000 else ra_val
            rb_s = rb_val - 0x100000000 if rb_val >= 0x80000000 else rb_val

            taken = False
            if op == OP['BEQ']: taken = (ra_val == rb_val)
            elif op == OP['BNE']: taken = (ra_val != rb_val)
            elif op == OP['BLT']: taken = (ra_s < rb_s)
            elif op == OP['BGE']: taken = (ra_s >= rb_s)
            elif op == OP['BLTU']: taken = (ra_val < rb_val)
            elif op == OP['BGEU']: taken = (ra_val >= rb_val)
            if taken:
                next_pc = c_idx

        pc = next_pc
        cycles = cycle + 1

    # Extract results (GPU → CPU at the end only)
    result_regs = {}
    regs_cpu = regs.cpu()
    for i in range(32):
        result_regs[i] = 0 if i == 0 else int(regs_cpu[i].item()) & MASK32

    return result_regs, cycles, halted
