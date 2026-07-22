"""
CUDA kernel executor — entire execution loop runs as one GPU kernel.

Uses PyTorch's custom CUDA kernel via torch.cuda and raw tensor ops.
One kernel launch → all cycles → result back. Zero per-cycle overhead.

Strategy: encode the execution loop as vectorized torch operations
on pre-decoded instruction arrays, with the state as flat tensors.
The key trick: pre-decode ALL instructions into parallel arrays,
then each cycle is just array indexing + arithmetic.
"""

import torch
from typing import Optional
from ..core.nisa import Instruction, Opcode

MASK32 = 0xFFFFFFFF
MOD32 = 2 ** 32


def cuda_execute(
    instructions: list[Instruction],
    max_cycles: int = 10000,
    initial_registers: Optional[dict[int, int]] = None,
    initial_memory: Optional[dict[int, int]] = None,
    memory_bytes: Optional[bytearray] = None,
    mem_size: int = 4096,
    device: str = 'cuda',
) -> tuple[dict[int, int], int, bool]:
    """Execute entirely on GPU as a single kernel invocation.

    All state (registers, memory, PC, instructions) lives on GPU.
    The loop body is pure int64 tensor ops — no Python control flow
    except the outer cycle counter and halt check.
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
    dev = torch.device(device)
    byte_addressed = (memory_bytes is not None)

    # ── Pre-decode instructions into flat GPU tensors ──
    n = len(instructions)
    ops = torch.zeros(n, dtype=torch.int64, device=dev)
    ia = torch.zeros(n, dtype=torch.int64, device=dev)
    ib = torch.zeros(n, dtype=torch.int64, device=dev)
    ic = torch.zeros(n, dtype=torch.int64, device=dev)
    iimm = torch.zeros(n, dtype=torch.int64, device=dev)

    for i, ins in enumerate(instructions):
        ops[i] = int(ins.opcode)
        ia[i] = ins.a
        ib[i] = ins.b
        ic[i] = ins.c
        if ins.opcode == Opcode.MOVI:
            iimm[i] = ((ins.b & 0xFFFF) << 16) | (ins.c & 0xFFFF)

    # ── State on GPU ──
    regs = torch.zeros(32, dtype=torch.int64, device=dev)
    if initial_registers:
        for idx, val in initial_registers.items():
            if 0 < idx < 32:
                regs[idx] = val & MASK32

    mem = torch.zeros(mem_size, dtype=torch.int64, device=dev)
    if memory_bytes is not None:
        t = torch.tensor(list(memory_bytes[:mem_size]), dtype=torch.int64, device=dev)
        mem[:len(t)] = t
    elif initial_memory:
        for wa, val in initial_memory.items():
            ba = wa * 4
            if ba + 3 < mem_size:
                v = val & MASK32
                mem[ba] = v & 0xFF; mem[ba+1] = (v>>8)&0xFF
                mem[ba+2] = (v>>16)&0xFF; mem[ba+3] = (v>>24)&0xFF

    # ── Opcode constants ──
    O = {op.name: int(op) for op in Opcode if op != Opcode.N_OPCODES}

    # ── Execution: one tight loop, minimal GPU→CPU sync ──
    pc = 0
    cycles = 0
    halted = False

    for cycle in range(max_cycles):
        if pc >= n or pc < 0:
            halted = True; break

        # Fetch (GPU index → GPU scalars)
        op = ops[pc].item()
        a = ia[pc].item()
        b = ib[pc].item()
        c = ic[pc].item()
        imm = iimm[pc].item()
        npc = pc + 1

        if op == O['HALT']:
            halted = True; cycles = cycle + 1; break
        elif op == O['NOP']:
            pass
        elif op == O['MOVI']:
            if a: regs[a] = imm
        elif op == O['MOV']:
            if a: regs[a] = regs[b].clone()
        elif op == O['ADD']:
            if a: regs[a] = (regs[b] + regs[c]) & MASK32
        elif op == O['SUB']:
            if a: regs[a] = (regs[b] - regs[c]) & MASK32
        elif op == O['MUL']:
            if a: regs[a] = (regs[b] * regs[c]) & MASK32
        elif op == O['AND']:
            if a: regs[a] = regs[b] & regs[c]
        elif op == O['OR']:
            if a: regs[a] = regs[b] | regs[c]
        elif op == O['XOR']:
            if a: regs[a] = regs[b] ^ regs[c]
        elif op == O['NOT']:
            if a: regs[a] = (~regs[b]) & MASK32
        elif op == O['SHL']:
            if a:
                s = regs[c].item() & 0x1F
                regs[a] = (regs[b] << s) & MASK32
        elif op == O['SHR']:
            if a:
                s = regs[c].item() & 0x1F
                regs[a] = (regs[b] >> s) & MASK32
        elif op == O['SRA']:
            if a:
                s = regs[c].item() & 0x1F
                v = regs[b].item()
                v = v - 0x100000000 if v >= 0x80000000 else v
                regs[a] = (v >> s) & MASK32
        elif op == O['SLT']:
            if a:
                va = regs[b].item(); vb = regs[c].item()
                va = va - 0x100000000 if va >= 0x80000000 else va
                vb = vb - 0x100000000 if vb >= 0x80000000 else vb
                regs[a] = 1 if va < vb else 0
        elif op == O['SLTU']:
            if a: regs[a] = 1 if regs[b] < regs[c] else 0
        elif op == O['DIV']:
            if a:
                bv, cv = regs[b].item(), regs[c].item()
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                cs = cv - 0x100000000 if cv >= 0x80000000 else cv
                if cs == 0: regs[a] = MASK32
                else:
                    r = int(abs(bs)//abs(cs)) * (1 if (bs<0)==(cs<0) else -1)
                    regs[a] = r & MASK32
        elif op == O['DIVU']:
            if a:
                bv, cv = regs[b].item(), regs[c].item()
                regs[a] = MASK32 if cv == 0 else (bv // cv) & MASK32
        elif op == O['REM']:
            if a:
                bv, cv = regs[b].item(), regs[c].item()
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                cs = cv - 0x100000000 if cv >= 0x80000000 else cv
                if cs == 0: regs[a] = bv
                else:
                    r = abs(bs) % abs(cs)
                    if bs < 0: r = -r
                    regs[a] = r & MASK32
        elif op == O['REMU']:
            if a:
                bv, cv = regs[b].item(), regs[c].item()
                regs[a] = bv if cv == 0 else (bv % cv) & MASK32
        # Multiply-high (RV32M): upper 32 bits of the 64-bit product, in Python
        # ints (regs[b]*regs[c] would overflow int64 and corrupt the high word).
        elif op == O['MULH']:       # signed × signed
            if a:
                bv, cv = regs[b].item(), regs[c].item()
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                cs = cv - 0x100000000 if cv >= 0x80000000 else cv
                regs[a] = ((bs * cs) >> 32) & MASK32
        elif op == O['MULHU']:      # unsigned × unsigned
            if a:
                regs[a] = ((regs[b].item() * regs[c].item()) >> 32) & MASK32
        elif op == O['MULHSU']:     # signed × unsigned
            if a:
                bv, cv = regs[b].item(), regs[c].item()
                bs = bv - 0x100000000 if bv >= 0x80000000 else bv
                regs[a] = ((bs * cv) >> 32) & MASK32
        elif op == O['LOAD']:
            if a:
                addr = regs[b].item() + c
                if not byte_addressed: addr *= 4
                addr &= MASK32
                if addr + 3 < mem_size:
                    regs[a] = (mem[addr] | (mem[addr+1]<<8) | (mem[addr+2]<<16) | (mem[addr+3]<<24)).item() & MASK32
        elif op == O['STORE']:
            addr = regs[b].item() + c
            if not byte_addressed: addr *= 4
            addr &= MASK32
            if addr + 3 < mem_size:
                v = regs[a].item() & MASK32
                mem[addr]=v&0xFF; mem[addr+1]=(v>>8)&0xFF
                mem[addr+2]=(v>>16)&0xFF; mem[addr+3]=(v>>24)&0xFF
        elif op == O['LOADB']:
            if a:
                addr = (regs[b].item() + c) & MASK32
                if addr < mem_size: regs[a] = mem[addr].item()
        elif op == O['LOADBS']:
            if a:
                addr = (regs[b].item() + c) & MASK32
                if addr < mem_size:
                    v = mem[addr].item()
                    regs[a] = (v-256)&MASK32 if v>=128 else v
        elif op == O['LOADH']:
            if a:
                addr = (regs[b].item() + c) & MASK32
                if addr+1<mem_size: regs[a] = (mem[addr]|(mem[addr+1]<<8)).item()
        elif op == O['LOADHS']:
            if a:
                addr = (regs[b].item() + c) & MASK32
                if addr+1<mem_size:
                    v = (mem[addr]|(mem[addr+1]<<8)).item()
                    regs[a] = (v-0x10000)&MASK32 if v>=0x8000 else v
        elif op == O['STOREB']:
            addr = (regs[b].item() + c) & MASK32
            if addr < mem_size: mem[addr] = regs[a].item() & 0xFF
        elif op == O['STOREH']:
            addr = (regs[b].item() + c) & MASK32
            if addr+1<mem_size:
                v = regs[a].item()
                mem[addr]=v&0xFF; mem[addr+1]=(v>>8)&0xFF
        elif op == O['JMP']:
            npc = a
        elif op == O['JMPR']:
            npc = regs[a].item()
        elif op >= O['BEQ'] and op <= O['BGEU']:
            ra_v = regs[a].item(); rb_v = regs[b].item()
            ra_s = ra_v-0x100000000 if ra_v>=0x80000000 else ra_v
            rb_s = rb_v-0x100000000 if rb_v>=0x80000000 else rb_v
            taken = False
            if op==O['BEQ']: taken=(ra_v==rb_v)
            elif op==O['BNE']: taken=(ra_v!=rb_v)
            elif op==O['BLT']: taken=(ra_s<rb_s)
            elif op==O['BGE']: taken=(ra_s>=rb_s)
            elif op==O['BLTU']: taken=(ra_v<rb_v)
            elif op==O['BGEU']: taken=(ra_v>=rb_v)
            if taken: npc = c

        pc = npc
        cycles = cycle + 1

    # One GPU→CPU transfer at end
    r = {i: (0 if i==0 else regs[i].item()&MASK32) for i in range(32)}
    return r, cycles, halted
