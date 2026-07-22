"""
Wide integer executor — native Python arbitrary precision.

For DeFi/crypto fuzzing where 256-bit integers are required.
Uses Python's native arbitrary-precision ints (no overflow ever),
with explicit modular reduction where Solidity would have it.

This is the FASTEST executor for integer-heavy code because
Python ints are already arbitrary precision — no conversion needed.
"""

from typing import Optional
from ..core.nisa import Instruction, Opcode


def wide_execute(
    instructions: list[Instruction],
    max_cycles: int = 10000,
    initial_registers: Optional[dict[int, int]] = None,
    bit_width: int = 256,
) -> tuple[dict[int, int], int, bool]:
    """Execute with arbitrary-width integers.

    Args:
        instructions: NISA program
        max_cycles: max cycles
        initial_registers: {reg_idx: value}
        bit_width: integer width (256 for Solidity, 0 for unlimited)

    Returns:
        (registers, cycles, halted)
    """
    MASK = (1 << bit_width) - 1 if bit_width > 0 else None
    SIGN_BIT = 1 << (bit_width - 1) if bit_width > 0 else None

    def wrap(v):
        return v & MASK if MASK else v

    def to_signed(v):
        if MASK and v >= SIGN_BIT:
            return v - (MASK + 1)
        return v

    # Pre-decode
    n = len(instructions)
    ops = [int(ins.opcode) for ins in instructions]
    aa = [ins.a for ins in instructions]
    bb = [ins.b for ins in instructions]
    cc = [ins.c for ins in instructions]
    imms = [((ins.b & 0xFFFF) << 16) | (ins.c & 0xFFFF)
            if ins.opcode == Opcode.MOVI else 0 for ins in instructions]

    regs = [0] * 32
    if initial_registers:
        for idx, val in initial_registers.items():
            if 0 < idx < 32:
                regs[idx] = wrap(val) if MASK else val

    O = {op.name: int(op) for op in Opcode if op != Opcode.N_OPCODES}
    pc = 0

    for cycle in range(max_cycles):
        if pc >= n or pc < 0:
            return {i: regs[i] for i in range(32)}, cycle, True

        op = ops[pc]; a = aa[pc]; b = bb[pc]; c = cc[pc]; imm = imms[pc]
        npc = pc + 1

        if op == O['HALT']:
            return {i: regs[i] for i in range(32)}, cycle + 1, True
        elif op == O['NOP']:
            pass
        elif op == O['MOVI']:
            if a: regs[a] = imm
        elif op == O['MOV']:
            if a: regs[a] = regs[b]
        elif op == O['ADD']:
            if a: regs[a] = wrap(regs[b] + regs[c])
        elif op == O['SUB']:
            if a: regs[a] = wrap(regs[b] - regs[c])
        elif op == O['MUL']:
            if a: regs[a] = wrap(regs[b] * regs[c])
        elif op == O['AND']:
            if a: regs[a] = regs[b] & regs[c]
        elif op == O['OR']:
            if a: regs[a] = regs[b] | regs[c]
        elif op == O['XOR']:
            if a: regs[a] = regs[b] ^ regs[c]
        elif op == O['NOT']:
            if a: regs[a] = wrap(~regs[b])
        elif op == O['SHL']:
            if a:
                s = regs[c] & 0xFF
                regs[a] = wrap(regs[b] << s)
        elif op == O['SHR']:
            if a:
                s = regs[c] & 0xFF
                regs[a] = regs[b] >> s
        elif op == O['SLT']:
            if a:
                regs[a] = 1 if to_signed(regs[b]) < to_signed(regs[c]) else 0
        elif op == O['SLTU']:
            if a:
                regs[a] = 1 if regs[b] < regs[c] else 0
        elif op == O['DIV']:
            if a:
                bs = to_signed(regs[b]); cs = to_signed(regs[c])
                if cs == 0:
                    regs[a] = MASK or 0
                else:
                    r = abs(bs) // abs(cs)
                    if (bs < 0) != (cs < 0): r = -r
                    regs[a] = wrap(r)
        elif op == O['DIVU']:
            if a:
                cv = regs[c]
                regs[a] = (MASK or 0) if cv == 0 else regs[b] // cv
        elif op == O['REM']:
            if a:
                bs = to_signed(regs[b]); cs = to_signed(regs[c])
                if cs == 0:
                    regs[a] = regs[b]
                else:
                    r = abs(bs) % abs(cs)
                    if bs < 0: r = -r
                    regs[a] = wrap(r)
        elif op == O['REMU']:
            if a:
                cv = regs[c]
                regs[a] = regs[b] if cv == 0 else regs[b] % cv
        elif op == O['JMP']:
            npc = a
        elif op == O['JMPR']:
            npc = regs[a]
        elif op >= O['BEQ'] and op <= O['BGEU']:
            rv = regs[a]; bv = regs[b]
            rs = to_signed(rv); bs = to_signed(bv)
            taken = False
            if op == O['BEQ']: taken = (rv == bv)
            elif op == O['BNE']: taken = (rv != bv)
            elif op == O['BLT']: taken = (rs < bs)
            elif op == O['BGE']: taken = (rs >= bs)
            elif op == O['BLTU']: taken = (rv < bv)
            elif op == O['BGEU']: taken = (rv >= bv)
            if taken: npc = c

        pc = npc

    return {i: regs[i] for i in range(32)}, max_cycles, False
