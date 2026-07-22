"""
NISA Assembler: parse text assembly into instruction lists.

Syntax:
    LABEL:           # define a label (resolves to instruction index)
    OPCODE args      # instruction
    # comment        # line comment (also inline after instruction)

    Register: r0-r31 (r0 = zero register)
    Immediate: decimal (42), hex (0xFF), negative (-1)

Examples:
    movi r1, 5
    movi r2, 3
    add r3, r1, r2
    beq r1, r2, done
    jmp loop
    done:
    halt

All opcodes are case-insensitive. Registers use 'r' prefix.
Labels are resolved in a second pass (forward references OK).
"""

import re
from typing import Optional
from ..core.nisa import Instruction, Opcode, N_REGS


# Map opcode names to Opcode enum
_OPCODE_MAP = {op.name.lower(): op for op in Opcode if op != Opcode.N_OPCODES}


def assemble(source: str) -> list[Instruction]:
    """Assemble NISA source text into a list of Instructions.

    Args:
        source: multi-line NISA assembly text

    Returns:
        list of Instruction objects

    Raises:
        SyntaxError: on parse errors
    """
    lines = source.strip().split('\n')

    # Pass 1: collect labels and parse instructions
    labels: dict[str, int] = {}
    raw_instrs: list[tuple[int, str, list[str]]] = []  # (line_no, opcode, args)
    instr_idx = 0

    for line_no, line in enumerate(lines, 1):
        # Strip comments
        line = line.split('#')[0].strip()
        if not line:
            continue

        # Check for label
        if ':' in line:
            parts = line.split(':', 1)
            label = parts[0].strip()
            if not re.match(r'^[a-zA-Z_]\w*$', label):
                raise SyntaxError(f"Line {line_no}: invalid label '{label}'")
            if label in labels:
                raise SyntaxError(f"Line {line_no}: duplicate label '{label}'")
            labels[label] = instr_idx
            # Check for instruction after label
            rest = parts[1].strip()
            if not rest:
                continue
            line = rest

        # Parse instruction
        tokens = re.split(r'[,\s]+', line.strip())
        tokens = [t for t in tokens if t]
        if not tokens:
            continue

        opcode_str = tokens[0].lower()
        if opcode_str not in _OPCODE_MAP:
            raise SyntaxError(f"Line {line_no}: unknown opcode '{tokens[0]}'")

        args = tokens[1:]
        raw_instrs.append((line_no, opcode_str, args))
        instr_idx += 1

    # Pass 2: resolve labels and build instructions
    instructions = []
    for line_no, opcode_str, args in raw_instrs:
        op = _OPCODE_MAP[opcode_str]
        try:
            instr = _build_instruction(op, args, labels, line_no)
        except Exception as e:
            raise SyntaxError(f"Line {line_no}: {e}") from e
        instructions.append(instr)

    return instructions


def _parse_register(s: str) -> int:
    """Parse a register reference like 'r5' or 'x5'."""
    s = s.strip().lower()
    if s.startswith('r') or s.startswith('x'):
        try:
            idx = int(s[1:])
        except ValueError:
            raise ValueError(f"Invalid register '{s}'")
        if not (0 <= idx < N_REGS):
            raise ValueError(f"Register index out of range: {s}")
        return idx
    raise ValueError(f"Expected register, got '{s}'")


def _parse_immediate(s: str, labels: dict[str, int]) -> int:
    """Parse an immediate value (decimal, hex) or label reference."""
    s = s.strip()
    # Check if it's a label
    if s in labels:
        return labels[s]
    # Try hex
    if s.startswith('0x') or s.startswith('0X'):
        return int(s, 16) & 0xFFFFFFFF
    # Try decimal (may be negative)
    try:
        v = int(s)
        return v & 0xFFFFFFFF
    except ValueError:
        raise ValueError(f"Cannot parse immediate '{s}' (not a number or known label)")


def _parse_reg_or_imm(s: str, labels: dict[str, int]) -> tuple[bool, int]:
    """Parse as register or immediate. Returns (is_register, value)."""
    s = s.strip().lower()
    if s.startswith('r') or s.startswith('x'):
        return True, _parse_register(s)
    return False, _parse_immediate(s, labels)


def _parse_mem_operand(s: str, labels: dict[str, int]) -> tuple[int, int]:
    """Parse memory operand like '[r5 + 3]' or '[r5]'. Returns (base_reg, offset)."""
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']')):
        raise ValueError(f"Expected memory operand [reg + offset], got '{s}'")
    inner = s[1:-1].strip()

    if '+' in inner:
        parts = inner.split('+')
        base = _parse_register(parts[0].strip())
        offset = _parse_immediate(parts[1].strip(), labels)
        return base, offset
    else:
        base = _parse_register(inner)
        return base, 0


def _build_instruction(op: Opcode, args: list[str], labels: dict[str, int],
                       line_no: int) -> Instruction:
    """Build an Instruction from parsed opcode and argument strings."""

    def reg(i: int) -> int:
        if i >= len(args):
            raise ValueError(f"{op.name} requires more arguments")
        return _parse_register(args[i])

    def imm(i: int) -> int:
        if i >= len(args):
            raise ValueError(f"{op.name} requires more arguments")
        return _parse_immediate(args[i], labels)

    if op in (Opcode.NOP, Opcode.HALT, Opcode.ECALL):
        return Instruction(op)

    elif op == Opcode.MOVI:
        # movi rd, imm
        rd = reg(0)
        val = imm(1)
        # Split into b (high 16) and c (low 16) for the Instruction constructor
        return Instruction(op, a=rd, b=(val >> 16) & 0xFFFF, c=val & 0xFFFF)

    elif op == Opcode.MOV:
        # mov rd, rs
        return Instruction(op, a=reg(0), b=reg(1))

    elif op == Opcode.NOT:
        # not rd, rs
        return Instruction(op, a=reg(0), b=reg(1))

    elif op in (Opcode.ADD, Opcode.SUB, Opcode.MUL,
                Opcode.AND, Opcode.OR, Opcode.XOR,
                Opcode.SHL, Opcode.SHR, Opcode.SRA,
                Opcode.SLT, Opcode.SLTU):
        # op rd, rs1, rs2
        return Instruction(op, a=reg(0), b=reg(1), c=reg(2))

    elif op == Opcode.LOAD:
        # load rd, [rb + offset]  OR  load rd, rb, offset
        rd = reg(0)
        # Check for bracket syntax
        remaining = ','.join(args[1:]).strip()
        if '[' in remaining:
            base, offset = _parse_mem_operand(remaining, labels)
            return Instruction(op, a=rd, b=base, c=offset)
        else:
            return Instruction(op, a=rd, b=reg(1), c=imm(2) if len(args) > 2 else 0)

    elif op == Opcode.STORE:
        # store rs, [rb + offset]  OR  store rs, rb, offset
        rs = reg(0)
        remaining = ','.join(args[1:]).strip()
        if '[' in remaining:
            base, offset = _parse_mem_operand(remaining, labels)
            return Instruction(op, a=rs, b=base, c=offset)
        else:
            return Instruction(op, a=rs, b=reg(1), c=imm(2) if len(args) > 2 else 0)

    elif op == Opcode.JMP:
        # jmp label_or_addr
        target = imm(0)
        return Instruction(op, a=target)

    elif op in (Opcode.BEQ, Opcode.BNE, Opcode.BLT, Opcode.BGE,
                Opcode.BLTU, Opcode.BGEU):
        # beq rs1, rs2, label_or_addr
        return Instruction(op, a=reg(0), b=reg(1), c=imm(2))

    else:
        raise ValueError(f"Unhandled opcode: {op.name}")
