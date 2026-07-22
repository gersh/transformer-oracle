"""
Neural ISA (NISA) - Instruction set optimized for transformer execution.

Fixed 4-field format: (opcode, a, b, c)
  - opcode: which operation
  - a: destination or first operand (register index or immediate)
  - b: second operand (register index or immediate)
  - c: third operand (register index, immediate, or unused)

This maps directly to what attention heads can route: the opcode determines
which ALU path activates, and a/b/c are routed via attention to the correct
register columns.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class Opcode(IntEnum):
    """NISA opcodes (~25 instructions)."""
    # Arithmetic
    ADD = 0      # a = b + c
    SUB = 1      # a = b - c
    MUL = 2      # a = b * c  (lower 32 bits)

    # Bitwise
    AND = 3      # a = b & c
    OR = 4       # a = b | c
    XOR = 5      # a = b ^ c
    NOT = 6      # a = ~b

    # Shifts
    SHL = 7      # a = b << c
    SHR = 8      # a = b >> c  (logical)
    SRA = 9      # a = b >> c  (arithmetic, sign-extending)

    # Data movement
    MOV = 10     # a = b
    MOVI = 11    # a = immediate(b, c)  -- load immediate value into register a

    # Memory (word-addressed, 32-bit)
    LOAD = 12    # a = mem_word[b + c]
    STORE = 13   # mem_word[b + c] = a

    # Control flow
    JMP = 14     # PC = a  (unconditional jump, constant target)
    # Note: opcodes 15-25 already allocated below
    BEQ = 15     # if reg[a] == reg[b]: PC = c
    BNE = 16     # if reg[a] != reg[b]: PC = c
    BLT = 17     # if reg[a] <  reg[b]: PC = c  (signed)
    BGE = 18     # if reg[a] >= reg[b]: PC = c  (signed)
    BLTU = 19    # if reg[a] <  reg[b]: PC = c  (unsigned)
    BGEU = 20    # if reg[a] >= reg[b]: PC = c  (unsigned)

    # Comparison (set less than)
    SLT = 21     # a = (b < c) ? 1 : 0  (signed)
    SLTU = 22    # a = (b < c) ? 1 : 0  (unsigned)

    # System
    NOP = 23     # no operation
    HALT = 24    # stop execution
    ECALL = 25   # system call

    # Byte/halfword memory (byte-addressed)
    LOADB = 26   # a = mem_byte[b + c] (zero-extend)
    LOADBS = 27  # a = mem_byte[b + c] (sign-extend)
    LOADH = 28   # a = mem_half[b + c] (zero-extend)
    LOADHS = 29  # a = mem_half[b + c] (sign-extend)
    STOREB = 30  # mem_byte[b + c] = a (low 8 bits)
    STOREH = 31  # mem_half[b + c] = a (low 16 bits)

    # Indirect jump (for function calls)
    JMPR = 32    # PC = reg[a] (jump to address in register)

    # Division/remainder (RV32M)
    DIV = 33     # a = b / c  (signed)
    DIVU = 34    # a = b / c  (unsigned)
    REM = 35     # a = b % c  (signed)
    REMU = 36    # a = b % c  (unsigned)

    # Multiply-high (RV32M — upper 32 bits of 64-bit product)
    MULH = 37    # a = (b * c) >> 32  (signed × signed)
    MULHU = 38   # a = (b * c) >> 32  (unsigned × unsigned)
    MULHSU = 39  # a = (b * c) >> 32  (signed × unsigned)

    N_OPCODES = 40


# Opcode categories for the transformer pipeline
ALU_OPS = {Opcode.ADD, Opcode.SUB, Opcode.MUL,
           Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.NOT,
           Opcode.SHL, Opcode.SHR, Opcode.SRA,
           Opcode.SLT, Opcode.SLTU}

BRANCH_OPS = {Opcode.JMP, Opcode.BEQ, Opcode.BNE,
              Opcode.BLT, Opcode.BGE, Opcode.BLTU, Opcode.BGEU}

MEMORY_OPS = {Opcode.LOAD, Opcode.STORE}

DATA_OPS = {Opcode.MOV, Opcode.MOVI}

SYSTEM_OPS = {Opcode.NOP, Opcode.HALT, Opcode.ECALL}


@dataclass
class Instruction:
    """A single NISA instruction."""
    opcode: Opcode
    a: int = 0  # destination / first operand
    b: int = 0  # second operand
    c: int = 0  # third operand / immediate / target

    def encode(self) -> tuple[int, int, int, int]:
        """Encode as a 4-tuple of integers."""
        return (int(self.opcode), self.a, self.b, self.c)

    def __repr__(self):
        name = self.opcode.name
        if self.opcode == Opcode.MOVI:
            # MOVI rd, imm  (imm stored across b and c as 16-bit halves)
            imm = (self.b << 16) | (self.c & 0xFFFF)
            return f"MOVI r{self.a}, {imm}"
        elif self.opcode in ALU_OPS:
            return f"{name} r{self.a}, r{self.b}, r{self.c}"
        elif self.opcode == Opcode.MOV:
            return f"MOV r{self.a}, r{self.b}"
        elif self.opcode == Opcode.LOAD:
            return f"LOAD r{self.a}, [r{self.b} + {self.c}]"
        elif self.opcode == Opcode.STORE:
            return f"STORE r{self.a}, [r{self.b} + {self.c}]"
        elif self.opcode == Opcode.JMP:
            return f"JMP {self.a}"
        elif self.opcode in BRANCH_OPS:
            return f"{name} r{self.a}, r{self.b}, {self.c}"
        elif self.opcode == Opcode.NOP:
            return "NOP"
        elif self.opcode == Opcode.HALT:
            return "HALT"
        elif self.opcode == Opcode.ECALL:
            return "ECALL"
        return f"{name} {self.a}, {self.b}, {self.c}"


def movi(rd: int, imm: int) -> Instruction:
    """Helper: load a 32-bit immediate into register rd.

    The immediate is split into upper 16 bits (field b) and lower 16 bits (field c).
    """
    imm = imm & 0xFFFFFFFF  # ensure 32-bit
    return Instruction(Opcode.MOVI, a=rd, b=(imm >> 16) & 0xFFFF, c=imm & 0xFFFF)


def add(rd: int, rs1: int, rs2: int) -> Instruction:
    return Instruction(Opcode.ADD, a=rd, b=rs1, c=rs2)


def sub(rd: int, rs1: int, rs2: int) -> Instruction:
    return Instruction(Opcode.SUB, a=rd, b=rs1, c=rs2)


def halt() -> Instruction:
    return Instruction(Opcode.HALT)


def nop() -> Instruction:
    return Instruction(Opcode.NOP)


# Number of general-purpose registers (matches RISC-V)
N_REGS = 32
# Register x0 is hardwired to zero
ZERO_REG = 0
