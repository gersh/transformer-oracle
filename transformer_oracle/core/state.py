"""
State tensor encoding/decoding for the transformer CPU.

The state is a 2D tensor of shape (d, n) where:
  - d = state dimension per column (bipolar encoding of values + metadata)
  - n = number of columns (registers + PC + memory + instructions)

Each 32-bit integer value is encoded in bipolar form: {-1, +1} per bit,
where +1 represents binary 1 and -1 represents binary 0.

Column layout:
  [0]        x0 (hardwired zero)
  [1..31]    General-purpose registers x1-x31
  [32]       Program counter
  [33..N_DATA+32]  Data memory words
  [N_DATA+33..]    Instruction memory (encoded NISA instructions)

Row layout per column:
  [0..31]    Value bits (bipolar)
  [32..42]   Position encoding (column address, 11 bits)
  [43]       Column type: -1=register, 0=PC, +1=data_mem, reserved for instr
"""

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass

from .nisa import Instruction, Opcode, N_REGS


# Layout constants
VALUE_BITS = 32
POSITION_BITS = 11  # supports up to 2048 columns
COLUMN_TYPE_BITS = 1

# Row offsets
VALUE_START = 0
VALUE_END = VALUE_BITS  # 32
POSITION_START = VALUE_END  # 32
POSITION_END = POSITION_START + POSITION_BITS  # 43
TYPE_START = POSITION_END  # 43
TYPE_END = TYPE_START + COLUMN_TYPE_BITS  # 44

# Instruction encoding rows (within instruction columns)
# An instruction has 4 fields: opcode(6 bits), a(5 bits), b(16 bits), c(16 bits)
INSTR_OPCODE_BITS = 6
INSTR_A_BITS = 5
INSTR_B_BITS = 16
INSTR_C_BITS = 16

# State dimension per column
D_STATE = TYPE_END  # 44

# Column type tags
COL_TYPE_REG = -1.0
COL_TYPE_PC = 0.0
COL_TYPE_DATAMEM = 1.0
COL_TYPE_INSTRMEM = 2.0  # reserved but not used in value bits encoding


@dataclass
class StateConfig:
    """Configuration for the state tensor dimensions."""
    n_regs: int = N_REGS          # 32 registers (including x0)
    n_data_words: int = 256       # 256 words of data memory
    n_instr_slots: int = 512      # max 512 instructions (can grow)
    d_state: int = D_STATE        # rows per column

    @property
    def n_columns(self) -> int:
        """Total number of columns in the state tensor."""
        return self.n_regs + 1 + self.n_data_words + self.n_instr_slots

    @property
    def pc_col(self) -> int:
        return self.n_regs  # column 32

    @property
    def data_start(self) -> int:
        return self.n_regs + 1  # column 33

    @property
    def data_end(self) -> int:
        return self.data_start + self.n_data_words

    @property
    def instr_start(self) -> int:
        return self.data_end

    @property
    def instr_end(self) -> int:
        return self.instr_start + self.n_instr_slots


DEFAULT_CONFIG = StateConfig()


def int_to_bipolar(value: int, n_bits: int = 32) -> torch.Tensor:
    """Convert an integer to bipolar {-1, +1} representation.

    Bit 0 (LSB) is at index 0, bit n_bits-1 (MSB) at index n_bits-1.
    Binary 1 → +1.0, binary 0 → -1.0.
    """
    value = value & ((1 << n_bits) - 1)  # mask to n_bits
    bits = torch.zeros(n_bits, dtype=torch.float64)
    for i in range(n_bits):
        bits[i] = 1.0 if (value >> i) & 1 else -1.0
    return bits


def bipolar_to_int(bipolar: torch.Tensor, signed: bool = False) -> int:
    """Convert bipolar {-1, +1} tensor back to an integer.

    Values > 0 are treated as binary 1, <= 0 as binary 0.
    """
    n_bits = bipolar.shape[0]
    result = 0
    for i in range(n_bits):
        if bipolar[i].item() > 0:
            result |= (1 << i)
    if signed and n_bits > 0 and (result >> (n_bits - 1)) & 1:
        result -= (1 << n_bits)
    return result


def position_encoding(col_idx: int) -> torch.Tensor:
    """Encode a column index as bipolar position bits."""
    return int_to_bipolar(col_idx, POSITION_BITS)


class StateTensor:
    """Manages the state tensor for the transformer CPU.

    The state tensor has shape (d_state, n_columns) and uses float64
    for precision during analytical weight computation.
    """

    def __init__(self, config: Optional[StateConfig] = None):
        self.config = config or DEFAULT_CONFIG
        c = self.config
        self.data = torch.zeros(c.d_state, c.n_columns, dtype=torch.float64)
        self._init_structure()

    def _init_structure(self):
        """Initialize position encodings and column types."""
        c = self.config
        for col in range(c.n_columns):
            # Position encoding
            self.data[POSITION_START:POSITION_END, col] = position_encoding(col)

            # Column type tag
            if col < c.n_regs:
                self.data[TYPE_START, col] = COL_TYPE_REG
            elif col == c.pc_col:
                self.data[TYPE_START, col] = COL_TYPE_PC
            elif c.data_start <= col < c.data_end:
                self.data[TYPE_START, col] = COL_TYPE_DATAMEM
            else:
                self.data[TYPE_START, col] = COL_TYPE_INSTRMEM

        # x0 is hardwired to zero
        self.data[VALUE_START:VALUE_END, 0] = int_to_bipolar(0)
        # PC starts at 0
        self.set_pc(0)

    def set_register(self, reg_idx: int, value: int):
        """Set a register value. Register 0 is ignored (hardwired zero)."""
        if reg_idx == 0:
            return  # x0 is always zero
        assert 0 < reg_idx < self.config.n_regs, f"Invalid register: {reg_idx}"
        self.data[VALUE_START:VALUE_END, reg_idx] = int_to_bipolar(value)

    def get_register(self, reg_idx: int) -> int:
        """Get a register value as an integer."""
        if reg_idx == 0:
            return 0
        assert 0 < reg_idx < self.config.n_regs, f"Invalid register: {reg_idx}"
        return bipolar_to_int(self.data[VALUE_START:VALUE_END, reg_idx])

    def set_pc(self, value: int):
        """Set the program counter."""
        self.data[VALUE_START:VALUE_END, self.config.pc_col] = int_to_bipolar(value)

    def get_pc(self) -> int:
        """Get the program counter value."""
        return bipolar_to_int(self.data[VALUE_START:VALUE_END, self.config.pc_col])

    def set_memory(self, addr: int, value: int):
        """Set a data memory word."""
        assert 0 <= addr < self.config.n_data_words, f"Memory address out of range: {addr}"
        col = self.config.data_start + addr
        self.data[VALUE_START:VALUE_END, col] = int_to_bipolar(value)

    def get_memory(self, addr: int) -> int:
        """Get a data memory word."""
        assert 0 <= addr < self.config.n_data_words, f"Memory address out of range: {addr}"
        col = self.config.data_start + addr
        return bipolar_to_int(self.data[VALUE_START:VALUE_END, col])

    def load_program(self, instructions: list[Instruction]):
        """Load a program into instruction memory.

        Each instruction is encoded into a column:
        - Bits 0..31 encode the 4 fields packed as a 32-bit word:
          [opcode(6)][a(5)][b(16)][c(16)] = 43 bits, but we pack into 32 bits
          by using the value rows for the primary encoding.

        For Phase 1, we encode each field directly into the value bits:
          bits 0-5: opcode (6 bits)
          bits 6-10: a (5 bits, register index)
          bits 11-26: b (16 bits)
          bits 27-31: c low 5 bits (remaining c bits overflow — will expand later)

        Actually, for simplicity in Phase 1, we store the instruction as a
        packed 32-bit word where we allocate bits efficiently.
        """
        assert len(instructions) <= self.config.n_instr_slots, \
            f"Program too large: {len(instructions)} > {self.config.n_instr_slots}"

        for i, instr in enumerate(instructions):
            col = self.config.instr_start + i
            packed = self._pack_instruction(instr)
            self.data[VALUE_START:VALUE_END, col] = int_to_bipolar(packed)

    def get_instruction_at(self, instr_idx: int) -> Instruction:
        """Decode an instruction from instruction memory."""
        col = self.config.instr_start + instr_idx
        packed = bipolar_to_int(self.data[VALUE_START:VALUE_END, col])
        return self._unpack_instruction(packed)

    def _pack_instruction(self, instr: Instruction) -> int:
        """Pack an instruction into a 32-bit word.

        Layout (LSB first):
          bits 0-5:   opcode (6 bits, supports up to 64 opcodes -- the ISA has 40)
          bits 6-10:  a field (5 bits, register index 0-31)
          bits 11-15: b field low 5 bits (register index for reg-reg ops)
          bits 16-20: c field low 5 bits (register index for reg-reg ops)
          bits 16-27: immediate/target bits (12 bits for branch offsets, etc.)

        For MOVI, the immediate is packed into the high bits (21 bits).
        """
        opv = int(instr.opcode)
        if opv > 0x3F:      # fail loud: opcode does not fit the 6-bit field
            raise ValueError(
                f"opcode {instr.opcode} ({opv}) exceeds the 6-bit instruction field; "
                f"widen the encoding to add it")
        op = opv & 0x3F                      # 6 bits
        a = instr.a & 0x1F                   # 5 bits
        b = instr.b & 0x1F                   # 5 bits
        c = instr.c & 0x1F                   # 5 bits
        # For MOVI, pack the immediate into the high bits (21 bits)
        if instr.opcode == Opcode.MOVI:
            imm = ((instr.b & 0xFFFF) << 16) | (instr.c & 0xFFFF)
            return op | (a << 6) | ((imm & 0x1FFFFF) << 11)
        # For branches, c is a target address (use more bits)
        if instr.opcode in {Opcode.BEQ, Opcode.BNE, Opcode.BLT,
                            Opcode.BGE, Opcode.BLTU, Opcode.BGEU, Opcode.JMP}:
            target = instr.c & 0xFFF if instr.opcode != Opcode.JMP else instr.a & 0xFFF
            if instr.opcode == Opcode.JMP:
                return op | (target << 6)
            return op | (a << 6) | (b << 11) | (target << 16)

        return op | (a << 6) | (b << 11) | (c << 16)

    def _unpack_instruction(self, packed: int) -> Instruction:
        """Unpack a 32-bit word back into an Instruction (6-bit opcode)."""
        op = Opcode(packed & 0x3F)
        if op == Opcode.MOVI:
            a = (packed >> 6) & 0x1F
            imm = (packed >> 11) & 0x1FFFFF
            return Instruction(op, a=a, b=(imm >> 16) & 0xFFFF, c=imm & 0xFFFF)
        if op == Opcode.JMP:
            target = (packed >> 6) & 0xFFF
            return Instruction(op, a=target)
        if op in {Opcode.BEQ, Opcode.BNE, Opcode.BLT,
                  Opcode.BGE, Opcode.BLTU, Opcode.BGEU}:
            a = (packed >> 6) & 0x1F
            b = (packed >> 11) & 0x1F
            target = (packed >> 16) & 0xFFF
            return Instruction(op, a=a, b=b, c=target)

        a = (packed >> 6) & 0x1F
        b = (packed >> 11) & 0x1F
        c = (packed >> 16) & 0x1F
        return Instruction(op, a=a, b=b, c=c)

    def snap_to_bipolar(self):
        """Error correction: snap all value bits back to exact {-1, +1}.

        This prevents floating-point errors from accumulating over
        many forward passes. Values > 0 → +1, values <= 0 → -1.
        """
        values = self.data[VALUE_START:VALUE_END, :]
        self.data[VALUE_START:VALUE_END, :] = torch.where(values > 0, 1.0, -1.0)
        # Re-enforce x0 = 0
        self.data[VALUE_START:VALUE_END, 0] = int_to_bipolar(0)

    def dump_registers(self) -> dict[int, int]:
        """Return all register values as a dict."""
        regs = {}
        for i in range(self.config.n_regs):
            regs[i] = self.get_register(i) if i > 0 else 0
        return regs

    def clone(self) -> 'StateTensor':
        """Create a deep copy of this state."""
        new = StateTensor.__new__(StateTensor)
        new.config = self.config
        new.data = self.data.clone()
        return new

    def __repr__(self):
        pc = self.get_pc()
        regs = {i: self.get_register(i) for i in range(1, 8)}  # show first 7 regs
        return f"StateTensor(PC={pc}, regs={regs})"
