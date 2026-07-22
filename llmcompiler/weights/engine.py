"""
Weight Generation Engine.

Top-level module that generates all transformer weight matrices for the
10-layer CPU transformer. In Phase 1, this is a reference implementation
that directly executes NISA instructions using the bipolar arithmetic
functions. The transformer weights themselves are constructed analytically
(no training) and will be used in Phase 2+ for actual transformer execution.

Phase 1 Architecture:
  The "engine" directly interprets instructions using:
  - instruction_fetch: reads PC, selects instruction from memory
  - decode: unpacks instruction fields
  - execute: runs ALU via bipolar_arithmetic
  - writeback: stores result to destination register
  - error_correct: snaps values to bipolar

This serves as the ground truth reference for verifying the transformer
implementation in later phases.
"""

import torch
from ..core.state import StateTensor, int_to_bipolar, bipolar_to_int
from ..core.nisa import (
    Instruction, Opcode, ALU_OPS, BRANCH_OPS, MEMORY_OPS, DATA_OPS
)
from .bipolar_arithmetic import (
    bipolar_add_32bit, bipolar_sub_32bit,
    bipolar_and_32bit, bipolar_or_32bit, bipolar_xor_32bit, bipolar_not_32bit,
    bipolar_shl_32bit, bipolar_shr_32bit, bipolar_sra_32bit,
)
from .instruction_fetch import fetch_instruction_direct
from .writeback import writeback_register, update_pc
from .error_correction import error_correct


def decode_instruction(state: StateTensor, instr_idx: int) -> Instruction:
    """Decode an instruction from instruction memory."""
    return state.get_instruction_at(instr_idx)


def read_register_bipolar(state: StateTensor, reg_idx: int) -> torch.Tensor:
    """Read a register's bipolar value from the state tensor."""
    if reg_idx == 0:
        return int_to_bipolar(0)
    from ..core.state import VALUE_START, VALUE_END
    return state.data[VALUE_START:VALUE_END, reg_idx].clone()


def execute_one_cycle(state: StateTensor) -> bool:
    """Execute one instruction cycle on the state tensor.

    This is the Phase 1 reference executor that directly interprets
    instructions using bipolar arithmetic. In Phase 2+, this will be
    replaced by a single transformer forward pass.

    Returns:
        True if execution should continue, False if HALT was reached.
    """
    # --- Stage 1: Instruction Fetch ---
    instr_idx = fetch_instruction_direct(state)
    if instr_idx >= state.config.n_instr_slots:
        return False  # PC out of bounds

    # --- Stage 2: Decode ---
    instr = decode_instruction(state, instr_idx)

    # --- System instructions ---
    if instr.opcode == Opcode.HALT:
        return False
    if instr.opcode == Opcode.NOP:
        update_pc(state, state.get_pc() + 1)
        error_correct(state)
        return True

    # --- Stage 3: Execute ---
    next_pc = state.get_pc() + 1  # default: advance to next instruction

    if instr.opcode == Opcode.MOVI:
        # Load immediate into register a
        imm = ((instr.b & 0xFFFF) << 16) | (instr.c & 0xFFFF)
        # But since we pack into 22 bits in the instruction encoding,
        # use the actual instruction fields
        result = int_to_bipolar(imm)
        writeback_register(state, instr.a, result)

    elif instr.opcode == Opcode.MOV:
        # Copy register b to register a
        val = read_register_bipolar(state, instr.b)
        writeback_register(state, instr.a, val)

    elif instr.opcode in ALU_OPS:
        # Read operands
        rs1_val = read_register_bipolar(state, instr.b)
        rs2_val = read_register_bipolar(state, instr.c)

        # Execute ALU operation
        if instr.opcode == Opcode.ADD:
            result = bipolar_add_32bit(rs1_val, rs2_val)
        elif instr.opcode == Opcode.SUB:
            result = bipolar_sub_32bit(rs1_val, rs2_val)
        elif instr.opcode == Opcode.AND:
            result = bipolar_and_32bit(rs1_val, rs2_val)
        elif instr.opcode == Opcode.OR:
            result = bipolar_or_32bit(rs1_val, rs2_val)
        elif instr.opcode == Opcode.XOR:
            result = bipolar_xor_32bit(rs1_val, rs2_val)
        elif instr.opcode == Opcode.NOT:
            result = bipolar_not_32bit(rs1_val)
        elif instr.opcode == Opcode.SHL:
            shift_amt = bipolar_to_int(rs2_val) & 0x1F
            result = bipolar_shl_32bit(rs1_val, shift_amt)
        elif instr.opcode == Opcode.SHR:
            shift_amt = bipolar_to_int(rs2_val) & 0x1F
            result = bipolar_shr_32bit(rs1_val, shift_amt)
        elif instr.opcode == Opcode.SRA:
            shift_amt = bipolar_to_int(rs2_val) & 0x1F
            result = bipolar_sra_32bit(rs1_val, shift_amt)
        elif instr.opcode == Opcode.SLT:
            va = bipolar_to_int(rs1_val, signed=True)
            vb = bipolar_to_int(rs2_val, signed=True)
            result = int_to_bipolar(1 if va < vb else 0)
        elif instr.opcode == Opcode.SLTU:
            va = bipolar_to_int(rs1_val, signed=False)
            vb = bipolar_to_int(rs2_val, signed=False)
            result = int_to_bipolar(1 if va < vb else 0)
        elif instr.opcode == Opcode.MUL:
            va = bipolar_to_int(rs1_val, signed=True)
            vb = bipolar_to_int(rs2_val, signed=True)
            result = int_to_bipolar((va * vb) & 0xFFFFFFFF)
        else:
            raise ValueError(f"Unhandled ALU op: {instr.opcode}")

        writeback_register(state, instr.a, result)

    elif instr.opcode in {Opcode.MULHU, Opcode.MULH, Opcode.MULHSU,
                          Opcode.DIV, Opcode.DIVU, Opcode.REM, Opcode.REMU}:
        # Multiply-high (RV32M) and divide/remainder. Computed in Python ints.
        au = bipolar_to_int(read_register_bipolar(state, instr.b), signed=False)
        bu = bipolar_to_int(read_register_bipolar(state, instr.c), signed=False)
        as_ = au - 2**32 if au >= 2**31 else au
        bs_ = bu - 2**32 if bu >= 2**31 else bu
        if instr.opcode == Opcode.MULHU:
            res = ((au * bu) >> 32) & 0xFFFFFFFF
        elif instr.opcode == Opcode.MULH:
            res = ((as_ * bs_) >> 32) & 0xFFFFFFFF
        elif instr.opcode == Opcode.MULHSU:
            res = ((as_ * bu) >> 32) & 0xFFFFFFFF
        elif instr.opcode == Opcode.DIVU:
            res = 0xFFFFFFFF if bu == 0 else (au // bu) & 0xFFFFFFFF
        elif instr.opcode == Opcode.REMU:
            res = au if bu == 0 else (au % bu)
        elif instr.opcode == Opcode.DIV:
            if bs_ == 0:
                res = 0xFFFFFFFF
            else:
                q = abs(as_) // abs(bs_)
                if (as_ < 0) != (bs_ < 0): q = -q
                res = q & 0xFFFFFFFF
        else:  # REM
            if bs_ == 0:
                res = as_ & 0xFFFFFFFF
            else:
                r = abs(as_) % abs(bs_)
                if as_ < 0: r = -r
                res = r & 0xFFFFFFFF
        writeback_register(state, instr.a, int_to_bipolar(res))

    elif instr.opcode == Opcode.LOAD:
        # LOAD rd, [rs1 + offset]
        base = bipolar_to_int(read_register_bipolar(state, instr.b))
        addr = (base + instr.c) & 0xFFFFFFFF
        if addr < state.config.n_data_words:
            val = state.get_memory(addr)
            writeback_register(state, instr.a, int_to_bipolar(val))

    elif instr.opcode == Opcode.STORE:
        # STORE rs, [rb + offset]
        val = bipolar_to_int(read_register_bipolar(state, instr.a))
        base = bipolar_to_int(read_register_bipolar(state, instr.b))
        addr = (base + instr.c) & 0xFFFFFFFF
        if addr < state.config.n_data_words:
            state.set_memory(addr, val)

    elif instr.opcode == Opcode.JMP:
        next_pc = instr.a

    elif instr.opcode in BRANCH_OPS:
        rs1_val = read_register_bipolar(state, instr.a)
        rs2_val = read_register_bipolar(state, instr.b)
        va = bipolar_to_int(rs1_val, signed=True)
        vb = bipolar_to_int(rs2_val, signed=True)
        va_u = bipolar_to_int(rs1_val, signed=False)
        vb_u = bipolar_to_int(rs2_val, signed=False)

        taken = False
        if instr.opcode == Opcode.BEQ:
            taken = (va == vb)
        elif instr.opcode == Opcode.BNE:
            taken = (va != vb)
        elif instr.opcode == Opcode.BLT:
            taken = (va < vb)
        elif instr.opcode == Opcode.BGE:
            taken = (va >= vb)
        elif instr.opcode == Opcode.BLTU:
            taken = (va_u < vb_u)
        elif instr.opcode == Opcode.BGEU:
            taken = (va_u >= vb_u)

        if taken:
            next_pc = instr.c

    # --- Stage 5: PC Update ---
    update_pc(state, next_pc)

    # --- Stage 6: Error Correction ---
    error_correct(state)

    return True
