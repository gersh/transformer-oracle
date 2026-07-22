"""
RV32I to NISA Translator.

Translates parsed RV32I instructions into NISA instruction sequences.
Each RV32I instruction maps to 1-4 NISA instructions.

Key translations:
  - R-type (add, sub, etc.) → 1 NISA instruction (direct mapping)
  - I-type (addi, etc.) → 2 NISA instructions (MOVI imm + ALU op)
  - Load/Store → NISA LOAD/STORE with address computation
  - Branches → NISA branches (direct mapping)
  - JAL/JALR → MOVI + JMP + link register save
  - LUI/AUIPC → MOVI + SHL or MOVI with shifted immediate
  - Pseudo-instructions → expanded to base instructions first

Memory model:
  RISC-V uses byte-addressable memory, but our NISA uses word-addressed
  memory. For Phase 3, we support word-aligned LW/SW only (4-byte aligned).
  Sub-word loads (LB, LH) and stores (SB, SH) will be added in Phase 4.

Stack:
  SP (x2) is initialized to the top of data memory. The stack grows
  downward. Stack accesses are translated to NISA LOAD/STORE with
  computed word addresses (byte_addr / 4).
"""

from ..core.nisa import Instruction, Opcode, movi, add, halt, nop
from .rv32i_parser import RV32IInstr, RV32IOp, _R_TYPE, _I_TYPE, _LOAD_TYPE, _STORE_TYPE, _BRANCH_TYPE


# Temporary register used by the translator for intermediate values.
# Must not conflict with user registers. We use r30 and r31.
_TMP1 = 30  # scratch register for translator
_TMP2 = 31  # scratch register for translator


def translate_program(rv_instrs: list[RV32IInstr],
                      rv_labels: dict[str, int],
                      stack_top: int = 252,
                      data_image: dict[int, int] = None) -> list[Instruction]:
    """Translate a complete RV32I program to NISA.

    Args:
        rv_instrs: parsed RV32I instructions
        rv_labels: label → RV32I instruction index
        stack_top: initial stack pointer (word address in data memory)
        data_image: byte-addr → byte for initialized globals (.data/.bss initializers);
            emitted as store instructions in the preamble so the program self-initializes
            regardless of how memory is seeded at execution time.

    Returns:
        list of NISA instructions
    """
    # Phase 1: Translate each RV32I instruction to NISA, tracking
    # the mapping from RV32I index to NISA index for label resolution.

    nisa_instrs: list[Instruction] = []
    # Map RV32I instruction index → NISA instruction index
    rv_to_nisa: dict[int, int] = {}

    # Initialize SP (x2) to stack top (byte address)
    nisa_instrs.append(movi(2, stack_top * 4))
    # Initialize RA (x1) to point to halt instruction at end of program.
    ra_init_idx = len(nisa_instrs)
    nisa_instrs.append(movi(1, 0))  # placeholder, patched below

    # Emit initialized-data stores (self-initializing .data/.bss), so the program
    # works under any execution harness without externally seeding memory.
    if data_image:
        _emit_data_init(nisa_instrs, data_image)

    # If _start is not at the beginning, emit a jump to it.
    # This handles GCC emitting helper functions before _start.
    start_jump_idx = None
    if '_start' in rv_labels and rv_labels['_start'] > 0:
        start_jump_idx = len(nisa_instrs)
        nisa_instrs.append(Instruction(Opcode.JMP, a=0))  # patched below

    init_offset = len(nisa_instrs)

    for rv_idx, rv_instr in enumerate(rv_instrs):
        rv_to_nisa[rv_idx] = len(nisa_instrs)
        _translate_one(rv_instr, nisa_instrs)

    # Map RV32I end to NISA end (for labels pointing past last instruction)
    rv_to_nisa[len(rv_instrs)] = len(nisa_instrs)

    # Append a HALT for ret-from-_start to land on
    halt_idx = len(nisa_instrs)
    nisa_instrs.append(halt())

    # Patch RA initialization to point to the halt
    nisa_instrs[ra_init_idx] = movi(1, halt_idx)

    # Patch jump-to-_start if needed
    if start_jump_idx is not None and '_start' in rv_labels:
        start_nisa_idx = rv_to_nisa.get(rv_labels['_start'], init_offset)
        nisa_instrs[start_jump_idx] = Instruction(Opcode.JMP, a=start_nisa_idx)

    # Phase 2: Resolve label references in branch/jump instructions.
    # NISA branch targets need to be NISA instruction indices.
    _resolve_labels(nisa_instrs, rv_instrs, rv_labels, rv_to_nisa, init_offset)

    return nisa_instrs


def _translate_one(rv: RV32IInstr, out: list[Instruction]):
    """Translate a single RV32I instruction, appending NISA instructions to out."""

    op = rv.op

    # ── Pseudo-instruction expansion ──

    if op == RV32IOp.NOP:
        out.append(nop())
        return

    if op == RV32IOp.MV:
        # mv rd, rs1 → NISA MOV rd, rs1
        out.append(Instruction(Opcode.MOV, a=rv.rd, b=rv.rs1))
        return

    if op == RV32IOp.LI:
        # li rd, imm → NISA MOVI rd, imm
        _emit_load_immediate(out, rv.rd, rv.imm)
        return

    if op == RV32IOp.NEG:
        # neg rd, rs → sub rd, x0, rs
        out.append(Instruction(Opcode.SUB, a=rv.rd, b=0, c=rv.rs2))
        return

    if op == RV32IOp.NOT:
        # not rd, rs → xori rd, rs, -1
        _emit_load_immediate(out, _TMP1, 0xFFFFFFFF)
        out.append(Instruction(Opcode.XOR, a=rv.rd, b=rv.rs1, c=_TMP1))
        return

    if op == RV32IOp.SEQZ:
        # seqz rd, rs → sltiu rd, rs, 1
        _emit_load_immediate(out, _TMP1, 1)
        out.append(Instruction(Opcode.SLTU, a=rv.rd, b=rv.rs1, c=_TMP1))
        return

    if op == RV32IOp.SNEZ:
        # snez rd, rs → sltu rd, x0, rs
        out.append(Instruction(Opcode.SLTU, a=rv.rd, b=0, c=rv.rs2))
        return

    if op == RV32IOp.J:
        # j label → jal x0, label (label resolved in phase 2)
        out.append(Instruction(Opcode.JMP, a=0))  # target patched later
        return

    if op == RV32IOp.JR:
        # jr rs → jalr x0, rs, 0 — indirect jump to address in register
        out.append(Instruction(Opcode.JMPR, a=rv.rs1))
        return

    if op == RV32IOp.RET:
        # ret → jalr x0, x1, 0 → jump to address in ra (x1)
        out.append(Instruction(Opcode.JMPR, a=1))  # jump to reg[ra]
        return

    if op in (RV32IOp.BEQZ, RV32IOp.BNEZ, RV32IOp.BLTZ,
              RV32IOp.BGEZ, RV32IOp.BLEZ, RV32IOp.BGTZ):
        _translate_branch_zero(rv, out)
        return

    if op in (RV32IOp.BLE, RV32IOp.BGT, RV32IOp.BLEU, RV32IOp.BGTU):
        # Pseudo-instructions with swapped operands
        swap_map = {
            RV32IOp.BLE: Opcode.BGE,   # ble a,b → bge b,a
            RV32IOp.BGT: Opcode.BLT,   # bgt a,b → blt b,a
            RV32IOp.BLEU: Opcode.BGEU, # bleu a,b → bgeu b,a
            RV32IOp.BGTU: Opcode.BLTU, # bgtu a,b → bltu b,a
        }
        # Note: operands are swapped (rs2 as first arg, rs1 as second)
        out.append(Instruction(swap_map[op], a=rv.rs2, b=rv.rs1, c=0))
        return

    if op in (RV32IOp.CALL, RV32IOp.TAIL):
        # call label → jal ra, label / tail label → jal x0, label
        # Same as JAL translation
        if rv.rd != 0:
            out.append(movi(rv.rd, 0))  # return addr, patched in phase 2
        out.append(Instruction(Opcode.JMP, a=0))  # target patched in phase 2
        return

    if op in (RV32IOp.LA, RV32IOp.LLA):
        # la/lla rd, symbol — load address. The parser resolves data symbols to their
        # absolute byte address in rv.imm (0 if the symbol is unknown / a text label).
        _emit_load_immediate(out, rv.rd, rv.imm & 0xFFFFFFFF)
        return

    # ── R-type: register-register ALU ──

    # sgt/sgtu rd,rs1,rs2 = "rs1 > rs2" = slt/sltu with operands swapped
    if op == RV32IOp.SGT:
        out.append(Instruction(Opcode.SLT, a=rv.rd, b=rv.rs2, c=rv.rs1))
        return
    if op == RV32IOp.SGTU:
        out.append(Instruction(Opcode.SLTU, a=rv.rd, b=rv.rs2, c=rv.rs1))
        return

    if op in _R_TYPE:
        nisa_op = _R_TYPE_MAP.get(op)
        if nisa_op is None:
            raise NotImplementedError(
                f"rv32i_to_nisa: R-type op {op.name} has no NISA mapping (would emit "
                f"nothing and silently drop the instruction)")
        out.append(Instruction(nisa_op, a=rv.rd, b=rv.rs1, c=rv.rs2))
        return

    # ── I-type: register-immediate ALU ──

    if op in _I_TYPE:
        _translate_i_type(rv, out)
        return

    # ── Load instructions ──

    if op in _LOAD_TYPE:
        _translate_load(rv, out)
        return

    # ── Store instructions ──

    if op in _STORE_TYPE:
        _translate_store(rv, out)
        return

    # ── Branch instructions ──

    if op in _BRANCH_TYPE:
        nisa_op = _BRANCH_MAP.get(op)
        if nisa_op is not None:
            # Target is resolved in phase 2; emit with placeholder
            out.append(Instruction(nisa_op, a=rv.rs1, b=rv.rs2, c=0))
        return

    # ── JAL (jump and link) ──

    if op == RV32IOp.JAL:
        # jal rd, label → save PC+1 in rd, then jump to label
        # NISA: MOVI rd, <return_addr>; JMP <target>
        # Return addr and target resolved in phase 2
        if rv.rd != 0:
            # Save return address (NISA index of instruction after the JMP)
            out.append(movi(rv.rd, 0))  # patched in phase 2
        out.append(Instruction(Opcode.JMP, a=0))  # patched in phase 2
        return

    # ── JALR (jump and link register) ──

    if op == RV32IOp.JALR:
        # jalr rd, rs1, imm → rd = PC+1; PC = (rs1 + imm) & ~1
        if rv.rd != 0:
            out.append(movi(rv.rd, 0))  # return addr, patched in phase 2
        # Compute target: rs1 + imm
        if rv.imm != 0:
            _emit_load_immediate(out, _TMP1, rv.imm)
            out.append(Instruction(Opcode.ADD, a=_TMP1, b=rv.rs1, c=_TMP1))
            out.append(Instruction(Opcode.JMPR, a=_TMP1))
        else:
            out.append(Instruction(Opcode.JMPR, a=rv.rs1))
        return

    # ── LUI (load upper immediate) ──

    if op == RV32IOp.LUI:
        # lui rd, imm → rd = imm << 12
        val = (rv.imm & 0xFFFFF) << 12
        _emit_load_immediate(out, rv.rd, val)
        return

    # ── AUIPC ──

    if op == RV32IOp.AUIPC:
        # auipc rd, imm → rd = PC + (imm << 12)
        # In NISA we don't have a PC-relative operation, so we compute
        # the absolute value. This is resolved during label resolution.
        val = (rv.imm & 0xFFFFF) << 12
        _emit_load_immediate(out, rv.rd, val)
        # Note: AUIPC needs the PC value added; for static compilation,
        # this is resolved at link time. For Phase 3, we handle simple cases.
        return

    # ── System ──

    if op == RV32IOp.ECALL:
        out.append(Instruction(Opcode.ECALL))
        return

    if op == RV32IOp.EBREAK:
        out.append(halt())
        return

    if op == RV32IOp.FENCE:
        out.append(nop())  # no-op for single-threaded
        return

    # Unknown instruction — emit NOP
    out.append(nop())


# ── R-type mapping ──

_R_TYPE_MAP = {
    RV32IOp.ADD: Opcode.ADD,
    RV32IOp.SUB: Opcode.SUB,
    RV32IOp.SLL: Opcode.SHL,
    RV32IOp.SLT: Opcode.SLT,
    RV32IOp.SLTU: Opcode.SLTU,
    RV32IOp.XOR: Opcode.XOR,
    RV32IOp.SRL: Opcode.SHR,
    RV32IOp.SRA: Opcode.SRA,
    RV32IOp.OR: Opcode.OR,
    RV32IOp.AND: Opcode.AND,
    RV32IOp.MUL: Opcode.MUL,
    RV32IOp.MULH: Opcode.MULH,
    RV32IOp.MULHU: Opcode.MULHU,
    RV32IOp.MULHSU: Opcode.MULHSU,
    RV32IOp.DIV: Opcode.DIV,
    RV32IOp.DIVU: Opcode.DIVU,
    RV32IOp.REM: Opcode.REM,
    RV32IOp.REMU: Opcode.REMU,
}

_BRANCH_MAP = {
    RV32IOp.BEQ: Opcode.BEQ,
    RV32IOp.BNE: Opcode.BNE,
    RV32IOp.BLT: Opcode.BLT,
    RV32IOp.BGE: Opcode.BGE,
    RV32IOp.BLTU: Opcode.BLTU,
    RV32IOp.BGEU: Opcode.BGEU,
}


def _translate_i_type(rv: RV32IInstr, out: list[Instruction]):
    """Translate I-type (register-immediate) instructions.

    Pattern: op rd, rs1, imm → MOVI tmp, imm; NISA_OP rd, rs1, tmp
    """
    imm = rv.imm & 0xFFFFFFFF

    # Map I-type to base ALU op
    op_map = {
        RV32IOp.ADDI: Opcode.ADD,
        RV32IOp.SLTI: Opcode.SLT,
        RV32IOp.SLTIU: Opcode.SLTU,
        RV32IOp.XORI: Opcode.XOR,
        RV32IOp.ORI: Opcode.OR,
        RV32IOp.ANDI: Opcode.AND,
        RV32IOp.SLLI: Opcode.SHL,
        RV32IOp.SRLI: Opcode.SHR,
        RV32IOp.SRAI: Opcode.SRA,
    }

    nisa_op = op_map.get(rv.op)
    if nisa_op is None:
        out.append(nop())
        return

    # For shift instructions, the immediate is the shift amount (5 bits)
    if rv.op in (RV32IOp.SLLI, RV32IOp.SRLI, RV32IOp.SRAI):
        imm = rv.imm & 0x1F

    # Load immediate into tmp register, then perform operation
    _emit_load_immediate(out, _TMP1, imm)
    out.append(Instruction(nisa_op, a=rv.rd, b=rv.rs1, c=_TMP1))


def _translate_load(rv: RV32IInstr, out: list[Instruction]):
    """Translate load instructions using byte-addressed memory.

    RV32I and NISA both use byte addresses now.
    """
    offset = rv.imm

    # Compute effective address: rs1 + offset → _TMP1
    if offset != 0:
        _emit_load_immediate(out, _TMP1, offset)
        out.append(Instruction(Opcode.ADD, a=_TMP1, b=rv.rs1, c=_TMP1))
    else:
        out.append(Instruction(Opcode.MOV, a=_TMP1, b=rv.rs1))

    if rv.op == RV32IOp.LW:
        # Word load: use LOAD with byte address in register
        # LOAD treats register value as byte address when >= mem_size//4
        # We use the byte-addressed path by ensuring address is large enough
        # or we can use a direct word read. For now, read 4 bytes via word load.
        out.append(Instruction(Opcode.LOAD, a=rv.rd, b=_TMP1, c=0))
    elif rv.op == RV32IOp.LBU:
        out.append(Instruction(Opcode.LOADB, a=rv.rd, b=_TMP1, c=0))
    elif rv.op == RV32IOp.LB:
        out.append(Instruction(Opcode.LOADBS, a=rv.rd, b=_TMP1, c=0))
    elif rv.op == RV32IOp.LHU:
        out.append(Instruction(Opcode.LOADH, a=rv.rd, b=_TMP1, c=0))
    elif rv.op == RV32IOp.LH:
        out.append(Instruction(Opcode.LOADHS, a=rv.rd, b=_TMP1, c=0))


def _translate_store(rv: RV32IInstr, out: list[Instruction]):
    """Translate store instructions using byte-addressed memory."""
    offset = rv.imm

    # Compute effective address: rs1 + offset → _TMP1
    if offset != 0:
        _emit_load_immediate(out, _TMP1, offset)
        out.append(Instruction(Opcode.ADD, a=_TMP1, b=rv.rs1, c=_TMP1))
    else:
        out.append(Instruction(Opcode.MOV, a=_TMP1, b=rv.rs1))

    if rv.op == RV32IOp.SW:
        out.append(Instruction(Opcode.STORE, a=rv.rs2, b=_TMP1, c=0))
    elif rv.op == RV32IOp.SB:
        out.append(Instruction(Opcode.STOREB, a=rv.rs2, b=_TMP1, c=0))
    elif rv.op == RV32IOp.SH:
        out.append(Instruction(Opcode.STOREH, a=rv.rs2, b=_TMP1, c=0))


def _translate_branch_zero(rv: RV32IInstr, out: list[Instruction]):
    """Translate branch-against-zero pseudo-instructions."""
    op_map = {
        RV32IOp.BEQZ: Opcode.BEQ,    # beqz rs, label → beq rs, x0, label
        RV32IOp.BNEZ: Opcode.BNE,    # bnez rs, label → bne rs, x0, label
        RV32IOp.BLTZ: Opcode.BLT,    # bltz rs, label → blt rs, x0, label
        RV32IOp.BGEZ: Opcode.BGE,    # bgez rs, label → bge rs, x0, label
        RV32IOp.BLEZ: Opcode.BGE,    # blez rs, label → bge x0, rs, label
        RV32IOp.BGTZ: Opcode.BLT,    # bgtz rs, label → blt x0, rs, label
    }
    nisa_op = op_map[rv.op]

    if rv.op in (RV32IOp.BLEZ, RV32IOp.BGTZ):
        # Swap: compare x0 against rs
        out.append(Instruction(nisa_op, a=0, b=rv.rs1, c=0))
    else:
        out.append(Instruction(nisa_op, a=rv.rs1, b=0, c=0))


def _emit_data_init(out: list[Instruction], data_image: dict[int, int]):
    """Emit stores that write the initialized-data image into memory at startup.

    Groups bytes into aligned 32-bit words and stores each word that has any nonzero
    byte (zero words match the default-zero memory, so they're skipped). Uses _TMP1 for
    the address and _TMP2 for the value."""
    words: dict[int, int] = {}
    for addr, byte in data_image.items():
        w = addr & ~3
        words[w] = words.get(w, 0) | ((byte & 0xFF) << (8 * (addr - w)))
    for w in sorted(words):
        val = words[w] & 0xFFFFFFFF
        if val == 0:
            continue
        _emit_load_immediate(out, _TMP1, w)
        _emit_load_immediate(out, _TMP2, val)
        out.append(Instruction(Opcode.STORE, a=_TMP2, b=_TMP1, c=0))


def _emit_load_immediate(out: list[Instruction], rd: int, value: int):
    """Emit NISA instruction(s) to load a 32-bit immediate into rd.

    MOVI's immediate is only 21 bits in the packed state encoding (the matmul
    transformer backends decode 21 bits), so a value >= 2^21 loaded via a single MOVI
    silently truncates there — which is exactly how LUI-derived constants broke on
    pure/analyt. Materialize as rd = (hi16 << 16) | lo16; the shift is a MUL by 2^16
    (65536 < 2^21) so every emitted immediate fits."""
    value = value & 0xFFFFFFFF
    if value < (1 << 21):
        out.append(movi(rd, value))
    else:
        scratch = _TMP2 if rd != _TMP2 else _TMP1
        out.append(movi(rd, (value >> 16) & 0xFFFF))
        out.append(movi(scratch, 1 << 16))                          # 65536 < 2^21
        out.append(Instruction(Opcode.MUL, a=rd, b=rd, c=scratch))  # rd = hi16 << 16
        out.append(movi(scratch, value & 0xFFFF))
        out.append(Instruction(Opcode.OR, a=rd, b=rd, c=scratch))   # rd |= lo16


def _resolve_labels(nisa_instrs: list[Instruction],
                    rv_instrs: list[RV32IInstr],
                    rv_labels: dict[str, int],
                    rv_to_nisa: dict[int, int],
                    init_offset: int):
    """Resolve label references in NISA branch/jump instructions.

    Walks through the NISA instructions and patches branch targets
    and return addresses using the RV32I→NISA index mapping.
    """
    # Build a map from RV32I index to the NISA instructions it generated
    nisa_idx = init_offset  # skip SP initialization

    for rv_idx, rv_instr in enumerate(rv_instrs):
        nisa_start = rv_to_nisa[rv_idx]
        nisa_end = rv_to_nisa.get(rv_idx + 1, len(nisa_instrs))
        n_generated = nisa_end - nisa_start

        # Resolve label if this instruction has one
        label = rv_instr.label
        if label and label not in rv_labels:
            # Try stripping leading dot (GCC generates .L2 style labels)
            label = label.lstrip('.')
        if label and label in rv_labels:
            rv_target = rv_labels[label]
            nisa_target = rv_to_nisa.get(rv_target, len(nisa_instrs))

            # Patch the NISA instructions generated from this RV instruction
            for ni in range(nisa_start, nisa_end):
                instr = nisa_instrs[ni]

                if instr.opcode == Opcode.JMP:
                    nisa_instrs[ni] = Instruction(Opcode.JMP, a=nisa_target)

                elif instr.opcode in (Opcode.BEQ, Opcode.BNE, Opcode.BLT,
                                       Opcode.BGE, Opcode.BLTU, Opcode.BGEU):
                    nisa_instrs[ni] = Instruction(
                        instr.opcode, a=instr.a, b=instr.b, c=nisa_target)

        # Patch return addresses for JAL
        if rv_instr.op in (RV32IOp.JAL, RV32IOp.CALL) and rv_instr.rd != 0:
            # The MOVI for return address is the first instruction
            return_addr = nisa_end  # NISA index after the JMP
            nisa_instrs[nisa_start] = movi(rv_instr.rd, return_addr)
