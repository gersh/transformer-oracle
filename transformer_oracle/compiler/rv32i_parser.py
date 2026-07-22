"""
RV32I Assembly Parser.

Parses RISC-V assembly output from GCC/clang into structured
instruction objects. Handles the standard GAS (GNU Assembler) syntax.

Supported:
  - All RV32I base instructions (47 instructions)
  - Labels and label references
  - Directives (.text, .globl, .word, .section, etc.) — skipped
  - Register aliases (zero, ra, sp, etc.)
  - Memory operands: offset(reg) syntax
  - Comments (# and //)
"""

import re
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto


class UnknownInstruction(Exception):
    """Raised when an unrecognized instruction mnemonic is encountered, so it fails
    loud instead of being silently dropped (which produces wrong results)."""


class RV32IOp(Enum):
    """RV32I instruction opcodes."""
    # R-type (register-register)
    ADD = auto()
    SUB = auto()
    SLL = auto()
    SLT = auto()
    SLTU = auto()
    SGT = auto()      # set greater-than (pseudo: sgt rd,rs,rt = slt rd,rt,rs)
    SGTU = auto()     # set greater-than unsigned (pseudo: sgtu rd,rs,rt = sltu rd,rt,rs)
    XOR = auto()
    SRL = auto()
    SRA = auto()
    OR = auto()
    AND = auto()

    # RV32M (multiply/divide)
    MUL = auto()
    MULH = auto()
    MULHSU = auto()
    MULHU = auto()
    DIV = auto()
    DIVU = auto()
    REM = auto()
    REMU = auto()

    # I-type (register-immediate)
    ADDI = auto()
    SLTI = auto()
    SLTIU = auto()
    XORI = auto()
    ORI = auto()
    ANDI = auto()
    SLLI = auto()
    SRLI = auto()
    SRAI = auto()

    # Load
    LB = auto()
    LH = auto()
    LW = auto()
    LBU = auto()
    LHU = auto()

    # Store
    SB = auto()
    SH = auto()
    SW = auto()

    # Branch
    BEQ = auto()
    BNE = auto()
    BLT = auto()
    BGE = auto()
    BLTU = auto()
    BGEU = auto()

    # Jump
    JAL = auto()
    JALR = auto()

    # Upper immediate
    LUI = auto()
    AUIPC = auto()

    # System
    ECALL = auto()
    EBREAK = auto()
    FENCE = auto()

    # Pseudo-instructions (expanded during parsing)
    LI = auto()       # load immediate (pseudo)
    MV = auto()       # move (pseudo)
    NOP = auto()      # no-op (pseudo)
    J = auto()        # jump (pseudo for jal x0, offset)
    JR = auto()       # jump register (pseudo for jalr x0, rs, 0)
    RET = auto()      # return (pseudo for jalr x0, x1, 0)
    CALL = auto()     # call (pseudo)
    TAIL = auto()     # tail call (pseudo)
    LA = auto()       # load address (pseudo)
    LLA = auto()      # load local address (pseudo)
    NEG = auto()      # negate (pseudo)
    NOT = auto()      # bitwise not (pseudo)
    SEQZ = auto()     # set equal zero (pseudo)
    SNEZ = auto()     # set not equal zero (pseudo)
    BEQZ = auto()     # branch if equal zero (pseudo)
    BNEZ = auto()     # branch if not equal zero (pseudo)
    BLEZ = auto()     # branch if <= zero (pseudo)
    BGEZ = auto()     # branch if >= zero (pseudo)
    BLTZ = auto()     # branch if < zero (pseudo)
    BGTZ = auto()     # branch if > zero (pseudo)
    BLE = auto()      # branch if <= (pseudo: bge with swapped args)
    BGT = auto()      # branch if > (pseudo: blt with swapped args)
    BLEU = auto()     # branch if <= unsigned (pseudo)
    BGTU = auto()     # branch if > unsigned (pseudo)


# Register name aliases
_REG_ALIASES = {
    'zero': 0, 'ra': 1, 'sp': 2, 'gp': 3, 'tp': 4,
    't0': 5, 't1': 6, 't2': 7,
    's0': 8, 'fp': 8, 's1': 9,
    'a0': 10, 'a1': 11, 'a2': 12, 'a3': 13,
    'a4': 14, 'a5': 15, 'a6': 16, 'a7': 17,
    's2': 18, 's3': 19, 's4': 20, 's5': 21,
    's6': 22, 's7': 23, 's8': 24, 's9': 25,
    's10': 26, 's11': 27,
    't3': 28, 't4': 29, 't5': 30, 't6': 31,
}

# Opcode string → enum
_OP_MAP = {op.name.lower(): op for op in RV32IOp}


@dataclass
class RV32IInstr:
    """A parsed RV32I instruction."""
    op: RV32IOp
    rd: int = 0        # destination register
    rs1: int = 0       # source register 1
    rs2: int = 0       # source register 2
    imm: int = 0       # immediate value
    label: str = ""    # label reference (for branches/jumps)
    line: int = 0      # source line number
    reloc_kind: str = ""  # symbol relocation kind: 'hi' | 'lo' | 'pcrel_hi' | 'pcrel_lo' | 'full'
    reloc_sym: str = ""   # symbol name for the relocation (resolved against the data layout)

    def __repr__(self):
        name = self.op.name.lower()
        if self.op in _R_TYPE:
            return f"{name} x{self.rd}, x{self.rs1}, x{self.rs2}"
        elif self.op in _I_TYPE:
            return f"{name} x{self.rd}, x{self.rs1}, {self.imm}"
        elif self.op in _LOAD_TYPE:
            return f"{name} x{self.rd}, {self.imm}(x{self.rs1})"
        elif self.op in _STORE_TYPE:
            return f"{name} x{self.rs2}, {self.imm}(x{self.rs1})"
        elif self.op in _BRANCH_TYPE:
            target = self.label or str(self.imm)
            return f"{name} x{self.rs1}, x{self.rs2}, {target}"
        elif self.op == RV32IOp.JAL:
            target = self.label or str(self.imm)
            return f"jal x{self.rd}, {target}"
        elif self.op == RV32IOp.JALR:
            return f"jalr x{self.rd}, x{self.rs1}, {self.imm}"
        elif self.op in (RV32IOp.LUI, RV32IOp.AUIPC):
            return f"{name} x{self.rd}, {self.imm}"
        elif self.op == RV32IOp.LI:
            return f"li x{self.rd}, {self.imm}"
        return f"{name}"


_R_TYPE = {RV32IOp.ADD, RV32IOp.SUB, RV32IOp.SLL, RV32IOp.SLT, RV32IOp.SLTU,
           RV32IOp.SGT, RV32IOp.SGTU,
           RV32IOp.XOR, RV32IOp.SRL, RV32IOp.SRA, RV32IOp.OR, RV32IOp.AND,
           RV32IOp.MUL, RV32IOp.MULH, RV32IOp.MULHSU, RV32IOp.MULHU,
           RV32IOp.DIV, RV32IOp.DIVU, RV32IOp.REM, RV32IOp.REMU}

_I_TYPE = {RV32IOp.ADDI, RV32IOp.SLTI, RV32IOp.SLTIU, RV32IOp.XORI,
           RV32IOp.ORI, RV32IOp.ANDI, RV32IOp.SLLI, RV32IOp.SRLI, RV32IOp.SRAI}

_LOAD_TYPE = {RV32IOp.LB, RV32IOp.LH, RV32IOp.LW, RV32IOp.LBU, RV32IOp.LHU}

_STORE_TYPE = {RV32IOp.SB, RV32IOp.SH, RV32IOp.SW}

_BRANCH_TYPE = {RV32IOp.BEQ, RV32IOp.BNE, RV32IOp.BLT, RV32IOp.BGE,
                RV32IOp.BLTU, RV32IOp.BGEU}


def parse_register(s: str) -> int:
    """Parse a register name to index (0-31)."""
    s = s.strip().lower()
    if s in _REG_ALIASES:
        return _REG_ALIASES[s]
    if s.startswith('x'):
        try:
            idx = int(s[1:])
            if 0 <= idx < 32:
                return idx
        except ValueError:
            pass
    raise ValueError(f"Invalid register: '{s}'")


def parse_immediate(s: str) -> int:
    """Parse an immediate value."""
    s = s.strip()
    if s.startswith('0x') or s.startswith('0X'):
        return int(s, 16)
    if s.startswith('-0x') or s.startswith('-0X'):
        return -int(s[1:], 16)
    return int(s)


_RELOC_RE = re.compile(r'^%(hi|lo|pcrel_hi|pcrel_lo)\(([^)]+)\)$')


def _parse_imm_or_reloc(s: str, instr: 'RV32IInstr') -> int:
    """Parse an immediate that may be a %hi/%lo(sym) relocation.

    On a relocation, records (kind, sym) on `instr` and returns 0 (patched after the
    data layout is known). Otherwise returns the numeric immediate.
    """
    s = s.strip()
    m = _RELOC_RE.match(s)
    if m:
        instr.reloc_kind = m.group(1)
        instr.reloc_sym = m.group(2).strip()
        return 0
    return parse_immediate(s)


# Base byte address where the data/bss/rodata sections are laid out. Placed above the
# regions test/harness programs hardcode for scratch/heap (params ~0x0FF0, buffers
# 0x1000–0x1200, heap 0x4000) and below the 64KB memory top, so compiler-managed globals
# never alias either the stack (low, grows down from ~0x3F0) or a program's own scratch.
DATA_BASE = 0x8000


def parse_rv32i_assembly(source: str) -> tuple[list[RV32IInstr], dict[str, int]]:
    """Parse RV32I assembly into (instructions, labels). Back-compat 2-tuple wrapper.

    Data-section symbols are laid out and %hi/%lo relocations are resolved internally;
    use `parse_rv32i_assembly_with_data` if you also need the initialized-data image.
    """
    instrs, labels, _data = parse_rv32i_assembly_with_data(source)
    return instrs, labels


def parse_rv32i_assembly_with_data(
        source: str) -> tuple[list[RV32IInstr], dict[str, int], dict[int, int]]:
    """Parse RV32I assembly, laying out the data section and resolving relocations.

    Returns:
        (instructions, labels, data_image) where
          - labels maps text-section name → instruction index,
          - data_image maps byte-address → byte value for initialized data (.word/.string/…).

    This is the piece that makes global variables work: GCC accesses globals via the
    `lui rd,%hi(sym); lw/sw rd2,%lo(sym)(rd)` idiom. We assign every data symbol an
    absolute address (from .data/.bss/.sbss/.comm/.set and size directives) and rewrite
    each %hi/%lo relocation to the corresponding numeric immediate. Previously the
    immediate `%hi(sym)` failed to parse and the whole instruction was silently dropped,
    leaving global loads/stores missing (so a global RMW counter used as an index read a
    stale register).
    """
    lines = source.split('\n')
    instructions: list[RV32IInstr] = []
    labels: dict[str, int] = {}

    # Data layout state
    section = 'text'                 # 'text' | 'data' | 'bss'
    data_loc = DATA_BASE             # current byte address in the data area
    data_symbols: dict[str, int] = {}
    data_image: dict[int, int] = {}  # byte-addr → byte value (initialized data only)
    # Deferred .word/.dword values that are symbol references, resolved after full layout.
    pending_data_syms: list[tuple[int, str, int]] = []  # (addr, symbol, nbytes)

    def align_to(loc: int, n: int) -> int:
        if n <= 1:
            return loc
        return (loc + n - 1) & ~(n - 1)

    for line_no, line in enumerate(lines, 1):
        line = line.split('#')[0].split('//')[0].strip()
        if not line:
            continue

        # Handle a leading label (possibly followed by an instruction/directive).
        if ':' in line and not line.startswith('.string') and not line.startswith('.asciz'):
            head, _, rest = line.partition(':')
            raw_label = head.strip()
            if re.match(r'^\.?[a-zA-Z_]\w*$', raw_label):
                label = raw_label.lstrip('.')
                if section == 'text':
                    labels[label] = len(instructions)
                    if raw_label != label:
                        labels[raw_label] = len(instructions)
                else:
                    data_symbols[raw_label] = data_loc
                    if raw_label != label:
                        data_symbols[label] = data_loc
                line = rest.strip()
                if not line:
                    continue

        # Directives
        if line.startswith('.'):
            data_loc = _handle_directive(line, section, data_loc, data_symbols,
                                         data_image, pending_data_syms, align_to)
            new_section = _section_of(line)
            if new_section is not None:
                section = new_section
            continue

        if section != 'text':
            # Non-directive, non-label content inside a data section: ignore.
            continue

        # Parse a text-section instruction.
        try:
            instr = _parse_instruction(line, line_no)
            if instr is not None:
                instructions.append(instr)
        except UnknownInstruction:
            raise
        except Exception:
            pass

    # Resolve deferred symbolic .word values now that all symbols are laid out.
    for addr, sym, nbytes in pending_data_syms:
        val = data_symbols.get(sym, 0)
        for k in range(nbytes):
            data_image[addr + k] = (val >> (8 * k)) & 0xFF

    # Resolve instruction relocations against the data layout.
    for instr in instructions:
        if instr.reloc_kind:
            instr.imm = _resolve_reloc(instr.reloc_kind, instr.reloc_sym, data_symbols)
        elif instr.op in (RV32IOp.LA, RV32IOp.LLA) and instr.label in data_symbols:
            instr.imm = data_symbols[instr.label]
            instr.reloc_kind = 'full'

    return instructions, labels, data_image


def _section_of(directive: str):
    """Return the section a directive switches to, or None if it doesn't switch."""
    d = directive.split()[0]
    if d in ('.text',):
        return 'text'
    if d in ('.data', '.sdata', '.rodata', '.srodata'):
        return 'data'
    if d in ('.bss', '.sbss'):
        return 'bss'
    if d == '.section':
        name = directive.split(None, 1)[1] if len(directive.split()) > 1 else ''
        name = name.split(',')[0].strip()
        if name.startswith('.text'):
            return 'text'
        if name.startswith(('.bss', '.sbss', '.tbss')):
            return 'bss'
        if name.startswith(('.data', '.sdata', '.rodata', '.srodata', '.sbss')):
            return 'data'
        return None
    return None


def _resolve_reloc(kind: str, sym: str, data_symbols: dict[str, int]) -> int:
    """Compute the numeric immediate for a %hi/%lo relocation of `sym`."""
    addr = data_symbols.get(sym, 0)
    if kind in ('hi', 'pcrel_hi'):
        return ((addr + 0x800) >> 12) & 0xFFFFF
    if kind in ('lo', 'pcrel_lo'):
        # Signed low-12 remainder such that (hi<<12) + lo == addr.
        return (addr - (((addr + 0x800) >> 12) << 12)) & 0xFFFFFFFF
    if kind == 'full':
        return addr & 0xFFFFFFFF
    return 0


def _parse_data_values(rest: str) -> list:
    """Split the operand list of a .word/.byte/... directive into int|str tokens."""
    out = []
    for tok in _split_operands(rest):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(parse_immediate(tok))
        except ValueError:
            out.append(tok)  # symbol reference, resolved later
    return out


def _decode_asm_string(s: str) -> bytes:
    """Decode a GAS quoted string operand (handles common C escapes)."""
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return bytes(s, 'utf-8').decode('unicode_escape').encode('latin-1', 'replace')


def _handle_directive(line, section, data_loc, data_symbols, data_image,
                      pending_data_syms, align_to):
    """Apply one assembler directive; returns the (possibly advanced) data location."""
    parts = line.split(None, 1)
    d = parts[0]
    rest = parts[1].strip() if len(parts) > 1 else ''

    if d in ('.align', '.p2align'):
        try:
            n = int(rest.split(',')[0])
            if section != 'text':
                data_loc = align_to(data_loc, 1 << n)
        except (ValueError, IndexError):
            pass
    elif d == '.balign':
        try:
            n = int(rest.split(',')[0])
            if section != 'text':
                data_loc = align_to(data_loc, n)
        except (ValueError, IndexError):
            pass
    elif d in ('.zero', '.space', '.skip'):
        try:
            data_loc += int(rest.split(',')[0])
        except (ValueError, IndexError):
            pass
    elif d in ('.word', '.4byte', '.long', '.half', '.short', '.2byte',
               '.byte', '.1byte', '.dword', '.8byte'):
        nbytes = {'.word': 4, '.4byte': 4, '.long': 4, '.half': 2, '.short': 2,
                  '.2byte': 2, '.byte': 1, '.1byte': 1, '.dword': 8, '.8byte': 8}[d]
        for val in _parse_data_values(rest):
            if isinstance(val, str):
                pending_data_syms.append((data_loc, val, nbytes))
            else:
                for k in range(nbytes):
                    data_image[data_loc + k] = (val >> (8 * k)) & 0xFF
            data_loc += nbytes
    elif d in ('.string', '.asciz', '.ascii'):
        raw = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ''
        b = _decode_asm_string(raw)
        for k, byte in enumerate(b):
            data_image[data_loc + k] = byte
        data_loc += len(b)
        if d in ('.string', '.asciz'):
            data_image[data_loc] = 0
            data_loc += 1
    elif d in ('.set', '.equ', '.equiv'):
        # .set sym, expr  (expr may reference '.' = current location)
        try:
            name, expr = rest.split(',', 1)
            name = name.strip()
            expr = expr.strip()
            data_symbols[name] = _eval_data_expr(expr, data_loc, data_symbols)
        except ValueError:
            pass
    elif d in ('.comm', '.lcomm'):
        # .comm sym, size[, align]
        toks = [t.strip() for t in rest.split(',')]
        try:
            name = toks[0]
            size = int(toks[1])
            if len(toks) > 2:
                data_loc = align_to(data_loc, int(toks[2]))
            data_symbols[name] = data_loc
            data_loc += size
        except (ValueError, IndexError):
            pass
    # All other directives (.globl, .type, .size, .file, .option, .attribute,
    # .ident, .cfi_*, .note, ...) don't affect layout.
    return data_loc


def _eval_data_expr(expr: str, data_loc: int, data_symbols: dict[str, int]) -> int:
    """Evaluate a simple data-location expression like '. + 0', '.', or 'sym + 4'."""
    expr = expr.strip()
    m = re.match(r'^(\.|\w+)\s*([+-])\s*(\d+)$', expr)
    if m:
        base = data_loc if m.group(1) == '.' else data_symbols.get(m.group(1), 0)
        n = int(m.group(3))
        return base + n if m.group(2) == '+' else base - n
    if expr == '.':
        return data_loc
    if expr in data_symbols:
        return data_symbols[expr]
    try:
        return int(expr)
    except ValueError:
        return data_loc


def _parse_instruction(line: str, line_no: int) -> Optional[RV32IInstr]:
    """Parse a single assembly instruction line."""
    # Tokenize: split on whitespace and commas
    # Handle offset(reg) syntax first by normalizing
    line = line.strip()
    if not line:
        return None

    # Split into mnemonic and operands
    parts = line.split(None, 1)
    mnemonic = parts[0].lower()
    operand_str = parts[1].strip() if len(parts) > 1 else ""

    if mnemonic not in _OP_MAP:
        # Fail loud on a plausible-but-unsupported instruction mnemonic (would otherwise
        # be silently dropped → wrong results, as sgtu was). Only raise for clean
        # identifiers; assembler/directive noise (e.g. "(ubuntu" from a .ident string)
        # is skipped as before.
        if mnemonic and mnemonic[0].isalpha() and all(ch.isalnum() or ch in '._' for ch in mnemonic):
            raise UnknownInstruction(mnemonic)
        return None

    op = _OP_MAP[mnemonic]
    operands = _split_operands(operand_str) if operand_str else []

    instr = RV32IInstr(op=op, line=line_no)

    if op in _R_TYPE:
        # add rd, rs1, rs2
        instr.rd = parse_register(operands[0])
        instr.rs1 = parse_register(operands[1])
        instr.rs2 = parse_register(operands[2])

    elif op in _I_TYPE:
        # addi rd, rs1, imm   (imm may be %lo(sym))
        instr.rd = parse_register(operands[0])
        instr.rs1 = parse_register(operands[1])
        instr.imm = _parse_imm_or_reloc(operands[2], instr)

    elif op in _LOAD_TYPE:
        # lw rd, offset(rs1)   (offset may be %lo(sym))
        instr.rd = parse_register(operands[0])
        offset, base = _parse_mem_operand(operands[1], instr)
        instr.imm = offset
        instr.rs1 = base

    elif op in _STORE_TYPE:
        # sw rs2, offset(rs1)   (offset may be %lo(sym))
        instr.rs2 = parse_register(operands[0])
        offset, base = _parse_mem_operand(operands[1], instr)
        instr.imm = offset
        instr.rs1 = base

    elif op in _BRANCH_TYPE:
        # beq rs1, rs2, label
        instr.rs1 = parse_register(operands[0])
        instr.rs2 = parse_register(operands[1])
        instr.label = operands[2].strip()
        try:
            instr.imm = parse_immediate(operands[2])
        except ValueError:
            pass  # label reference, resolved later

    elif op == RV32IOp.JAL:
        if len(operands) == 1:
            # jal label (rd = ra = x1)
            instr.rd = 1
            instr.label = operands[0].strip()
        else:
            # jal rd, label
            instr.rd = parse_register(operands[0])
            instr.label = operands[1].strip()
        try:
            instr.imm = parse_immediate(instr.label)
        except ValueError:
            pass

    elif op == RV32IOp.JALR:
        if len(operands) == 3:
            instr.rd = parse_register(operands[0])
            instr.rs1 = parse_register(operands[1])
            instr.imm = parse_immediate(operands[2])
        elif len(operands) == 2:
            # jalr rd, offset(rs1)  or  jalr rd, rs1
            instr.rd = parse_register(operands[0])
            if '(' in operands[1]:
                offset, base = _parse_mem_operand(operands[1], instr)
                instr.imm = offset
                instr.rs1 = base
            else:
                instr.rs1 = parse_register(operands[1])
        elif len(operands) == 1:
            # jalr rs1 (rd = ra)
            instr.rd = 1
            instr.rs1 = parse_register(operands[0])

    elif op in (RV32IOp.LUI, RV32IOp.AUIPC):
        # lui rd, %hi(sym)  — the %hi relocation is resolved after data layout.
        instr.rd = parse_register(operands[0])
        instr.imm = _parse_imm_or_reloc(operands[1], instr)

    elif op == RV32IOp.LI:
        instr.rd = parse_register(operands[0])
        instr.imm = parse_immediate(operands[1])

    elif op == RV32IOp.MV:
        instr.rd = parse_register(operands[0])
        instr.rs1 = parse_register(operands[1])

    elif op == RV32IOp.NEG:
        instr.rd = parse_register(operands[0])
        instr.rs2 = parse_register(operands[1])

    elif op == RV32IOp.NOT:
        instr.rd = parse_register(operands[0])
        instr.rs1 = parse_register(operands[1])

    elif op in (RV32IOp.LA, RV32IOp.LLA):
        # la/lla rd, symbol — load address of symbol
        instr.rd = parse_register(operands[0])
        instr.label = operands[1].strip()

    elif op == RV32IOp.J:
        instr.label = operands[0].strip()

    elif op == RV32IOp.JR:
        instr.rs1 = parse_register(operands[0])

    elif op == RV32IOp.RET:
        pass  # no operands

    elif op in (RV32IOp.BEQZ, RV32IOp.BNEZ, RV32IOp.BLEZ,
                RV32IOp.BGEZ, RV32IOp.BLTZ, RV32IOp.BGTZ):
        instr.rs1 = parse_register(operands[0])
        instr.label = operands[1].strip()

    elif op in (RV32IOp.BLE, RV32IOp.BGT, RV32IOp.BLEU, RV32IOp.BGTU):
        # ble rs1, rs2, label (pseudo — swapped operand branches)
        instr.rs1 = parse_register(operands[0])
        instr.rs2 = parse_register(operands[1])
        instr.label = operands[2].strip()

    elif op == RV32IOp.SEQZ:
        instr.rd = parse_register(operands[0])
        instr.rs1 = parse_register(operands[1])

    elif op == RV32IOp.SNEZ:
        instr.rd = parse_register(operands[0])
        instr.rs2 = parse_register(operands[1])

    elif op == RV32IOp.CALL:
        # call label → jal ra, label
        instr.rd = 1  # ra
        instr.label = operands[0].strip()

    elif op == RV32IOp.TAIL:
        # tail label → jal x0, label (tail call, no return)
        instr.rd = 0
        instr.label = operands[0].strip()

    elif op in (RV32IOp.NOP, RV32IOp.ECALL, RV32IOp.EBREAK, RV32IOp.FENCE):
        pass

    return instr


def _split_operands(s: str) -> list[str]:
    """Split operand string, respecting parentheses for offset(reg) syntax."""
    result = []
    current = []
    depth = 0
    for ch in s:
        if ch == '(' :
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            result.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        result.append(''.join(current).strip())
    return [x for x in result if x]


def _parse_mem_operand(s: str, instr: 'RV32IInstr' = None) -> tuple[int, int]:
    """Parse memory operand like '8(sp)', '0(a0)', or '%lo(sym)(a5)'.

    Returns (offset, base_reg). A `%lo(sym)` offset records a relocation on `instr`
    (offset returned as 0, patched once the data layout is known).
    """
    s = s.strip()
    # %lo(sym)(reg) / %pcrel_lo(label)(reg)
    m = re.match(r'^%(lo|pcrel_lo)\(([^)]+)\)\((\w+)\)$', s)
    if m and instr is not None:
        instr.reloc_kind = m.group(1)
        instr.reloc_sym = m.group(2).strip()
        return 0, parse_register(m.group(3))
    m = re.match(r'(-?\d+)\((\w+)\)', s)
    if m:
        offset = int(m.group(1))
        base = parse_register(m.group(2))
        return offset, base
    # Try just (reg) with implicit offset 0
    m = re.match(r'\((\w+)\)', s)
    if m:
        return 0, parse_register(m.group(1))
    raise ValueError(f"Cannot parse memory operand: '{s}'")
