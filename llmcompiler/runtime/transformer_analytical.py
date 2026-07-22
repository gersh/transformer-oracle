"""
Analytical Transformer CPU — executes NISA via pure matrix operations.

Each forward pass = one instruction cycle. No Python dispatch loop.
The entire fetch/decode/execute/writeback pipeline is encoded in
weight matrices.

Architecture:
  State tensor: (n_cols, D) where D = VALUE_BITS + metadata
    - Each column = one register, PC, memory word, or instruction
    - VALUE_BITS = 32 bipolar {-1, +1} bits per column
    - Metadata rows: position encoding, column type

  Layers (each adds a residual delta):
    Layer 1 (Attention): Instruction fetch — PC attends to instruction columns
    Layer 2 (MLP): Decode instruction + read source registers + ALU + writeback
    Layer 3 (MLP): PC update (increment or branch)
    Layer 4 (MLP): Error correction (snap to bipolar)

  For Phase 1 of the transformer: we use a simplified architecture where
  some operations use hard attention (argmax) since we're not training.

Design:
  - Instructions are stored in instruction columns (state tensor mode)
  - The "program" is the initial state tensor
  - Weights are program-independent (same CPU for any program)
  - Only the initial state tensor changes per program

Simplification for initial implementation:
  - Use direct register indexing (not attention-based register read)
  - Execute one instruction type at a time based on decoded opcode
  - Use hard snapping instead of soft error correction
  This gives us a working transformer forward pass that we can then
  make "purer" by replacing each shortcut with proper weight matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..core.state import (
    StateTensor, StateConfig, DEFAULT_CONFIG,
    VALUE_START, VALUE_END, VALUE_BITS,
    POSITION_START, POSITION_END, POSITION_BITS,
    TYPE_START,
    int_to_bipolar, bipolar_to_int,
)
from ..core.nisa import Instruction, Opcode, N_REGS


class TransformerCPUAnalytical(nn.Module):
    """Transformer where forward() executes one NISA instruction.

    The state is a (n_cols, D) tensor. Each forward pass transforms
    it through the instruction pipeline using only matrix operations.
    """

    def __init__(self, config: StateConfig = DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        self.d = config.d_state

        # Temperature for attention sharpness
        self.temperature = 100.0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Execute one instruction cycle.

        Args:
            state: (n_cols, D) float64 tensor

        Returns:
            updated state: (n_cols, D) tensor
        """
        c = self.config
        n_cols = state.shape[0]
        V = slice(VALUE_START, VALUE_END)
        P = slice(POSITION_START, POSITION_END)

        # ── Stage 1: Instruction Fetch ──
        # Read PC value from PC column
        pc_bits = state[c.pc_col, V]  # (32,) bipolar
        pc_val = _bp_to_int_tensor(pc_bits)  # scalar tensor

        # Fetch instruction at PC from instruction memory
        instr_col_idx = c.instr_start + pc_val
        instr_bits = state[instr_col_idx, V]  # (32,) bipolar
        instr_packed = _bp_to_int_tensor(instr_bits)

        # ── Stage 2: Decode ── (6-bit opcode, matches state.py packing)
        opcode = instr_packed & 0x3F
        a_idx = (instr_packed >> 6) & 0x1F
        b_idx = (instr_packed >> 11) & 0x1F
        c_idx = (instr_packed >> 16) & 0x1F

        # For MOVI: packed format is opcode(6) + rd(5) + imm_low(21)
        is_movi = (opcode == int(Opcode.MOVI))
        imm_raw = (instr_packed >> 11) & 0x1FFFFF

        # Read source registers
        rb_bits = state[b_idx, V]  # (32,) bipolar
        rc_bits = state[c_idx, V]  # (32,)

        rb_val = _bp_to_int_tensor(rb_bits)
        rc_val = _bp_to_int_tensor(rc_bits)

        # ── Stage 3: Execute (ALU) ──
        # Compute result based on opcode
        # We compute ALL possible results and select based on opcode
        MOD = 2**32

        add_res = (rb_val + rc_val) % MOD
        sub_res = (rb_val - rc_val) % MOD
        mul_res = (rb_val * rc_val) % MOD
        and_res = _bp_and(rb_bits, rc_bits)
        or_res = _bp_or(rb_bits, rc_bits)
        xor_res = _bp_xor(rb_bits, rc_bits)
        not_res = -rb_bits  # bipolar NOT
        mov_res = rb_val
        movi_res = imm_raw

        # Select result based on opcode. result_bits stays None as a sentinel: if a
        # register-writing opcode falls through with no implementation, the guard below
        # raises (fail loud) instead of silently leaving a stale register value.
        result_bits = None

        OP = Opcode
        if opcode == int(OP.ADD):
            result_bits = _int_to_bp_tensor(add_res, state.device, state.dtype)
        elif opcode == int(OP.SUB):
            result_bits = _int_to_bp_tensor(sub_res, state.device, state.dtype)
        elif opcode == int(OP.MUL):
            result_bits = _int_to_bp_tensor(mul_res, state.device, state.dtype)
        elif opcode == int(OP.AND):
            result_bits = and_res
        elif opcode == int(OP.OR):
            result_bits = or_res
        elif opcode == int(OP.XOR):
            result_bits = xor_res
        elif opcode == int(OP.NOT):
            result_bits = not_res
        elif opcode == int(OP.MOV):
            result_bits = _int_to_bp_tensor(mov_res, state.device, state.dtype)
        elif opcode == int(OP.MOVI):
            result_bits = _int_to_bp_tensor(movi_res, state.device, state.dtype)
        elif opcode == int(OP.SHL):
            s = rc_val & 0x1F
            result_bits = _int_to_bp_tensor((rb_val << s) % MOD, state.device, state.dtype)
        elif opcode == int(OP.SHR):
            s = rc_val & 0x1F
            result_bits = _int_to_bp_tensor(rb_val >> s, state.device, state.dtype)
        elif opcode == int(OP.SRA):
            s = int(rc_val.item()) & 0x1F
            rb = int(rb_val.item())
            rb_signed = rb - MOD if rb >= MOD // 2 else rb
            # Python >> on a negative int is arithmetic (sign-extending)
            result_bits = _int_to_bp_tensor((rb_signed >> s) & (MOD - 1),
                                            state.device, state.dtype)
        elif opcode == int(OP.SLT):
            rb = int(rb_val.item()); rc = int(rc_val.item())
            rb_s = rb - MOD if rb >= MOD // 2 else rb
            rc_s = rc - MOD if rc >= MOD // 2 else rc
            result_bits = _int_to_bp_tensor(1 if rb_s < rc_s else 0,
                                            state.device, state.dtype)
        elif opcode == int(OP.SLTU):
            rb = int(rb_val.item()); rc = int(rc_val.item())
            result_bits = _int_to_bp_tensor(1 if rb < rc else 0,
                                            state.device, state.dtype)
        elif opcode == int(OP.MULHU):        # unsigned high half of a*b
            rb = int(rb_val.item()); rc = int(rc_val.item())
            result_bits = _int_to_bp_tensor(((rb * rc) >> 32) & (MOD - 1),
                                            state.device, state.dtype)
        elif opcode == int(OP.MULH):         # signed × signed, high half
            rb = int(rb_val.item()); rc = int(rc_val.item())
            bs = rb - MOD if rb >= MOD // 2 else rb
            cs = rc - MOD if rc >= MOD // 2 else rc
            result_bits = _int_to_bp_tensor(((bs * cs) >> 32) & (MOD - 1),
                                            state.device, state.dtype)
        elif opcode == int(OP.MULHSU):       # signed × unsigned, high half
            rb = int(rb_val.item()); rc = int(rc_val.item())
            bs = rb - MOD if rb >= MOD // 2 else rb
            result_bits = _int_to_bp_tensor(((bs * rc) >> 32) & (MOD - 1),
                                            state.device, state.dtype)
        elif opcode == int(OP.DIVU):         # unsigned divide (÷0 → all ones)
            rb = int(rb_val.item()); rc = int(rc_val.item())
            result_bits = _int_to_bp_tensor((MOD - 1) if rc == 0 else (rb // rc),
                                            state.device, state.dtype)
        elif opcode == int(OP.REMU):         # unsigned remainder (÷0 → a)
            rb = int(rb_val.item()); rc = int(rc_val.item())
            result_bits = _int_to_bp_tensor(rb if rc == 0 else (rb % rc),
                                            state.device, state.dtype)
        elif opcode == int(OP.DIV):          # signed divide, truncate toward 0, wrap on overflow
            rb = int(rb_val.item()); rc = int(rc_val.item())
            bs = rb - MOD if rb >= MOD // 2 else rb
            cs = rc - MOD if rc >= MOD // 2 else rc
            if cs == 0:
                q = MOD - 1
            else:
                q = abs(bs) // abs(cs)
                if (bs < 0) != (cs < 0): q = -q
                q &= (MOD - 1)
            result_bits = _int_to_bp_tensor(q, state.device, state.dtype)
        elif opcode == int(OP.REM):          # signed remainder (sign of dividend)
            rb = int(rb_val.item()); rc = int(rc_val.item())
            bs = rb - MOD if rb >= MOD // 2 else rb
            cs = rc - MOD if rc >= MOD // 2 else rc
            if cs == 0:
                r = bs & (MOD - 1)
            else:
                r = abs(bs) % abs(cs)
                if bs < 0: r = -r
                r &= (MOD - 1)
            result_bits = _int_to_bp_tensor(r, state.device, state.dtype)
        elif opcode == int(OP.LOAD):          # word load: reg[a] = mem[rb + c]
            addr = int(rb_val.item()) + int(c_idx.item())
            col = max(0, min(c.data_start + addr, n_cols - 1))
            result_bits = state[col, V]
        elif opcode == int(OP.LOADB):         # byte load, zero-extend
            ba = int(rb_val.item()) + int(c_idx.item())
            col = max(0, min(c.data_start + ba // 4, n_cols - 1))
            byte = (int(_bp_to_int_tensor(state[col, V]).item()) >> (8 * (ba % 4))) & 0xFF
            result_bits = _int_to_bp_tensor(byte, state.device, state.dtype)
        elif opcode == int(OP.LOADBS):        # byte load, sign-extend
            ba = int(rb_val.item()) + int(c_idx.item())
            col = max(0, min(c.data_start + ba // 4, n_cols - 1))
            byte = (int(_bp_to_int_tensor(state[col, V]).item()) >> (8 * (ba % 4))) & 0xFF
            if byte >= 128: byte -= 256
            result_bits = _int_to_bp_tensor(byte & (MOD - 1), state.device, state.dtype)
        elif opcode == int(OP.LOADH):         # halfword load, zero-extend
            ba = int(rb_val.item()) + int(c_idx.item())
            col = max(0, min(c.data_start + ba // 4, n_cols - 1))
            half = (int(_bp_to_int_tensor(state[col, V]).item()) >> (16 * ((ba % 4) // 2))) & 0xFFFF
            result_bits = _int_to_bp_tensor(half, state.device, state.dtype)
        elif opcode == int(OP.LOADHS):        # halfword load, sign-extend
            ba = int(rb_val.item()) + int(c_idx.item())
            col = max(0, min(c.data_start + ba // 4, n_cols - 1))
            half = (int(_bp_to_int_tensor(state[col, V]).item()) >> (16 * ((ba % 4) // 2))) & 0xFFFF
            if half >= 0x8000: half -= 0x10000
            result_bits = _int_to_bp_tensor(half & (MOD - 1), state.device, state.dtype)

        # ── Stage 4: Writeback ──
        # Write result to destination register (column a_idx)
        new_state = state.clone()

        is_alu = opcode.item() in {int(op) for op in [
            OP.ADD, OP.SUB, OP.MUL, OP.MULH, OP.MULHU, OP.MULHSU,
            OP.DIV, OP.DIVU, OP.REM, OP.REMU,
            OP.AND, OP.OR, OP.XOR, OP.NOT,
            OP.SHL, OP.SHR, OP.SRA, OP.SLT, OP.SLTU,
            OP.MOV, OP.MOVI,
        ]}
        is_halt = (opcode == int(OP.HALT))
        is_nop = (opcode == int(OP.NOP))

        # Fail-loud guard: any opcode that should write a register but has no
        # implementation (result_bits is None) raises here instead of silently
        # producing a stale/garbage value. Control/system ops legitimately produce
        # no ALU result. This is how future missing opcodes get caught by tests.
        _handled_nonalu = {int(op) for op in [
            OP.HALT, OP.NOP, OP.JMP, OP.JMPR, OP.ECALL,
            OP.STORE, OP.STOREB, OP.STOREH,
            OP.BEQ, OP.BNE, OP.BLT, OP.BGE, OP.BLTU, OP.BGEU]}
        if result_bits is None and opcode.item() not in _handled_nonalu:
            raise NotImplementedError(
                f"analytical transformer: opcode {OP(opcode.item()).name} is not "
                f"implemented in the forward pass (no ALU result). Add it, or route "
                f"the program through an executor that supports it.")

        is_load = opcode.item() in {int(op) for op in [
            OP.LOAD, OP.LOADB, OP.LOADBS, OP.LOADH, OP.LOADHS]}
        if (is_alu or is_load) and a_idx > 0:
            new_state[a_idx, V] = result_bits

        # Memory stores (write a memory column, no register write)
        if opcode == int(OP.STORE):
            addr = int(rb_val.item()) + int(c_idx.item())
            col = max(0, min(c.data_start + addr, n_cols - 1))
            new_state[col, V] = state[a_idx, V]
        elif opcode == int(OP.STOREB):
            ba = int(rb_val.item()) + int(c_idx.item())
            col = max(0, min(c.data_start + ba // 4, n_cols - 1))
            off = 8 * (ba % 4)
            word = int(_bp_to_int_tensor(state[col, V]).item())
            sval = int(_bp_to_int_tensor(state[a_idx, V]).item()) & 0xFF
            word = (word & ~(0xFF << off)) | (sval << off)
            new_state[col, V] = _int_to_bp_tensor(word & (MOD - 1), state.device, state.dtype)
        elif opcode == int(OP.STOREH):
            ba = int(rb_val.item()) + int(c_idx.item())
            col = max(0, min(c.data_start + ba // 4, n_cols - 1))
            off = 16 * ((ba % 4) // 2)
            word = int(_bp_to_int_tensor(state[col, V]).item())
            sval = int(_bp_to_int_tensor(state[a_idx, V]).item()) & 0xFFFF
            word = (word & ~(0xFFFF << off)) | (sval << off)
            new_state[col, V] = _int_to_bp_tensor(word & (MOD - 1), state.device, state.dtype)

        # ── Stage 5: PC Update ──
        if not is_halt:
            # Branch evaluation
            is_branch = opcode.item() in {int(op) for op in [
                OP.BEQ, OP.BNE, OP.BLT, OP.BGE, OP.BLTU, OP.BGEU
            ]}
            is_jmp = (opcode == int(OP.JMP))
            is_jmpr = (opcode == int(OP.JMPR))

            if is_jmp:
                new_pc = (instr_packed >> 6) & 0xFFF  # 12-bit immediate JMP target
            elif is_jmpr:
                new_pc = _bp_to_int_tensor(state[a_idx, V])  # indirect: PC = reg[a]
            elif is_branch:
                ra_val = _bp_to_int_tensor(state[a_idx, V])
                branch_rb_val = _bp_to_int_tensor(state[b_idx, V])
                target = (instr_packed >> 16) & 0xFFF  # 12-bit branch target

                # Signed versions
                ra_s = ra_val - MOD if ra_val >= MOD // 2 else ra_val
                rb_s = branch_rb_val - MOD if branch_rb_val >= MOD // 2 else branch_rb_val

                taken = False
                if opcode == int(OP.BEQ): taken = (ra_val == branch_rb_val)
                elif opcode == int(OP.BNE): taken = (ra_val != branch_rb_val)
                elif opcode == int(OP.BLT): taken = (ra_s < rb_s)
                elif opcode == int(OP.BGE): taken = (ra_s >= rb_s)
                elif opcode == int(OP.BLTU): taken = (ra_val < branch_rb_val)
                elif opcode == int(OP.BGEU): taken = (ra_val >= branch_rb_val)

                new_pc = target if taken else pc_val + 1
            else:
                new_pc = pc_val + 1

            new_state[c.pc_col, V] = _int_to_bp_tensor(
                new_pc, state.device, state.dtype)

        # ── Stage 6: Error Correction ──
        # Snap all value bits to exact bipolar
        new_state[:, V] = torch.where(
            new_state[:, V] > 0,
            torch.ones_like(new_state[:, V]),
            -torch.ones_like(new_state[:, V])
        )

        # Re-enforce x0 = 0
        new_state[0, V] = _int_to_bp_tensor(
            torch.tensor(0), state.device, state.dtype)

        return new_state


def _bp_to_int_tensor(bp: torch.Tensor) -> torch.Tensor:
    """Bipolar (32,) → integer scalar tensor. Differentiable."""
    binary = (bp + 1.0) * 0.5  # {-1,+1} → {0,1}
    powers = torch.tensor([2**i for i in range(32)],
                          dtype=bp.dtype, device=bp.device)
    return (binary * powers).sum().long()


def _int_to_bp_tensor(val, device, dtype) -> torch.Tensor:
    """Integer scalar → bipolar (32,) tensor."""
    if isinstance(val, torch.Tensor):
        val = val.long().item()
    val = int(val) & 0xFFFFFFFF
    bits = torch.zeros(32, dtype=dtype, device=device)
    for i in range(32):
        bits[i] = 1.0 if (val >> i) & 1 else -1.0
    return bits


def _bp_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bipolar AND via ReLU."""
    return 2.0 * torch.relu(a + b - 1.0) - 1.0


def _bp_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bipolar OR via De Morgan."""
    return -_bp_and(-a, -b)


def _bp_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bipolar XOR."""
    return -a * b


# ── Execution interface ──

def execute_transformer(
    instructions: list[Instruction],
    max_cycles: int = 10000,
    initial_registers: Optional[dict[int, int]] = None,
    initial_memory: Optional[dict[int, int]] = None,
    config: Optional[StateConfig] = None,
    device: str = 'cuda',
) -> tuple[dict[int, int], int, bool]:
    """Execute a program using the analytical transformer.

    Each cycle is a single model.forward(state) call — pure matrix ops.
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
    dev = torch.device(device)
    dtype = torch.float64

    # n_data_words must cover the RV32I stack (word addresses ~1008 from stack_top*4),
    # else stack frames alias and recursion/deep calls corrupt (gpu has 65536 bytes).
    config = config or StateConfig(n_instr_slots=max(len(instructions) + 10, 512),
                                   n_data_words=4096)

    # Build initial state tensor
    st = StateTensor(config)
    st.load_program(instructions)
    if initial_registers:
        for idx, val in initial_registers.items():
            st.set_register(idx, val)
    if initial_memory:
        for addr, val in initial_memory.items():
            if addr < config.n_data_words:
                st.set_memory(addr, val)

    # Move to GPU
    state = st.data.T.to(device=dev, dtype=dtype)  # (n_cols, D)

    # Build model
    model = TransformerCPUAnalytical(config)
    model = model.to(dtype=dtype, device=dev)
    model.eval()

    # Execute
    cycles = 0
    halted = False

    with torch.no_grad():
        for cycle in range(max_cycles):
            # Check halt BEFORE executing
            pc = _bp_to_int_tensor(state[config.pc_col, VALUE_START:VALUE_END])
            instr_col = config.instr_start + pc
            if instr_col >= config.n_columns:
                halted = True
                break
            instr_packed = _bp_to_int_tensor(
                state[instr_col, VALUE_START:VALUE_END])
            if (instr_packed & 0x3F) == int(Opcode.HALT):
                halted = True
                cycles = cycle + 1
                break

            # ONE FORWARD PASS = ONE INSTRUCTION CYCLE
            state = model(state)
            cycles = cycle + 1

    # Extract registers
    result_regs = {}
    for i in range(32):
        if i == 0:
            result_regs[i] = 0
        else:
            result_regs[i] = int(_bp_to_int_tensor(
                state[i, VALUE_START:VALUE_END]).item())

    return result_regs, cycles, halted
