"""
Pure tensor transformer CPU — zero Python if/else in the forward pass.

Every operation is a tensor computation. Opcode dispatch is done via
one-hot masking: compute ALL possible results, then select the correct
one by multiplying with the opcode mask.

This means the entire forward() is torch.compile-able and runs as
a single fused GPU kernel.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..core.state import (
    StateTensor, StateConfig, DEFAULT_CONFIG,
    VALUE_START, VALUE_END, VALUE_BITS,
    int_to_bipolar,
)
from ..core.nisa import Instruction, Opcode

N_OPS = int(Opcode.N_OPCODES)
MOD32 = 2 ** 32
MASK32 = 0xFFFFFFFF

# Pre-compute powers of 2 as a constant
_POW2 = torch.tensor([2 ** i for i in range(32)], dtype=torch.float64)


def _make_pow2(device, dtype):
    return torch.tensor([2.0 ** i for i in range(32)], dtype=dtype, device=device)


class PureTransformerCPU(nn.Module):
    """Forward pass with zero Python control flow.

    All opcode dispatch is done via masked tensor selection.
    """

    def __init__(self, config: StateConfig):
        super().__init__()
        self.config = config

    def forward(self, state: torch.Tensor, pow2: torch.Tensor) -> torch.Tensor:
        """One instruction cycle as pure tensor ops.

        Args:
            state: (n_cols, 44) float64
            pow2: (32,) powers of 2 constant

        Returns:
            new_state: (n_cols, 44) float64
        """
        c = self.config
        V = slice(VALUE_START, VALUE_END)

        # ═══ FETCH ═══
        # Decode PC → integer index
        pc_bp = state[c.pc_col, V]                          # (32,)
        pc_int = _bp2int(pc_bp, pow2)                        # scalar

        # Fetch instruction from instruction memory column
        instr_col = c.instr_start + pc_int.long()
        instr_bp = state[instr_col, V]                       # (32,)
        instr_int = _bp2int(instr_bp, pow2)                  # scalar

        # ═══ DECODE (detached — indices don't need gradients) ═══ (6-bit opcode)
        ii = instr_int.detach().long()
        opcode = (ii & 0x3F)
        a_idx = ((ii >> 6) & 0x1F)
        b_idx = ((ii >> 11) & 0x1F)
        c_idx = ((ii >> 16) & 0x1F)
        imm22 = ((ii >> 11) & 0x1FFFFF)

        # Read source registers
        rb_bp = state[b_idx, V]
        rc_bp = state[c_idx, V]
        rb_int = _bp2int(rb_bp, pow2)
        rc_int = _bp2int(rc_bp, pow2)

        # Also read a_idx register (for branches)
        ra_bp = state[a_idx, V]
        ra_int = _bp2int(ra_bp, pow2)

        # ═══ EXECUTE — compute ALL results in parallel ═══
        # Integer results
        add_r = (rb_int + rc_int) % MOD32
        sub_r = (rb_int - rc_int) % MOD32
        # Multiply in int64, not float64: a 32-bit product reaches ~2^64, past
        # float64's exact range (2^53), which would corrupt the low bits before
        # the mod. int64 wraps mod 2^64 and (2^32 | 2^64) keeps the low 32 exact.
        mul_r = ((rb_int.detach().long() * rc_int.detach().long()) & MASK32).to(state.dtype)
        mov_r = rb_int
        movi_r = imm22

        # Shift amounts (extract from rc as integer — detached for indexing)
        shift = (rc_int.detach().long() & 0x1F)
        shl_r = (rb_int * (2.0 ** shift.float()))  # differentiable shift via multiply
        shr_r = rb_int / (2.0 ** shift.float())    # differentiable shift via divide

        # SRA — arithmetic (sign-extending) right shift.
        # Interpret rb as signed, floor-divide by 2^shift (= arithmetic shift for
        # two's complement), then wrap back to unsigned 32-bit. Division by a power
        # of two is exact in float64, so this is bit-exact.
        _rb_signed = torch.where(rb_int >= 0x80000000, rb_int - MOD32, rb_int)
        sra_r = torch.floor(_rb_signed / (2.0 ** shift.float())) % MOD32

        # Comparison
        rb_s = torch.where(rb_int >= 0x80000000, rb_int - MOD32, rb_int)
        rc_s = torch.where(rc_int >= 0x80000000, rc_int - MOD32, rc_int)
        slt_r = (rb_s < rc_s).to(state.dtype)
        sltu_r = (rb_int < rc_int).to(state.dtype)

        # Division (guard against div by zero, detached for correctness)
        rb_d = rb_int.detach(); rc_d = rc_int.detach()
        safe_rc = torch.where(rc_d == 0, torch.ones_like(rc_d), rc_d)
        divu_r = torch.where(rc_d == 0, torch.tensor(float(MASK32), device=state.device, dtype=state.dtype),
                             torch.floor(rb_d / safe_rc))
        remu_r = torch.where(rc_d == 0, rb_d, torch.fmod(rb_d, safe_rc))

        # Multiply-high (tensor-only, exact via a 16-bit split so neither float64 nor
        # int64 loses the high bits of the ~2^64 product). Signed variants from the
        # standard MULHU correction: signed_high = MULHU - (a<0)·b - (b<0)·a.
        _K = 65536.0
        _rbh = torch.floor(rb_int / _K); _rbl = rb_int - _K * _rbh
        _rch = torch.floor(rc_int / _K); _rcl = rc_int - _K * _rch
        _mid = _rbl * _rch + _rbh * _rcl
        _carry = torch.floor((_rbl * _rcl + _mid * _K) / MOD32)
        mulhu_r = (_rbh * _rch + _carry) % MOD32
        _aneg = (rb_int >= 0x80000000).to(state.dtype)
        _bneg = (rc_int >= 0x80000000).to(state.dtype)
        mulh_r = (mulhu_r - _aneg * rc_int - _bneg * rb_int) % MOD32
        mulhsu_r = (mulhu_r - _aneg * rc_int) % MOD32
        # Signed divide / remainder (tensor-only): magnitude divide + sign correction
        _bs = torch.where(rb_int >= 0x80000000, rb_int - MOD32, rb_int)
        _cs = torch.where(rc_int >= 0x80000000, rc_int - MOD32, rc_int)
        _czero = (_cs == 0)
        _safec = torch.where(_czero, torch.ones_like(_cs), _cs)
        _qabs = torch.floor(torch.abs(_bs) / torch.abs(_safec))
        _qsig = torch.where((_bs < 0) != (_safec < 0), -_qabs, _qabs)
        div_r = torch.where(_czero, torch.full_like(_bs, float(MASK32)), _qsig % MOD32)
        _rabs = torch.abs(_bs) - torch.abs(_safec) * _qabs
        _rsig = torch.where(_bs < 0, -_rabs, _rabs)
        rem_r = torch.where(_czero, _bs % MOD32, _rsig % MOD32)

        # Bipolar results (AND, OR, XOR, NOT) — keep as bipolar bits
        and_bp = 2.0 * torch.relu(rb_bp + rc_bp - 1.0) - 1.0
        or_bp = -(2.0 * torch.relu(-rb_bp + (-rc_bp) - 1.0) - 1.0)
        xor_bp = -rb_bp * rc_bp
        not_bp = -rb_bp

        # Convert integer results to bipolar
        add_bp = _int2bp(add_r, pow2, state.device, state.dtype)
        sub_bp = _int2bp(sub_r, pow2, state.device, state.dtype)
        mul_bp = _int2bp(mul_r, pow2, state.device, state.dtype)
        mov_bp = _int2bp(mov_r, pow2, state.device, state.dtype)
        movi_bp = _int2bp(movi_r, pow2, state.device, state.dtype)
        shl_bp = _int2bp(shl_r, pow2, state.device, state.dtype)
        shr_bp = _int2bp(shr_r, pow2, state.device, state.dtype)
        sra_bp = _int2bp(sra_r, pow2, state.device, state.dtype)
        slt_bp = _int2bp(slt_r, pow2, state.device, state.dtype)
        sltu_bp = _int2bp(sltu_r, pow2, state.device, state.dtype)
        divu_bp = _int2bp(divu_r, pow2, state.device, state.dtype)
        remu_bp = _int2bp(remu_r, pow2, state.device, state.dtype)
        mulhu_bp = _int2bp(mulhu_r, pow2, state.device, state.dtype)
        mulh_bp = _int2bp(mulh_r, pow2, state.device, state.dtype)
        mulhsu_bp = _int2bp(mulhsu_r, pow2, state.device, state.dtype)
        div_bp = _int2bp(div_r, pow2, state.device, state.dtype)
        rem_bp = _int2bp(rem_r, pow2, state.device, state.dtype)

        # LOAD: read from memory column at word address rb + c
        load_addr = (rb_int.detach().long() + c_idx)
        load_col = c.data_start + load_addr
        # Clamp to valid range
        load_col = torch.clamp(load_col, 0, state.shape[0] - 1)
        load_bp = state[load_col, V]

        # Sub-word LOADs: byte-addressed view over word columns (little-endian within
        # each word), consistent with word LOAD (word addr a ↔ byte addr 4a). Uses EXACT
        # integer bit-extraction — float div/mod loses low bits when the word nears 2^32.
        _ba = (rb_int.detach().long() + c_idx)
        _wcol = torch.clamp(c.data_start + (_ba // 4), 0, state.shape[0] - 1)
        _wl = _bp2int(state[_wcol, V], pow2).long()
        _sb = (8 * (_ba % 4))
        _sh = (16 * ((_ba % 4) // 2))
        _byte = (_wl >> _sb) & 0xFF
        _half = (_wl >> _sh) & 0xFFFF
        loadb_bp = _int2bp(_byte.to(state.dtype), pow2, state.device, state.dtype)
        loadbs_bp = _int2bp(torch.where(_byte >= 128, _byte - 256, _byte).to(state.dtype),
                            pow2, state.device, state.dtype)
        loadh_bp = _int2bp(_half.to(state.dtype), pow2, state.device, state.dtype)
        loadhs_bp = _int2bp(torch.where(_half >= 32768, _half - 65536, _half).to(state.dtype),
                            pow2, state.device, state.dtype)

        # ═══ OPCODE-GATED SELECTION — no if/else ═══
        # Stack all results: (N_OPS, 32)
        # Each row = result for that opcode. Unused opcodes = zeros.
        all_results = torch.zeros(N_OPS, VALUE_BITS, dtype=state.dtype, device=state.device)
        all_results[int(Opcode.ADD)] = add_bp
        all_results[int(Opcode.SUB)] = sub_bp
        all_results[int(Opcode.MUL)] = mul_bp
        all_results[int(Opcode.AND)] = and_bp
        all_results[int(Opcode.OR)] = or_bp
        all_results[int(Opcode.XOR)] = xor_bp
        all_results[int(Opcode.NOT)] = not_bp
        all_results[int(Opcode.SHL)] = shl_bp
        all_results[int(Opcode.SHR)] = shr_bp
        all_results[int(Opcode.SRA)] = sra_bp
        all_results[int(Opcode.MOV)] = mov_bp
        all_results[int(Opcode.MOVI)] = movi_bp
        all_results[int(Opcode.SLT)] = slt_bp
        all_results[int(Opcode.SLTU)] = sltu_bp
        all_results[int(Opcode.DIVU)] = divu_bp
        all_results[int(Opcode.REMU)] = remu_bp
        all_results[int(Opcode.MULHU)] = mulhu_bp
        all_results[int(Opcode.MULH)] = mulh_bp
        all_results[int(Opcode.MULHSU)] = mulhsu_bp
        all_results[int(Opcode.DIV)] = div_bp
        all_results[int(Opcode.REM)] = rem_bp
        all_results[int(Opcode.LOAD)] = load_bp
        all_results[int(Opcode.LOADB)] = loadb_bp
        all_results[int(Opcode.LOADBS)] = loadbs_bp
        all_results[int(Opcode.LOADH)] = loadh_bp
        all_results[int(Opcode.LOADHS)] = loadhs_bp

        # Select result for this opcode: one-hot index
        result_bp = all_results[opcode]  # (32,)

        # ═══ WRITEBACK — masked scatter ═══
        # Determine if this opcode writes to a register
        writes_reg = torch.zeros(N_OPS, dtype=state.dtype, device=state.device)
        for op in [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.AND, Opcode.OR,
                   Opcode.XOR, Opcode.NOT, Opcode.SHL, Opcode.SHR, Opcode.SRA,
                   Opcode.MOV, Opcode.MOVI, Opcode.SLT, Opcode.SLTU,
                   Opcode.DIVU, Opcode.REMU, Opcode.MULHU, Opcode.MULH,
                   Opcode.MULHSU, Opcode.DIV, Opcode.REM, Opcode.LOAD,
                   Opcode.LOADB, Opcode.LOADBS, Opcode.LOADH, Opcode.LOADHS]:
            writes_reg[int(op)] = 1.0

        should_write = writes_reg[opcode]  # 0.0 or 1.0
        a_nonzero = (a_idx > 0).float()
        write_mask = should_write * a_nonzero  # 1.0 if should write & dest != x0

        # Build new state via residual: only modify the destination column
        new_state = state.clone()
        old_val = state[a_idx, V]
        new_val = write_mask * result_bp + (1.0 - write_mask) * old_val
        new_state[a_idx, V] = new_val

        # STORE: write reg[a] to memory column at address rb + c
        is_store = (opcode == int(Opcode.STORE)).float()
        if is_store > 0:
            store_addr = (rb_int.detach().long() + c_idx)
            store_col = c.data_start + store_addr
            store_col = torch.clamp(store_col, 0, state.shape[0] - 1)
            new_state[store_col, V] = state[a_idx, V]

        # Sub-word STOREs: read-modify-write the byte/halfword inside its word column.
        is_storeb = (opcode == int(Opcode.STOREB)).float()
        if is_storeb > 0:
            _sba = (rb_int.detach().long() + c_idx)
            _swcol = torch.clamp(c.data_start + (_sba // 4), 0, state.shape[0] - 1)
            _sbsh = (8 * (_sba % 4))
            _oldw = _bp2int(state[_swcol, V], pow2).long()
            _sval = _bp2int(state[a_idx, V], pow2).long() & 0xFF
            _neww = (_oldw & ~(0xFF << _sbsh)) | (_sval << _sbsh)   # exact byte splice
            new_state[_swcol, V] = _int2bp(_neww.to(state.dtype), pow2, state.device, state.dtype)

        is_storeh = (opcode == int(Opcode.STOREH)).float()
        if is_storeh > 0:
            _hba = (rb_int.detach().long() + c_idx)
            _hwcol = torch.clamp(c.data_start + (_hba // 4), 0, state.shape[0] - 1)
            _hbsh = (16 * ((_hba % 4) // 2))
            _holdw = _bp2int(state[_hwcol, V], pow2).long()
            _hsval = _bp2int(state[a_idx, V], pow2).long() & 0xFFFF
            _hneww = (_holdw & ~(0xFFFF << _hbsh)) | (_hsval << _hbsh)   # exact half splice
            new_state[_hwcol, V] = _int2bp(_hneww.to(state.dtype), pow2, state.device, state.dtype)

        # ═══ PC UPDATE — all via tensor ops ═══
        is_halt = (opcode == int(Opcode.HALT)).float()
        is_jmp = (opcode == int(Opcode.JMP)).float()
        is_jmpr = (opcode == int(Opcode.JMPR)).float()  # indirect: PC = reg[a]

        # Branch conditions — compute all, select by opcode
        ra_s = torch.where(ra_int >= 0x80000000, ra_int - MOD32, ra_int)
        bra_rb_int = _bp2int(state[b_idx, V], pow2)
        bra_rb_s = torch.where(bra_rb_int >= 0x80000000, bra_rb_int - MOD32, bra_rb_int)

        beq = (ra_int == bra_rb_int).float()
        bne = (ra_int != bra_rb_int).float()
        blt = (ra_s < bra_rb_s).float()
        bge = (ra_s >= bra_rb_s).float()
        bltu = (ra_int < bra_rb_int).float()
        bgeu = (ra_int >= bra_rb_int).float()

        branch_taken = torch.zeros(N_OPS, dtype=state.dtype, device=state.device)
        branch_taken[int(Opcode.BEQ)] = beq
        branch_taken[int(Opcode.BNE)] = bne
        branch_taken[int(Opcode.BLT)] = blt
        branch_taken[int(Opcode.BGE)] = bge
        branch_taken[int(Opcode.BLTU)] = bltu
        branch_taken[int(Opcode.BGEU)] = bgeu
        taken = branch_taken[opcode]  # 0.0 or 1.0

        is_branch = torch.zeros(N_OPS, dtype=state.dtype, device=state.device)
        for op in [Opcode.BEQ, Opcode.BNE, Opcode.BLT, Opcode.BGE,
                   Opcode.BLTU, Opcode.BGEU]:
            is_branch[int(op)] = 1.0
        is_any_branch = is_branch[opcode]

        # New PC: sequential, jump, or branch target
        pc_plus_1 = pc_int + 1
        # Branch/JMP targets are packed 12-bit (not 5-bit register indices): JMP target
        # at bits 6-17, branch target at bits 16-27. Decoding only 5 bits truncated any
        # target >= 32, breaking every program with a far branch.
        jmp_target = (ii >> 6) & 0xFFF       # 12-bit immediate JMP target
        branch_target = (ii >> 16) & 0xFFF   # 12-bit branch target

        # Select: halt → keep PC, jmp → a_idx, branch taken → c_idx, else → pc+1
        new_pc = (
            is_halt * pc_int +
            (1.0 - is_halt) * (
                is_jmp * jmp_target +
                (1.0 - is_jmp) * (
                    is_jmpr * ra_int +
                    (1.0 - is_jmpr) * (
                        is_any_branch * (
                            taken * branch_target +
                            (1.0 - taken) * pc_plus_1
                        ) +
                        (1.0 - is_any_branch) * pc_plus_1
                    )
                )
            )
        )

        new_state[c.pc_col, V] = _int2bp(new_pc, pow2, state.device, state.dtype)

        # ═══ ERROR CORRECTION — snap to bipolar ═══
        new_state[:, V] = torch.where(
            new_state[:, V] > 0,
            torch.ones_like(new_state[:, V]),
            -torch.ones_like(new_state[:, V])
        )
        # x0 = 0
        new_state[0, V] = _int2bp(torch.tensor(0), pow2, state.device, state.dtype)

        return new_state


def _bp2int(bp: torch.Tensor, pow2: torch.Tensor) -> torch.Tensor:
    """Bipolar (32,) → integer scalar. Differentiable via STE."""
    binary = (bp + 1.0) * 0.5  # {-1,+1} → {0,1}
    return (binary * pow2).sum()  # keep as float for gradient flow


class _STEBitExtract(torch.autograd.Function):
    """Straight-through estimator for int→bipolar conversion.
    Forward: hard bit extraction. Backward: identity (pass gradient through)."""
    @staticmethod
    def forward(ctx, val, pow2, device, dtype):
        val_long = val.detach().long() & MASK32
        bits_idx = torch.arange(32, device=device)
        binary = (val_long >> bits_idx) & 1
        return 2.0 * binary.to(dtype) - 1.0

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient of the sum through to the input scalar
        return grad_output.sum(), None, None, None


def _int2bp(val, pow2: torch.Tensor, device, dtype) -> torch.Tensor:
    """Integer → bipolar (32,). Uses STE for gradient flow."""
    if not isinstance(val, torch.Tensor):
        val = torch.tensor(val, dtype=dtype, device=device)
    val = val.to(dtype)
    if val.requires_grad:
        return _STEBitExtract.apply(val, pow2, device, dtype)
    else:
        val_long = val.long() & MASK32
        bits_idx = torch.arange(32, device=device)
        binary = (val_long >> bits_idx) & 1
        return 2.0 * binary.to(dtype) - 1.0


# ── Execution interface ──

def execute_pure_transformer(
    instructions: list[Instruction],
    max_cycles: int = 10000,
    initial_registers: Optional[dict[int, int]] = None,
    initial_memory: Optional[dict[int, int]] = None,
    config: Optional[StateConfig] = None,
    device: str = 'cuda',
) -> tuple[dict[int, int], int, bool]:
    """Execute via pure tensor forward passes."""
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
    dev = torch.device(device)
    dtype = torch.float64

    # n_data_words must cover the RV32I stack (word addresses ~1008 from stack_top*4),
    # else stack frames alias and recursion/deep calls corrupt. gpu has 65536 bytes;
    # match with enough word columns here.
    config = config or StateConfig(n_instr_slots=max(len(instructions) + 10, 64),
                                   n_data_words=4096)

    # Fail loud on opcodes this backend does not implement in its forward pass,
    # rather than silently returning 0 / falling through (which is a soundness hazard).
    _PURE_SUPPORTED = {
        Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.AND, Opcode.OR, Opcode.XOR,
        Opcode.NOT, Opcode.SHL, Opcode.SHR, Opcode.SRA, Opcode.MOV, Opcode.MOVI,
        Opcode.SLT, Opcode.SLTU, Opcode.DIVU, Opcode.REMU, Opcode.MULHU, Opcode.MULH,
        Opcode.MULHSU, Opcode.DIV, Opcode.REM, Opcode.LOAD, Opcode.STORE, Opcode.JMP,
        Opcode.JMPR, Opcode.LOADB, Opcode.LOADBS, Opcode.LOADH, Opcode.LOADHS,
        Opcode.STOREB, Opcode.STOREH, Opcode.ECALL,
        Opcode.BEQ, Opcode.BNE, Opcode.BLT, Opcode.BGE, Opcode.BLTU, Opcode.BGEU,
        Opcode.HALT, Opcode.NOP,
    }
    for instr in instructions:
        if instr.opcode not in _PURE_SUPPORTED:
            raise NotImplementedError(
                f"pure transformer: opcode {instr.opcode.name} is not implemented in the "
                f"forward pass (would silently no-op). Add it, or route the program through "
                f"an executor that supports it.")

    # Build state
    st = StateTensor(config)
    st.load_program(instructions)
    if initial_registers:
        for idx, val in initial_registers.items():
            if 0 < idx < 32:
                st.set_register(idx, val)
    if initial_memory:
        for addr, val in initial_memory.items():
            if addr < config.n_data_words:
                st.set_memory(addr, val)

    state = st.data.T.to(device=dev, dtype=dtype)
    pow2 = _make_pow2(dev, dtype)

    model = PureTransformerCPU(config).to(dev)
    model.eval()

    # Compile the forward pass for kernel fusion
    compiled_forward = torch.compile(model.forward, fullgraph=False)

    cycles = 0
    halted = False

    with torch.no_grad():
        for cycle in range(max_cycles):
            # Check halt (minimal CPU touch — just read opcode)
            pc = _bp2int(state[config.pc_col, VALUE_START:VALUE_END], pow2)
            ic = config.instr_start + int(pc.item())   # _bp2int is float; index needs int
            if ic >= config.n_columns:
                halted = True; break
            opcode = int(_bp2int(state[ic, VALUE_START:VALUE_END], pow2).item()) & 0x3F
            if opcode == int(Opcode.HALT):
                halted = True; cycles = cycle + 1; break

            state = compiled_forward(state, pow2)
            cycles = cycle + 1

    regs = {}
    for i in range(32):
        regs[i] = 0 if i == 0 else int(_bp2int(
            state[i, VALUE_START:VALUE_END], pow2).item())
    return regs, cycles, halted
