"""
Differentiable Executor for gradient-guided fuzzing.

De-quantizes the GPU executor: registers are float64 tensors with
requires_grad=True instead of bipolar-encoded. Arithmetic stays in
float space (exact for 32-bit integers in float64's 53-bit mantissa).

At branch points, the executor records a differentiable "branch distance"
— how close the condition was to flipping. Backpropagating through these
distances gives gradients on the input, which gradient descent uses to
steer inputs toward uncovered branches.

The control flow (which instruction executes next) is still discrete/hard.
Only the data flow is differentiable. This is the standard "trace-based"
approach used in search-based testing and concolic execution.
"""

import torch
from typing import Optional
from dataclasses import dataclass, field

from ..core.nisa import Instruction, Opcode


MOD32 = 2 ** 32


@dataclass
class BranchEvent:
    """A branch encountered during execution."""
    cycle: int
    pc: int
    opcode: Opcode
    distance: torch.Tensor  # differentiable: negative = taken, positive = not taken
    taken: bool
    target: int


@dataclass
class DiffExecutionResult:
    """Result from differentiable execution."""
    registers: dict[int, torch.Tensor]  # differentiable register values
    branch_events: list[BranchEvent]
    cycles: int
    halted: bool

    def reg_value(self, idx: int) -> int:
        """Get integer register value (detached)."""
        if idx == 0:
            return 0
        return int(self.registers.get(idx, torch.tensor(0.0)).detach().item()) & 0xFFFFFFFF

    @property
    def branch_distances(self) -> list[torch.Tensor]:
        """All branch distances as a list of differentiable tensors."""
        return [b.distance for b in self.branch_events]

    @property
    def coverage(self) -> set[tuple[int, bool]]:
        """Set of (pc, taken) pairs — which branches were hit and which direction."""
        return {(b.pc, b.taken) for b in self.branch_events}


def execute_differentiable(
    instructions: list[Instruction],
    input_values: dict[int, torch.Tensor],
    max_cycles: int = 10000,
    initial_memory: Optional[dict[int, float]] = None,
) -> DiffExecutionResult:
    """Execute a program with differentiable data flow.

    Args:
        instructions: NISA program
        input_values: {reg_idx: tensor} — input registers as differentiable tensors.
                      Use torch.tensor(value, dtype=torch.float64, requires_grad=True)
        max_cycles: max instruction cycles
        initial_memory: {addr: value} for pre-initialized memory

    Returns:
        DiffExecutionResult with differentiable register values and branch distances
    """
    dtype = torch.float64

    # Registers as differentiable scalars
    regs: dict[int, torch.Tensor] = {}
    for i in range(32):
        if i in input_values:
            regs[i] = input_values[i]
        else:
            regs[i] = torch.tensor(0.0, dtype=dtype)

    # Memory as differentiable tensors
    mem: dict[int, torch.Tensor] = {}
    if initial_memory:
        for addr, val in initial_memory.items():
            if isinstance(val, torch.Tensor):
                mem[addr] = val
            else:
                mem[addr] = torch.tensor(float(val), dtype=dtype)

    # Pre-decode
    decoded = [(instr.opcode, instr.a, instr.b, instr.c) for instr in instructions]

    branch_events: list[BranchEvent] = []
    pc = 0
    n_instr = len(instructions)
    cycles = 0
    halted = False

    for cycle in range(max_cycles):
        if pc >= n_instr:
            halted = True
            break

        op, a, b, c = decoded[pc]
        next_pc = pc + 1

        if op == Opcode.HALT:
            halted = True
            cycles = cycle + 1
            break

        elif op == Opcode.NOP:
            pass

        elif op == Opcode.MOVI:
            if a != 0:
                imm = ((b & 0xFFFF) << 16) | (c & 0xFFFF)
                regs[a] = torch.tensor(float(imm), dtype=dtype)

        elif op == Opcode.MOV:
            if a != 0:
                regs[a] = regs[b] + 0  # preserve gradient connection

        elif op == Opcode.ADD:
            if a != 0:
                regs[a] = _mod32(regs[b] + regs[c])

        elif op == Opcode.SUB:
            if a != 0:
                regs[a] = _mod32(regs[b] - regs[c])

        elif op == Opcode.MUL:
            if a != 0:
                regs[a] = _mod32(regs[b] * regs[c])

        elif op == Opcode.AND:
            if a != 0:
                regs[a] = _soft_and(regs[b], regs[c])

        elif op == Opcode.OR:
            if a != 0:
                regs[a] = _soft_or(regs[b], regs[c])

        elif op == Opcode.XOR:
            if a != 0:
                regs[a] = _soft_xor(regs[b], regs[c])

        elif op == Opcode.NOT:
            if a != 0:
                regs[a] = _soft_not(regs[b])

        elif op == Opcode.SHL:
            if a != 0:
                shift = int(regs[c].detach().item()) & 0x1F
                regs[a] = _mod32(regs[b] * (2 ** shift))

        elif op == Opcode.SHR:
            if a != 0:
                shift = int(regs[c].detach().item()) & 0x1F
                # Logical right shift: floor division by 2^shift
                regs[a] = torch.floor(regs[b] / (2 ** shift))

        elif op == Opcode.SRA:
            if a != 0:
                shift = int(regs[c].detach().item()) & 0x1F
                val_s = _to_signed(regs[b])
                regs[a] = _to_unsigned(torch.floor(val_s / (2 ** shift)))

        elif op == Opcode.SLT:
            if a != 0:
                diff = _to_signed(regs[b]) - _to_signed(regs[c])
                regs[a] = torch.sigmoid(-diff * 100)  # soft less-than

        elif op == Opcode.SLTU:
            if a != 0:
                diff = regs[b] - regs[c]
                regs[a] = torch.sigmoid(-diff * 100)

        elif op == Opcode.LOAD:
            if a != 0:
                base = int(regs[b].detach().item())
                addr = (base + c) & 0xFFFFFFFF
                if addr in mem:
                    regs[a] = mem[addr] + 0
                else:
                    regs[a] = torch.tensor(0.0, dtype=dtype)

        elif op == Opcode.STORE:
            base = int(regs[b].detach().item())
            addr = (base + c) & 0xFFFFFFFF
            mem[addr] = regs[a] + 0

        elif op == Opcode.JMP:
            next_pc = a

        elif op in (Opcode.BEQ, Opcode.BNE, Opcode.BLT, Opcode.BGE,
                    Opcode.BLTU, Opcode.BGEU):
            # Compute differentiable branch distance
            distance, taken = _branch_distance(op, regs[a], regs[b])
            branch_events.append(BranchEvent(
                cycle=cycle, pc=pc, opcode=op,
                distance=distance, taken=taken, target=c
            ))
            if taken:
                next_pc = c

        pc = next_pc
        cycles = cycle + 1

    return DiffExecutionResult(
        registers=regs,
        branch_events=branch_events,
        cycles=cycles,
        halted=halted,
    )


def _mod32(x: torch.Tensor) -> torch.Tensor:
    """Differentiable mod 2^32. Uses fmod which has gradients."""
    return torch.fmod(x, MOD32)


def _to_signed(x: torch.Tensor) -> torch.Tensor:
    """Convert unsigned 32-bit float to signed interpretation."""
    return torch.where(x >= 2**31, x - MOD32, x)


def _to_unsigned(x: torch.Tensor) -> torch.Tensor:
    """Convert signed to unsigned 32-bit."""
    return torch.where(x < 0, x + MOD32, x)


def _soft_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Approximate bitwise AND in float space.
    For integer values, this is exact when used with integer inputs."""
    ai, bi = int(a.detach().item()), int(b.detach().item())
    result = ai & bi
    # Use STE: forward = hard result, backward = gradient from a*b/max(a,b,1)
    return a - a.detach() + result  # straight-through estimator


def _soft_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ai, bi = int(a.detach().item()), int(b.detach().item())
    result = ai | bi
    return a - a.detach() + result


def _soft_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ai, bi = int(a.detach().item()), int(b.detach().item())
    result = ai ^ bi
    return a - a.detach() + result


def _soft_not(a: torch.Tensor) -> torch.Tensor:
    ai = int(a.detach().item()) & 0xFFFFFFFF
    result = (~ai) & 0xFFFFFFFF
    return a - a.detach() + result


def _branch_distance(op: Opcode, ra: torch.Tensor, rb: torch.Tensor
                     ) -> tuple[torch.Tensor, bool]:
    """Compute differentiable branch distance.

    Returns (distance, taken) where:
      distance < 0 means branch IS taken
      distance > 0 means branch is NOT taken
      distance ≈ 0 means branch is at the boundary

    The gradient of distance w.r.t. inputs tells us how to change
    inputs to flip the branch.
    """
    if op == Opcode.BEQ:
        # Taken if ra == rb → distance = |ra - rb|, taken when 0
        diff = ra - rb
        distance = diff * diff  # always positive, zero when equal
        taken = abs(diff.detach().item()) < 0.5
        if taken:
            distance = -distance  # negative = taken
    elif op == Opcode.BNE:
        diff = ra - rb
        distance = -(diff * diff)  # negative when not equal (taken)
        taken = abs(diff.detach().item()) >= 0.5
        if not taken:
            distance = -distance
    elif op == Opcode.BLT:
        # Taken if ra < rb (signed)
        distance = _to_signed(ra) - _to_signed(rb)  # negative when taken
        taken = distance.detach().item() < 0
    elif op == Opcode.BGE:
        # Taken if ra >= rb (signed)
        distance = _to_signed(rb) - _to_signed(ra)  # negative when taken
        taken = _to_signed(ra).detach().item() >= _to_signed(rb).detach().item()
    elif op == Opcode.BLTU:
        distance = ra - rb  # negative when taken
        taken = ra.detach().item() < rb.detach().item()
    elif op == Opcode.BGEU:
        distance = rb - ra
        taken = ra.detach().item() >= rb.detach().item()
    else:
        distance = torch.tensor(0.0, dtype=ra.dtype)
        taken = False

    return distance, taken


# ── Fuzzing interface ──

def fuzz(
    instructions: list[Instruction],
    n_input_regs: int = 1,
    n_iterations: int = 200,
    lr: float = 10.0,
    max_cycles: int = 5000,
    initial_memory: Optional[dict[int, float]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Gradient-guided fuzzer.

    Optimizes input register values to maximize branch coverage.

    Args:
        instructions: NISA program to fuzz
        n_input_regs: number of input registers (r1, r2, ...)
        n_iterations: optimization iterations
        lr: learning rate for Adam optimizer
        max_cycles: max execution cycles per run
        initial_memory: pre-initialized memory
        seed: random seed
        verbose: print progress

    Returns:
        dict with:
          'best_inputs': dict of input values that achieved best coverage
          'best_coverage': set of (pc, taken) pairs
          'coverage_history': list of coverage counts per iteration
          'inputs_history': list of input dicts tried
    """
    torch.manual_seed(seed)

    # Initialize inputs as random differentiable parameters
    params = []
    for i in range(1, n_input_regs + 1):
        p = torch.tensor(torch.randn(1).item() * 100, dtype=torch.float64,
                         requires_grad=True)
        params.append(p)

    optimizer = torch.optim.Adam(params, lr=lr)

    best_coverage: set[tuple[int, bool]] = set()
    best_inputs: dict[int, int] = {}
    coverage_history: list[int] = []
    inputs_history: list[dict[int, int]] = []
    all_branches: set[int] = set()  # all branch PCs seen

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Build input dict
        input_values = {}
        for i, p in enumerate(params):
            # Clamp to valid 32-bit range
            val = torch.clamp(p, 0, MOD32 - 1)
            input_values[i + 1] = val

        # Execute
        result = execute_differentiable(
            instructions, input_values,
            max_cycles=max_cycles,
            initial_memory=initial_memory,
        )

        # Track coverage
        current_coverage = result.coverage
        coverage_history.append(len(current_coverage))

        if len(current_coverage) > len(best_coverage):
            best_coverage = current_coverage
            best_inputs = {i + 1: int(p.detach().item()) & 0xFFFFFFFF
                           for i, p in enumerate(params)}

        inputs_history.append(
            {i + 1: int(p.detach().item()) & 0xFFFFFFFF for i, p in enumerate(params)}
        )

        # Track all branch PCs
        for b in result.branch_events:
            all_branches.add(b.pc)

        # Compute loss: minimize distance to flipping uncovered branches
        if result.branch_distances:
            # For each branch, we want to get close to flipping it
            # Loss = sum of |distance| for branches we haven't covered in the opposite direction
            loss = torch.tensor(0.0, dtype=torch.float64)

            for event in result.branch_events:
                opposite = (event.pc, not event.taken)
                if opposite not in best_coverage:
                    # We want distance → 0 to flip this branch
                    # Use smooth L1 to avoid gradient explosion
                    loss = loss + torch.nn.functional.smooth_l1_loss(
                        event.distance, torch.tensor(0.0, dtype=torch.float64),
                        reduction='sum'
                    )

            # Also add a diversity term: push inputs apart across iterations
            loss = loss + 0.01 * sum(p * p for p in params)

            loss.backward()
            optimizer.step()
        else:
            # No branches → random perturbation
            with torch.no_grad():
                for p in params:
                    p.add_(torch.randn(1).item() * 50)

        if verbose and (iteration % 50 == 0 or iteration == n_iterations - 1):
            covered = len(best_coverage)
            total_branches = len(all_branches)
            max_possible = total_branches * 2  # each branch can be taken or not
            pct = covered / max_possible * 100 if max_possible > 0 else 0
            input_str = ", ".join(f"r{i+1}={int(p.detach().item()) & 0xFFFFFFFF}"
                                  for i, p in enumerate(params))
            print(f"  iter {iteration:4d}: coverage {covered}/{max_possible} "
                  f"({pct:.0f}%) branches={total_branches} | {input_str}")

    return {
        'best_inputs': best_inputs,
        'best_coverage': best_coverage,
        'coverage_history': coverage_history,
        'inputs_history': inputs_history,
        'all_branches': all_branches,
    }
