"""
Runtime Executor: Iterative forward pass loop.

Loads a program into the state tensor, then repeatedly executes
instruction cycles until HALT or a cycle limit is reached.

Phase 1: Uses the reference executor (direct interpretation via
bipolar arithmetic). Phase 2+: Will use actual transformer forward
passes with constructed weight matrices.
"""

import torch
from typing import Optional
from ..core.state import StateTensor, StateConfig
from ..core.nisa import Instruction, Opcode, movi, add, halt
from ..weights.engine import execute_one_cycle


class ExecutionResult:
    """Result of program execution."""

    def __init__(self, state: StateTensor, cycles: int, halted: bool):
        self.state = state
        self.cycles = cycles
        self.halted = halted

    @property
    def registers(self) -> dict[int, int]:
        return self.state.dump_registers()

    def reg(self, idx: int) -> int:
        return self.state.get_register(idx)

    def __repr__(self):
        status = "HALTED" if self.halted else "RUNNING"
        return f"ExecutionResult({status}, cycles={self.cycles}, PC={self.state.get_pc()})"


def execute_program(
    instructions: list[Instruction],
    max_cycles: int = 10000,
    initial_registers: Optional[dict[int, int]] = None,
    initial_memory: Optional[dict[int, int]] = None,
    config: Optional[StateConfig] = None,
    trace: bool = False,
) -> ExecutionResult:
    """Execute a NISA program from start to finish.

    Args:
        instructions: list of NISA instructions to execute
        max_cycles: maximum number of instruction cycles before forced stop
        initial_registers: optional dict of {reg_idx: value} to initialize
        initial_memory: optional dict of {addr: value} to initialize
        config: optional state configuration (defaults to DEFAULT_CONFIG)
        trace: if True, print each instruction as it executes

    Returns:
        ExecutionResult with final state, cycle count, and halt status
    """
    # Initialize state
    state = StateTensor(config)

    # Load program
    state.load_program(instructions)

    # Set initial register values
    if initial_registers:
        for reg, val in initial_registers.items():
            state.set_register(reg, val)

    # Set initial memory values
    if initial_memory:
        for addr, val in initial_memory.items():
            state.set_memory(addr, val)

    # Execute
    cycles = 0
    halted = False

    for cycle in range(max_cycles):
        if trace:
            pc = state.get_pc()
            if pc < len(instructions):
                instr = instructions[pc]
                print(f"  cycle {cycle:4d}: PC={pc:3d}  {instr}")

        running = execute_one_cycle(state)
        cycles = cycle + 1

        if not running:
            halted = True
            break

    return ExecutionResult(state, cycles, halted)


# --- Convenience functions for quick testing ---

def run_simple_add(a: int, b: int) -> int:
    """Quick test: compute a + b using the full pipeline."""
    program = [
        movi(1, a),     # r1 = a
        movi(2, b),     # r2 = b
        add(3, 1, 2),   # r3 = r1 + r2
        halt(),
    ]
    result = execute_program(program)
    return result.reg(3)
