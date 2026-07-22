"""
Tests for single instruction execution through the full pipeline.

Tests the complete cycle: load program → execute → verify registers.
"""

import pytest
from ..core.nisa import (
    Instruction, Opcode, movi, add, sub, halt, nop,
)
from ..core.state import StateTensor, StateConfig
from ..runtime.executor import execute_program, run_simple_add


class TestMOVI:
    def test_movi_small(self):
        program = [movi(1, 42), halt()]
        result = execute_program(program)
        assert result.halted
        assert result.reg(1) == 42

    def test_movi_zero(self):
        program = [movi(1, 0), halt()]
        result = execute_program(program)
        assert result.reg(1) == 0

    def test_movi_large(self):
        # 21-bit immediate max (opcode field widened 5->6 bits to fit all 40 opcodes)
        program = [movi(1, 0x1FFFFF), halt()]
        result = execute_program(program)
        assert result.reg(1) == 0x1FFFFF

    def test_movi_to_x0_ignored(self):
        """Writing to x0 should have no effect."""
        program = [movi(0, 42), halt()]
        result = execute_program(program)
        assert result.reg(0) == 0


class TestADD:
    def test_add_simple(self):
        program = [
            movi(1, 5),
            movi(2, 3),
            add(3, 1, 2),
            halt(),
        ]
        result = execute_program(program)
        assert result.halted
        assert result.reg(3) == 8
        assert result.cycles == 4

    def test_add_zero(self):
        program = [
            movi(1, 100),
            movi(2, 0),
            add(3, 1, 2),
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 100

    def test_add_commutative(self):
        program = [
            movi(1, 7),
            movi(2, 13),
            add(3, 1, 2),  # 7 + 13
            add(4, 2, 1),  # 13 + 7
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 20
        assert result.reg(4) == 20

    def test_add_self(self):
        program = [
            movi(1, 50),
            add(2, 1, 1),  # 50 + 50
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(2) == 100

    def test_add_chain(self):
        program = [
            movi(1, 1),
            movi(2, 2),
            add(3, 1, 2),  # 3
            add(4, 3, 2),  # 5
            add(5, 4, 3),  # 8
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 3
        assert result.reg(4) == 5
        assert result.reg(5) == 8

    def test_run_simple_add(self):
        assert run_simple_add(5, 3) == 8
        assert run_simple_add(0, 0) == 0
        assert run_simple_add(100, 200) == 300


class TestSUB:
    def test_sub_simple(self):
        program = [
            movi(1, 10),
            movi(2, 3),
            Instruction(Opcode.SUB, a=3, b=1, c=2),
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 7

    def test_sub_to_zero(self):
        program = [
            movi(1, 42),
            Instruction(Opcode.SUB, a=2, b=1, c=1),  # r1 - r1
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(2) == 0


class TestALU:
    def test_and(self):
        program = [
            movi(1, 0xFF),
            movi(2, 0x0F),
            Instruction(Opcode.AND, a=3, b=1, c=2),
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 0x0F

    def test_or(self):
        program = [
            movi(1, 0xF0),
            movi(2, 0x0F),
            Instruction(Opcode.OR, a=3, b=1, c=2),
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 0xFF

    def test_xor(self):
        program = [
            movi(1, 0xFF),
            movi(2, 0x0F),
            Instruction(Opcode.XOR, a=3, b=1, c=2),
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 0xF0

    def test_not(self):
        program = [
            movi(1, 0xFF),
            Instruction(Opcode.NOT, a=2, b=1, c=0),
            halt(),
        ]
        result = execute_program(program)
        # NOT of 0xFF (lower 8 bits set) = 0xFFFFFF00
        assert result.reg(2) == 0xFFFFFF00


class TestControlFlow:
    def test_halt(self):
        program = [halt()]
        result = execute_program(program)
        assert result.halted
        assert result.cycles == 1

    def test_nop(self):
        program = [nop(), nop(), halt()]
        result = execute_program(program)
        assert result.halted
        assert result.cycles == 3

    def test_jmp(self):
        program = [
            Instruction(Opcode.JMP, a=2),  # jump to instruction 2
            halt(),                         # should be skipped
            movi(1, 42),                    # should execute
            halt(),
        ]
        result = execute_program(program)
        assert result.halted
        assert result.reg(1) == 42

    def test_beq_taken(self):
        program = [
            movi(1, 5),
            movi(2, 5),
            Instruction(Opcode.BEQ, a=1, b=2, c=4),  # if r1==r2, goto 4
            halt(),  # should be skipped
            movi(3, 99),  # should execute
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 99

    def test_beq_not_taken(self):
        program = [
            movi(1, 5),
            movi(2, 6),
            Instruction(Opcode.BEQ, a=1, b=2, c=5),  # if r1==r2, goto 5
            movi(3, 42),  # should execute (branch not taken)
            halt(),
            movi(3, 99),  # should NOT execute
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 42

    def test_bne(self):
        program = [
            movi(1, 5),
            movi(2, 6),
            Instruction(Opcode.BNE, a=1, b=2, c=4),  # if r1!=r2, goto 4
            halt(),
            movi(3, 77),
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 77


class TestMemory:
    def test_store_load(self):
        program = [
            movi(1, 42),
            movi(2, 0),  # base address = 0
            Instruction(Opcode.STORE, a=1, b=2, c=0),  # mem[0] = 42
            Instruction(Opcode.LOAD, a=3, b=2, c=0),   # r3 = mem[0]
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 42

    def test_store_load_offset(self):
        program = [
            movi(1, 99),
            movi(2, 0),
            Instruction(Opcode.STORE, a=1, b=2, c=5),  # mem[5] = 99
            Instruction(Opcode.LOAD, a=3, b=2, c=5),   # r3 = mem[5]
            halt(),
        ]
        result = execute_program(program)
        assert result.reg(3) == 99


class TestTrace:
    def test_trace_output(self, capsys):
        program = [movi(1, 5), movi(2, 3), add(3, 1, 2), halt()]
        result = execute_program(program, trace=True)
        captured = capsys.readouterr()
        assert "MOVI" in captured.out
        assert "ADD" in captured.out
        assert result.reg(3) == 8


class TestMaxCycles:
    def test_infinite_loop_stops(self):
        program = [
            Instruction(Opcode.JMP, a=0),  # infinite loop
        ]
        result = execute_program(program, max_cycles=100)
        assert not result.halted
        assert result.cycles == 100


class TestFibonacci:
    def test_fibonacci_5(self):
        """Compute fib(5) = 5 using a loop."""
        # r1 = n (input), r2 = fib(i-1), r3 = fib(i), r4 = temp, r5 = counter
        program = [
            movi(1, 5),     # 0: n = 5
            movi(2, 0),     # 1: fib_prev = 0
            movi(3, 1),     # 2: fib_curr = 1
            movi(5, 1),     # 3: counter = 1
            # Loop start (instruction 4):
            Instruction(Opcode.BGE, a=5, b=1, c=10),  # 4: if counter >= n, goto end
            Instruction(Opcode.ADD, a=4, b=2, c=3),   # 5: temp = prev + curr
            Instruction(Opcode.MOV, a=2, b=3, c=0),   # 6: prev = curr
            Instruction(Opcode.MOV, a=3, b=4, c=0),   # 7: curr = temp
            movi(6, 1),                                 # 8: r6 = 1
            Instruction(Opcode.ADD, a=5, b=5, c=6),   # 9: counter++
            Instruction(Opcode.JMP, a=4),               # 10: goto loop start -- wait, should be 4
            halt(),                                      # 11: end
        ]
        # Fix: BGE target should be 11 (halt), JMP target should be 4
        program[4] = Instruction(Opcode.BGE, a=5, b=1, c=11)
        program[10] = Instruction(Opcode.JMP, a=4)

        result = execute_program(program, trace=False)
        assert result.halted
        assert result.reg(3) == 5, f"fib(5) = {result.reg(3)}, expected 5"

    def test_fibonacci_10(self):
        """Compute fib(10) = 55."""
        program = [
            movi(1, 10),    # 0: n = 10
            movi(2, 0),     # 1: fib_prev = 0
            movi(3, 1),     # 2: fib_curr = 1
            movi(5, 1),     # 3: counter = 1
            Instruction(Opcode.BGE, a=5, b=1, c=11),  # 4: if counter >= n, end
            Instruction(Opcode.ADD, a=4, b=2, c=3),   # 5: temp = prev + curr
            Instruction(Opcode.MOV, a=2, b=3, c=0),   # 6: prev = curr
            Instruction(Opcode.MOV, a=3, b=4, c=0),   # 7: curr = temp
            movi(6, 1),                                 # 8: r6 = 1
            Instruction(Opcode.ADD, a=5, b=5, c=6),   # 9: counter++
            Instruction(Opcode.JMP, a=4),               # 10: goto loop
            halt(),                                      # 11: end
        ]
        result = execute_program(program)
        assert result.halted
        assert result.reg(3) == 55, f"fib(10) = {result.reg(3)}, expected 55"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
