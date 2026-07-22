"""
Tests for the GPU executor.

Verifies correctness against the reference executor and benchmarks
GPU vs CPU performance.
"""

import time
import pytest
import torch

from ..core.nisa import Instruction, Opcode, movi, add, sub, halt, nop
from ..core.state import StateConfig
from ..runtime.executor import execute_program
from ..runtime.gpu_executor import gpu_execute
from ..tests.test_md5 import generate_md5_program, format_md5_hash, md5_reference


class TestGPUCorrectness:
    """Verify GPU executor matches reference executor."""

    def test_simple_add(self):
        prog = [movi(1, 5), movi(2, 3), add(3, 1, 2), halt()]
        result = gpu_execute(prog, device='cuda')
        assert result.halted
        assert result.reg(3) == 8

    def test_fibonacci_10(self):
        prog = [
            movi(1, 10), movi(2, 0), movi(3, 1), movi(5, 1),
            Instruction(Opcode.BGE, a=5, b=1, c=11),
            Instruction(Opcode.ADD, a=4, b=2, c=3),
            Instruction(Opcode.MOV, a=2, b=3, c=0),
            Instruction(Opcode.MOV, a=3, b=4, c=0),
            movi(6, 1),
            Instruction(Opcode.ADD, a=5, b=5, c=6),
            Instruction(Opcode.JMP, a=4),
            halt(),
        ]
        result = gpu_execute(prog, device='cuda')
        assert result.halted
        assert result.reg(3) == 55

    def test_all_alu_ops(self):
        prog = [
            movi(1, 0xFF),
            movi(2, 0x0F),
            Instruction(Opcode.AND, a=3, b=1, c=2),   # 0x0F
            Instruction(Opcode.OR, a=4, b=1, c=2),    # 0xFF
            Instruction(Opcode.XOR, a=5, b=1, c=2),   # 0xF0
            Instruction(Opcode.NOT, a=6, b=1, c=0),   # 0xFFFFFF00
            Instruction(Opcode.SUB, a=7, b=1, c=2),   # 0xF0
            halt(),
        ]
        result = gpu_execute(prog, device='cuda')
        assert result.reg(3) == 0x0F
        assert result.reg(4) == 0xFF
        assert result.reg(5) == 0xF0
        assert result.reg(6) == 0xFFFFFF00
        assert result.reg(7) == 0xF0

    def test_shifts(self):
        prog = [
            movi(1, 1),
            movi(2, 4),
            Instruction(Opcode.SHL, a=3, b=1, c=2),   # 1 << 4 = 16
            movi(1, 256),
            Instruction(Opcode.SHR, a=4, b=1, c=2),   # 256 >> 4 = 16
            halt(),
        ]
        result = gpu_execute(prog, device='cuda')
        assert result.reg(3) == 16
        assert result.reg(4) == 16

    def test_memory(self):
        prog = [
            movi(1, 42),
            movi(2, 0),
            Instruction(Opcode.STORE, a=1, b=2, c=0),
            Instruction(Opcode.LOAD, a=3, b=2, c=0),
            halt(),
        ]
        result = gpu_execute(prog, device='cuda')
        assert result.reg(3) == 42

    def test_branches(self):
        prog = [
            movi(1, 5), movi(2, 5),
            Instruction(Opcode.BEQ, a=1, b=2, c=4),
            halt(),
            movi(3, 99),
            halt(),
        ]
        result = gpu_execute(prog, device='cuda')
        assert result.reg(3) == 99

    def test_md5_empty(self):
        """The big test: MD5 of empty string on GPU."""
        prog, mem = generate_md5_program(b"")
        result = gpu_execute(prog, initial_memory=mem, device='cuda',
                             max_cycles=len(prog) + 100)
        assert result.halted
        got = format_md5_hash(result.reg(1), result.reg(2),
                              result.reg(3), result.reg(4))
        expected = md5_reference(b"")
        assert got == expected, f"GPU MD5('') = {got}, expected {expected}"

    def test_md5_abc(self):
        """MD5 of 'abc' on GPU."""
        prog, mem = generate_md5_program(b"abc")
        result = gpu_execute(prog, initial_memory=mem, device='cuda',
                             max_cycles=len(prog) + 100)
        assert result.halted
        got = format_md5_hash(result.reg(1), result.reg(2),
                              result.reg(3), result.reg(4))
        expected = md5_reference(b"abc")
        assert got == expected, f"GPU MD5('abc') = {got}, expected {expected}"


class TestGPUBenchmark:
    """Benchmark GPU vs reference executor."""

    def test_benchmark_md5(self):
        """Benchmark MD5 execution: GPU vs reference."""
        prog, mem = generate_md5_program(b"")
        config = StateConfig(n_instr_slots=2048)

        # Reference executor (CPU, ripple-carry)
        t0 = time.time()
        ref_result = execute_program(prog, initial_memory=mem, config=config,
                                     max_cycles=len(prog) + 100)
        t_ref = time.time() - t0

        # GPU executor (vectorized bipolar)
        # Warmup
        gpu_execute(prog, initial_memory=mem, device='cuda',
                    max_cycles=len(prog) + 100)

        t0 = time.time()
        gpu_result = gpu_execute(prog, initial_memory=mem, device='cuda',
                                 max_cycles=len(prog) + 100)
        t_gpu = time.time() - t0

        # Verify same result
        for r in [1, 2, 3, 4]:
            assert ref_result.reg(r) == gpu_result.reg(r), \
                f"Register r{r} mismatch: ref={ref_result.reg(r)}, gpu={gpu_result.reg(r)}"

        speedup = t_ref / t_gpu if t_gpu > 0 else float('inf')

        print(f"\n{'='*50}")
        print(f"MD5 Benchmark ({ref_result.cycles} cycles)")
        print(f"  Reference (CPU, ripple-carry): {t_ref:.3f}s")
        print(f"  GPU (vectorized bipolar):      {t_gpu:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        print(f"{'='*50}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
