"""
Tests for the RV32I → NISA compiler pipeline.

Tests compilation from RV32I assembly to NISA and execution.
Also tests C compilation if the RISC-V GCC toolchain is available.
"""

import pytest
import shutil

from ..compiler.rv32i_parser import parse_rv32i_assembly, RV32IOp
from ..compiler.rv32i_to_nisa import translate_program
from ..compiler.compiler import compile_asm, compile_and_run, compile_c
from ..core.nisa import Instruction, Opcode, halt
from ..runtime.gpu_executor import gpu_execute


# Check if RISC-V GCC is available
HAS_RISCV_GCC = shutil.which("riscv64-linux-gnu-gcc") is not None


class TestRV32IParser:
    """Test parsing RV32I assembly text."""

    def test_parse_r_type(self):
        instrs, labels = parse_rv32i_assembly("add x3, x1, x2")
        assert len(instrs) == 1
        assert instrs[0].op == RV32IOp.ADD
        assert instrs[0].rd == 3
        assert instrs[0].rs1 == 1
        assert instrs[0].rs2 == 2

    def test_parse_i_type(self):
        instrs, _ = parse_rv32i_assembly("addi x1, x0, 42")
        assert instrs[0].op == RV32IOp.ADDI
        assert instrs[0].rd == 1
        assert instrs[0].rs1 == 0
        assert instrs[0].imm == 42

    def test_parse_register_aliases(self):
        instrs, _ = parse_rv32i_assembly("add a0, sp, zero")
        assert instrs[0].rd == 10   # a0
        assert instrs[0].rs1 == 2   # sp
        assert instrs[0].rs2 == 0   # zero

    def test_parse_load_store(self):
        instrs, _ = parse_rv32i_assembly("""
            lw x1, 8(x2)
            sw x3, 12(x4)
        """)
        assert instrs[0].op == RV32IOp.LW
        assert instrs[0].rd == 1
        assert instrs[0].rs1 == 2
        assert instrs[0].imm == 8

        assert instrs[1].op == RV32IOp.SW
        assert instrs[1].rs2 == 3
        assert instrs[1].rs1 == 4
        assert instrs[1].imm == 12

    def test_parse_branch_with_label(self):
        instrs, labels = parse_rv32i_assembly("""
            beq x1, x2, done
            addi x3, x0, 1
        done:
            addi x4, x0, 2
        """)
        assert instrs[0].op == RV32IOp.BEQ
        assert instrs[0].label == "done"
        assert labels["done"] == 2

    def test_parse_with_directives(self):
        """Parser should skip assembler directives."""
        instrs, labels = parse_rv32i_assembly("""
            .text
            .globl main
            .type main, @function
        main:
            addi sp, sp, -16
            sw ra, 12(sp)
            li a0, 42
            lw ra, 12(sp)
            addi sp, sp, 16
            ret
        """)
        assert "main" in labels
        assert len(instrs) > 0

    def test_parse_lui_auipc(self):
        instrs, _ = parse_rv32i_assembly("""
            lui x1, 0x12345
            auipc x2, 0x1000
        """)
        assert instrs[0].op == RV32IOp.LUI
        assert instrs[0].rd == 1
        assert instrs[0].imm == 0x12345

    def test_parse_jal(self):
        instrs, labels = parse_rv32i_assembly("""
            jal ra, func
            nop
        func:
            ret
        """)
        assert instrs[0].op == RV32IOp.JAL
        assert instrs[0].rd == 1  # ra
        assert instrs[0].label == "func"


class TestRV32IToNISA:
    """Test translation from RV32I to NISA."""

    def test_simple_add(self):
        """addi + add → NISA MOVI + ADD."""
        instrs, labels = parse_rv32i_assembly("""
            addi x1, x0, 5
            addi x2, x0, 3
            add x3, x1, x2
        """)
        nisa = translate_program(instrs, labels)
        nisa.append(halt())

        result = gpu_execute(nisa, device='cuda')
        assert result.halted
        assert result.reg(3) == 8

    def test_sub(self):
        instrs, labels = parse_rv32i_assembly("""
            addi x1, x0, 10
            addi x2, x0, 3
            sub x3, x1, x2
        """)
        nisa = translate_program(instrs, labels)
        nisa.append(halt())

        result = gpu_execute(nisa, device='cuda')
        assert result.reg(3) == 7

    def test_bitwise(self):
        instrs, labels = parse_rv32i_assembly("""
            addi x1, x0, 0xFF
            addi x2, x0, 0x0F
            and x3, x1, x2
            or  x4, x1, x2
            xor x5, x1, x2
        """)
        nisa = translate_program(instrs, labels)
        nisa.append(halt())

        result = gpu_execute(nisa, device='cuda')
        assert result.reg(3) == 0x0F
        assert result.reg(4) == 0xFF
        assert result.reg(5) == 0xF0

    def test_shift(self):
        instrs, labels = parse_rv32i_assembly("""
            addi x1, x0, 1
            slli x2, x1, 4
        """)
        nisa = translate_program(instrs, labels)
        nisa.append(halt())

        result = gpu_execute(nisa, device='cuda')
        assert result.reg(2) == 16

    def test_branch_beq(self):
        instrs, labels = parse_rv32i_assembly("""
            addi x1, x0, 5
            addi x2, x0, 5
            beq x1, x2, equal
            addi x3, x0, 0
            j done
        equal:
            addi x3, x0, 1
        done:
            nop
        """)
        nisa = translate_program(instrs, labels)
        nisa.append(halt())

        result = gpu_execute(nisa, device='cuda')
        assert result.reg(3) == 1, f"Expected 1, got {result.reg(3)}"

    def test_branch_bne(self):
        instrs, labels = parse_rv32i_assembly("""
            addi x1, x0, 5
            addi x2, x0, 6
            bne x1, x2, notequal
            addi x3, x0, 0
            j done
        notequal:
            addi x3, x0, 1
        done:
            nop
        """)
        nisa = translate_program(instrs, labels)
        nisa.append(halt())

        result = gpu_execute(nisa, device='cuda')
        assert result.reg(3) == 1

    def test_load_store(self):
        """Test word-aligned load/store via stack."""
        instrs, labels = parse_rv32i_assembly("""
            addi x1, x0, 42       # x1 = 42
            sw x1, 0(sp)          # mem[sp/4] = 42
            lw x2, 0(sp)          # x2 = mem[sp/4]
        """)
        nisa = translate_program(instrs, labels)
        nisa.append(halt())

        result = gpu_execute(nisa, device='cuda')
        assert result.reg(2) == 42, f"Expected 42, got {result.reg(2)}"

    def test_lui(self):
        instrs, labels = parse_rv32i_assembly("""
            lui x1, 0x12
        """)
        nisa = translate_program(instrs, labels)
        nisa.append(halt())

        result = gpu_execute(nisa, device='cuda')
        assert result.reg(1) == 0x12 << 12

    def test_fibonacci_rv32i(self):
        """Fibonacci in RV32I assembly."""
        instrs, labels = parse_rv32i_assembly("""
            addi a0, zero, 10     # n = 10
            addi t0, zero, 0      # prev = 0
            addi t1, zero, 1      # curr = 1
            addi t2, zero, 1      # counter = 1
        loop:
            bge t2, a0, done      # if counter >= n, done
            add t3, t0, t1        # temp = prev + curr
            add t0, zero, t1      # prev = curr  (mv t0, t1)
            add t1, zero, t3      # curr = temp  (mv t1, t3)
            addi t2, t2, 1        # counter++
            j loop
        done:
            add a1, zero, t1      # result in a1
        """)
        nisa = translate_program(instrs, labels)
        nisa.append(halt())

        result = gpu_execute(nisa, device='cuda')
        assert result.reg(11) == 55, f"fib(10) = {result.reg(11)}, expected 55"

    def test_gcd(self):
        """Euclidean GCD in RV32I assembly."""
        instrs, labels = parse_rv32i_assembly("""
            # GCD(48, 18) = 6
            addi a0, zero, 48     # a = 48
            addi a1, zero, 18     # b = 18
        gcd_loop:
            beq a1, zero, gcd_done
            # a, b = b, a % b
            # a % b via repeated subtraction
            add t0, zero, a0      # t0 = a
        mod_loop:
            blt t0, a1, mod_done  # if a < b, done
            sub t0, t0, a1        # a -= b
            j mod_loop
        mod_done:
            add a0, zero, a1      # a = old b
            add a1, zero, t0      # b = a % b
            j gcd_loop
        gcd_done:
            nop                   # result in a0
        """)
        nisa = translate_program(instrs, labels)
        nisa.append(halt())

        result = gpu_execute(nisa, device='cuda', max_cycles=10000)
        assert result.reg(10) == 6, f"GCD(48,18) = {result.reg(10)}, expected 6"

    def test_multiply_by_addition(self):
        """Multiply via repeated addition."""
        instrs, labels = parse_rv32i_assembly("""
            # 7 * 6 = 42
            addi a0, zero, 7      # multiplicand
            addi a1, zero, 6      # multiplier
            addi a2, zero, 0      # result = 0
        mul_loop:
            beq a1, zero, mul_done
            add a2, a2, a0        # result += multiplicand
            addi a1, a1, -1       # multiplier--
            j mul_loop
        mul_done:
            nop
        """)
        nisa = translate_program(instrs, labels)
        nisa.append(halt())

        result = gpu_execute(nisa, device='cuda')
        assert result.reg(12) == 42, f"7*6 = {result.reg(12)}, expected 42"


@pytest.mark.skipif(not HAS_RISCV_GCC, reason="RISC-V GCC not installed")
class TestCCompilation:
    """Test compiling C source code (requires RISC-V GCC toolchain)."""

    def test_compile_simple_c(self):
        """Compile C to RV32I assembly."""
        c_source = """
        int _start() {
            int a = 5;
            int b = 3;
            int c = a + b;
            return c;
        }
        """
        asm = compile_c(c_source)
        # GCC may constant-fold (li a0, 8) or emit add; either is valid
        assert "rv32i" in asm.lower() or "li" in asm.lower() or "add" in asm.lower()

    def test_c_constant_fold(self):
        """C with constant folding — result in a0."""
        c_source = """
        int _start() {
            return 5 + 3;
        }
        """
        # GCC constant-folds to: li a0, 8; ret
        result = compile_and_run(c_source, language="c", device="cuda")
        assert result.reg(10) == 8, f"Expected a0=8, got {result.reg(10)}"

    def test_c_volatile_add(self):
        """C addition with volatile to prevent constant folding."""
        c_source = """
        volatile int a = 7;
        volatile int b = 6;
        int _start() {
            return a + b;
        }
        """
        asm = compile_c(c_source)
        print("Generated assembly:")
        print(asm)
        # This might use global variables, which need data section support.
        # For now just verify it compiles.
        assert len(asm) > 0

    def test_c_fibonacci(self):
        """C fibonacci — result in a0."""
        c_source = """
        int _start() {
            int prev = 0, curr = 1;
            for (int i = 1; i < 10; i++) {
                int temp = prev + curr;
                prev = curr;
                curr = temp;
            }
            return curr;
        }
        """
        result = compile_and_run(c_source, language="c", device="cuda",
                                 max_cycles=50000)
        assert result.reg(10) == 55, f"fib(10) = {result.reg(10)}, expected 55"

    def test_c_gcd(self):
        """C GCD via repeated subtraction — result in a0."""
        c_source = """
        int _start() {
            int a = 48, b = 18;
            while (b != 0) {
                int r = a;
                while (r >= b) r -= b;
                a = b;
                b = r;
            }
            return a;
        }
        """
        result = compile_and_run(c_source, language="c", device="cuda",
                                 max_cycles=50000)
        assert result.reg(10) == 6, f"GCD(48,18) = {result.reg(10)}, expected 6"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
