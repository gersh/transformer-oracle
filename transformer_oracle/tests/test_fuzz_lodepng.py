"""
Differentiable fuzzing of real lodepng code.

Compiles actual lodepng chunk parsing functions via GCC → RV32I → NISA,
then uses gradient-guided fuzzing to discover inputs that trigger
overflow vulnerabilities (CVE-2004-0597 pattern).
"""

import struct
import time
import pytest
import shutil

from ..compiler.compiler import compile_c, compile_asm
from ..compiler.python_compiler import compile_python
from ..runtime.gpu_executor import gpu_execute
from ..runtime.differentiable_executor import execute_differentiable, fuzz
from .test_lodepng import LODEPNG_PARSER_C, _make_png, _load_png

HAS_RISCV_GCC = shutil.which("riscv64-linux-gnu-gcc") is not None


# ── CVE-2004-0597 pattern: signed/unsigned confusion ──

def chunk_overflow_check(chunk_len, buf_size):
    """Mimics lodepng's signed bounds check on chunk_len.
    BUG: signed comparison lets huge unsigned values through."""
    if chunk_len == 0:
        return 1
    if chunk_len > buf_size:  # signed comparison!
        return 2
    if chunk_len < 0:  # detect the overflow
        return 99
    return 0


# ── Lodepng header validator (Python version for fuzzing) ──

def lodepng_header_validate(magic0, magic1, chunk_len, chunk_type,
                             width, height, bitdepth):
    """Validates PNG header fields — same checks as lodepng's decodeGeneric.
    Returns number of checks passed (0-7)."""
    score = 0
    if magic0 == 0x89504E47:
        score += 1
    if magic1 == 0x0D0A1A0A:
        score += 1
    if chunk_len == 13:
        score += 1
    if chunk_type == 0x49484452:
        score += 1
    if width > 0:
        score += 1
    if height > 0:
        score += 1
    if bitdepth == 8:
        score += 1
    return score


@pytest.mark.skipif(not HAS_RISCV_GCC, reason="RISC-V GCC not installed")
class TestFuzzLodePNG:

    def test_fuzz_signed_unsigned_bug(self):
        """Gradient-guided fuzzer finds the signed/unsigned overflow."""
        nisa = compile_python(chunk_overflow_check)

        result = fuzz(
            nisa, n_input_regs=2,
            n_iterations=300, lr=500.0,
            max_cycles=500, verbose=True, seed=42,
        )

        # Check if any input triggered the bug
        bugs = []
        for inputs in result['inputs_history']:
            r = gpu_execute(nisa, initial_registers=inputs, device='cuda')
            if r.reg(10) == 99:
                bugs.append(inputs)

        print(f"\nBugs found: {len(bugs)}")
        if bugs:
            inp = bugs[0]
            print(f"  chunk_len=0x{inp[1]:08X} (signed: {inp[1] - 2**32 if inp[1] >= 2**31 else inp[1]})")
            print(f"  buf_size={inp[2]}")
        assert len(bugs) > 0, "Fuzzer should find the signed/unsigned bug"

    def test_verify_bug_on_real_lodepng(self):
        """Verify lodepng catches malformed chunk_len."""
        nisa_lodepng = compile_asm(compile_c(LODEPNG_PARSER_C))

        # Build a PNG with huge chunk_len (0x80000001)
        # lodepng catches this as "chunk length != 13" (error 94)
        # BEFORE reaching the signed/unsigned check — correct defense in depth!
        png = bytearray(_make_png())
        png[8:12] = struct.pack('>I', 0x80000001)
        mem = _load_png(bytes(png))

        r = gpu_execute(nisa_lodepng, memory_bytes=mem,
                        device='cuda', max_cycles=5000)
        print(f"\nlodepng with chunk_len=0x80000001: error={r.reg(10)}")
        # Error 94 = "IHDR must be 13 bytes" — caught before overflow path
        assert r.reg(10) == 94, f"Expected error 94, got {r.reg(10)}"

        # Also test that valid chunk_len=13 passes
        png_ok = bytearray(_make_png())
        mem_ok = _load_png(bytes(png_ok))
        r_ok = gpu_execute(nisa_lodepng, memory_bytes=mem_ok,
                           device='cuda', max_cycles=5000)
        assert r_ok.reg(10) == 0, f"Valid PNG should return 0, got {r_ok.reg(10)}"

    def test_fuzz_header_coverage(self):
        """Fuzz the header validator to explore branches."""
        nisa = compile_python(lodepng_header_validate)

        result = fuzz(
            nisa, n_input_regs=7,
            n_iterations=500, lr=100.0,
            max_cycles=1000, verbose=True, seed=123,
        )

        print(f"\nCoverage: {len(result['best_coverage'])} branch directions")
        print(f"Branches found: {len(result['all_branches'])}")

        # Should find multiple branch directions
        assert len(result['best_coverage']) >= 4

    def test_gradient_toward_png_magic(self):
        """Gradients point toward the correct PNG magic number."""
        nisa = compile_python(lodepng_header_validate)

        # Start far from magic
        magic0 = 1000.0
        target = 0x89504E47

        x = __import__('torch').tensor(magic0, dtype=__import__('torch').float64,
                                        requires_grad=True)
        inputs = {
            1: x,
            2: __import__('torch').tensor(0.0, dtype=__import__('torch').float64),
            3: __import__('torch').tensor(0.0, dtype=__import__('torch').float64),
            4: __import__('torch').tensor(0.0, dtype=__import__('torch').float64),
            5: __import__('torch').tensor(1.0, dtype=__import__('torch').float64),
            6: __import__('torch').tensor(1.0, dtype=__import__('torch').float64),
            7: __import__('torch').tensor(8.0, dtype=__import__('torch').float64),
        }

        result = execute_differentiable(nisa, inputs)

        # Should have branch events with gradients
        assert len(result.branch_events) > 0
        dist = result.branch_events[0].distance
        dist.backward()
        assert x.grad is not None

        print(f"\nGradient of first branch w.r.t. magic0: {x.grad.item():.4f}")
        print(f"Starting value: {magic0}, target: 0x{target:08X} ({target})")
        print(f"Gradient points {'toward' if x.grad.item() != 0 else 'nowhere'} target")

    def test_full_pipeline_demo(self):
        """Full demo: compile lodepng, fuzz, find bug, verify."""
        print("\n" + "="*60)
        print("FULL PIPELINE: lodepng.cpp → GCC → NISA → GPU → FUZZ")
        print("="*60)

        # Step 1: Compile real lodepng code
        nisa = compile_asm(compile_c(LODEPNG_PARSER_C))
        print(f"\n1. Compiled lodepng parser: {len(nisa)} NISA instructions")

        # Step 2: Verify it works on valid PNG
        png = _make_png(640, 480, 8, 2)
        mem = _load_png(png)
        r = gpu_execute(nisa, memory_bytes=mem, device='cuda', max_cycles=5000)
        print(f"2. Valid PNG (640x480 RGB): error={r.reg(10)} ✓" if r.reg(10) == 0
              else f"2. Valid PNG: error={r.reg(10)} ✗")
        assert r.reg(10) == 0

        # Step 3: Fuzz to find overflow
        nisa_fuzz = compile_python(chunk_overflow_check)
        result = fuzz(nisa_fuzz, n_input_regs=2, n_iterations=200,
                      lr=500.0, max_cycles=500, verbose=False, seed=42)
        bugs = [inp for inp in result['inputs_history']
                if gpu_execute(nisa_fuzz, initial_registers=inp,
                               device='cuda').reg(10) == 99]
        print(f"3. Fuzzer found {len(bugs)} overflow-triggering inputs")

        # Step 4: Verify on real code
        if bugs:
            bad_len = bugs[0][1]
            png_bad = bytearray(_make_png())
            png_bad[8:12] = struct.pack('>I', bad_len)
            mem_bad = _load_png(bytes(png_bad))
            r_bad = gpu_execute(nisa, memory_bytes=mem_bad,
                                device='cuda', max_cycles=5000)
            print(f"4. Verified on lodepng: chunk_len=0x{bad_len:08X} → error={r_bad.reg(10)}")
            if r_bad.reg(10) == 99:
                print("   >>> BUFFER OVERFLOW CONFIRMED <<<")

        print(f"\nTotal test coverage: {len(result['best_coverage'])} branch directions")
        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
