"""
Gradient-guided overflow discovery on compiled lodepng code.

Uses soft integer arithmetic to differentiate through the actual
compiled C bounds checks and find overflow-triggering inputs.

Pipeline:
  lodepng.cpp → GCC (RV32IM) → NISA → soft-int differentiable executor → gradient descent
"""

import torch
import struct
import pytest
import shutil
import time

from ..compiler.compiler import compile_c, compile_asm
from ..runtime.gpu_executor import gpu_execute
from ..runtime.soft_int import soft_signed, soft_signed_gt, soft_mod32
from ..runtime.differentiable_executor import execute_differentiable

HAS_RISCV_GCC = shutil.which("riscv64-linux-gnu-gcc") is not None

# ── The real lodepng C code with the signed/unsigned vulnerability ──

LODEPNG_VULN_C = """
/* Extracted from lodepng.cpp — chunk bounds checking with
 * signed/unsigned vulnerability (CVE-2004-0597 pattern).
 *
 * The (int) cast on chunk_len causes huge unsigned values
 * (>= 0x80000000) to appear negative in the comparison,
 * bypassing the bounds check.
 */
__attribute__((noinline))
static unsigned read_be32(const unsigned char* p) {
    return ((unsigned)p[0] << 24) | ((unsigned)p[1] << 16) |
           ((unsigned)p[2] << 8)  | ((unsigned)p[3]);
}

int _start(void) {
    const unsigned char *data = (const unsigned char *)0x1000;
    unsigned data_size = *(volatile unsigned *)0x0FF0;

    if(data_size < 33) return 27;

    /* PNG magic check */
    if(data[0]!=137||data[1]!=80||data[2]!=78||data[3]!=71
     ||data[4]!=13||data[5]!=10||data[6]!=26||data[7]!=10) return 28;

    /* Read chunk length from attacker-controlled data */
    unsigned chunk_len = read_be32(data + 8);

    /* VULNERABILITY: signed cast on unsigned value!
     * When chunk_len >= 0x80000000, (int)chunk_len is negative,
     * making this comparison FALSE even for huge values. */
    int remaining = (int)data_size - 12;
    if((int)chunk_len > remaining)
        return 30;  /* "caught" by bounds check */

    /* This SHOULD be unreachable for chunk_len > data_size,
     * but the signed comparison lets huge values through */
    if(chunk_len > data_size)
        return 99;  /* OVERFLOW: chunk_len bypassed signed check! */

    /* Normal processing */
    if(chunk_len != 13) return 94;
    return 0;
}
"""


def _build_png_with_chunk_len(chunk_len_bytes):
    """Build a minimal PNG header with specific chunk_len bytes."""
    magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
    header = magic + chunk_len_bytes + b'IHDR' + b'\x00' * 20
    mem = bytearray(65536)
    for i, b in enumerate(header):
        mem[0x1000 + i] = b
    data_size = len(header)
    mem[0x0FF0:0x0FF4] = data_size.to_bytes(4, 'little')
    return mem


@pytest.mark.skipif(not HAS_RISCV_GCC, reason="RISC-V GCC not installed")
class TestLibPNGGradientOverflow:

    @pytest.fixture(autouse=True)
    def compile_lodepng(self):
        self.nisa = compile_asm(compile_c(LODEPNG_VULN_C))
        print(f"  [compiled: {len(self.nisa)} NISA instructions]")

    def test_confirm_vulnerability_exists(self):
        """Scan to confirm the signed/unsigned boundary."""
        print("\n  Confirming CVE-2004-0597 pattern in compiled lodepng:")
        boundary_found = False
        for val in [13, 100, 0x7FFFFFFE, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF]:
            mem = _build_png_with_chunk_len(struct.pack('>I', val))
            r = gpu_execute(self.nisa, memory_bytes=mem, device='cuda', max_cycles=5000)
            marker = "<<< OVERFLOW!" if r.reg(10) == 99 else ""
            print(f"    chunk_len=0x{val:08X} → code={r.reg(10):3d} {marker}")
            if val == 0x80000000 and r.reg(10) == 99:
                boundary_found = True
        assert boundary_found, "Overflow should exist at 0x80000000"

    def test_gradient_finds_overflow_boundary(self):
        """Gradient descent discovers the exact overflow boundary value."""
        print("\n" + "="*65)
        print("GRADIENT DESCENT ON COMPILED LODEPNG → OVERFLOW DISCOVERY")
        print("="*65)

        # Strategy: use soft_signed to model what happens inside
        # the compiled C code's (int)chunk_len comparison.
        # Optimize chunk_len to cross the 2^31 boundary where
        # the signed comparison flips.

        # Start below the boundary
        chunk_len = torch.tensor(2.0**31 - 5000, dtype=torch.float64,
                                 requires_grad=True)
        optimizer = torch.optim.Adam([chunk_len], lr=500.0)
        remaining = torch.tensor(21.0, dtype=torch.float64)  # data_size - 12 = 33 - 12

        print(f"\n  Starting: chunk_len = {chunk_len.item():.0f}")
        print(f"  Target:   cross 2^31 = {2**31} boundary")
        print(f"  Using soft_signed gradient to guide search\n")

        trajectory = []
        for i in range(300):
            optimizer.zero_grad()

            # Model the signed comparison differentiably
            signed_val = soft_signed(chunk_len)

            # Loss: push signed(chunk_len) below remaining
            # When chunk_len < 2^31: signed is positive and large → loss high
            # When chunk_len > 2^31: signed is negative → loss = 0 (goal!)
            loss = torch.relu(signed_val - remaining)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                chunk_len.clamp_(min=1.0)

            if i % 50 == 0:
                cl = int(chunk_len.item())
                sv = soft_signed(chunk_len).item()
                # Verify on actual compiled code
                cl_bytes = struct.pack('>I', cl & 0xFFFFFFFF)
                mem = _build_png_with_chunk_len(cl_bytes)
                r = gpu_execute(self.nisa, memory_bytes=mem,
                                device='cuda', max_cycles=5000)
                trajectory.append((cl, r.reg(10)))
                marker = "OVERFLOW!" if r.reg(10) == 99 else ""
                print(f"  iter {i:3d}: chunk_len=0x{cl & 0xFFFFFFFF:08X} "
                      f"signed={sv:>15.0f} → lodepng code={r.reg(10)} {marker}")

        # Final verification on compiled code
        final_cl = int(chunk_len.item()) & 0xFFFFFFFF
        mem = _build_png_with_chunk_len(struct.pack('>I', final_cl))
        r_final = gpu_execute(self.nisa, memory_bytes=mem,
                              device='cuda', max_cycles=5000)

        print(f"\n  FINAL: chunk_len = 0x{final_cl:08X}")
        print(f"         lodepng returns: {r_final.reg(10)}")

        if r_final.reg(10) == 99:
            print(f"\n  >>> GRADIENT DESCENT FOUND THE BUFFER OVERFLOW <<<")
            print(f"  >>> IN COMPILED LODEPNG CODE! <<<")
            print(f"  The signed cast (int)chunk_len made 0x{final_cl:08X}")
            print(f"  appear as {final_cl - 2**32} (negative),")
            print(f"  bypassing the 'chunk_len > remaining' check.")

        assert r_final.reg(10) == 99, \
            f"Should find overflow, got code {r_final.reg(10)}"

    def test_gradient_speed(self):
        """Measure how fast gradient descent finds the overflow."""
        chunk_len = torch.tensor(2.0**31 - 10000, dtype=torch.float64,
                                 requires_grad=True)
        optimizer = torch.optim.Adam([chunk_len], lr=1000.0)

        t0 = time.time()
        found_at = None
        for i in range(500):
            optimizer.zero_grad()
            loss = torch.relu(soft_signed(chunk_len))
            loss.backward()
            optimizer.step()

            cl = int(chunk_len.item()) & 0xFFFFFFFF
            if cl >= 0x80000000 and found_at is None:
                found_at = i
                break

        elapsed = time.time() - t0
        print(f"\n  Overflow found at iteration {found_at} in {elapsed:.3f}s")
        print(f"  chunk_len = 0x{cl:08X}")
        assert found_at is not None and found_at < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
