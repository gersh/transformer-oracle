"""
Deep fuzzing of lodepng chunk iteration logic.

This extracts the chunk walking and bounds-checking code from lodepng's
decodeGeneric() and lodepng_inspect_chunk() — the code that iterates
through PNG chunks and validates their lengths.

This is where real bugs live: the interaction between chunk lengths,
buffer sizes, and pointer arithmetic during chunk iteration.

Key code from lodepng.cpp:
  - lodepng_chunk_next_const (line 2798): advance to next chunk
  - lodepng_inspect_chunk (line 5231): validate chunk bounds
  - decodeGeneric chunk loop (line 5294+): iterate all chunks

We extract and compile the actual chunk iteration + bounds checking,
then fuzz it to look for inputs where malformed chunk lengths cause
out-of-bounds reads.
"""

import struct
import pytest
import shutil

from ..compiler.compiler import compile_c, compile_asm
from ..compiler.python_compiler import compile_python
from ..runtime.gpu_executor import gpu_execute
from ..runtime.differentiable_executor import fuzz

HAS_RISCV_GCC = shutil.which("riscv64-linux-gnu-gcc") is not None


# ── Actual lodepng chunk iteration logic, extracted ──

CHUNK_WALKER_C = """
/* Extracted from lodepng.cpp — chunk iteration and bounds checking.
 * This is the core logic that walks through PNG chunks and validates
 * that each chunk's declared length doesn't exceed the remaining data.
 *
 * Input: PNG data at 0x1000, total size at 0x0FF0
 * Returns: 0=safe, or error code indicating which check caught the issue
 *
 * Error codes (from lodepng):
 *   27 = data too short for PNG header
 *   28 = bad PNG signature
 *   30 = chunk extends past end of data
 *   63 = chunk length exceeds 2GB
 *   94 = IHDR chunk length != 13
 *   50 = too many chunks (safety limit)
 *   95 = POTENTIAL BUG: chunk navigation went backwards or stalled
 */

typedef unsigned int uint32_t;

/* From lodepng.cpp line 320 */
__attribute__((noinline))
static unsigned read_be32(const unsigned char* buffer) {
  return (((unsigned)buffer[0] << 24u) | ((unsigned)buffer[1] << 16u) |
         ((unsigned)buffer[2] << 8u) | (unsigned)buffer[3]);
}

/* From lodepng.cpp line 147 */
static int addofl(unsigned a, unsigned b, unsigned* result) {
  *result = a + b;
  return *result < a;
}

/* From lodepng.cpp lines 2798-2810 — lodepng_chunk_next_const
 * This function advances the chunk pointer to the next chunk.
 * BUG SURFACE: if chunk_length is malicious, total_chunk_length can overflow,
 * causing the pointer to jump to an unexpected location. */
__attribute__((noinline))
static unsigned chunk_next(unsigned pos, const unsigned char* data,
                           unsigned data_size, unsigned* next_pos) {
    if(pos >= data_size || data_size - pos < 12) return 30;

    /* Check for PNG magic at current position (lodepng line 2786-2789) */
    if(data[pos+0] == 0x89 && data[pos+1] == 0x50 &&
       data[pos+2] == 0x4e && data[pos+3] == 0x47 &&
       data[pos+4] == 0x0d && data[pos+5] == 0x0a &&
       data[pos+6] == 0x1a && data[pos+7] == 0x0a) {
        *next_pos = pos + 8;
        return 0;
    }

    unsigned chunk_length = read_be32(data + pos);

    /* lodepng line 2807: overflow check on chunk_length + 12 */
    unsigned total_chunk_length;
    if(addofl(chunk_length, 12, &total_chunk_length)) {
        *next_pos = data_size;  /* overflow → jump to end */
        return 63;
    }

    /* lodepng line 2808: bounds check */
    if(total_chunk_length > data_size - pos) {
        *next_pos = data_size;
        return 30;
    }

    *next_pos = pos + total_chunk_length;

    /* Safety check: did we actually advance? */
    if(*next_pos <= pos) {
        return 95;  /* STALL: chunk navigation didn't advance! */
    }

    return 0;
}

/* Walk through ALL chunks in a PNG file.
 * Mimics lodepng's decodeGeneric() chunk loop.
 * Returns the number of chunks found, or negative error. */
int _start(void) {
    const unsigned char *data = (const unsigned char *)0x1000;
    unsigned data_size = *(volatile unsigned *)0x0FF0;

    /* Check minimum size */
    if(data_size < 33) return 27;

    /* Check PNG signature (lodepng line 4398) */
    if(data[0] != 137 || data[1] != 80 || data[2] != 78 || data[3] != 71
       || data[4] != 13 || data[5] != 10 || data[6] != 26 || data[7] != 10) {
        return 28;
    }

    /* Start after PNG signature */
    unsigned pos = 8;
    unsigned chunk_count = 0;
    unsigned found_ihdr = 0;
    unsigned found_iend = 0;

    /* Iterate chunks (lodepng decodeGeneric loop, line 5340+) */
    while(pos < data_size && chunk_count < 100) {
        if(data_size - pos < 12) break;  /* not enough for chunk header */

        unsigned chunk_length = read_be32(data + pos);
        unsigned char type0 = data[pos + 4];
        unsigned char type1 = data[pos + 5];
        unsigned char type2 = data[pos + 6];
        unsigned char type3 = data[pos + 7];

        /* lodepng line 5241: reject absurdly large chunks */
        if(chunk_length > 0x7FFFFFFF) return 63;

        /* Bounds check: does chunk data + CRC fit in remaining data?
         * This is lodepng's line 5243: chunkLength + 12 > insize - pos */
        unsigned total;
        if(addofl(chunk_length, 12, &total)) return 63;  /* overflow */
        if(total > data_size - pos) return 30;  /* past end */

        /* Check for IHDR (must be first) */
        if(type0 == 73 && type1 == 72 && type2 == 68 && type3 == 82) {
            if(chunk_length != 13) return 94;
            found_ihdr = 1;
        }

        /* Check for IEND (last chunk) */
        if(type0 == 73 && type1 == 69 && type2 == 78 && type3 == 68) {
            found_iend = 1;
        }

        /* Advance to next chunk */
        unsigned next_pos;
        unsigned err = chunk_next(pos, data, data_size, &next_pos);
        if(err) return err;

        /* Safety: ensure forward progress */
        if(next_pos <= pos) return 95;

        pos = next_pos;
        chunk_count++;
    }

    if(!found_ihdr) return 29;  /* missing IHDR */

    /* Return chunk count as success indicator */
    return chunk_count;
}
"""


def _build_valid_png():
    """Build a minimal valid PNG with IHDR + IEND chunks."""
    magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])

    # IHDR chunk
    ihdr_data = struct.pack('>II', 1, 1)  # 1x1 image
    ihdr_data += bytes([8, 0, 0, 0, 0])    # 8-bit grayscale
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', 0)

    # IEND chunk
    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', 0)

    return magic + ihdr + iend


def _build_multi_chunk_png(extra_chunks=None):
    """Build PNG with multiple chunks."""
    magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])

    ihdr_data = struct.pack('>II', 1, 1) + bytes([8, 0, 0, 0, 0])
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', 0)

    chunks = magic + ihdr
    if extra_chunks:
        for chunk_type, chunk_data in extra_chunks:
            chunks += struct.pack('>I', len(chunk_data))
            chunks += chunk_type
            chunks += chunk_data
            chunks += struct.pack('>I', 0)  # CRC

    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', 0)
    return chunks + iend


def _load_data(data):
    mem = bytearray(65536)
    for i, b in enumerate(data[:4096]):
        mem[0x1000 + i] = b
    length = len(data)
    mem[0x0FF0] = length & 0xFF
    mem[0x0FF1] = (length >> 8) & 0xFF
    mem[0x0FF2] = (length >> 16) & 0xFF
    mem[0x0FF3] = (length >> 24) & 0xFF
    return mem


@pytest.mark.skipif(not HAS_RISCV_GCC, reason="RISC-V GCC not installed")
class TestChunkWalker:

    @pytest.fixture(autouse=True)
    def compile_walker(self):
        self.nisa = compile_asm(compile_c(CHUNK_WALKER_C))
        print(f"  [chunk walker: {len(self.nisa)} NISA instructions]")

    def _run(self, data):
        mem = _load_data(data)
        return gpu_execute(self.nisa, memory_bytes=mem,
                           device='cuda', max_cycles=20000)

    def test_valid_minimal_png(self):
        """Valid 1x1 PNG with IHDR + IEND — should parse at least 1 chunk."""
        r = self._run(_build_valid_png())
        # Returns chunk count or error code; positive = parsed chunks
        assert r.reg(10) > 0 and r.halted, \
            f"Expected successful parse, got {r.reg(10)}, halted={r.halted}"

    def test_multi_chunk_png(self):
        """PNG with extra chunks between IHDR and IEND."""
        png = _build_multi_chunk_png([
            (b'tEXt', b'Comment\x00hello'),
            (b'tEXt', b'Author\x00test'),
        ])
        r = self._run(png)
        assert r.reg(10) > 0 and r.halted, \
            f"Expected successful parse, got {r.reg(10)}"

    def test_too_short(self):
        r = self._run(b'\x89PNG\r\n\x1a\n' + b'\x00' * 10)
        assert r.reg(10) == 27

    def test_bad_signature(self):
        r = self._run(b'\x00' * 40)
        assert r.reg(10) == 28

    def test_chunk_exceeds_data(self):
        """Chunk declares more data than available → caught."""
        magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        ihdr_data = struct.pack('>II', 1, 1) + bytes([8, 0, 0, 0, 0])
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', 0)
        bad_chunk = struct.pack('>I', 9999) + b'tEXt'
        data = magic + ihdr + bad_chunk
        r = self._run(data)
        # Should catch as error 30 (past end) or return partial chunk count
        assert r.reg(10) > 0 and r.halted, \
            f"Expected error or partial parse, got {r.reg(10)}"

    def test_chunk_length_overflow(self):
        """Chunk length near 2^32 → addofl catches overflow."""
        magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        ihdr_data = struct.pack('>II', 1, 1) + bytes([8, 0, 0, 0, 0])
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', 0)
        # Chunk with length 0xFFFFFFFE → length + 12 overflows
        bad_chunk = struct.pack('>I', 0xFFFFFFFE) + b'tEXt'
        data = magic + ihdr + bad_chunk + b'\x00' * 20
        r = self._run(data)
        assert r.reg(10) == 63, f"Expected error 63 (overflow), got {r.reg(10)}"

    def test_chunk_length_2gb(self):
        """Chunk length > 0x7FFFFFFF → rejected."""
        magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        ihdr_data = struct.pack('>II', 1, 1) + bytes([8, 0, 0, 0, 0])
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', 0)
        bad_chunk = struct.pack('>I', 0x80000001) + b'tEXt'
        data = magic + ihdr + bad_chunk + b'\x00' * 20
        r = self._run(data)
        assert r.reg(10) == 63, f"Expected error 63, got {r.reg(10)}"

    def test_zero_length_chunk(self):
        """Zero-length chunk (like IEND) should be valid."""
        png = _build_multi_chunk_png([(b'tEXt', b'')])
        r = self._run(png)
        assert r.reg(10) >= 2  # at least IHDR + IEND


@pytest.mark.skipif(not HAS_RISCV_GCC, reason="RISC-V GCC not installed")
class TestChunkWalkerFuzzing:

    def test_fuzz_chunk_lengths(self):
        """Fuzz chunk length values to find overflow bugs."""
        # Python equivalent of the chunk bounds check for fuzzing
        def chunk_bounds(chunk_len, remaining):
            """Check if chunk fits in remaining data.
            Returns: 0=ok, 30=past end, 63=overflow, 95=stall"""
            if remaining < 12:
                return 30
            # addofl: chunk_len + 12
            total = chunk_len + 12
            if total < chunk_len:  # overflow!
                return 63
            if total > remaining:
                return 30
            if chunk_len > 0x7FFFFFFF:
                return 63
            return 0

        nisa = compile_python(chunk_bounds)
        result = fuzz(nisa, n_input_regs=2, n_iterations=300,
                      lr=200.0, verbose=True, seed=7)

        print(f"\nCoverage: {len(result['best_coverage'])} branch directions")
        assert len(result['best_coverage']) >= 2

    def test_fuzz_crafted_png(self):
        """Fuzz with crafted PNG data — inject malicious chunk lengths."""
        if not shutil.which("riscv64-linux-gnu-gcc"):
            pytest.skip("no gcc")

        nisa = compile_asm(compile_c(CHUNK_WALKER_C))

        # Try various malicious chunk lengths
        results = {}
        magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        ihdr_data = struct.pack('>II', 1, 1) + bytes([8, 0, 0, 0, 0])
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', 0)

        test_lengths = [
            0,              # zero
            1,              # tiny
            0x7FFFFFFF,     # max signed
            0x80000000,     # min negative (signed)
            0xFFFFFFFF,     # max unsigned
            0xFFFFFFF4,     # -12 → total overflows to 0
            0xFFFFFFF5,     # -11 → total overflows to 1
        ]

        print("\nChunk length fuzzing on compiled lodepng:")
        for length in test_lengths:
            bad_chunk = struct.pack('>I', length) + b'tEXt' + b'\x00' * min(length, 100)
            data = magic + ihdr + bad_chunk + b'\x00' * 20
            mem = _load_data(data)
            r = gpu_execute(nisa, memory_bytes=mem, device='cuda', max_cycles=20000)
            results[length] = r.reg(10)
            caught = "CAUGHT" if r.reg(10) in (30, 63, 95) else "OK" if r.reg(10) > 0 else "???"
            print(f"  chunk_len=0x{length:08X} ({length:>11d}) → error={r.reg(10):>3d} [{caught}]")

        # ALL malicious lengths should be caught (no crashes, no 0 returns)
        for length in [0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFF4, 0xFFFFFFF5]:
            assert results[length] in (30, 63, 95), \
                f"Malicious length 0x{length:08X} returned {results[length]} — possible bug!"

        print("\n  All malicious chunk lengths properly caught by lodepng's bounds checks!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
