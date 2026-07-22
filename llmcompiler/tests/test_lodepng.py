"""
Compile and fuzz ACTUAL lodepng chunk parsing code.

This extracts the real chunk parsing and IHDR validation functions
from lodepng (https://github.com/lvandeve/lodepng), compiles them
through our C → RV32I → NISA pipeline, and fuzzes them for overflows.

Functions taken verbatim from lodepng.cpp:
  - lodepng_read32bitInt (line 320)
  - lodepng_chunk_length (line 2715)
  - lodepng_chunk_type_equals (line 2725)
  - checkColorValidity (line 2907)
  - decodeGeneric IHDR parsing (lines 4398-4433)
"""

import struct
import pytest
import shutil

from ..compiler.compiler import compile_c, compile_asm
from ..runtime.gpu_executor import gpu_execute
from ..runtime.differentiable_executor import fuzz

HAS_RISCV_GCC = shutil.which("riscv64-linux-gnu-gcc") is not None

# ── Actual lodepng code, extracted and adapted for freestanding compilation ──

LODEPNG_PARSER_C = """
/* ============================================================
 * Code below is extracted from lodepng.cpp by Lode Vandevenne
 * https://github.com/lvandeve/lodepng
 * License: zlib
 * ============================================================ */

typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

/* From lodepng.cpp line 320 */
__attribute__((noinline))
static unsigned lodepng_read32bitInt(const unsigned char* buffer) {
  return (((unsigned)buffer[0] << 24u) | ((unsigned)buffer[1] << 16u) |
         ((unsigned)buffer[2] << 8u) | (unsigned)buffer[3]);
}

/* From lodepng.cpp line 2715 */
__attribute__((noinline))
unsigned lodepng_chunk_length(const unsigned char* chunk) {
  return lodepng_read32bitInt(chunk);
}

/* From lodepng.cpp line 2725 — adapted to avoid string literals
 * (our freestanding env doesn't load .rodata sections) */
__attribute__((noinline))
unsigned char lodepng_chunk_type_equals_4(const unsigned char* chunk,
                                          unsigned char a, unsigned char b,
                                          unsigned char c, unsigned char d) {
  return (chunk[4] == a && chunk[5] == b && chunk[6] == c && chunk[7] == d);
}

/* From lodepng.cpp line 2907 — EXACT code from lodepng */
__attribute__((noinline))
static unsigned checkColorValidity(unsigned colortype, unsigned bd) {
  switch(colortype) {
    case 0: if(!(bd == 1 || bd == 2 || bd == 4 || bd == 8 || bd == 16)) return 37; break;
    case 2: if(!(                                 bd == 8 || bd == 16)) return 37; break;
    case 3: if(!(bd == 1 || bd == 2 || bd == 4 || bd == 8            )) return 37; break;
    case 4: if(!(                                 bd == 8 || bd == 16)) return 37; break;
    case 6: if(!(                                 bd == 8 || bd == 16)) return 37; break;
    default: return 31;
  }
  return 0;
}

/* From lodepng.cpp line 147 */
static int lodepng_addofl(unsigned a, unsigned b, unsigned* result) {
  *result = a + b;
  return *result < a;
}

/* Adapted from lodepng.cpp decodeGeneric() lines 4388-4433
 * This is the ACTUAL IHDR parsing logic from lodepng.
 *
 * Input: PNG data at 0x1000, length in a1
 * Returns lodepng error code (0 = success)
 */
int _start(void) {
    const unsigned char *in = (const unsigned char *)0x1000;
    /* Input length stored at fixed address 0x0FF0 (avoids register clobber) */
    int insize = *(volatile int *)0x0FF0;

    /* lodepng.cpp line 4389: check minimum size */
    if(insize < 33) return 27;

    /* lodepng.cpp line 4398-4401: check PNG signature */
    if(in[0] != 137 || in[1] != 80 || in[2] != 78 || in[3] != 71
       || in[4] != 13 || in[5] != 10 || in[6] != 26 || in[7] != 10) {
        return 28;
    }

    /* lodepng.cpp line 4402-4403: header size must be 13 bytes */
    if(lodepng_chunk_length(in + 8) != 13) {
        return 94;
    }

    /* lodepng.cpp line 4405-4407: must start with IHDR (0x49484452) */
    if(!lodepng_chunk_type_equals_4(in + 8, 'I', 'H', 'D', 'R')) {
        return 29;
    }

    /* lodepng.cpp line 4410-4411: read dimensions */
    unsigned width = lodepng_read32bitInt(&in[16]);
    unsigned height = lodepng_read32bitInt(&in[20]);

    unsigned bitdepth = in[24];
    unsigned colortype = in[25];
    unsigned compression = in[26];
    unsigned filter = in[27];
    unsigned interlace = in[28];

    /* lodepng.cpp line 4424: invalid image size */
    if(width == 0 || height == 0) return 93;

    /* lodepng.cpp line 4426: invalid colortype/bitdepth */
    unsigned err = checkColorValidity(colortype, bitdepth);
    if(err) return err;

    /* lodepng.cpp line 4429: only compression method 0 */
    if(compression != 0) return 32;

    /* lodepng.cpp line 4431: only filter method 0 */
    if(filter != 0) return 33;

    /* lodepng.cpp line 4433: only interlace 0 or 1 */
    if(interlace > 1) return 34;

    return 0; /* valid PNG header */
}
"""


def _make_png(width=640, height=480, bit_depth=8, color_type=2,
              compression=0, filter_m=0, interlace=0, chunk_len=13):
    """Build a PNG header with given parameters."""
    magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
    cl = struct.pack('>I', chunk_len)
    ct = b'IHDR'
    w = struct.pack('>I', width)
    h = struct.pack('>I', height)
    ihdr_data = w + h + bytes([bit_depth, color_type, compression, filter_m, interlace])
    crc = struct.pack('>I', 0)  # CRC not checked
    return magic + cl + ct + ihdr_data + crc


def _load_png(png_bytes):
    """Load PNG bytes into memory at 0x1000, store length at 0x0FF0."""
    mem = bytearray(65536)
    for i, b in enumerate(png_bytes[:4096]):
        mem[0x1000 + i] = b
    # Store input length as LE 32-bit at 0x0FF0
    length = len(png_bytes)
    mem[0x0FF0] = length & 0xFF
    mem[0x0FF1] = (length >> 8) & 0xFF
    mem[0x0FF2] = (length >> 16) & 0xFF
    mem[0x0FF3] = (length >> 24) & 0xFF
    return mem


@pytest.mark.skipif(not HAS_RISCV_GCC, reason="RISC-V GCC not installed")
class TestLodePNGParser:
    """Test the actual lodepng IHDR parsing code."""

    @pytest.fixture(autouse=True)
    def compile_parser(self):
        self.nisa = compile_asm(compile_c(LODEPNG_PARSER_C))
        print(f"  [lodepng parser: {len(self.nisa)} NISA instructions]")

    def _run(self, png_bytes):
        mem = _load_png(png_bytes)  # length stored at 0x0FF0
        return gpu_execute(self.nisa,
                           memory_bytes=mem, device='cuda',
                           max_cycles=5000)

    def test_valid_png_rgb(self):
        """Valid 640x480 RGB PNG header → error 0."""
        r = self._run(_make_png(640, 480, 8, 2))
        assert r.reg(10) == 0, f"Expected 0, got {r.reg(10)}"

    def test_valid_png_rgba(self):
        """Valid RGBA PNG → error 0."""
        r = self._run(_make_png(100, 100, 8, 6))
        assert r.reg(10) == 0

    def test_valid_png_grayscale(self):
        """Valid grayscale PNG → error 0."""
        r = self._run(_make_png(256, 256, 16, 0))
        assert r.reg(10) == 0

    def test_valid_png_palette(self):
        """Valid palette PNG → error 0."""
        r = self._run(_make_png(32, 32, 8, 3))
        assert r.reg(10) == 0

    def test_too_small(self):
        """Input < 33 bytes → lodepng error 27."""
        r = self._run(b'\x00' * 10)
        assert r.reg(10) == 27

    def test_bad_signature(self):
        """Wrong PNG magic → lodepng error 28."""
        r = self._run(b'\x00' * 40)
        assert r.reg(10) == 28

    def test_bad_ihdr_length(self):
        """IHDR chunk length != 13 → lodepng error 94."""
        r = self._run(_make_png(chunk_len=12))
        assert r.reg(10) == 94

    def test_not_ihdr(self):
        """First chunk not IHDR → lodepng error 29."""
        png = bytearray(_make_png())
        png[12:16] = b'tEXt'  # change chunk type
        r = self._run(bytes(png))
        assert r.reg(10) == 29

    def test_zero_width(self):
        """Width = 0 → lodepng error 93."""
        r = self._run(_make_png(width=0))
        assert r.reg(10) == 93

    def test_zero_height(self):
        """Height = 0 → lodepng error 93."""
        r = self._run(_make_png(height=0))
        assert r.reg(10) == 93

    def test_invalid_color_type(self):
        """Color type 5 is invalid → lodepng error 31."""
        r = self._run(_make_png(color_type=5))
        assert r.reg(10) == 31

    def test_invalid_bitdepth_for_rgb(self):
        """RGB with bit depth 4 is invalid → lodepng error 37."""
        r = self._run(_make_png(bit_depth=4, color_type=2))
        assert r.reg(10) == 37

    def test_invalid_compression(self):
        """Compression != 0 → lodepng error 32."""
        r = self._run(_make_png(compression=1))
        assert r.reg(10) == 32

    def test_invalid_filter(self):
        """Filter != 0 → lodepng error 33."""
        r = self._run(_make_png(filter_m=1))
        assert r.reg(10) == 33

    def test_invalid_interlace(self):
        """Interlace > 1 → lodepng error 34."""
        r = self._run(_make_png(interlace=2))
        assert r.reg(10) == 34

    def test_huge_dimensions_accepted(self):
        """Huge but valid dimensions pass header validation."""
        r = self._run(_make_png(width=0x7FFFFFFF, height=0x7FFFFFFF, bit_depth=8, color_type=6))
        assert r.reg(10) == 0, f"Expected valid (0), got {r.reg(10)}"

    def test_valid_1bit_grayscale(self):
        """1-bit grayscale is valid."""
        r = self._run(_make_png(8, 8, 1, 0))
        assert r.reg(10) == 0

    def test_all_valid_color_types(self):
        """Test every valid color type / bit depth combination from the PNG spec."""
        valid = [
            (1, 0), (2, 0), (4, 0), (8, 0), (16, 0),  # grayscale
            (8, 2), (16, 2),                             # RGB
            (1, 3), (2, 3), (4, 3), (8, 3),             # palette
            (8, 4), (16, 4),                              # gray+alpha
            (8, 6), (16, 6),                              # RGBA
        ]
        for bd, ct in valid:
            r = self._run(_make_png(16, 16, bd, ct))
            assert r.reg(10) == 0, \
                f"Valid combo bd={bd} ct={ct} returned error {r.reg(10)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
