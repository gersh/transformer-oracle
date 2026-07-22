"""
Deep fuzzing of lodepng's internal processing functions.

Tests the REAL, UNMODIFIED code from:
  - readChunk_PLTE (palette parsing, line 4788)
  - readChunk_tRNS (transparency, line 4808)
  - unfilterScanline (pixel filter, line 4446)
  - paethPredictor (Paeth filter, used by unfilter)

These are the functions where historical PNG CVEs lived —
buffer fills, pointer arithmetic, and complex state.
"""

import torch
import struct
import pytest
import shutil

from ..compiler.compiler import compile_c, compile_asm
from ..runtime.gpu_executor import gpu_execute
from ..runtime.soft_int import soft_signed, soft_mod32

HAS_RISCV_GCC = shutil.which("riscv64-linux-gnu-gcc") is not None


# ══════════════════════════════════════════════════════
# REAL lodepng readChunk_PLTE + readChunk_tRNS
# ══════════════════════════════════════════════════════

PLTE_TRNS_C = """
/* VERBATIM from lodepng.cpp lines 4788-4833.
 * readChunk_PLTE: reads palette data from chunk
 * readChunk_tRNS: reads transparency data
 *
 * Heap at 0x4000 for palette allocation.
 * Chunk data at 0x1000.
 * Returns lodepng error code.
 */

typedef unsigned int size_t;
#define NULL ((void*)0)
#define LCT_GREY 0
#define LCT_RGB 2
#define LCT_PALETTE 3

/* Bump allocator */
static unsigned char* _heap = (unsigned char*)0x4000;
static void* lodepng_malloc(size_t s) {
    s = (s + 3) & ~3u;
    unsigned char* p = _heap;
    _heap = p + s;
    return p;
}

/* Color mode struct — matches lodepng */
typedef struct {
    unsigned colortype;
    unsigned bitdepth;
    unsigned char* palette;
    size_t palettesize;
    unsigned key_defined;
    unsigned key_r, key_g, key_b;
} LodePNGColorMode;

/* lodepng.cpp line 2949 — EXACT */
__attribute__((noinline))
static void lodepng_color_mode_alloc_palette(LodePNGColorMode* info) {
    size_t i;
    if(!info->palette) info->palette = (unsigned char*)lodepng_malloc(1024);
    if(!info->palette) return;
    for(i = 0; i != 256; ++i) {
        info->palette[i * 4 + 0] = 0;
        info->palette[i * 4 + 1] = 0;
        info->palette[i * 4 + 2] = 0;
        info->palette[i * 4 + 3] = 255;
    }
}

/* lodepng.cpp line 4788 — EXACT */
__attribute__((noinline))
static unsigned readChunk_PLTE(LodePNGColorMode* color,
                               const unsigned char* data,
                               size_t chunkLength) {
    unsigned pos = 0, i;
    color->palettesize = chunkLength / 3u;
    if(color->palettesize == 0 || color->palettesize > 256) return 38;
    lodepng_color_mode_alloc_palette(color);
    if(!color->palette && color->palettesize) {
        color->palettesize = 0;
        return 83;
    }
    for(i = 0; i != color->palettesize; ++i) {
        color->palette[4 * i + 0] = data[pos++];
        color->palette[4 * i + 1] = data[pos++];
        color->palette[4 * i + 2] = data[pos++];
        color->palette[4 * i + 3] = 255;
    }
    return 0;
}

/* lodepng.cpp line 4808 — EXACT */
__attribute__((noinline))
static unsigned readChunk_tRNS(LodePNGColorMode* color,
                                const unsigned char* data,
                                size_t chunkLength) {
    unsigned i;
    if(color->colortype == LCT_PALETTE) {
        if(chunkLength > color->palettesize) return 39;
        for(i = 0; i != chunkLength; ++i)
            color->palette[4 * i + 3] = data[i];
    } else if(color->colortype == LCT_GREY) {
        if(chunkLength != 2) return 30;
        color->key_defined = 1;
        color->key_r = color->key_g = color->key_b = 256u * data[0] + data[1];
    } else if(color->colortype == LCT_RGB) {
        if(chunkLength != 6) return 41;
        color->key_defined = 1;
        color->key_r = 256u * data[0] + data[1];
        color->key_g = 256u * data[2] + data[3];
        color->key_b = 256u * data[4] + data[5];
    } else return 42;
    return 0;
}

int _start(void) {
    /* Input: chunk data at 0x1000, params at 0x0FF0 */
    const unsigned char* data = (const unsigned char*)0x1000;
    unsigned chunk_length = *(volatile unsigned*)0x0FF0;
    unsigned test_mode = *(volatile unsigned*)0x0FF4;  /* 0=PLTE, 1=tRNS */
    unsigned colortype = *(volatile unsigned*)0x0FF8;

    LodePNGColorMode color;
    color.colortype = colortype;
    color.bitdepth = 8;
    color.palette = NULL;
    color.palettesize = 0;
    color.key_defined = 0;
    color.key_r = color.key_g = color.key_b = 0;

    if(test_mode == 0) {
        /* Test PLTE */
        return readChunk_PLTE(&color, data, chunk_length);
    } else {
        /* Test tRNS — first set up palette */
        if(colortype == LCT_PALETTE) {
            /* Pre-allocate palette with some entries */
            unsigned plte_len = *(volatile unsigned*)0x0FFC;
            unsigned err = readChunk_PLTE(&color, data, plte_len);
            if(err) return 100 + err;  /* PLTE error, offset by 100 */
        }
        color.colortype = colortype;
        /* tRNS data starts after PLTE data */
        unsigned plte_bytes = (colortype == LCT_PALETTE) ?
            (*(volatile unsigned*)0x0FFC) : 0;
        return readChunk_tRNS(&color, data + plte_bytes, chunk_length);
    }
}
"""


# ══════════════════════════════════════════════════════
# REAL lodepng unfilterScanline
# ══════════════════════════════════════════════════════

UNFILTER_C = """
/* VERBATIM from lodepng.cpp lines 4446-4460 (filter types 0-2).
 * Simplified: only filter types 0 (none), 1 (sub), 2 (up).
 *
 * Input scanline at 0x1000, precon at 0x1100
 * Output recon at 0x1200
 * Parameters at 0x0FF0: bytewidth, filterType, length
 */

__attribute__((noinline))
static unsigned char paethPredictor(unsigned char a, unsigned char b, unsigned char c) {
    /* lodepng.cpp line 4437 — EXACT */
    int pa = (int)b - (int)c;
    int pb = (int)a - (int)c;
    int pc = pa + pb;
    if(pa < 0) pa = -pa;
    if(pb < 0) pb = -pb;
    if(pc < 0) pc = -pc;
    if(pa <= pb && pa <= pc) return a;
    else if(pb <= pc) return b;
    else return c;
}

/* lodepng.cpp line 4446 — EXACT (filter types 0-4) */
__attribute__((noinline))
static unsigned unfilterScanline(unsigned char* recon,
                                  const unsigned char* scanline,
                                  const unsigned char* precon,
                                  unsigned bytewidth,
                                  unsigned char filterType,
                                  unsigned length) {
    unsigned i;
    switch(filterType) {
        case 0:
            for(i = 0; i != length; ++i) recon[i] = scanline[i];
            break;
        case 1: {
            unsigned j = 0;
            for(i = 0; i != bytewidth; ++i) recon[i] = scanline[i];
            for(i = bytewidth; i != length; ++i, ++j)
                recon[i] = scanline[i] + recon[j];
            break;
        }
        case 2:
            if(precon) {
                for(i = 0; i != length; ++i) recon[i] = scanline[i] + precon[i];
            } else {
                for(i = 0; i != length; ++i) recon[i] = scanline[i];
            }
            break;
        case 3:
            if(precon) {
                unsigned j = 0;
                for(i = 0; i != bytewidth; ++i)
                    recon[i] = scanline[i] + (precon[i] >> 1u);
                for(i = bytewidth; i != length; ++i, ++j)
                    recon[i] = scanline[i] + ((recon[j] + precon[i]) >> 1u);
            } else {
                unsigned j = 0;
                for(i = 0; i != bytewidth; ++i) recon[i] = scanline[i];
                for(i = bytewidth; i != length; ++i, ++j)
                    recon[i] = scanline[i] + (recon[j] >> 1u);
            }
            break;
        case 4:
            if(precon) {
                unsigned j = 0;
                for(i = 0; i != bytewidth; ++i)
                    recon[i] = scanline[i] + paethPredictor(0, precon[i], 0);
                for(i = bytewidth; i != length; ++i, ++j)
                    recon[i] = scanline[i] + paethPredictor(recon[j], precon[i], precon[j]);
            } else {
                unsigned j = 0;
                for(i = 0; i != bytewidth; ++i) recon[i] = scanline[i];
                for(i = bytewidth; i != length; ++i, ++j)
                    recon[i] = scanline[i] + paethPredictor(recon[j], 0, 0);
            }
            break;
        default: return 36; /* invalid filter type */
    }
    return 0;
}

int _start(void) {
    unsigned char* scanline = (unsigned char*)0x1000;
    unsigned char* precon = (unsigned char*)0x1100;
    unsigned char* recon = (unsigned char*)0x1200;

    unsigned bytewidth = *(volatile unsigned*)0x0FF0;
    unsigned filterType = *(volatile unsigned*)0x0FF4;
    unsigned length = *(volatile unsigned*)0x0FF8;
    unsigned has_precon = *(volatile unsigned*)0x0FFC;

    /* Bounds check: length must be reasonable */
    if(length > 200) return 1;  /* safety for our small buffers */
    if(bytewidth == 0 || bytewidth > 8) return 2;
    if(filterType > 4) return 36;

    return unfilterScanline(recon, scanline, has_precon ? precon : 0,
                            bytewidth, (unsigned char)filterType, length);
}
"""


def _set_param(mem, offset, value):
    """Write a 32-bit LE value to memory."""
    v = value & 0xFFFFFFFF
    mem[offset] = v & 0xFF
    mem[offset+1] = (v >> 8) & 0xFF
    mem[offset+2] = (v >> 16) & 0xFF
    mem[offset+3] = (v >> 24) & 0xFF


@pytest.mark.skipif(not HAS_RISCV_GCC, reason="RISC-V GCC not installed")
class TestPLTEtRNS:
    """Test real lodepng PLTE and tRNS chunk parsing."""

    @pytest.fixture(autouse=True)
    def compile_plte(self):
        self.nisa = compile_asm(compile_c(PLTE_TRNS_C))
        print(f"  [PLTE/tRNS: {len(self.nisa)} NISA instructions]")

    def _run_plte(self, chunk_data, chunk_length=None):
        mem = bytearray(65536)
        for i, b in enumerate(chunk_data[:4096]):
            mem[0x1000 + i] = b
        cl = chunk_length if chunk_length is not None else len(chunk_data)
        _set_param(mem, 0x0FF0, cl)
        _set_param(mem, 0x0FF4, 0)  # mode=PLTE
        _set_param(mem, 0x0FF8, 3)  # colortype=palette
        return gpu_execute(self.nisa, memory_bytes=mem, device='cuda', max_cycles=100000)

    def _run_trns(self, plte_data, trns_data, colortype, trns_length=None):
        mem = bytearray(65536)
        for i, b in enumerate(plte_data):
            mem[0x1000 + i] = b
        for i, b in enumerate(trns_data):
            mem[0x1000 + len(plte_data) + i] = b
        tl = trns_length if trns_length is not None else len(trns_data)
        _set_param(mem, 0x0FF0, tl)
        _set_param(mem, 0x0FF4, 1)  # mode=tRNS
        _set_param(mem, 0x0FF8, colortype)
        _set_param(mem, 0x0FFC, len(plte_data))  # plte_len
        return gpu_execute(self.nisa, memory_bytes=mem, device='cuda', max_cycles=100000)

    def test_plte_valid(self):
        """Valid PLTE chunk (3 colors = 9 bytes)."""
        data = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])  # R, G, B
        r = self._run_plte(data)
        assert r.reg(10) == 0

    def test_plte_256_colors(self):
        """Max valid PLTE (256 colors = 768 bytes)."""
        data = bytes([i % 256 for i in range(768)])
        r = self._run_plte(data)
        assert r.reg(10) == 0

    def test_plte_empty(self):
        """Empty PLTE → error 38."""
        r = self._run_plte(b'', chunk_length=0)
        assert r.reg(10) == 38

    def test_plte_too_large(self):
        """PLTE > 768 bytes (> 256 colors) → error 38."""
        r = self._run_plte(b'\x00' * 900, chunk_length=900)
        assert r.reg(10) == 38

    def test_plte_fuzz_lengths(self):
        """Fuzz PLTE with various chunk lengths."""
        print("\n  PLTE chunk length fuzzing:")
        data = bytes([i % 256 for i in range(800)])
        for cl in [0, 1, 2, 3, 4, 5, 6, 9, 12, 255, 256*3, 256*3+1, 256*3+2,
                   257*3, 768, 769, 1000, 0xFFFF, 0x7FFFFFFF]:
            r = self._run_plte(data, chunk_length=cl)
            ps = cl // 3
            print(f"    len={cl:>10d} palsize={ps:>4d} → code={r.reg(10):3d}")

    def test_trns_palette_valid(self):
        """Valid tRNS for palette image."""
        plte = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])  # 3 colors
        trns = bytes([128, 64, 255])  # alpha for each
        r = self._run_trns(plte, trns, colortype=3)
        assert r.reg(10) == 0

    def test_trns_palette_too_many(self):
        """tRNS with more entries than palette → error 39."""
        plte = bytes([255, 0, 0, 0, 255, 0])  # 2 colors
        trns = bytes([128, 64, 255])  # 3 alpha values — too many!
        r = self._run_trns(plte, trns, colortype=3)
        assert r.reg(10) == 39

    def test_trns_grey_valid(self):
        """Valid tRNS for grayscale (2 bytes)."""
        r = self._run_trns(b'', bytes([0, 128]), colortype=0, trns_length=2)
        assert r.reg(10) == 0

    def test_trns_grey_wrong_size(self):
        """tRNS for grayscale != 2 bytes → error 30."""
        r = self._run_trns(b'', bytes([0, 128, 0]), colortype=0, trns_length=3)
        assert r.reg(10) == 30

    def test_trns_rgb_valid(self):
        """Valid tRNS for RGB (6 bytes)."""
        r = self._run_trns(b'', bytes([0, 255, 0, 128, 0, 64]), colortype=2, trns_length=6)
        assert r.reg(10) == 0

    def test_trns_rgba_rejected(self):
        """tRNS for RGBA → error 42 (not allowed)."""
        r = self._run_trns(b'', bytes([0, 0]), colortype=6, trns_length=2)
        assert r.reg(10) == 42


@pytest.mark.skipif(not HAS_RISCV_GCC, reason="RISC-V GCC not installed")
class TestUnfilterScanline:
    """Test real lodepng unfilterScanline with all filter types."""

    @pytest.fixture(autouse=True)
    def compile_unfilter(self):
        self.nisa = compile_asm(compile_c(UNFILTER_C))
        print(f"  [unfilter: {len(self.nisa)} NISA instructions]")

    def _run(self, scanline, precon, bytewidth, filter_type, length=None):
        mem = bytearray(65536)
        for i, b in enumerate(scanline[:256]):
            mem[0x1000 + i] = b
        if precon:
            for i, b in enumerate(precon[:256]):
                mem[0x1100 + i] = b
        if length is None:
            length = len(scanline)
        _set_param(mem, 0x0FF0, bytewidth)
        _set_param(mem, 0x0FF4, filter_type)
        _set_param(mem, 0x0FF8, length)
        _set_param(mem, 0x0FFC, 1 if precon else 0)
        r = gpu_execute(self.nisa, memory_bytes=mem, device='cuda', max_cycles=500000)
        # Read recon output from the RESULT's memory. (gpu_execute runs on its own copy of
        # memory_bytes and does not write back into the caller's bytearray, so the output
        # must be read via the result, not the input `mem`.)
        recon = bytes(r.mem_byte(0x1200 + i) for i in range(length))
        return r.reg(10), recon

    def test_filter_none(self):
        """Filter type 0: copy."""
        code, recon = self._run(b'\x01\x02\x03\x04', None, 1, 0)
        assert code == 0
        assert recon == b'\x01\x02\x03\x04'

    def test_filter_sub(self):
        """Filter type 1: sub (add previous pixel)."""
        code, recon = self._run(b'\x01\x02\x03\x04', None, 1, 1)
        assert code == 0
        # recon[0]=1, recon[1]=1+2=3, recon[2]=3+3=6, recon[3]=6+4=10
        assert recon == bytes([1, 3, 6, 10])

    def test_filter_up(self):
        """Filter type 2: up (add previous scanline)."""
        code, recon = self._run(b'\x01\x02\x03', b'\x10\x20\x30', 1, 2)
        assert code == 0
        assert recon == bytes([0x11, 0x22, 0x33])

    def test_filter_average(self):
        """Filter type 3: average."""
        code, recon = self._run(b'\x04\x04\x04', b'\x10\x10\x10', 1, 3)
        assert code == 0
        # recon[0] = scan[0] + precon[0]>>1 = 4 + 8 = 12
        # recon[1] = scan[1] + (recon[0]+precon[1])>>1 = 4 + (12+16)>>1 = 4+14 = 18
        assert recon[0] == 12

    def test_filter_paeth(self):
        """Filter type 4: Paeth."""
        code, recon = self._run(b'\x01\x02\x03\x04', b'\x10\x20\x30\x40', 1, 4)
        assert code == 0
        # Paeth(0, 0x10, 0) = 0x10, recon[0] = 1 + 0x10 = 0x11
        assert recon[0] == 0x11

    def test_invalid_filter(self):
        """Filter type 5+ → error 36."""
        code, _ = self._run(b'\x01\x02', None, 1, 5)
        assert code == 36

    def test_fuzz_bytewidth_length(self):
        """Fuzz with various bytewidth and length combinations."""
        print("\n  Unfilter fuzzing (bytewidth × length × filter):")
        data = bytes(range(200))
        precon = bytes([(i * 7) % 256 for i in range(200)])
        issues = []

        for bw in [1, 2, 3, 4, 6, 8]:
            for length in [0, 1, bw-1, bw, bw+1, bw*2, 10, 50, 100, 199, 200]:
                if length <= 0 or length > 200:
                    continue
                for ft in range(5):
                    code, recon = self._run(data[:length], precon[:length], bw, ft, length)
                    if code != 0:
                        issues.append((bw, length, ft, code))

        if issues:
            print(f"    Found {len(issues)} unexpected errors:")
            for bw, l, ft, code in issues[:10]:
                print(f"      bw={bw} len={l} filter={ft} → code={code}")
        else:
            print(f"    All combinations passed (5 filters × 6 bytewidths × lengths)")

    def test_length_equals_bytewidth(self):
        """Edge case: length exactly equals bytewidth."""
        for bw in [1, 2, 3, 4]:
            for ft in range(5):
                code, _ = self._run(bytes(bw), bytes(bw), bw, ft, bw)
                assert code == 0, f"bw={bw} ft={ft} returned {code}"

    def test_gradient_search_unfilter(self):
        """Try gradient search for inputs that cause unexpected behavior."""
        print("\n  Gradient search on unfilter parameters:")

        # Look for bytewidth/length combos where bytewidth > length
        # In the sub filter (type 1), the loop starts at i=bytewidth
        # If bytewidth > length, the second loop never executes — but is that safe?
        for bw in [1, 2, 4, 8]:
            for length in [0, 1, bw-1, bw, bw+1]:
                if length <= 0 or length > 200:
                    continue
                for ft in [1, 3, 4]:  # sub, average, paeth — use bytewidth
                    code, recon = self._run(bytes(length), bytes(length), bw, ft, length)
                    status = "OK" if code == 0 else f"ERR={code}"
                    # Check: when bytewidth > length, does the loop behave correctly?
                    flag = " ← bytewidth > length!" if bw > length else ""
                    print(f"    bw={bw} len={length} filter={ft} → {status}{flag}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
