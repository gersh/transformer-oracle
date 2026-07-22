"""
Test MD5 hashing via NISA execution.

Generates an unrolled MD5 program in NISA, executes it on the
reference executor, and verifies against Python's hashlib.

This tests the full pipeline: all ALU operations (ADD, AND, OR, XOR, NOT),
shifts (for rotations), memory (LOAD for constants), and control flow.
"""

import hashlib
import math
import struct
import pytest

from ..core.nisa import Instruction, Opcode, movi, halt
from ..core.state import StateConfig
from ..runtime.executor import execute_program


# ── MD5 constants ──

MD5_K = [int(abs(math.sin(i + 1)) * (2**32)) & 0xFFFFFFFF for i in range(64)]

MD5_S = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
]

MD5_INIT = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476]


# ── MD5 padding ──

def md5_pad(message: bytes) -> list[int]:
    """Pad message per MD5 spec. Returns list of 32-bit LE words.

    Only supports single-block messages (up to 55 bytes).
    """
    msg_len = len(message)
    bit_len = msg_len * 8

    msg = bytearray(message) + b'\x80'
    while len(msg) % 64 != 56:
        msg.append(0)
    msg += struct.pack('<Q', bit_len)

    assert len(msg) == 64, f"Multi-block messages not yet supported (got {len(msg)} bytes)"

    words = []
    for i in range(0, 64, 4):
        words.append(struct.unpack('<I', msg[i:i+4])[0])
    return words


# ── NISA code generation for MD5 ──

# Register allocation:
#   r1 = A, r2 = B, r3 = C, r4 = D  (MD5 working state)
#   r5 = F (accumulator for current round)
#   r6 = temp (loaded constants)
#   r7 = temp (memory addresses, shift amounts)
#   r8 = temp (shift left result)
#   r9 = temp (shift right result)
#   r10 = saved D (before ABCD rotation)
#
# Memory layout:
#   Words 0-15:  Message block M[0..15]
#   Words 16-79: K constants K[0..63]
#   Words 80-83: Initial hash values (a0, b0, c0, d0)


def generate_md5_program(message: bytes):
    """Generate NISA program and initial memory for MD5 hashing.

    Args:
        message: input message bytes (max 55 bytes for single block)

    Returns:
        (instructions, initial_memory) tuple
    """
    # Pad message
    words = md5_pad(message)

    # Build initial memory
    initial_memory = {}
    for i, w in enumerate(words):
        initial_memory[i] = w
    for i, k in enumerate(MD5_K):
        initial_memory[16 + i] = k
    for i, h in enumerate(MD5_INIT):
        initial_memory[80 + i] = h

    # Generate instructions
    prog = []

    # ── Initialization: load hash state from memory ──
    for i, reg in enumerate([1, 2, 3, 4]):
        prog.append(movi(7, 80 + i))
        prog.append(Instruction(Opcode.LOAD, a=reg, b=7, c=0))

    # ── 64 rounds (fully unrolled) ──
    for i in range(64):
        _emit_md5_round(prog, i)

    # ── Finalization: add initial hash values back ──
    for i, reg in enumerate([1, 2, 3, 4]):
        prog.append(movi(7, 80 + i))
        prog.append(Instruction(Opcode.LOAD, a=6, b=7, c=0))
        prog.append(Instruction(Opcode.ADD, a=reg, b=reg, c=6))

    prog.append(halt())
    return prog, initial_memory


def _emit_md5_round(prog: list, i: int):
    """Emit NISA instructions for one MD5 round.

    Before: r1=A, r2=B, r3=C, r4=D
    After:  r1=D_old, r2=B+rotl(F,s), r3=B_old, r4=C_old
    """
    # Step 1: Compute F and determine g
    if i < 16:
        # F = (B & C) | (~B & D)
        prog.append(Instruction(Opcode.AND, a=5, b=2, c=3))   # r5 = B & C
        prog.append(Instruction(Opcode.NOT, a=6, b=2, c=0))   # r6 = ~B
        prog.append(Instruction(Opcode.AND, a=6, b=6, c=4))   # r6 = ~B & D
        prog.append(Instruction(Opcode.OR, a=5, b=5, c=6))    # r5 = F
        g = i
    elif i < 32:
        # F = (D & B) | (~D & C)
        prog.append(Instruction(Opcode.AND, a=5, b=4, c=2))   # r5 = D & B
        prog.append(Instruction(Opcode.NOT, a=6, b=4, c=0))   # r6 = ~D
        prog.append(Instruction(Opcode.AND, a=6, b=6, c=3))   # r6 = ~D & C
        prog.append(Instruction(Opcode.OR, a=5, b=5, c=6))    # r5 = F
        g = (5 * i + 1) % 16
    elif i < 48:
        # F = B ^ C ^ D
        prog.append(Instruction(Opcode.XOR, a=5, b=2, c=3))   # r5 = B ^ C
        prog.append(Instruction(Opcode.XOR, a=5, b=5, c=4))   # r5 = F
        g = (3 * i + 5) % 16
    else:
        # F = C ^ (B | ~D)
        prog.append(Instruction(Opcode.NOT, a=6, b=4, c=0))   # r6 = ~D
        prog.append(Instruction(Opcode.OR, a=6, b=2, c=6))    # r6 = B | ~D
        prog.append(Instruction(Opcode.XOR, a=5, b=3, c=6))   # r5 = F
        g = (7 * i) % 16

    # Step 2: Save D (needed for ABCD rotation at end)
    prog.append(Instruction(Opcode.MOV, a=10, b=4, c=0))      # r10 = D

    # Step 3: F = F + A
    prog.append(Instruction(Opcode.ADD, a=5, b=5, c=1))       # r5 = F + A

    # Step 4: F = F + K[i]  (load from memory)
    prog.append(movi(7, 16 + i))                               # r7 = addr of K[i]
    prog.append(Instruction(Opcode.LOAD, a=6, b=7, c=0))      # r6 = K[i]
    prog.append(Instruction(Opcode.ADD, a=5, b=5, c=6))       # r5 += K[i]

    # Step 5: F = F + M[g]  (load from memory)
    prog.append(movi(7, g))                                     # r7 = addr of M[g]
    prog.append(Instruction(Opcode.LOAD, a=6, b=7, c=0))      # r6 = M[g]
    prog.append(Instruction(Opcode.ADD, a=5, b=5, c=6))       # r5 += M[g]

    # Step 6: Left rotate F by s[i] bits
    #   rotl(x, s) = (x << s) | (x >> (32 - s))
    s = MD5_S[i]
    prog.append(movi(7, s))                                     # r7 = s
    prog.append(Instruction(Opcode.SHL, a=8, b=5, c=7))       # r8 = F << s
    prog.append(movi(7, 32 - s))                                # r7 = 32 - s
    prog.append(Instruction(Opcode.SHR, a=9, b=5, c=7))       # r9 = F >> (32-s)
    prog.append(Instruction(Opcode.OR, a=5, b=8, c=9))        # r5 = rotl(F, s)

    # Step 7: B_new = B + rotl(F, s)
    prog.append(Instruction(Opcode.ADD, a=5, b=5, c=2))       # r5 = B + rotl(F, s)

    # Step 8: Rotate ABCD
    #   A_new = D_old, D_new = C_old, C_new = B_old, B_new = result
    prog.append(Instruction(Opcode.MOV, a=1, b=10, c=0))      # A = saved D
    prog.append(Instruction(Opcode.MOV, a=4, b=3, c=0))       # D = old C
    prog.append(Instruction(Opcode.MOV, a=3, b=2, c=0))       # C = old B
    prog.append(Instruction(Opcode.MOV, a=2, b=5, c=0))       # B = result


# ── Hash output formatting ──

def format_md5_hash(a0: int, b0: int, c0: int, d0: int) -> str:
    """Format 4 x 32-bit words as MD5 hex digest (little-endian byte order)."""
    digest = struct.pack('<4I', a0, b0, c0, d0)
    return digest.hex()


# ── Reference MD5 ──

def md5_reference(message: bytes) -> str:
    """Compute MD5 using Python's hashlib (reference)."""
    return hashlib.md5(message).hexdigest()


def md5_reference_words(message: bytes) -> tuple[int, int, int, int]:
    """Compute MD5 and return as 4 x 32-bit LE words."""
    digest = hashlib.md5(message).digest()
    return struct.unpack('<4I', digest)


# ── Tests ──

class TestMD5CodeGen:
    def test_program_size(self):
        """Verify the generated program fits in instruction memory."""
        prog, mem = generate_md5_program(b"")
        # Should be around 1380 instructions
        assert len(prog) < 2048, f"Program too large: {len(prog)} instructions"
        assert len(prog) > 1000, f"Program suspiciously small: {len(prog)} instructions"

    def test_memory_layout(self):
        """Verify memory is correctly initialized."""
        prog, mem = generate_md5_program(b"")
        # Message words
        assert mem[0] == 0x00000080  # first byte is 0x80 (padding)
        for i in range(1, 14):
            assert mem[i] == 0  # padding zeros
        assert mem[14] == 0  # bit length low (empty message = 0 bits)
        assert mem[15] == 0  # bit length high
        # K constants
        assert mem[16] == MD5_K[0]  # 0xd76aa478
        assert mem[79] == MD5_K[63]
        # Initial hash
        assert mem[80] == 0x67452301
        assert mem[83] == 0x10325476

    def test_padding_abc(self):
        """Verify padding of 'abc'."""
        words = md5_pad(b"abc")
        assert len(words) == 16
        # 'a'=0x61, 'b'=0x62, 'c'=0x63, pad=0x80
        # In LE word: bytes 61 62 63 80 → 0x80636261
        assert words[0] == 0x80636261
        assert words[14] == 24  # 3 bytes = 24 bits


class TestMD5Execution:
    """Run MD5 on the NISA executor and verify against hashlib."""

    def _run_md5(self, message: bytes) -> str:
        """Compile and execute MD5 for a message, return hex digest."""
        prog, mem = generate_md5_program(message)
        config = StateConfig(n_instr_slots=2048)
        result = execute_program(
            prog,
            initial_memory=mem,
            config=config,
            max_cycles=len(prog) + 100,
        )
        assert result.halted, f"Program did not halt after {result.cycles} cycles"

        a0 = result.reg(1)
        b0 = result.reg(2)
        c0 = result.reg(3)
        d0 = result.reg(4)
        return format_md5_hash(a0, b0, c0, d0)

    def test_md5_empty(self):
        """MD5 of empty string."""
        expected = md5_reference(b"")
        got = self._run_md5(b"")
        assert got == expected, f"MD5('') = {got}, expected {expected}"

    def test_md5_a(self):
        """MD5 of 'a'."""
        expected = md5_reference(b"a")
        got = self._run_md5(b"a")
        assert got == expected, f"MD5('a') = {got}, expected {expected}"

    def test_md5_abc(self):
        """MD5 of 'abc'."""
        expected = md5_reference(b"abc")
        got = self._run_md5(b"abc")
        assert got == expected, f"MD5('abc') = {got}, expected {expected}"

    def test_md5_hello(self):
        """MD5 of 'hello'."""
        expected = md5_reference(b"hello")
        got = self._run_md5(b"hello")
        assert got == expected, f"MD5('hello') = {got}, expected {expected}"

    def test_md5_longer(self):
        """MD5 of a longer string (still single block, < 56 bytes)."""
        msg = b"The quick brown fox"
        expected = md5_reference(msg)
        got = self._run_md5(msg)
        assert got == expected, f"MD5('{msg.decode()}') = {got}, expected {expected}"

    def test_md5_max_single_block(self):
        """MD5 of a 55-byte message (maximum for single block)."""
        msg = b"A" * 55
        expected = md5_reference(msg)
        got = self._run_md5(msg)
        assert got == expected, f"MD5('A'*55) = {got}, expected {expected}"


class TestMD5ReferenceValues:
    """Verify our reference matches known MD5 values."""

    def test_known_hashes(self):
        assert md5_reference(b"") == "d41d8cd98f00b204e9800998ecf8427e"
        assert md5_reference(b"a") == "0cc175b9c0f1b6a831c399e269772661"
        assert md5_reference(b"abc") == "900150983cd24fb0d6963f7d28e17f72"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
