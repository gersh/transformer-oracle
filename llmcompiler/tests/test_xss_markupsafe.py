"""
XSS bypass analysis: MarkupSafe escape vs Chrome HTML parser.

MarkupSafe escapes: & < > " '
Chrome's HTML parser has additional behaviors:
  - Null bytes (0x00) can terminate strings or be stripped
  - Various Unicode characters that normalize or confuse parsers
  - Backtick (`) in older IE (not Chrome)
  - JavaScript: protocol in href/src attributes
  - Event handlers (onclick, onerror, etc.) — not tag-level
  - CSS expression() — old IE only
  - SVG/MathML namespace confusion

The analysis: compile MarkupSafe's escape function to NISA,
run it on every byte value 0-255, and check what passes through.
Then analyze whether any unescaped character could trigger XSS
in Chrome's HTML parser.
"""

import pytest
from ..compiler.python_compiler import compile_python
from ..runtime.gpu_executor import gpu_execute


# ── MarkupSafe escape logic (exact port to our Python subset) ──

def markupsafe_escape_byte(b):
    """Apply MarkupSafe's escape to a single byte value.

    Returns:
      0 if the byte is escaped (replaced with entity)
      1 if the byte passes through unchanged
    """
    # MarkupSafe escapes exactly these 5 characters:
    if b == 38:   return 0   # & → &amp;
    if b == 60:   return 0   # < → &lt;
    if b == 62:   return 0   # > → &gt;
    if b == 34:   return 0   # " → &#34;
    if b == 39:   return 0   # ' → &#39;
    return 1  # passes through


# ── Chrome HTML parser dangerous characters ──

# Characters that could be dangerous in different HTML contexts:
# In tag context:  < (open tag) — ESCAPED ✓
# In attribute value: " ' (break out of attribute) — ESCAPED ✓
# In attribute name: = (assign attribute)
# In unquoted attr: space, tab, /, > (terminate value)
# Special: null byte (0x00), BOM (0xFEFF), various control chars

# Context-dependent dangers:
# Inside <script>: any character can be dangerous
# Inside href="": javascript: protocol
# Inside style="": expression(), url()
# Inside event handler: any JS code

# The KEY question: can any unescaped byte value create a NEW HTML
# context that MarkupSafe doesn't account for?


class TestMarkupSafeEscaping:

    def test_all_bytes_escaped_or_safe(self):
        """Analyze every byte value 0-255 for XSS potential."""
        nisa = compile_python(markupsafe_escape_byte)

        print("\n" + "="*65)
        print("MARKUPSAFE ESCAPE ANALYSIS: ALL 256 BYTE VALUES")
        print("="*65)

        escaped = []
        passed = []

        for b in range(256):
            r = gpu_execute(nisa, initial_registers={1: b}, device='cuda')
            if r.reg(10) == 0:
                escaped.append(b)
            else:
                passed.append(b)

        print(f"\n  Escaped ({len(escaped)}): {[chr(b) if 32<=b<127 else f'0x{b:02X}' for b in escaped]}")
        print(f"  Passed through: {len(passed)} byte values")

        # Analyze passed-through bytes for potential danger
        print(f"\n  SECURITY ANALYSIS of passed-through bytes:")

        dangerous = []

        for b in passed:
            concerns = []

            # Null byte
            if b == 0x00:
                concerns.append("NULL byte — can truncate strings in C parsers, "
                               "Chrome strips them in HTML")

            # Control characters
            if b < 0x20 and b not in (0x09, 0x0A, 0x0D):
                concerns.append(f"Control char 0x{b:02X} — some parsers handle unexpectedly")

            # Tab, newline, CR — can break attribute values in unquoted contexts
            if b in (0x09, 0x0A, 0x0C, 0x0D):
                concerns.append("Whitespace — can separate attributes in unquoted context")

            # Backtick — IE6 treated as attribute quote (Chrome: no)
            if b == 0x60:
                concerns.append("Backtick — was attribute delimiter in IE6 (NOT Chrome)")

            # Forward slash — can close tags like </
            if b == 0x2F:
                concerns.append("/ — part of closing tag </...>, but needs < which is escaped")

            # Equals — attribute assignment
            if b == 0x3D:
                concerns.append("= — attribute assignment, but needs to be in tag context")

            # Space — attribute separator
            if b == 0x20:
                concerns.append("Space — separates attributes, but only in tag context")

            # Semicolon — entity terminator
            if b == 0x3B:
                concerns.append("; — entity terminator, but entities need & which is escaped")

            # Parentheses — JavaScript calls
            if b in (0x28, 0x29):
                concerns.append("() — JavaScript function calls, but only in JS context")

            if concerns:
                ch = chr(b) if 32 <= b < 127 else f'0x{b:02X}'
                for c in concerns:
                    dangerous.append((b, ch, c))

        print()
        for b, ch, concern in dangerous:
            print(f"    [{ch:>4s}] (0x{b:02X}): {concern}")

        print(f"\n  VERDICT:")
        print(f"    MarkupSafe escapes the 5 critical characters: & < > \" '")
        print(f"    These are the ONLY characters that can create or break")
        print(f"    HTML tags and attribute values in compliant parsers.")
        print()
        print(f"    Potential issues (all CONTEXT-DEPENDENT):")
        print(f"    1. If escaped output is placed in an UNQUOTED attribute,")
        print(f"       space/tab/newline can terminate the value.")
        print(f"       Fix: always quote attribute values.")
        print(f"    2. If placed inside <script> or event handlers,")
        print(f"       ANY character is dangerous (MarkupSafe doesn't help).")
        print(f"       Fix: use JSON encoding for JS contexts.")
        print(f"    3. Null bytes (0x00) are stripped by Chrome but could")
        print(f"       confuse server-side processing that uses C strings.")
        print(f"    4. The characters () [] {{}} are passed through —")
        print(f"       harmless in HTML context but dangerous in JS/CSS.")

    def test_context_specific_attacks(self):
        """Test known context-specific XSS vectors."""
        print("\n" + "="*65)
        print("CONTEXT-SPECIFIC XSS VECTOR ANALYSIS")
        print("="*65)

        # Test vectors: (input, context, whether MarkupSafe prevents it)
        vectors = [
            # Tag injection — prevented
            ('<script>alert(1)</script>',
             'tag body',
             'BLOCKED — < and > are escaped'),

            # Attribute breakout — prevented
            ('" onmouseover="alert(1)',
             'quoted attribute',
             'BLOCKED — " is escaped'),

            # Single-quote breakout — prevented
            ("' onmouseover='alert(1)",
             'single-quoted attribute',
             "BLOCKED — ' is escaped"),

            # Entity injection — prevented
            ('&lt;script&gt;',
             'tag body',
             'BLOCKED — & is escaped (double-encoding prevented)'),

            # javascript: in href — NOT prevented by MarkupSafe alone
            ('javascript:alert(1)',
             'href attribute value',
             'PASSES THROUGH — MarkupSafe does not filter URL schemes! '
             'The application must validate URLs separately.'),

            # Event handler content — NOT prevented
            ('alert(1)',
             'onclick attribute value',
             'PASSES THROUGH — MarkupSafe cannot help when output is '
             'already in a JavaScript context.'),

            # CSS expression — NOT prevented (but Chrome ignores it)
            ('expression(alert(1))',
             'style attribute',
             'PASSES THROUGH — but Chrome does not support CSS expressions.'),

            # Null byte injection
            ('test\x00<script>',
             'tag body',
             'PARTIALLY BLOCKED — < is escaped, but null byte passes. '
             'Chrome strips null bytes, so <script> cannot form.'),

            # Tab in attribute value
            ('x\tonmouseover=alert(1)',
             'unquoted attribute',
             'DANGEROUS IF UNQUOTED — tab separates attributes. '
             'Fix: always use quoted attribute values.'),
        ]

        for payload, context, analysis in vectors:
            # Check which chars MarkupSafe would escape
            escaped_chars = [c for c in payload if c in '&<>"\'']
            safe_chars = [c for c in payload if c not in '&<>"\'']

            display = payload.replace('\x00', '\\0').replace('\t', '\\t')
            print(f"\n  Vector: {display[:50]}")
            print(f"  Context: {context}")
            print(f"  Escaped: {len(escaped_chars)} chars  Passed: {len(safe_chars)} chars")
            print(f"  → {analysis}")

        print(f"\n  SUMMARY:")
        print(f"    MarkupSafe prevents XSS in the HTML body and quoted attributes.")
        print(f"    It does NOT prevent:")
        print(f"      - javascript: URLs in href/src (need URL validation)")
        print(f"      - Code injection in event handlers (need CSP)")
        print(f"      - Attacks via unquoted attributes (need quoting)")
        print(f"    These are well-documented limitations, not bugs.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
