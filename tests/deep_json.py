"""JSON text nested too deeply for any JSON parser to read.

The JSON parser's depth limit is not a fixed count: on CPython 3.14 it follows the C stack the thread has, which differs
by platform (about 11,000 levels of ``[`` in a 1 MB thread stack on Windows; 100,000 levels parsed on a Linux runner
with its 8 MB default). ``UNREADABLE_DEPTH`` lies far beyond any such stack: each level takes some 80 to 180 bytes of stack,
so 2,000,000 levels would need 160 MB or more. A document nested this deeply fails to parse on every supported platform.
The text is 4 MB, and the parser stops at its stack limit long before the end, so the attempt is fast.
"""

from __future__ import annotations

UNREADABLE_DEPTH = 2_000_000


def deep_array(depth: int = UNREADABLE_DEPTH) -> str:
    """A JSON array nested ``depth`` levels deep: ``[[[...]]]``."""
    return "[" * depth + "]" * depth
