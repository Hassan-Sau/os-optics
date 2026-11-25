"""ECMAScript ``Number::toString`` serialization (RFC 8785 §3.2.2.3), stdlib-only.

Not a public module. Python's ``float.__repr__`` already produces the
shortest decimal digit string that round-trips to the same IEEE-754 binary64
value (correctly rounded, ties-to-even) -- the same digit sequence ECMAScript
is required to produce. This module reuses ``repr`` purely as a digit-string
extractor and re-renders those digits under ECMAScript's own fixed/exponential
formatting thresholds (RFC 8785 Appendix B; ECMA-262 §6.1.6.1.20 "Number::
toString"), instead of Python's own display thresholds.
"""

from __future__ import annotations

import math
import re


_REPR_PATTERN = re.compile(r"^(?P<int>\d+)(?:\.(?P<frac>\d+))?(?:[eE](?P<exp>[+-]?\d+))?$")


def format_ecmascript_number(value: float) -> str:
    """Return the ECMAScript ``Number::toString`` decimal text for a finite float.

    Caller is responsible for rejecting non-finite values first; this function
    assumes ``value`` is finite. ``-0.0`` and ``0.0`` both render as ``"0"``.
    """

    if value == 0.0:
        return "0"
    sign = "-" if math.copysign(1.0, value) < 0 else ""
    magnitude = -value if sign else value
    digits, n = _shortest_digits_and_exponent(magnitude)
    return sign + _render(digits, n)


def _shortest_digits_and_exponent(x: float) -> tuple[str, int]:
    """Return (significant digits with no leading/trailing zero, decimal exponent n).

    ``n`` is defined so that ``x == 0.<digits> * 10**n`` (RFC 8785 Appendix B /
    ECMA-262 "let n, k, and s be integers such that ... the Number value for
    s x 10^(n-k) is x"). ``x`` must be strictly positive and finite.
    """

    match = _REPR_PATTERN.match(repr(x))
    if match is None:  # pragma: no cover - repr() of a finite float always matches
        raise AssertionError(f"unexpected float repr form: {x!r}")

    int_part = match.group("int")
    frac_part = match.group("frac") or ""
    exp = int(match.group("exp") or "0")

    full_digits = int_part + frac_part
    frac_len = len(frac_part)

    stripped_leading = full_digits.lstrip("0")
    sig_digits = stripped_leading.rstrip("0")
    trailing_zero_count = len(stripped_leading) - len(sig_digits)

    k = len(sig_digits)
    e10 = trailing_zero_count + exp - frac_len
    n = e10 + k
    return sig_digits, n


def _render(digits: str, n: int) -> str:
    """RFC 8785 Appendix B / ECMA-262 6.1.6.1.20 steps 4-8 (positive magnitude)."""

    k = len(digits)
    if k <= n <= 21:
        return digits + ("0" * (n - k))
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + ("0" * -n) + digits
    e = n - 1
    exp_sign = "+" if e >= 0 else "-"
    mantissa = digits[0] if k == 1 else digits[0] + "." + digits[1:]
    return f"{mantissa}e{exp_sign}{abs(e)}"
