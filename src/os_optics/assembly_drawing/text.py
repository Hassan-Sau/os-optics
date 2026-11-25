"""Text metrics and number formatting of the drawing sheets (pure, deterministic).

* ``text_width`` uses the standard Helvetica / Helvetica-Bold advance widths (AFM units per 1000 em) of the printable
  ASCII range, the fonts the PDF backend names and the SVG font stack approximates; every other character (Chinese
  text and full-width punctuation) is one em wide, the advance of the predefined CJK font the PDF backend names
  (``pdf.CJK_FONT``). Layout is identical on every runtime and needs no font file;
* ``wrap`` breaks at spaces and, in Chinese text, between any two wide characters (never before closing punctuation);
* ``paper`` formats a paper coordinate (2 decimals, trailing zeros stripped, never ``-0``);
* ``value_text`` formats a measured millimetre value as the drawing result records it (at most 3 decimals, trailing
  zeros stripped), the form ``DrawingDimensionV1.value_mm`` carries;
* ``mm`` / ``deg`` format a value as the sheet prints it: one decimal (0.1 mm, 0.1 deg), trailing zero stripped.

Python's float formatting is correctly rounded (the same digits on CPython for Windows, Linux and Pyodide).
"""

from __future__ import annotations

PAPER_DECIMALS = 2
VALUE_DECIMALS = 3
PRINT_DECIMALS = 1
EM = 1000.0
FIRST_PRINTABLE = 32
LAST_PRINTABLE = 126
WIDE_WIDTH = 1000  # every non-ASCII character: one em (the CJK font's fixed advance)

# Advance widths of ASCII 32..126 (space .. tilde).
_W_REGULAR = (
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556,
    278, 278, 584, 584, 584, 556, 1015,
    667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944,
    667, 667, 611,
    278, 278, 278, 469, 556, 333,
    556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722,
    500, 500, 500,
    334, 260, 334, 584,
)
_W_BOLD = (
    278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556,
    333, 333, 584, 584, 584, 611, 975,
    722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944,
    667, 667, 611,
    333, 278, 333, 584, 556, 333,
    556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611, 611, 611, 389, 556, 333, 611, 556, 778,
    556, 556, 500,
    389, 280, 389, 584,
)
#: Wide punctuation that never starts a line (it stays with the character before it).
NO_LINE_START = frozenset("，。、；：！？）》」』】〕”’")
#: Wide punctuation that never ends a line (it stays with the character after it).
NO_LINE_END = frozenset("（《「『【〔“‘")


def is_ascii(ch: str) -> bool:
    return FIRST_PRINTABLE <= ord(ch) <= LAST_PRINTABLE


def text_width(text: str, size: float, bold: bool = False) -> float:
    """Advance width of ``text`` at font ``size`` (paper mm)."""

    table = _W_BOLD if bold else _W_REGULAR
    total = 0
    for ch in text:
        code = ord(ch)
        total += table[code - FIRST_PRINTABLE] if FIRST_PRINTABLE <= code <= LAST_PRINTABLE else WIDE_WIDTH
    return total * size / EM


def _units(text: str) -> list[str]:
    """Break opportunities of ``text``: ASCII words (with their trailing spaces kept apart) and single wide characters
    (a wide closing punctuation mark is glued to the unit before it)."""

    units: list[str] = []
    word = ""
    opening = ""
    for ch in text:
        if opening and ch != " ":
            if is_ascii(ch):
                word = opening + word
            else:
                ch = opening + ch
            opening = ""
        if ch in NO_LINE_END:
            if word:
                units.append(word)
                word = ""
            opening = ch
            continue
        if is_ascii(ch[-1]):
            if ch == " ":
                if word:
                    units.append(word)
                    word = ""
                units.append(" ")
            else:
                word += ch
            continue
        if word:
            units.append(word)
            word = ""
        if ch in NO_LINE_START and units and units[-1] != " ":
            units[-1] += ch
        else:
            units.append(ch)
    if word or opening:
        units.append(word + opening)
    return units


def wrap(text: str, width: float, size: float, bold: bool = False) -> list[str]:
    """Greedy wrap of ``text`` into lines no wider than ``width`` (a single over-long unit keeps its own line). ASCII
    text breaks at spaces only; Chinese text also between wide characters."""

    lines: list[str] = []
    current = ""
    for unit in _units(text):
        if unit == " ":
            if current:
                current += " "
            continue
        trial = current + unit
        if current.strip() and text_width(trial.rstrip(), size, bold) > width:
            lines.append(current.rstrip())
            current = unit
        else:
            current = trial
    if current.strip():
        lines.append(current.rstrip())
    return lines


def _stripped(number: float, places: int) -> str:
    text = f"{number:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "", "-") else text


def paper(value: float) -> str:
    """A paper coordinate as written into the SVG."""

    return _stripped(value, PAPER_DECIMALS)


def value_text(value: float) -> str:
    """A measured value as the drawing result records it (``DrawingDimensionV1.value_mm``)."""

    return _stripped(value, VALUE_DECIMALS)


def mm(value: float) -> str:
    """A length as the sheet prints it: 0.1 mm."""

    return _stripped(value, PRINT_DECIMALS)


def deg(value: float) -> str:
    """An angle as the sheet prints it: 0.1 deg."""

    return _stripped(value, PRINT_DECIMALS)


def ascii_only(text: str) -> bool:
    return all(is_ascii(ch) for ch in text)
