"""Minimal PDF 1.4 backend: the same ``Canvas`` as a one-page PDF of the canvas' sheet with deterministic bytes.

Fonts, none embedded: base-14 Helvetica / Helvetica-Bold (WinAnsiEncoding) for ASCII text, and -- only when the sheet
prints a non-ASCII character (Chinese text) -- the predefined Adobe-GB1 CID font ``STSong-Light`` as a Type0 font with
the predefined CMap ``UniGB-UCS2-H`` (text written as UCS-2BE hex strings, every glyph one em wide: ``/DW 1000``), which
every PDF viewer with Chinese support provides. A text operation switches font per run (ASCII / wide), so the advance of
every glyph is the one ``text.text_width`` lays out. Bold Chinese text is drawn filled and stroked (text render mode 2).
A character outside the Basic Multilingual Plane is written as ``?``. Viewers with Chinese font support render the
predefined font (Acrobat / Reader, PDFium in Chrome and Edge, pdf.js, macOS Preview, PyMuPDF); Poppler needs its
``poppler-data`` package. The SVG of the same sheet needs no PDF font.

Uncompressed content stream, fixed object order, no creation date, no ``/ID``. Numbers are written with fixed decimals;
circles are four cubic Beziers with the standard literal constant; text rotation uses the canvas' fixed rotation table
(no libm). The document title (Info) is a UTF-16BE string.
"""

from __future__ import annotations

from .._version import DISTRIBUTION_NAME
from .canvas import ROTATIONS, Canvas, Style
from .text import is_ascii, text_width

MM_TO_PT = 72.0 / 25.4
BEZIER_CIRCLE = 0.5522847498  # 4/3 (sqrt(2) - 1): quarter-circle control-point distance
HEX_BASE = 16
CHANNEL_MAX = 255.0
_CAPS = {"butt": 0, "round": 1, "square": 2}
PRODUCER = f"OpenSci {DISTRIBUTION_NAME} drawing"
BMP_MAX = 0xFFFF
#: Stroke width of bold Chinese text, as a fraction of the font size (text render mode 2: fill, then stroke).
CJK_BOLD_STROKE = 0.03
CJK_FONT = "STSong-Light"
CJK_CMAP = "UniGB-UCS2-H"
#: The predefined font's descriptor values (Adobe's published metrics of STSong-Light).
CJK_DESCRIPTOR = ("<< /Type /FontDescriptor /FontName /STSong-Light /Flags 6 /FontBBox [-25 -254 1000 880] "
                  "/ItalicAngle 0 /Ascent 880 /Descent -120 /CapHeight 880 /StemV 93 >>")


def _rgb(color: str) -> str:
    r, g, b = (int(color[i:i + 2], HEX_BASE) / CHANNEL_MAX for i in (1, 3, 5))
    return f"{r:.4f} {g:.4f} {b:.4f}"


def pdf_string(text: str) -> str:
    """An ASCII literal string (anything else becomes ``?``)."""

    out = []
    for ch in text:
        if ch in "\\()":
            out.append("\\" + ch)
        elif is_ascii(ch):
            out.append(ch)
        else:
            out.append("?")
    return "(" + "".join(out) + ")"


def ucs2_hex(text: str) -> str:
    """A UCS-2BE hex string (a character beyond the BMP becomes ``?``)."""

    return "<" + "".join(f"{ord(ch) if ord(ch) <= BMP_MAX else ord('?'):04X}" for ch in text) + ">"


def info_string(text: str) -> str:
    """A text string of the document information dictionary: ASCII as a literal, otherwise UTF-16BE with its BOM."""

    if all(is_ascii(ch) for ch in text):
        return pdf_string(text)
    return "<FEFF" + "".join(f"{ord(ch) if ord(ch) <= BMP_MAX else ord('?'):04X}" for ch in text) + ">"


def runs(text: str) -> list[tuple[bool, str]]:
    """``text`` split into (wide, run) pieces: ASCII runs and runs of every other character."""

    out: list[tuple[bool, str]] = []
    for ch in text:
        wide = not is_ascii(ch)
        if out and out[-1][0] == wide:
            out[-1] = (wide, out[-1][1] + ch)
        else:
            out.append((wide, ch))
    return out


def _style(st: Style) -> tuple[str, str]:
    ops = []
    if st.stroke:
        ops.append(f"{_rgb(st.stroke)} RG {st.width:.3f} w {_CAPS[st.cap]} J 1 j")
        ops.append("[" + " ".join(f"{d:.3f}" for d in st.dash) + "] 0 d" if st.dash else "[] 0 d")
    if st.fill:
        ops.append(f"{_rgb(st.fill)} rg")
    if st.fill and st.stroke:
        paint = "B*" if st.evenodd else "B"
    elif st.fill:
        paint = "f*" if st.evenodd else "f"
    else:
        paint = "S"
    return " ".join(ops), paint


def _text_op(op: tuple, uses_cjk: list[bool]) -> str:
    (x, y), s, size, anchor, bold, rotate, color = op[1:]
    w = text_width(s, size, bold)
    shift = {"start": 0.0, "middle": -w / 2.0, "end": -w}[anchor]
    ca, sa = ROTATIONS[rotate]
    tx, ty = x + shift * ca, y + shift * sa
    shows = []
    for wide, piece in runs(s):
        if not wide:
            shows.append(f"/{'F2' if bold else 'F1'} {size:.3f} Tf {pdf_string(piece)} Tj")
            continue
        uses_cjk[0] = True
        if bold:
            shows.append(f"{_rgb(color)} RG {size * CJK_BOLD_STROKE:.3f} w 2 Tr /F3 {size:.3f} Tf {ucs2_hex(piece)} Tj 0 Tr")
        else:
            shows.append(f"/F3 {size:.3f} Tf {ucs2_hex(piece)} Tj")
    body = (f"BT {_rgb(color)} rg {ca:.6f} {sa:.6f} {sa:.6f} {-ca:.6f} {tx:.3f} {ty:.3f} Tm " + " ".join(shows) + " ET")
    return f"q {body} Q" if bold and any(wide for wide, _ in runs(s)) else body


def render_pdf(canvas: Canvas, title: str) -> bytes:
    body: list[str] = [f"q {MM_TO_PT:.6f} 0 0 {-MM_TO_PT:.6f} 0 {canvas.height * MM_TO_PT:.4f} cm"]
    k = BEZIER_CIRCLE
    uses_cjk = [False]
    for op in canvas.ops:
        kind = op[0]
        if kind == "path":
            state, paint = _style(op[2])
            segs = []
            for pts, closed in op[1]:
                segs.append(f"{pts[0][0]:.3f} {pts[0][1]:.3f} m " + " ".join(f"{x:.3f} {y:.3f} l" for x, y in pts[1:])
                            + (" h" if closed else ""))
            body.append(f"{state} " + " ".join(segs) + f" {paint}")
        elif kind == "circle":
            (cx, cy), r, st = op[1], op[2], op[3]
            state, paint = _style(st)
            c = r * k
            body.append(
                f"{state} {cx + r:.3f} {cy:.3f} m {cx + r:.3f} {cy + c:.3f} {cx + c:.3f} {cy + r:.3f} {cx:.3f} {cy + r:.3f} c "
                f"{cx - c:.3f} {cy + r:.3f} {cx - r:.3f} {cy + c:.3f} {cx - r:.3f} {cy:.3f} c "
                f"{cx - r:.3f} {cy - c:.3f} {cx - c:.3f} {cy - r:.3f} {cx:.3f} {cy - r:.3f} c "
                f"{cx + c:.3f} {cy - r:.3f} {cx + r:.3f} {cy - c:.3f} {cx + r:.3f} {cy:.3f} c h {paint}"
            )
        elif kind == "text":
            body.append(_text_op(op, uses_cjk))
    body.append("Q")
    content = ("\n".join(body) + "\n").encode("latin-1")
    fonts = "/F1 5 0 R /F2 6 0 R" + (" /F3 7 0 R" if uses_cjk[0] else "")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {canvas.width * MM_TO_PT:.4f} {canvas.height * MM_TO_PT:.4f}] "
         f"/Resources << /Font << {fonts} >> >> /Contents 4 0 R >>").encode("latin-1"),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
    ]
    if uses_cjk[0]:
        objects += [
            f"<< /Type /Font /Subtype /Type0 /BaseFont /{CJK_FONT} /Encoding /{CJK_CMAP} /DescendantFonts [8 0 R] >>".encode("latin-1"),
            (f"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /{CJK_FONT} /CIDSystemInfo << /Registry (Adobe) "
             "/Ordering (GB1) /Supplement 2 >> /FontDescriptor 9 0 R /DW 1000 >>").encode("latin-1"),
            CJK_DESCRIPTOR.encode("latin-1"),
        ]
    objects.append(b"<< /Title " + info_string(title).encode("latin-1") + b" /Producer " + pdf_string(PRODUCER).encode("latin-1") + b" >>")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {len(objects)} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
