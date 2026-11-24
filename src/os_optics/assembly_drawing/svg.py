"""SVG backend: a ``Canvas`` as one deterministic SVG 1.1 document (1 user unit = 1 mm on paper, the canvas' sheet).

The document carries a ``<metadata id="opensci-drawing-metadata">`` element holding compact, sorted JSON of the
drawing's bindings (package / detail content hashes, resolution id, BOM hash, status, generator version). Coordinates
are written with two decimals, attributes in a fixed order, no timestamps, LF line endings. The root names the sheet's
language (``xml:lang``) and a font stack that falls back to common Simplified Chinese fonts for Chinese text.
"""

from __future__ import annotations

import json
from typing import Any

from .canvas import PAPER, Canvas, Style
from .text import paper

SVG_METADATA_ID = "opensci-drawing-metadata"
FONT_FAMILY = ("Helvetica, Arial, 'Liberation Sans', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei', "
               "'PingFang SC', SimHei, SimSun, sans-serif")
_ESCAPES = (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;"))


def escape(text: str) -> str:
    for raw, entity in _ESCAPES:
        text = text.replace(raw, entity)
    return text


def _style(st: Style) -> str:
    parts = [f'fill="{st.fill or "none"}"']
    if st.evenodd:
        parts.append('fill-rule="evenodd"')
    if st.stroke:
        parts.append(f'stroke="{st.stroke}" stroke-width="{paper(st.width)}" stroke-linecap="{st.cap}" stroke-linejoin="round"')
        if st.dash:
            parts.append(f'stroke-dasharray="{" ".join(paper(d) for d in st.dash)}"')
    else:
        parts.append('stroke="none"')
    return " ".join(parts)


def _d(subs: tuple[tuple[tuple[tuple[float, float], ...], bool], ...]) -> str:
    out = []
    for pts, closed in subs:
        out.append("M" + " L".join(f"{paper(x)} {paper(y)}" for x, y in pts) + (" Z" if closed else ""))
    return " ".join(out)


def render_svg(canvas: Canvas, title: str, description: str, metadata: dict[str, Any], *, locale: str = "en",
               metadata_id: str = SVG_METADATA_ID) -> bytes:
    width, height = canvas.width, canvas.height
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" xml:lang="{escape(locale)}" width="{paper(width)}mm" '
        f'height="{paper(height)}mm" viewBox="0 0 {paper(width)} {paper(height)}" font-family="{escape(FONT_FAMILY)}">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(description)}</desc>",
        f'<metadata id="{metadata_id}">{escape(json.dumps(metadata, sort_keys=True, ensure_ascii=True, separators=(",", ":")))}</metadata>',
        f'<rect x="0" y="0" width="{paper(width)}" height="{paper(height)}" fill="{PAPER}" stroke="none"/>',
    ]
    for op in canvas.ops:
        kind = op[0]
        if kind == "g":
            attrs = " ".join(f'{k}="{escape(v)}"' for k, v in op[1])
            lines.append(f"<g {attrs}>")
        elif kind == "/g":
            lines.append("</g>")
        elif kind == "path":
            lines.append(f'<path d="{_d(op[1])}" {_style(op[2])}/>')
        elif kind == "circle":
            (cx, cy), r, st = op[1], op[2], op[3]
            lines.append(f'<circle cx="{paper(cx)}" cy="{paper(cy)}" r="{paper(r)}" {_style(st)}/>')
        elif kind == "text":
            (x, y), s, size, anchor, bold, rotate, color = op[1:]
            attrs = f'x="{paper(x)}" y="{paper(y)}" font-size="{paper(size)}" fill="{color}"'
            if anchor != "start":
                attrs += f' text-anchor="{anchor}"'
            if bold:
                attrs += ' font-weight="bold"'
            if rotate:
                attrs += f' transform="rotate({paper(rotate)} {paper(x)} {paper(y)})"'
            lines.append(f"<text {attrs}>{escape(s)}</text>")
    lines.append("</svg>")
    return ("\n".join(lines) + "\n").encode("utf-8")
