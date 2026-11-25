"""Display colour of a beam from its vacuum wavelength. Pure presentation: no radiometry, no physics.

Visible light (380-750 nm) uses the common piecewise-linear spectrum approximation. Invisible light gets a fixed,
clearly artificial false colour so a near-infrared beam (for example the 780 nm reference) is still drawn.
"""

from __future__ import annotations

NEAR_INFRARED_FALSE_COLOUR = (0.85, 0.12, 0.55)
ULTRAVIOLET_FALSE_COLOUR = (0.45, 0.2, 0.95)
VISIBLE_MIN_NM = 380.0
VISIBLE_MAX_NM = 750.0
RGB_DECIMALS = 4
# (band start nm, band end nm, (r, g, b)); a channel is a constant, or "+" / "-" for a linear ramp 0 -> 1 / 1 -> 0 across
# the band. The last band also covers VISIBLE_MAX_NM itself.
SPECTRUM_BANDS = (
    (380.0, 440.0, ("-", 0.0, 1.0)),
    (440.0, 490.0, (0.0, "+", 1.0)),
    (490.0, 510.0, (0.0, 1.0, "-")),
    (510.0, 580.0, ("+", 1.0, 0.0)),
    (580.0, 645.0, (1.0, "-", 0.0)),
    (645.0, 750.0, (1.0, 0.0, 0.0)),
)


def _channel(spec: float | str, w: float, start: float, end: float) -> float:
    if spec == "+":
        return (w - start) / (end - start)
    if spec == "-":
        return (end - w) / (end - start)
    return float(spec)


def wavelength_display_rgb(wavelength_nm: float) -> tuple[float, float, float]:
    w = float(wavelength_nm)
    if w > VISIBLE_MAX_NM:
        return NEAR_INFRARED_FALSE_COLOUR
    if w < VISIBLE_MIN_NM:
        return ULTRAVIOLET_FALSE_COLOUR
    start, end, specs = next((band for band in SPECTRUM_BANDS if w < band[1]), SPECTRUM_BANDS[-1])
    r, g, b = (_channel(spec, w, start, end) for spec in specs)
    return (round(r, RGB_DECIMALS), round(g, RGB_DECIMALS), round(b, RGB_DECIMALS))
