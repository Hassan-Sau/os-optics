"""Regression tests for the semantic fix to ``_capture_margin`` (radial-direction extent, deterministic at the centre).

The frozen contract (``results.py`` ``ArrivalResult.capture_margin_mm``) defines the metric as the distance from the 1/e^2 beam
edge *along the radial direction* to the active-area edge.  A prior change globally replaced the radial extent with the ellipse
semi-major axis; these tests pin the correct radial semantics and the deterministic centred-beam convention.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics.config import NumericalConfig
from os_optics.optics.gaussian import initial_state
from os_optics.optics.rays import ArrivalRecord
from os_optics.scene.model import ApertureG
from os_optics.validation.observations import _capture_margin

CFG = NumericalConfig()

# Beam is deliberately astigmatic (wy >> wx) so radial-direction extent and semi-major extent differ clearly.
WX, WY = 0.5, 2.0
U = np.array([1.0, 0.0, 0.0])
V = np.array([0.0, 1.0, 0.0])


def _gstate(w0x: float = WX, w0y: float = WY) -> object:
    return initial_state(
        wavelength_nm=633.0, n=1.0, w0x=w0x, w0y=w0y, z_waist=0.0, m2x=1.0, m2y=1.0,
        bx=U.copy(), by=V.copy(), cfg=CFG,
    )


def _rec(gstate, uv: tuple[float, float]) -> ArrivalRecord:
    return ArrivalRecord(
        observation_id="obs", surface_id="surf", path_id="path", source_id="src", spectral_index=0,
        wavelength_nm=gstate.wavelength_nm, medium="vacuum", n=1.0,
        position=np.zeros(3), direction=np.array([0.0, 0.0, 1.0]), power_W=1e-3,
        opl_mm=0.0, length_mm=0.0, jones=np.array([1.0 + 0.0j, 0.0 + 0.0j]),
        e1=U.copy(), e2=V.copy(), gstate=gstate, uv=uv, aoi_deg=0.0,
        pupil_uv=np.zeros((0, 2)), pupil_total=0, coherence_group="grp", coherence_length_mm=None,
        lineage=(), plane_u=U.copy(), plane_v=V.copy(),
    )


def _semi_major_extent(gstate) -> float:
    """The (incorrect) semi-major-axis extent the prior implementation returned."""
    wx, wy = gstate.radius("x"), gstate.radius("y")
    bxu = np.array([gstate.bx @ U, gstate.bx @ V])
    byu = np.array([gstate.by @ U, gstate.by @ V])
    s = np.outer(bxu, bxu) / wx**2 + np.outer(byu, byu) / wy**2
    s_inv = np.linalg.inv(s)
    return math.sqrt(float(np.linalg.eigvalsh(s_inv)[1]))


def test_centered_beam_uses_deterministic_convention():
    """A centred astigmatic beam gives the +u radial extent regardless of last-bit position residue."""
    g = _gstate()
    ap = ApertureG("circle", 5.0, 0.0, 0.0, 0.0, 0.0)
    exact = _capture_margin(_rec(g, (0.0, 0.0)), ap, CFG)
    residue = _capture_margin(_rec(g, (1e-14, -1e-14)), ap, CFG)
    # Both land on n = +u (extent = wx); any noise-direction extent would be larger (up to wy).
    assert exact == pytest.approx(5.0 - WX, abs=1e-12)
    assert residue == pytest.approx(5.0 - WX, abs=1e-12)
    assert exact == pytest.approx(residue, abs=1e-13)


def test_near_center_noise_is_deterministic():
    """Offsets below aperture_edge_tol_mm in four different directions collapse to the identical centred value."""
    g = _gstate()
    ap = ApertureG("circle", 5.0, 0.0, 0.0, 0.0, 0.0)
    eps = 1e-12  # << CFG.aperture_edge_tol_mm (1e-9)
    margins = [_capture_margin(_rec(g, (du, dv)), ap, CFG) for du, dv in ((eps, 0.0), (-eps, 0.0), (0.0, eps), (0.0, -eps))]
    assert all(m == margins[0] for m in margins[1:]), "the four near-centre offsets must produce identical margins"
    assert margins[0] == pytest.approx(5.0 - (eps + WX), abs=1e-15)


def test_off_center_uses_radial_not_semi_major():
    """A genuinely off-centre anisotropic beam uses radial-direction extent, not the global semi-major axis."""
    g = _gstate()
    ap = ApertureG("circle", 10.0, 0.0, 0.0, 0.0, 0.0)
    d = 2.0  # >> aperture_edge_tol_mm
    got = _capture_margin(_rec(g, (d, 0.0)), ap, CFG)
    radial = 10.0 - (d + WX)          # n = +u, extent = wx = 0.5
    semi = 10.0 - (d + _semi_major_extent(g))  # extent = wy = 2.0
    assert got == pytest.approx(radial, abs=1e-12)
    assert abs(semi - got) > 1.0, "semi-major and radial extents must differ for this anisotropic beam"


def test_negative_margin_means_clipped():
    """A beam edge crossing the active-area edge still yields a negative capture margin."""
    g = _gstate()
    ap = ApertureG("circle", 2.0, 0.0, 0.0, 0.0, 0.0)
    # centre 1.9 mm off-axis with radial extent 0.5 mm -> edge at 2.4 mm crosses the 2.0 mm edge.
    got = _capture_margin(_rec(g, (1.9, 0.0)), ap, CFG)
    assert got == pytest.approx(2.0 - (1.9 + WX), abs=1e-12)
    assert got < 0.0
