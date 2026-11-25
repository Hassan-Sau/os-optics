"""The window bound (``validation/window_bound.py``): a lower bound of a surface's exposure that light of unknown place --
untraced light of unknown size and light left out as negligible, both bounded by their diffraction limit -- cannot undo.

Before it, the lower bound was taken at a traced beam's centre with that light there at its full diffraction-limited
amplitude in antiphase: in a multi-element system almost any ghost cancels the beam, so an exceedance of a rating far
below the traced exposure was UNKNOWN (the review's AR two-singlet expander, 100 x over its rating). The light of unknown
place has a small power, though: averaged against a window over the beam's area it takes away at most ``H sqrt(P)``,
whatever its directions. Checked here against explicit fields on a plane (the largest exposure on a fine grid is never
below the bound), with another coherent beam of known profile, at oblique incidence, and in the expander.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import PowerRating, ProjectSnapshot, Target
from os_optics.contracts.materials import ConstantCoating
from os_optics.validation.window_bound import Anchor, WindowBeam, surface_room, window_lower

from helpers import run

FIXTURE = "test fixture rating, not product data"
LAM_NM = 1064.0
K = 2.0 * math.pi / (LAM_NM * 1e-6)


def _anchor(w: float, theta: float, value: float) -> Anchor:
    """A round beam of 1/e^2 radius ``w`` meeting the plane z = 0 at ``theta`` (rad) from its normal, in air."""
    c, s = math.cos(theta), math.sin(theta)
    beam = WindowBeam(value, (0.0, 0.0, 0.0), (c, 0.0, -s), (0.0, 1.0, 0.0), w, w)
    return Anchor(beam, (0.0, 0.0, 1.0), c, 1.0, LAM_NM, 0.0, 0.0)


def _grid(half: float, n: int):
    x = np.linspace(-half, half, n)
    return np.meshgrid(x, x)


@pytest.mark.parametrize("theta_deg", [0.0, 35.0])
def test_light_of_unknown_place_cannot_undo_the_window_bound(theta_deg: float):
    """Explicit fields of known flux on a plane: a traced beam (peak V, transverse 1/e^2 radius w) and, in antiphase with
    its phase across the plane (the most that cancels it), a Gaussian of another size and place carrying ``eps`` of the
    beam's power across the plane (``flux = integral |E|^2 cos``), or tilted to fringes of 30 um and more. The largest
    exposure on a 3 um grid is never below the window bound computed with ``sqrt(flux)``; for a faint field it is close to
    the beam's peak (above 0.9 V at eps <= 1e-4), where the bound at the centre with the diffraction limit there gives 0."""
    theta = math.radians(theta_deg)
    rng = np.random.default_rng(11)
    for _ in range(12):
        w = float(10 ** rng.uniform(-1.0, 0.0))
        power = 1.0
        peak = 2.0 * power / (math.pi * w * w)  # W/mm2 (transverse)
        X, Y = _grid(3.0 * w / math.cos(theta), 801)
        k_t = K * math.sin(theta)
        traced = math.sqrt(peak) * np.exp(-((X * math.cos(theta)) ** 2 + Y ** 2) / (w * w)) * np.exp(1j * k_t * X)
        eps = float(10 ** rng.uniform(-6.0, -1.0))
        a, b = w * float(10 ** rng.uniform(-1.0, 0.7)), w * float(10 ** rng.uniform(-1.0, 0.7))
        cx, cy = (float(v) for v in rng.normal(0.0, 0.5 * w, 2))
        tilt = 0.0 if rng.uniform() < 0.6 else float(rng.uniform(-1.0, 1.0)) * 2.0 * math.pi / 0.03
        cos_u = math.sqrt(1.0 - ((k_t + tilt) / K) ** 2)
        shape = np.exp(-((X - cx) ** 2 / (a * a) + (Y - cy) ** 2 / (b * b)))
        flux = eps * power
        amplitude = math.sqrt(flux / (cos_u * math.pi * a * b / 2.0))
        unknown = -amplitude * shape * np.exp(1j * (k_t + tilt) * X)
        largest = float(np.max(np.abs(traced + unknown) ** 2)) * 100.0
        bound = window_lower(_anchor(w, theta, peak * 100.0), [], 0.0, math.sqrt(flux), 1.0, None, math.inf)
        assert bound <= largest * (1.0 + 1e-9), (w, eps, a, b, tilt)
        if eps <= 1e-4:
            assert bound > 0.9 * peak * 100.0


def test_another_coherent_beam_takes_away_at_most_its_overlap_with_the_window():
    """A second beam of known profile on the plane, in antiphase, offset by about a radius: the window bound subtracts its
    mean against the window (closed form, ``_overlap``), never more than the true cancellation allows -- the largest
    exposure on the grid stays above the bound -- and less than its full peak amplitude would."""
    w, peak = 0.5, 2.0 / (math.pi * 0.25)
    X, Y = _grid(2.5, 1001)
    traced = math.sqrt(peak) * np.exp(-(X ** 2 + Y ** 2) / (w * w))
    for offset, w_j, share in ((0.4, 0.3, 0.2), (0.0, 0.5, 0.5), (1.0, 0.2, 0.9)):
        other_peak = share * peak
        other = WindowBeam(other_peak * 100.0, (offset, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), w_j, w_j)
        field = traced - math.sqrt(other_peak) * np.exp(-((X - offset) ** 2 + Y ** 2) / (w_j * w_j))
        largest = float(np.max(np.abs(field) ** 2)) * 100.0
        bound = window_lower(_anchor(w, 0.0, peak * 100.0), [other], 0.0, 0.0, 1.0, None, math.inf)
        crude = (math.sqrt(peak) - math.sqrt(other_peak)) ** 2 * 100.0
        assert 0.0 < bound <= largest * (1.0 + 1e-9)
        if offset > 0.0:
            assert crude < bound  # (an offset beam takes away less than its peak amplitude over the window)


def test_the_room_to_the_clear_aperture_bounds_the_window():
    """``surface_room``: a plane's distance to its clear aperture's edge, a sphere's (with its radius) to its cap's edge
    across its axis; a beam near the edge gets a smaller bound than at the centre (what the window would reach beyond the
    edge counts as lost), one outside the clear aperture none."""
    from os_optics.config import NumericalConfig
    from os_optics.optics.materials.library import MaterialLibrary
    from os_optics.scene.lower import lower_scene

    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    lens = B.singlet_lens("L", r1_mm=50.0, r2_mm=float("inf"), thickness_mm=4.0, clear_radius_mm=10.0, material="N-BK7",
                          position=(100.0, 0.0, 0.0))
    src = B.laser("S", wavelength_nm=LAM_NM, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    cfg = NumericalConfig()
    scene, _ = lower_scene(ProjectSnapshot(components=(src, window, lens)), cfg, MaterialLibrary((), cfg))
    room = surface_room(scene)
    plane = next(s for s in scene.surfaces if s.owner_id == "W" and s.geometry == "plane")
    point = np.array(plane.origin) + 3.0 * np.array(plane.v_axis)
    assert room(plane.surface_id, point) == (None, pytest.approx(9.7, rel=1e-12), None)
    sphere = next(s for s in scene.surfaces if s.owner_id == "L" and s.geometry == "sphere")
    radius, gap, spread = room(sphere.surface_id, np.array(sphere.center) + sphere.radius * np.array(sphere.cap_axis))
    assert radius == pytest.approx(50.0, rel=1e-12) and gap == pytest.approx(10.0, rel=1e-12)
    # (the normals within a 10 mm cap of a 50 mm sphere: twice the chord to the rim's, 2 sqrt(2 (1 - cos(asin(0.2)))))
    assert spread == pytest.approx(2.0 * math.sqrt(2.0 * (1.0 - math.cos(math.asin(0.2)))), rel=1e-12)
    anchor = _anchor(0.5, 0.0, 100.0)
    centre = window_lower(anchor, [], 0.0, 1e-3, 1.5, None, 9.7)
    edge = window_lower(anchor, [], 0.0, 1e-3, 1.5, None, 0.2)
    assert 0.0 < edge < centre and window_lower(anchor, [], 0.0, 1e-3, 1.5, None, -0.1) == 0.0
    curved = window_lower(anchor, [], 0.0, 1e-3, 1.5, 50.0, 9.7)
    assert 0.0 < curved < centre  # (a sphere's sag spreads the window's overlap with the plane waves)
    # (on a sphere the second hemisphere of plane waves: in full without the aperture's spread, else only near grazing)
    assert curved < window_lower(anchor, [], 0.0, 1e-3, 1.5, 50.0, 9.7, spread) < centre


def _expander(rating: float, unit: str = "W/cm2", test_beam_mm: float | None = None) -> ProjectSnapshot:
    """The review's case (``review_physics/p8_old_vs_new.py``): a 1064 nm, 1 W beam expanded by two AR-coated (0.25 %)
    N-BK7 plano-convex singlets, the second one rated (a W/cm2 value stated as a peak irradiance, ``value_definition``
    PEAK: its threshold is the value). ``test_beam_mm``: the rating's test beam (1/e^2), for linear light (the source's)."""
    ar = ConstantCoating(reflectance_s=0.0025, reflectance_p=0.0025, transmittance_s=0.9975, transmittance_p=0.9975)

    def coat(c):
        return c.model_copy(update={"surfaces": tuple(s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": ar})})
                                                      for s in c.surfaces)})

    src = B.laser("S", wavelength_nm=LAM_NM, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    l1 = coat(B.singlet_lens("L1", r1_mm=25.8, r2_mm=float("inf"), thickness_mm=4.0, clear_radius_mm=11.0, material="N-BK7",
                             position=(0.0, 0.0, 0.0)))
    l2 = coat(B.singlet_lens("L2", r1_mm=51.5, r2_mm=float("inf"), thickness_mm=4.0, clear_radius_mm=11.0, material="N-BK7",
                             position=(150.0, 0.0, 0.0)))
    extra = {"value_definition": "PEAK"} if unit != "W/cm" else {"beam_diameter_definition": "1/E2"}
    if test_beam_mm is not None:
        extra.update(beam_diameter_mm=test_beam_mm, beam_diameter_definition="1/E2", polarization="LINEAR",
                     polarization_condition="linear")
    l2 = l2.model_copy(update={"power_ratings": (PowerRating(regime="CW", value=rating, unit=unit, wavelength_nm=LAM_NM,
                                                               status="DECLARED", basis=FIXTURE, **extra),)})
    dump = B.beam_dump("D", position=(400.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=20.0)
    return ProjectSnapshot(components=(src, l1, l2, dump),
                           targets=(Target(target_id="u", device_id="L2", metric="power_rating_utilization", maximum=1.0),))


def test_an_ar_coated_expander_far_over_its_rating_exceeds_it():
    """The review's reproducer: the second singlet rated at 1 % and at 50 % of its traced peak. Its faces meet ghosts of
    unknown size (aberration-dominated, at their diffraction limit, thousands of W/cm2 each) and the light left out as
    negligible: at a beam's centre they could cancel it, so the bound there is 0 -- but over the beam's area they can take
    away little: both faces EXCEED (the plane back face and the curved front face), the notice is raised and the target
    FAILs at 1 / 0.01 and 1 / 0.5 of the traced utilization's lower end. The rating states a test beam a quarter of the
    traced beam's 1/e^2 diameter -- smaller than every beam on the faces, the ghosts included, so its damage threshold is
    at most the rated value whatever light of unknown size adds -- and linear polarization."""
    probe = run(_expander(1e12))
    traced = {s.surface_id: s.hits[0].peak_irradiance_W_per_cm2 for s in probe.exposures.surfaces if s.element_id == "L2"}
    largest = max(traced.values())
    smaller = 0.25 * min(2.0 * s.hits[0].beam_radius_x_mm for s in probe.exposures.surfaces if s.element_id == "L2")
    assert any(h.peak_irradiance_W_per_cm2 is None for s in probe.exposures.surfaces if s.element_id == "L2" for h in s.untraced_hits)
    for k in (0.01, 0.5):
        result = run(_expander(k * largest, test_beam_mm=smaller))
        for surface in (s for s in result.exposures.surfaces if s.element_id == "L2"):
            (comparison,) = surface.ratings
            assert (comparison.applicability, comparison.status) == ("THRESHOLD_AT_MOST_RATED", "EXCEEDS_RATING"), (k, surface.surface_id)
            assert 0.8 * traced[surface.surface_id] < comparison.exposure_lower_bound < traced[surface.surface_id]
        (target,) = result.targets
        assert target.status.value == "FAIL" and "POWER_RATING_EXCEEDED" in target.issue_codes
        assert target.value > 0.8 / k
        assert sum(1 for i in result.issues if i.code.value == "POWER_RATING_EXCEEDED") == 2


def test_the_p_over_d_window_lifts_the_expanders_w_per_cm_bounds(monkeypatch):
    """``P / d`` (W/cm) has a window bound of its own (``linear_lower``): light of unknown place counts as Gaussian beams of
    its power crossing the surface from one side, and such a beam's ``P / d`` covers at most a strip of the window. On the
    expander's second singlet rated in W/cm at 1 % and 50 % of its traced ``P / d`` the lower bounds lie above 0.6 of the
    traced values -- above those at the beams' centres alone (``linear_lower`` switched off) -- and the irradiance bounds
    still come from the field window. The rating states no test beam or polarization: a W/cm rating has no established
    relationship for a beam larger than its test beam, and no one test beam is within 5 % of every beam on these faces
    (their ghosts), so none would let it decide an exceedance; it is not verified, the target not decided."""
    import os_optics.validation.exposures as exposures

    probe = run(_expander(1e12, "W/cm"))
    traced = {s.surface_id: s.hits[0].linear_power_density_W_per_cm for s in probe.exposures.surfaces if s.element_id == "L2"}

    def bounds(value: float) -> dict[str, float | None]:
        result = run(_expander(value, "W/cm"))
        assert result.targets[0].status.value == "UNKNOWN"
        return {s.surface_id: (s.ratings[0].status, s.ratings[0].exposure_lower_bound) for s in result.exposures.surfaces
                if s.element_id == "L2"}

    for k in (0.01, 0.5):
        found = bounds(k * max(traced.values()))
        for surface_id, (status, lower) in found.items():
            assert status == "CONDITIONS_NOT_VERIFIED" and 0.6 * traced[surface_id] < lower < traced[surface_id], (k, surface_id)
    with_window = bounds(0.01 * max(traced.values()))
    monkeypatch.setattr(exposures, "linear_lower", lambda *args: 0.0)
    without = bounds(0.01 * max(traced.values()))
    assert all(without[i][1] < with_window[i][1] for i in traced)


# ------------------------------------------------------------------ P / d: light of unknown place against the strip bound

KAPPA = 5.0  # W/cm of 1 W over a narrower 1/e^2 radius of 1 mm (P / (2 w))
WAIST = LAM_NM * 1e-6 / (math.pi * 1.5)  # mm, the diffraction-limited waist at index 1.5


def _piece_axes(n0: np.ndarray, cp: float, azimuth: float, turn: float):
    """A direction ``d`` with ``d . n0 = cp`` and its transverse axes, the narrow one turned by ``turn`` from the
    surface's (s) direction toward the plane of incidence (``pi / 2``: the most toward the normal)."""
    t = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
    d = cp * n0 + math.sqrt(1.0 - cp * cp) * t
    a = np.cross(d, t)
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    b1 = math.cos(turn) * a + math.sin(turn) * b
    return d, b1, np.cross(d, b1)


def _region(sphere: float | None, frame, s_index: int):
    """The window's region on a grid: tangent-plane points, chi, the surface's points and normals there."""
    from os_optics.validation import window_bound as wb

    a = wb._S[s_index] * frame.low
    distance = frame.distance if sphere is None else min(frame.distance, wb.LINEAR_REACH / math.sqrt(a))
    reach = min(distance, 6.0 / math.sqrt(a))
    x = np.linspace(-reach, reach, 701)
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = frame.form
    inside = X * X + Y * Y <= distance * distance
    chi = np.exp(-wb._S[s_index] * (f[0] * X * X + 2.0 * f[1] * X * Y + f[2] * Y * Y)) * inside
    e1, e2, n0 = np.array(frame.e1), np.array(frame.e2), np.array(frame.n0)
    points = X[..., None] * e1 + Y[..., None] * e2
    normals = np.broadcast_to(n0, points.shape).copy()
    if sphere is not None:
        sag = sphere - np.sqrt(sphere * sphere - X * X - Y * Y)
        points = points + sag[..., None] * n0
        normals = (sphere * n0 - points) / sphere
    return chi, points, normals[inside], (x[1] - x[0]) ** 2, reach


@pytest.mark.parametrize("sphere", [None, 30.0, 5.0], ids=["plane", "sphere30", "sphere5"])
def test_light_crossing_from_one_side_cannot_undo_the_p_over_d_window(sphere: float | None):
    """Pieces of 1 W of light of unknown place, as the ``P / d`` window counts them: Gaussian beams no more focused than the
    diffraction limit (narrower radius ``w1``, far-field slopes ``WAIST / w``) whose 1/e^2 far-field cone lies on one side
    of the surface at every point of the window's region -- thin sheets at the edge of that, nearly grazing with the narrow
    axis toward the normal; line foci; turned axes -- with the ``P / d`` amplitude ``sqrt(kappa / w1) exp(-rho^2)`` (the
    Engine's profile). On a 701 x 701 grid their ``integral chi |u|`` over the region stays below the bound the window
    subtracts per watt (``_unknown_linear``), on a plane and on spheres of 30 mm and 5 mm."""
    from os_optics.validation import window_bound as wb

    rng = np.random.default_rng(5)
    n0 = np.array([0.0, 0.0, 1.0])
    beam = WindowBeam(1.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5, 0.35)
    frame = wb._frame(beam, tuple(n0), [], sphere, 5.0)
    bound, _ = wb._unknown_linear(frame, KAPPA, WAIST, sphere)
    worst = 0.0
    for k in (6, 14, 22):
        chi, points, normals, cell, reach = _region(sphere, frame, k)
        normals = normals[:: max(1, len(normals) // 300)]
        tried = 0
        while tried < 18:
            w1 = WAIST * float(np.exp(rng.uniform(0.0, 5.5)))
            w2 = w1 * float(np.exp(rng.uniform(0.0, 7.0)))
            th1, th2 = WAIST / w1, WAIST / w2
            kind = tried % 3
            turn = (math.pi / 2.0, 0.0, float(rng.uniform(0.0, math.pi)))[kind]
            cp = th1 / math.sqrt(1.0 + th1 * th1) * float(rng.uniform(1.0, 1.5)) if kind != 1 else float(rng.uniform(0.3, 1.0))
            d, b1, b2 = _piece_axes(n0, min(cp, 0.999), float(rng.uniform(0.0, 2.0 * math.pi)), turn)
            if not np.all(normals @ d > np.hypot(th1 * (normals @ b1), th2 * (normals @ b2))):
                continue  # (its cone not on one side of the surface everywhere in the region)
            tried += 1
            centre = np.array([rng.uniform(-0.5, 0.5) * reach, rng.uniform(-0.5, 0.5) * reach, 0.0])
            if sphere is not None:
                centre[2] = sphere - math.sqrt(sphere * sphere - centre[0] ** 2 - centre[1] ** 2)
            rel = points - centre
            u = math.sqrt(KAPPA / w1) * np.exp(-((rel @ b1) ** 2 / (w1 * w1) + (rel @ b2) ** 2 / (w2 * w2)))
            worst = max(worst, float(np.sum(chi * u) * cell) / float(bound[k]))
    assert 0.05 < worst < 1.0


def test_a_shallow_thin_sheet_covers_a_strip_a_line_focus_a_line():
    """Why ``P / d`` gains less from the window than the irradiance: ``P / d`` is a beam's power over its narrower
    transverse diameter, so a thin sheet of light meeting the surface at a shallow angle keeps it along a strip of the
    surface about ``w^2 / waist`` wide -- here 30 % and more of what the window bound allows for -- while the same sheet
    at normal incidence (a line focus) covers a line ``w`` wide, a few per cent of it."""
    from os_optics.validation import window_bound as wb

    n0 = np.array([0.0, 0.0, 1.0])
    beam = WindowBeam(1.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5, 0.5)
    frame = wb._frame(beam, tuple(n0), [], None, 5.0)
    bound, _ = wb._unknown_linear(frame, KAPPA, WAIST, None)
    k = 14
    chi, points, _, cell, _ = _region(None, frame, k)
    w1 = math.sqrt(WAIST * math.sqrt(math.pi / (wb._S[k] * frame.low)))  # (about where the strip covers the window)
    th1 = WAIST / w1
    shares = []
    for cp, turn in ((1.01 * th1 / math.sqrt(1.0 + th1 * th1), math.pi / 2.0), (0.999, 0.0)):
        d, b1, b2 = _piece_axes(n0, cp, 0.0, turn)
        u = math.sqrt(KAPPA / w1) * np.exp(-((points @ b1) ** 2 / (w1 * w1) + (points @ b2) ** 2 / (100.0 * w1) ** 2))
        shares.append(float(np.sum(chi * u) * cell) / float(bound[k]))
    sheet, line = shares
    assert sheet > 0.3 and line < 0.05
