"""Light the trace leaves out returned between two surfaces facing each other (``validation/resonance.py``), and the model's
scope.

Two reflecting surfaces that face each other -- a window's or a lens's faces, a window and a mirror, the mirrors of a
linear cavity, also with a lens between them -- return the left-out light to themselves and to every surface between
them: each round trip keeps at most ``g = r_a r_b`` of its field, so all its returns bring ``g / (1 - g)`` times one
crossing more. What each return sends out of the pair is at most that return's field and may reach any surface, so every
surface and plane has ``f = 1 + sum g / (1 - g)`` over the scene's pairs, from either side. The reflectances are bounded
over every angle at which the pair's light can meet them: along the legs between the surfaces' apertures, between
parallel planes from the momentum along them that the rest of the scene brings in; from an interface's denser side only
below the critical angle (a range reaching it counts 1); a coating met outside its data counts 1. Light circulating
through three or more reflecting surfaces (a ring, a folded cavity, a periscope's windows through its folds, a stack of
plates) and light held by total internal reflection are outside the model.

The verifier's round-6 and round-7 findings each have a test here: the medium-name rule that dropped a cavity's arrivals
(a source in 'vacuum', mirrors in 'air'), the thin lens's deflection (the slope-form lens turns light by up to 2 atan(rho
/ 2f), not atan(rho / f); rho measured from the lens's axis, a decentred aperture's offset included) and the sides light
after a lens meets, a lens that turns a cavity's light far off the line between its mirrors, a coating clipped to its
data, the cone of directions a refraction at a plane widens, the light a cavity sends out through its mirror, and the
returns between parallel planes of different apertures. Hand values are written out; every power rating is fixture data
declared in the test project, never product data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import PowerRating, ProjectSnapshot, Target
from os_optics.contracts.materials import ConstantCoating, TabulatedCoating
from os_optics.contracts import EngineeringComponentSnapshot
from os_optics.contracts.optics import CircularAperture, PlaneSurface, RefractInteraction
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.optics.surfaces.interaction import thin_lens_dir
from os_optics.scene.lower import lower_scene
from os_optics.validation import resonance

from helpers import run
from test_group_delay_and_returns import _bk7, window_returns

FIX = "verifier fixture rating, not product data"
LAM = 1064.0
LAM_MM = LAM * 1e-6


def _rating(value: float, wavelength_nm: float = LAM) -> PowerRating:
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=wavelength_nm, status="DECLARED", basis=FIX)


def _constant(r: float, max_aoi_deg: float = 90.0) -> ConstantCoating:
    return ConstantCoating(reflectance_s=r, reflectance_p=r, transmittance_s=1.0 - r, transmittance_p=1.0 - r, max_aoi_deg=max_aoi_deg)


def _coat(component, coating, both: bool = False):
    surfaces = tuple(s.model_copy(update={"interaction": s.interaction.model_copy(
        update={"coating": coating, **({"reflective_side": "both"} if both else {})})}) for s in component.surfaces)
    return component.model_copy(update={"surfaces": surfaces})


def _surface(result, surface_id: str):
    return next(s for s in result.exposures.surfaces if s.surface_id == surface_id)


def _targets(*devices: str) -> tuple[Target, ...]:
    return tuple(Target(target_id=d.lower(), device_id=d, metric="power_rating_utilization", maximum=1.0) for d in devices)


def _scene(project: ProjectSnapshot):
    cfg = NumericalConfig()
    library = MaterialLibrary(project.materials, cfg)
    scene, _ = lower_scene(project, cfg, library)
    return scene, library


def _gains(project: ProjectSnapshot, wavelength_nm: float = LAM) -> resonance.ReturnGains:
    scene, library = _scene(project)
    return resonance.return_gains(scene, library, [wavelength_nm])[wavelength_nm]


def _peak_limit(power_w: float, wavelength_nm: float = LAM, n: float = 1.0) -> float:
    """The diffraction limit 2 pi n^2 P / lambda^2 of light of power ``power_w`` (W/cm2)."""
    lam_mm = wavelength_nm * 1e-6
    return 2.0 * math.pi * n * n * power_w / (lam_mm * lam_mm) * 100.0


# ------------------------------------------------------------------------------------------------ a window and a cavity


def test_a_window_s_faces_return_the_left_out_light_as_a_pair():
    """An uncoated window: its faces face each other across the glass, and the light between them keeps the momentum along
    them the source's beam brings (8 far-field divergences): its returns bring at most R / (1 - R) more, R the Fresnel s
    reflectance from the glass at the widest internal angle, and leave the window on either side: each face's factor is
    1 / (1 - R) (``window_returns``)."""
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    window = window.model_copy(update={"power_ratings": (_rating(1e9),)})
    result = run(ProjectSnapshot(components=(src, window), targets=_targets("W")))
    for face in ("W/front", "W/back"):
        surface = _surface(result, face)
        assert surface.faint_light_return_factor == pytest.approx(window_returns(_bk7(1.064), LAM, 0.5), rel=1e-12)
        assert "FAINT_LIGHT_RECIRCULATES" in surface.reason_codes
    (pair,) = _gains(ProjectSnapshot(components=(src, window))).pairs
    assert set(pair.surfaces) == {"W/front", "W/back"} and pair.crossed == ()


def _cavity(rating: float, *, medium: str = "air", coating=None, coupler: float = 1e-5):
    """M1 leaks 5e-10 of a 1 W, 1064 nm beam (w0 0.5
    mm; left out), a thin lens mode-matches the leak into a hemispherical cavity -- flat C1 (clear radius 5 mm), concave C2
    (radius of curvature 10.1 mm, clear radius 3 mm) 10 mm apart, both reflecting on either side with ``coating`` (default
    R = 1 - ``coupler``). ``medium``: the source's declared medium (the mirrors keep the builders' air)."""
    w_mode = math.sqrt(LAM_MM / math.pi * math.sqrt(10.0 * (10.1 - 10.0)))
    w_lens = 0.5 * math.sqrt(1.0 + (150.0 / (math.pi * 0.25 / LAM_MM)) ** 2)
    f = w_mode * math.pi * w_lens / LAM_MM
    coating = coating or _constant(1.0 - coupler)
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    src = src.model_copy(update={"source": src.source.model_copy(update={"medium": medium})})
    m1 = _coat(B.mirror_for_reflection("M1", position=(100.0, 0.0, 0.0), d_in=(1, 0, 0), d_out=(0, 1, 0), clear_radius_mm=12.7),
               _constant(1.0 - 5e-10))
    dump = B.beam_dump("D", position=(100.0, 100.0, 0.0), axis=(0.0, 1.0, 0.0), radius_mm=12.7)
    lens = B.thin_lens("L", focal_length_mm=f, clear_radius_mm=12.7, position=(150.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0))
    c1 = _coat(B.mirror("C1", position=(150.0 + f, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear_radius_mm=5.0), coating, both=True)
    c2 = _coat(B.spherical_mirror("C2", position=(160.0 + f, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_of_curvature_mm=10.1,
                                  clear_radius_mm=3.0), coating, both=True)
    c1, c2 = (x.model_copy(update={"power_ratings": (_rating(rating),)}) for x in (c1, c2))
    return ProjectSnapshot(components=(src, m1, dump, lens, c1, c2), targets=_targets("C1", "C2"))


def _stored_peak() -> float:
    """The cavity on resonance (the verifier's hand model): the leak's q at C1 against the mode (waist at the flat C1),
    the mode match eta, the circulating power eta P T / (1 - R)^2 and its peak 2 P / (pi w0^2) on C1 (W/cm2)."""
    w_mode = math.sqrt(LAM_MM / math.pi * math.sqrt(10.0 * (10.1 - 10.0)))
    w_lens = 0.5 * math.sqrt(1.0 + (150.0 / (math.pi * 0.25 / LAM_MM)) ** 2)
    f = w_mode * math.pi * w_lens / LAM_MM
    q = complex(150.0, math.pi * 0.25 / LAM_MM)
    q = q / (-q / f + 1.0) + f
    inv = 1.0 / q
    w_in = math.sqrt(-LAM_MM / (math.pi * inv.imag))
    eta = 4.0 / ((w_in / w_mode + w_mode / w_in) ** 2 + (math.pi * w_in * w_mode / LAM_MM) ** 2 * inv.real ** 2)
    p_circ = eta * 5e-10 * 1e-5 / 1e-5 ** 2
    return 2.0 * p_circ / (math.pi * w_mode ** 2) * 100.0


def test_a_cavity_returns_its_round_trips_and_holds_its_stored_light():
    """C1 and C2 face each other along their common axis: g = sqrt(R_C1 R_C2) = 1 - 1e-5, so every surface gets 1 / (1 -
    g) = 1e5 from either side (the cavity's returns leave it through its mirrors and past their edges). The light stored
    on resonance (9.3 W/cm2 on C1) lies within C1's bound, and the verifier's rating of 2.22 W/cm2 is not within it. (A
    rating of 1e12 W/cm2 decides on the first pass; one the first pass does not decide makes the run follow the left-out
    light deeper, into the cavity, where it meets the interaction limit and bounds nothing.)"""
    project = _cavity(1e12)
    gains = _gains(project)
    g = 1.0 - 1e-5
    (pair,) = gains.pairs
    assert set(pair.surfaces) == {"C1/face", "C2/face"} and pair.gain == pytest.approx(g, rel=1e-12)
    assert pair.returns == pytest.approx(g / (1.0 - g), rel=1e-9)
    for surface in ("C1/face", "C2/face", "L/lens", "M1/face", "D/dump"):
        assert gains.sides(surface) == pytest.approx((1.0 / (1.0 - g),) * 2, rel=1e-9)
    result = run(project)
    peak = _stored_peak()
    assert 7.0 < peak < 10.0
    assert _surface(result, "C1/face").ratings[0].exposure_upper_bound >= peak
    verdict = run(_cavity(2.22))
    assert all(t.status.value != "PASS" for t in verdict.targets)


def test_a_cavity_s_returns_do_not_depend_on_the_medium_s_name():
    """With the source declared in 'vacuum' and the mirrors in the builders' 'air' (both
    n = 1), an earlier version formed no arrival in the cavity -- a medium-name rule -- and a rating of 2.22 W/cm2 PASSed against
    9.3 W/cm2 stored. The screen compares indices, and only at an interface (which takes light in its own medium): the
    same factors as with the source in air, and the rating is not decided."""
    by_air, by_vacuum = _gains(_cavity(1e9)), _gains(_cavity(1e9, medium="vacuum"))
    assert by_vacuum.by_surface == by_air.by_surface and by_air.by_surface
    verdict = run(_cavity(2.22, medium="vacuum"))
    assert all(t.status.value != "PASS" for t in verdict.targets)


def _cavity_output(rating: float, obs_maximum: float, transmittance: float = 5e-10):
    """A 1 W, 1064 nm beam (w0 0.5 mm) mode-matched by a thin lens into a hemispherical
    cavity -- flat C1, concave C2 (10.1 mm), 10 mm apart, both transmitting ``transmittance`` -- whose transmission through
    C1 (5e-10 W) is left out; an observation plane OBS 20 mm and a window W (N-BK7, 3 mm, rated ``rating``) 50 mm behind
    C2. On resonance the symmetric cavity sends nearly all of the input out through C2: 0.993 W across OBS, 51.8 W/cm2 on
    W's front face."""
    w_mode = math.sqrt(LAM_MM / math.pi * math.sqrt(10.0 * (10.1 - 10.0)))
    w_lens = 0.5 * math.sqrt(1.0 + (150.0 / (math.pi * 0.25 / LAM_MM)) ** 2)
    f = w_mode * math.pi * w_lens / LAM_MM
    coating = _constant(1.0 - transmittance)
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    lens = B.thin_lens("L", focal_length_mm=f, clear_radius_mm=12.7, position=(150.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0))
    c1 = _coat(B.mirror("C1", position=(150.0 + f, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear_radius_mm=5.0), coating, both=True)
    c2 = _coat(B.spherical_mirror("C2", position=(160.0 + f, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_of_curvature_mm=10.1,
                                  clear_radius_mm=3.0), coating, both=True)
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(210.0 + f, 0.0, 0.0))
    window = window.model_copy(update={"power_ratings": (_rating(rating),)})
    obs = B.observation_plane("OBS", position=(180.0 + f, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=12.7)
    targets = (*_targets("W"), Target(target_id="obs", observation_id="OBS", metric="coherent_power_W", maximum=obs_maximum))
    return ProjectSnapshot(components=(src, lens, c1, c2, window), observations=(obs,), targets=targets)


def test_the_light_a_cavity_sends_out_through_its_mirror_reaches_the_surfaces_behind_it():
    """An earlier version raised the left-out light only on a pair's own surfaces and those between
    them, so a plane and a window behind C2 counted the cavity's transmission once: OBS PASSed a coherent power of at most
    1e-6 W (bound 8.8e-9 W) and W a rating of 17.4 W/cm2, while on resonance the cavity sends 0.993 W out through C2. Every
    return's light that leaves the pair is at most that return's field: every surface and plane behind it gets the
    cavity's factor 1 / (1 - g), and neither target is decided."""
    project = _cavity_output(1e9, 1.0)
    gains = _gains(project)
    g = 1.0 - 5e-10
    pair = next(p for p in gains.pairs if set(p.surfaces) == {"C1/face", "C2/face"})
    assert pair.gain == pytest.approx(g, rel=1e-12)
    for surface in ("W/front", "W/back", "observation/OBS"):
        assert min(gains.sides(surface)) >= 1.0 / (1.0 - g) * (1.0 - 1e-6)
    verdict = run(_cavity_output(17.4, 1e-6))
    assert all(t.status.value != "PASS" for t in verdict.targets)


@pytest.mark.parametrize("variant", ["within", "beyond", "table"])
def test_a_coating_met_outside_its_data_bounds_nothing(variant: str):
    """The cavity's mirrors coated R 0.9 / T 0.1. The pair's light meets them at up to 62
    and 80 deg (the angle between the apertures and C2's normals). With data for 0-90 deg the pair's gain is 0.9 and the
    factors 1 / (1 - 0.9) = 10 on the inner sides; with data only up to 3 deg (``beyond``), or a table that agrees there
    and reflects everything further on (``table``: a completion of the same data), the coating counts as reflecting
    everything: g = 1, no upper bound (an earlier version clipped the range to the data and took 0.9: a rating of 121 W/cm2 PASSed for
    the 3-deg coating)."""
    if variant == "table":
        aoi = (0.0, 3.0, 3.5, 90.0)
        r = ((0.9, 0.9, 1.0, 1.0),) * 2
        t = ((0.1, 0.1, 0.0, 0.0),) * 2
        coating = TabulatedCoating(wavelength_nm=(1000.0, 1100.0), aoi_deg=aoi, reflectance_s=r, reflectance_p=r,
                                   transmittance_s=t, transmittance_p=t)
    else:
        coating = _constant(0.9, 90.0 if variant == "within" else 3.0)
    project = _cavity(1e9, coating=coating)
    gains = _gains(project)
    (pair,) = gains.pairs
    (low1, high1), (low2, high2) = pair.angles_deg
    assert high1 > 60.0 and high2 > 75.0
    result = run(project)
    upper = _surface(result, "C1/face").ratings[0].exposure_upper_bound
    if variant == "within":
        assert pair.gain == pytest.approx(0.9, rel=1e-12)
        assert gains.sides("C1/face") == pytest.approx((10.0, 10.0), rel=1e-12)
        assert upper is not None
    else:
        assert pair.gain == 1.0 and math.inf in gains.sides("C1/face") and math.inf in gains.sides("M1/face")
        assert upper is None
        assert all(t.status.value != "PASS" for t in run(_cavity(121.3, coating=coating)).targets)


# ------------------------------------------------------------------------------------------------ thin lenses


def test_a_thin_lens_turns_light_by_at_most_its_slope_form_maximum():
    """The Engine's thin lens (``thin_lens_dir``, slope form s' = s - h / f) turns a ray met at
    height h by atan(s) - atan(s - h / f), largest, 2 atan(h / 2f), at s = h / 2f; an earlier version took atan(rho / f) (up to 34 deg
    short for a fast lens). The screen's bound is the largest deflection over the aperture, and no ray of the Engine's lens
    turns further."""
    class _Lens:
        def __init__(self, f):
            self.interaction = type("I", (), {"focal_length": f, "table": None})()

    for rho, f in ((12.7, 100.0), (12.7, 12.7), (12.7, 6.35), (5.0, 2.0)):
        bend = resonance._lens_bend(_Lens(f), LAM, rho)  # type: ignore[arg-type]
        assert bend == pytest.approx(2.0 * math.atan(rho / (2.0 * f)), rel=1e-12)
        slopes = np.linspace(-5.0, 5.0, 20001)
        d = np.stack([slopes, np.zeros_like(slopes), np.ones_like(slopes)], axis=1)
        d = d / np.sqrt((d * d).sum(axis=1))[:, None]
        out, _ = thin_lens_dir(np.tile(np.array([rho, 0.0, 0.0]), (len(slopes), 1)), np.zeros(3), np.array([0.0, 0.0, 1.0]), d, f, 1e-12)
        largest = float(np.arccos(np.clip((d * out).sum(axis=1), -1.0, 1.0)).max())
        assert largest <= bend * (1.0 + 1e-9)


def _fast_lens_cavity(rating: float):
    """M1 leaks 5e-10 of a 1 W, 1064 nm beam along +x into a cavity: E (flat, x = -200,
    R 1 - 1e-6 on either side) -- 200 mm -- an ideal thin lens L (f 4 mm, clear radius 12.7 mm) met at 0.8 of its radius,
    its axis turned so that the lens turns the ray by 2 atan(1.27) = 103.6 deg -- 6.48 mm -- B (flat, clear radius 2 mm,
    R 1 - 1e-6 on either side), normal to that ray. The lens straddles B's plane."""
    f, rho, h = 4.0, 12.7, 0.8 * 12.7
    s0 = h / (2.0 * f)
    phi = -math.atan(s0)
    axis = (math.cos(phi), math.sin(phi), 0.0)
    eh = (-math.sin(phi), math.cos(phi), 0.0)
    centre = (-h * eh[0], -h * eh[1], 0.0)
    sp = s0 - h / f
    d = (axis[0] + sp * eh[0], axis[1] + sp * eh[1], 0.0)
    norm = math.hypot(d[0], d[1])
    out = (d[0] / norm, d[1] / norm, 0.0)
    l2 = 6.478718455497198
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-400.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    m1 = _coat(B.mirror_for_reflection("M1", position=(-350.0, 0.0, 0.0), d_in=(1, 0, 0), d_out=(0, 1, 0), clear_radius_mm=12.7),
               _constant(1.0 - 5e-10), both=True)
    dump = B.beam_dump("D", position=(-350.0, 100.0, 0.0), axis=(0.0, 1.0, 0.0), radius_mm=12.7)
    e = _coat(B.mirror("E", position=(-200.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear_radius_mm=12.7), _constant(1.0 - 1e-6), both=True)
    lens = B.thin_lens("L", focal_length_mm=f, clear_radius_mm=rho, position=centre, axis=axis)
    b = _coat(B.mirror("B", position=(l2 * out[0], l2 * out[1], 0.0), normal=(-out[0], -out[1], 0.0), clear_radius_mm=2.0),
              _constant(1.0 - 1e-6), both=True)
    e, b = (x.model_copy(update={"power_ratings": (_rating(rating),)}) for x in (e, b))
    return ProjectSnapshot(components=(src, m1, dump, e, lens, b), targets=_targets("E", "B"))


def test_a_cavity_closed_through_a_fast_thin_lens_is_a_pair_met_from_either_side():
    """E and B face each other through the lens: B's light arrives along the leg from the lens (every direction from the
    lens's aperture to B's), and the lens straddles B's plane, so light after it may meet either side of B: both of B's
    sides get 1 / (1 - g), g = sqrt(R_E R_B) = 1 - 1e-6, and so do both sides of the lens between them. An earlier version widened
    the legs by atan(rho / f) and took B's side from E: a rating 7.6e8 times below the stored light PASSed. The bound
    holds the light such a cavity can store on resonance: the leak's power times E's transmittance over (1 - g)^2,
    focused to its diffraction limit."""
    project = _fast_lens_cavity(1e9)
    gains = _gains(project)
    g = 1.0 - 1e-6
    (pair,) = gains.pairs
    assert set(pair.surfaces) == {"E/face", "B/face"} and pair.gain == pytest.approx(g, rel=1e-12) and pair.crossed == ("L/lens",)
    assert gains.sides("B/face") == pytest.approx((1.0 / (1.0 - g),) * 2, rel=1e-9)
    assert gains.sides("L/lens") == pytest.approx((1.0 / (1.0 - g),) * 2, rel=1e-9)
    result = run(project)
    upper = _surface(result, "B/face").ratings[0].exposure_upper_bound
    stored = 5e-10 * 1e-6 / (1.0 - g) ** 2
    assert upper is not None and upper >= _peak_limit(stored)


def _decentred(project: ProjectSnapshot) -> ProjectSnapshot:
    """The fast-lens cavity with the lens's clear aperture a 3 mm circle centred on the beam's hit point, 10.16 mm off the
    lens's axis (the stored mode is unchanged: the beam is clipped by 3e-12 per pass)."""
    scene, _ = _scene(project)
    lens_surface = next(s for s in scene.surfaces if s.surface_id == "L/lens")
    rel = -lens_surface.origin
    offset = (float(rel @ lens_surface.u_axis), float(rel @ lens_surface.v_axis))
    components = []
    for c in project.components:
        if c.instance_id == "L":
            (s,) = c.surfaces
            c = c.model_copy(update={"surfaces": (s.model_copy(update={
                "aperture": CircularAperture(radius_mm=3.0, center_offset_mm=offset)}),)})
        components.append(c)
    return project.model_copy(update={"components": tuple(components)})


def test_a_decentred_thin_lens_turns_light_by_its_height_above_its_axis():
    """The Engine's thin lens turns a ray by its height above the lens's axis (the
    surface's origin), so a clear aperture 10.16 mm off the axis turns light by up to 2 atan((10.16 + 3) / 2f) = 117 deg;
    an earlier version took the aperture's own radius (41 deg), lost the cavity E-L-B and PASSed a rating 811 times below the light it
    stores. The bend is measured from the axis: the pair is found with the factor of the centred lens."""
    centred = _fast_lens_cavity(1e9)
    project = _decentred(centred)
    scene, _ = _scene(project)
    lens_surface = next(s for s in scene.surfaces if s.surface_id == "L/lens")
    height = math.hypot(lens_surface.aperture.offset_u, lens_surface.aperture.offset_v) + 3.0
    assert height == pytest.approx(0.8 * 12.7 + 3.0, rel=1e-9)
    assert resonance._lens_bend(lens_surface, LAM, resonance._aperture_radius(lens_surface.aperture)) == pytest.approx(
        2.0 * math.atan(height / (2.0 * 4.0)), rel=1e-12)
    gains = _gains(project)
    g = 1.0 - 1e-6
    (pair,) = gains.pairs
    assert set(pair.surfaces) == {"E/face", "B/face"} and pair.gain == pytest.approx(g, rel=1e-12)
    assert gains.sides("B/face") == pytest.approx((1.0 / (1.0 - g),) * 2, rel=1e-9)


def _v_cavity():
    """A cavity whose light a fast thin lens turns by 151 deg far off the line between its mirrors: E (flat, 200 mm before
    the lens along +x, R 1 - 1e-6 on either side), an ideal thin lens (f 1.3 mm, clear radius 12.7 mm) met 10.16 mm off
    its axis with the slope that turns the ray most, 2 atan(10.16 / 2.6), and B (flat, clear radius 2 mm, R 1 - 1e-6) 100
    mm on, normal to the turned ray: the ray retraces itself. The lens lies 72 mm from the line between E and B."""
    f, h = 1.3, 0.8 * 12.7
    s0 = h / (2.0 * f)
    phi = -math.atan(s0)
    axis = (math.cos(phi), math.sin(phi), 0.0)
    eh = (-math.sin(phi), math.cos(phi), 0.0)
    centre = (-h * eh[0], -h * eh[1], 0.0)
    sp = s0 - h / f
    d = (axis[0] + sp * eh[0], axis[1] + sp * eh[1], 0.0)
    norm = math.hypot(d[0], d[1])
    out = (d[0] / norm, d[1] / norm, 0.0)
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-400.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    m1 = _coat(B.mirror_for_reflection("M1", position=(-350.0, 0.0, 0.0), d_in=(1, 0, 0), d_out=(0, 1, 0), clear_radius_mm=12.7),
               _constant(1.0 - 5e-10), both=True)
    dump = B.beam_dump("D", position=(-350.0, 100.0, 0.0), axis=(0.0, 1.0, 0.0), radius_mm=12.7)
    e = _coat(B.mirror("E", position=(-200.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear_radius_mm=12.7), _constant(1.0 - 1e-6), both=True)
    lens = B.thin_lens("L", focal_length_mm=f, clear_radius_mm=12.7, position=centre, axis=axis)
    b = _coat(B.mirror("B", position=(100.0 * out[0], 100.0 * out[1], 0.0), normal=(-out[0], -out[1], 0.0), clear_radius_mm=2.0),
              _constant(1.0 - 1e-6), both=True)
    return ProjectSnapshot(components=(src, m1, dump, e, lens, b))


def test_a_cavity_closed_through_a_lens_far_off_the_line_between_its_mirrors_is_a_pair():
    """A lens that turns the cavity's light far off the line between its mirrors is between them on the light's way: E
    and B lie on opposite sides of its plane, so it counts among the surfaces between them although it does not meet
    their hull, and the pair is found through it."""
    gains = _gains(_v_cavity())
    g = 1.0 - 1e-6
    (pair,) = gains.pairs
    assert set(pair.surfaces) == {"E/face", "B/face"} and pair.gain == pytest.approx(g, rel=1e-12)
    assert pair.crossed == ("L/lens",)
    assert gains.sides("E/face") == pytest.approx((1.0 / (1.0 - g),) * 2, rel=1e-9)


# ------------------------------------------------------------------------------------------------ reflectance bounds


def test_an_interface_counts_its_reflectance_below_the_critical_angle():
    """From glass into air: Fresnel's s reflectance at the widest angle below the critical angle; a range that reaches the
    critical angle counts 1 (the reflectance just below it); a range wholly beyond it 0 (totally reflected: outside the
    model). From air: up to grazing incidence."""
    n = 1.5
    critical = math.asin(1.0 / n)
    t = math.radians(30.0)
    cos_t = math.sqrt(1.0 - (n * math.sin(t)) ** 2)
    rs = abs((n * math.cos(t) - cos_t) / (n * math.cos(t) + cos_t))
    assert resonance._interface(n, 1.0, 0.0, t) == pytest.approx(rs, rel=1e-12)
    assert resonance._interface(n, 1.0, 0.0, critical + 1e-6) == 1.0
    assert resonance._interface(n, 1.0, critical + 1e-6, 1.2) == 0.0
    assert resonance._interface(1.0, n, 0.0, math.radians(89.0)) == pytest.approx(
        abs((math.cos(math.radians(89.0)) - n * math.sqrt(1.0 - (math.sin(math.radians(89.0)) / n) ** 2))
            / (math.cos(math.radians(89.0)) + n * math.sqrt(1.0 - (math.sin(math.radians(89.0)) / n) ** 2))), rel=1e-9)


def _lens(coating=None, rating: float = 1e9):
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=1.0, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    lens = B.singlet_lens("L", r1_mm=101.32, r2_mm=-101.32, thickness_mm=4.0, clear_radius_mm=12.7, material="N-BK7",
                          position=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0))
    if coating is not None:
        lens = _coat(lens, coating)
    lens = lens.model_copy(update={"power_ratings": (_rating(rating),)})
    return ProjectSnapshot(components=(src, lens), targets=_targets("L"))


def test_a_coated_lens_returns_its_body_s_light_by_the_coating_s_reflectance():
    """A lens with a 0.25 % coating that holds for every angle: its faces face each other through the glass, and the
    light inside meets them at every angle up to the critical angle -- the coating's reflectance there, g = 0.0025: each
    face's factor 1 / (1 - g) from either side. A lens's body is not a lossless reflector (formerly no upper bound for
    every lens)."""
    gains = _gains(_lens(_constant(0.0025)))
    (pair,) = gains.pairs
    assert pair.gain == pytest.approx(0.0025, rel=1e-12)
    for face in ("L/front", "L/back"):
        assert gains.sides(face) == pytest.approx((1.0 / (1.0 - 0.0025),) * 2, rel=1e-12)
        assert gains.factor(face) == pytest.approx(1.0 / (1.0 - 0.0025), rel=1e-12)
    result = run(_lens(_constant(0.0025)))
    assert all(_surface(result, f).ratings[0].exposure_upper_bound is not None for f in ("L/front", "L/back"))


def test_an_uncoated_lens_s_faces_have_no_upper_bound_near_their_critical_angle():
    """An uncoated lens: light inside it can meet both faces at every angle up to the critical angle (its faces' apertures
    subtend more), where a bare face reflects nearly everything -- the pair's gain bound is 1: no upper bound on its faces,
    the rating not decided."""
    gains = _gains(_lens())
    (pair,) = gains.pairs
    assert pair.gain == 1.0
    result = run(_lens())
    for face in ("L/front", "L/back"):
        surface = _surface(result, face)
        assert surface.ratings[0].exposure_upper_bound is None and "FAINT_LIGHT_RECIRCULATES" in surface.reason_codes
    assert result.targets[0].status.value != "PASS"


def test_a_pair_at_a_gain_bound_of_1_states_why_and_its_centre_line_value_beside_the_missing_bound():
    """The faces of the uncoated lens: each reflectance bound is 1 because the angles at which the body's light can meet
    the face reach its critical angle (``CRITICAL_ANGLE_IN_RANGE``). The proven bound stays absent -- that supremum is
    real, and no cheap proof bounds how much light meets a face near its critical angle. Beside it the pair is evaluated
    at its centre-line incidence, where the line between the faces' centres meets them at normal incidence.

    Closed form: at normal incidence a bare interface reflects ``R0 = ((n - 1) / (n + 1))^2`` of the power, so the round
    trip keeps the amplitude ``g0 = r0^2 = R0`` and all returns bring ``g0 / (1 - g0)`` more: the factor ``1 / (1 - R0)``.
    Second method: a lens whose faces carry a coating of that same reflectance at every angle has exactly this factor as
    its *proven* bound (``test_a_coated_lens_returns_its_body_s_light_by_the_coating_s_reflectance``). The estimate
    enters no proven field: ``factor`` stays inf, ``exposure_upper_bound`` absent, the comparison and the target
    undecided -- only ``faint_light_return_factor`` and ``faint_light_peak_W_per_cm2`` carry it, as an ESTIMATE whose
    entry names the pair, the single ray standing for its light and the return it does not bound."""
    n = _bk7_index()
    r0 = ((n - 1.0) / (n + 1.0)) ** 2
    gains = _gains(_lens())
    (pair,) = gains.pairs
    assert pair.saturation == ("CRITICAL_ANGLE_IN_RANGE", "CRITICAL_ANGLE_IN_RANGE")
    assert pair.nominal_angles_deg == (0.0, 0.0)
    assert pair.nominal_gain == pytest.approx(r0, rel=1e-12) and pair.nominal_returns == pytest.approx(r0 / (1.0 - r0), rel=1e-12)
    coated = _gains(_lens(_constant(r0)))
    for face in ("L/front", "L/back"):
        assert math.isinf(gains.factor(face)) and math.isinf(gains.exposure_factor(face)) and math.isinf(gains.power_factor(face))
        assert gains.nominal_factor(face) == pytest.approx(1.0 / (1.0 - r0), rel=1e-12)
        assert gains.nominal_factor(face) == pytest.approx(coated.factor(face), rel=1e-12)  # (the second method)
        assert coated.nominal_factor(face) is None  # (a proven factor: nothing to estimate)
    result = run(_lens())
    (line,) = result.exposures.faint_light
    for face in ("L/front", "L/back"):
        surface = _surface(result, face)
        (comparison,) = surface.ratings
        assert comparison.exposure_upper_bound is None and comparison.status != "WITHIN_RATING"
        assert surface.faint_light_return_factor == pytest.approx(1.0 / (1.0 - r0), rel=1e-12)
        # (the left-out light's diffraction limit in the glass, times its open paths' crossings and the centre-line factor)
        amplitude = gains.crossing_field(face) * gains.nominal_factor(face)
        assert surface.faint_light_peak_W_per_cm2 == pytest.approx(amplitude ** 2 * _peak_limit(line.in_phase_power_W, n=n), rel=1e-9)
        for name in ("faint_light_return_factor", "faint_light_peak_W_per_cm2"):
            entry = surface.value_basis[name]
            assert entry.kind.value == "ESTIMATE"
            (assumed,) = entry.assumed
            assert (assumed.input.value, assumed.at) == ("SINGLE_RAY_FOR_BEAM", ("L/front", "L/back"))
            assert assumed.value == pytest.approx(r0, rel=1e-12)
            assert [(o.part.value, o.at, o.code) for o in entry.omitted] == [
                ("POSSIBLE_RETURN", ("L/front", "L/back"), "EXPOSURE_BOUND_WITHHELD")]
    assert result.targets[0].status.value != "PASS" and result.targets[0].assessment.value == "INDETERMINATE"
    notice = next(i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD")
    context = notice.numerical_context
    assert context["cause"] == "RETURN_PAIR_GAIN_BOUND_1" and context["return_scope"] == "SCENE_WIDE_NOT_LOCALIZED"
    assert context["return_pair_saturation"] == ["CRITICAL_ANGLE_IN_RANGE, CRITICAL_ANGLE_IN_RANGE"]
    assert context["return_pair_nominal_surfaces"] == ["L/front, L/back"]
    assert context["return_pair_nominal_gain"] == pytest.approx([r0], rel=1e-12)


def test_a_bound_a_pair_leaves_open_names_that_pair():
    """The cause recorded for a plane or a surface whose left-out light a pair leaves unbounded names the pair's surfaces
    (the clause the plane's and the target's messages carry), not "two surfaces" of the scene; a pair with a proven bound
    is no cause."""
    assert _gains(_lens()).return_causes("L/front") == ("the surfaces L/front and L/back, facing each other, may hold it",)
    assert _gains(_lens(_constant(0.0025))).return_causes("L/front") == ()


def test_a_bounded_pair_reports_no_centre_line_value_and_keeps_its_bytes():
    """A pair with a proven bound (the coated lens) has nothing to estimate: no saturation code, no centre-line value, no
    entry on the surface, and the surface's record serializes without ``value_basis``."""
    gains = _gains(_lens(_constant(0.0025)))
    (pair,) = gains.pairs
    assert (pair.saturation, pair.nominal_gain, pair.nominal_returns, pair.nominal_angles_deg) == (None, None, None, None)
    result = run(_lens(_constant(0.0025)))
    for face in ("L/front", "L/back"):
        surface = _surface(result, face)
        assert surface.value_basis is None and "value_basis" not in surface.model_dump(mode="json")
        assert surface.faint_light_return_factor == pytest.approx(1.0 / (1.0 - 0.0025), rel=1e-12)


# ------------------------------------------------------------------------------------------------ parallel planes


def test_light_between_parallel_planes_returns_only_where_its_walk_lets_it():
    """Two parallel mirrors (R 0.99) facing each other on one axis form a pair; the same two offset 100 mm along their
    planes do not: light that goes from one aperture to the other moves along them by at least the offset less both radii
    (75 mm), and cannot come back onto a 12.7 mm aperture, which needs a walk of at most its radius."""
    def scene(offset: float):
        src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
        a = _coat(B.mirror("A", position=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear_radius_mm=12.7), _constant(0.99), both=True)
        b = _coat(B.mirror("B", position=(150.0, offset, 0.0), normal=(-1.0, 0.0, 0.0), clear_radius_mm=12.7), _constant(0.99), both=True)
        return ProjectSnapshot(components=(src, a, b))

    aligned, offset = _gains(scene(0.0)), _gains(scene(100.0))
    (pair,) = aligned.pairs
    assert pair.gain == pytest.approx(0.99, rel=1e-12)
    assert offset.pairs == () and offset.by_surface == {}


@pytest.mark.parametrize("theta_deg, returns", [(31.0, 2), (56.3, 1)])
def test_returns_between_parallel_planes_count_each_plane_s_own_aperture(theta_deg: float, returns: int):
    """Flat mirrors A (clear radius 20 mm) and B (1 mm), R 0.999 on either side, 1
    mm apart, and a beam crossing the gap at ``theta``. A ray met first on A hits A, B, A, ... ``s = t tan(theta)`` apart:
    at 31 deg (s = 0.60 mm) it comes back onto A twice, at 56.3 deg (s = 1.50 mm, more than B's radius) once. An earlier version took
    the smaller radius for both planes (one return, and no pair at 56.3 deg); each return onto one plane needs a hit on the
    other, so a plane gets at most min(rho_1 / s, rho_2 / s + 1) returns, and the factor holds them: at least 1 + sum g^m
    over the ray's returns."""
    g = 0.999
    t = math.radians(theta_deg)
    d = (-math.cos(t), math.sin(t), 0.0)
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=1.0, position=(1.0 - 50.0 * d[0], -5.0 - 50.0 * d[1], 0.0),
                  direction=d)
    a = _coat(B.mirror("A", position=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear_radius_mm=20.0), _constant(g), both=True)
    b = _coat(B.mirror("B", position=(1.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), clear_radius_mm=1.0), _constant(g), both=True)
    gains = _gains(ProjectSnapshot(components=(src, a, b)))
    (pair,) = gains.pairs
    assert pair.gain == pytest.approx(g, rel=1e-12)
    hand = 1.0 + sum(g ** m for m in range(1, returns + 1))
    for surface in ("A/face", "B/face"):
        low, high = sorted(gains.sides(surface))
        assert low >= hand * (1.0 - 1e-12) and math.isfinite(high)


def test_parallel_planes_that_lenses_off_their_hull_may_close_into_a_cavity_are_screened_through_them():
    """Two parallel mirrors (R 0.99) 200 mm apart and 100 mm apart along their planes: light between them alone walks off
    their 2 mm apertures at once (no pair; with a tilted plate beside the line between them they are screened through it,
    and neither sends the light it passes back toward the other). Two fast thin lenses off the line between them, each
    with the mirrors (or the other lens) on opposite sides of its plane, may turn light from one mirror toward the other
    and back: the mirrors are then screened as a pair through those lenses."""
    inv = math.sqrt(0.5)

    def scene(lenses: bool, plate: bool = False):
        src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
        a = _coat(B.mirror("A", position=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear_radius_mm=2.0), _constant(0.99), both=True)
        b = _coat(B.mirror("B", position=(200.0, 100.0, 0.0), normal=(-1.0, 0.0, 0.0), clear_radius_mm=2.0), _constant(0.99), both=True)
        parts = [src, a, b]
        if lenses:
            parts += [B.thin_lens("L1", focal_length_mm=3.0, clear_radius_mm=12.7, position=(50.0, 0.0, 0.0), axis=(inv, inv, 0.0)),
                      B.thin_lens("L2", focal_length_mm=3.0, clear_radius_mm=12.7, position=(50.0, 100.0, 0.0), axis=(inv, inv, 0.0))]
        if plate:
            parts.append(B.window("P", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(60.0, 100.0, 0.0),
                                  axis=(inv, inv, 0.0)))
        return ProjectSnapshot(components=tuple(parts))

    assert _gains(scene(False)).pairs == ()
    assert not [p for p in _gains(scene(False, plate=True)).pairs if set(p.surfaces) == {"A/face", "B/face"}]
    gains = _gains(scene(True))
    (pair,) = [p for p in gains.pairs if set(p.surfaces) == {"A/face", "B/face"}]
    assert pair.gain == pytest.approx(0.99, rel=1e-12) and {"L1/lens", "L2/lens"} <= set(pair.crossed)


AR0 = ConstantCoating(reflectance_s=0.0, reflectance_p=0.0, transmittance_s=1.0, transmittance_p=1.0)


def _plane_body(iid: str, faces) -> EngineeringComponentSnapshot:
    """A body of N-BK7 between plane faces ``(id, origin, outward normal)`` (global frame), each 3 mm in radius and coated
    R 0 / T 1 (no ghosts of their own)."""
    surfaces = tuple(PlaneSurface(surface_id=sid, origin_mm=origin, normal=normal, aperture=CircularAperture(radius_mm=3.0),
                                  mount_margin_mm=0.5, medium_positive="air", medium_negative="N-BK7",
                                  interaction=RefractInteraction(coating=AR0)) for sid, origin, normal in faces)
    return EngineeringComponentSnapshot(instance_id=iid, role="window", pose=B._pose((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
                                        surfaces=surfaces)


def _displaced_cavity(body, shift: float, maximum: float):
    """A 1 W, 1064 nm beam along +x onto the back of mirror A at
    the origin (radius 2 mm, R 1 - 5e-10 on either side: its 5e-10 W transmission is left out), parallel mirror B 200 mm
    away facing -x and ``shift`` along z (R 0.999 on either side), beyond the walk of light between the two alone; a
    transparent plane OBS 20 mm before B (radius 2 mm about B's axis) with a power maximum. ``body`` turns light along +x
    from A onto B's axis and back."""
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    a = _coat(B.mirror("A", position=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), clear_radius_mm=2.0), _constant(1.0 - 5e-10), both=True)
    b = _coat(B.mirror("B", position=(200.0, 0.0, shift), normal=(-1.0, 0.0, 0.0), clear_radius_mm=2.0), _constant(0.999), both=True)
    obs = B.observation_plane("OBS", position=(180.0, 0.0, shift), normal=(1.0, 0.0, 0.0), radius_mm=2.0)
    return ProjectSnapshot(components=(src, a, *body, b), observations=(obs,),
                           targets=(Target(target_id="p", observation_id="OBS", metric="power_W", maximum=maximum),))


def _bk7_index() -> float:
    return MaterialLibrary([], NumericalConfig()).evaluator("N-BK7").index(LAM)[0]


def _tilted_plate():
    """A 30 mm N-BK7 plate tilted 70 deg (face normal (cos 70, 0, sin 70)), front face at (60, 0, 0): light along +x
    leaves it along +x shifted by t sin(theta - theta') / cos(theta') along z, written out with Snell's law."""
    n = _bk7_index()
    th = math.radians(70.0)
    normal = (math.cos(th), 0.0, math.sin(th))
    c1 = normal[0]  # (cos of the angle of incidence of +x light on the front face)
    c2 = math.sqrt(1.0 - (1.0 - c1 * c1) / (n * n))
    inside = tuple(d / n + (c1 / n - c2) * q for d, q in zip((1.0, 0.0, 0.0), normal))  # (the refracted direction)
    length = 30.0 / sum(x * y for x, y in zip(inside, normal))
    exit_point = tuple(e + length * x for e, x in zip((60.0, 0.0, 0.0), inside))
    body = _plane_body("PL", [("front", (60.0, 0.0, 0.0), tuple(-x for x in normal)), ("back", exit_point, normal)])
    return (body,), exit_point[2]


def _wedge_pair():
    """Two 20.5 deg N-BK7 wedges (2 mm thick on axis), the second turned 180 deg about the y axis: each has a face normal to
    x and a tilted one, the faces of the two pairwise parallel; light along +x leaves the first deviated by
    asin(n sin 20.5 deg) - 20.5 deg toward -z and the second back along +x, shifted by 99 mm x tan of that deviation."""
    n = _bk7_index()
    alpha = math.radians(20.5)
    tilt = (math.cos(alpha), 0.0, math.sin(alpha))
    shift = -99.0 * math.tan(math.asin(n * math.sin(alpha)) - alpha)
    w1 = _plane_body("W1", [("front", (49.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), ("back", (51.0, 0.0, 0.0), tilt)])
    w2 = _plane_body("W2", [("front", (151.0, 0.0, shift), (1.0, 0.0, 0.0)),
                            ("back", (149.0, 0.0, shift), (-tilt[0], 0.0, -tilt[2]))])
    return (w1, w2), shift


@pytest.mark.parametrize("which", ["plate", "wedges"])
def test_parallel_mirrors_closed_by_a_displacer_off_their_hull_are_a_pair(which: str):
    """A plate tilted off the line between two parallel mirrors,
    or two wedges whose faces are pairwise parallel (a lateral beam displacer), shift light along +x from A sideways onto
    B, which sends it back: a linear cavity. An earlier version took those faces for a plate's -- any two parallel refracting planes
    between the same media -- and screened A and B as bare parallel planes: no pair, and the plane between them PASSed
    2.5 times the left-out power the cavity holds many times over. A tilted face turns light and shifts it sideways, so
    the walk-off bound does not hold through it: the two are a pair (round-trip gain sqrt((1 - 5e-10) 0.999)), and the
    plane's power is not decided at 4 times that light."""
    body, shift = _tilted_plate() if which == "plate" else _wedge_pair()
    project = _displaced_cavity(body, shift, 1.0)
    gains = _gains(project)
    (pair,) = [p for p in gains.pairs if set(p.surfaces) == {"A/face", "B/face"}]
    assert pair.gain == pytest.approx(math.sqrt((1.0 - 5e-10) * 0.999), rel=1e-12)
    faces = {s.surface_id for c in body for s in _scene(ProjectSnapshot(components=(c,)))[0].surfaces}
    assert faces <= set(pair.crossed)
    assert gains.factor("observation/OBS") >= 1.0 / (1.0 - pair.gain) * (1.0 - 1e-9)
    (target,) = run(_displaced_cavity(body, shift, 4.0 * 5e-10)).targets
    assert target.status.value != "PASS"


def test_a_plate_splitter_parallel_to_two_fold_mirrors_keeps_their_walk_off():
    """The Designer's two-beam interferometer: its plate splitter's faces are parallel to its two fold mirrors, which face
    each other along the diagonal. A refracting plane parallel to the mirrors keeps the light's momentum along them and
    counts in the walk (its glass the largest index between them), so the mirrors walk light off each other's apertures:
    no pair, and every surface keeps the factor of the splitter's own faces, 1 / (1 - g), g their round-trip gain."""
    from os_optics.local.example import example_project_of_kind
    from os_optics.local.lowering import lower_project

    project = lower_project(example_project_of_kind("two_beam_interference"), None).request.project
    scene, library = _scene(project)
    wavelength = min(w for w, _ in scene.sources[0].spectrum)
    gains = resonance.return_gains(scene, library, [wavelength])[wavelength]
    (pair,) = gains.pairs
    assert set(pair.surfaces) == {"BS1/S1", "BS1/S2"} and 0.0 < pair.gain < 0.1
    assert gains.factor("M1/S1") == pytest.approx(1.0 / (1.0 - pair.gain), rel=1e-12)


def test_a_refraction_widens_a_cone_by_snell_s_largest_magnification():
    """A cone of directions refracted at a plane spreads by Snell's derivative, n1 cos i / (n2
    cos t) in the plane of incidence: out of a prism face met at 40 deg inside N-BK7 (critical angle 41.6 deg) a cone of
    0.0054 rad leaves 3.2 times wider. The image cone the screen takes holds every exactly refracted direction of the
    cone's edge."""
    n = 1.5066
    spread = 8.0 * LAM_MM / (math.pi * 0.5)
    inside = math.radians(40.0)
    normal = (math.cos(inside), math.sin(inside), 0.0)
    centre = (1.0, 0.0, 0.0)
    cone = resonance._cone(centre, spread, resonance.POSITIVE)
    image = resonance._snell(cone, normal, n, 1.0, resonance.NEGATIVE)
    assert not image.wide and image.reach >= 3.0 * spread
    axis = image.u
    for k in range(64):
        phi = 2.0 * math.pi * k / 64
        edge = resonance._unit((math.cos(spread), math.sin(spread) * math.cos(phi), math.sin(spread) * math.sin(phi)))
        out = resonance._refracted(edge, normal, n, 1.0)
        assert out is not None and resonance._angle(axis, out) <= image.reach * (1.0 + 1e-9)


# ------------------------------------------------------------------------------------------------ outside the model


def _ring():
    """A three-mirror ring A (0, 0) -> B (100, 0) -> C (50, 30) -> A (mirrors R 1 - 1e-6): no
    two of its mirrors face each other."""
    a, b, c = (0.0, 0.0), (100.0, 0.0), (50.0, 30.0)

    def unit(p, q):
        d = math.hypot(q[0] - p[0], q[1] - p[1])
        return ((q[0] - p[0]) / d, (q[1] - p[1]) / d, 0.0)

    hr = _constant(1.0 - 1e-6)
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-400.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    ma = _coat(B.mirror_for_reflection("A", position=(0.0, 0.0, 0.0), d_in=unit(c, a), d_out=unit(a, b), clear_radius_mm=3.0), hr, both=True)
    mb = _coat(B.mirror_for_reflection("B", position=(100.0, 0.0, 0.0), d_in=unit(a, b), d_out=unit(b, c), clear_radius_mm=3.0), hr)
    mc = _coat(B.mirror_for_reflection("C", position=(50.0, 30.0, 0.0), d_in=unit(b, c), d_out=unit(c, a), clear_radius_mm=3.0), hr)
    return ProjectSnapshot(components=(src, ma, mb, mc))


def test_a_ring_s_returns_are_outside_the_model():
    """Left-out light circulating through three or more reflecting surfaces is outside the model: a ring's mirrors form no
    pair, and no factor is taken for the ring's round trips (the traced light that enters a ring meets the interaction
    limit and leaves its surfaces without an upper bound)."""
    gains = _gains(_ring())
    assert gains.pairs == () and gains.by_surface == {}


def test_a_periscope_s_windows_pair_their_own_faces_only():
    """A window before and after a periscope of two HR fold mirrors: the light one window reflects can reach the other only
    through the folds -- a closed path of four surfaces, outside the model; the fold mirrors, parallel and 100 mm apart
    along their planes, form no pair. Each window's faces form a pair, and the two windows' parallel faces, 100 mm apart
    along their planes, form none."""
    hr = _constant(0.995)
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    w1 = B.window("W1", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    m1 = _coat(B.mirror_for_reflection("M1", position=(100.0, 0.0, 0.0), d_in=(1, 0, 0), d_out=(0, 0, 1), clear_radius_mm=12.7), hr)
    m2 = _coat(B.mirror_for_reflection("M2", position=(100.0, 0.0, 100.0), d_in=(0, 0, 1), d_out=(1, 0, 0), clear_radius_mm=12.7), hr)
    w2 = B.window("W2", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(150.0, 0.0, 100.0))
    gains = _gains(ProjectSnapshot(components=(src, w1, m1, m2, w2)))
    assert sorted(tuple(sorted(p.surfaces)) for p in gains.pairs) == [("W1/back", "W1/front"), ("W2/back", "W2/front")]
    assert all(p.gain < 0.5 for p in gains.pairs)


def test_light_that_enters_a_window_at_grazing_incidence_walks_off_its_faces():
    """A window with a mirror beside it, whose light can graze the window's faces: that light enters at up to 90 deg and
    meets the faces from inside up to the critical angle, where they reflect nearly everything (the pair's gain bound 1).
    It walks along the faces by at least t k / sqrt(n^2 - k^2) per pass, so it comes back onto the 12.7 mm aperture only a
    few times: the faces' returns stay finite (``min(M g, g / (1 - g))`` over the momenta)."""
    src = B.laser("S", wavelength_nm=800.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    mirror = B.mirror("M", position=(0.0, 200.0, 0.0), normal=(0.0, -1.0, 0.0))
    gains = _gains(ProjectSnapshot(components=(src, window, mirror)), 800.0)
    (pair,) = gains.pairs
    assert set(pair.surfaces) == {"W/front", "W/back"} and pair.gain == 1.0
    n = _bk7(0.8)
    k = 0.9  # (a momentum below the top piece's lower end: its walk is shorter, so it allows more returns)
    most = math.floor(12.7 / (3.0 * k / math.sqrt(n * n - k * k)))
    for face in ("W/front", "W/back"):
        sides = gains.sides(face)
        assert all(math.isfinite(f) for f in sides) and max(sides) <= 1.0 + most


def test_light_through_lenses_between_two_faces_keeps_the_cone_they_pass_it_on_in():
    """Two faces of different uncoated lenses facing each other across the train: the pair's light reaches each through
    the faces between them, which pass it on within their refraction of the legs into them (``_Scene.passed``), not in
    every direction from their apertures -- the faces meet it well below the critical angle and the pair's gain is a few
    percent, while each lens's own two faces, which light inside it can meet up to the critical angle, bound nothing."""
    n = 1.5066  # (N-BK7 near 1064 nm: f = 100 mm for R = 2 (n - 1) f)
    r = 2.0 * (n - 1.0) * 100.0
    src = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=1.0, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    lenses = [B.singlet_lens(f"L{k + 1}", r1_mm=r, r2_mm=-r, thickness_mm=4.0, clear_radius_mm=12.7, material="N-BK7",
                             position=(x, 0.0, 0.0), axis=(1.0, 0.0, 0.0)) for k, x in enumerate((0.0, 200.0))]
    gains = _gains(ProjectSnapshot(components=(src, *lenses)))
    pairs = {tuple(sorted(p.surfaces)): p for p in gains.pairs}
    across = pairs[("L1/back", "L2/back")]
    assert across.gain < 0.1 and across.crossed == ("L2/front",)
    assert max(across.angles_deg[1]) < 30.0
    assert pairs[("L1/back", "L1/front")].gain == 1.0 and pairs[("L2/back", "L2/front")].gain == 1.0
