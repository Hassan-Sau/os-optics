"""Regression tests of three bounded low-level fixes (PHYSICS_SCOPE P1 / P2):

* a coating is judged on the rays the tracer still follows: a live pupil ray outside the coating's angle range while the
  chief ray is inside it goes on with the coefficient at the nearest stated angle, a chief ray outside it goes on with the
  coating's values at the nearest stated angle (``COATING_OUT_OF_RANGE``, the power behind an estimate), and a retired
  ray's stale row never decides anything;
* a ray that misses a spherical cap gets a finite unit normal (the cap axis), not ``(0 - centre) / R``;
* an AOI-independent ``TabulatedCoating`` is interpolated over wavelength only, so its R / T cannot depend on the last bits
  of an AOI computed by another libm.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import (
    CircularAperture,
    ConstantCoating,
    EngineeringComponentSnapshot,
    MirrorInteraction,
    PlaneSurface,
    Pose3D,
    ProjectSnapshot,
    RefractInteraction,
    SphereSurface,
    TabulatedCoating,
)
from os_optics.optics.materials.coatings import CoatingEvaluator
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.optics.surfaces import intersect
from os_optics.scene.lower import lower_scene

from helpers import CFG, arrivals, issue_codes, run

# Far from the global origin, so a missed intersection point left at (0, 0, 0) is far from the sphere.
X0 = 500.0
SQRT_HALF = math.sqrt(0.5)


def _coated_mirror(aoi_lo: float, aoi_hi: float, *, normal: tuple[float, float, float]) -> EngineeringComponentSnapshot:
    coating = TabulatedCoating(
        wavelength_nm=(400.0, 900.0), aoi_deg=(aoi_lo, aoi_hi),
        reflectance_s=((0.9, 0.9), (0.9, 0.9)), reflectance_p=((0.9, 0.9), (0.9, 0.9)),
        transmittance_s=((0.0, 0.0), (0.0, 0.0)), transmittance_p=((0.0, 0.0), (0.0, 0.0)),
    )
    surface = PlaneSurface(surface_id="face", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                           aperture=CircularAperture(radius_mm=12.0), interaction=MirrorInteraction(coating=coating))
    return EngineeringComponentSnapshot(instance_id="M", role="mirror", pose=Pose3D.from_axis_direction((X0 + 40.0, 0.0, 0.0), normal),
                                        surfaces=(surface,))


def _converging_fold(aoi_lo: float, aoi_hi: float, normal=(-SQRT_HALF, SQRT_HALF, 0.0)) -> ProjectSnapshot:
    """A 2 mm beam focused by an f = 50 mm thin lens onto a 45 deg fold mirror: pupil rays meet it 45 +- ~2.3 deg."""

    return ProjectSnapshot(
        components=(
            B.laser("L", wavelength_nm=633.0, waist_radius_mm=2.0, position=(X0 - 20.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)),
            B.thin_lens("TL", focal_length_mm=50.0, clear_radius_mm=12.0, position=(X0, 0.0, 0.0), axis=(1.0, 0.0, 0.0)),
            _coated_mirror(aoi_lo, aoi_hi, normal=normal),
        ),
        observations=(B.observation_plane("out", position=(X0 + 40.0, 30.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=20.0),),
    )


def test_live_pupil_rays_outside_a_coating_range_keep_their_geometry_with_the_edge_coefficient():
    """The marginal pupil rays meet the mirror up to ~2.3 deg beside the chief ray's 45 deg, outside the coating's 44-46 deg:
    their directions follow the law of reflection, which does not use the coating, so they go on with its coefficient held
    at the range's edge. The beam's power is the chief ray's (inside the range): the same number as before, and the ray
    geometry is that of the same mirror with a coating stated over every angle (the independent reference)."""
    result = run(_converging_fold(44.0, 46.0))
    reference = run(_converging_fold(0.0, 90.0))
    (arrival,) = arrivals(result, "out")
    (wide,) = arrivals(reference, "out")
    assert arrival.beam.power_W == wide.beam.power_W == pytest.approx(1e-3 * 0.9, rel=1e-12)
    assert arrival.pupil_ray_count == CFG.pupil_ring_count * CFG.pupil_points_per_ring
    assert arrival.pupil_rays_arrived == wide.pupil_rays_arrived == arrival.pupil_ray_count
    assert arrival.pupil_rms_radius_mm == wide.pupil_rms_radius_mm
    assert arrival.pupil_centroid_uv_mm == wide.pupil_centroid_uv_mm
    assert dict(result.power_budget.lines[0].sinks) == dict(reference.power_budget.lines[0].sinks)
    # one warning at the mirror: how many rays, the largest angle met, the range
    (held,) = [i for i in result.issues if i.code.value == "COATING_OUT_OF_RANGE"]
    assert (held.status.value, held.location) == ("WARNING", "surface:M/face")
    assert held.numerical_context["valid_aoi_deg"] == [44.0, 46.0]
    assert 46.0 < held.numerical_context["aoi_max_deg"] < 48.0
    assert 0 < held.numerical_context["pupil_rays_held"] < arrival.pupil_ray_count


def test_a_chief_ray_outside_the_coating_range_goes_on_with_the_nearest_stated_angle():
    """The chief ray meets the mirror at 45 deg, its coating is stated for 0-40 deg (R = 0.9 at every angle): the beam is
    reflected with the 40 deg values, to the place and with the pupil rays of the mirror stated for 0-50 deg, and its power
    (1 mW x 0.9) is an estimate that names the angle range."""

    result, inside = run(_converging_fold(0.0, 40.0)), run(_converging_fold(0.0, 50.0))
    (issue,) = [i for i in result.issues if i.code.value == "COATING_OUT_OF_RANGE"]
    assert issue.status.value == "NOT_APPLICABLE" and issue.location == "surface:M/face"
    assert issue.numerical_context["held_aoi_deg"] == 40.0 and issue.numerical_context["aoi_deg"] == pytest.approx(45.0, abs=1e-9)
    (arrival,), (expected,) = arrivals(result, "out"), arrivals(inside, "out")
    assert arrival.beam.power_W == pytest.approx(1e-3 * 0.9, rel=1e-12)
    assert arrival.beam.model_dump() == expected.beam.model_dump()
    assert arrival.pupil_rays_arrived == expected.pupil_rays_arrived == arrival.pupil_ray_count  # no ray retired by the range
    entry = arrival.value_basis["beam.power_W"]
    (assumed,), (check,) = entry.assumed, entry.validity
    assert entry.kind.value == "ESTIMATE" and (assumed.input.value, assumed.at, assumed.value, assumed.unit) == (
        "HELD_AT_NEAREST_STATED_POINT", ("M/face",), 40.0, "deg")
    assert (check.parameter, check.minimum, check.maximum, check.within, check.code) == ("aoi_deg", 0.0, 40.0, False, "COATING_OUT_OF_RANGE")
    assert check.observed == pytest.approx(45.0, abs=1e-9)
    (path,) = result.paths
    assert path.termination != "COATING_OUT_OF_RANGE" and path.power_out_of_scope_at == ("M/face",)


def _small_coated_cap() -> EngineeringComponentSnapshot:
    """One refracting sphere (R = 20 mm, 1 mm clear radius, no rim) whose coating is valid up to 10 deg only."""

    coating = ConstantCoating(reflectance_s=0.01, reflectance_p=0.01, transmittance_s=0.99, transmittance_p=0.99, max_aoi_deg=10.0)
    cap = SphereSurface(surface_id="cap", center_mm=(0.0, 0.0, 20.0), radius_mm=20.0, cap_axis=(0.0, 0.0, -1.0), aperture_radius_mm=1.0,
                        medium_positive="air", medium_negative="N-BK7", interaction=RefractInteraction(coating=coating))
    exit_face = PlaneSurface(surface_id="exit", origin_mm=(0.0, 0.0, 3.0), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=1.0),
                             medium_positive="air", medium_negative="N-BK7", interaction=RefractInteraction())
    return EngineeringComponentSnapshot(instance_id="C", role="lens", pose=Pose3D.from_axis_direction((X0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                                        surfaces=(cap, exit_face))


def test_rays_that_miss_a_sphere_never_decide_its_coating():
    """A 3 mm beam on a 1 mm cap: every outer pupil ray misses the cap (retired). Their rows must not fail the coating."""

    project = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=633.0, waist_radius_mm=3.0, position=(X0 - 20.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)),
                    _small_coated_cap()),
        observations=(B.observation_plane("out", position=(X0 + 100.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=50.0),),
    )
    result = run(project)
    assert "COATING_OUT_OF_RANGE" not in issue_codes(result)
    assert len(arrivals(result, "out")) == 1


def test_a_missed_sphere_intersection_has_a_finite_unit_normal():
    project = ProjectSnapshot(components=(_small_coated_cap(),))
    scene, _ = lower_scene(project, CFG, MaterialLibrary((), CFG))
    cap = next(s for s in scene.surfaces if s.surface_id == "C/cap")
    origins = np.array([[X0 - 10.0, 0.0, 0.0], [X0 - 10.0, 5.0, 0.0], [X0 - 10.0, 0.0, 50.0]])
    directions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hits = intersect(cap, origins, directions, CFG)
    assert np.isfinite(hits.t[0]) and not np.isfinite(hits.t[1:]).any()
    np.testing.assert_allclose(np.linalg.norm(hits.normal, axis=1), 1.0, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(hits.normal[1:], np.repeat(cap.cap_axis[None, :], 2, axis=0))


def _flat_in_aoi_table() -> TabulatedCoating:
    """Two bands (R 0.95 below 600 nm, R 0.1 above 650 nm), AOI-independent."""

    wl = (100.0, 400.0, 600.0, 650.0, 1100.0, 20000.0)
    r = (0.95, 0.95, 0.95, 0.1, 0.1, 0.1)
    t = (0.05, 0.05, 0.05, 0.9, 0.9, 0.9)
    rows = lambda values: tuple((v, v) for v in values)  # noqa: E731
    return TabulatedCoating(wavelength_nm=wl, aoi_deg=(0.0, 90.0), reflectance_s=rows(r), reflectance_p=rows(r),
                            transmittance_s=rows(t), transmittance_p=rows(t))


def test_an_aoi_independent_table_is_exact_and_independent_of_the_aoi():
    ev = CoatingEvaluator(_flat_in_aoi_table(), CFG)
    aoi = np.array([0.0, 44.999999999999993, 45.0, 45.000000000000007, 89.0])
    for wavelength, (r, t) in ((500.0, (0.95, 0.05)), (600.0, (0.95, 0.05)), (800.0, (0.1, 0.9)), (400.0, (0.95, 0.05))):
        c = ev.coefficients(wavelength, aoi, location="x")
        assert c.issue is None
        # exactly the band values, bit for bit, at every AOI
        assert c.rs.tolist() == [r] * aoi.size and c.rp.tolist() == [r] * aoi.size
        assert c.ts.tolist() == [t] * aoi.size and c.tp.tolist() == [t] * aoi.size
    gap = ev.coefficients(625.0, aoi, location="x")
    assert np.all(gap.rs == gap.rs[0]) and gap.rs[0] == pytest.approx(0.525, rel=1e-12)


def test_an_aoi_dependent_table_keeps_bilinear_interpolation():
    table = _flat_in_aoi_table()
    tilted = table.model_copy(update={"reflectance_s": tuple((v, v / 2) for v, _ in table.reflectance_s)})
    ev = CoatingEvaluator(tilted, CFG)
    c = ev.coefficients(500.0, np.array([0.0, 45.0, 90.0]), location="x")
    np.testing.assert_allclose(c.rs, [0.95, 0.95 * 0.75, 0.475], rtol=1e-12)
    assert c.rp.tolist() == [0.95, 0.95, 0.95]


def test_a_held_coating_keeps_the_entry_the_target_had_and_adds_its_own():
    """B6: a power target whose beam met a coating outside its data reports the estimate with the union of the entries it
    reads -- what its own value already stated (here light not followed, as on a BOUNDED plane) and the held coating."""
    from os_optics.config import NumericalConfig
    from os_optics.contracts import Target
    from os_optics.contracts.basis import Omitted, ValueBasis
    from os_optics.errors import Status
    from os_optics.errors.status import OmittedPart, ValueKind
    from os_optics.validation import metrics as M

    target = Target(target_id="p", metric="power_W", observation_id="out", minimum=0.0, unit="W")
    result = run(_converging_fold(0.0, 40.0).model_copy(update={"targets": (target,)}))
    (obs,) = [o for o in result.observations if o.observation_id == "out"]
    (outcome,) = result.targets
    assert [a.input.value for a in outcome.value_basis["value"].assumed] == ["HELD_AT_NEAREST_STATED_POINT"]
    own = ValueBasis(kind=ValueKind.LOWER_BOUND, omitted=(Omitted(part=OmittedPart.NOT_FOLLOWED, at=("TL/lens",), upper=1e-6,
                                                                  code="UNTRACED_LIGHT_BOUNDED"),))
    given = outcome.model_copy(update={"status": Status.PASS, "value_basis": {"value": own}})
    scoped = M._power_scope(target, given, obs, NumericalConfig())
    entry = scoped.value_basis["value"]
    assert scoped.status.value == "OUT_OF_MODEL_SCOPE" and entry.kind.value == "ESTIMATE"
    assert [(o.part.value, o.at, o.upper) for o in entry.omitted] == [("NOT_FOLLOWED", ("TL/lens",), 1e-6)]
    assert [a.input.value for a in entry.assumed] == ["HELD_AT_NEAREST_STATED_POINT"]
