"""A transparent observation plane is virtual: adding one anywhere changes no other result.

Observation planes never move, turn, retire or count a ray. For realistic scenes -- a lens, a fold mirror, a window --
and transparent planes placed where they used to interfere (tilted beside a mirror's or a lens's hit point, grazing along
the beam, through a surface's hit point, between two surfaces, before or after the other planes in the declaration
order), every other plane's results are bit-identical with and without the added plane: its arrivals (power, bounds, pupil
points and spot metrics), status and power, its targets and the issues placed on it, the power ledger and the surface
exposures. Also for a rated element the beam passes beside: a plane, or a physical surface, whose plane the
chief ray crosses exactly where it crosses the rated surface's plane does not hide that surface's exposure.
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.contracts import PowerRating, ProjectSnapshot, Target

from helpers import run

Q = math.sqrt(0.5)
X = (1.0, 0.0, 0.0)


def _unit(v) -> tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)  # type: ignore[return-value]


def _laser():
    return B.laser("L", wavelength_nm=780.0, waist_radius_mm=0.5, power_W=1e-3, position=(-10.0, 0.0, 0.0), direction=X)


def _plane(oid: str, position, normal, **kwargs):
    kwargs.setdefault("radius_mm", 50.0)
    return B.observation_plane(oid, position=position, normal=normal, **kwargs)


def _targets(oid: str) -> tuple[Target, ...]:
    return (Target(target_id=f"{oid}_power", metric="power_W", observation_id=oid, maximum=1.5e-3, unit="W"),
            Target(target_id=f"{oid}_rms", metric="spot_rms_mm", observation_id=oid, maximum=0.3, unit="mm"),
            Target(target_id=f"{oid}_cu", metric="spot_centroid_u_mm", observation_id=oid, minimum=-0.01, maximum=0.01, unit="mm"))


SCENES = {
    "lens": ((B.thin_lens("F", position=(0.0, 0.0, 0.0), focal_length_mm=200.0, clear_radius_mm=10.0),),
             (_plane("D", (60.0, 0.0, 0.0), X), _plane("A", (-5.0, 0.0, 0.0), X))),
    "fold": ((B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0), reflectance=0.99),),
             (_plane("D", (0.0, 60.0, 0.0), (0.0, 1.0, 0.0)), _plane("A", (0.0, 20.0, 0.0), (0.0, 1.0, 0.0)))),
    "window": ((B.window("W", thickness_mm=5.0, clear_radius_mm=10.0, material="N-BK7", position=(2.5, 0.0, 0.0)),),
               (_plane("D", (40.0, 0.0, 0.0), X),)),
}
#: Transparent planes added to each scene: tilted beside the first surface's hit point (part of the beam crosses it before
#: the surface, the chief ray after), grazing along the beam, through the hit point, before the surface, behind it.
ADDED = {
    "tilted_after": _plane("T", (0.3, 0.0, 0.0), _unit((1.0, 1.0, 0.0))),
    "tilted_beside_fold": _plane("T", (0.3, 0.3, 0.0), _unit((1.0, 1.0, 0.0))),
    "grazing": _plane("T", (0.2, 0.0, 0.0), _unit((0.05, 1.0, 0.0))),
    "through_hit_point": _plane("T", (0.0, 0.0, 0.0), _unit((1.0, -0.4, 0.2))),
    "before": _plane("T", (-3.0, 0.0, 0.0), _unit((1.0, 0.3, 0.0))),
    "small_area": _plane("T", (-1.0, 0.2, 0.0), X, radius_mm=0.1),
}


def _others(result, added: str):
    """Everything about the planes other than ``added``: observations, their targets and issues, and the power ledger."""
    observations = {o.observation_id: o for o in result.observations if o.observation_id != added}
    targets = {t.target_id: t for t in result.targets if t.observation_id != added}
    issues = sorted(((i.location, i.code.value, i.status.value, i.message, repr(sorted(i.numerical_context.items())))
                     for i in result.issues if i.location and i.location.startswith("observation:") and i.location != f"observation:{added}"))
    budget = (result.power_budget.total_sinks, [line.sinks for line in result.power_budget.lines])
    exposures = result.exposures.model_dump(mode="json") if result.exposures is not None else None
    return observations, targets, issues, budget, exposures


@pytest.mark.parametrize("scene", sorted(SCENES))
@pytest.mark.parametrize("added", sorted(ADDED))
@pytest.mark.parametrize("first", (False, True), ids=("declared_last", "declared_first"))
def test_adding_a_transparent_plane_changes_no_other_planes_results(scene: str, added: str, first: bool):
    components, planes = SCENES[scene]
    targets = tuple(t for p in planes for t in _targets(p.observation_id))
    reference = run(ProjectSnapshot(components=(_laser(), *components), observations=planes, targets=targets))
    extra = ADDED[added]
    with_plane = run(ProjectSnapshot(components=(_laser(), *components),
                                     observations=(extra, *planes) if first else (*planes, extra), targets=targets))
    assert _others(with_plane, "T") == _others(reference, "T")


def test_the_spurious_record_of_review_c16_is_gone():
    """c16 of the review: a 45 deg monitor 0.3 mm beside a fold mirror's hit point used to retire the rays that had
    crossed it before the mirror, so the detector behind saw 86 of 96 pupil rays (a smaller spot, a false MET) and got a
    partial-crossing record of the main beam; the detector's results now equal those without the monitor."""
    fold = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0), reflectance=0.99)
    d = _plane("D", (0.0, 60.0, 0.0), (0.0, 1.0, 0.0))
    t = _plane("T", (0.3, 0.3, 0.0), _unit((1.0, 1.0, 0.0)))
    targets = (Target(target_id="dmax", metric="power_W", observation_id="D", maximum=1.5e-3, unit="W"),
               Target(target_id="drms", metric="spot_rms_mm", observation_id="D", maximum=0.34, unit="mm"))
    control = run(ProjectSnapshot(components=(_laser(), fold), observations=(d,), targets=targets))
    result = run(ProjectSnapshot(components=(_laser(), fold), observations=(t, d), targets=targets))
    assert _others(result, "T") == _others(control, "T")
    detector = next(o for o in result.observations if o.observation_id == "D").arrivals[0]
    assert detector.pupil_rays_arrived == detector.pupil_ray_count == 96
    assert {t.target_id: t.assessment.value for t in result.targets} == {"dmax": "MET", "drms": "UNMET"}


def test_the_rays_a_plane_does_not_sample_go_on_in_the_ray_data():
    """In c16 the rays above y = 0.3 mm cross T before the mirror, so on the folded leg they do not cross it again: T's step
    has no point of them, and they go on to D and leave the scene with the beam (``PATH_END``, not ``NOT_FOLLOWED``)."""
    from os_optics import EngineConfiguration, ValidationRequest, validate_with_ray_paths  # noqa: PLC0415

    fold = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-Q, Q, 0.0), reflectance=0.99)
    project = ProjectSnapshot(components=(_laser(), fold), observations=(
        _plane("T", (0.3, 0.3, 0.0), _unit((1.0, 1.0, 0.0))), _plane("D", (0.0, 60.0, 0.0), (0.0, 1.0, 0.0))))
    _result, trace = validate_with_ray_paths(ValidationRequest(
        project=project, configuration=EngineConfiguration(run_layout=False, run_control=False)))
    (ray_path,) = trace.paths
    assert {ray.end for ray in ray_path.pupil_rays} == {"PATH_END"}
    step = ray_path.chief_surface_ids.index("observation/T")
    skipped = [ray for ray in ray_path.pupil_rays if step not in ray.steps]
    assert 0 < len(skipped) < len(ray_path.pupil_rays)
    assert all(ray.surface_ids[-1] == "observation/D" for ray in skipped)


# --------------------------------------------------------- a rated element beside the beam (``_beside``)


def _rated(component, value: float = 1e3):
    rating = PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=780.0, status="DECLARED", basis="test",
                         value_definition="PEAK", beam_diameter_mm=1.0, beam_diameter_definition="1/E2", polarization="LINEAR",
                         polarization_condition="linear")
    return component.model_copy(update={"power_ratings": (rating,)})


def _utilization(device: str) -> Target:
    return Target(target_id=f"u_{device}", device_id=device, metric="power_rating_utilization", maximum=1.0)


BESIDE = {
    # a 1 mm mirror on a 2 mm mount ring, 0.6 / 0.3 mm beside the beam's footprint edge: the tail reaches its support
    "mirror_0.6": (_rated(B.mirror("M", position=(5.0, 3.6, 0.0), normal=(-1.0, 0.0, 0.0), clear_radius_mm=1.0, mount_margin_mm=2.0,
                                   reflectance=0.99), 1e-3), 5.0, "M"),
    "mirror_0.3": (_rated(B.mirror("M", position=(5.0, 3.3, 0.0), normal=(-1.0, 0.0, 0.0), clear_radius_mm=1.0, mount_margin_mm=2.0,
                                   reflectance=0.99), 1e-1), 5.0, "M"),
    "window_front": (_rated(B.window("W", thickness_mm=3.0, clear_radius_mm=1.0, material="N-BK7", position=(5.0, 3.6, 0.0))), 3.5, "W"),
    "window_back": (_rated(B.window("W", thickness_mm=3.0, clear_radius_mm=1.0, material="N-BK7", position=(5.0, 3.9, 0.0))), 6.5, "W"),
}


@pytest.mark.parametrize("scene", sorted(BESIDE))
@pytest.mark.parametrize("offset", (0.0, 1e-6), ids=("coplanar", "1nm_behind"))
@pytest.mark.parametrize("first", (False, True), ids=("declared_last", "declared_first"))
def test_a_plane_coplanar_with_a_rated_surface_beside_the_beam_leaves_its_exposure(scene: str, offset: float, first: bool):
    element, x, device = BESIDE[scene]
    d = _plane("D", (50.0, 0.0, 0.0), X)
    targets = (*_targets("D"), _utilization(device))
    reference = run(ProjectSnapshot(components=(_laser(), element), observations=(d,), targets=targets))
    t = _plane("T", (x + offset, 0.0, 0.0), X)
    with_plane = run(ProjectSnapshot(components=(_laser(), element), observations=(t, d) if first else (d, t), targets=targets))
    assert _others(with_plane, "T") == _others(reference, "T")
    utilization = next(r for r in reference.targets if r.target_id == f"u_{device}")
    assert utilization.value is None and reference.exposures.surfaces  # (the tail reaches it: bounded, never a known 0)


def test_a_lens_in_the_beam_coplanar_with_a_rated_mirror_beside_it_leaves_its_exposure():
    """Two mounts in one row: the beam passes through a lens at x = 5 while a rated mirror's support beside the beam is
    reached by its tail at the same x. The mirror keeps its exposure (bounded, INDETERMINATE), as with the lens 1 um away."""
    mirror = BESIDE["mirror_0.3"][0]
    for x in (5.0, 5.001):
        lens = B.thin_lens("F", position=(x, 0.0, 0.0), focal_length_mm=1e6, clear_radius_mm=1.5)
        result = run(ProjectSnapshot(components=(_laser(), mirror, lens), observations=(_plane("D", (50.0, 0.0, 0.0), X),),
                                     targets=(_utilization("M"),)))
        (u,) = result.targets
        assert u.value is None and u.assessment.value == "INDETERMINATE"
        assert [s.surface_id for s in result.exposures.surfaces if s.element_id == "M"]
