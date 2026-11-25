"""Diffuse surface scatter and bulk scattering, against closed-form references.

Every reference is computed here from the geometry stated in the test, independently of the Engine:

* a point scatterer with intensity ``I(theta) proportional to cos^n(theta)`` (``n = 1``: Lambertian) sends the share
  ``1 - cos^(n + 1)(theta_max)`` of its power into a coaxial cone of half angle ``theta_max``; a coaxial disc of radius ``a`` at
  distance ``h`` subtends ``cos(theta_max) = h / sqrt(h^2 + a^2)`` (Lambertian: ``a^2 / (a^2 + h^2)``, the configuration factor
  ``sin^2(theta_max)`` of a differential element to a coaxial disc);
* a specular beam keeps ``(1 - f)`` of the power that meets the scattering surface; through a plate of index ``n`` at normal
  incidence it keeps ``(1 - R)^2 exp(-(alpha + mu_s) L)`` with ``R = ((n - 1) / (n + 1))^2``; the medium's loss splits into
  absorption and scattering in the ratio ``alpha : mu_s``.

Tolerance of the sampled fractions (4096 rays, stratified): absolute 0.004 of the scattered power, as for the extended emitters
(``test_extended_source_radiometry.py``).
"""

from __future__ import annotations

import copy
import math

import pytest
from declared_helpers import declared_component, sequence, surface, with_models
from local_helpers import catalog_document, plane, project, run, source
from pydantic import ValidationError

from os_optics import EngineConfiguration, ValidationRequest, validate
from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import AbsorberInteraction, MirrorInteraction, ProjectSnapshot, RefractInteraction, SurfaceScatter
from os_optics.contracts.materials import ConstantIndexMaterial
from os_optics.contracts.materials import ConstantCoating
from os_optics.local import LocalRunError
from os_optics.validation.pipeline import BEAMS_ALONE, run_physics

INV = 1.0 / math.sqrt(2.0)
FRACTION_ABS_TOL = 0.004
RAYS = 4096


def _dump_scene(interaction, *, h: float = 10.0, a: float = 5.0, power_w: float = 1.0) -> ProjectSnapshot:
    """A beam at 45 deg onto a plane at the origin (normal +z) and a coaxial absorbing disc of radius ``a`` at height ``h``."""
    src = B.laser("S", wavelength_nm=633.0, power_W=power_w, waist_radius_mm=0.2, position=(-50.0, 0.0, 50.0), direction=(INV, 0.0, -INV))
    dump = B.beam_dump("DUMP", position=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius_mm=12.0)
    face = dump.surfaces[0].model_copy(update={"interaction": interaction})
    dump = dump.model_copy(update={"surfaces": (face,)})
    det = B.observation_plane("D", position=(0.0, 0.0, h), normal=(0.0, 0.0, -1.0), radius_mm=a, absorbing=True)
    return ProjectSnapshot(name="scatter", components=(src, dump), observations=(det,))


def _validate(snapshot: ProjectSnapshot):
    return validate(ValidationRequest(project=snapshot, configuration=EngineConfiguration(run_layout=False, run_control=False)))


def _cone_share(n: float, h: float, a: float) -> float:
    return 1.0 - (h / math.sqrt(h * h + a * a)) ** (n + 1.0)


# ------------------------------------------------------------------------------------------------ surface scatter


@pytest.mark.parametrize("bsdf, exponent", [("LAMBERTIAN", None), ("COSINE_POWER", 3.0), ("COSINE_POWER", 0.5)])
def test_scattered_share_on_a_coaxial_disc_is_the_cosine_power_cone_integral(bsdf, exponent):
    f, h, a = 0.1, 10.0, 5.0
    scatter = SurfaceScatter(bsdf=bsdf, exponent=exponent, fraction=f, side="reflect", sample_count=RAYS)
    result = _validate(_dump_scene(AbsorberInteraction(scatter=scatter), h=h, a=a))
    plane_d = result.observations[0]
    (line,) = plane_d.sampled_light
    reference = _cone_share(1.0 if exponent is None else exponent, h, a)
    if exponent is None:
        assert reference == pytest.approx(a * a / (a * a + h * h), rel=1e-12)
    assert abs(line.collection_fraction - reference) < FRACTION_ABS_TOL
    assert abs(line.collection_fraction_half_samples - reference) < 2 * FRACTION_ABS_TOL
    assert line.scatter_surface_id == "DUMP/dump" and line.parent_path_id == "S#0" and line.emitted_W == pytest.approx(f)
    # the plane's power is the scattered light it collects, and the run states it whole
    assert plane_d.total_power_W == pytest.approx(f * line.collection_fraction, rel=1e-12)
    assert plane_d.quantity_state == "KNOWN"
    (src_line,) = result.sampled_sources
    assert src_line.shape == "POINT" and src_line.angular_law == bsdf and src_line.scatter_side == "REFLECT"
    assert src_line.scattered_fraction == f and src_line.incident_power_W == pytest.approx(1.0)
    n = 1.0 if exponent is None else exponent
    assert src_line.intensity_W_per_sr == pytest.approx(f * (n + 1.0) / (2.0 * math.pi), rel=1e-12)
    # every scattered ray's crossing is on the stated hemisphere: it travels away from the dump (+z), toward the plane
    assert all(dn < 0.0 for dn in line.crossings.direction_n)  # (the plane's normal faces the dump)


def test_the_ledger_closes_over_the_specular_and_the_scattered_light():
    scatter = SurfaceScatter(fraction=0.25, side="reflect", sample_count=1024)
    result = _validate(_dump_scene(AbsorberInteraction(scatter=scatter)))
    (line,) = result.power_budget.lines
    assert line.conserved and abs(line.residual_W) < 1e-12
    # the absorber takes the unscattered share; the scattered rays end on the detector or leave the scene
    assert line.sinks["ABSORBED"] == pytest.approx(0.75, rel=1e-14)
    assert line.sinks["DETECTED"] + line.sinks["ESCAPED"] == pytest.approx(0.25, rel=1e-12)
    assert "SURFACE_SCATTER" not in line.sinks  # (traced: its rays' fates replace it)
    assert line.observed_W["D"] == pytest.approx(line.sinks["DETECTED"], rel=1e-12)


def test_a_mirror_keeps_one_minus_the_scattered_share_in_its_specular_beam():
    f, reflectance = 0.02, 0.9
    coating = ConstantCoating(reflectance_s=reflectance, reflectance_p=reflectance, transmittance_s=0.0, transmittance_p=0.0)
    src = B.laser("S", wavelength_nm=633.0, power_W=1.0, waist_radius_mm=0.2, position=(-50.0, 0.0, 50.0), direction=(INV, 0.0, -INV))
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), clear_radius_mm=12.0)
    face = mirror.surfaces[0].model_copy(update={"interaction": MirrorInteraction(coating=coating, scatter=SurfaceScatter(fraction=f, sample_count=256))})
    mirror = mirror.model_copy(update={"surfaces": (face,)})
    out = B.observation_plane("OUT", position=(50.0, 0.0, 50.0), normal=(INV, 0.0, INV), radius_mm=5.0, absorbing=True)
    result = _validate(ProjectSnapshot(name="rough mirror", components=(src, mirror), observations=(out,)))
    (arrival,) = result.observations[0].arrivals
    assert arrival.beam.power_W == pytest.approx(reflectance * (1.0 - f), rel=1e-12)
    (line,) = result.power_budget.lines
    assert line.conserved
    assert line.sinks["COATING_ABSORPTION"] == pytest.approx((1.0 - reflectance) * (1.0 - f), rel=1e-12)


def test_a_beam_that_meets_the_back_of_a_reflector_does_not_scatter():
    src = B.laser("S", wavelength_nm=633.0, power_W=1.0, waist_radius_mm=0.2, position=(0.0, 0.0, -50.0), direction=(0.0, 0.0, 1.0))
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), clear_radius_mm=12.0)
    face = mirror.surfaces[0].model_copy(update={"interaction": MirrorInteraction(scatter=SurfaceScatter(fraction=0.5, sample_count=64))})
    mirror = mirror.model_copy(update={"surfaces": (face,)})
    result = _validate(ProjectSnapshot(name="back side", components=(src, mirror), observations=()))
    (line,) = result.power_budget.lines
    assert "SURFACE_SCATTER" not in line.sinks and not result.sampled_sources
    assert line.sinks["BLOCKED_BACK_SIDE"] == pytest.approx(1.0)


def test_scattered_light_the_run_does_not_trace_stays_unknown():
    # the beams alone (as control readings and the network export read them): the scattered power keeps its booking
    scatter = SurfaceScatter(fraction=0.1, sample_count=64)
    snapshot = _dump_scene(AbsorberInteraction(scatter=scatter))
    outcome = run_physics(snapshot, NumericalConfig(), plan=BEAMS_ALONE)
    flows = [f for f in outcome.trace.power_flows if f.kind == "SURFACE_SCATTER"]
    assert len(flows) == 1 and flows[0].power_W == pytest.approx(0.1) and flows[0].surface_id == "DUMP/dump"
    issue = next(i for i in outcome.issues if i.code == "SCATTERED_LIGHT_NOT_FOLLOWED")
    # (a notice: the planes carry the consequence in their own state and untraced_power_W)
    assert issue.status == "NOT_APPLICABLE" and issue.location == "surface:DUMP/dump" and not issue.affected_targets
    assert issue.suggested_action == "read power_upper_bound_W and untraced_power_W of the planes"
    assert outcome.observations[0].power_complete is False


def _two_rough_dumps() -> ProjectSnapshot:
    """Two beams, 1 W and 0.5 W, each onto its own absorber that scatters 10 % back: two scatter events, 0.1 W and 0.05 W."""
    scatter = SurfaceScatter(fraction=0.1, side="reflect", sample_count=256)
    parts = []
    for name, x, power in (("S", 0.0, 1.0), ("T", 200.0, 0.5)):
        parts.append(B.laser(name, wavelength_nm=633.0, power_W=power, waist_radius_mm=0.2, position=(x - 50.0, 0.0, 50.0),
                             direction=(INV, 0.0, -INV)))
        dump = B.beam_dump(f"DUMP_{name}", position=(x, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius_mm=12.0)
        face = dump.surfaces[0].model_copy(update={"interaction": AbsorberInteraction(scatter=scatter)})
        parts.append(dump.model_copy(update={"surfaces": (face,)}))
    return ProjectSnapshot(name="two rough dumps", components=tuple(parts), observations=())


def _with_origin_budget(snapshot: ProjectSnapshot, budget: int | None):
    numerical = NumericalConfig() if budget is None else NumericalConfig(max_scatter_origins=budget)
    return validate(ValidationRequest(project=snapshot, configuration=EngineConfiguration(run_layout=False, run_control=False,
                                                                                         numerical=numerical)))


def test_the_scatter_origin_budget_of_the_run_decides_how_many_events_are_traced():
    """``max_scatter_origins`` (the run's compute budget) is the number of scatter events traced as rays, the strongest
    first. With one of two: the weaker event keeps its booking -- 10 % of 0.5 W, closed form -- and its surface's issue states
    the budget and what lies beyond it; with two, as at the default, both are traced and each line's ledger closes over its
    rays."""
    one = _with_origin_budget(_two_rough_dumps(), 1)
    lines = {line.source_id: line for line in one.power_budget.lines}
    assert "SURFACE_SCATTER" not in lines["S"].sinks and lines["S"].conserved  # (the stronger event: traced)
    assert lines["T"].sinks["SURFACE_SCATTER"] == pytest.approx(0.05, rel=1e-12) and lines["T"].conserved
    assert lines["T"].sinks["ABSORBED"] == pytest.approx(0.45, rel=1e-12)
    assert [s.scatter_surface_id for s in one.sampled_sources] == ["DUMP_S/dump"]
    (issue,) = [i for i in one.issues if i.code == "SCATTERED_LIGHT_NOT_FOLLOWED"]
    # (a notice: the planes carry the light not followed in their own state and bounds)
    assert issue.location == "surface:DUMP_T/dump" and issue.status == "NOT_APPLICABLE"
    context = issue.numerical_context
    assert (context["max_scatter_origins"], context["origins_beyond_budget"]) == (1, 1)
    assert context["power_beyond_budget_W"] == context["scattered_power_W"] == pytest.approx(0.05, rel=1e-12)
    assert "1 of its scatter events lie beyond the 1 strongest of the run" in issue.message
    assert "raise compute_budget.max_scatter_origins (set_compute_budget)" in issue.suggested_action
    two, default = _with_origin_budget(_two_rough_dumps(), 2), _with_origin_budget(_two_rough_dumps(), None)
    for result in (two, default):
        assert not [i for i in result.issues if i.code == "SCATTERED_LIGHT_NOT_FOLLOWED"]
        assert sorted(s.scatter_surface_id for s in result.sampled_sources) == ["DUMP_S/dump", "DUMP_T/dump"]
        for line in result.power_budget.lines:
            share = {"S": 1.0, "T": 0.5}[line.source_id]
            assert line.conserved and "SURFACE_SCATTER" not in line.sinks
            assert line.sinks["ABSORBED"] + line.sinks.get("ESCAPED", 0.0) == pytest.approx(share, rel=1e-12)
            assert line.sinks["ABSORBED"] >= 0.9 * share * (1 - 1e-12)
    # the budget changes nothing it does not reach: the same ledger at 2 as at the default 64
    assert [line.sinks for line in two.power_budget.lines] == [line.sinks for line in default.power_budget.lines]
    # ... and the traced event has the same numbers whether or not the other is traced
    assert lines["S"].sinks == {line.source_id: line for line in default.power_budget.lines}["S"].sinks


def test_light_a_sampled_ray_scatters_again_is_booked_and_not_followed():
    # a rough mirror facing the dump: the dump's scattered rays meet the mirror, which scatters again (not followed)
    scatter = SurfaceScatter(fraction=0.5, sample_count=256)
    snapshot = _dump_scene(AbsorberInteraction(scatter=scatter), h=30.0, a=2.0)
    mirror = B.mirror("M", position=(0.0, 0.0, 20.0), normal=(0.0, 0.0, -1.0), clear_radius_mm=6.0)
    face = mirror.surfaces[0].model_copy(update={"interaction": MirrorInteraction(scatter=SurfaceScatter(fraction=0.3, sample_count=64))})
    snapshot = snapshot.model_copy(update={"components": (*snapshot.components, mirror.model_copy(update={"surfaces": (face,)}))})
    result = _validate(snapshot)
    (line,) = result.power_budget.lines
    assert line.conserved
    assert line.sinks["SURFACE_SCATTER"] > 0.0  # (the mirror's share of the rays that reached it)
    # (the mirror's, and the dump's own where the mirror sends rays back onto it)
    issues = [i for i in result.issues if i.code == "SCATTERED_LIGHT_NOT_FOLLOWED"]
    assert {i.location for i in issues} == {"surface:M/face", "surface:DUMP/dump"}
    # (no event lies beyond the budget in force: the issues say so)
    assert all((i.numerical_context["max_scatter_origins"], i.numerical_context["origins_beyond_budget"]) == (64, 0) for i in issues)
    assert not any("compute_budget.max_scatter_origins" in i.message + (i.suggested_action or "") for i in issues)
    assert math.fsum(i.numerical_context["scattered_power_W"] for i in issues) == pytest.approx(line.sinks["SURFACE_SCATTER"], rel=1e-12)
    assert result.observations[0].power_complete is False


def test_scatter_is_reproducible_and_seeded():
    snap = _dump_scene(AbsorberInteraction(scatter=SurfaceScatter(fraction=0.1, sample_count=512, seed=3)))
    a = _validate(snap).observations[0].sampled_light[0]
    b = _validate(snap).observations[0].sampled_light[0]
    assert a.crossings == b.crossings
    other = _validate(_dump_scene(AbsorberInteraction(scatter=SurfaceScatter(fraction=0.1, sample_count=512, seed=4))))
    assert other.observations[0].sampled_light[0].crossings != a.crossings


# ------------------------------------------------------------------------------------------------ contracts


def test_a_surface_without_scatter_keeps_its_earlier_form():
    assert "scatter" not in RefractInteraction().model_dump()
    assert "scatter" not in MirrorInteraction().model_dump()
    assert AbsorberInteraction().model_dump() == {"kind": "absorber"}
    material = ConstantIndexMaterial(material_id="g", valid_wavelength_nm=(400.0, 900.0), refractive_index=1.5)
    assert "scattering_coefficient_per_mm" not in material.model_dump() and "anisotropy_g" not in material.model_dump()


@pytest.mark.parametrize("fields, message", [
    ({"bsdf": "ABG", "fraction": 0.1}, "bsdf"),
    ({"bsdf": "TABULATED", "fraction": 0.1}, "bsdf"),
    ({"bsdf": "COSINE_POWER", "fraction": 0.1}, "exponent"),
    ({"bsdf": "LAMBERTIAN", "exponent": 2.0, "fraction": 0.1}, "exponent"),
    ({"fraction": 0.0}, "fraction"),
    ({"fraction": 1.5}, "fraction"),
])
def test_unsupported_scatter_statements_are_refused(fields, message):
    with pytest.raises(ValidationError, match=message):
        SurfaceScatter(**fields)


def test_reflectors_and_absorbers_scatter_back_only():
    with pytest.raises(ValidationError, match="reflect"):
        MirrorInteraction(scatter=SurfaceScatter(fraction=0.1, side="transmit"))
    with pytest.raises(ValidationError, match="reflect"):
        AbsorberInteraction(scatter=SurfaceScatter(fraction=0.1, side="transmit"))
    with pytest.raises(ValidationError, match="anisotropy_g"):
        ConstantIndexMaterial(material_id="g", valid_wavelength_nm=(400.0, 900.0), refractive_index=1.5, anisotropy_g=0.5)


# ------------------------------------------------------------------------------------------------ bulk scattering (Project)

ALPHA, MU_S, G, THICKNESS, INDEX, FRACTION = 0.01, 0.02, 0.8, 5.0, 1.5, 0.05


def _milky_plate_document(*, scatter: dict | None = None, mu_s: str | None = "0.02") -> dict:
    glass = {"material_id": "MILKY", "display_name": "Milky glass", "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "1.5"},
             "valid_band_nm": {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"}, "absorption_coefficient_per_mm": "0.01"}
    if mu_s is not None:
        glass.update(scattering_coefficient_per_mm=mu_s, anisotropy_g="0.8")
    front = surface(1, "0", clear="9", **({"scatter": scatter} if scatter else {}))
    model = {"model_id": "plate", "model": sequence([front, surface(2, "5", clear="9")], ["MILKY"])}
    doc = project([source("SRC", (-100.0, 0.0, 0.0), wavelength_nm=633.0, waist_mm=0.5),
                   declared_component("PLATE", "window", "plate", (0.0, 0.0, 0.0))],
                  [plane("D", (50.0, 0.0, 0.0), radius_mm=5.0), plane("BACK", (-20.0, 0.0, 0.0), radius_mm=15.0)])
    return with_models(doc, [model], [glass])


def test_bulk_scattering_extinguishes_the_beam_and_books_the_scattered_share():
    scatter = {"bsdf": "LAMBERTIAN", "fraction": "0.05", "side": "REFLECT", "sample_count": 1024}
    result = run(_milky_plate_document(scatter=scatter)).result
    power_w = 1e-3  # (the test laser's power)
    reflectance = ((INDEX - 1.0) / (INDEX + 1.0)) ** 2
    expected = power_w * (1.0 - FRACTION) * (1.0 - reflectance) ** 2 * math.exp(-(ALPHA + MU_S) * THICKNESS)
    plane_d = next(o for o in result.observations if o.observation_id == "D")
    assert plane_d.total_power_W == pytest.approx(expected, rel=1e-12)
    (line,) = result.power_budget.lines
    assert line.conserved and abs(line.residual_W) < 1e-15
    entering = power_w * (1.0 - FRACTION) * (1.0 - reflectance)
    lost = entering * (1.0 - math.exp(-(ALPHA + MU_S) * THICKNESS))
    assert line.sinks["BULK_SCATTER"] == pytest.approx(lost * MU_S / (ALPHA + MU_S), rel=1e-12)
    assert line.sinks["BULK_ABSORPTION"] == pytest.approx(lost * ALPHA / (ALPHA + MU_S), rel=1e-12)
    # the scattered light is not followed: no plane of the line reads a complete power, and the run says why
    assert all(o.power_complete is False for o in result.observations)
    # (the lit plane behind the plate: its traced value is a lower end, and its entry names the scattered light, which
    # nothing bounds on a transparent plane)
    assert plane_d.quantity_state == "BOUNDED" and plane_d.power_upper_bound_W is None
    assert plane_d.untraced_power_W >= line.sinks["BULK_SCATTER"]
    entry = plane_d.value_basis["total_power_W"]
    scattered = [o for o in entry.omitted if o.code == "SCATTERED_LIGHT_NOT_FOLLOWED"]
    assert entry.kind.value == "LOWER_BOUND" and scattered and all(o.part.value == "NOT_FOLLOWED" and o.upper is None for o in scattered)
    issue = next(i for i in result.issues if i.code == "SCATTERED_LIGHT_NOT_FOLLOWED" and i.location == "material:project:MILKY")
    assert issue.numerical_context["anisotropy_g"] == G and issue.numerical_context["scattering_coefficient_per_mm"] == MU_S
    # the front face's scattered light reaches the plane before the plate (it is part of that plane's power)
    back = next(o for o in result.observations if o.observation_id == "BACK")
    (scattered,) = back.sampled_light
    assert scattered.scatter_surface_id == "PLATE/S1" and scattered.counted_crossing_count > 0


def test_bulk_scatter_at_or_below_the_negligible_level_is_left_out_inside_the_bound():
    # mu_s = 1e-10 /mm books about 4.7e-13 W of the 1 mW line (below its negligible level 1e-12 W): no notice, every plane
    # reads as without scattering, and the plane's upper bound grows by at least the scattered power
    clear = run(_milky_plate_document(mu_s=None)).result
    faint = run(_milky_plate_document(mu_s="0.0000000001")).result
    (line,) = faint.power_budget.lines
    entering = 1e-3 * (1.0 - ((INDEX - 1.0) / (INDEX + 1.0)) ** 2)
    assert line.sinks["BULK_SCATTER"] == pytest.approx(entering * 1e-10 * THICKNESS, rel=1e-3)
    assert line.sinks["BULK_SCATTER"] < NumericalConfig().energy_conservation_tol * 1e-3
    assert not any(i.code == "SCATTERED_LIGHT_NOT_FOLLOWED" for i in faint.issues)
    assert faint.status == clear.status
    for a, b in zip(clear.observations, faint.observations, strict=True):
        assert (b.quantity_state, b.power_complete, b.status) == (a.quantity_state, a.power_complete, a.status)
        assert b.power_upper_bound_W is not None
        assert b.power_upper_bound_W - b.total_power_W - b.untraced_power_W >= (
            a.power_upper_bound_W - a.total_power_W - a.untraced_power_W) + line.sinks["BULK_SCATTER"]


def test_a_material_without_scattering_keeps_its_beam_and_ledger():
    result = run(_milky_plate_document(mu_s=None)).result
    (line,) = result.power_budget.lines
    assert "BULK_SCATTER" not in line.sinks and not result.sampled_sources
    assert not any(i.code == "SCATTERED_LIGHT_NOT_FOLLOWED" for i in result.issues)
    # (the plane is bounded by the plate's untraced reflections alone, as before)
    assert next(o for o in result.observations if o.observation_id == "D").quantity_state == "BOUNDED"


def test_project_scatter_refusals():
    doc = _milky_plate_document(scatter={"bsdf": "LAMBERTIAN", "fraction": "0.1", "side": "TRANSMIT"})
    doc["optical_models"][0]["model"]["surfaces"][0]["traced"] = "REFLECTED"
    with pytest.raises(LocalRunError, match="side REFLECT"):
        run(doc)
    doc = _milky_plate_document(scatter={"bsdf": "HARVEY", "fraction": "0.1", "side": "REFLECT"})
    with pytest.raises(LocalRunError, match="bsdf"):
        run(doc)
    doc = _milky_plate_document()
    del doc["model_materials"][0]["scattering_coefficient_per_mm"]
    with pytest.raises(LocalRunError, match="anisotropy_g"):
        run(doc)


def test_a_published_catalog_does_not_carry_surface_scatter():
    from os_optics._vendor.shared_contracts.design_catalog import PublicOpticalCatalogV2

    document = copy.deepcopy(catalog_document())
    for product in document["products"]:
        for model in product["simulation_models"]:
            if model.get("surfaces"):
                model["surfaces"][0]["scatter"] = {"bsdf": "LAMBERTIAN", "fraction": "0.01", "side": "REFLECT"}
                break
        else:
            continue
        break
    with pytest.raises(ValidationError, match="surface scatter is declared in Project optical models"):
        PublicOpticalCatalogV2.model_validate(document)
