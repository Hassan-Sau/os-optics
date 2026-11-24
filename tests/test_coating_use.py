"""The left-out light's use of a coating from its dense side, and the run-level notice of a withheld exposure bound (S2).

An air-equivalent coating record is evaluated at the left-out light's own bounded extent, converted by the Snell invariant, and
refused where that extent leaves the record's range or has no real air-equivalent angle (``validation/coating_use.py``). The
expected angles here come from Snell's law, a Gaussian beam's far-field angle and real rays traced through the elements in this
file, never from the engine under test. Every power rating is fixture data declared in the test project.
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import EngineeringComponentSnapshot, PowerRating, ProjectSnapshot, Target
from os_optics.contracts.materials import ConstantCoating, DeclaredCoatingValidity, TabulatedCoating
from os_optics.contracts.optics import CircularAperture, PlaneSurface, RefractInteraction
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.optics.rays.tracer import Tracer
from os_optics.scene.lower import lower_scene
from os_optics.validation import resonance
from os_optics.validation.coating_use import CoatingUse, NO_AIR_EQUIVALENT, NOT_BOUNDED, OUTSIDE_VALIDITY, OUTSIDE_WAVELENGTH, coating_uses, judge
from os_optics.validation.exposures import _Withheld, _withheld_notices
from os_optics.validation.resonance import NEGATIVE, POSITIVE
from os_optics.validation.untraced import follow_untraced

from helpers import run
from test_group_delay_and_returns import _bk7

LAM = 1064.0
WAIST = 1.0
LEVEL = 1e-9  # negligible_power_fraction x a 1 W line
RATING = 1e12
FIX = "fixture rating, not product data"
N = _bk7(LAM * 1e-3)
THETA = LAM * 1e-6 / (math.pi * WAIST)  # the source's far-field half angle in air (rad)


def declared(low: float = 0.0, high: float = 10.0, reference: str = "AIR_EQUIVALENT", slack: float = 1e-6) -> DeclaredCoatingValidity:
    return DeclaredCoatingValidity(record_id="REC", aoi_reference=reference, min_deg=low, max_deg=high, slack_deg=slack)


def coating(r: float = 0.01, validity: DeclaredCoatingValidity | None = None) -> ConstantCoating:
    return ConstantCoating(reflectance_s=r, reflectance_p=r, transmittance_s=1.0 - r, transmittance_p=1.0 - r, declared_validity=validity)


def coated(component, model):
    surfaces = tuple(s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": model})}) for s in component.surfaces)
    return component.model_copy(update={"surfaces": surfaces})


def rating(value: float = RATING) -> PowerRating:
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=LAM, status="DECLARED", basis=FIX, value_definition="PEAK")


def rated(project: ProjectSnapshot, device: str, value: float = RATING, maximum: float = 1.0) -> ProjectSnapshot:
    parts = tuple(c.model_copy(update={"power_ratings": (rating(value),)}) if c.instance_id == device else c for c in project.components)
    target = Target(target_id=f"u-{device}", device_id=device, metric="power_rating_utilization", maximum=maximum)
    return project.model_copy(update={"components": parts, "targets": (target,)})


def uses_of(project: ProjectSnapshot) -> dict:
    """``coating_uses`` of one run's followed light at ``LAM``."""
    cfg = NumericalConfig()
    library = MaterialLibrary(project.materials, cfg)
    scene, _ = lower_scene(project, cfg, library)
    trace = Tracer(scene, library, cfg, include_pupil=True).run()
    probes = follow_untraced(trace, scene, library, cfg)
    hits = [*trace.surface_hits, *(h for _, outcome in probes.results for h in outcome.surface_hits)]
    return coating_uses(scene, library, cfg, hits, {(i.source_id, i.spectral_index): i.power_W for i in trace.inputs}).get(LAM, {})


def k_of(power: float) -> float:
    """The envelope's ``K`` of a followed beam: its share past ``K`` radii is an eighth of the level (``0.5 exp(-2 K^2) P = L / 8``)."""
    return math.sqrt(0.5 * math.log(4.0 * power / LEVEL))


def parent_power(r: float) -> float:
    """The largest followed power inside a coated plate or lens whose reflection is left out (its reflection at most the level):
    1 W in, ``T`` on through the first face, each further pass ``r`` of the last."""
    power = 1.0 - r
    while power * r > LEVEL:
        power *= r
    return power


def tilted_window(incidence_deg: float, model) -> ProjectSnapshot:
    """A plate on +x, its normal along x, met by a collimated beam at ``incidence_deg`` in air."""
    c, s = math.cos(math.radians(incidence_deg)), math.sin(math.radians(incidence_deg))
    source = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=WAIST, position=(-100.0 * c, -100.0 * s, 0.0), direction=(c, s, 0.0))
    plate = coated(B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0)), model)
    return ProjectSnapshot(components=(source, plate))


# ------------------------------------------------------------------------------------------------ the record on a compiled coating


def _record(coating_id: str):
    from local_helpers import load_test_catalog
    from os_optics.local import engine_coating

    return engine_coating(load_test_catalog().coating(coating_id)).declared_validity


def test_a_compiled_coating_carries_its_record_and_the_engines_own_allowances():
    ranged = _record("ANTI_REFLECTION.400-1100R0.005T0.995.AOI0-10")
    assert (ranged.record_id, ranged.aoi_reference, ranged.min_deg, ranged.max_deg) == ("ANTI_REFLECTION.400-1100R0.005T0.995.AOI0-10", "AIR_EQUIVALENT", 0.0, 10.0)
    assert ranged.slack_deg == pytest.approx(1e-6)  # (a range: AOI_TOLERANCE_DEG)
    assert _record("ANTI_REFLECTION.400-1100R0.01T0.99.AOI45-45").slack_deg == pytest.approx(3.0 + 1e-6)  # (one angle of an AR coating)
    assert _record("BEAMSPLITTER.400-1100R0.5T0.5.AOI45-45").slack_deg == pytest.approx(2.0 + 1e-6)
    assert ranged.wavelength_bands_nm == ((400.0, 1100.0),)


# ------------------------------------------------------------------------------------------------ the extent on a plate


def test_the_extent_on_a_plate_is_snell_s_law_of_the_beams_far_field_angle():
    """Between parallel faces the glass angle is Snell's ``t`` of the air-side ``i``, and a ray of the beam deviates from the
    chief ray by at most ``K theta_air / n``: the left-out reflection of the last followed pass meets the other face from the
    glass within ``t +- K theta_air / n``, air-equivalent ``asin(n sin(...))``."""
    incidence = 6.0
    uses = uses_of(tilted_window(incidence, coating(0.01, declared())))
    assert set(uses) == {("W/front", NEGATIVE)} or set(uses) == {("W/front", POSITIVE)}
    use = next(iter(uses.values()))
    spread = k_of(parent_power(0.01)) * THETA / N
    glass = math.asin(math.sin(math.radians(incidence)) / N)
    assert use.high == pytest.approx(glass + spread, rel=1e-12)
    assert use.low == pytest.approx(glass - spread, rel=1e-12)
    assert use.air_deg[1] == pytest.approx(math.degrees(math.asin(N * math.sin(glass + spread))), rel=1e-12)
    assert use.air_deg[0] == pytest.approx(math.degrees(math.asin(N * math.sin(glass - spread))), rel=1e-12)
    assert use.index == pytest.approx(N, rel=1e-12)
    assert use.refused is None and use.record_id == "REC" and use.valid_aoi_deg == (0.0, 10.0)


def edge_incidence(limit_deg: float) -> float:
    """The air-side incidence that puts the extent's top on the record's edge (Snell, the far-field angle in the glass)."""
    glass = math.asin(math.sin(math.radians(limit_deg)) / N) - k_of(parent_power(0.01)) * THETA / N
    return math.degrees(math.asin(N * math.sin(glass)))


@pytest.mark.parametrize("step, refused", [(-0.001, None), (0.001, OUTSIDE_VALIDITY)])
def test_the_validity_edge_refuses_just_outside_and_not_just_inside(step, refused):
    use = next(iter(uses_of(tilted_window(edge_incidence(10.0) + step, coating(0.01, declared()))).values()))
    assert use.refused == refused
    assert (use.air_deg[1] > 10.0) == (refused is not None)


@pytest.mark.parametrize("step", [-0.001, 0.001])
def test_the_edge_decides_the_bound_of_a_whole_run(step):
    project = rated(tilted_window(edge_incidence(10.0) + step, coating(0.01, declared())), "W")
    result = run(project)
    face = next(s for s in result.exposures.surfaces if s.surface_id == "W/front")
    notices = [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"]
    if step < 0.0:
        assert face.ratings[0].exposure_upper_bound is not None and not notices
    else:
        assert face.ratings[0].exposure_upper_bound is None and "FAINT_LIGHT_RECIRCULATES" in face.reason_codes
        (notice,) = notices
        ctx = notice.numerical_context
        assert notice.status.value == "NOT_APPLICABLE" and notice.location == "surface:W/front"
        assert ctx["coating_record"] == "REC" and ctx["cause"] == "COATING_USE_REFUSED"
        assert ctx["incident_medium"] == "N-BK7" and ctx["incident_index"] == pytest.approx(N, rel=1e-12)
        assert ctx["air_equivalent_deg"][1] > 10.0 and ctx["extent_in_medium_deg"][1] < 10.0
        assert ctx["quantity"] == "exposure_upper_bound" and ctx["wavelength_nm"] == [LAM]
        # the other face reads the record's stated value: its bound is that of the same run without the refusal
        back = next(s for s in result.exposures.surfaces if s.surface_id == "W/back")
        same = run(rated(tilted_window(edge_incidence(10.0) + step, coating(0.01)), "W"))
        assert back.ratings[0].exposure_upper_bound == next(s for s in same.exposures.surfaces if s.surface_id == "W/back").ratings[0].exposure_upper_bound


def test_a_single_design_angle_takes_its_allowance():
    """A record of one angle, with the coating kind's tolerance as its allowance (3 deg for an anti-reflection coating)."""
    model = coating(0.01, declared(5.0, 5.0, slack=3.0 + 1e-6))
    assert next(iter(uses_of(tilted_window(7.5, model)).values())).refused is None
    assert next(iter(uses_of(tilted_window(2.5, model)).values())).refused is None
    assert next(iter(uses_of(tilted_window(8.5, model)).values())).refused == OUTSIDE_VALIDITY
    assert next(iter(uses_of(tilted_window(1.5, model)).values())).refused == OUTSIDE_VALIDITY


# ------------------------------------------------------------------------------------------------ the extent on a lens


def trace_marginal(x0: float, u0: float, r: float, thickness: float, reflections: int, n: float) -> float:
    """A real meridional ray of a biconvex lens (vertices at -t/2, +t/2; radii r, -r), from the first face's vertex plane at
    height ``x0``, slope ``u0``: refracted in at the first face, reflected ``reflections`` times inside (second face first), and
    its incidence (rad) where it next meets a face. Plain trigonometry, not the engine's."""
    half = thickness / 2.0

    def hit(z, y, cz_dir, face):
        """Intersection with the sphere of ``face`` (0: centre at -half + r; 1: centre at half - r), moving along ``cz_dir``."""
        centre = -half + r if face == 0 else half - r
        dz, dy = cz_dir
        oz, oy = z - centre, y
        b = oz * dz + oy * dy
        c = oz * oz + oy * oy - r * r
        root = math.sqrt(b * b - c)
        # (the vertex side of each sphere: the first face at negative z of its centre, the second at positive z)
        t = -b - root if (face == 0) == (dz > 0.0) else -b + root
        pz, py = z + t * dz, y + t * dy
        nz, ny = (pz - centre) / r, py / r  # outward from the sphere's centre
        return pz, py, nz, ny

    # the first face from air
    d = (math.cos(math.atan(u0)), math.sin(math.atan(u0)))
    z, y, nz, ny = hit(-half - 1e-3, x0, d, 0)
    c = -(d[0] * nz + d[1] * ny)
    ratio = 1.0 / n
    k = 1.0 - ratio * ratio * (1.0 - c * c)
    d = (ratio * d[0] + (ratio * c - math.sqrt(k)) * nz, ratio * d[1] + (ratio * c - math.sqrt(k)) * ny)
    face = 1
    for _ in range(reflections):
        z, y, nz, ny = hit(z, y, d, face)
        c = d[0] * nz + d[1] * ny
        d = (d[0] - 2.0 * c * nz, d[1] - 2.0 * c * ny)
        face = 1 - face
    z, y, nz, ny = hit(z, y, d, face)
    c = abs(d[0] * nz + d[1] * ny)
    return math.acos(min(c, 1.0))


def test_the_extent_on_a_lens_holds_the_real_rays_of_the_envelope():
    """The Gaussian beam's ``K`` ellipse, launched as real rays at its waist and traced with their reflections through the lens,
    meets the face the left-out reflection lands on within the extent, and the extent is that close to them."""
    r, thickness = 100.0, 4.0
    lens = coated(B.singlet_lens("L", r1_mm=r, r2_mm=-r, thickness_mm=thickness, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0)),
                  coating(0.01, declared(0.0, 60.0)))
    # (the waist on the first vertex: the launch is the ellipse there)
    source = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=WAIST, position=(-2.0 - 1e-3, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    use = next(iter(uses_of(ProjectSnapshot(components=(source, lens))).values()))
    assert use.surface_id == "L/front" and use.refused is None
    parent = parent_power(0.01)
    k = k_of(parent)
    # the followed passes: back 0.99, front 9.9e-3, back 9.9e-5, front 9.9e-7, back 9.9e-9 (parent): five reflections inside
    assert parent == pytest.approx(0.99e-8)
    widest = 0.0
    for step in range(720):
        angle = 2.0 * math.pi * step / 720.0
        widest = max(widest, trace_marginal(k * WAIST * math.cos(angle), k * THETA * math.sin(angle), r, thickness, 5, N))
    assert use.high >= widest
    assert use.high == pytest.approx(widest, rel=0.03)


def test_a_beam_that_fills_a_strong_lens_is_not_bounded():
    """The beam is too large against the surface for its Gaussian state, and inside a lens the leg between the faces bounds no
    direction: the use is refused whatever the chief ray does (it is at normal incidence)."""
    lens = coated(B.singlet_lens("L", r1_mm=12.0, r2_mm=-12.0, thickness_mm=4.0, clear_radius_mm=6.0, material="N-BK7", position=(0.0, 0.0, 0.0)),
                  coating(0.01, declared()))
    source = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=4.0, position=(-50.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    uses = uses_of(ProjectSnapshot(components=(source, lens)))
    assert uses and all(use.refused in (NOT_BOUNDED, OUTSIDE_VALIDITY, NO_AIR_EQUIVALENT) for use in uses.values())
    assert any(use.refused == NOT_BOUNDED for use in uses.values())


# ------------------------------------------------------------------------------------------------ above the critical angle


def right_angle_prism(model) -> ProjectSnapshot:
    """A beam along +x enters the face at x = 0, folds by total internal reflection at the hypotenuse (45 deg) and leaves
    through the face at y = 10; every face carries the coating."""
    ap = CircularAperture(radius_mm=8.0)
    s = 1.0 / math.sqrt(2.0)
    interaction = RefractInteraction(coating=model)

    def face(name, origin, normal):
        return PlaneSurface(surface_id=name, origin_mm=origin, normal=normal, aperture=ap, medium_positive="air", medium_negative="N-BK7", interaction=interaction)

    prism = EngineeringComponentSnapshot(
        instance_id="P", role="window", pose=B._pose((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        surfaces=(face("in", (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)), face("fold", (0.0, 0.0, 10.0), (0.0, -s, s)), face("out", (0.0, 10.0, 10.0), (0.0, 1.0, 0.0))))
    source = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=WAIST, position=(0.0, 0.0, -100.0), direction=(0.0, 0.0, 1.0))
    return ProjectSnapshot(components=(source, prism))


def test_a_glass_side_use_above_the_critical_angle_has_no_air_equivalent_angle():
    """45 deg in N-BK7 is above the critical angle: ``n sin(45 deg) > 1``. The table is never read there."""
    uses = uses_of(right_angle_prism(coating(0.01, declared(0.0, 90.0))))
    use = next(u for key, u in uses.items() if key[0] == "P/fold")
    assert N * math.sin(math.radians(45.0)) > 1.0
    assert use.refused == NO_AIR_EQUIVALENT and use.air_deg is None


def test_judge_at_the_critical_angle():
    critical = math.asin(1.0 / N)
    wide = declared(0.0, 90.0)
    air, refused = judge(0.0, critical * (1.0 - 1e-9), N, wide, LAM)
    assert refused is None and air[1] < 90.0
    assert judge(0.0, critical * (1.0 + 1e-9), N, wide, LAM) == (None, NO_AIR_EQUIVALENT)
    assert judge(0.0, math.pi / 2.0, N, wide, LAM) == (None, NOT_BOUNDED)


def test_a_wavelength_the_record_does_not_state_is_refused():
    def banded(*bands):
        return coating(0.01, DeclaredCoatingValidity(record_id="REC", aoi_reference="AIR_EQUIVALENT", min_deg=0.0, max_deg=10.0, slack_deg=1e-6, wavelength_bands_nm=bands))

    assert next(iter(uses_of(tilted_window(5.0, banded((400.0, 1100.0)))).values())).refused is None
    assert next(iter(uses_of(tilted_window(5.0, banded((400.0, 700.0), (1500.0, 1600.0)))).values())).refused == OUTSIDE_WAVELENGTH
    result = run(rated(tilted_window(5.0, banded((400.0, 700.0))), "W"))
    (notice,) = [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"]
    assert notice.numerical_context["coating_refusal"] == OUTSIDE_WAVELENGTH
    assert "states 400-700 nm" in notice.message and "1064 nm" in notice.message


# ------------------------------------------------------------------------------------------------ what reads as before


def _bounds(result):
    return {s.surface_id: (s.rating_status, [(c.exposure_lower_bound, c.exposure_upper_bound, c.exposure_value) for c in s.ratings], s.reason_codes)
            for s in result.exposures.surfaces}


def lens_train(model) -> ProjectSnapshot:
    source = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    return ProjectSnapshot(components=(source, *(coated(
        B.singlet_lens(f"L{i + 1}", r1_mm=101.32, r2_mm=-101.32, thickness_mm=4.0, clear_radius_mm=12.7, material="N-BK7", position=(60.0 * i, 0.0, 0.0)), model)
        for i in range(3))))


def test_an_incident_medium_record_reads_as_a_coating_without_a_record():
    plain = run(rated(lens_train(coating(0.0025)), "L2"))
    stated = run(rated(lens_train(coating(0.0025, declared(0.0, 1.0, "INCIDENT_MEDIUM"))), "L2"))
    assert _bounds(stated) == _bounds(plain) and stated.status == plain.status and stated.issues == plain.issues


def test_air_side_incidence_of_an_air_equivalent_record_is_not_judged():
    """A mirror in air at 45 deg against a 0-10 deg record: an air-side use, read as it always was."""
    fold = B.mirror_for_reflection("M", position=(0.0, 0.0, 0.0), d_in=(1.0, 0.0, 0.0), d_out=(0.0, 1.0, 0.0), clear_radius_mm=12.7)
    source = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=WAIST, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))

    def project(model):
        return rated(ProjectSnapshot(components=(source, coated(fold, model))), "M")

    assert uses_of(project(coating(0.9, declared()))) == {}
    assert _bounds(run(project(coating(0.9, declared())))) == _bounds(run(project(coating(0.9))))


def test_the_ar_lens_train_keeps_its_recorded_bounds():
    """The AR lens train of the ring-return census: a constant 0.25 % coating with no record, three singlets, 1064 nm, its rating at a
    50 mm test beam (the recorded bounds of ``probe_363``, re-recorded after the share beside a support was resolved upward
    to the ledger resolution, 887d482a6 and 2cb461bb1: both rose by about 1e-8 relative)."""
    project = lens_train(coating(0.0025))
    parts = tuple(c.model_copy(update={"power_ratings": (_catalog_rating(LAM, 1e6),)}) if c.instance_id == "L2" else c for c in project.components)
    result = run(project.model_copy(update={"components": parts}))
    bounds = {s.surface_id: s.ratings[0].exposure_upper_bound for s in result.exposures.surfaces if s.ratings}
    assert bounds["L2/front"] == pytest.approx(11475.816081, rel=1e-9)
    assert bounds["L2/back"] == pytest.approx(7395.672375, rel=1e-9)
    assert not [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"]


def test_ghosts_that_fill_a_lens_train_refuse_a_narrow_air_equivalent_record():
    """The same train under a 0-10 deg air-equivalent record: its faint ghosts' pupil rays are aberration dominated inside the
    lenses, whose faces bound no direction between them, so no extent is bounded there and the bound is withheld, named."""
    result = run(rated(lens_train(coating(0.0025, declared(0.0, 10.0))), "L2", 1e6))
    notices = [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD" and i.numerical_context["cause"] == "COATING_USE_REFUSED"]
    assert notices and {i.numerical_context["coating_refusal"] for i in notices} == {NOT_BOUNDED}
    assert all(i.numerical_context["coating_record"] == "REC" for i in notices)
    face = next(s for s in result.exposures.surfaces if s.surface_id == notices[0].location.removeprefix("surface:"))
    assert not face.ratings or face.ratings[0].exposure_upper_bound is None


def test_a_coating_without_a_record_serializes_as_before():
    assert "declared_validity" not in coating(0.01).model_dump(mode="json")
    assert "declared_validity" in coating(0.01, declared()).model_dump(mode="json")
    table = TabulatedCoating(wavelength_nm=(500.0, 600.0), aoi_deg=(0.0, 90.0), reflectance_s=((0.1, 0.1), (0.1, 0.1)), reflectance_p=((0.1, 0.1), (0.1, 0.1)),
                             transmittance_s=((0.9, 0.9), (0.9, 0.9)), transmittance_p=((0.9, 0.9), (0.9, 0.9)))
    assert "declared_validity" not in table.model_dump(mode="json")


def angle_table(validity: DeclaredCoatingValidity | None) -> TabulatedCoating:
    """R = 0.01 at 0 deg rising linearly to 0.31 at 30 deg, then flat."""
    row_r = (0.01, 0.31, 0.31)
    row_t = tuple(1.0 - r for r in row_r)
    return TabulatedCoating(wavelength_nm=(100.0, 20000.0), aoi_deg=(0.0, 30.0, 90.0), reflectance_s=(row_r, row_r), reflectance_p=(row_r, row_r),
                            transmittance_s=(row_t, row_t), transmittance_p=(row_t, row_t), declared_validity=validity)


def test_an_angle_resolved_table_is_read_at_the_air_equivalent_angle_from_the_glass():
    """The bound reads the table at the light's air-equivalent angle: at 20 deg in air, 12.97 deg in the glass, R(air
    equivalent) is what the glass-side face gets, not R(12.97 deg)."""
    project = tilted_window(20.0, angle_table(declared(0.0, 30.0)))
    cfg = NumericalConfig()
    library = MaterialLibrary(project.materials, cfg)
    scene, _ = lower_scene(project, cfg, library)
    uses = uses_of(project)
    (key, use), = uses.items()
    surface = next(s for s in scene.surfaces if s.surface_id == key[0])
    r, _ = resonance._coefficients(surface, key[1], 0.0, math.radians(30.0), library, LAM, uses)
    expected = 0.01 + 0.30 * use.air_deg[1] / 30.0
    assert use.air_deg[1] == pytest.approx(20.0, abs=0.1)
    assert r == pytest.approx(math.sqrt(expected), rel=1e-12)
    # (the in-glass angle would read about 0.14 here)
    assert r * r > 0.2


# ------------------------------------------------------------------------------------------------ the run-level notice


def uncoated_lens() -> ProjectSnapshot:
    source = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    lens = B.singlet_lens("L1", r1_mm=101.32, r2_mm=-101.32, thickness_mm=4.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    return ProjectSnapshot(components=(source, lens))


def test_a_pair_refusal_names_each_surface_at_run_level():
    result = run(rated(uncoated_lens(), "L1"))
    notices = {i.location: i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"}
    assert set(notices) == {"surface:L1/front", "surface:L1/back"}
    for location, notice in notices.items():
        sid = location.removeprefix("surface:")
        ctx = notice.numerical_context
        assert notice.status.value == "NOT_APPLICABLE"
        assert ctx["element_id"] == "L1" and ctx["surface_id"] == sid and ctx["quantity"] == "exposure_upper_bound"
        assert ctx["wavelength_nm"] == [LAM] and ctx["cause"] == "RETURN_PAIR_GAIN_BOUND_1"
        assert ctx["return_pair_surfaces"] == ["L1/front, L1/back"]
        assert notice.message.startswith(f"The exposure upper bound on {sid} is withheld")
        assert "state a coating" in notice.suggested_action and "to obtain a bound" not in notice.message
        for hedge in ("may be", "caution", "cannot guarantee", "unreliable"):
            assert hedge not in notice.message
    face = next(s for s in result.exposures.surfaces if s.surface_id == "L1/front")
    assert "FAINT_LIGHT_RECIRCULATES" in face.reason_codes and face.ratings[0].exposure_upper_bound is None
    assert result.summary.issue_counts["EXPOSURE_BOUND_WITHHELD"] == 2


def test_the_notice_changes_no_status_and_names_its_undecided_target():
    project = rated(uncoated_lens(), "L1")
    result = run(project)
    target = result.targets[0]
    assert target.status.value == "UNKNOWN"
    others = [i.status for i in result.issues if i.code.value != "EXPOSURE_BOUND_WITHHELD"] + [t.status for t in result.targets]
    assert result.status == max(others, key=lambda s: ["NOT_APPLICABLE", "PASS", "WARNING", "UNKNOWN", "OUT_OF_MODEL_SCOPE", "FAIL"].index(s.value))
    for notice in (i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"):
        assert notice.affected_targets == ("u-L1",)
    assert any(w.startswith("EXPOSURE_BOUND_WITHHELD") for w in result.manifest.warnings)


def test_a_right_angle_prism_names_each_face_it_leaves_without_a_bound():
    """R4 of the ring-return census on a solid: the three faces of a right-angle prism (total internal reflection at the hypotenuse) are three pairs
    at a gain bound of 1; each rated face has no upper bound, names ``FAINT_LIGHT_RECIRCULATES`` and has its own run-level
    notice, and the element's undecided target is named by every one of them."""
    import test_catalog3_physics as T
    from catalog3_products import catalog3_document
    from local_helpers import project as document
    from os_optics.local.lowering import lower_project

    wavelength = 633.0
    source = T._source_at(T._in_point(T.RIGHT_ANGLE), wavelength_nm=wavelength, power_w=1.0)
    lowered = lower_project(document([source, T._element("P", "prism", T.RIGHT_ANGLE)], []), catalog3_document()).request.project
    rating_633 = PowerRating(regime="CW", value=RATING, unit="W/cm2", wavelength_nm=wavelength, status="DECLARED", basis=FIX, value_definition="PEAK")
    parts = tuple(c.model_copy(update={"power_ratings": (rating_633,)}) if c.instance_id == "P" else c for c in lowered.components)
    target = Target(target_id="u-P", device_id="P", metric="power_rating_utilization", maximum=1.0)
    result = run(lowered.model_copy(update={"components": parts, "targets": (target,)}))
    faces = [s for s in result.exposures.surfaces if s.element_id == "P"]
    assert len(faces) == 3
    notices = {i.location: i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"}
    assert set(notices) == {f"surface:{s.surface_id}" for s in faces}
    for face in faces:
        assert face.ratings[0].exposure_upper_bound is None and "FAINT_LIGHT_RECIRCULATES" in face.reason_codes
        notice = notices[f"surface:{face.surface_id}"]
        ctx = notice.numerical_context
        assert notice.status.value == "NOT_APPLICABLE" and notice.affected_targets == ("u-P",)
        assert ctx["cause"] == "RETURN_PAIR_GAIN_BOUND_1" and ctx["wavelength_nm"] == [wavelength] and ctx["surface_id"] == face.surface_id
    assert result.targets[0].status.value == "UNKNOWN"


def test_a_coated_lens_with_a_finite_bound_has_no_notice():
    result = run(rated(lens_train(coating(0.0025)), "L2", 1e6))
    assert not [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"]


def test_a_closed_return_has_one_run_notice_and_no_pair_notice():
    from test_return_pairs import _constant, _coat, _ring

    ring = _ring()
    src, a, b, c = ring.components
    ring = ring.model_copy(update={"components": (src, _coat(a, _constant(1.0 - 1e-12), both=True), b, c)})
    parts = tuple(x.model_copy(update={"power_ratings": (rating(100.0),)}) if x.instance_id == "B" else x for x in ring.components)
    result = run(ring.model_copy(update={"components": parts}))
    # (no issue per surface and no pair notice: the run's one notice of the closed return names B's face)
    assert not [i for i in result.issues if i.location == "surface:B/face"]
    (notice,) = [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"
                 and i.numerical_context["quantity"] == "exposure_upper_bound"]
    assert (notice.status.value, notice.numerical_context["cause"]) == ("NOT_APPLICABLE", "CLOSED_RETURN_NOT_BOUNDED")
    assert "B/face" in notice.numerical_context["surface_ids"] and "B" in notice.numerical_context["element_ids"]


@pytest.mark.parametrize("project", [lambda: uncoated_lens(), lambda: tilted_window(edge_incidence(10.0) + 0.001, coating(0.01, declared()))],
                         ids=["pair", "coating"])
def test_a_target_threshold_moves_no_bound_and_no_notice(project):
    """An optional target's threshold is no input of the physics."""
    device = project().components[1].instance_id
    results = [run(rated(project(), device, maximum=maximum)) for maximum in (1.0, 1e-6)]
    assert _bounds(results[0]) == _bounds(results[1])
    notices = [[(i.code, i.location, i.message) for i in r.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"] for r in results]
    assert notices[0] == notices[1] and notices[0]


def plane_lens(model=None, source_power: float = 1.0) -> ProjectSnapshot:
    """A lens (uncoated without ``model``) and a plane behind it, on one axis, no rating."""
    source = B.laser("S", wavelength_nm=LAM, power_W=source_power, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    lens = B.singlet_lens("L1", r1_mm=101.32, r2_mm=-101.32, thickness_mm=4.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    screen = B.observation_plane("screen", position=(100.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_mm=12.7)
    return ProjectSnapshot(components=(source, coated(lens, model) if model is not None else lens), observations=(screen,))


def test_a_run_without_a_rating_names_a_plane_a_pair_leaves_without_an_upper_bound():
    result = run(plane_lens())
    (plane,) = result.observations
    assert plane.power_upper_bound_W is None
    (notice,) = [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"]
    ctx = notice.numerical_context
    assert notice.status.value == "NOT_APPLICABLE" and notice.location == "observation:screen"
    assert ctx["quantity"] == "power_upper_bound" and ctx["wavelength_nm"] == [LAM] and ctx["cause"] == "RETURN_PAIR_GAIN_BOUND_1"
    assert ctx["return_pair_surfaces"] == ["L1/front, L1/back"]
    assert notice.message.startswith("The power upper bound on the plane screen is withheld at 1064 nm because L1/front and L1/back")
    assert any(w.startswith("EXPOSURE_BOUND_WITHHELD") for w in result.manifest.warnings)


def test_a_closed_return_names_a_lit_plane_it_leaves_without_an_upper_bound():
    """A three-mirror ring fed by a 1e-12 coupler transmission, a plane on the incoming beam, no rating: the plane is lit, its
    upper bound is withheld, and the notice with its own cause is the only change against the same run's other results."""
    from test_return_pairs import _constant, _coat, _ring

    ring = _ring()
    src, a, b, c = ring.components
    lit = B.observation_plane("lit", position=(-50.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=400.0)
    result = run(ring.model_copy(update={"components": (src, _coat(a, _constant(1.0 - 1e-12), both=True), b, c), "observations": (lit,)}))
    (plane,) = result.observations
    # (lit: the traced value is the lower end, the upper end open -- the left-out light may circulate in the ring)
    assert plane.lit and plane.power_upper_bound_W is None and plane.quantity_state.value == "BOUNDED"
    from os_optics.results import basis_of

    assert basis_of(plane, "total_power_W").omitted[0].part.value == "POSSIBLE_RETURN"
    (notice,) = [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"]
    ctx = notice.numerical_context
    assert notice.status.value == "NOT_APPLICABLE" and notice.location == "observation:lit"
    assert ctx["cause"] == "CLOSED_RETURN_NOT_BOUNDED" and ctx["quantity"] == "power_upper_bound"
    assert ctx["closed_return_surfaces"] == ["A/face, B/face, C/face"] and ctx["wavelength_nm"] == [LAM]
    assert "closed return through A/face, B/face, C/face" in notice.message
    assert result.status.value == "PASS"  # (the notice changes no status)


def test_a_plane_with_a_finite_power_bound_has_no_notice():
    result = run(plane_lens(coating(0.0025)))
    assert result.observations[0].power_upper_bound_W is not None
    assert not [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"]


def filled_lens_project(model, *, rated_lens: bool) -> ProjectSnapshot:
    """A beam that fills a strong lens (its uses not bounded), a plane behind it."""
    source = B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=4.0, position=(-50.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    lens = coated(B.singlet_lens("L1", r1_mm=12.0, r2_mm=-12.0, thickness_mm=4.0, clear_radius_mm=6.0, material="N-BK7", position=(0.0, 0.0, 0.0)), model)
    screen = B.observation_plane("screen", position=(30.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_mm=25.0)
    project = ProjectSnapshot(components=(source, lens), observations=(screen,))
    return rated(project, "L1") if rated_lens else project


@pytest.mark.parametrize("rated_lens", [False, True])
def test_a_refusal_leaves_a_planes_power_bound_as_the_record_states_and_withholds_the_faces(rated_lens):
    """The refusal speaks of the face's exposure bound: the plane's upper bound equals, bit for bit, that of the same coating
    without a record, and the refused face has no exposure upper bound."""
    refused = run(filled_lens_project(coating(0.01, declared()), rated_lens=rated_lens))
    plain = run(filled_lens_project(coating(0.01), rated_lens=rated_lens))
    assert refused.observations[0].power_upper_bound_W == plain.observations[0].power_upper_bound_W
    assert plain.observations[0].power_upper_bound_W is not None
    assert refused.observations[0].quantity_state == plain.observations[0].quantity_state
    assert not [i for i in refused.issues if i.location == "observation:screen" and i.code.value == "EXPOSURE_BOUND_WITHHELD"]
    if rated_lens:
        faces = {s.surface_id: s for s in refused.exposures.surfaces}
        notices = {i.location: i for i in refused.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"}
        assert "surface:L1/front" in notices and faces["L1/front"].ratings[0].exposure_upper_bound is None
        assert notices["surface:L1/front"].numerical_context["cause"] == "COATING_USE_REFUSED"


def _upper(result, sid: str):
    return next(s for s in result.exposures.surfaces if s.surface_id == sid).ratings[0].exposure_upper_bound


def test_a_refusal_leaves_another_surface_as_the_record_states():
    """A rated mirror behind a plate whose front face is refused: the mirror's bound is that of the run without a record, bit
    for bit."""
    mirror = coated(B.mirror("M", position=(100.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), clear_radius_mm=25.0), coating(0.5))

    def project(model):
        base = rated(tilted_window(edge_incidence(10.0) + 0.001, model), "W")
        return base.model_copy(update={"components": (*base.components, mirror.model_copy(update={"power_ratings": (rating(),)}))})

    refused, plain = run(project(coating(0.01, declared()))), run(project(coating(0.01)))
    assert _upper(refused, "W/front") is None
    assert _upper(plain, "M/face") is not None and _upper(refused, "M/face") == _upper(plain, "M/face")


def test_a_refusal_keeps_the_stated_factors_and_withholds_the_pair_it_leaves_at_a_gain_bound_of_1():
    """Both faces of a plate at normal incidence refused (no walk bounds their returns): their pair's gain bound is 1 in the refusal's reading; the factors a plane reads stay the
    record's stated ones, bit for bit."""
    project = tilted_window(0.0, coating(0.01, declared()))
    cfg = NumericalConfig()
    library = MaterialLibrary(project.materials, cfg)
    scene, _ = lower_scene(project, cfg, library)

    def use(surface_id: str) -> CoatingUse:
        return CoatingUse(surface_id, "W", NEGATIVE, LAM, "REC", "N-BK7", N, (0.0, 10.0), (), 0.0, math.radians(20.0), (0.0, 31.0), OUTSIDE_VALIDITY)

    stated = resonance.return_gains(scene, library, [LAM])[LAM]
    both = {LAM: {("W/front", NEGATIVE): use("W/front"), ("W/back", NEGATIVE): use("W/back")}}
    found = resonance.return_gains(scene, library, [LAM], both)[LAM]
    assert found.by_surface == stated.by_surface and found.reflecting == stated.reflecting
    assert found.factor("W/front") == stated.factor("W/front") < math.inf
    assert found.withheld == {"W/front", "W/back"} and [p.surfaces for p in found.refused_pairs] == [("W/front", "W/back")]
    assert found.exposure_factor("W/front") == math.inf
    one = {LAM: {("W/front", NEGATIVE): use("W/front")}}
    single = resonance.return_gains(scene, library, [LAM], one)[LAM]
    assert single.withheld == {"W/front"} and single.refused_pairs == ()


def test_a_face_a_refusal_holds_through_its_pair_has_its_own_notice():
    withheld = _Withheld(partners=((LAM, (("L/front", "M/face"),)),))
    (notice,) = _withheld_notices("M/face", "M", withheld)
    assert notice.numerical_context["cause"] == "COATING_USE_REFUSED_ON_FACING_SURFACE"
    assert notice.numerical_context["return_pair_surfaces"] == ["L/front, M/face"]
    assert notice.message.startswith("The exposure upper bound on M/face is withheld at 1064 nm because a refused coating use leaves L/front and M/face")


# ------------------------------------------------------------------------------------------------ the catalog's reference designs


def _example(kind: str, *, assume_lossless: bool = False):
    from os_optics.local.example import example_project_of_kind
    from os_optics.local.lowering import lower_project

    project = lower_project(example_project_of_kind(kind), None).request.project.model_copy(update={"targets": ()})
    if not assume_lossless:
        return project  # Public N-BK7 states index, not bulk loss: do not claim known transmission.
    materials = []
    for material in project.materials:
        if material.material_id == "catalog:N-BK7":
            assert "absorption_coefficient_per_mm" not in material.model_fields_set and material.bulk_loss is None
            material = material.model_copy(update={"absorption_coefficient_per_mm": 0.0})
        materials.append(material)
    # Only this test's idealized copy assumes alpha=0; every surface, index law and geometry stay identical.
    return project.model_copy(update={"materials": tuple(materials)})


def _catalog_rating(wavelength: float, value: float, diameter: float = 50.0) -> PowerRating:
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=wavelength, status="DECLARED", basis=FIX, value_definition="PEAK",
                       beam_diameter_mm=diameter, beam_diameter_definition="1/E2", polarization="LINEAR", polarization_condition="linear")


# (re-recorded -- the examples are designed and traced in air at 20 C and 101325 Pa, n_air ~ 1.00027: the lens sits at its
# focal distance in that air and its faces reflect by the index relative to it, so the bounds moved by ~1e-5 of themselves)
@pytest.mark.parametrize("kind, bound, lower", [("focus", 0.0678118219129554, 0.06322741613546895), ("folded", 0.07043914888547562, 0.06225003017381916)])
def test_the_reference_lens_keeps_its_recorded_bound(real_catalog_store, kind, bound, lower):
    """The Designer's ``focus`` and ``folded`` lens, coated ``AR_400-1100R0.01T0.99_AOI0-10`` (0-10 deg air-equivalent, N-BK7 at 780
    nm): the public glass has unknown transmission; an explicit test-only lossless copy retains the census's
    quantitative coating-use and geometry bounds at the unchanged design beam."""
    actual = run(_example(kind))
    assert any(i.code.value == "BULK_LOSS_NOT_ESTABLISHED" for i in actual.issues)
    project = _example(kind, assume_lossless=True)
    parts = tuple(c.model_copy(update={"power_ratings": (_catalog_rating(780.0, 0.1, 2.0 * 1.0012),)}) if c.instance_id == "L1" else c for c in project.components)
    project = project.model_copy(update={"components": parts, "targets": (Target(target_id="u", device_id="L1", metric="power_rating_utilization", maximum=1.0),)})
    result = run(project)
    face = next(s for s in result.exposures.surfaces if s.surface_id == "L1/S2")
    assert face.ratings[0].exposure_upper_bound == pytest.approx(bound, rel=1e-12)
    assert face.ratings[0].exposure_lower_bound == pytest.approx(lower, rel=1e-12)
    assert not [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"]
    cfg = NumericalConfig()
    library = MaterialLibrary(project.materials, cfg)
    scene, _ = lower_scene(project, cfg, library)
    trace = Tracer(scene, library, cfg, include_pupil=True).run()
    probes = follow_untraced(trace, scene, library, cfg)
    hits = [*trace.surface_hits, *(h for _, outcome in probes.results for h in outcome.surface_hits)]
    found = coating_uses(scene, library, cfg, hits, {(i.source_id, i.spectral_index): i.power_W for i in trace.inputs})[780.0]
    for use in found.values():
        assert use.record_id == "AR_400-1100R0.01T0.99_AOI0-10" and use.refused is None and use.air_deg[1] < 10.0


@pytest.mark.parametrize("kind", ("beam_expander", "relay_4f"))
def test_a_lens_filled_by_a_returning_ghost_is_refused_with_its_record(real_catalog_store, kind):
    """On the test-only lossless copy, the 8 faces whose returns nothing bounds: no coating-use
    decision moves, and these faces have no exposure bound. Each design at its design beam: the ghosts' hits have no
    Gaussian state that describes them, and the leg between a
    lens's faces bounds no direction, so no extent is bounded below the record's 10 deg. The chief rays are at normal incidence;
    the extent is not."""
    project = _example(kind, assume_lossless=True)
    optical = [c.instance_id for c in project.components if c.source is None and c.surfaces]
    parts = tuple(c.model_copy(update={"power_ratings": (_catalog_rating(780.0, 1e4),)}) if c.instance_id in optical else c for c in project.components)
    result = run(project.model_copy(update={"components": parts}))
    notices = [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"]
    refused = [i for i in notices if i.numerical_context["cause"] == "COATING_USE_REFUSED"]
    assert {i.numerical_context["coating_record"] for i in refused} == {"AR_400-1100R0.01T0.99_AOI0-10"}
    assert {i.location for i in refused} == {"surface:L1/S1", "surface:L1/S2", "surface:L2/S1", "surface:L2/S2"}
    target_free = {s.surface_id: s.ratings[0].exposure_upper_bound for s in result.exposures.surfaces if s.ratings}
    assert all(bound is None for bound in target_free.values())


@pytest.mark.parametrize("kind, plane", [("beam_expander", "output"), ("relay_4f", "image_plane")])
def test_a_refused_lens_leaves_the_reference_designs_plane_bounded(real_catalog_store, kind, plane):
    """The public glass leaves transmission unknown; with test-only assumed alpha=0, the plane's upper bound
    retains the record's stated value (1 mW source), and a 1 W maximum is MET."""
    # Physical air sets the interface indices. Each omitted share is rounded upward to the ledger resolution.
    recorded = {"output": 0.0009610146023572602, "image_plane": 0.0009612450472993279}[plane]
    real = run(_example(kind))
    actual = next(o for o in real.observations if o.observation_id == plane)
    assert actual.total_power_W is None and actual.power_upper_bound_W is None and actual.quantity_state.value == "UNKNOWN"
    assert any(i.code.value == "BULK_LOSS_NOT_ESTABLISHED" for i in real.issues)
    project = _example(kind, assume_lossless=True)
    target = Target(target_id="p", observation_id=plane, metric="power_W", maximum=1.0, unit="W")
    result = run(project.model_copy(update={"targets": (target,)}))
    observation = next(o for o in result.observations if o.observation_id == plane)
    assert observation.power_upper_bound_W == recorded
    assert observation.quantity_state.value == "BOUNDED"
    assert result.targets[0].assessment.value == "MET"
    assert not [i for i in result.issues if i.code.value == "EXPOSURE_BOUND_WITHHELD"]
