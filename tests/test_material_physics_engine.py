"""The Engine's material physics: one material-evaluation rule on vacuum wavelengths, the environment's air,
VACUUM gaps, spectral bulk loss, analytic group dispersion and temperature.

Every expected value here is computed independently from its source (the source's own law and coefficients, the shared air
formula ``air_index`` of the contracts, the manufacturer's tabulated internal transmittance), never with the Engine's
evaluator:

* the rule: ``l_law = l / n_ref(l)`` (air wavelengths), ``n_abs = law(l_law) n_ref(l)`` (relative to air);
* native Projects operate in the air of their environment (default 20 C, 101325 Pa), ``VACUUM`` gaps have index 1;
* spectral bulk loss ``alpha`` from ``k`` (``4 pi k / l``) or internal transmittance (``-ln T / d``), unknown outside its rows;
* group index / GDD analytic through the chain rule (a measured table states none);
* Schott dn/dT at the material's temperature (the OpticStudio BK7 example).
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from os_optics._vendor.shared_contracts.design_catalog.material_physics import (
    IndexReferenceV1,
    ReferenceAirV1,
    air_index,
)
from os_optics._vendor.shared_contracts.optical_design.optical_model import ProjectModelMaterialV1
from os_optics.config import DEFAULT_NUMERICAL_CONFIG as CFG
from os_optics.contracts.materials import (
    ExtinctionCoefficientLoss,
    HerzbergerMaterial,
    IndexReference,
    InternalTransmittanceLoss,
    PowerSeriesMaterial,
    SchottThermoOptic,
    SellmeierMaterial,
    TabulatedIndexMaterial,
    TemperatureSellmeierMaterial,
)
from os_optics.local import paraxial_analysis, run_optical_project
from os_optics.local.compile import engine_material
from os_optics.optics.materials.group import MEASURED_TABLE_LIMITATION, group_dispersion
from os_optics.optics.materials.index import build_evaluator
from os_optics.optics.materials.physics import internal_transmittance, material_point

from declared_helpers import BAND, NBK7_SELLMEIER, declared_component, sequence, surface, with_models
from local_helpers import ALONG_X, plane, project, quat_multiply, source, turn_about_z

STANDARD = (20.0, 101325.0)
NBK7_B = tuple(float(v) for v in NBK7_SELLMEIER["b"])
NBK7_C = tuple(float(v) for v in NBK7_SELLMEIER["c_um2"])


def dec(value) -> str:
    """A number as the contracts' canonical decimal string (no exponent, no trailing zeros)."""

    text = format(Decimal(value if isinstance(value, (str, Decimal)) else repr(float(value))).normalize(), "f")
    return text if text != "-0" else "0"


def sellmeier(b, c, law_nm: float) -> float:
    lam2 = (law_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(bi * lam2 / (lam2 - ci) for bi, ci in zip(b, c)))


def herzberger(coefficients, law_nm: float) -> float:
    a, b, c, d, e, f = coefficients
    s = (law_nm / 1000.0) ** 2
    big_l = 1.0 / (s - 0.028)
    return a + b * big_l + c * big_l ** 2 + d * s + e * s ** 2 + f * s ** 3


def reference(temperature_c: str = "20", pressure_pa: str = "101325", band=("400", "1100"), stated_by="CATALOGUE_FORMAT") -> IndexReferenceV1:
    return IndexReferenceV1(wavelength="AIR", index="RELATIVE_TO_AIR", reference_air=ReferenceAirV1(temperature_c=temperature_c, pressure_pa=pressure_pa),
                            source_band_nm=band, stated_by=stated_by)


def air_material(material_id: str, dispersion: dict, ref: IndexReferenceV1, **extra) -> ProjectModelMaterialV1:
    """A declared material whose law is on air wavelengths relative to ``ref``'s air, valid on its vacuum band."""

    low, high = ref.vacuum_band()
    return ProjectModelMaterialV1.model_validate({
        "material_id": material_id, "display_name": material_id, "dispersion": dispersion,
        "valid_band_nm": {"min_nm": f"{low:f}", "max_nm": f"{high:f}", "status": "SPECIFIED"},
        "index_reference": ref.model_dump(mode="json"), **extra})


def engine_ref(t: float = 20.0, p: float = 101325.0) -> IndexReference:
    return IndexReference(wavelength="air", index="relative_to_air", reference_air_temperature_c=t, reference_air_pressure_pa=p)


def test_extinction_coefficient_interpolates_k_before_converting_at_the_requested_vacuum_wavelength() -> None:
    """Measured k is a property of the complex index, not an absorption coefficient sampled at the endpoints. If k is
    constant across the two rows, alpha must vary as 1 / vacuum wavelength between them, including when the source
    wavelengths are measured in air."""

    loss = ExtinctionCoefficientLoss(wavelength_nm=(1000.0, 2000.0), k=(1e-6, 1e-6))
    model = SellmeierMaterial(material_id="K_INTERPOLATION", valid_wavelength_nm=(1000.0, 2000.0),
                               b=(1.0,), c_um2=(0.01,), bulk_loss=loss)
    evaluator = build_evaluator(model, CFG)
    assert evaluator.absorption_at(1500.0) == pytest.approx(4 * math.pi * 1e-6 / (1500e-6), rel=1e-13)
    assert math.exp(-evaluator.absorption_at(1500.0) * 100) == pytest.approx(
        math.exp(-4 * math.pi * 1e-6 / (1500e-6) * 100), rel=1e-13)
    air_reference = engine_ref()
    air_source = SellmeierMaterial(material_id="AIR_K_INTERPOLATION", valid_wavelength_nm=(1000.28, 2000.55),
                                   b=(1.0,), c_um2=(0.01,), bulk_loss=loss, index_reference=air_reference)
    at = 1500.0 * air_index(1500.0, *STANDARD)
    assert build_evaluator(air_source, CFG).absorption_at(at) == pytest.approx(4 * math.pi * 1e-6 / (at * 1e-6), rel=1e-12)


# --------------------------------------------------------------------------------------------- air and vacuum media


def _free_space(environment):
    return project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8)], [plane("P", (100.0, 0.0, 0.0))], environment=environment)


def test_replacing_air_by_vacuum_changes_the_optical_path_by_one_minus_n_air_times_the_length() -> None:
    """A straight 200 mm path from the source to a plane, at the fixed vacuum wavelength 632.8 nm and fixed geometry:
    ``OPL_vac - OPL_air = (1 - n_air) L`` with ``n_air = air_index(632.8 nm, 20 C, 101325 Pa)``."""

    in_air = run_optical_project(_free_space(None))
    in_vacuum = run_optical_project(_free_space({"pressure_pa": "0"}))
    (a,) = next(o for o in in_air.result.observations if o.observation_id == "P").arrivals
    (v,) = next(o for o in in_vacuum.result.observations if o.observation_id == "P").arrivals
    n_air = air_index(632.8, *STANDARD)
    assert a.beam.geometric_length_mm == v.beam.geometric_length_mm == pytest.approx(200.0, abs=1e-12)
    assert v.beam.optical_path_length_mm == pytest.approx(200.0, abs=1e-12)
    assert v.beam.optical_path_length_mm - a.beam.optical_path_length_mm == pytest.approx((1.0 - n_air) * 200.0, abs=1e-11)
    materials = {m.material_id: m for m in in_air.request.project.materials}  # (the environment's air, stated in the request)
    assert materials["air"].kind == "air" and (materials["air"].temperature_c, materials["air"].pressure_pa) == STANDARD


def _cell(gap: str) -> dict:
    """Two 2 mm N-BK7 windows around a 10 mm gap of ``gap`` (``AIR`` or ``VACUUM``)."""

    surfaces = [surface(1, "0"), surface(2, "2"), surface(3, "12"), surface(4, "14")]
    return {"model_id": "cell", "model": sequence(surfaces, ["N-BK7", gap, "N-BK7"])}


def _nbk7() -> dict:
    return {"material_id": "N-BK7", "display_name": "N-BK7", "dispersion": dict(NBK7_SELLMEIER), "valid_band_nm": dict(BAND)}


def _cell_project(gap: str, *, quaternion=ALONG_X) -> dict:
    document = project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=632.8, waist_mm=0.5), declared_component("C", "window", "cell", quaternion=quaternion)],
                       [plane("P", (60.0, 0.0, 0.0), radius_mm=20.0)], environment=None)
    return with_models(document, [_cell(gap)], [_nbk7()])


def test_a_vacuum_gap_inside_an_element_is_vacuum_whatever_the_environment() -> None:
    """The same cell with its gap ``AIR`` and ``VACUUM`` in air at 20 C: the windows and the outside are identical, so the
    optical paths differ by exactly ``(1 - n_air) x 10 mm``; the gap is traced in the Engine's ``vacuum`` (index 1)."""

    air_run = run_optical_project(_cell_project("AIR"))
    vacuum_run = run_optical_project(_cell_project("VACUUM"))
    (a,) = next(o for o in air_run.result.observations if o.observation_id == "P").arrivals
    (v,) = next(o for o in vacuum_run.result.observations if o.observation_id == "P").arrivals
    n_air = air_index(632.8, *STANDARD)
    assert v.beam.optical_path_length_mm - a.beam.optical_path_length_mm == pytest.approx((1.0 - n_air) * 10.0, abs=1e-11)
    n_glass = sellmeier(NBK7_B, NBK7_C, 632.8)
    assert a.beam.optical_path_length_mm == pytest.approx(n_air * (110.0 - 4.0 - 10.0) + n_glass * 4.0 + n_air * 10.0, abs=1e-9)
    media = [(s.medium_id, s.refractive_index) for s in vacuum_run.result.paths[0].segments]
    assert ("vacuum", 1.0) in media and ("air", n_air) in media


def test_a_window_between_air_and_vacuum_uses_each_side_s_own_index() -> None:
    """The cell turned 20 deg: the chief ray refracts from the room air (n_air) into the window and from it into the vacuum
    gap (index 1), so ``n_air sin(theta_air) = sin(theta_vacuum)`` about the windows' normal -- not the environment's index on
    both sides -- and it leaves parallel to its incidence."""

    half = math.radians(20.0) / 2.0
    turned = quat_multiply(turn_about_z(math.cos(half), math.sin(half)), ALONG_X)
    run = run_optical_project(_cell_project("VACUUM", quaternion=turned))
    normal = (math.cos(2.0 * half), math.sin(2.0 * half), 0.0)
    n_air = air_index(632.8, *STANDARD)

    def sine(segment) -> float:
        d = [e - s for e, s in zip(segment.end_mm, segment.start_mm)]
        norm = math.sqrt(sum(x * x for x in d))
        cos = sum(x * n for x, n in zip(d, normal)) / norm
        return math.sqrt(max(0.0, 1.0 - cos * cos))

    segments = run.result.paths[0].segments
    first_air = next(s for s in segments if s.medium_id == "air")
    gap = next(s for s in segments if s.medium_id == "vacuum")
    glass = next(s for s in segments if s.medium_id.endswith("N-BK7"))
    assert first_air.refractive_index == n_air and gap.refractive_index == 1.0
    assert n_air * sine(first_air) == pytest.approx(1.0 * sine(gap), abs=1e-12)
    assert n_air * sine(first_air) == pytest.approx(glass.refractive_index * sine(glass), abs=1e-12)
    assert sine(gap) > sine(first_air)  # (vacuum is less dense than air: the ray bends away from the normal)
    (arrival,) = next(o for o in run.result.observations if o.observation_id == "P").arrivals
    assert arrival.beam.direction == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)


# ------------------------------------------------------------------------ equivalent representations, paraxial = trace


def _lens_project(material: dict, environment) -> dict:
    lens = {"model_id": "pcx", "model": sequence([surface(1, "0", radius="52"), surface(2, "4.5")], [material["material_id"]],
                                                 band=dict(material["valid_band_nm"]))}
    document = project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=632.8, waist_mm=0.5), declared_component("L", "lens", "pcx")],
                       [plane("P", (80.0, 0.0, 0.0), radius_mm=20.0)], environment=environment)
    return with_models(document, [lens], [material])


def test_a_law_on_air_wavelengths_and_its_absolute_vacuum_form_give_the_same_focus_and_optical_path() -> None:
    """One source law (the N-BK7 Sellmeier formula) written two ways at fixed conditions (632.8 nm vacuum, air at 20 C):
    (a) on air wavelengths, relative to standard air -- the catalogue's convention; (b) the absolute index on the vacuum
    wavelength, ``law(l / n_air) n_air`` computed here, as a constant. The run's optical path and the paraxial focal length
    agree; the focal length is the catalogue convention's (index ``law(l_air)`` in unit air), as OpticStudio computes it."""

    ref = reference()
    relative = air_material("GLASS", dict(NBK7_SELLMEIER), ref).model_dump(mode="json", exclude_none=True)
    n_air = air_index(632.8, *STANDARD)
    n_rel = sellmeier(NBK7_B, NBK7_C, 632.8 / n_air)
    absolute = {"material_id": "GLASS", "display_name": "GLASS", "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": dec(n_rel * n_air)},
                "valid_band_nm": relative["valid_band_nm"]}
    runs = [run_optical_project(_lens_project(m, None)) for m in (relative, absolute)]
    beams = [next(o for o in r.result.observations if o.observation_id == "P").arrivals[0].beam for r in runs]
    assert beams[0].optical_path_length_mm == pytest.approx(beams[1].optical_path_length_mm, abs=1e-11)
    assert beams[0].position_mm == pytest.approx(beams[1].position_mm, abs=1e-12)
    efl = [paraxial_analysis(_lens_project(m, None), {"components": ["L"], "wavelength_nm": 632.8})["system"]["efl_mm"] for m in (relative, absolute)]
    assert efl[0] == pytest.approx(efl[1], rel=1e-13)
    assert efl[0] == pytest.approx(52.0 / (n_rel - 1.0), rel=1e-13)  # (plano-convex, R = 52 mm: f = R / (n_rel - 1) in that air)


# ------------------------------------------------------------------------------------- source checks (air wavelengths)

LBBH1_B = (2.56597628, 0.526892298, 1.27957402)  # rii specs/ohara/optical/L-BBH1.yml formula 2 (OHARA 2017-11-30)
LBBH1_C = (0.0205164012, 0.0831737967, 109.78774)
OHARA_AIR = ("25", "101325")  # OHARA states its data relative to air at its TD reference temperature, 25 C


def _lbbh1() -> ProjectModelMaterialV1:
    sellmeier_law = {"kind": "SELLMEIER", "b": [dec(b) for b in LBBH1_B], "c_um2": [dec(c) for c in LBBH1_C]}
    return air_material("L-BBH1", sellmeier_law, reference(*OHARA_AIR, band=("300", "2400")))


def test_l_bbh1_at_300_1_nm_evaluates_its_source_law_at_the_air_wavelength() -> None:
    """OHARA L-BBH1 at 300.1 nm (vacuum): the source law at ``l_air = 300.1 / n_air(25 C)`` (about 300.0155 nm), times
    ``n_air``; relative index 3.3556767. The old engine evaluated the law at 300.1 nm itself: 3.3485499 (no longer)."""

    material = _lbbh1()
    n_ref = air_index(300.1, 25.0, 101325.0)
    law_nm = 300.1 / n_ref
    expected_rel = sellmeier(LBBH1_B, LBBH1_C, law_nm)
    n_abs, issue = build_evaluator(engine_material(material, "project:"), CFG).index(300.1)
    assert issue is None
    assert n_abs == pytest.approx(expected_rel * n_ref, rel=1e-14)
    assert n_abs / n_ref == pytest.approx(3.3556767, abs=5e-8)
    assert abs(n_abs / n_ref - 3.3485499) > 7e-3 and sellmeier(LBBH1_B, LBBH1_C, 300.1) == pytest.approx(3.3485499, abs=5e-8)
    point = material_point(material, 300.1)
    assert point.law_wavelength_nm == pytest.approx(law_nm, rel=1e-15) and point.convention == "AIR/RELATIVE_TO_AIR"


def test_a_vacuum_wavelength_inside_the_old_interval_but_whose_air_wavelength_is_outside_the_source_band_is_refused() -> None:
    """The source band is 300-2400 nm on air wavelengths: its vacuum band starts at 300.085 nm. 300.08 nm (vacuum) lies inside
    the old numeric interval [300, 2400] but its air wavelength (299.994 nm) does not: MATERIAL_OUT_OF_RANGE."""

    material = _lbbh1()
    assert Decimal(material.valid_band_nm.min_nm) == Decimal("300.085")
    assert 300.08 / air_index(300.08, 25.0, 101325.0) < 300.0
    evaluator = build_evaluator(engine_material(material, "project:"), CFG)
    n, issue = evaluator.index(300.08)
    assert n is None and issue.code.value == "MATERIAL_OUT_OF_RANGE"
    assert evaluator.index(300.085)[0] is not None
    assert material_point(material, 300.08).reason.startswith("MATERIAL_OUT_OF_RANGE")


IG2_LAW = {"kind": "SELLMEIER", "b": ["2.341388", "2.962098", "0.831035"], "c_um2": ["0", dec(0.360017 ** 2), dec(35.023342 ** 2)]}


def test_vitron_ig2_at_its_lower_band_edge_and_at_2_um() -> None:
    """VITRON IG2 (rii formula 1, air wavelengths relative to standard air, 1-12 um): its vacuum band is 1000.27-12003.215
    nm; 1000.1 nm (vacuum) is outside it, the edge inside; at 2 um the index is ``law(2000 / n_air) n_air``."""

    material = air_material("IG2", IG2_LAW, reference(band=("1000", "12000"), stated_by="SOURCE_TEXT"))
    assert (material.valid_band_nm.min_nm, material.valid_band_nm.max_nm) == ("1000.27", "12003.215")
    evaluator = build_evaluator(engine_material(material, "project:"), CFG)
    assert evaluator.index(1000.1)[1].code.value == "MATERIAL_OUT_OF_RANGE"
    b = [float(v) for v in IG2_LAW["b"]]
    c = [float(v) for v in IG2_LAW["c_um2"]]
    for wavelength in (1000.27, 2000.0):
        n_ref = air_index(wavelength, *STANDARD)
        assert evaluator.index(wavelength)[0] == pytest.approx(sellmeier(b, c, wavelength / n_ref) * n_ref, rel=1e-14)
    assert 1000.27 / air_index(1000.27, *STANDARD) >= 1000.0 > 1000.1 / air_index(1000.1, *STANDARD)


# ------------------------------------------------------------------------------------------------------- bulk loss

HWS3_HERZBERGER = ("2.507599900E+000", "1.116728900E-001", "-9.339157600E-002", "-4.457253300E-005", "-1.103713300E-006", "3.541481900E-009")
HWS9_HERZBERGER = ("3.298702950E+000", "3.030472970E-001", "1.504280870E-001", "-7.781002050E-005", "8.974389770E-008", "-3.702107570E-010")
# CDGM-ZEMAX202609.AGF IT lines (wavelength um on air wavelengths, internal transmittance, thickness 2 mm), as the public
# CDGM optical glass catalogue (2026-09) writes them
HWS3_IT = ((1.3, .981), (1.5, .981), (2.0, .984), (2.5, .985), (3.0, .983), (3.5, .987), (4.0, .988), (4.5, .982), (5.0, .983), (5.5, .983),
           (6.0, .988), (6.5, .987), (7.0, .987), (7.5, .986), (8.0, .966), (8.5, .985), (9.0, .984), (9.5, .987), (10.0, .988), (10.5, .976),
           (11.0, .961), (11.5, .872), (12.0, .768), (12.5, .639), (13.0, .650), (13.5, .712), (14.0, .825))
HWS9_IT = ((2.5, .965), (3.0, .950), (3.5, .973), (4.0, .979), (4.5, .984), (5.0, .985), (5.5, .986), (6.0, .987), (6.5, .986), (7.0, .992),
           (7.5, .996), (8.0, .970), (8.5, .986), (9.0, .998), (9.5, .998), (10.0, .998), (10.5, .994), (11.0, .984), (11.5, .955), (12.0, .709))


def _cdgm(material_id: str, coefficients, rows, band) -> ProjectModelMaterialV1:
    loss = {"kind": "INTERNAL_TRANSMITTANCE", "wavelength_nm": [dec(Decimal(repr(w)) * 1000) for w, _t in rows],
            "internal_transmittance": [dec(t) for _w, t in rows], "thickness_mm": ["2"] * len(rows),
            "basis": "CDGM optical glass catalogue 2026-09, AGF IT lines"}
    return air_material(material_id, {"kind": "HERZBERGER", "coefficients": [dec(c) for c in coefficients]}, reference(band=band), bulk_loss=loss)


HWS3 = _cdgm("HWS3", HWS3_HERZBERGER, HWS3_IT, ("2000", "14000"))
HWS9 = _cdgm("HWS9", HWS9_HERZBERGER, HWS9_IT, ("2000", "12000"))


@pytest.mark.parametrize("material, expected", [(HWS3, 0.988), (HWS9, 0.998)])
def test_cdgm_internal_transmittance_at_10_um_and_2_mm_and_its_thickness_scaling(material, expected) -> None:
    """At 10 um (the catalogue's air wavelength; vacuum ``10000 n_air``) a 2 mm sample transmits the manufacturer's 0.988
    (HWS3) / 0.998 (HWS9) internally; ``T(L) = T(2 mm)^(L / 2 mm)`` (Beer-Lambert); at 10 um vacuum the loss is linear in
    alpha between the 9.5 and 10 um rows."""

    vacuum = 10000.0 * air_index(10000.0 * air_index(10000.0, *STANDARD), *STANDARD)
    t2, reason = internal_transmittance(material, vacuum, 2.0)
    assert reason is None and t2 == pytest.approx(expected, abs=1e-12)
    for thickness in (0.5, 5.0, 10.0):
        assert internal_transmittance(material, vacuum, thickness)[0] == pytest.approx(expected ** (thickness / 2.0), rel=1e-12)
    law = 10000.0 / air_index(10000.0, *STANDARD)
    rows = dict((round(w * 1000.0), t) for w, t in (HWS3_IT if material is HWS3 else HWS9_IT))
    a0, a1 = -math.log(rows[9500]) / 2.0, -math.log(rows[10000]) / 2.0
    alpha = a0 + (a1 - a0) * (law - 9500.0) / 500.0
    assert internal_transmittance(material, 10000.0, 2.0)[0] == pytest.approx(math.exp(-2.0 * alpha), rel=1e-12)


def test_the_extinction_coefficient_route_gives_the_same_absorption() -> None:
    """``k = -ln(T) l / (4 pi d)`` at each row (``l`` its vacuum wavelength) turns the internal transmittance into the
    equivalent extinction coefficient; evaluated through ``alpha = 4 pi k / l`` it gives the same alpha at the rows."""

    ref = engine_ref()
    waves = tuple(w * 1000.0 for w, _t in HWS3_IT)
    it = InternalTransmittanceLoss(wavelength_nm=waves, internal_transmittance=tuple(t for _w, t in HWS3_IT), thickness_mm=(2.0,) * len(waves))
    vacuum = [w * air_index(w * air_index(w, *STANDARD), *STANDARD) for w in waves]
    vacuum = [w * air_index(v, *STANDARD) for w, v in zip(waves, vacuum)]
    k = tuple(-math.log(t) * (v * 1e-6) / (4.0 * math.pi * 2.0) for (_w, t), v in zip(HWS3_IT, vacuum))
    common = {"valid_wavelength_nm": (2000.546, 14003.0), "coefficients": tuple(float(v) for v in HWS3_HERZBERGER), "index_reference": ref}
    by_t = build_evaluator(HerzbergerMaterial(material_id="T", bulk_loss=it, **common), CFG)
    by_k = build_evaluator(HerzbergerMaterial(material_id="K", bulk_loss=ExtinctionCoefficientLoss(wavelength_nm=waves, k=k), **common), CFG)
    for v in vacuum[3:]:
        assert by_k.absorption_at(v) == pytest.approx(by_t.absorption_at(v), rel=1e-9)
    # The two measured quantities interpolate differently between rows: k is linear as a complex-index measurement;
    # internal T at a known thickness is linear in alpha = -ln(T) / d. They coincide exactly at sample wavelengths.
    middle = 5250.0 * air_index(5250.0, *STANDARD)
    law_mid = middle / air_index(middle, *STANDARD)
    for (w0, t0), (w1, t1) in zip(HWS3_IT, HWS3_IT[1:]):
        if w0 * 1000 <= law_mid <= w1 * 1000:
            fraction = (law_mid - w0 * 1000) / ((w1 - w0) * 1000)
            k_mid = k[HWS3_IT.index((w0, t0))] * (1 - fraction) + k[HWS3_IT.index((w1, t1))] * fraction
            alpha_k = 4 * math.pi * k_mid / (middle * 1e-6)
            alpha_t = ((-math.log(t0) / 2) * (1 - fraction) + (-math.log(t1) / 2) * fraction)
            assert by_k.absorption_at(middle) == pytest.approx(alpha_k, rel=1e-12)
            assert by_t.absorption_at(middle) == pytest.approx(alpha_t, rel=1e-12)
            break
    assert by_k.absorption_at(middle) != by_t.absorption_at(middle)


def test_loss_outside_its_rows_is_unknown_not_zero_and_a_trace_says_so() -> None:
    """HWS9's rows start at 2.5 um (air): at 2.2 um the index is defined (band from 2 um) but the bulk loss is not: the
    evaluator gives ``None``, the selection API a reason, and a run through a window of it traces the geometry and warns
    ``BULK_LOSS_NOT_ESTABLISHED`` (its transmitted power is an upper bound). A material without bulk loss has no stated loss."""

    assert build_evaluator(engine_material(HWS9, "project:"), CFG).absorption_at(2200.0) is None
    value, reason = internal_transmittance(HWS9, 2200.0, 2.0)
    assert value is None and reason.startswith("BULK_LOSS_NOT_ESTABLISHED")
    plain = ProjectModelMaterialV1.model_validate(_nbk7())
    assert internal_transmittance(plain, 632.8, 2.0)[1].startswith("NO_STATED_BULK_LOSS")
    window = {"model_id": "w", "model": sequence([surface(1, "0"), surface(2, "2")], ["HWS9"],
                                                 band={"min_nm": "2000.546", "max_nm": "12003", "status": "SPECIFIED"})}
    document = with_models(project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=2200.0, waist_mm=0.5), declared_component("W", "window", "w")],
                                   [plane("P", (30.0, 0.0, 0.0), radius_mm=20.0)], environment=None), [window], [HWS9.model_dump(mode="json", exclude_none=True)])
    run = run_optical_project(document)
    codes = [i.code.value for i in run.result.issues]
    assert "BULK_LOSS_NOT_ESTABLISHED" in codes
    (arrival,) = next(o for o in run.result.observations if o.observation_id == "P").arrivals
    assert arrival.beam.optical_path_length_mm is not None


# ------------------------------------------------------------------------------------------- group index and GDD


def _numeric_derivatives(evaluator, wavelength_nm: float, h: float = 0.5) -> tuple[float, float]:
    """dn/dl and d2n/dl2 (per nm) of the evaluator's own index by central differences (an independent check of the chain)."""

    n = [evaluator.index(wavelength_nm + k * h)[0] for k in (-2, -1, 0, 1, 2)]
    first = (n[0] - 8.0 * n[1] + 8.0 * n[3] - n[4]) / (12.0 * h)
    second = (-n[0] + 16.0 * n[1] - 30.0 * n[2] + 16.0 * n[3] - n[4]) / (12.0 * h * h)
    return first, second


def test_hws3_gdd_at_6_um_has_the_source_law_s_sign_and_the_analytic_chain_matches_numeric_derivatives() -> None:
    """CDGM HWS3 at 6 um (inside its 2-14 um band), now its native Herzberger law: the source law's own curvature at its
    wavelength and the Engine's group-delay dispersion (through the air conversion) have the same sign; the Engine's analytic
    n_g and k2 agree with central differences of its index."""

    model = engine_material(HWS3, "project:")
    got = group_dispersion(model, 6000.0, CFG)
    assert got.method == "analytic" and got.limitation is None
    coefficients = tuple(float(v) for v in HWS3_HERZBERGER)
    law = 6000.0 / air_index(6000.0, *STANDARD)
    h = 1.0
    source_curvature = (herzberger(coefficients, law + h) - 2.0 * herzberger(coefficients, law) + herzberger(coefficients, law - h)) / (h * h)
    assert math.copysign(1.0, got.k2) == math.copysign(1.0, source_curvature)
    first, second = _numeric_derivatives(build_evaluator(model, CFG), 6000.0)
    assert got.n_group == pytest.approx(got.n - 6000.0 * first, rel=1e-9)
    assert got.k2 == pytest.approx(6000.0 ** 3 * second * 1e-6 / (2.0 * math.pi), rel=1e-4)


@pytest.mark.parametrize("model", [
    SellmeierMaterial(material_id="S", valid_wavelength_nm=(400.0, 1100.0), b=NBK7_B, c_um2=NBK7_C, index_reference=engine_ref(),
                      thermo_optic=SchottThermoOptic(d0=1.86e-6, d1=1.31e-8, d2=-1.37e-11, e0=4.34e-7, e1=6.27e-10, lambda_tk_um=0.17,
                                                     reference_temperature_c=20.0), temperature_c=60.0),
    PowerSeriesMaterial(material_id="P", valid_wavelength_nm=(400.0, 1100.0), coefficients=(2.2718929, -1.0108077e-2, 1.0592509e-2, 2.0816965e-4,
                                                                                            -7.6472538e-6, 4.9240991e-7),
                        exponents=(0, 2, -2, -4, -6, -8), index_reference=engine_ref()),
    TemperatureSellmeierMaterial(material_id="T", valid_wavelength_nm=(1200.0, 5000.0), s_coefficients=((10.4907, -1.346e-3, 4.13e-6),),
                                 pole_coefficients_um=((0.29, 1e-4),), temperature_k_range=(20.0, 300.0), reference_temperature_k=295.0),
])
def test_analytic_group_dispersion_of_every_formula_kind_matches_numeric_derivatives(model) -> None:
    wavelength = 1550.0 if model.kind == "temperature_sellmeier" else 700.0
    got = group_dispersion(model, wavelength, CFG)
    first, second = _numeric_derivatives(build_evaluator(model, CFG), wavelength)
    assert got.n_group == pytest.approx(got.n - wavelength * first, rel=1e-10)
    assert got.k2 == pytest.approx(wavelength ** 3 * second * 1e-6 / (2.0 * math.pi), rel=1e-5)


def test_a_measured_table_states_no_group_index_or_gdd() -> None:
    table = TabulatedIndexMaterial(material_id="M", valid_wavelength_nm=(8000.0, 12000.0), wavelength_nm=(8000.0, 9000.0, 10000.0, 11000.0, 12000.0),
                                   refractive_index=(2.6055, 2.6040, 2.6023, 2.6003, 2.5981), origin="measured")
    got = group_dispersion(table, 10000.0, CFG)
    assert got.n == pytest.approx(2.6023) and got.n_group is None and got.k2 is None and got.limitation == MEASURED_TABLE_LIMITATION


# -------------------------------------------------------------------------------------------------------- temperature


def test_schott_dn_dt_reproduces_the_opticstudio_bk7_example() -> None:
    """OpticStudio's example (Ansys article 42661799783443): BK7 (Schott Sellmeier and TD data, T0 = 20 C) at 30 C, 2 atm,
    the system-air wavelength 0.55 um. Our convention: the vacuum wavelength is ``0.55 um x n_air(30 C, 2 atm)``; the Engine
    gives the absolute index at 30 C; divided by ``n_air(30 C, 2 atm)`` it is OpticStudio's relative index 1.51814375. The air
    index is evaluated at the vacuum wavelength (OpticStudio: at the air wavelength), which moves n_air by < 1e-8; every
    intermediate value of the article is compared to its 8 decimals."""

    t, p = 30.0, 2.0 * 101325.0
    vacuum = 550.0
    for _ in range(5):
        vacuum = 550.0 * air_index(vacuum, t, p)
    model = SellmeierMaterial(material_id="BK7", valid_wavelength_nm=(300.0, 2500.0), b=NBK7_B, c_um2=NBK7_C, index_reference=engine_ref(),
                              thermo_optic=SchottThermoOptic(d0=1.86e-6, d1=1.31e-8, d2=-1.37e-11, e0=4.34e-7, e1=6.27e-10, lambda_tk_um=0.17,
                                                             reference_temperature_c=20.0), temperature_c=t)
    n_ref, n_sys = air_index(vacuum, *STANDARD), air_index(vacuum, t, p)
    assert n_ref == pytest.approx(1.00027308, abs=2e-8) and n_sys == pytest.approx(1.00052810, abs=2e-8)
    assert vacuum / n_ref == pytest.approx(550.14022, abs=1e-5)
    n_rel_t0 = sellmeier(NBK7_B, NBK7_C, vacuum / n_ref)
    assert n_rel_t0 == pytest.approx(1.51851533, abs=1e-8)
    assert n_rel_t0 * n_ref == pytest.approx(1.51893001, abs=2e-8)
    n_abs, issue = build_evaluator(model, CFG).index(vacuum)
    assert issue is None
    assert n_abs - n_rel_t0 * n_ref == pytest.approx(0.00001547, abs=1e-8)
    assert n_abs == pytest.approx(1.51894548, abs=2e-8)
    assert n_abs / n_sys == pytest.approx(1.51814375, abs=2e-8)
    at_reference = build_evaluator(model.model_copy(update={"temperature_c": 20.0}), CFG).index(vacuum)[0]
    assert at_reference == n_rel_t0 * n_ref or at_reference == pytest.approx(n_rel_t0 * n_ref, rel=1e-15)


def test_the_environment_temperature_is_the_material_temperature_in_a_native_run() -> None:
    """A Project at 40 C: a declared material with Schott data is lowered at 40 C; one without stays as stated."""

    thermo = {"kind": "SCHOTT_DN_DT", "d0": dec("1.86e-6"), "d1": dec("1.31e-8"), "d2": dec("-1.37e-11"), "e0": dec("4.34e-7"), "e1": dec("6.27e-10"),
              "lambda_tk_um": "0.17", "reference_temperature_c": "20", "basis": "Schott N-BK7 TD data"}
    warm = air_material("WARM", dict(NBK7_SELLMEIER), reference(), thermo_optic=thermo).model_dump(mode="json", exclude_none=True)
    from os_optics.local import lower_project

    lens = {"model_id": "pcx", "model": sequence([surface(1, "0", radius="52"), surface(2, "4.5")], ["WARM"], band=dict(warm["valid_band_nm"]))}
    document = with_models(project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=632.8), declared_component("L", "lens", "pcx")],
                                   [plane("P", (80.0, 0.0, 0.0))], environment={"temperature_c": "40"}), [lens], [warm])
    materials = {m.material_id: m for m in lower_project(document).request.project.materials}
    assert materials["project:WARM"].temperature_c == 40.0
    assert (materials["air"].temperature_c, materials["air"].pressure_pa) == (40.0, 101325.0)


def test_a_temperature_law_is_refused_outside_its_temperature_range() -> None:
    model = TemperatureSellmeierMaterial(material_id="SI", valid_wavelength_nm=(1200.0, 5000.0), s_coefficients=((10.4907, -1.346e-3, 4.13e-6),),
                                         pole_coefficients_um=((0.29, 1e-4),), temperature_k_range=(20.0, 300.0), reference_temperature_k=295.0)
    inside = build_evaluator(model, CFG, 20.0).index(2000.0)
    kelvin = 20.0 + 273.15
    s = 10.4907 - 1.346e-3 * kelvin + 4.13e-6 * kelvin * kelvin
    pole = 0.29 + 1e-4 * kelvin
    lam2 = 4.0
    assert inside[0] == pytest.approx(math.sqrt(1.0 + s * lam2 / (lam2 - pole * pole)), rel=1e-14)
    n, issue = build_evaluator(model, CFG, 40.0).index(2000.0)
    assert n is None and issue.code.value == "MATERIAL_OUT_OF_RANGE"
