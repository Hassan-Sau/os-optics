"""Catalog polarizing prisms (catalog schema 3.5 ``JOINED_BODIES``) traced from the catalog entry: catalog -> Project -> Engine
trace -> results.

The catalog is a SYNTHETIC test fixture written by the catalog generator (``fixtures/catalog/test_polarizing_prisms.v3_5.json``;
the generator's side checks that it still writes these bytes): Glan-Thompson, Glan-Taylor, Wollaston and
Rochon prisms of a test calcite (G. Ghosh's Sellmeier coefficients, Opt. Commun. 163, 95 (1999)) and a test cement of index 1.55;
not products.

References are computed here, independent of the Engine: the indices from the Sellmeier formula, the vector Snell law at the
contact and the exit face (every wave there travels perpendicular to its crystal's optic axis or along it), the isotropic
s-polarized Fresnel transmittances of the Glan-Thompson (optic axes perpendicular to the plane of incidence), and for the
Glan-Taylor (optic axes in the plane of incidence at the gap) the exact transmittance of the transverse-magnetic wave across a
uniaxial / isotropic interface from the boundary conditions (``tm_transmittance``). Since the
public-format fixture states no bulk absorption, these Fresnel values are upper bounds, not measured output powers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from os_optics.local import lower_project, run_optical_project
from os_optics.local.compile import compile_element
from os_optics.local.declared import declared_copy
from os_optics.local.errors import LocalRunError
from os_optics._vendor.shared_contracts.design_catalog import PublicOpticalCatalogV2
from os_optics.native_library import build_native_model

from local_helpers import ALONG_X, POWER_W, element, plane, project, source
from test_native_crystals import CLASSES_CRYSTAL, place

FIXTURE = Path(__file__).parent / "fixtures" / "catalog" / "test_polarizing_prisms.v3_5.json"
WAVELENGTH = 633.0
CALCITE_O = ([0.73358749, 0.96464345, 1.82831454], [0.0, 0.0194325203, 120.0])
CALCITE_E = ([0.35859695, 0.8242783, 0.14429128], [0.0, 0.0106689543, 120.0])
N_CEMENT = 1.55
AR_T = 0.995  # the fixture's anti-reflection coating: R < 0.5 percent, lossless


def sellmeier(terms, wavelength_nm: float) -> float:
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(b * lam2 / (lam2 - c) for b, c in zip(*terms)))


N_O, N_E = sellmeier(CALCITE_O, WAVELENGTH), sellmeier(CALCITE_E, WAVELENGTH)


def document() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def sku(form: str) -> str:
    return next(p["optical_sku"] for p in document()["products"] if p["form"] == form)


def cut_deg(form: str) -> float:
    return float(next(p for p in document()["products"] if p["form"] == form)["optical_sku"].split("-CUT")[1].split("-")[0].replace("P", "."))


def run(form: str, *, polarization_deg: float = 45.0):
    role = "polarizer" if form.startswith("GLAN") else "splitter"
    doc = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=WAVELENGTH, waist_mm=0.3, polarization_deg=polarization_deg),
                   element("X", role, sku(form), catalog_version="2000-01-01.1")], [plane("OUT", (60.0, 0.0, 0.0), radius_mm=40.0)])
    return run_optical_project(doc, catalog=document())


def bounded_power(arrival) -> float:
    """A catalog with unstated bulk attenuation yields an upper bound, never a known output power."""
    assert arrival.beam.power_W is None
    assert arrival.power_upper_bound_W is not None and 0.0 <= arrival.power_upper_bound_W <= POWER_W
    return arrival.power_upper_bound_W


def outputs(result):
    observation = next(o for o in result.observations if o.observation_id == "OUT")
    assert observation.quantity_state.value == "UNKNOWN" and observation.total_power_W is None
    assert any(issue.code.value == "BULK_LOSS_NOT_ESTABLISHED" for issue in result.issues)
    return tuple(a for a in observation.arrivals if bounded_power(a) > 1e-12 * POWER_W)


def polarized_along(arrival) -> np.ndarray:
    # The unknown bulk attenuation withholds a Jones amplitude, but each split wave's
    # normalized polarization direction remains geometrically determined.
    state = arrival.polarization_state
    assert state is not None and state.handedness == "LINEAR"
    field = np.abs(np.array(state.major_axis))
    return field / np.linalg.norm(field)


def refract(d: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> np.ndarray:
    n = normal if np.dot(d, normal) < 0 else -normal
    cos_i = -float(np.dot(d, n))
    eta = n1 / n2
    return eta * d + (eta * cos_i - math.sqrt(1.0 - eta * eta * (1.0 - cos_i * cos_i))) * n


def exit_direction(n_a: float, n_b: float, theta_deg: float) -> np.ndarray:
    """The axial beam through a split prism (contact normal ``(-sin theta, 0, cos theta)``, exit face normal +Z) with index ``n_a``
    in prism A and ``n_b`` in prism B, as a design-frame direction (local Z = design X, local X = design Y)."""

    theta = math.radians(theta_deg)
    inside = refract(np.array([0.0, 0.0, 1.0]), np.array([-math.sin(theta), 0.0, math.cos(theta)]), n_a, n_b)
    out = refract(inside, np.array([0.0, 0.0, 1.0]), n_b, 1.0)
    out = out / np.linalg.norm(out)
    return np.array([out[2], out[0], out[1]])


def t_normal(n: float) -> float:
    return 4.0 * n / (1.0 + n) ** 2


def t_s(n1: float, n2: float, cos_i: float) -> float:
    sin_t = n1 / n2 * math.sqrt(1.0 - cos_i * cos_i)
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    return 1.0 - ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2


def tm_transmittance(n_o: float, n_e: float, axis_tm: tuple[float, float], k_t: float) -> tuple[float, float]:
    """Exact power transmittances (crystal -> air, air -> crystal) of the transverse-magnetic wave across a plane face of a
    uniaxial crystal whose optic axis lies in the plane of incidence (components ``axis_tm`` along the face's in-plane
    direction and its normal), for tangential wave-vector component ``k_t`` (units of the vacuum wave number), the crystal's
    wave travelling toward the face (+normal). With H along y: ``eta_xx k_z^2 - 2 eta_xz k_x k_z + eta_zz k_x^2 = 1``
    (``eta`` the inverse dielectric tensor), tangential E ``(eta_xx k_z - eta_xz k_x) H``, normal Poynting flux proportional to
    it times ``|H|^2``; continuity of H_y and E_x."""

    d = 1.0 / n_e ** 2 - 1.0 / n_o ** 2
    xx = 1.0 / n_o ** 2 + d * axis_tm[0] ** 2
    zz = 1.0 / n_o ** 2 + d * axis_tm[1] ** 2
    xz = d * axis_tm[0] * axis_tm[1]
    b, c = -2.0 * xz * k_t, zz * k_t * k_t - 1.0
    disc = math.sqrt(b * b - 4.0 * xx * c)
    toward, back = (-b + disc) / (2.0 * xx), (-b - disc) / (2.0 * xx)
    a_in, a_back = xx * toward - xz * k_t, xx * back - xz * k_t
    k_air = math.sqrt(1.0 - k_t * k_t)
    h_r = (k_air - a_in) / (a_back - k_air)
    t_out = k_air * (1.0 + h_r) ** 2 / a_in
    assert t_out + (-a_back * h_r * h_r / a_in) == pytest.approx(1.0, abs=1e-13)  # energy balance of the solution
    h_r2 = (k_air - a_in) / (k_air + a_in)
    t_in = a_in * (1.0 + h_r2) ** 2 / k_air
    assert t_in + h_r2 * h_r2 == pytest.approx(1.0, abs=1e-13)
    return t_out, t_in


def test_the_fixture_catalog_compiles_each_prism_to_one_component_of_its_bodies() -> None:
    catalog = PublicOpticalCatalogV2.model_validate(document())
    wollaston = compile_element("X", catalog.get(sku("WOLLASTON_PRISM")), catalog)  # type: ignore[arg-type]
    assert wollaston.is_compound and [m.member_id for m in wollaston.members] == ["A", "B"]
    assert [s.surface_id for s in wollaston.surfaces] == ["A-S1", "A-S2", "B-S2"]  # (the contact is one interface)
    contact = wollaston.surfaces[1]
    assert (contact.medium_negative, contact.medium_positive) == ("catalog:TEST-CALCITE@X-A", "catalog:TEST-CALCITE@X-B")
    axes = {m.material_id: m.optic_axis for m in wollaston.materials}
    assert axes == {"catalog:TEST-CALCITE@X-A": (1.0, 0.0, 0.0), "catalog:TEST-CALCITE@X-B": (0.0, 1.0, 0.0)}
    thompson = compile_element("X", catalog.get(sku("GLAN_THOMPSON_PRISM")), catalog)  # type: ignore[arg-type]
    assert [s.surface_id for s in thompson.surfaces] == ["A-S1", "A-S2", "A-S3", "B-S1", "B-S2"]
    assert {s.surface_id: s.medium_positive for s in thompson.surfaces if s.surface_id in ("A-S2", "B-S1")} == {
        "A-S2": "catalog:TEST-CEMENT", "B-S1": "catalog:TEST-CEMENT"}
    assert "catalog:TEST-CEMENT" in {m.material_id for m in thompson.materials}


def test_a_catalog_glan_thompson_passes_the_extraordinary_wave_with_the_exact_fresnel_transmittance() -> None:
    """Axes along local Y (design Z), perpendicular to the plane of incidence at the cement layer: exact isotropic s-Fresnel at
    n_e through the uncoated entrance, both layer faces and the exit; the ordinary wave is totally reflected at the layer and
    leaves prism A through its side face."""

    out = run("GLAN_THOMPSON_PRISM")
    result = out.result
    assert out.status == "UNKNOWN" and result.power_budget.conserved and not out.model_scope_issues
    assert any(i.code.value == "CLIPPED" for i in result.issues)  # rejected ray misses the side aperture
    angle = math.radians(cut_deg("GLAN_THOMPSON_PRISM"))
    through = next(a for a in outputs(result) if abs(a.beam.direction[0] - 1.0) < 1e-12)
    cos_c = math.sqrt(1.0 - (N_E * math.sin(angle) / N_CEMENT) ** 2)
    expected = 0.5 * POWER_W * t_normal(N_E) ** 2 * t_s(N_E, N_CEMENT, math.cos(angle)) * t_s(N_CEMENT, N_E, cos_c)
    assert bounded_power(through) == pytest.approx(expected, rel=1e-9)
    assert polarized_along(through)[2] == pytest.approx(1.0, abs=1e-12)
    assert "BIREFRINGENT_INTERFACE_APPROXIMATE" not in {i.code for i in result.issues}
    rejected = [p for p in result.paths if any(step.endswith("tir_o") for step in p.lineage)]
    assert rejected and any("X/A-S3" in step for p in rejected for step in p.lineage)
    critical = math.degrees(math.asin(N_CEMENT / N_O))
    assert critical < cut_deg("GLAN_THOMPSON_PRISM")  # (the ordinary wave's critical angle at the layer)


def test_a_catalog_glan_taylor_passes_the_extraordinary_wave_with_the_exact_tm_transmittance() -> None:
    """Axes along local X (design Y), in the plane of incidence at the air gap: the extraordinary wave (local X, p) crosses the
    gap; the reference is the exact transverse-magnetic transmittance of each gap face (``tm_transmittance``) and the
    anti-reflection coating's T = 0.995 at each outer face."""

    out = run("GLAN_TAYLOR_PRISM")
    result = out.result
    assert out.status == "UNKNOWN" and result.power_budget.conserved and not out.model_scope_issues
    angle = math.radians(cut_deg("GLAN_TAYLOR_PRISM"))
    (through,) = outputs(result)
    np.testing.assert_allclose(through.beam.direction, (1.0, 0.0, 0.0), atol=1e-12)
    assert polarized_along(through)[1] == pytest.approx(1.0, abs=1e-12)
    t_out, t_in = tm_transmittance(N_O, N_E, (math.cos(angle), -math.sin(angle)), N_E * math.sin(angle))
    expected = 0.5 * POWER_W * AR_T ** 2 * t_out * t_in
    assert bounded_power(through) == pytest.approx(expected, rel=1e-9)
    assert any(step.endswith("tir_o") for p in result.paths for step in p.lineage)


@pytest.mark.parametrize("form", ["WOLLASTON_PRISM", "ROCHON_PRISM"])
def test_a_catalog_split_prism_sends_each_polarization_where_snells_law_does(form: str) -> None:
    out = run(form)
    result = out.result
    assert out.status == "UNKNOWN" and result.power_budget.conserved and not out.model_scope_issues
    theta = cut_deg(form)
    beams = outputs(result)
    assert len(beams) == 2
    along_y = next(a for a in beams if polarized_along(a)[2] < 1e-9)
    along_z = next(a for a in beams if polarized_along(a)[2] > 0.999999)
    pairs = {"WOLLASTON_PRISM": ((N_E, N_O), (N_O, N_E)), "ROCHON_PRISM": ((N_O, N_O), (N_O, N_E))}[form]
    np.testing.assert_allclose(along_y.beam.direction, exit_direction(*pairs[0], theta), atol=1e-12)
    np.testing.assert_allclose(along_z.beam.direction, exit_direction(*pairs[1], theta), atol=1e-12)
    if form == "WOLLASTON_PRISM":
        separation = math.degrees(math.acos(float(np.dot(along_y.beam.direction, along_z.beam.direction))))
        assert separation == pytest.approx(2.0 * math.degrees(math.asin((N_O - N_E) * math.tan(math.radians(theta)))), rel=0.01)


def test_the_catalog_wollaston_traces_as_the_native_one_of_the_same_geometry() -> None:
    """Two independent geometry generators (product-data's rule, the native library's ``splitter.wollaston``) of the same prism
    (calcite, 10 mm, 20 deg, 1 mm thin ends, optical contact): equal beam directions and powers."""

    native = run_optical_project(place(build_native_model("splitter.wollaston", CLASSES_CRYSTAL["splitter.wollaston"])), catalog=None).result
    catalog = run("WOLLASTON_PRISM").result
    key = lambda a: a.beam.direction[1]  # noqa: E731
    native_arrivals = next(o for o in native.observations if o.observation_id == "OUT").arrivals
    for mine, theirs in zip(sorted(outputs(catalog), key=key), sorted(native_arrivals, key=key), strict=True):
        np.testing.assert_allclose(mine.beam.direction, theirs.beam.direction, atol=1e-12)
        assert bounded_power(mine) == pytest.approx(theirs.beam.power_W, rel=1e-12)


def test_the_view_draws_a_catalog_prism_as_the_unit_of_its_bodies(tmp_path: Path) -> None:
    out = run_optical_project(project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=WAVELENGTH, waist_mm=0.3),
                                       element("X", "splitter", sku("WOLLASTON_PRISM"), catalog_version="2000-01-01.1")],
                                      [plane("OUT", (60.0, 0.0, 0.0), radius_mm=40.0)]), catalog=document(), output_dir=tmp_path)
    written = json.loads((tmp_path / "optical_scene.json").read_text(encoding="utf-8"))
    assert out.optical_view.elements[1].geometry.kind == "OPTICAL_UNIT"
    assert (tmp_path / "optical_scene.glb").stat().st_size > 0
    (prism,) = [e for e in written["elements"] if e["element_id"] == "X"]
    assert prism["geometry"]["kind"] == "OPTICAL_UNIT" and prism["geometry"]["basis"] == "PUBLIC_CATALOG_DESIGN_GEOMETRY"
    assert prism["geometry"]["form"] == "WOLLASTON_PRISM" and [m["member_id"] for m in prism["geometry"]["members"]] == ["A", "B"]
    assert all("optical_model_id" not in m and m["body"]["solid"]["medium"] == "TEST-CALCITE" for m in prism["geometry"]["members"])


def test_a_joined_prism_is_not_copied_as_one_declared_model() -> None:
    catalog = PublicOpticalCatalogV2.model_validate(document())
    with pytest.raises(LocalRunError, match="COMPOUND unit of one SURFACE_SET model per body"):
        declared_copy(catalog.get(sku("ROCHON_PRISM")), catalog)  # type: ignore[arg-type]


def test_the_lowered_request_carries_each_body_crystal_cut_and_the_cement() -> None:
    doc = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=WAVELENGTH, waist_mm=0.3),
                   element("X", "polarizer", sku("GLAN_THOMPSON_PRISM"), catalog_version="2000-01-01.1", quaternion=ALONG_X)],
                  [plane("OUT", (60.0, 0.0, 0.0), radius_mm=40.0)])
    request = lower_project(doc, catalog=document()).request
    ids = {m.material_id for m in request.project.materials}
    assert {"catalog:TEST-CALCITE@X-A", "catalog:TEST-CALCITE@X-B", "catalog:TEST-CEMENT"} <= ids
