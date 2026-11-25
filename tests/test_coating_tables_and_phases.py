"""Project-declared coatings with polarization-resolved spectra, angle-resolved (wavelength x AOI) tables and stated
reflection / transmission phases: what the Engine computes, checked against independent references
(bilinear interpolation written out here, Snell's law, the s / p field decomposition of a reflection and a refraction),
and the equivalence of a declared coating with the same low-level Engine request.

Fixture data: every coating value and phase below is made up for the test; none is supplier data.
"""

from __future__ import annotations

import cmath
import copy
import math

import numpy as np
import pytest

from os_optics import validate
from os_optics.contracts import (
    InterfacePhases,
    JonesVector,
    MirrorInteraction,
    RefractInteraction,
    TabulatedCoating,
    ValidationRequest,
)
from os_optics.local import json_api, lower_project
from os_optics.local.compile import engine_phases
from os_optics.manifest import hash_exact
from os_optics._vendor.shared_contracts.design_catalog import PublicCoatingV1

from declared_helpers import NBK7_SELLMEIER, declared_component, nbk7, sequence, surface, with_models
from local_helpers import ALONG_X, POWER_W, plane, project, quat_multiply, run, source, turn_about_z

WAVELENGTH = 780.0
TILT_DEG = 30.0

# Fixture angle-resolved table: rows 400 / 1100 nm, columns 0 / 20 / 45 deg (in the incident medium).
TABLE_NM = ("400", "1100")
TABLE_AOI = ("0", "20", "45")
T_S = (("0.99", "0.97", "0.9"), ("0.98", "0.95", "0.86"))
T_P = (("0.99", "0.995", "0.999"), ("0.985", "0.99", "0.996"))
R_S = (("0.009", "0.029", "0.099"), ("0.019", "0.049", "0.139"))
R_P = (("0.009", "0.004", "0.0009"), ("0.014", "0.009", "0.0039"))


def angular(aoi=TABLE_AOI, t_s=T_S, t_p=T_P, r_s=R_S, r_p=R_P) -> dict:
    return {"wavelength_nm": list(TABLE_NM), "aoi_deg": list(aoi), "reflectance_s": [list(r) for r in r_s],
            "reflectance_p": [list(r) for r in r_p], "transmittance_s": [list(r) for r in t_s], "transmittance_p": [list(r) for r in t_p],
            "status": "DECLARED"}


def coating(coating_id: str = "OWN", **fields) -> dict:
    data = {"coating_id": coating_id, "kind": "ANTI_REFLECTION", "display_name": "Fixture coating",
            "valid_aoi_deg": {"min_deg": "0", "max_deg": "45", "status": "DECLARED"}, "aoi_reference": "INCIDENT_MEDIUM",
            "basis": "fixture data"}
    data.update(fields)
    return data


def window_project(coat: dict, *, tilt_deg: float = TILT_DEG, polarization_deg: float = 90.0, wavelength_nm: float = WAVELENGTH) -> dict:
    """A 3 mm N-BK7 window with ``coat`` on both faces, turned by ``tilt_deg`` about global Z in a beam along +X; the source
    is s-polarized (global Z) at 90 deg and p-polarized at 0 deg; the plane OUT receives the transmitted beam."""

    half = math.radians(tilt_deg) / 2.0
    pose = quat_multiply(turn_about_z(math.cos(half), math.sin(half)), ALONG_X)
    model = {"model_id": "win", "model": sequence([surface(1, "0", coating_id=coat["coating_id"]),
                                                  surface(2, "3", coating_id=coat["coating_id"])], ["N-BK7"])}
    document = project([source(polarization_deg=polarization_deg, wavelength_nm=wavelength_nm),
                        declared_component("W", "window", "win", quaternion=pose)], [plane("OUT", (100.0, 0.0, 0.0))])
    return with_models(document, [model], [nbk7()], [coat])


def nbk7_index(wavelength_nm: float) -> float:
    b = [float(v) for v in NBK7_SELLMEIER["b"]]
    c = [float(v) for v in NBK7_SELLMEIER["c_um2"]]
    l2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


def bilinear(table, wavelength_nm: float, aoi_deg: float, nodes_nm=TABLE_NM, nodes_aoi=TABLE_AOI) -> float:
    """Independent bilinear interpolation of ``table[i][j]`` (plain Python)."""

    w = [float(x) for x in nodes_nm]
    a = [float(x) for x in nodes_aoi]
    i = max(k for k in range(len(w) - 1) if w[k] <= wavelength_nm)
    j = max(k for k in range(len(a) - 1) if a[k] <= aoi_deg)
    u = (wavelength_nm - w[i]) / (w[i + 1] - w[i])
    v = (aoi_deg - a[j]) / (a[j + 1] - a[j])
    t = [[float(x) for x in row] for row in table]
    return (1 - u) * ((1 - v) * t[i][j] + v * t[i][j + 1]) + u * ((1 - v) * t[i + 1][j] + v * t[i + 1][j + 1])


def arrival(run_, observation_id: str = "OUT"):
    (obs,) = [o for o in run_.result.observations if o.observation_id == observation_id]
    assert len(obs.arrivals) == 1
    return obs.arrivals[0]


def field_of(arr) -> np.ndarray:
    jx, jy = complex(*arr.beam.polarization.ex), complex(*arr.beam.polarization.ey)
    return jx * np.array(arr.beam.basis.e1) + jy * np.array(arr.beam.basis.e2)


# ------------------------------------------------------------------------------------------- angle-resolved tables


@pytest.mark.parametrize("polarization_deg, table", [(90.0, T_S), (0.0, T_P)], ids=["s", "p"])
def test_an_angle_resolved_coating_transmits_its_bilinear_value_at_each_faces_own_angle(polarization_deg, table):
    """Front face: AOI 30 deg in air; back face: the refraction angle inside the glass (the table is read in the incident
    medium). Transmitted power = P T(780 nm, 30 deg) T(780 nm, theta_t), each by independent bilinear interpolation."""

    run_ = run(window_project(coating(angular_spectrum=angular()), polarization_deg=polarization_deg))
    theta_t = math.degrees(math.asin(math.sin(math.radians(TILT_DEG)) / nbk7_index(WAVELENGTH)))
    expected = POWER_W * bilinear(table, WAVELENGTH, TILT_DEG) * bilinear(table, WAVELENGTH, theta_t)
    assert arrival(run_).beam.power_W == pytest.approx(expected, rel=1e-9)
    assert run_.model_scope_issues == ()
    aois = sorted(s.angle_of_incidence_deg for p in run_.result.paths for s in p.segments if s.end_surface_id and s.end_surface_id.startswith("W/"))
    assert aois == pytest.approx(sorted([theta_t, TILT_DEG]), abs=1e-9)


def _padded_table() -> TabulatedCoating:
    """The low-level table a declared angular coating means: its own grid, held flat out to 100 / 20000 nm and 0 / 90 deg."""

    def grid(table):
        rows = [[float(v) for v in row] for row in table]
        rows = [rows[0], *rows, rows[-1]]
        return tuple(tuple([*row, row[-1]]) for row in rows)

    return TabulatedCoating(wavelength_nm=(100.0, 400.0, 1100.0, 20000.0), aoi_deg=(0.0, 20.0, 45.0, 90.0),
                            reflectance_s=grid(R_S), reflectance_p=grid(R_P), transmittance_s=grid(T_S), transmittance_p=grid(T_P))


def test_a_declared_angle_resolved_coating_is_the_same_low_level_engine_request():
    lowered = lower_project(window_project(coating(angular_spectrum=angular())))
    (component,) = [c for c in lowered.request.project.components if c.instance_id == "W"]
    # (the compiled coating also carries its record: validation/coating_use.py)
    table = _padded_table().model_copy(update={"declared_validity": component.surfaces[0].interaction.coating.declared_validity})
    assert all(s.interaction == RefractInteraction(coating=table) for s in component.surfaces)
    # the hand-written low-level request (the same surfaces with the hand-built table) has the declared run's input hash
    surfaces = tuple(s.model_copy(update={"interaction": RefractInteraction(coating=table)}) for s in component.surfaces)
    components = tuple(c.model_copy(update={"surfaces": surfaces}) if c.instance_id == "W" else c for c in lowered.request.project.components)
    request = lowered.request.model_copy(update={"project": lowered.request.project.model_copy(update={"components": components})})
    request = ValidationRequest.model_validate_json(request.model_dump_json())
    assert hash_exact(request.model_dump(mode="json")) == lowered.engine_input_hash


def test_an_angle_outside_the_table_is_out_of_model_scope_and_held_flat():
    """A table starting at 10 deg met at 5 deg (and about 3.3 deg inside the glass): the Engine holds the 10 deg column (no
    extrapolation) and the run is OUT_OF_MODEL_SCOPE (COATING_AOI_OUTSIDE_RANGE) -- never a value invented below the table."""

    coat = coating(angular_spectrum=angular(aoi=("10", "20", "45")), valid_aoi_deg={"min_deg": "10", "max_deg": "45", "status": "DECLARED"})
    run_ = run(window_project(coat, tilt_deg=5.0))
    expected = POWER_W * bilinear(T_S, WAVELENGTH, 10.0, nodes_aoi=("10", "20", "45")) ** 2
    assert arrival(run_).beam.power_W == pytest.approx(expected, rel=1e-9)
    assert {issue.code for issue in run_.model_scope_issues} == {"COATING_AOI_OUTSIDE_RANGE"}
    assert run_.status.value == "OUT_OF_MODEL_SCOPE"


def test_the_engine_bilinear_table_is_identical_to_the_previous_interpolator():
    """The explicit bilinear arithmetic equals SciPy's RegularGridInterpolator bit for bit (the earlier implementation), so
    low-level requests with AOI-dependent tables keep their exact results."""

    from scipy.interpolate import RegularGridInterpolator

    from os_optics.optics.materials.coatings import _cell, bilinear as engine_bilinear

    rng = np.random.default_rng(7)
    for _ in range(50):
        w = np.sort(rng.uniform(300.0, 1500.0, rng.integers(2, 7)))
        a = np.sort(rng.uniform(0.0, 90.0, rng.integers(2, 7)))
        table = rng.uniform(0.0, 1.0, (w.size, a.size))
        wl = float(rng.uniform(w[0], w[-1]))
        aoi = np.concatenate([rng.uniform(a[0], a[-1], 20), a])
        reference = RegularGridInterpolator((w, a), table, method="linear", bounds_error=True)(np.column_stack([np.full_like(aoi, wl), aoi]))
        assert np.array_equal(engine_bilinear(table, _cell(w, np.array([wl])), _cell(a, aoi)), reference)


# --------------------------------------------------------------------------------------- polarization-resolved spectra


SP_SPECTRUM = {"wavelength_nm": ["400", "1100"], "reflectance": ["0.0475", "0.0675"], "transmittance": ["0.9475", "0.9275"],
               "reflectance_s": ["0.08", "0.1"], "reflectance_p": ["0.015", "0.035"], "transmittance_s": ["0.915", "0.895"],
               "transmittance_p": ["0.98", "0.96"], "status": "DECLARED"}


@pytest.mark.parametrize("polarization_deg, key", [(90.0, "transmittance_s"), (0.0, "transmittance_p")], ids=["s", "p"])
def test_a_polarization_resolved_spectrum_transmits_s_and_p_separately(polarization_deg, key):
    run_ = run(window_project(coating(spectrum=SP_SPECTRUM), polarization_deg=polarization_deg))
    low, high = (float(v) for v in SP_SPECTRUM[key])
    value = low + (high - low) * (WAVELENGTH - 400.0) / 700.0
    assert arrival(run_).beam.power_W == pytest.approx(POWER_W * value * value, rel=1e-12)


# ---------------------------------------------------------------------------------------------------------- phases


def _phase_of(arr, s_axis: np.ndarray) -> float:
    """arg(E_p / E_s) of an arrival whose plane of incidence contains global X and Y (s along global Z)."""

    e = field_of(arr)
    d = np.array(arr.beam.direction)
    p_axis = np.cross(d, s_axis)
    return cmath.phase(complex(e @ p_axis) / complex(e @ s_axis))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def test_stated_transmission_phases_replace_the_fresnel_phases_of_a_refracting_face():
    """Both faces of the tilted window state t_s = 0.4 rad, t_p = 1.1 rad: light at 45 deg leaves with arg(E_p / E_s)
    shifted by 2 (1.1 - 0.4) from the input's (the coating's phases, twice), instead of the bare Fresnel phases."""

    phase = {"transmission_s_rad": "0.4", "transmission_p_rad": "1.1", "status": "DECLARED"}
    run_ = run(window_project(coating(angular_spectrum=angular(), phase=phase), polarization_deg=45.0))
    z = np.array([0.0, 0.0, 1.0])
    # the input (global Y + Z) / sqrt 2 along +X: E . s = 1 / sqrt 2, E . p = E . (x cross z) = -1 / sqrt 2, i.e. arg = pi
    assert _wrap(_phase_of(arrival(run_), z) - math.pi - 2.0 * (1.1 - 0.4)) == pytest.approx(0.0, abs=1e-12)
    # without phases the bare Fresnel transmission phases (both 0 for a dielectric below the critical angle) apply
    plain = run(window_project(coating(angular_spectrum=angular()), polarization_deg=45.0))
    assert _wrap(_phase_of(arrival(plain), z) - math.pi) == pytest.approx(0.0, abs=1e-12)


def test_a_wavelength_tabulated_phase_is_linear_between_its_wavelengths():
    phase = {"wavelength_nm": ["400", "1100"], "transmission_s_rad": ["0", "0.7"], "transmission_p_rad": ["1.4", "0"], "status": "DECLARED"}
    run_ = run(window_project(coating(angular_spectrum=angular(), phase=phase), polarization_deg=45.0))
    f = (WAVELENGTH - 400.0) / 700.0
    expected = math.pi + 2.0 * ((1.4 * (1.0 - f)) - (0.7 * f))
    assert _wrap(_phase_of(arrival(run_), np.array([0.0, 0.0, 1.0])) - expected) == pytest.approx(0.0, abs=1e-12)


def _folded_mirror(phase: dict | None, polarization_deg: float = 45.0) -> dict:
    built = json_api.build_native_model("mirror.flat", {"reflectance": "0.9", "valid_band_nm": ["400", "1100"], "diameter_mm": "25.4"})
    assert built["ok"], built
    model, (coat,) = copy.deepcopy(built["optical_model"]), copy.deepcopy(built["model_coatings"])
    if phase is not None:
        coat["phase"] = phase
    inv = 1.0 / math.sqrt(2.0)
    # local +Z (the reflecting side) faces the beam: (-1, 1, 0) / sqrt 2; local +X along global Z; a fold from +X to +Y
    pose = _quaternion(x_axis=(0.0, 0.0, 1.0), z_axis=(-inv, inv, 0.0))
    document = project([source(polarization_deg=polarization_deg), declared_component("M", "mirror", model["model_id"], quaternion=pose)],
                       [plane("OUT", (0.0, 100.0, 0.0), quaternion=_quaternion(x_axis=(0.0, 0.0, 1.0), z_axis=(0.0, 1.0, 0.0)))])
    return with_models(document, [model], None, [coat])


def _quaternion(*, x_axis, z_axis) -> tuple[float, float, float, float]:
    """The rotation (w, x, y, z) taking local X / Z to ``x_axis`` / ``z_axis`` (test helper, plain float algebra)."""

    x, z = np.array(x_axis, float), np.array(z_axis, float)
    y = np.cross(z, x)
    m = np.column_stack([x, y, z])
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w > 1e-6:
        return (w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w))
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    r = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k, j] - m[j, k]) / (2 * r)
    q[1 + i] = r / 2
    q[1 + j] = (m[j, i] + m[i, j]) / (2 * r)
    q[1 + k] = (m[k, i] + m[i, k]) / (2 * r)
    return tuple(q)  # type: ignore[return-value]


def test_stated_reflection_phases_replace_a_mirrors_pi_and_zero():
    """E_out = r_s (E . s) s + r_p (E . p_in) p_out with r = sqrt(R) e^{i phi}: the arriving 3D field equals the
    independent decomposition for the stated phases (s = d x n normalized, p = d x s), and equals the low-level
    MirrorInteraction(phase_s_rad, phase_p_rad) request (to 1e-15: stated phases use the libm-free exp(i phi))."""

    phi_s, phi_p = 0.3, 2.2
    declared = run(_folded_mirror({"reflection_s_rad": str(phi_s), "reflection_p_rad": str(phi_p), "status": "DECLARED"}))
    d_in, d_out = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    inv = 1.0 / math.sqrt(2.0)
    n = np.array([-inv, inv, 0.0])
    s = np.cross(d_in, n)
    s /= np.linalg.norm(s)
    e_in = inv * np.array([0.0, 1.0, 0.0]) + inv * np.array([0.0, 0.0, 1.0])  # source local X = global Y, local Y = global Z
    p_in, p_out = np.cross(d_in, s), np.cross(d_out, s)
    expected = math.sqrt(0.9) * (cmath.exp(1j * phi_s) * (e_in @ s) * s + cmath.exp(1j * phi_p) * (e_in @ p_in) * p_out)
    got = field_of(arrival(declared))
    # compare up to the common propagation phase, which the Jones vector does not carry
    ratio = got @ expected.conj() / (expected @ expected.conj())
    assert abs(ratio) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(got, ratio * expected, atol=1e-12)
    # the same physics as the low-level phase_s_rad / phase_p_rad of the mirror
    lowered = lower_project(_folded_mirror({"reflection_s_rad": str(phi_s), "reflection_p_rad": str(phi_p), "status": "DECLARED"}))
    (mirror,) = [c for c in lowered.request.project.components if c.instance_id == "M"]
    (face,) = mirror.surfaces
    assert face.interaction.phases == InterfacePhases(reflection_s_rad=(phi_s,), reflection_p_rad=(phi_p,))
    low_level = face.interaction.model_copy(update={"phases": None, "phase_s_rad": phi_s, "phase_p_rad": phi_p})
    components = tuple(c.model_copy(update={"surfaces": (face.model_copy(update={"interaction": low_level}),)}) if c.instance_id == "M" else c
                       for c in lowered.request.project.components)
    request = lowered.request.model_copy(update={"project": lowered.request.project.model_copy(update={"components": components})})
    by_phase_fields = validate(ValidationRequest.model_validate_json(request.model_dump_json()))
    by_declared = validate(lowered.request)
    pol = [(*a.beam.polarization.ex, *a.beam.polarization.ey) for o in by_phase_fields.observations for a in o.arrivals]
    declared_pol = [(*a.beam.polarization.ex, *a.beam.polarization.ey) for o in by_declared.observations for a in o.arrivals]
    assert len(pol) == len(declared_pol) == 1
    assert pol[0] == pytest.approx(declared_pol[0], abs=1e-15)  # stated phases use libm-free exp(i phi)


def test_a_coating_without_phases_keeps_the_engine_input_unchanged():
    """No ``phase``: the interactions carry no ``phases`` and their JSON has no such key (every earlier input hash holds)."""

    lowered = lower_project(_folded_mirror(None))
    (mirror,) = [c for c in lowered.request.project.components if c.instance_id == "M"]
    assert mirror.surfaces[0].interaction.phases is None
    assert "phases" not in mirror.surfaces[0].interaction.model_dump(mode="json")
    assert "phases" not in lowered.request.model_dump_json()


def test_engine_phases_hold_a_table_flat_beyond_its_wavelengths():
    coat = PublicCoatingV1.model_validate(coating(spectrum=SP_SPECTRUM, phase={
        "wavelength_nm": ["400", "1100"], "reflection_s_rad": ["0.1", "0.2"], "reflection_p_rad": ["0.3", "0.4"], "status": "DECLARED"}))
    assert engine_phases(coat) == InterfacePhases(wavelength_nm=(100.0, 400.0, 1100.0, 20000.0), reflection_s_rad=(0.1, 0.1, 0.2, 0.2),
                                                  reflection_p_rad=(0.3, 0.3, 0.4, 0.4))
    constant = PublicCoatingV1.model_validate(coating(spectrum=SP_SPECTRUM, phase={"transmission_s_rad": "0.5", "transmission_p_rad": "-0.5",
                                                                                    "status": "DECLARED"}))
    assert engine_phases(constant) == InterfacePhases(transmission_s_rad=(0.5,), transmission_p_rad=(-0.5,))
    assert engine_phases(PublicCoatingV1.model_validate(coating(spectrum=SP_SPECTRUM))) is None


def test_a_low_level_phase_table_that_does_not_cover_the_line_states_no_phase_there():
    """A low-level InterfacePhases table of 400-700 nm met by 780 nm light is never extrapolated: the mirror states no
    phase at that line, so the run is the run of the same mirror without stated phases -- every observation and path
    equal -- plus the COATING_OUT_OF_RANGE issue that names the surface, the path and the table's range."""

    lowered = lower_project(_folded_mirror(None))
    (mirror,) = [c for c in lowered.request.project.components if c.instance_id == "M"]
    assert mirror.surfaces[0].interaction.coating is not None and mirror.surfaces[0].interaction.phases is None
    phases = InterfacePhases(wavelength_nm=(400.0, 700.0), reflection_s_rad=(0.0, 0.1), reflection_p_rad=(0.0, 0.1))
    face = mirror.surfaces[0].model_copy(update={"interaction": mirror.surfaces[0].interaction.model_copy(update={"phases": phases})})
    components = tuple(c.model_copy(update={"surfaces": (face,)}) if c.instance_id == "M" else c for c in lowered.request.project.components)
    request = lowered.request.model_copy(update={"project": lowered.request.project.model_copy(update={"components": components})})
    result = validate(ValidationRequest.model_validate_json(request.model_dump_json()))
    unstated = validate(ValidationRequest.model_validate_json(lowered.request.model_dump_json()))
    (issue,) = [issue for issue in result.issues if issue.code == "COATING_OUT_OF_RANGE"]
    assert issue.status.value == "OUT_OF_MODEL_SCOPE" and issue.location == f"surface:M/{face.surface_id}"
    assert issue.numerical_context["valid_wavelength_nm"] == [400.0, 700.0] and issue.numerical_context["wavelength_nm"] == 780.0
    assert issue.numerical_context["path_id"] == result.paths[0].path_id
    assert all(o.arrivals for o in result.observations)
    assert [o.model_dump(mode="json") for o in result.observations] == [o.model_dump(mode="json") for o in unstated.observations]
    assert [p.model_dump(mode="json") for p in result.paths] == [p.model_dump(mode="json") for p in unstated.paths]
    assert [i.code for i in result.issues if i is not issue] == [i.code for i in unstated.issues]
    # (the pass is recorded as one whose phase is not stated, as for the mirror without stated phases)
    (arrival,) = result.observations[0].arrivals
    assert arrival.group_delay_unknown_at and arrival.group_delay_unknown_at == unstated.observations[0].arrivals[0].group_delay_unknown_at


def test_low_level_phase_contracts_refuse_ambiguity():
    with pytest.raises(ValueError, match="state the reflection phases once"):
        MirrorInteraction(phase_s_rad=0.0, phases=InterfacePhases(reflection_s_rad=(0.1,), reflection_p_rad=(0.2,)))
    with pytest.raises(ValueError, match="needs a coating"):
        RefractInteraction(phases=InterfacePhases(transmission_s_rad=(0.1,), transmission_p_rad=(0.2,)))
    with pytest.raises(ValueError, match="stated together"):
        InterfacePhases(reflection_s_rad=(0.1,))
    with pytest.raises(ValueError, match="one per wavelength_nm node"):
        InterfacePhases(wavelength_nm=(400.0, 500.0), reflection_s_rad=(0.1,), reflection_p_rad=(0.2,))


def test_source_polarization_is_unchanged_for_linear_sources():
    """A linear source keeps its exact Jones vector (the elliptical path is used only for ELLIPTICAL)."""

    lowered = lower_project(window_project(coating(spectrum=SP_SPECTRUM), polarization_deg=90.0))
    (src,) = [c for c in lowered.request.project.components if c.source is not None]
    assert src.source.polarization == JonesVector(ex=(0.0, 0.0), ey=(1.0, 0.0))


# ------------------------------------------------------------------------------------- network export phases


def _phased(project, instance_id: str, phases: InterfacePhases, **fields):
    """``project`` with the (single) surface of component ``instance_id`` given ``phases`` (and interaction ``fields``)."""

    def update(component):
        (face,) = component.surfaces
        interaction = face.interaction.model_copy(update={"phases": phases, **fields})
        interaction = type(interaction).model_validate(interaction.model_dump())
        return component.model_copy(update={"surfaces": (face.model_copy(update={"interaction": interaction}),)})

    return project.model_copy(update={"components": tuple(update(c) if c.instance_id == instance_id else c for c in project.components)})


def _network(project, **options):
    from os_optics.local.network import export_request_network

    return export_request_network(ValidationRequest(project=project), options=options or None)


def _entry(net, row: str, col: str) -> complex:
    tm = net.transfer_matrix
    return complex(*tm.entries[tm.rows.index(row)][tm.columns.index(col)])


def test_the_network_export_carries_stated_splitter_phases():
    """BS1 states r_s = e^{i(0.2 + pi/2)}, r_p = e^{i(0.2 - pi/2)}, t_s = t_p = e^{0.2 i} (|r| = |t| = 1 / sqrt 2, a lossless
    unitary splitter): its element matrices are exactly those amplitudes on (H = p, V = s) of the horizontal fold, the
    network stays lossless, and the notes name the surface whose phases are stated."""

    from test_golden_interference import _mzi_project

    phi = 0.2
    phases = InterfacePhases(reflection_s_rad=(phi + math.pi / 2,), reflection_p_rad=(phi - math.pi / 2,),
                             transmission_s_rad=(phi,), transmission_p_rad=(phi,))
    net = _network(_phased(_mzi_project(0.0), "BS1", phases))
    by = {(e.surface_id, e.event): np.array([[complex(*z) for z in row] for row in e.matrix]) for e in net.elements}
    inv = 1.0 / math.sqrt(2.0)
    assert by[("BS1/coating", "split_reflect")] == pytest.approx(inv * np.diag([cmath.exp(1j * (phi - math.pi / 2)),
                                                                                cmath.exp(1j * (phi + math.pi / 2))]), abs=1e-9)
    assert by[("BS1/coating", "split_transmit")] == pytest.approx(inv * cmath.exp(1j * phi) * np.eye(2), abs=1e-9)
    assert net.transfer_matrix.lossless and net.classical_check.status == "PASS"
    notes = {n.code: n.message for n in net.hints.notes}
    assert "BS1/coating" in notes["REFLECTION_PHASE_CONVENTION"]
    # equal arms: a common phase on both of BS1's outputs moves no light between the ports
    assert abs(_entry(net, "out:port_x:0:V", "in:L0#0:V")) ** 2 == pytest.approx(1.0, abs=2e-9)


def test_a_stated_mirror_phase_shifts_the_interference_of_the_network():
    """A phase delta added to the s reflection of mirror MA (one arm): V light splits cos^2(delta / 2) / sin^2(delta / 2)
    between the two ports (port_x gets everything at delta = 0); H light (p, unchanged phase) still leaves port_x."""

    from test_golden_interference import _mzi_project

    delta = 1.1
    net = _network(_phased(_mzi_project(0.0), "MA", InterfacePhases(reflection_s_rad=(math.pi + delta,), reflection_p_rad=(0.0,))))
    x = abs(_entry(net, "out:port_x:0:V", "in:L0#0:V")) ** 2
    y = abs(_entry(net, "out:port_y:0:V", "in:L0#0:V")) ** 2
    assert (x, y) == pytest.approx((math.cos(delta / 2) ** 2, math.sin(delta / 2) ** 2), abs=2e-9)
    assert abs(_entry(net, "out:port_x:0:H", "in:L0#0:H")) ** 2 == pytest.approx(1.0, abs=2e-9)


def test_stated_phases_that_break_energy_conservation_are_refused():
    """Splitters stating r and t in phase (r_s = t_s, a non-physical lossless splitter) put out more light than they
    receive at an interferometer: NETWORK_NOT_PASSIVE, never a gain matrix."""

    from os_optics.network import NetworkExportError

    from test_golden_interference import _mzi_project

    in_phase = InterfacePhases(reflection_s_rad=(0.0,), reflection_p_rad=(0.0,), transmission_s_rad=(0.0,), transmission_p_rad=(0.0,))
    project = _phased(_phased(_mzi_project(0.0), "BS1", in_phase), "BS2", in_phase)
    with pytest.raises(NetworkExportError) as info:
        _network(project)
    assert {p.code.value for p in info.value.problems} == {"NETWORK_NOT_PASSIVE"}


def test_the_network_conventions_state_how_coating_phases_are_used():
    from os_optics.network.conventions import conventions

    stated = conventions()
    models = {m.interaction: m.model for m in stated.reflection_models}
    assert "states its phases" in models["mirror"] and "states its phases" in models["beam_splitter"]
    assert "stated transmission / reflection phases" in models["refract"]
    assert not any("measured coating phases" in item for item in stated.not_modelled)


def test_libm_free_trigonometry_matches_libm_to_a_few_ulps():
    from os_optics.math.exact_trig import acos_deg, cis

    c = np.linspace(0.0, 1.0, 20001)
    assert np.max(np.abs(acos_deg(c) - np.degrees(np.arccos(c)))) < 1e-12
    assert tuple(acos_deg(np.array([0.0, 0.5, 1.0]))) == pytest.approx((90.0, 60.0, 0.0), abs=1e-13)
    assert max(abs(cis(p) - cmath.exp(1j * p)) for p in np.linspace(-20.0, 20.0, 4001)) < 1e-14
