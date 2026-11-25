"""The Faraday rotator (``FARADAY_ROTATOR``), against Jones calculations derived by hand here, not through Engine functions.

Frame of the hand derivation. Every layout runs along design +X. Local +Z of a component placed with ``ALONG_X`` is design
+X, local +X is design +Y, local +Y is design +Z, so an angle ``phi`` in a transverse plane is measured from design +Y
toward design +Z, right-handed about +X. A rotator with its field axis along +X turns the laboratory field by ``+theta``
in that sense for light travelling along +X and along -X alike:

    Y -> cos(theta) Y + sin(theta) Z,   Z -> -sin(theta) Y + cos(theta) Z.

In a right-handed transverse basis of the ray (``e1 x e2 = d``):

* along +X, in the basis ``(Y, Z)``: ``J -> [[c, -s], [s, c]] J``;
* along -X, in the basis ``(Z, Y)`` (``Z x Y = -X``): writing ``J = (a, b)`` for ``a Z + b Y``, the turn gives
  ``a (-s Y + c Z) + b (c Y + s Z)``, so ``J -> [[c, s], [-s, c]] J``: the rotation of the opposite sign about ``d``.

A reciprocal half-wave plate with its fast axis at ``theta / 2`` sends the forward state to the same output, but mirrors
about its axis in both directions, so the reverse state returns to the input: it isolates nothing and a double pass
cancels. A polarizer of transmittance ``T`` and extinction ratio ``ER`` passes ``T`` of the power along its axis and
``T / ER`` across it.

Isolation metric of the path polarizer / rotator / analyzer. Reference planes: ``IN`` on the laser side of the polarizer and
``OUT`` on the load side of the analyzer. Source ``FWD`` is launched at ``IN`` toward ``OUT``, source ``REV`` independently at
``OUT`` toward ``IN``; both are read per source. Insertion loss ``= 10 log10(P[FWD, IN] / P[FWD, OUT])``; isolation ``= 10
log10(P[REV, OUT] / P[REV, IN])``, both for the stated launch polarizations (``FWD`` along the polarizer axis, ``REV`` along
the analyzer axis). Hand-derived with the rotator at ``theta`` and the analyzer at 45 deg:

    P[FWD, OUT] / P[FWD, IN] = T1 TF (T2 cos^2(theta - 45) + (T2 / ER2) sin^2(theta - 45))
    P[REV, IN] / P[REV, OUT] = T2 TF (T1 cos^2(theta + 45) + (T1 / ER1) sin^2(theta + 45))

Tolerances, chosen before running: powers and field overlaps 1e-12 relative (double-precision products of a few factors).
"""

from __future__ import annotations

import copy
import json
import math

import pytest
from pydantic import ValidationError

from os_optics._vendor.shared_contracts.optical_design import ProjectOpticalModelV1
from os_optics.local import json_api, lower_project, run_optical_project
from os_optics.optical_view import optic_mesh, section
from os_optics.optics.polarization.routing import ROUTE_PASS, LineState, PolarizationRouter
from os_optics.results import load_run

from declared_helpers import compound_model, cylinder, declared_component, with_models
from local_helpers import ALONG_X, AGAINST_X, arrivals, plane, project, quat_multiply, source, turn_about_z

REL = 1e-12
POWER_W = 0.001
SHA = "ab" * 32
EVIDENCE = {"method": "Test fixture values, not a measured device", "tools": [{"name": "test fixture", "version": "1"}],
            "source_data_sha256": SHA, "fit_model": "none: the stated rows", "validation_summary": "test fixture", "date": "2026-10-30"}
AOI = {"min_deg": "0", "max_deg": "5", "status": "SPECIFIED"}
LAMBDA_NM = 1064.0
#: Stated rows of the test rotator (fixture values): at 1047 nm, halfway between the first two, theta 46.5 deg and T 0.94.
ROWS = [("1030", "48", "0.93"), ("1064", "45", "0.95"), ("1090", "42.5", "0.94")]
T1, ER1 = 0.9, 1.0e4
T2, ER2 = 0.85, 2.0e4
TF = 0.95


def rotator_model(model_id: str = "fr", rows=None, *, interpolation: str = "LINEAR", aoi: dict | None = None, **extra) -> dict:
    stated = rows if rows is not None else ROWS
    return {"model_id": model_id, "model": {
        "model_kind": "FARADAY_ROTATOR", "model_version": "1",
        "rows": [{"wavelength_nm": w, "rotation_deg": r, "transmittance": t} for w, r, t in stated], "interpolation": interpolation,
        "clear_radius_mm": "4", "edge_radius_mm": "5", "valid_aoi_deg": aoi or dict(AOI),
        "operating_conditions": "fixture: 25 C, magnet assembly as supplied", "evidence": dict(EVIDENCE), "status": "SPECIFIED", **extra}}


def _plain(value: float) -> str:
    """A number as a contract decimal: no exponent, no trailing zeros."""

    text = f"{value:.12f}".rstrip("0").rstrip(".")
    return text


def _band() -> dict:
    return {"min_nm": "400", "max_nm": "1600", "status": "SPECIFIED"}


def polarizer_model(model_id: str, t: float, er: float) -> dict:
    return {"model_id": model_id, "model": {
        "model_kind": "POLARIZATION_ELEMENT", "model_version": "1", "fidelity": "IDEAL_THIN_ELEMENT", "element": "LINEAR_POLARIZER", "plane_z_mm": "0",
        "transmittance": {"value": _plain(t), "status": "SPECIFIED"}, "extinction_ratio": {"value": _plain(er), "status": "SPECIFIED"},
        "clear_radius_mm": "5", "edge_radius_mm": "6", "valid_band_nm": _band(), "valid_aoi_deg": dict(AOI), "approximations": ["THIN_ELEMENT"],
        "status": "SPECIFIED"}}


def retarder_model(model_id: str, waves: str, t: float = 1.0) -> dict:
    return {"model_id": model_id, "model": {
        "model_kind": "POLARIZATION_ELEMENT", "model_version": "1", "fidelity": "IDEAL_THIN_ELEMENT", "element": "RETARDER", "plane_z_mm": "0",
        "retardance_waves": waves, "design_wavelength_nm": _plain(LAMBDA_NM), "retardance_dispersion": "CONSTANT",
        "transmittance": {"value": _plain(t), "status": "SPECIFIED"}, "clear_radius_mm": "5", "edge_radius_mm": "6", "valid_band_nm": _band(),
        "valid_aoi_deg": dict(AOI), "approximations": ["CONSTANT_RETARDANCE", "THIN_ELEMENT"], "status": "SPECIFIED"}}


def mirror() -> tuple[dict, list[dict]]:
    """A lossless flat mirror, its model and coating: ``E_out = E_in`` for the transverse field at normal incidence, up to sign."""

    built = json_api.build_native_model("mirror.flat", {"diameter_mm": "25.4", "reflectance": "1", "valid_band_nm": ["400", "1600"], "model_id": "mirror"})
    assert built["ok"] is True, built
    return built["optical_model"], built["model_coatings"]


def along(deg: float) -> tuple[float, float, float, float]:
    """``ALONG_X`` turned about the beam by ``deg``, right-handed about +X."""

    half = math.radians(deg) / 2.0
    return quat_multiply(ALONG_X, turn_about_z(math.cos(half), math.sin(half)))


def forward_source(instance_id: str = "FWD", x: float = -100.0, *, phi_deg: float = 0.0, wavelength_nm: float = LAMBDA_NM) -> dict:
    return source(instance_id, (x, 0.0, 0.0), wavelength_nm=wavelength_nm, polarization_deg=phi_deg, power_w=POWER_W)


def reverse_source(instance_id: str = "REV", x: float = 100.0, *, phi_deg: float = 0.0, wavelength_nm: float = LAMBDA_NM) -> dict:
    # AGAINST_X: local +X is design -Y, local +Y is design +Z, so a laboratory angle phi is the local angle -phi (up to the sign of the field)
    return source(instance_id, (x, 0.0, 0.0), quaternion=AGAINST_X, wavelength_nm=wavelength_nm, polarization_deg=-phi_deg, power_w=POWER_W)


def _run(document: dict, **kwargs):
    return run_optical_project(document, catalog=None, **kwargs)


def _beam(arrival) -> tuple[list[complex], float]:
    """The arriving field vector in the design frame (unit length) and the power."""

    beam = arrival.beam
    ex, ey = complex(*beam.polarization.ex), complex(*beam.polarization.ey)
    field = [ex * a + ey * b for a, b in zip(beam.basis.e1, beam.basis.e2)]
    norm = math.sqrt(sum(abs(c) ** 2 for c in field))
    return [c / norm for c in field], beam.power_W


def _overlap(field: list[complex], expected: list[complex]) -> float:
    return abs(sum(a.conjugate() * b for a, b in zip(field, expected)))


def _lab(phi_deg: float) -> list[complex]:
    phi = math.radians(phi_deg)
    return [0j, complex(math.cos(phi)), complex(math.sin(phi))]


def _through(result, observation: str, source_id: str):
    (arrival,) = arrivals(result, observation, source_id)
    return arrival


# ---------------------------------------------------------------------------------------------- the model


def test_a_customer_declares_it_without_a_product():
    document = with_models(project([forward_source(), declared_component("FR", "other_optical", "fr")], [plane("OUT", (50.0, 0.0, 0.0))]),
                           [rotator_model()], design_intent="OPTICAL_SYSTEM")
    checked = json_api.check_project(document)
    assert checked["ok"] is True, checked
    (declared,) = checked["declared_models"]
    assert declared["model_kind"] == "FARADAY_ROTATOR" and declared["optical_model_id"] == "fr"
    assert checked["simulated_parts"] == []


@pytest.mark.parametrize("change, message", [
    ({"plane_z_mm": "1"}, "plane_z_mm is 0"),
    ({"clear_radius_mm": "6"}, "clear_radius_mm cannot exceed"),
    ({"valid_aoi_deg": {"min_deg": "0", "max_deg": "90", "status": "SPECIFIED"}}, "below 90"),
    ({"rows": [{"wavelength_nm": "1064", "rotation_deg": "45", "transmittance": "1.2"}]}, "transmittance must lie"),
    ({"rows": [{"wavelength_nm": "1090", "rotation_deg": "45", "transmittance": "0.9"}, {"wavelength_nm": "1064", "rotation_deg": "45", "transmittance": "0.9"}]}, "strictly increasing"),
])
def test_an_inconsistent_response_is_refused(change, message):
    with pytest.raises(ValidationError, match=message):
        ProjectOpticalModelV1.model_validate(rotator_model(**change) if "rows" not in change else rotator_model(rows=[(r["wavelength_nm"], r["rotation_deg"], r["transmittance"]) for r in change["rows"]]))


@pytest.mark.parametrize("field", ["rows", "interpolation"])
def test_the_response_is_never_filled_in(field):
    model = rotator_model()
    del model["model"][field]
    with pytest.raises(ValidationError, match=field):
        ProjectOpticalModelV1.model_validate(model)


def test_a_response_without_stated_angles_conditions_or_evidence_is_an_assumed_table_and_nothing_is_filled_in():
    """The rows alone are a complete model. Tilted 8 deg -- outside the 0-5 deg of the stated twin -- the two give the same
    field and power to the last bit; the stated one flags the angle, the unstated one has no angle to check; the summary
    says which is which."""

    def tilted(model: dict):
        half = math.radians(8.0) / 2.0
        pose = quat_multiply(ALONG_X, (math.cos(half), math.sin(half), 0.0, 0.0))
        return _run(with_models(project([forward_source(), declared_component("FR", "other_optical", "fr", quaternion=pose)], [plane("OUT", (50.0, 0.0, 0.0))]),
                                [model], design_intent="OPTICAL_SYSTEM"))

    stated = rotator_model()
    bare = rotator_model()
    for field in ("valid_aoi_deg", "operating_conditions", "evidence"):
        del bare["model"][field]
    parsed = ProjectOpticalModelV1.model_validate(bare).model
    assert (parsed.valid_aoi_deg, parsed.operating_conditions, parsed.evidence, parsed.data_kind) == (None, None, None, "ASSUMED")
    with_range, without = tilted(stated), tilted(bare)
    assert [i.code for i in with_range.model_scope_issues] == ["ELEMENT_AOI_OUTSIDE_RANGE"] and with_range.status.value == "OUT_OF_MODEL_SCOPE"
    assert without.model_scope_issues == () and without.status.value == "PASS"
    a, b = _through(with_range.result, "OUT", "FWD"), _through(without.result, "OUT", "FWD")
    assert b.beam.power_W == a.beam.power_W and b.beam.polarization == a.beam.polarization and b.beam.basis == a.beam.basis
    assert b.beam.power_W == pytest.approx(POWER_W * TF, rel=REL)  # the stated transmittance at 1064 nm
    field, _ = _beam(b)
    assert _overlap(field, _lab(45.0)) == pytest.approx(1.0, rel=1e-9)  # the stated rotation at 1064 nm
    (row,) = [r for r in without.summary()["element_models"] if r["instance_id"] == "FR"]
    response = row["response"]
    assert response["data_kind"] == "ASSUMED" and response["evidence"] is None and response["operating_conditions"] is None
    assert response["domain"] == {"interpolation": "LINEAR", "valid_aoi_deg": None}
    (row,) = [r for r in with_range.summary()["element_models"] if r["instance_id"] == "FR"]
    assert row["response"]["data_kind"] == "EVIDENCED" and row["response"]["evidence"] == EVIDENCE
    assert row["response"]["domain"]["valid_aoi_deg"] == ["0", "5"]


def test_a_frame_is_not_stated():
    with pytest.raises(ValidationError, match="fixed frame"):
        ProjectOpticalModelV1.model_validate({**rotator_model(), "frame": {"origin": "FIRST_SURFACE_VERTEX", "symmetry": "ROTATIONAL_ABOUT_LOCAL_Z"}})


# --------------------------------------------------------------------------------- forward and reverse Jones


@pytest.mark.parametrize("theta_deg", [45.0, 30.0, -45.0, 90.0, 12.5])
@pytest.mark.parametrize("phi_deg", [0.0, 20.0, 90.0])
def test_both_directions_match_the_hand_derived_jones_matrices(theta_deg, phi_deg):
    rows = [("1064", _plain(theta_deg), _plain(TF))]
    document = with_models(
        project([forward_source(phi_deg=phi_deg), reverse_source(phi_deg=phi_deg), declared_component("FR", "other_optical", "fr")],
                [plane("OUT", (50.0, 0.0, 0.0)), plane("IN", (-50.0, 0.0, 0.0))]),
        [rotator_model(rows=rows, interpolation="POINTS_ONLY")], design_intent="OPTICAL_SYSTEM")
    result = _run(document).result
    # in the laboratory both directions turn the same way: phi + theta (module docstring)
    for observation, source_id in (("OUT", "FWD"), ("IN", "REV")):
        field, power = _beam(_through(result, observation, source_id))
        assert power == pytest.approx(POWER_W * TF, rel=REL)
        assert _overlap(field, _lab(phi_deg + theta_deg)) == pytest.approx(1.0, rel=REL)
    # the same statement in each ray's own basis: the matrices of the module docstring
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    p, q = math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))
    forward_yz = (c * p - s * q, s * p + c * q)  # J = (p, q) in (Y, Z)
    field, _ = _beam(_through(result, "OUT", "FWD"))
    assert abs(abs(field[1]) - abs(forward_yz[0])) < REL and abs(abs(field[2]) - abs(forward_yz[1])) < REL
    reverse_zy = (c * q + s * p, -s * q + c * p)  # J = (q, p) in (Z, Y)
    field, _ = _beam(_through(result, "IN", "REV"))
    assert abs(abs(field[2]) - abs(reverse_zy[0])) < REL and abs(abs(field[1]) - abs(reverse_zy[1])) < REL


def test_a_circular_state_keeps_its_handedness_and_power():
    """R(b) (1, i) = exp(-i b) (1, i): an eigenstate of the rotation, whichever way it travels."""

    circular = forward_source()
    circular["source_model"]["polarization"] = {"kind": "ELLIPTICAL", "orientation_deg": 0.0, "ellipticity_deg": 45.0}
    document = with_models(project([circular, declared_component("FR", "other_optical", "fr")], [plane("OUT", (50.0, 0.0, 0.0))]),
                           [rotator_model(rows=[("1064", "45", _plain(TF))], interpolation="POINTS_ONLY")], design_intent="OPTICAL_SYSTEM")
    arrival = _through(_run(document).result, "OUT", "FWD")
    assert arrival.beam.power_W == pytest.approx(POWER_W * TF, rel=REL)
    assert arrival.polarization_state.ellipticity_deg == pytest.approx(45.0, abs=1e-9)


def test_turning_the_element_around_reverses_its_sense_in_the_laboratory():
    """A pose that sends the field axis to -X: the field turns by -theta about +X for light along +X."""

    document = with_models(project([forward_source(), declared_component("FR", "other_optical", "fr", quaternion=AGAINST_X)], [plane("OUT", (50.0, 0.0, 0.0))]),
                           [rotator_model(rows=[("1064", "30", "1")], interpolation="POINTS_ONLY")], design_intent="OPTICAL_SYSTEM")
    field, _ = _beam(_through(_run(document).result, "OUT", "FWD"))
    assert _overlap(field, _lab(-30.0)) == pytest.approx(1.0, rel=REL)


def test_the_response_is_symmetric_about_its_axis():
    """Turning the element about its own axis changes nothing: only the axis is a direction of the element."""

    for turn in (0.0, 33.0):
        document = with_models(project([forward_source(phi_deg=10.0), declared_component("FR", "other_optical", "fr", quaternion=along(turn))],
                                       [plane("OUT", (50.0, 0.0, 0.0))]),
                               [rotator_model(rows=[("1064", "30", "1")], interpolation="POINTS_ONLY")], design_intent="OPTICAL_SYSTEM")
        field, _ = _beam(_through(_run(document).result, "OUT", "FWD"))
        assert _overlap(field, _lab(40.0)) == pytest.approx(1.0, rel=REL)


def test_a_compound_member_keeps_the_axis_it_is_placed_with():
    """A member turned half a turn about local X has its field axis along the unit's -Z: the opposite sense."""

    unit = compound_model("isolator", [("FR", "fr", (0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))], cylinder("12", "-1", "1"))
    document = with_models(project([forward_source(), declared_component("U", "other_optical", "isolator")], [plane("OUT", (50.0, 0.0, 0.0))]),
                           [rotator_model(rows=[("1064", "30", "1")], interpolation="POINTS_ONLY"), unit], design_intent="OPTICAL_SYSTEM")
    field, _ = _beam(_through(_run(document).result, "OUT", "FWD"))
    assert _overlap(field, _lab(-30.0)) == pytest.approx(1.0, rel=REL)


# ------------------------------------------------------------------------------ round trip against a reciprocal element


def _double_pass(element_model: dict, quaternion) -> list[complex]:
    """Light along +X through the element, a normal-incidence mirror behind it, back through the element, read on a plane
    in front of it; returns the field there."""

    mirror_doc, coatings = mirror()
    document = with_models(
        project([forward_source(), declared_component("E", "other_optical", element_model["model_id"], quaternion=quaternion),
                 declared_component("M", "mirror", "mirror", (40.0, 0.0, 0.0), quaternion=AGAINST_X)],
                [plane("BACK", (-50.0, 0.0, 0.0))]),
        [element_model, mirror_doc], coatings=coatings, design_intent="OPTICAL_SYSTEM")
    result = _run(document).result
    back = [a for a in arrivals(result, "BACK", "FWD") if a.beam.direction[0] < 0.0]
    assert len(back) == 1
    return _beam(back[0])[0]


def test_a_round_trip_adds_the_rotation_and_a_reciprocal_plate_cancels():
    """Both elements send +Y to 45 deg on the way out (the isolator test below). Coming back, the rotator goes on to 90 deg
    (a Faraday mirror: the state at the input is orthogonal to the launch), the half-wave plate returns to 0 deg."""

    rotated = _double_pass(rotator_model(rows=[("1064", "45", "1")], interpolation="POINTS_ONLY"), ALONG_X)
    assert _overlap(rotated, _lab(90.0)) == pytest.approx(1.0, rel=REL)
    assert _overlap(rotated, _lab(0.0)) < 1e-12
    plate = _double_pass({**retarder_model("hwp", "0.5"), "model_id": "hwp"}, along(22.5))
    assert _overlap(plate, _lab(0.0)) == pytest.approx(1.0, rel=REL)
    assert _overlap(plate, _lab(90.0)) < 1e-12


# ------------------------------------------------------------------------------------------------ the isolation path


def isolator(middle: dict, middle_pose, *, wavelength_nm: float = LAMBDA_NM, fwd_phi: float = 0.0, rev_phi: float = 45.0) -> dict:
    return with_models(project(
        [forward_source(wavelength_nm=wavelength_nm, phi_deg=fwd_phi), reverse_source(wavelength_nm=wavelength_nm, phi_deg=rev_phi),
         declared_component("P1", "polarizer", "p1", (-20.0, 0.0, 0.0), along(0.0)),
         declared_component("MID", "other_optical", middle["model_id"], (0.0, 0.0, 0.0), middle_pose),
         declared_component("P2", "polarizer", "p2", (20.0, 0.0, 0.0), along(45.0))],
        [plane("IN", (-50.0, 0.0, 0.0)), plane("OUT", (50.0, 0.0, 0.0))]),
        [polarizer_model("p1", T1, ER1), polarizer_model("p2", T2, ER2), middle], design_intent="OPTICAL_SYSTEM")


def _ports(result) -> dict[str, float]:
    return {"fwd_in": _through(result, "IN", "FWD").beam.power_W, "fwd_out": _through(result, "OUT", "FWD").beam.power_W,
            "rev_out": _through(result, "OUT", "REV").beam.power_W, "rev_in": _through(result, "IN", "REV").beam.power_W}


def _isolation_db(ports: dict[str, float]) -> float:
    return 10.0 * math.log10(ports["rev_out"] / ports["rev_in"])


def _insertion_db(ports: dict[str, float]) -> float:
    return 10.0 * math.log10(ports["fwd_in"] / ports["fwd_out"])


def _hand_ports(theta_deg: float, tf: float) -> tuple[float, float]:
    """(forward transmission IN -> OUT, reverse transmission OUT -> IN) of the module docstring."""

    forward = T1 * tf * (T2 * math.cos(math.radians(theta_deg - 45.0)) ** 2 + T2 / ER2 * math.sin(math.radians(theta_deg - 45.0)) ** 2)
    reverse = T2 * tf * (T1 * math.cos(math.radians(theta_deg + 45.0)) ** 2 + T1 / ER1 * math.sin(math.radians(theta_deg + 45.0)) ** 2)
    return forward, reverse


# (wavelength, theta and T there: the rows, and 1047 nm halfway between the first two)
ISOLATION_CONDITIONS = [(1030.0, 48.0, 0.93), (1047.0, 46.5, 0.94), (1064.0, 45.0, 0.95), (1090.0, 42.5, 0.94)]


@pytest.mark.parametrize("wavelength_nm, theta_deg, tf", ISOLATION_CONDITIONS)
def test_the_isolator_ports_match_the_hand_derivation(wavelength_nm, theta_deg, tf):
    result = _run(isolator(rotator_model(), ALONG_X, wavelength_nm=wavelength_nm)).result
    ports = _ports(result)
    forward, reverse = _hand_ports(theta_deg, tf)
    assert ports["fwd_in"] == pytest.approx(POWER_W, rel=REL) and ports["rev_out"] == pytest.approx(POWER_W, rel=REL)
    assert ports["fwd_out"] / ports["fwd_in"] == pytest.approx(forward, rel=REL)
    assert ports["rev_in"] / ports["rev_out"] == pytest.approx(reverse, rel=REL)
    assert _isolation_db(ports) == pytest.approx(-10.0 * math.log10(reverse), rel=REL)


def test_isolation_at_the_design_point_and_its_limits():
    """At theta = 45 deg the leak is the polarizer's own: isolation = -10 log10(T2 TF T1 / ER1) (41.4 dB here: 40 dB of the
    extinction ratio and the path losses); 3 deg of rotation error (1030 nm, 27 dB) brings it to the rotator's sin^2 limit."""

    ports = _ports(_run(isolator(rotator_model(), ALONG_X)).result)
    assert _isolation_db(ports) == pytest.approx(-10.0 * math.log10(T2 * TF * T1 / ER1), rel=REL)
    assert _insertion_db(ports) == pytest.approx(-10.0 * math.log10(T1 * TF * T2), rel=REL)
    off = _ports(_run(isolator(rotator_model(), ALONG_X, wavelength_nm=1030.0)).result)
    assert _isolation_db(off) == pytest.approx(-10.0 * math.log10(_hand_ports(48.0, 0.93)[1]), rel=REL)
    assert _isolation_db(off) < _isolation_db(ports) - 10.0


def test_a_design_turned_as_a_whole_gives_the_same_ports():
    """A rigid turn of the whole design about an oblique axis: every direction the rotator depends on turns with it."""

    half = 0.6
    axis = (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
    turn = (math.cos(half), *(math.sin(half) * a for a in axis))
    document = isolator(rotator_model(), ALONG_X)
    reference = _ports(_run(document).result)

    def rotate(v):
        w, x, y, z = turn
        q = quat_multiply(quat_multiply(turn, (0.0, *v)), (w, -x, -y, -z))
        return list(q[1:])

    for item in document["components"]:
        pose = item["nominal_pose"]
        pose["position_mm"], pose["quaternion_wxyz"] = rotate(pose["position_mm"]), list(quat_multiply(turn, pose["quaternion_wxyz"]))
    for item in document["observations"]:
        pose = item["pose"]
        pose["position_mm"], pose["quaternion_wxyz"] = rotate(pose["position_mm"]), list(quat_multiply(turn, pose["quaternion_wxyz"]))
    turned = _ports(_run(document).result)
    assert turned == pytest.approx(reference, rel=1e-9)


def test_a_reciprocal_half_wave_plate_in_the_same_path_isolates_nothing():
    """The plate passes the forward beam like the rotator (+Y to 45 deg, here T = 0.95) and the reverse beam straight back:
    the reverse transmission equals the forward one, isolation = insertion loss."""

    ports = _ports(_run(isolator(retarder_model("hwp", "0.5", TF), along(22.5))).result)
    assert ports["fwd_out"] / ports["fwd_in"] == pytest.approx(T1 * TF * T2, rel=REL)
    assert ports["rev_in"] / ports["rev_out"] == pytest.approx(T1 * TF * T2, rel=REL)
    assert _isolation_db(ports) == pytest.approx(_insertion_db(ports), rel=REL)


def test_a_reciprocal_return_trap_is_unchanged():
    """Polarizer, quarter-wave plate at 45 deg, mirror: the return is orthogonal to the polarizer axis by a retarder's double
    pass, a trap for that one reflection and not an isolator. The Jones of a retarder has no direction in it."""

    mirror_doc, coatings = mirror()
    document = with_models(project(
        [forward_source(), declared_component("P", "polarizer", "p1", (-20.0, 0.0, 0.0), along(0.0)),
         declared_component("Q", "waveplate", "qwp", (0.0, 0.0, 0.0), along(45.0)),
         declared_component("M", "mirror", "mirror", (40.0, 0.0, 0.0), quaternion=AGAINST_X)],
        [plane("BACK", (-50.0, 0.0, 0.0))]),
        [polarizer_model("p1", T1, ER1), retarder_model("qwp", "0.25"), mirror_doc], coatings=coatings, design_intent="OPTICAL_SYSTEM")
    back = [a for a in arrivals(_run(document).result, "BACK", "FWD") if a.beam.direction[0] < 0.0]
    assert len(back) == 1
    assert back[0].beam.power_W == pytest.approx(POWER_W * T1 * T1 / ER1, rel=REL)


def test_the_power_budget_reads_each_launch_on_its_own():
    """The two sources are independent launches: each line's budget carries its own power at each plane, the rotator's loss is
    an element transmission loss, and every line's ledger closes."""

    budget = _run(isolator(rotator_model(), ALONG_X)).result.power_budget
    lines = {line.source_id: line for line in budget.lines}
    forward, reverse = _hand_ports(45.0, TF)
    assert lines["FWD"].observed_W["OUT"] == pytest.approx(POWER_W * forward, rel=REL)
    assert lines["REV"].observed_W["IN"] == pytest.approx(POWER_W * reverse, rel=REL)
    assert lines["FWD"].conserved and lines["REV"].conserved
    # the forward line loses (1 - TF) of what P1 passes at the rotator
    element = next(e for e in budget.by_element if e.element_id == "MID")
    assert element.sinks["ELEMENT_TRANSMISSION_LOSS"] == pytest.approx(POWER_W * (T1 * (1.0 - TF) + T2 * (1.0 - TF)), rel=1e-9)


# ------------------------------------------------------------------------------------------------- domain and applicability


def _single(rotator: dict, wavelength_nm: float, **source_fields) -> dict:
    src = forward_source(wavelength_nm=wavelength_nm)
    src["source_model"].update(source_fields)
    return with_models(project([src, declared_component("FR", "other_optical", "fr")], [plane("OUT", (50.0, 0.0, 0.0))],
                               [{"target_id": "T", "metric": "power_W", "observation_id": "OUT", "minimum": 0.0, "unit": "W"}]),
                       [rotator], design_intent="OPTICAL_SYSTEM")


@pytest.mark.parametrize("rotator, wavelength_nm", [
    (rotator_model(), 1029.0),  # below the span
    (rotator_model(), 1091.0),  # above it
    (rotator_model(interpolation="POINTS_ONLY"), 1047.0),  # between rows, at the rows only
    (rotator_model(rows=[("1064", "45", "0.95")]), 1064.1),  # one row: exactly it
])
def test_a_wavelength_without_a_stated_response_has_no_power_behind_the_element(rotator, wavelength_nm):
    """Its rotation and transmittance are not known: the light passes straight on (the element does not deflect it) with a
    power and polarization that are not computed, at most the power that arrived."""
    run = _run(_single(rotator, wavelength_nm))
    result = run.result
    (arrival,) = arrivals(result, "OUT")
    assert arrival.beam.power_W is None and arrival.power_upper_bound_W == pytest.approx(POWER_W, rel=REL)
    assert arrival.beam.direction == pytest.approx((1.0, 0.0, 0.0), abs=1e-15)
    (path,) = result.paths
    assert path.termination == "ESCAPED" and path.final_power_W is None
    (issue,) = [i for i in result.issues if i.code.value == "MODEL_WAVELENGTH_OUTSIDE_BAND"]
    assert issue.status.value == "OUT_OF_MODEL_SCOPE" and issue.location == "surface:FR/rotator"
    assert issue.numerical_context["power_upper_bound_W"] == arrival.power_upper_bound_W
    assert run.status.value == "OUT_OF_MODEL_SCOPE" or run.status.value == "FAIL"
    (scope,) = [i for i in run.model_scope_issues if i.code == "MODEL_WAVELENGTH_OUTSIDE_BAND"]
    assert scope.instance_id == "FR" and scope.surface_id == "rotator"
    (target,) = result.targets
    # (no value; the plane's proven interval [0, bound] meets the target's minimum of 0 W, and nothing above it)
    assert target.value is None and target.assessment.value == "MET"
    assert result.power_budget.lines[0].sinks["TERMINATED_OUT_OF_MODEL"] == pytest.approx(POWER_W, rel=REL)


def test_the_check_warns_about_a_line_outside_the_table_before_any_run():
    checked = json_api.check_project(_single(rotator_model(interpolation="POINTS_ONLY"), 1047.0))
    (warning,) = [w for w in checked["warnings"] if w["code"] == "SOURCE_WAVELENGTH_OUTSIDE_COMPONENT_BAND"]
    assert warning["details"]["reported_in"] == "engine_issues" and warning["details"]["wavelengths_nm"] == [1047.0]
    assert json_api.check_project(_single(rotator_model(), 1047.0))["warnings"] == []


def test_a_line_inside_the_table_keeps_its_result_beside_one_outside_it():
    """Partial results: the in-domain line is delivered with its stated transmittance; the other has no value and is named."""

    lines = [{"wavelength_nm": 1064.0, "power_w": POWER_W / 2.0}, {"wavelength_nm": 1200.0, "power_w": POWER_W / 2.0}]
    document = _single(rotator_model(), LAMBDA_NM)
    model = document["components"][0]["source_model"]
    del model["wavelength_nm"], model["power_w"]
    model["spectral_lines"] = lines
    run = _run(document)
    (arrival, outside) = sorted(arrivals(run.result, "OUT"), key=lambda a: a.beam.wavelength_nm)
    assert arrival.beam.wavelength_nm == 1064.0
    assert arrival.beam.power_W == pytest.approx(POWER_W / 2.0 * TF, rel=REL)
    assert (outside.beam.wavelength_nm, outside.beam.power_W) == (1200.0, None)
    assert outside.power_upper_bound_W == pytest.approx(POWER_W / 2.0, rel=REL)
    unknown = [p for p in run.result.paths if p.final_power_W is None]
    assert [p.wavelength_nm for p in unknown] == [1200.0]
    assert "1200" in next(i for i in run.result.issues if i.code.value == "MODEL_WAVELENGTH_OUTSIDE_BAND").message


def test_an_angle_outside_the_stated_range_is_out_of_model_scope_and_inside_it_is_not():
    """A rotator tilted 4 deg is inside 0-5 deg (same rotation in the laboratory about the ray, chief ray undeviated); tilted 8 deg
    the values are computed and the use is flagged."""

    def tilted(deg: float):
        half = math.radians(deg) / 2.0
        pose = quat_multiply(ALONG_X, (math.cos(half), math.sin(half), 0.0, 0.0))
        return _run(with_models(project([forward_source(), declared_component("FR", "other_optical", "fr", quaternion=pose)], [plane("OUT", (50.0, 0.0, 0.0))]),
                                [rotator_model(rows=[("1064", "45", "1")], interpolation="POINTS_ONLY")], design_intent="OPTICAL_SYSTEM"))

    inside = tilted(4.0)
    assert not [i for i in inside.model_scope_issues if i.code == "ELEMENT_AOI_OUTSIDE_RANGE"]
    field, _ = _beam(_through(inside.result, "OUT", "FWD"))
    assert _overlap(field, _lab(45.0)) == pytest.approx(1.0, rel=REL)
    outside = tilted(8.0)
    assert [i.code for i in outside.model_scope_issues] == ["ELEMENT_AOI_OUTSIDE_RANGE"]
    assert outside.status.value == "OUT_OF_MODEL_SCOPE"


def test_a_stated_power_rating_is_compared_with_the_exposure_on_it():
    rating = {"regime": "CW", "value": "1000000", "unit": "W/cm2", "wavelength_nm": "1064", "value_definition": "PEAK", "status": "DECLARED", "basis": "fixture",
              "beam_diameter_mm": "2", "beam_diameter_definition": "1/E2"}
    document = with_models(project([forward_source(), declared_component("FR", "other_optical", "fr")], [plane("OUT", (50.0, 0.0, 0.0))]),
                           [{**rotator_model(), "power_ratings": [rating]}], design_intent="OPTICAL_SYSTEM")
    surface = next(s for s in _run(document).result.exposures.surfaces if s.surface_id == "FR/rotator")
    assert surface.hits[0].incident_power_W == pytest.approx(POWER_W, rel=REL)
    assert len(surface.ratings) == 1


def test_the_optical_path_behind_it_has_no_value(tmp_path):
    """It states no optical path: phase, optical path length and wavefront are withheld behind it, while power and polarization
    stay; the pass is recorded as of undetermined delay (with the polarization state), so a coherence with a beam that does not
    make it is not stated."""

    document = with_models(project([forward_source(), declared_component("FR", "other_optical", "fr")], [plane("OUT", (50.0, 0.0, 0.0))]),
                           [rotator_model()], design_intent="OPTICAL_SYSTEM")
    run = _run(document)
    arrival = _through(run.result, "OUT", "FWD")
    assert arrival.ray_opl_defined is False
    run.write_artifacts(tmp_path)
    results = load_run(tmp_path)
    (beam,) = results.handoff("OUT")["beams"]
    assert beam["phase_rad"] is None and beam["optical_path_length_mm"] is None and beam["validity"]["phase_defined"] is False
    assert beam["power_W"] == pytest.approx(POWER_W * TF, rel=REL)
    assert results.wavefront("OUT", "FWD#0")["reason"] == "RAY_OPL_UNDEFINED"
    (record,) = [a for a in run.result.paths[0].segments if a.end_event == "faraday_rotator"]
    assert record.end_surface_id == "FR/rotator"
    assert any(key.startswith("FR/rotator:faraday_rotator:-@0.000000[") for key in arrival.group_delay_unknown_at)


def test_a_network_through_it_is_refused():
    document = with_models(project([forward_source(), declared_component("FR", "other_optical", "fr")],
                                   [{**plane("OUT", (50.0, 0.0, 0.0)), "absorbing": True, "medium": "air"}]),
                           [rotator_model()], design_intent="OPTICAL_SYSTEM")
    reply = json_api.linear_optical_network(document)
    assert reply["ok"] is False
    (problem,) = reply["error"]["problems"]
    assert problem["code"] == "PHASE_UNDEFINED_FARADAY_ROTATOR" and "FR/rotator" in problem["message"]


# ------------------------------------------------------------------------------------------- views of the same model


def test_the_router_follows_the_same_matrices_in_both_directions():
    document = with_models(project([forward_source(), declared_component("FR", "other_optical", "fr")], [plane("OUT", (50.0, 0.0, 0.0))]),
                           [rotator_model(rows=[("1064", "30", "0.95")], interpolation="POINTS_ONLY")], design_intent="OPTICAL_SYSTEM")
    router = PolarizationRouter.for_lowered(lower_project(document))
    y, z = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    forward = router.step("FR", ROUTE_PASS, LAMBDA_NM, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          LineState((1.0, 0.0, 0.0), y, z, (1 + 0j, 0j)))
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    assert forward.share == pytest.approx(0.95, rel=REL)
    assert forward.state.jones[0] == pytest.approx(c, rel=1e-12) or forward.state.field()[1] == pytest.approx(c, rel=1e-12)
    backward = router.step("FR", ROUTE_PASS, LAMBDA_NM, (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                           LineState((-1.0, 0.0, 0.0), z, y, (0j, 1 + 0j)))  # +Y in the basis (Z, Y)
    field = backward.state.field()
    assert field[1].real == pytest.approx(c, rel=1e-12) and field[2].real == pytest.approx(s, rel=1e-12)  # the laboratory turn: +30 deg
    assert router.step("FR", ROUTE_PASS, 1000.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), None) is None


def test_the_body_is_drawn_as_a_rod_and_not_as_a_lens():
    model = {**rotator_model(), "design_geometry": cylinder("10", "-10", "10")}
    document = with_models(project([forward_source(), declared_component("FR", "other_optical", "fr")], [plane("OUT", (50.0, 0.0, 0.0))]),
                           [model], design_intent="OPTICAL_SYSTEM")
    scene = _run(document).optical_view
    geom = next(e.geometry for e in scene.elements if e.element_id == "FR")
    assert geom.model_kind == "FARADAY_ROTATOR" and geom.surfaces[0].role.value == "FARADAY_ROTATOR" and geom.surfaces[0].surface_id == "rotator"
    materials = {name for name, _ in optic_mesh.body_meshes(geom)}
    assert "faraday_body" in materials and "faraday_plane" in materials and "polarization_axis" in materials
    assert "thin_lens" not in materials and "optic_glass" not in materials
    side = section.element_section(geom, "XZ")
    assert side["plate"]["polygon"] == [[-5.0, -10.0], [5.0, -10.0], [5.0, 10.0], [-5.0, 10.0]]


def test_a_stated_box_body_is_drawn_as_a_plate_and_no_body_as_a_plane():
    box = {"kind": "BOX", "center_mm": ["0", "0", "0"], "half_extents_mm": ["6", "6", "1"], "status": "NOMINAL"}
    for design_geometry, has_body in ((box, True), (None, False)):
        model = {**rotator_model(), **({"design_geometry": design_geometry} if design_geometry else {})}
        document = with_models(project([forward_source(), declared_component("FR", "other_optical", "fr")], [plane("OUT", (50.0, 0.0, 0.0))]),
                               [model], design_intent="OPTICAL_SYSTEM")
        geom = next(e.geometry for e in _run(document).optical_view.elements if e.element_id == "FR")
        materials = {name for name, _ in optic_mesh.body_meshes(geom)}
        assert ("faraday_body" in materials) is has_body and "faraday_plane" in materials and "thin_lens" not in materials


# --------------------------------------------------------------------------------------- native, saved, and browser paths


def _native_parameters(**extra) -> dict:
    return {"rows": [{"wavelength_nm": w, "rotation_deg": r, "transmittance": t} for w, r, t in ROWS], "diameter_mm": "10", "valid_aoi_deg": ["0", "5"],
            "operating_conditions": "fixture: 25 C, magnet assembly as supplied", "evidence": dict(EVIDENCE), **extra}


def test_the_native_class_is_discoverable_and_builds_the_same_model():
    catalog = json_api.native_components()
    assert catalog["ok"] is True
    (described,) = [c for c in catalog["classes"] if c["class_id"] == "rotator.faraday"]
    assert described["model_kind"] == "FARADAY_ROTATOR" and described["optical_role"] == "other_optical" and described["catalog_equivalents"] == []
    assert "sign(d . z)" in described["physics"]
    assert "optical isolator" in described["keywords"]
    built = json_api.build_native_model("rotator.faraday", _native_parameters(model_id="fr", instance_id="FR", length_mm="20"))
    assert built["ok"] is True, built
    model = built["optical_model"]
    assert model["design_geometry"]["z_min_mm"] == "-10" and model["design_geometry"]["z_max_mm"] == "10"
    assert built["component_template"]["optical_role"] == "other_optical"
    ProjectOpticalModelV1.model_validate(model)


@pytest.mark.parametrize("parameters, where", [
    (lambda p: {**p, "rows": [{"wavelength_nm": "1064", "rotation_deg": "45"}]}, "rows"),
    (lambda p: {k: v for k, v in p.items() if k != "rows"}, "rows"),
    (lambda p: {**p, "evidence": {"method": "fixture"}}, "evidence"),
    (lambda p: {**p, "valid_aoi_deg": ["0", "95"]}, "valid_aoi_deg"),
    (lambda p: {**p, "clear_aperture_diameter_mm": "12"}, "clear_aperture_diameter_mm"),
])
def test_the_native_class_refuses_a_missing_or_impossible_statement(parameters, where):
    reply = json_api.build_native_model("rotator.faraday", parameters(_native_parameters()))
    assert reply["ok"] is False
    assert where in json.dumps(reply["error"])


def test_the_native_class_builds_from_the_rows_alone():
    """Angles, operating conditions and evidence are stated when the author has them; the built model holds what was stated."""

    parameters = {k: v for k, v in _native_parameters(model_id="fr").items() if k not in ("valid_aoi_deg", "operating_conditions", "evidence")}
    built = json_api.build_native_model("rotator.faraday", parameters)
    assert built["ok"] is True, built
    physics = built["optical_model"]["model"]
    assert not {"valid_aoi_deg", "operating_conditions", "evidence"} & set(physics)
    assert ProjectOpticalModelV1.model_validate(built["optical_model"]).model.data_kind == "ASSUMED"
    full = json_api.build_native_model("rotator.faraday", _native_parameters(model_id="fr"))["optical_model"]["model"]
    assert {k: v for k, v in full.items() if k not in ("valid_aoi_deg", "operating_conditions", "evidence")} == physics
    (listed,) = [c for c in json_api.native_components()["classes"] if c["class_id"] == "rotator.faraday"]
    described = {p["name"]: p for p in listed["parameters"]}
    assert [described[name]["required"] for name in ("rows", "valid_aoi_deg", "operating_conditions", "evidence")] == [True, False, False, False]


def test_a_saved_project_authored_through_the_edit_operations_keeps_its_meaning(tmp_path):
    """Native build, ``add_optical_model`` and ``set_component_model``, the document saved as JSON and read back: same hash, same
    isolator values. (``json_api`` is the browser runtime's entry too.)"""

    built = json_api.build_native_model("rotator.faraday", _native_parameters(model_id="fr", length_mm="20"))
    base = isolator(polarizer_model("unused", 0.9, 100.0), ALONG_X)
    base["components"] = [c for c in base["components"] if c["instance_id"] != "MID"]
    base["optical_models"] = [m for m in base["optical_models"] if m["model_id"] != "unused"]
    ops = [{"op": "add_optical_model", "optical_model": built["optical_model"]},
           {"op": "add_component", "component": {**built["component_template"], "instance_id": "MID",
                                                 "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}}]
    updated = json_api.update_project(base, ops)
    assert updated["ok"] is True, updated
    saved = tmp_path / "isolator.project.json"
    saved.write_text(json.dumps(updated["project"]), encoding="utf-8")
    reread = json.loads(saved.read_text(encoding="utf-8"))
    checked = json_api.check_project(reread)
    assert checked["project_hash"] == updated["project_hash"]
    ran = json_api.run_project(copy.deepcopy(reread))
    assert ran["ok"] is True, ran
    run = _run(reread)
    forward, reverse = _hand_ports(45.0, TF)
    ports = _ports(run.result)
    assert ports["fwd_out"] / ports["fwd_in"] == pytest.approx(forward, rel=REL)
    assert ports["rev_in"] / ports["rev_out"] == pytest.approx(reverse, rel=REL)


def test_the_run_keeps_the_response_with_its_provenance():
    (row,) = [r for r in _run(_single(rotator_model(), LAMBDA_NM)).element_models if r.get("optical_model_id") == "fr"]
    assert row["model_kind"] == "FARADAY_ROTATOR" and row["fidelity"] == "IDEAL_THIN_ELEMENT"
    response = row["response"]
    assert response["operating_conditions"].startswith("fixture") and response["evidence"]["source_data_sha256"] == SHA
    assert "whichever way" in response["rotation_sense"] and len(response["rows"]) == 3


def test_a_project_without_a_rotator_keeps_its_bytes():
    """A Project document that does not use the kind serializes as before: the model field set is untouched."""

    document = with_models(project([forward_source(), declared_component("P", "polarizer", "p1")], [plane("OUT", (50.0, 0.0, 0.0))]),
                           [polarizer_model("p1", T1, ER1)], design_intent="OPTICAL_SYSTEM")
    payload = json.dumps(json_api.check_project(document)["project"])
    assert "FARADAY" not in payload and "rotator" not in payload
