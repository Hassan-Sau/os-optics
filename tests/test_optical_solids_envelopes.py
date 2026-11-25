"""A compound unit's envelope is its whole-unit boundary (final follow-up review F2).

Another component's optical body that shares volume with a unit's declared envelope (a lens put inside a Keplerian
telescope's tube) is refused like any two optical bodies sharing volume: ``OPTICAL_BODIES_OVERLAP`` on the intruding
component, ``other_instance_id`` the unit, ``other_region`` ``UNIT_ENVELOPE``, with the depth and position. Touching the
envelope is allowed, the unit's own members are not intruders, and the envelopes the native library and the prescription
importer write bound their members tightly enough that a part placed just outside a unit stays valid.

Independent references: the envelope of the built unit (a cylinder about the unit axis), a declared flint plate of known
thickness on the axis (shared axial length = its thickness) and beside it (depth = tube radius - distance of the plate's
nearest edge from the axis).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from declared_helpers import flint, sequence, surface
from os_optics.local import json_api
from os_optics.native_library import add_to_project, build_native_model

I = [1.0, 0.0, 0.0, 0.0]
#: A flint plate (constant index: no catalog copy), 2 mm thick, radius 5.
PLATE = {"model_id": "plate", "model": sequence([surface(1, "0", clear="4.5"), surface(2, "2", clear="4.5")], ["FLINT"], edge="5")}
THICKNESS, PLATE_RADIUS = 2.0, 5.0
SINGLET = {"focal_length_mm": "100", "design_wavelength_nm": "632.8", "material": "N-BK7", "diameter_mm": "25.4"}
NATIVE_UNITS = {
    "compound.keplerian_telescope": {
        "lens1": {"class_id": "lens.singlet", "parameters": {**SINGLET, "focal_length_mm": "50", "diameter_mm": "12.7", "form": "PLANO_EXIT"}},
        "lens2": {"class_id": "lens.singlet", "parameters": {**SINGLET, "focal_length_mm": "150"}},
        "design_wavelength_nm": "632.8", "focal_stop_diameter_mm": "0.03"},
    "compound.galilean_telescope": {
        "lens1": {"class_id": "lens.ideal_thin", "parameters": {"focal_length_mm": "-25", "clear_aperture_diameter_mm": "10"}},
        "lens2": {"class_id": "lens.ideal_thin", "parameters": {"focal_length_mm": "100", "clear_aperture_diameter_mm": "30"}}},
    "compound.relay_4f": {
        "lens1": {"class_id": "lens.singlet", "parameters": {**SINGLET, "form": "PLANO_EXIT"}},
        "lens2": {"class_id": "lens.singlet", "parameters": {**SINGLET, "focal_length_mm": "200", "form": "PLANO_ENTRY"}},
        "fourier_stop_diameter_mm": "2", "fourier_obstruction_diameter_mm": "0.2"},
}
FIXTURES = Path(__file__).parent / "fixtures" / "prescriptions"


def _base() -> dict:
    return {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM", "components": [{
        "instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "Test laser."},
        "nominal_pose": {"position_mm": [0.0, 0.0, -500.0], "quaternion_wxyz": I},
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 632.8, "power_w": 0.001, "waist_radius_mm": 1.0,
                         "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}],
        "observations": [{"observation_id": "far", "pose": {"position_mm": [0.0, 0.0, 2000.0], "quaternion_wxyz": I},
                          "aperture": {"shape": "circle", "radius_mm": 50.0}}]}


def _with_plate(document: dict, position) -> dict:
    out = copy.deepcopy(document)
    out["optical_models"] = [*out.get("optical_models", []), copy.deepcopy(PLATE)]
    out["model_materials"] = [*out.get("model_materials", []), flint()]
    out["components"].append({"instance_id": "P", "optical_role": "window", "optical_model_id": "plate",
                              "custom_part": {"custom_part_key": "own-plate", "description": "My plate."},
                              "nominal_pose": {"position_mm": list(position), "quaternion_wxyz": I}})
    return out


def _native_unit(class_id: str) -> tuple[dict, dict]:
    build = build_native_model(class_id, NATIVE_UNITS[class_id])
    document = add_to_project(_base(), build, instance_id="U", position_mm=[0.0, 0.0, 0.0], quaternion_wxyz=I)
    (unit,) = [m for m in document["optical_models"] if m["model"]["model_kind"] == "COMPOUND"]
    return document, unit["model"]["envelope"]


def _imported_unit(name: str) -> tuple[dict, dict]:
    answer = json_api.import_prescription((FIXTURES / name).read_text(encoding="utf-8"),
                                          {"model_id": "rx", "into": _base(), "instance_id": "U", "pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": I}})
    assert answer["ok"], answer
    (unit,) = [m for m in answer["project"]["optical_models"] if m["model"]["model_kind"] == "COMPOUND"]
    return answer["project"], unit["model"]["envelope"]


def _issues(document: dict) -> list[dict]:
    report = json_api.inspect_optical_solids(document)
    assert report["ok"], report
    return report["issues"]


def test_a_part_inside_a_telescope_tube_is_refused_with_the_unit_depth_and_position():
    document, envelope = _native_unit("compound.keplerian_telescope")
    assert _issues(document) == []  # the unit alone: its members are not intruders
    z_mid = (float(envelope["z_min_mm"]) + float(envelope["z_max_mm"])) / 2.0
    tube = float(envelope["diameter_mm"]) / 2.0
    # on the axis, between the lenses: the plate's whole thickness is inside the tube (exact coaxial measure)
    (issue,) = _issues(_with_plate(document, (0.0, 0.0, z_mid)))
    assert (issue["code"], issue["severity"], issue["instance_id"], issue["other_instance_id"], issue["other_region"]) == (
        "OPTICAL_BODIES_OVERLAP", "ERROR", "P", "U", "UNIT_ENVELOPE")
    assert issue["value_mm"] == pytest.approx(THICKNESS, abs=1e-6) and issue["location"] == "component P"
    assert issue["zone"] == "UNIT_ENVELOPE"
    assert "envelope of compound unit U" in issue["message"] and "outside the unit" in issue["message"]
    # beside the axis, its nearest edge 10 mm from it: 12.7 - 10 mm inside the tube (sampled and refined)
    (beside,) = _issues(_with_plate(document, (0.0, 15.0, z_mid)))
    assert beside["value_mm"] == pytest.approx(tube - (15.0 - PLATE_RADIUS), abs=1e-6)
    # the refusal reaches check_project / run_project, structured
    refused = json_api.check_project(_with_plate(document, (0.0, 0.0, z_mid)))
    (problem,) = refused["error"]["problems"]
    assert problem["code"] == "OPTICAL_BODIES_OVERLAP" and problem["details"]["other_region"] == "UNIT_ENVELOPE"
    assert problem["details"]["zone"] == "UNIT_ENVELOPE"
    assert problem["details"]["other_instance_id"] == "U" and problem["details"]["value_mm"] == pytest.approx(THICKNESS, abs=1e-6)


def test_touching_or_clearing_a_unit_envelope_is_valid():
    document, envelope = _native_unit("compound.keplerian_telescope")
    z_min, z_max = float(envelope["z_min_mm"]), float(envelope["z_max_mm"])
    tube = float(envelope["diameter_mm"]) / 2.0
    # on the exit face beside the axis (on the axis it would touch L2's vertex: a contact light crosses, refused on its own),
    # just past the exit, just before the entrance, and alongside the tube touching its side
    for position in ((0.0, 8.0, z_max), (0.0, 0.0, z_max + 0.001), (0.0, 0.0, z_min - THICKNESS - 0.001),
                     (0.0, tube + PLATE_RADIUS, (z_min + z_max) / 2.0)):
        issues = [i for i in _issues(_with_plate(document, position)) if i["severity"] == "ERROR"]
        assert issues == [], position


@pytest.mark.parametrize("producer", [*NATIVE_UNITS, "cooke_triplet.csv", "doublet_singlet.csv"])
def test_library_and_imported_units_bound_their_members_tightly(producer, real_catalog_store):
    """A part 0.01 mm before a unit's first and after its last member body is valid; one inside is refused."""

    document, envelope = _native_unit(producer) if producer.startswith("compound.") else _imported_unit(producer)
    assert _issues(document) == [], producer
    z_min, z_max = float(envelope["z_min_mm"]), float(envelope["z_max_mm"])
    assert json_api.check_project(_with_plate(document, (0.0, 0.0, z_max + 0.01)))["ok"]
    assert json_api.check_project(_with_plate(document, (0.0, 0.0, z_min - THICKNESS - 0.01)))["ok"]
    inside = [i for i in _issues(_with_plate(document, (0.0, 0.0, (z_min + z_max) / 2.0 - 1.0))) if i.get("other_region") == "UNIT_ENVELOPE"]
    assert len(inside) == 1 and inside[0]["other_instance_id"] == "U"


def test_a_declared_polarization_elements_plane_rule_is_stated_plainly():
    """Final follow-up NIT: the contract keeps ``plane_z_mm`` free (it is the catalog's field too); the Engine refuses a
    declared polarization element off its frame origin and says what to do instead."""

    built = json_api.build_native_model("polarizer.linear", {"transmittance": "0.9", "extinction_ratio": "1000", "diameter_mm": "25.4",
                                                             "valid_band_nm": ["400", "700"]})
    document = _base()
    for plane_z, ok in (("0", True), ("2", False)):
        candidate = copy.deepcopy(document)
        model = copy.deepcopy(built["optical_model"])
        model["model"]["plane_z_mm"] = plane_z
        candidate["optical_models"] = [model]
        candidate["components"].append(copy.deepcopy(built["component_template"]))
        reply = json_api.check_project(candidate)
        assert reply["ok"] is ok, reply
    (problem,) = reply["error"]["problems"]
    assert problem["code"] == "OPTICAL_MODEL_INVALID"
    message = problem["message"]
    assert "plane is its frame origin, so plane_z_mm must be 0 (it is 2)" in message
    assert "move the component 2 mm along its local +Z" in message and "Engine's rule for declared polarization elements" in message
