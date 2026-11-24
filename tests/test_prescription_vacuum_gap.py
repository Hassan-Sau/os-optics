"""A CSV vacuum cell survives Project serialization and is traced as physical vacuum.

The compound's inter-member space is Project air. A cell enclosing evacuated space
must therefore be one native SURFACE_SEQUENCE with VACUUM among its declared media.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.local import paraxial_analysis, run_optical_project
from os_optics.optical_view.section import element_section
from os_optics.prescription import PrescriptionErrorCode, PrescriptionImportError, import_prescription, merge_into_project

WAVELENGTH_NM = 632.8
N_AIR = 1.0 + 1e-8 * (6432.8 + 2949810.0 / (146.0 - (1000 / WAVELENGTH_NM) ** 2)
                       + 25540.0 / (41.0 - (1000 / WAVELENGTH_NM) ** 2)) / (1 + 0.0034785 * 5)


def _csv(gap: str) -> str:
    return ("#! wavelength_min_nm = 632.8\n#! wavelength_max_nm = 632.8\n"
            "#! temperature_c = 20\n#! pressure_pa = 101325\n"
            "surface,radius,thickness,material,semi_diameter\n"
            "1,80,2,WINDOW,5\n"
            f"2,-60,10,{gap},5\n"
            "3,60,2,WINDOW,5\n"
            "4,-80,70,AIR,5\n"
            "[materials]\nname,kind,index,min_nm,max_nm\n"
            "WINDOW,CONSTANT_INDEX,1.52,600,700\n")


def _base_project() -> dict:
    return {
        "schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM",
        "components": [{
            "instance_id": "SRC", "optical_role": "source",
            "custom_part": {"custom_part_key": "laser", "description": "On-axis test source"},
            "nominal_pose": {"position_mm": [0, 0, -50], "quaternion_wxyz": [1, 0, 0, 0]},
            "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": WAVELENGTH_NM,
                             "power_w": 0.001, "waist_radius_mm": 0.5,
                             "waist_position_mm": 0, "m2": 1,
                             "polarization": {"kind": "LINEAR", "angle_deg": 0}},
        }],
        "observations": [{"observation_id": "OUT", "pose": {
            "position_mm": [0, 0, 80], "quaternion_wxyz": [1, 0, 0, 0]},
            "aperture": {"shape": "circle", "radius_mm": 10}}],
    }


def _reopened(tmp_path, gap: str) -> ProjectDocumentV2:
    imported = import_prescription(_csv(gap), model_id="cell", glass_source="inline")
    if gap == "VACUUM":
        (sequence,) = [model.model for model in imported.optical_models if model.model.model_kind == "SURFACE_SEQUENCE"]
        assert sequence.media == ("WINDOW", "VACUUM", "WINDOW")
        assert [s.vertex_z_mm for s in sequence.surfaces] == ["0", "2", "12", "14"]
        assert imported.report["counts"]["cemented_elements"] == 0
        assert imported.report["members"][0]["edge_thickness"] and len(imported.report["members"][0]["edge_thickness"]) == 2
        assert imported.model_materials[0].material_id == "WINDOW" and len(imported.model_materials) == 1
    return _save_and_reopen(tmp_path / f"cell_{gap}.json", imported)


def _save_and_reopen(file, imported) -> ProjectDocumentV2:
    document = merge_into_project(_base_project(), imported, instance_id="CELL")
    file.write_text(json.dumps(document.payload()), encoding="utf-8")
    reopened = ProjectDocumentV2.model_validate_json(file.read_text(encoding="utf-8"))
    assert reopened.content_hash() == document.content_hash()
    return reopened


def _unequal_csv(front_radius: int, *, rear_radius: str = "", gap: str = "VACUUM",
                 edges: tuple[str, str, str, str] = ("", "", "", "")) -> str:
    first, second, third, fourth = edges
    return ("surface,radius,thickness,material,semi_diameter,mech_semi_diameter\n"
            f"1,{front_radius},2,WINDOW,3,{first}\n"
            f"2,-{front_radius},10,{gap},3,{second}\n"
            f"3,{rear_radius},2,WINDOW,16,{third}\n"
            f"4,,40,AIR,16,{fourth}\n"
            "[materials]\nname,kind,index,min_nm,max_nm\n"
            "WINDOW,CONSTANT_INDEX,1.52,500,700\n")


def _import_unequal(source: str):
    return import_prescription(source, model_id="cell", glass_source="inline",
                               wavelength_band_nm=(Decimal("600"), Decimal("650")))


@pytest.mark.parametrize("front_radius,rear_radius", [(15, ""), (400, ""), (15, "-16")],
                         ids=["steep-front", "gentle-front", "curved-rear"])
def test_unequal_windows_keep_their_own_outlines_and_vacuum_after_reopen(tmp_path, front_radius, rear_radius):
    inferred = _import_unequal(_unequal_csv(front_radius, rear_radius=rear_radius))
    explicit = _import_unequal(_unequal_csv(front_radius, rear_radius=rear_radius, edges=("3", "3", "16", "16")))
    air = _import_unequal(_unequal_csv(front_radius, rear_radius=rear_radius, gap="AIR"))
    assert [m.model.edge_radius_mm for m in air.optical_models[1:]] == ["3", "16"]
    assert [m.model for m in inferred.optical_models] == [m.model for m in explicit.optical_models]
    unit, sequence = [entry.model for entry in inferred.optical_models]
    assert len(unit.members) == 1 and unit.envelope.diameter_mm == "32"
    assert sequence.valid_band_nm.min_nm == "600" and sequence.valid_band_nm.max_nm == "650"
    (member,) = inferred.report["members"]
    assert member["surface_edge_radii_mm"] == ["3", "3", "16", "16"]
    assert "derived_edges" not in member
    assert member["edge_thickness"] == explicit.report["members"][0]["edge_thickness"]
    assert not inferred.report["warnings"]

    def trace(file, imported):
        project = _save_and_reopen(file, imported)
        (sequence,) = [entry.model for entry in project.optical_models if entry.model.model_kind == "SURFACE_SEQUENCE"]
        assert sequence.media == ("WINDOW", "VACUUM", "WINDOW")
        assert [s.vertex_z_mm for s in sequence.surfaces] == ["0", "2", "12", "14"]
        assert [sequence.gap_outline_mm(i)[0] for i in range(3)] == [Decimal(3), Decimal(3), Decimal(16)]

        run = run_optical_project(project)
        cell = next(element.geometry for element in run.optical_view.elements if element.element_id == "CELL")
        (window_pair,) = cell.members
        section = element_section(window_pair.body)
        assert [max(abs(t) for t, _ in g["polygon"]) for g in section["media"]] == [3, 16]
        assert [max(abs(t) for segment in s["segments"] for t, _ in segment["points"])
                for s in section["surfaces"]] == [3, 3, 16, 16]
        (path,) = run.result.paths
        (gap,) = [segment for segment in path.segments if segment.medium_id == "vacuum"]
        assert gap.length_mm == pytest.approx(10, abs=1e-12)
        assert gap.refractive_index == 1.0
        assert gap.optical_path_length_mm == pytest.approx(10, abs=1e-12)
        (arrival,) = run.result.observations[0].arrivals
        return path.segments, arrival

    assert trace(tmp_path / "inferred.json", inferred) == trace(tmp_path / "explicit.json", explicit)


def test_explicit_mechanical_edges_set_the_default_only_in_their_own_window():
    inferred = _import_unequal(_unequal_csv(15, edges=("4", "", "17", "")))
    explicit = _import_unequal(_unequal_csv(15, edges=("4", "4", "17", "17")))
    assert [m.model for m in inferred.optical_models] == [m.model for m in explicit.optical_models]
    assert inferred.report["members"][0]["surface_edge_radii_mm"] == ["4", "4", "17", "17"]

    with pytest.raises(PrescriptionImportError) as exc:
        _import_unequal(_unequal_csv(15, edges=("16", "", "", "")))
    (problem,) = exc.value.problems
    assert problem.code is PrescriptionErrorCode.EDGE_INVALID and problem.location.row == 2
    assert "its mechanical semi-diameter 16 mm" in problem.message


def test_vacuum_windows_cannot_overlap_outside_the_clear_aperture():
    source = _unequal_csv(400, rear_radius="-16", edges=("4", "4", "", ""))
    source = source.replace(",10,VACUUM,", ",0.35,VACUUM,")
    with pytest.raises(PrescriptionImportError) as exc:
        _import_unequal(source)
    (problem,) = exc.value.problems
    assert problem.code is PrescriptionErrorCode.EDGE_INVALID and problem.location.row == 4
    assert "windows facing across the vacuum gap between surfaces 2 and 3" in problem.message


def test_vacuum_gap_is_traced_after_project_save_and_reopen(tmp_path):
    vacuum = _reopened(tmp_path, "VACUUM")
    air = _reopened(tmp_path, "AIR")
    v_run, a_run = (run_optical_project(document) for document in (vacuum, air))
    v_path, a_path = (run.result.paths[0] for run in (v_run, a_run))
    vacuum_segments = [seg for seg in v_path.segments if seg.medium_id == "vacuum"]
    assert len(vacuum_segments) == 1
    (gap,) = vacuum_segments
    assert gap.length_mm == pytest.approx(10, abs=1e-12)
    assert gap.refractive_index == 1.0
    assert gap.optical_path_length_mm == pytest.approx(10, abs=1e-12)
    assert any(seg.medium_id == "air" and seg.refractive_index == pytest.approx(N_AIR) for seg in v_path.segments)
    assert any(seg.medium_id == "air" and seg.length_mm == pytest.approx(10) for seg in a_path.segments)
    (v_arrival,) = v_run.result.observations[0].arrivals
    (a_arrival,) = a_run.result.observations[0].arrivals
    assert v_arrival.beam.geometric_length_mm == a_arrival.beam.geometric_length_mm == pytest.approx(130, abs=1e-11)
    assert (v_arrival.beam.optical_path_length_mm - a_arrival.beam.optical_path_length_mm) == pytest.approx(
        10 * (1 - N_AIR), abs=2e-10)
    assert 1000 * (a_arrival.beam.optical_path_length_mm - v_arrival.beam.optical_path_length_mm) == pytest.approx(
        2.717919, abs=0.00001)  # micrometres; independent Edlen 1953 reference


def test_curved_cell_back_focus_reflects_vacuum_refraction_after_reopen(tmp_path):
    def bfl(gap):
        document = _reopened(tmp_path, gap)
        result = paraxial_analysis(document, {"components": ["CELL"], "wavelength_nm": WAVELENGTH_NM})
        return result["system"]["bfl_mm"]

    air_bfl, vacuum_bfl = bfl("AIR"), bfl("VACUUM")
    assert air_bfl == pytest.approx(29.360913, abs=0.00003)
    assert vacuum_bfl == pytest.approx(29.350082, abs=0.00003)
    assert air_bfl - vacuum_bfl == pytest.approx(0.010831, abs=0.00006)


def test_unclosed_vacuum_gap_is_refused_not_silently_relabelled_air():
    source = _csv("VACUUM").replace("4,-80,70,AIR,5", "4,-80,70,VACUUM,5")
    with pytest.raises(PrescriptionImportError, match="close the evacuated space"):
        import_prescription(source, glass_source="inline")


def test_dummy_vacuum_plane_does_not_change_the_gap_medium():
    source = _csv("VACUUM").replace("2,-60,10,VACUUM,5", "2,-60,4,VACUUM,5\nD,,6,VACUUM,5")
    imported = import_prescription(source, glass_source="inline")
    (sequence,) = [model.model for model in imported.optical_models if model.model.model_kind == "SURFACE_SEQUENCE"]
    assert sequence.media == ("WINDOW", "VACUUM", "WINDOW")
    assert [surface.vertex_z_mm for surface in sequence.surfaces] == ["0", "2", "12", "14"]
    assert imported.report["counts"]["dummy_surfaces"] == 1


def test_a_separate_stop_in_vacuum_is_refused_instead_of_moving_it_into_air():
    source = _csv("VACUUM").replace("2,-60,10,VACUUM,5", "2,-60,4,VACUUM,5\nSTO,,6,VACUUM,5")
    source = source.replace("semi_diameter\n", "semi_diameter,stop\n", 1).replace("STO,,6,VACUUM,5", "STO,,6,VACUUM,5,yes")
    with pytest.raises(PrescriptionImportError, match="inter-member medium is Project air"):
        import_prescription(source, glass_source="inline")
