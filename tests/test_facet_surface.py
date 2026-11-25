"""Triangle-facet surfaces: the contract validates the mesh, states what the facets stand for and which facets take their own
interaction; a run holding a valid one traces it facet by facet (tests/test_facet_tracing.py has the optical references). A
mesh that breaks the contract's checks stays invalid input."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import json

from os_optics import EngineConfiguration, ValidationRequest, validate, validate_json
from os_optics import builders as B
from os_optics.contracts import (EngineeringComponentSnapshot, FacetApproximation, MeshFacetGroup, MeshSurface, MirrorInteraction,
                                 ProjectSnapshot)
from os_optics.contracts.optics import AbsorberInteraction
from os_optics.interop import entries

CUBE_VERTICES = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                 (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0))
# counter-clockwise seen from outside
CUBE_TRIANGLES = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
SHEET = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
EXACT = FacetApproximation(kind="FACETED_EXACT", source="polyhedral beam dump, CAD export")


def _mesh(vertices, triangles, *, closed: bool = False, interaction=None) -> MeshSurface:
    return MeshSurface(surface_id="facets", vertices_mm=vertices, triangles=triangles, approximation=EXACT, closed=closed,
                       interaction=interaction or AbsorberInteraction())


def test_a_closed_outward_facing_cube_and_an_open_sheet_are_valid():
    cube = _mesh(CUBE_VERTICES, CUBE_TRIANGLES, closed=True)
    assert len(cube.triangles) == 12 and cube.closed
    sheet = _mesh(SHEET, ((0, 1, 2), (0, 2, 3)), interaction=MirrorInteraction())
    assert sheet.geometry == "mesh"


@pytest.mark.parametrize("vertices, triangles, closed, message", [
    (SHEET, ((0, 1, 4),), False, "outside 0 .. 3"),
    (SHEET, ((0, 1, 1),), False, "repeats a vertex"),
    (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)), ((0, 1, 2),), False, "degenerate"),
    (SHEET, ((0, 1, 2), (0, 3, 2)), False, "same direction"),  # (the second facet turned over)
    (SHEET, ((0, 1, 2), (0, 2, 3)), True, "one facet only"),  # (an open sheet stated closed)
    (CUBE_VERTICES, tuple((a, c, b) for a, b, c in CUBE_TRIANGLES), True, "face inward"),
])
def test_invalid_meshes_are_refused(vertices, triangles, closed, message):
    with pytest.raises(ValidationError, match=message):
        _mesh(vertices, triangles, closed=closed)


def test_a_tessellated_surface_states_its_chord_deviation():
    with pytest.raises(ValidationError, match="max_chord_deviation_mm"):
        FacetApproximation(kind="TESSELLATED_FROM_CAD")
    with pytest.raises(ValidationError, match="max_chord_deviation_mm"):
        FacetApproximation(kind="FACETED_EXACT", max_chord_deviation_mm=0.01)
    assert FacetApproximation(kind="TESSELLATED_FROM_CAD", max_chord_deviation_mm=0.002).max_chord_deviation_mm == 0.002


def _facet_request() -> ValidationRequest:
    dump = EngineeringComponentSnapshot(instance_id="DUMP", role="dump", surfaces=(_mesh(CUBE_VERTICES, CUBE_TRIANGLES, closed=True),))
    src = B.laser("S", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.2, position=(0.5, 0.5, -50.0), direction=(0.0, 0.0, 1.0))
    return ValidationRequest(project=ProjectSnapshot(name="facets", components=(src, dump)),
                             configuration=EngineConfiguration(run_layout=False, run_control=False))


def test_a_run_with_a_facet_surface_traces_it():
    result = validate(_facet_request())
    # before: refused as out of model scope (FACET_SURFACE_NOT_TRACED); now the polyhedral dump absorbs the beam
    execution = result.outcome.execution
    assert execution.outcome == "COMPLETED" and execution.termination == "COMPLETED"
    (path,) = result.paths
    assert path.termination == "ABSORBED" and path.segments[-1].end_surface_id == "DUMP/facets.t0"  # the bottom face z = 0
    # the 1 mm face holds all of the 0.2 mm beam but its far tails: they fall beyond the face's edges, stated and not followed
    (edge,) = [i for i in result.issues if i.code.value.startswith("FACET")]
    assert edge.code.value == "FACET_EDGE_DIVIDES_BEAM" and edge.status.value == "WARNING" and edge.numerical_context["loss_fraction"] < 1e-5
    sinks = result.power_budget.total_sinks
    assert sinks["ABSORBED"] + sinks["TERMINATED_OUT_OF_MODEL"] == pytest.approx(1e-3, rel=1e-12)
    assert any(entry.entry_id == "cad-brep-geometry" for entry in entries.ENTRIES)


@pytest.mark.parametrize("groups, message", [
    (((0, 1), (1, 2)), "is in facet groups 0 and 1"),
    (((12,),), "outside 0 .. 11"),
])
def test_facet_groups_name_each_facet_once(groups, message):
    with pytest.raises(ValidationError, match=message):
        MeshSurface(surface_id="facets", vertices_mm=CUBE_VERTICES, triangles=CUBE_TRIANGLES, approximation=EXACT, closed=True,
                    interaction=AbsorberInteraction(),
                    facet_groups=tuple(MeshFacetGroup(triangles=g, interaction=MirrorInteraction()) for g in groups))


def test_a_mesh_that_breaks_the_contract_stays_invalid_input():
    document = json.loads(_facet_request().model_dump_json())
    surface = document["project"]["components"][1]["surfaces"][0]
    surface["triangles"] = [[a, c, b] for a, b, c in surface["triangles"]]  # (every face turned inward)
    result = json.loads(validate_json(json.dumps(document)))
    assert result["status"] == "FAIL"
    assert result["outcome"]["execution"]["termination"] == "INVALID_INPUT"
    assert [i["code"] for i in result["issues"]] == ["INVALID_INPUT"]
