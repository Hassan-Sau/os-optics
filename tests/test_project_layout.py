"""``json_api.project_layout``: the layout of a Project without a run (``os_optics.optical_view.layout``).

Independent references: the elements, observation planes, axis markers and display media of a layout are compared with
those of the run of the same Project (``run_project``'s ``optical_scene``: the scene builder bound to the Engine's own
trace); each source's nominal beam axis is compared with the chief ray the Engine traced for the same source (its points
must be the first points of that chief ray, exactly, wherever the axis is followed). Rules the run does not exercise
(the back of a mirror, oblique refraction, a Project without a source) are checked against hand-made geometry.
"""

from __future__ import annotations

import json
import math
import warnings

import pytest

from os_optics.local import EXAMPLE_KINDS, json_api
from os_optics.optical_view import parse_glb

LAYOUT_KEYS = {"schema", "schema_version", "view_kind", "view_notice", "frame", "length_unit", "up_axis", "project", "catalog_version",
               "settings", "elements", "observation_planes", "axis_markers", "display_media", "nominal_axes", "nominal_axes_unavailable",
               "bounds", "camera_hint"}
#: Keys of a run's results: none of them is part of a layout.
RESULT_WORDS = ("beam_paths", "power_W", "final_power_W", "arrivals", "engine_run", "outcome")


def _example(kind: str) -> dict:
    return json_api.example_project_document(kind=kind)["project"]


def _layout(document: dict) -> tuple[dict, bytes]:
    reply, files = json_api.call_method("project_layout", {"document": document})
    assert reply["ok"] is True, reply
    json.dumps(reply, allow_nan=False)
    return reply["layout"], files["optical_layout.glb"]


def _no_footprints(planes: list[dict]) -> list[dict]:
    return [{**p, "footprints": []} for p in planes]


@pytest.mark.parametrize("kind", EXAMPLE_KINDS)
def test_layout_draws_the_runs_elements_and_follows_the_traced_chief_ray(kind):
    document = _example(kind)
    layout, glb = _layout(document)
    assert set(layout) == LAYOUT_KEYS
    assert (layout["schema"], layout["schema_version"], layout["view_kind"]) == ("opensci.optical_layout_view", "1.0", "OPTICAL_LAYOUT_VIEW")
    run = json_api.run_project(document)
    assert run["ok"] is True
    scene = run["optical_scene"]
    # the same bodies, poses, planes, axis markers and display media as the run's Optical Simulation View
    assert layout["elements"] == scene["elements"]
    assert _no_footprints(layout["observation_planes"]) == _no_footprints(scene["observation_planes"])
    assert all(p["footprints"] == [] for p in layout["observation_planes"])
    assert layout["axis_markers"] == scene["axis_markers"]
    assert layout["display_media"] == scene["display_media"]
    assert layout["project"] == scene["project"]
    # each nominal axis is the start of the chief ray the Engine traced from that source (a prefix, exactly)
    chiefs = {p["source_element_id"]: p for p in scene["beam_paths"] if p["path_id"].endswith("#0")}
    assert {a["source_id"] for a in layout["nominal_axes"]} == set(chiefs)
    for axis in layout["nominal_axes"]:
        chief = chiefs[axis["source_id"]]["chief_ray"]
        n = len(axis["points_mm"])
        assert n >= 2
        assert chief["points_mm"][:n] == axis["points_mm"], (kind, axis)
        if axis["end"] == "FREE":
            assert chief["points_mm"] == axis["points_mm"]
            assert chief["display_extension_to_mm"] == axis["display_extension_to_mm"]
        else:
            assert axis["display_extension_to_mm"] is None
            assert axis["end_surface_id"] == axis["surface_ids"][-1]
    # nothing of a result: no beam path, power, status or target anywhere in the layout
    text = json.dumps(layout)
    for word in RESULT_WORDS:
        assert f'"{word}"' not in text, word
    assert glb[:4] == b"glTF"


def test_the_folded_layout_turns_at_the_mirror_and_is_deterministic():
    document = _example("folded")
    layout, glb = _layout(document)
    (axis,) = layout["nominal_axes"]
    # source at x = -150 along +X, the fold mirror at the origin, the lens at (0, 50, 0) on the reflected beam
    assert axis["points_mm"][:3] == [[-150.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 50.0, 0.0]]
    assert axis["surface_ids"][0] == "M1/S1"
    assert axis["end"] == "FREE"
    again, glb_again = _layout(document)
    assert again == layout and glb_again == glb


def test_layout_glb_names_the_layout_and_holds_no_beam():
    _, glb = _layout(_example("folded"))
    gltf, _binary = parse_glb(glb)
    assert gltf["asset"]["extras"]["opensci_view"] == "OPTICAL_LAYOUT_VIEW"
    assert gltf["asset"]["extras"]["layout_schema"] == "opensci.optical_layout_view"
    names = [n.get("name", "") for n in gltf["nodes"]]
    assert "optical_layout_view" in names
    assert {"element:SRC", "element:M1", "element:L1", "observation:focal_plane", "nominal_axis:SRC", "optical_axes"} <= set(names)
    assert not any(name.startswith("beam:") for name in names)
    assert all(n.get("extras", {}).get("opensci_view") == "OPTICAL_LAYOUT_VIEW" for n in gltf["nodes"] if "extras" in n)
    assert not any("footprints" in n.get("extras", {}) and n["extras"]["footprints"] for n in gltf["nodes"])


def test_the_runs_scene_glb_is_unchanged_by_the_layout_view():
    # the run's GLB still names the Optical Simulation View at every node (the layout's renaming is its own)
    run = json_api.run_project(_example("folded"))
    gltf, _ = parse_glb(run["files"]["optical_scene.glb"])
    assert gltf["scenes"][0]["name"] == "Optical Simulation View"
    assert [n["name"] for n in gltf["nodes"] if n.get("name") == "optical_simulation_view"]
    assert all(n["extras"]["opensci_view"] == "OPTICAL_SIMULATION_VIEW" for n in gltf["nodes"] if "extras" in n)


def _turn_component(document: dict, instance_id: str, quaternion_wxyz: list[float]) -> dict:
    for component in document["components"]:
        if component["instance_id"] == instance_id:
            component["nominal_pose"]["quaternion_wxyz"] = quaternion_wxyz
    return document


def test_a_mirror_met_on_its_back_ends_the_axis_absorbed():
    document = _example("folded")
    mirror = next(c for c in document["components"] if c["instance_id"] == "M1")
    w, x, y, z = mirror["nominal_pose"]["quaternion_wxyz"]
    # turned half a turn about its own local Y: the reflecting face now looks away from the source
    half = (0.0, 0.0, 1.0, 0.0)
    turned = [w * half[0] - x * half[1] - y * half[2] - z * half[3], w * half[1] + x * half[0] + y * half[3] - z * half[2],
              w * half[2] - x * half[3] + y * half[0] + z * half[1], w * half[3] + x * half[2] - y * half[1] + z * half[0]]
    layout, _ = _layout(_turn_component(document, "M1", turned))
    (axis,) = layout["nominal_axes"]
    assert axis["end"] == "ABSORBED"
    assert axis["end_surface_id"].startswith("M1/")


def _axis_is_the_start_of_the_traced_chief_ray(document: dict) -> dict:
    """The one nominal axis of the layout, checked point for point against the chief ray the run traces first from that source."""

    layout, _ = _layout(document)
    (axis,) = layout["nominal_axes"]
    run = json_api.run_project(document)
    assert run["ok"] is True
    (chief,) = [p["chief_ray"] for p in run["optical_scene"]["beam_paths"] if p["path_id"].endswith("#0") and p["source_element_id"] == axis["source_id"]]
    n = len(axis["points_mm"])
    assert chief["points_mm"][:n] == axis["points_mm"]
    return axis


def test_oblique_refraction_is_followed_by_snell_s_law_at_the_source_s_wavelength():
    """The reference lens tilted by 10 degrees about Z: the axis meets its surfaces off normal incidence and refracts there with
    the lens material's index at the source's wavelength, as the trace refracts its chief ray -- the axis is the start of that
    chief ray, point for point, through both surfaces (it used to end on the first, NOT_FOLLOWED)."""

    document = _example("focus")
    lens = next(c for c in document["components"] if c["instance_id"] == "L1")
    w, x, y, z = lens["nominal_pose"]["quaternion_wxyz"]
    a = math.radians(10.0) / 2.0
    rz = (math.cos(a), 0.0, 0.0, math.sin(a))
    turned = [rz[0] * w - rz[3] * z, rz[0] * x - rz[3] * y, rz[0] * y + rz[3] * x, rz[0] * z + rz[3] * w]
    axis = _axis_is_the_start_of_the_traced_chief_ray(_turn_component(document, "L1", turned))
    assert axis["end"] != "NOT_FOLLOWED" and axis["surface_ids"][:2] == ["L1/S1", "L1/S2"] and len(axis["points_mm"]) >= 3
    source = next(c for c in document["components"] if c["optical_role"] == "source")
    assert axis["wavelength_nm"] == source["source_model"]["wavelength_nm"]


def test_a_tilted_window_displaces_the_axis_by_the_plate_formula():
    """A 5 mm N-BK7 plate tilted by 30 degrees in a beam along +X: the axis leaves it parallel to itself, displaced by
    t sin(theta) (1 - cos(theta) / sqrt(n^2 - sin^2(theta))) with the Sellmeier index written here, and is the start of the
    traced chief ray."""

    from declared_helpers import NBK7_SELLMEIER, declared_component, nbk7, sequence, surface, with_models
    from local_helpers import ALONG_X, plane, project, quat_multiply, source, turn_about_z

    theta = math.radians(30.0)
    pose = quat_multiply(turn_about_z(math.cos(theta / 2.0), math.sin(theta / 2.0)), ALONG_X)
    model = {"model_id": "win", "model": sequence([surface(1, "0"), surface(2, "5")], ["N-BK7"])}
    document = with_models(project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=780.0), declared_component("W", "window", "win", quaternion=pose)],
                                   [plane("OUT", (100.0, 0.0, 0.0))]), [model], [nbk7()])
    axis = _axis_is_the_start_of_the_traced_chief_ray(document)
    assert axis["surface_ids"][:2] == ["W/S1", "W/S2"] and axis["wavelength_nm"] == 780.0 and axis["end"] != "NOT_FOLLOWED"
    lam = 0.78
    n = math.sqrt(1.0 + sum(float(b) * lam * lam / (lam * lam - float(c)) for b, c in zip(NBK7_SELLMEIER["b"], NBK7_SELLMEIER["c_um2"])))
    shift = 5.0 * math.sin(theta) * (1.0 - math.cos(theta) / math.sqrt(n * n - math.sin(theta) ** 2))
    entry, exit_, after = axis["points_mm"][1], axis["points_mm"][2], axis["points_mm"][3]
    assert entry == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert abs(exit_[1]) == pytest.approx(shift, rel=1e-6) and after[1] == pytest.approx(exit_[1], abs=1e-9)  # (parallel to +X again)
    assert after[0] == pytest.approx(100.0, abs=1e-9)


def test_total_internal_reflection_in_a_corner_cube_is_followed_as_the_trace_follows_it():
    """A glass corner cube: the axis refracts in, turns three times by total internal reflection and refracts out, as the traced
    chief ray does (it used to end on the entry face)."""

    from test_facet_solid_project import _corner_cube_design

    axis = _axis_is_the_start_of_the_traced_chief_ray(_corner_cube_design())
    facets = [k for k, surface_id in enumerate(axis["surface_ids"]) if surface_id.startswith("CC/facets")]
    assert len(facets) == 5 and axis["end"] != "NOT_FOLLOWED"  # (in, three reflections, out)
    points = axis["points_mm"]  # (points_mm[k + 1] is where the axis meets surface_ids[k])
    out = points[facets[-1] + 1]
    back = points[facets[-1] + 2] if len(points) > facets[-1] + 2 else axis["display_extension_to_mm"]
    assert back[0] < out[0] and back[1] == pytest.approx(out[1], abs=1e-6) and back[2] == pytest.approx(out[2], abs=1e-6)  # (retroreflected along -X)


def test_a_tabulated_thin_lens_met_off_its_centre_turns_the_axis_by_its_focal_length_at_the_first_line():
    """A tabulated thin lens moved 1 mm off the beam: the axis turns as the lens turns a ray, with the focal length the table gives at
    the source's first line (500 nm; the reference interpolates the rows here), and is the start of the traced chief ray. A line
    outside the table is not followed."""

    from test_tabulated_thin_lens import PLANE_X, focal, ttl_project

    document = ttl_project()
    next(c for c in document["components"] if c["instance_id"] == "ML")["nominal_pose"]["position_mm"] = [0.0, 1.0, 0.0]
    axis = _axis_is_the_start_of_the_traced_chief_ray(document)
    assert axis["wavelength_nm"] == 500.0 and axis["end"] != "NOT_FOLLOWED" and len(axis["points_mm"]) >= 3
    assert axis["points_mm"][2][1] == pytest.approx(PLANE_X / focal(500.0), rel=1e-6)  # (h = -1 mm: a slope of +1 / f behind the lens)
    outside = ttl_project(lines=((700.0, 0.001),))
    next(c for c in outside["components"] if c["instance_id"] == "ML")["nominal_pose"]["position_mm"] = [0.0, 1.0, 0.0]
    layout, _ = _layout(outside)
    (axis,) = layout["nominal_axes"]
    assert axis["end"] == "NOT_FOLLOWED" and axis["end_surface_id"] == axis["surface_ids"][-1] and axis["wavelength_nm"] == 700.0


def test_a_design_without_a_source_has_a_layout_without_axes():
    document = _example("focus")
    document["components"] = [c for c in document["components"] if c["optical_role"] != "source"]
    layout, _ = _layout(document)
    assert layout["nominal_axes"] == []
    assert [e["element_id"] for e in layout["elements"]] == ["L1"]


def test_overlapping_bodies_still_have_a_layout_and_a_refused_project_is_an_envelope():
    document = _example("beam_expander")
    lenses = [c for c in document["components"] if c["optical_role"] == "lens"]
    lenses[1]["nominal_pose"]["position_mm"] = list(lenses[0]["nominal_pose"]["position_mm"])  # two lenses in one place
    assert json_api.check_project(document)["ok"] is False
    layout, _ = _layout(document)
    assert len(layout["elements"]) == len(document["components"])
    refused = json_api.project_layout({**_example("focus"), "unexpected": 1})
    assert refused["ok"] is False and refused["error"]["code"] == "LOCAL_RUN_ERROR"


@pytest.mark.parametrize("position_mm", [1e300, 1e160, 1e30])
def test_a_design_too_large_to_draw_is_refused_in_customer_words(position_mm):
    # check_project accepts the pose; the drawing overflows (coordinates that are not finite, or a degenerate mesh)
    document = _example("focus")
    next(c for c in document["components"] if c["instance_id"] == "L1")["nominal_pose"]["position_mm"][0] = position_mm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # (numpy's overflow warnings on the way)
        refused = json_api.project_layout(document)
    json.dumps(refused, allow_nan=False)
    assert refused["ok"] is False and refused["error"]["code"] == "LAYOUT_INVALID"
    (problem,) = refused["error"]["problems"]
    message = problem["message"]
    assert message.startswith("the layout cannot be drawn")
    assert "positions" in message
    # no internal model or exception names, no Python wording
    for word in ("CameraHint", "V1", "validation error", "Error", "min()", "iterable"):
        assert word not in message, (word, message)
