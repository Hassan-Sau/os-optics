"""A spectral line a filter blocks completely is a known zero, not a missing result.

The real catalog's 850 nm typical-curve bandpass publishes T = 0 exactly at 898 nm. A source with lines at 850 and 898 nm
sends them through it onto an absorbing detector. The 898 nm line is traced to its end at the filter (its reflection is
declared not traced), so the detector receives 0 W of it: a maximum passes, a minimum fails with value 0, and the line is
listed in ``power_by_wavelength`` with 0 W and ``complete``. A line or path the run does not have selects nothing
(UNKNOWN, ``TARGET_SELECTION_EMPTY``). Reference: linear interpolation of the published curve, read here from the catalog
file (not through the Engine).
"""

from __future__ import annotations

import copy
import json

import pytest

from catalog_store_helpers import REAL_CATALOG, catalog_version_of
from os_optics.local import run_optical_project
from os_optics.results import RunResults


FILTER = "OS-OPT-FLT-BP-D25P4-C850-TYP-W50-300-1200-AOI0T10-T6-CA22"
ALONG_X = [0.5, 0.5, 0.5, 0.5]
LINE_W = 1e-3


def _published_transmittance(wavelength_nm: float) -> float:
    catalog = json.loads(REAL_CATALOG.read_bytes())
    product = next(p for p in catalog["products"] if p["optical_sku"] == FILTER)
    coating_id = product["simulation_models"][0]["surfaces"][0]["coating_id"]
    spectrum = next(c for c in catalog["coatings"] if c["coating_id"] == coating_id)["spectrum"]
    nodes = [(float(w), float(t)) for w, t in zip(spectrum["wavelength_nm"], spectrum["transmittance"])]
    for (w0, t0), (w1, t1) in zip(nodes, nodes[1:]):
        if w0 <= wavelength_nm <= w1:
            return t0 + (t1 - t0) * (wavelength_nm - w0) / (w1 - w0)
    raise AssertionError(wavelength_nm)


def _target(target_id: str, **kw) -> dict:
    return {"target_id": target_id, "metric": "power_W", "observation_id": "DET", "unit": "W", "parameters": [], **kw}


def _project(lines, targets) -> dict:
    model = {"model_kind": "GAUSSIAN_BEAM", "waist_radius_mm": 0.5, "waist_position_mm": 0, "m2": 1,
             "polarization": {"kind": "LINEAR", "angle_deg": 0}}
    if len(lines) > 1:
        model["spectral_lines"] = [{"wavelength_nm": w, "power_w": LINE_W} for w in lines]
    else:
        model.update(wavelength_nm=lines[0], power_w=LINE_W)
    return {
        "schema": "opensci.project", "project_schema_version": "2.0", "materials": [], "targets": targets,
        "components": [
            {"instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "laser"},
             "nominal_pose": {"position_mm": [-100, 0, 0], "quaternion_wxyz": ALONG_X}, "selected_parameters": [],
             "retained_dofs": [], "source_model": model},
            {"instance_id": "BP850", "optical_role": "filter",
             "product": {"optical_sku": FILTER, "catalog_version": catalog_version_of(REAL_CATALOG)},
             "nominal_pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": ALONG_X}, "selected_parameters": [], "retained_dofs": []},
        ],
        "observations": [
            {"observation_id": "DET", "pose": {"position_mm": [100, 0, 0], "quaternion_wxyz": ALONG_X},
             "aperture": {"shape": "circle", "radius_mm": 5, "center_offset_mm": [0, 0], "obstruction_radius_mm": 0},
             "absorbing": True, "medium": "air"},
        ],
    }


def test_a_line_the_filter_blocks_is_a_known_zero(real_catalog_store):
    assert _published_transmittance(898.0) == 0.0
    targets = [_target("blocked_max", wavelength_nm=898, maximum=1e-9), _target("blocked_min", wavelength_nm=898, minimum=1e-6),
               _target("pass_min", wavelength_nm=850, minimum=1e-4), _target("absent_line", wavelength_nm=700, maximum=1e-9),
               _target("absent_path", parameters=[{"key": "path_id", "value": "SRC#7"}], maximum=1e-9)]
    result = run_optical_project(_project([850, 898], targets)).result
    t = {x.target_id: x for x in result.targets}
    assert (t["blocked_max"].status.value, t["blocked_max"].value, t["blocked_max"].issue_codes) == ("PASS", 0.0, ("NO_TRACED_ARRIVAL",))
    assert (t["blocked_min"].status.value, t["blocked_min"].value) == ("FAIL", 0.0)
    assert t["pass_min"].status.value == "PASS"
    assert t["pass_min"].value == pytest.approx(LINE_W * _published_transmittance(850.0), rel=1e-9)
    for target_id in ("absent_line", "absent_path"):
        assert (t[target_id].status.value, t[target_id].value, t[target_id].issue_codes) == ("UNKNOWN", None, ("TARGET_SELECTION_EMPTY",))
    (det,) = result.observations
    assert [(s.wavelength_nm, s.complete) for s in det.power_by_wavelength] == [(850.0, True), (898.0, True)]
    assert det.power_by_wavelength[1].power_W == 0.0 and det.power_complete
    # the run fails only through the minimum on the blocked line; the absent selections make it UNKNOWN otherwise
    assert result.status.value == "FAIL"
    only_rejection = run_optical_project(_project([850, 898], targets[:1] + targets[2:3])).result
    assert only_rejection.status.value == "PASS"


def test_a_detector_that_only_a_blocked_line_would_reach_is_dark_not_failed(real_catalog_store):
    result = run_optical_project(_project([898], [_target("leak", maximum=1e-9)])).result
    (target,) = result.targets
    assert (target.status.value, target.value) == ("PASS", 0.0)
    (det,) = result.observations
    assert det.status.value == "PASS" and det.arrivals == () and det.total_power_W == 0.0 and det.power_complete
    (issue,) = [i for i in result.issues if i.code.value == "OBSERVATION_NOT_REACHED"]
    assert issue.status.value == "NOT_APPLICABLE"  # (a notice; the plane's record does not depend on the target)
    assert result.status.value == "PASS"


LENS_AR = "OS-OPT-LENS-PCX-D25P4-F100-T3P6-R51P5-NBK7-AR700-1100-W587P6"


@pytest.mark.parametrize("lossless", [False, True], ids=["unstated-bulk-loss", "lossless-fixture"])
def test_a_blocked_line_keeps_the_appropriate_bound_with_an_ar_coated_lens_upstream(real_catalog_store, lossless):
    """The filter stops the chief ray. The glass with unstated loss can send unresolved light around the filter.
    The lossless fixture isolates the much smaller bound from finite ghost-trace resolution."""
    document = _project([898], [_target("blocked_max", wavelength_nm=898, maximum=1e-9)])
    document["components"].insert(1, {
        "instance_id": "L1", "optical_role": "lens", "product": {"optical_sku": LENS_AR, "catalog_version": catalog_version_of(REAL_CATALOG)},
        "nominal_pose": {"position_mm": [-60, 0, 0], "quaternion_wxyz": ALONG_X}, "selected_parameters": [], "retained_dofs": []})
    from local_helpers import with_lossless_bulk

    catalog = with_lossless_bulk(json.loads(REAL_CATALOG.read_bytes())) if lossless else REAL_CATALOG
    result = run_optical_project(document, catalog=catalog).result
    (target,) = result.targets
    (det,) = result.observations
    assert not det.power_complete and det.quantity_state.value == "BOUNDED"
    if lossless:
        assert target.status.value == "PASS" and target.assessment.value == "MET"
        assert det.untraced_power_W == det.total_power_W == 0.0
        assert 1e-12 < det.power_upper_bound_W <= 1e-9
    else:
        assert target.status.value == result.status.value == "UNKNOWN" and target.value is None
        assert "BULK_LOSS_NOT_ESTABLISHED" in target.issue_codes
        assert det.untraced_power_W >= 0.000995 and det.total_power_W is None
        assert det.power_upper_bound_W >= 0.000995
        (arrival,) = det.arrivals
        assert arrival.beam.power_W is None and arrival.power_upper_bound_W == 0.0
        assert arrival.beam.optical_path_length_mm is not None
        (line,) = result.power_budget.lines
        assert line.conserved and line.sinks["UNTRACED_REFLECTION"] == pytest.approx(5e-6)
        assert line.sinks["BULK_LOSS_UNKNOWN"] == pytest.approx(0.000995)
        assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in result.issues}


def test_irradiance_map_preserves_unknown_fate_from_upstream_glass(real_catalog_store):
    """A known bypass beam must not make an unknown-loss glass route seem dark in the public map.

    Another source starts after the filter and lights the detector; the earlier
    glass route's chief ray has zero upper, but its upstream bulk-loss bucket
    remains unresolved when unmodelled bypasses cannot be ruled out.
    """
    document = _project([898], [])
    document["components"].insert(1, {
        "instance_id": "L1", "optical_role": "lens", "product": {"optical_sku": LENS_AR, "catalog_version": catalog_version_of(REAL_CATALOG)},
        "nominal_pose": {"position_mm": [-60, 0, 0], "quaternion_wxyz": ALONG_X}, "selected_parameters": [], "retained_dofs": []})
    bypass = copy.deepcopy(document["components"][0])
    bypass["instance_id"] = "BYPASS"
    bypass["custom_part"]["custom_part_key"] = "bypass-laser"
    bypass["nominal_pose"]["position_mm"] = [20, 0, 0]
    document["components"].append(bypass)
    result = run_optical_project(document).result
    (det,) = result.observations
    assert det.total_power_W == pytest.approx(LINE_W)
    assert any(a.beam.power_W is None and a.power_upper_bound_W == 0.0 for a in det.arrivals)
    assert result.power_budget.conserved
    field = RunResults(result=result).irradiance_map("DET")
    assert field["status"] == "OK"
    assert not field["map_complete"]
    assert field["untraced_power_W"] >= 0.000995  # do not pretend the upstream bucket is a known zero
    assert field["untraced_power_W"] <= det.power_upper_bound_W - det.total_power_W + 1e-15
