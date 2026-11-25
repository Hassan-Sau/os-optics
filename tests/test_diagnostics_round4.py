"""Round-4 verification of the diagnostics (G1, G3).

* **G1** -- light inside an element's own body (a lens's glass) that falls beyond a surface's support meets the element's
  edge or mount: an ordinary known loss, booked ``APERTURE_TRUNCATION`` like the main beam's, with no "passes beside the
  element" branch. Such a branch used to start inside the glass, skip the lens and end ``MEDIUM_MISMATCH`` at the next lens,
  so the dark planes behind 2-4 ordinary catalog lenses read UNKNOWN (3.7e-12 W ...). On a test-only copy whose N-BK7
  explicitly assumes alpha=0, relays of 2, 3 and 4 AR-coated PCX lenses bound the side detector far below 1 nW and no
  probe path ends out of model scope; the unchanged public catalog leaves bulk loss unknown.
* Untraced light whose fate a probe does not know is named by why (a path ended out of model scope, the probes' trace
  budget, ...), not by the kind of light it started as.
* **G3** -- the interval of a bounded detected power is printed with enough digits to tell its ends apart.

The relays are the review's (``OS-OPT-LENS-PCX-D25P4-F100-...-AR400-700``, 532 nm, 200 mm apart, a 5 mm detector 60 mm
off the axis): its Gaussian tails there are exp(-2 (60 / 1.3)^2) of the beam, and every ghost of the coaxial lenses stays
on the axis. A ghost that two lens surfaces focus to a sub-micron waist diverges to a radius of 43-172 mm at the last lens,
so most of it passes beside that lens, and the part of it whose flow crosses the detector's plane (along the beam) on the
detector's 10 mm is bounded (``PARTIAL_CROSSING``): the plane receives between 0 W and a few 1e-10 W.
"""

from __future__ import annotations

import math
import re

import pytest

from catalog_store_helpers import REAL_CATALOG, catalog_version_of
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.errors import Status
from os_optics.local import lower_project, run_optical_project
from os_optics.optics.rays import tracer as tracer_module

from helpers import run
from test_diagnostics_round3 import _defocused, _t

LENS = "OS-OPT-LENS-PCX-D25P4-F100-T3P6-R51P5-NBK7-AR400-700-W587P6"
ALONG_X = [0.5, 0.5, 0.5, 0.5]
FACING_MINUS_Y = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]


def _relay(lenses: int) -> dict:
    version = catalog_version_of(REAL_CATALOG)
    components = [{
        "instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "laser"},
        "nominal_pose": {"position_mm": [-400, 0, 0], "quaternion_wxyz": ALONG_X}, "selected_parameters": [], "retained_dofs": [],
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "waist_radius_mm": 0.5, "waist_position_mm": 0, "m2": 1,
                         "polarization": {"kind": "LINEAR", "angle_deg": 0}, "wavelength_nm": 532, "power_w": 1e-3}}]
    for k in range(lenses):
        components.append({"instance_id": f"L{k + 1}", "optical_role": "lens", "product": {"optical_sku": LENS, "catalog_version": version},
                           "nominal_pose": {"position_mm": [-350 + 200 * k, 0, 0], "quaternion_wxyz": ALONG_X},
                           "selected_parameters": [], "retained_dofs": []})
    last = -350 + 200 * (lenses - 1)

    def plane(oid, position, quaternion, radius):
        return {"observation_id": oid, "pose": {"position_mm": position, "quaternion_wxyz": quaternion},
                "aperture": {"shape": "circle", "radius_mm": radius, "center_offset_mm": [0, 0], "obstruction_radius_mm": 0},
                "absorbing": True, "medium": "air"}

    return {"schema": "opensci.project", "project_schema_version": "2.0", "materials": [], "components": components,
            "observations": [plane("MAIN", [last + 60, 0, 0], ALONG_X, 12.0), plane("SIDE", [last + 30, 60, 0], FACING_MINUS_Y, 5.0)],
            "targets": [{"target_id": "main", "metric": "power_W", "observation_id": "MAIN", "minimum": 1e-5, "unit": "W", "parameters": []},
                        {"target_id": "side_max", "metric": "power_W", "observation_id": "SIDE", "maximum": 1e-9, "unit": "W",
                         "parameters": []}]}


@pytest.mark.parametrize("lenses", [2, 3, 4])
def test_dark_planes_behind_relays_of_catalog_lenses_retain_faint_bounds(lenses, real_catalog_store, monkeypatch):
    """The relay's side detector has no traced power, only a bound below its 1 nW maximum (the faint light left out and the
    part of the ghosts spilled beside the last lens that crosses its plane on its active area). That followed light is not
    traced: the plane is UNKNOWN and states no power. Its maximum is graded on [0, bound] and is MET, the bound lying under
    it (targets only), and no probe path ends out of model scope."""
    probe_ends: list[str] = []
    original = tracer_module.Tracer.trace_probes

    def recording(self, roots):
        outcomes = original(self, roots)
        probe_ends.extend(path.termination for path in self.out.paths)
        return outcomes

    monkeypatch.setattr(tracer_module.Tracer, "trace_probes", recording)
    document = _relay(lenses)
    published = run_optical_project(document).result
    published_side = next(o for o in published.observations if o.observation_id == "SIDE")
    assert published_side.total_power_W is None and published_side.quantity_state.value == "UNKNOWN"
    assert 0 < published_side.power_upper_bound_W <= 1e-3  # unknown bulk loss permits zero through the line input
    assert next(t for t in published.targets if t.target_id == "side_max").issue_codes == ("BULK_LOSS_NOT_ESTABLISHED",)
    assert any(i.numerical_context.get("untraced_light") == ["BULK_LOSS_UNKNOWN"] for i in published.issues if i.location == "observation:SIDE")

    # A separate test-only ideal alpha=0 copy isolates G1's ghost geometry; the public catalog still states no transmission.
    project = lower_project(document).request.project
    glass = next(m for m in project.materials if m.material_id == "catalog:N-BK7")
    assert "absorption_coefficient_per_mm" not in glass.model_fields_set and glass.bulk_loss is None
    ideal = project.model_copy(update={"materials": tuple(m.model_copy(update={"absorption_coefficient_per_mm": 0.0}) if m is glass else m
                                                           for m in project.materials)})
    assert ideal.components == project.components and ideal.observations == project.observations
    result = run(ideal)
    t = {x.target_id: x for x in result.targets}
    side_max = t["side_max"]  # (no known 0 W target value: the proven bound decides the maximum)
    assert side_max.value is None and side_max.issue_codes == ("UNTRACED_LIGHT_BOUNDED",)
    assert (side_max.status.value, side_max.assessment.value) == ("PASS", "MET")
    side = next(o for o in result.observations if o.observation_id == "SIDE")
    assert not side.power_complete and 0 < side.untraced_power_W < side.power_upper_bound_W <= 1e-9
    # (followed light only, under its bound: between 0 and that bound, no value -- the bound grades the target)
    assert side.quantity_state.value == "BOUNDED" and side.power_lower_bound_W == 0.0 and side.total_power_W is None
    (issue,) = (i for i in result.issues if i.location == "observation:SIDE")
    assert issue.numerical_context["faint_power_upper_bound_W"] > 0  # (the light left out as negligible still bounds it)
    assert issue.numerical_context["untraced_light"] == ["UNTRACED_REFLECTION"]
    # (the plane's finding is a notice and its target is MET on the proven bound: nothing leaves the run undecided)
    assert result.status.value == "PASS"
    assert probe_ends and not {"MEDIUM_MISMATCH", "GRAZING_INCIDENCE", "TRACE_BUDGET", "MAX_INTERACTIONS"} & set(probe_ends)


X = (1.0, 0.0, 0.0)


def test_untraced_light_whose_fate_is_not_known_is_named_by_why():
    """A normal-incidence window's ghosts travel back past the source onto a mirror at grazing incidence (89.95 deg): the
    probe ends there out of model scope, so a dark detector is UNKNOWN for that reason -- the plane issue names untraced
    light that 'reached a path ended out of model scope' (UNTRACED_LIGHT_OUT_OF_MODEL), not untraced surface reflections."""
    laser = B.laser("SRC", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=X)
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=10.0, material="N-BK7", position=(20.0, 0.0, 0.0))
    graze = math.radians(89.95)
    mirror = B.mirror("M", position=(-30.0, 0.0, 0.0), normal=(math.cos(graze), math.sin(graze), 0.0), clear_radius_mm=50.0)
    dark = B.observation_plane("DARK", position=(60.0, 40.0, 0.0), normal=X, radius_mm=5.0, absorbing=True)
    result = run(ProjectSnapshot(components=(laser, window, mirror), observations=(dark,),
                                 targets=(Target(target_id="dark_max", metric="power_W", observation_id="DARK", unit="W", maximum=1e-9),)))
    (target,) = result.targets
    assert (target.status, target.value, target.issue_codes) == (Status.UNKNOWN, None, ("OUT_OF_MODEL_SCOPE",))
    (issue,) = [i for i in result.issues if i.code.value == "OBSERVATION_NOT_REACHED"]
    assert issue.numerical_context["untraced_light"] == ["UNTRACED_LIGHT_OUT_OF_MODEL"]
    assert "reached a path ended out of model scope" in issue.message and "surface reflections" not in issue.message


def test_a_narrow_interval_is_printed_with_ends_that_differ():
    """G3: a 1.5 mm detector holds all but ~1.5e-11 W of the annulus; the target message prints the two ends with enough
    digits to differ (they used to read 0.000606532 W and 0.000606532 W) and states the width."""
    result = run(_defocused(1.5, _t("p", "OUT", minimum=0.0)))
    (target,) = result.targets
    low, high = re.search(r"between (\S+) W and (\S+) W, a width of", target.message).groups()
    assert low != high and float(low) < float(high)
