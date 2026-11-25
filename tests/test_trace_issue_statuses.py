"""The status of a trace-layer finding says what the finding is.

* A physical outcome the calculation completes is a ``WARNING``: a beam whose chief ray falls on an opaque rim
  (``CLIPPED``) is stopped there, and every target on what it reaches is decided by the trace.
* A data-coverage or model limit for the quantities behind it is ``OUT_OF_MODEL_SCOPE``: a coating or a material outside
  its stated data (``COATING_OUT_OF_RANGE``, ``MATERIAL_OUT_OF_RANGE``), an angle of incidence beyond the interface
  model (``GRAZING_INCIDENCE`` at its limit), a ray parallel to a thin lens (``PARALLEL_RAY``), and between two beams of a
  pair a medium mismatch or a beam parallel to the plane.
* A scene that contradicts itself stays ``FAIL``: overlapping solids (the trace's ``MEDIUM_MISMATCH``), an element axis
  along the ray (``DEGENERATE_SURFACE``), a material model that gives no index (``MATERIAL_INVALID``), a numerical failure.

These statuses feed only the diagnostic aggregate ``ValidationResult.status``: ``outcome.execution`` and the targets'
assessments do not read them.
"""

from __future__ import annotations

import math

import numpy as np

from os_optics import builders as B
from os_optics.contracts import ConstantCoating, ProjectSnapshot, Target
from os_optics.optics.materials.coatings import CoatingEvaluator

from helpers import CFG, run

X = (1.0, 0.0, 0.0)


def _status(result, code: str) -> set[str]:
    return {i.status.value for i in result.issues if i.code.value == code}


def test_a_beam_clipped_by_a_rim_is_a_warning_and_the_run_is_not_failed_by_it():
    """The chief ray falls on the plate of a stop 1.5 mm beside its 1 mm opening: the beam is stopped there (a physical
    outcome); the part of it the opening passes is light the trace does not follow, so a power maximum behind is not
    failed by the clip itself."""
    project = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.3, position=(0.0, 0.0, 0.0), direction=X, power_W=1e-3),
                    B.aperture_stop("S", position=(50.0, 1.5, 0.0), axis=X, radius_mm=1.0)),
        observations=(B.observation_plane("P", position=(100.0, 0.0, 0.0), normal=X, radius_mm=10.0),),
        targets=(Target(target_id="dark", metric="power_W", observation_id="P", maximum=1e-6, unit="W"),))
    result = run(project)
    assert _status(result, "CLIPPED") == {"WARNING"}
    assert result.status.value != "FAIL" and result.outcome.execution.outcome.value == "COMPLETED"
    assert result.targets[0].assessment.value != "UNMET"


def test_grazing_incidence_beyond_the_model_is_out_of_model_scope():
    n = (-math.cos(math.radians(89.95)), math.sin(math.radians(89.95)), 0.0)
    project = ProjectSnapshot(components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.1, position=(0.0, 0.0, 20.0), direction=X),
                                          B.mirror("M", position=(50.0, 0.0, 20.0), normal=n, clear_radius_mm=200.0)))
    result = run(project)
    assert _status(result, "GRAZING_INCIDENCE") == {"OUT_OF_MODEL_SCOPE"}
    (issue,) = [i for i in result.issues if i.code.value == "GRAZING_INCIDENCE"]
    assert issue.numerical_context["aoi_deg"] > issue.numerical_context["limit_deg"] and issue.numerical_context["path_id"]


def test_data_outside_a_coating_or_a_material_is_out_of_model_scope():
    coating = CoatingEvaluator(ConstantCoating(reflectance_s=0.5, transmittance_s=0.5, reflectance_p=0.5, transmittance_p=0.5,
                                               valid_wavelength_nm=(400.0, 500.0), max_aoi_deg=40.0), CFG)
    for wavelength, aoi in ((650.0, 10.0), (450.0, 45.0)):
        assert coating.coefficients(wavelength, np.array([aoi]), location="x").issue.status.value == "OUT_OF_MODEL_SCOPE"
    window = B.window("W", thickness_mm=2.0, clear_radius_mm=10.0, material="N-BK7", position=(30.0, 0.0, 0.0))
    result = run(ProjectSnapshot(components=(B.laser("L", wavelength_nm=2600.0, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0),
                                                     direction=X), window)))
    assert _status(result, "MATERIAL_OUT_OF_RANGE") == {"OUT_OF_MODEL_SCOPE"}
    assert result.status.value == "OUT_OF_MODEL_SCOPE"


def test_a_pair_in_different_media_or_parallel_to_the_plane_is_out_of_model_scope():
    from os_optics.optics.interference import analyse_pair  # noqa: PLC0415
    from test_interferometer_open_phase import _record  # noqa: PLC0415

    a = _record("a", 1e-3, (0.0, 0.0), 1.0, 0.0, (), 100.0, 100.0)
    b = _record("b", 1e-3, (0.0, 0.0), 1.0, 0.0, (), 100.0, 100.0)
    b.n = 1.5
    pair, issues = analyse_pair(a, b, CFG)
    assert pair.status.value == "OUT_OF_MODEL_SCOPE" and {i.status.value for i in issues if i.code.value == "MEDIUM_MISMATCH"} == {
        "OUT_OF_MODEL_SCOPE"}
    b = _record("b", 1e-3, (0.0, 0.0), 1.0, 0.0, (), 100.0, 100.0)
    b.direction = np.array([1.0, 0.0, 0.0])  # (along the plane)
    pair, issues = analyse_pair(a, b, CFG)
    assert pair.status.value == "OUT_OF_MODEL_SCOPE"
    assert {i.status.value for i in issues if i.code.value == "GRAZING_INCIDENCE"} == {"OUT_OF_MODEL_SCOPE"}
