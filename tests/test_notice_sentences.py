"""The sentence rule of an Engine finding (R-19): ``message`` is one sentence -- what, where, how much -- and every
number in it is also a key of ``numerical_context``; the mechanism lives in the code's documentation entry and the
quantity's ``value_basis``. Checked on the templates of the tracer's not-computed power, coating and response limits,
grazing and parallel rays, the two-beam phase and coherence findings, and the pipeline's invalid input.
"""

from __future__ import annotations

import math
import re

import pytest

import json

from os_optics import SCHEMA_VERSION, builders as B
from os_optics.validation.pipeline import validate_json
from os_optics.contracts import ConstantCoating, ProjectSnapshot

from helpers import run
from test_central_obstruction import _project as obstructed, _stop
from test_geometric_power_basis import _run as geometric_run, _two_stops
from test_trace_issue_statuses import X

NUMBER = re.compile(r"(?<![\w./#])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w/])")
#: The longest message the rule allows (one sentence a reader takes in at once).
LONGEST = 220


def _numbers(value: object) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [x for item in value for x in _numbers(item)]
    if isinstance(value, dict):
        return [x for item in value.values() for x in _numbers(item)]
    return []


def _follows_the_rule(issue) -> None:
    message = issue.message
    assert len(message) <= LONGEST, (issue.code.value, len(message), message)
    assert message.count(". ") == 0 and "; " not in message, (issue.code.value, message)  # one sentence
    stated = _numbers(issue.numerical_context)
    for text in NUMBER.findall(message):
        value = float(text)
        assert any(math.isclose(value, x, rel_tol=5e-4, abs_tol=0.0) or (value == 0.0 and x == 0.0) for x in stated), (
            issue.code.value, text, message, issue.numerical_context)


def _issues(result, *codes: str):
    found = [i for i in result.issues if i.code.value in codes]
    assert found, codes
    return found


def test_the_not_computed_power_findings_follow_the_rule():
    for issue in _issues(run(obstructed(0.5, extra=(_stop("STOP2", 20.0, 0.6),))), "ANNULAR_BEAM_NOT_FOLLOWED"):
        _follows_the_rule(issue)
    for issue in [i for i in geometric_run(_two_stops(1.0, 0.1)).issues if i.numerical_context.get("power_upper_bound_W")]:
        _follows_the_rule(issue)


def test_the_coating_and_response_findings_follow_the_rule():
    splitter = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=650.0, waist_radius_mm=0.4, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.beam_splitter("BS", position=(30, 0, 20), normal=(-1, 1, 0), coating=ConstantCoating(
                        reflectance_s=0.5, reflectance_p=0.5, transmittance_s=0.5, transmittance_p=0.5, valid_wavelength_nm=(400.0, 500.0)))))
    for issue in _issues(run(splitter), "COATING_OUT_OF_RANGE"):
        _follows_the_rule(issue)


def test_the_grazing_finding_follows_the_rule():
    n = (-math.cos(math.radians(89.95)), math.sin(math.radians(89.95)), 0.0)
    result = run(ProjectSnapshot(components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.1, position=(0.0, 0.0, 20.0), direction=X),
                                             B.mirror("M", position=(50.0, 0.0, 20.0), normal=n, clear_radius_mm=200.0))))
    for issue in _issues(result, "GRAZING_INCIDENCE"):
        if issue.status.value != "WARNING":
            _follows_the_rule(issue)


def test_the_two_beam_findings_that_remain_follow_the_rule():
    from os_optics.config import NumericalConfig  # noqa: PLC0415
    from os_optics.optics.interference import analyse_pair  # noqa: PLC0415
    from test_interferometer_open_phase import _record  # noqa: PLC0415

    lens = tuple(f"S{i:02d}/lens:thin_lens:+@0.000000" for i in range(3))
    a = _record("a", 1e-3, (0.0, 0.0), 1.0, 0.0, lens, 100.0, 1000.0)
    b = _record("b", 1e-3, (0.0, 0.0), 1.0, 0.0, (), 100.0, 1000.0)
    _pair, issues = analyse_pair(a, b, NumericalConfig())
    (phase,) = [i for i in issues if i.code.value == "PHASE_UNDEFINED"]
    _follows_the_rule(phase)
    b.group_opl_mm = None
    _pair, issues = analyse_pair(a, b, NumericalConfig())
    (group_index,) = [i for i in issues if i.code.value == "COHERENCE_UNDEFINED"]
    _follows_the_rule(group_index)


def test_an_invalid_request_names_its_first_field_error():
    reply = json.loads(validate_json(json.dumps({"schema_version": SCHEMA_VERSION, "project": {"components": "none"}})))
    (issue,) = [i for i in reply["issues"] if i["code"] == "INVALID_INPUT"]
    first = issue["numerical_context"]["errors"][0]
    assert issue["message"] == f"the request does not satisfy the public contract at {first}"[: len(issue["message"])]
    assert first in issue["message"] or len(first) > 140
    assert len(issue["message"]) <= LONGEST


@pytest.mark.parametrize("message", ["the power of beam L#0 behind S2/stop is not computed, at most 0.000865 W"])
def test_the_rule_reads_numbers_and_ids(message):
    assert NUMBER.findall(message) == ["0.000865"]
