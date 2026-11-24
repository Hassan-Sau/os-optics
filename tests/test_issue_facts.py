"""Every code listed in ``os_optics.errors.issue_facts.ISSUE_FACTS`` carries its facts as context keys: a scene produces an
issue of the code and each key equals the value computed here without Engine code. A headless client reads these
structured keys rather than parsing human-readable sentences."""

from __future__ import annotations

import math
import re

import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import ProjectSnapshot
from os_optics.errors import ErrorCode
from os_optics.errors.issue_facts import ISSUE_FACTS, issue_fact_keys
from os_optics.local import run_optical_project
from os_optics.optics.attenuation import ND_FOOTPRINT_OD_TOLERANCE

from helpers import run
from test_exposure_bounds import _gradient_nd_document

CFG = NumericalConfig()
UNIT_SUFFIXES = ("_deg", "_rad", "_mm", "_W", "_nm")


def _issues(result, code: ErrorCode) -> list:
    found = [issue for issue in result.issues if issue.code is code]
    assert found, f"the scene produced no {code.value}"
    return found


def _laser_to_plane(*parts) -> ProjectSnapshot:
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=1.0, position=(0, 0, 20), direction=(1, 0, 0)), *parts),
        observations=(B.observation_plane("d", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=50.0),),
    )


def test_every_listed_key_is_named_with_its_unit():
    for code, facts in ISSUE_FACTS.items():
        keys = [fact.key for fact in facts]
        assert len(set(keys)) == len(keys), code
        for fact in facts:
            assert re.fullmatch(r"[a-z][a-z0-9_]*(_W)?", fact.key), fact.key
            assert fact.meaning, fact.key
            if fact.unit is None:
                assert not fact.key.endswith(UNIT_SUFFIXES), fact.key
            else:
                assert fact.key.endswith(f"_{fact.unit}"), fact.key
    assert issue_fact_keys(ErrorCode.TARGET_NOT_MET) == ()


@pytest.mark.parametrize(("radius_mm", "code"), [(1.6, ErrorCode.GAUSSIAN_TRUNCATION), (0.6, ErrorCode.GAUSSIAN_SEVERE_TRUNCATION)])
def test_a_truncating_stop_states_the_share_it_takes(radius_mm: float, code: ErrorCode):
    """A 1 mm (1/e^2) beam 10 mm from its waist (Rayleigh range 5.9 m) through a circular stop of radius r loses
    exp(-2 r^2 / w^2): 6.0e-3 at 1.6 mm (above the warning fraction), 0.49 at 0.6 mm (above the scope fraction)."""
    stop = B.aperture_stop("A", position=(10, 0, 20), axis=(1, 0, 0), radius_mm=radius_mm)
    (issue,) = _issues(run(_laser_to_plane(stop)), code)
    assert issue.location.startswith("surface:A")
    assert set(issue_fact_keys(code)) <= set(issue.numerical_context)
    assert issue.numerical_context["loss_fraction"] == pytest.approx(math.exp(-2.0 * radius_mm**2), rel=1e-4)


@pytest.mark.parametrize("kind", ["polarizer", "waveplate"])
def test_an_oblique_polarizing_element_states_the_angle_and_the_limit(kind: str):
    """The element's axis is turned 10 deg beyond the warning angle in the plane of the beam: the angle of incidence is that
    turn (geometry), the limit the configuration's ``retarder_warn_aoi_deg``."""
    turn = CFG.retarder_warn_aoi_deg + 10.0
    a = math.radians(turn)
    axis = (math.cos(a), math.sin(a), 0)
    part = (B.polarizer("E", position=(30, 0, 20), axis=axis, angle_deg=0.0) if kind == "polarizer" else
            B.waveplate("E", position=(30, 0, 20), axis=axis, fast_axis_angle_deg=0.0, retardance_rad=math.pi, design_wavelength_nm=532.0))
    issues = _issues(run(_laser_to_plane(part)), ErrorCode.RETARDER_OBLIQUE_INCIDENCE)
    for issue in issues:
        assert issue.location.startswith("surface:E")
        assert set(issue_fact_keys(ErrorCode.RETARDER_OBLIQUE_INCIDENCE)) <= set(issue.numerical_context)
        assert issue.numerical_context["aoi_deg"] == pytest.approx(turn, abs=1e-9)
        assert issue.numerical_context["warn_deg"] == CFG.retarder_warn_aoi_deg


def test_an_attenuator_whose_density_varies_across_the_beam_states_the_densities():
    """OD 0 -> 4 over x -2..2 mm (1 OD per mm) at the centre of a beam of 1/e^2 radius 1.5 mm (waist 120 mm before it, Rayleigh
    range several metres): the footprint spans OD 2 -/+ 1.5, a variation of 3 (2 w x the slope)."""
    (issue,) = _issues(run_optical_project(_gradient_nd_document(with_axicon=False)).result, ErrorCode.ND_DENSITY_VARIES_ACROSS_BEAM)
    assert issue.location == "surface:ND/attenuator"
    ctx = issue.numerical_context
    assert set(issue_fact_keys(ErrorCode.ND_DENSITY_VARIES_ACROSS_BEAM)) <= set(ctx)
    assert ctx["od_min"] == pytest.approx(0.5, abs=0.01) and ctx["od_max"] == pytest.approx(3.5, abs=0.01)
    assert ctx["od_variation"] == pytest.approx(ctx["od_max"] - ctx["od_min"], abs=1e-12)
    assert ctx["od_variation_tolerance"] == ND_FOOTPRINT_OD_TOLERANCE
