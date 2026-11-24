"""The level below which light is left out is a caller setting of its own (``NumericalConfig.negligible_power_fraction``).

Two different things used to share ``energy_conservation_tol``: the share of a source line's power below which a branch of
light is not followed, probed or listed (a statement about the result), and the numerical tolerance of the closure checks
(R + T <= 1, unit Jones norm, the ledger's residual, equality of indices). They are two fields now. The scene is a 1 W beam
through an uncoated window to a plane behind it; the hand values are Fresnel's: each face reflects ``R = ((n - 1) / (n + 1))^2``,
the beam passes ``(1 - R)^2`` and the window's ghosts toward the plane carry ``(1 - R)^2 R^(2k)`` (the first about 1.5e-3 W).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from os_optics import NumericalConfig
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.evaluation.capabilities import capability

from helpers import run

SRC = Path(__file__).resolve().parents[1] / "src" / "os_optics"


def _window() -> ProjectSnapshot:
    src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    window = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    behind = B.observation_plane("MON", position=(100.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0))
    return ProjectSnapshot(components=(src, window), observations=(behind,),
                           targets=(Target(target_id="p", observation_id="MON", metric="power_W", maximum=2.0),))


def _plane(result):
    (plane,) = result.observations
    return plane


def test_the_level_moves_with_its_own_setting_and_the_ledger_keeps_its_tolerance():
    default = run(_window())
    plane = _plane(default)
    passed = plane.total_power_W
    reflectance = 1.0 - passed ** 0.5  # (1 - R)^2 passes
    ghost = (1.0 - reflectance) ** 2 * reflectance ** 2
    assert 0.03 < reflectance < 0.05 and ghost > 1e-3
    # at the default level (1e-9 of the line) the ghosts are followed to the plane: the series (1 - R)^2 R^(2k), k >= 1
    assert plane.untraced_power_W == pytest.approx(ghost / (1.0 - reflectance ** 2), rel=1e-7)

    # a level above the ghost: it is left out, the plane's traced power is the same number and the ledger closes as before
    coarse = run(_window(), numerical=NumericalConfig(negligible_power_fraction=1e-2))
    assert _plane(coarse).total_power_W == passed and _plane(coarse).untraced_power_W == 0.0
    assert _plane(coarse).power_upper_bound_W > passed + ghost  # (the left-out light may still arrive: it is in the bound)
    budget = coarse.power_budget
    assert budget.conserved and abs(budget.residual_W) <= 1e-9 and budget.tolerance_W == default.power_budget.tolerance_W
    assert dict(budget.total_sinks) == dict(default.power_budget.total_sinks)
    assert coarse.manifest.numerical_tolerances["negligible_power_fraction"] == 1e-2

    # the closure tolerance alone moves no level: every reading is the default's
    loose = run(_window(), numerical=NumericalConfig(energy_conservation_tol=1e-2))
    assert _plane(loose).model_dump() == plane.model_dump()
    assert loose.power_budget.tolerance_W == pytest.approx(1e-2 + 1e-15)


def test_at_the_defaults_the_switch_of_the_negligible_reads_moves_no_number():
    """The reads that mean "negligible light" (``LineFate.level_W``, ``left_out_ledger_W``, the scattered-light notice)
    compared with ``energy_conservation_tol`` before; the level's default is that tolerance's default, so a run at the
    defaults reports the numbers a run whose level is set to the closure tolerance (the value those reads used) reports."""
    defaults = NumericalConfig()
    assert defaults.negligible_power_fraction == defaults.energy_conservation_tol == 1e-9
    default = run(_window())
    as_before = run(_window(), numerical=NumericalConfig(negligible_power_fraction=defaults.energy_conservation_tol))
    numbers = ("total_power_W", "untraced_power_W", "power_upper_bound_W", "power_lower_bound_W", "quantity_state")
    assert [getattr(_plane(default), f) for f in numbers] == [getattr(_plane(as_before), f) for f in numbers]
    assert default.power_budget.model_dump() == as_before.power_budget.model_dump()
    assert [i.code for i in default.issues] == [i.code for i in as_before.issues]


#: The modules that read each field, and no other (a physical use left on the closure tolerance, or a closure check moved
#: to the level, would make the two meanings diverge silently).
LEVEL_READERS = {
    "optics/rays/tracer.py", "validation/untraced.py", "validation/coating_use.py", "validation/power.py",
    "validation/scattered_light.py", "validation/sampled_light.py",
    # (the floor finding's example level and the cavity analysis's negligible injection read the same level)
    "validation/observations.py", "local/cavity.py",
}
CLOSURE_READERS = {
    "optics/rays/tracer.py", "validation/power.py", "scene/lower.py", "optics/interference/two_beam.py",
    "validation/pair_selection.py", "validation/superposition.py", "validation/polarization_mixture.py", "contracts/materials.py",
}


def _readers(name: str) -> set[str]:
    pattern = re.compile(rf"\b(?:cfg|self\.cfg|DEFAULT_NUMERICAL_CONFIG)\.{name}\b")
    return {path.relative_to(SRC).as_posix() for path in SRC.rglob("*.py")
            if "_vendor" not in path.parts and pattern.search(path.read_text(encoding="utf-8"))}


def test_each_field_is_read_by_its_own_modules():
    assert _readers("negligible_power_fraction") == LEVEL_READERS
    assert _readers("energy_conservation_tol") == CLOSURE_READERS
    tracer = (SRC / "optics/rays/tracer.py").read_text(encoding="utf-8")
    # in the tracer the closure tolerance is the interaction-loss residual only
    assert [line.strip() for line in tracer.splitlines() if "cfg.energy_conservation_tol" in line] == [
        "if abs(loss) > self.cfg.energy_conservation_tol * p_before and kind in INTERACTION_LOSS_SINKS:"]
    power = (SRC / "validation/power.py").read_text(encoding="utf-8")
    assert len([line for line in power.splitlines() if "cfg.energy_conservation_tol" in line]) == 2  # (line and total closure)
    assert len([line for line in power.splitlines() if "cfg.negligible_power_fraction" in line]) == 1  # (line_fate)


@pytest.mark.parametrize("capability_id, control", [
    ("beam_trace", "negligible_power_fraction"), ("untraced_light", "negligible_power_fraction"),
    ("plane_bounds", "negligible_power_fraction"), ("surface_exposures", "negligible_power_fraction"),
    ("power_ledger", "energy_conservation_tol"),
])
def test_the_capabilities_name_the_field_they_read(capability_id: str, control: str):
    controls = capability(capability_id).controls
    other = ({"negligible_power_fraction", "energy_conservation_tol"} - {control}).pop()
    assert control in controls and other not in controls
