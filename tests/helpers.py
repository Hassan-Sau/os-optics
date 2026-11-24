"""Shared helpers for engine tests."""

from __future__ import annotations

import math

import numpy as np

from os_optics import EngineConfiguration, NumericalConfig, ValidationRequest, validate
from os_optics.contracts import ProjectSnapshot, ValidationResult

CFG = NumericalConfig()
NM = 1e-6  # nm -> mm


def run(project: ProjectSnapshot, *, layout: bool = False, control: bool = False, seed: int | None = None,
        numerical: NumericalConfig | None = None) -> ValidationResult:
    conf = EngineConfiguration(run_layout=layout, run_control=control, random_seed=seed, numerical=numerical or CFG)
    return validate(ValidationRequest(project=project, configuration=conf))


def assert_dark_truncated(target) -> float:
    """A power target on a plane no traced or followed light of its lines reaches, in a run that follows every untraced
    branch: the plane receives 0 W, a known zero the target is graded on (``NO_TRACED_ARRIVAL``). The light left out as
    negligible is the stated truncation of that value: the target names ``FAINT_LIGHT_BOUNDED`` and its ``value_basis`` entry
    is the model's value with that light as an accounting omission, the most it may bring beside it (returned, W)."""
    assert target.status.value == "PASS" and target.value == 0.0, (target.status, target.value, target.message)
    assert target.issue_codes == ("NO_TRACED_ARRIVAL", "FAINT_LIGHT_BOUNDED"), target.issue_codes
    entry = target.value_basis["value"]
    (omitted,) = entry.omitted
    assert entry.kind.value == "MODEL" and not entry.assumed and not entry.validity, entry
    assert (omitted.part.value, omitted.setting, omitted.code) == ("BELOW_NEGLIGIBLE_LEVEL", "negligible_power_fraction", "FAINT_LIGHT_BOUNDED")
    assert omitted.upper is not None and omitted.upper > 0.0
    return omitted.upper


def assert_dark_bounded(target) -> None:
    """A power target on a plane no traced or followed light of its lines reaches, where the light left out as negligible
    may bring more than the lines' negligible level there (its returns or crossings, ``LineFate.faint_accounted``): the plane
    receives between 0 W and that light's bound -- no value, never a known 0 W (``NO_TRACED_ARRIVAL``,
    ``FAINT_LIGHT_BOUNDED``) -- and a maximum above that bound passes."""
    assert target.status.value == "PASS" and target.value is None, (target.status, target.value, target.message)
    assert target.issue_codes == ("NO_TRACED_ARRIVAL", "FAINT_LIGHT_BOUNDED"), target.issue_codes
    assert "light the trace leaves out as negligible may" in target.message


def arrivals(result: ValidationResult, observation_id: str, source_id: str | None = None):
    obs = next(o for o in result.observations if o.observation_id == observation_id)
    return [a for a in obs.arrivals if source_id is None or a.beam.source_id == source_id]


def one_arrival(result: ValidationResult, observation_id: str, source_id: str | None = None):
    arr = arrivals(result, observation_id, source_id)
    assert len(arr) == 1, f"expected one arrival at {observation_id}, got {len(arr)}"
    return arr[0]


def issue_codes(result: ValidationResult) -> set[str]:
    return {i.code.value for i in result.issues}


def assert_power_not_computed(result: ValidationResult, observation_id: str, *, state: str | None = "UNKNOWN"):
    """A plane reached by a beam whose power is not computed from some surface on (it goes on as geometry): one arrival
    with no power and the bound of what it may carry -- the amount the ledger booked once to light of unknown fate, the
    ledger closing with it --, a path that goes on to where the geometry ends, and a plane that states no power. Returns
    the plane and the arrival."""
    obs = next(o for o in result.observations if o.observation_id == observation_id)
    arrival = one_arrival(result, observation_id)
    assert arrival.beam.power_W is None and arrival.detected_power_W is None
    (line,) = [entry for entry in result.power_budget.lines
               if (entry.source_id, entry.spectral_index) == (arrival.beam.source_id, arrival.beam.spectral_index)]
    assert arrival.power_upper_bound_W == line.sinks["TERMINATED_OUT_OF_MODEL"] > 0.0
    assert line.conserved and result.power_budget.conserved
    path = next(p for p in result.paths if p.path_id == arrival.beam.path_id)
    assert path.final_power_W is None and path.termination in ("ESCAPED", "DETECTED")
    assert not [p.termination for p in result.paths if p.termination in ("TRUNCATED_BEAM_NOT_FOLLOWED", "ANNULAR_BEAM_NOT_FOLLOWED")]
    assert (obs.total_power_W, obs.power_complete) == (None, False)
    assert obs.quantity_state.value == state if state is not None else obs.quantity_state.value not in ("KNOWN", "KNOWN_ZERO")
    (issue,) = [i for i in result.issues if i.numerical_context.get("power_upper_bound_W") is not None and i.status.value == "OUT_OF_MODEL_SCOPE"
                and i.code.value != "GRATING_RESPONSE_UNAVAILABLE"]
    assert issue.numerical_context["power_upper_bound_W"] == arrival.power_upper_bound_W
    return obs, arrival


def rayleigh_range(w0_mm: float, wavelength_nm: float, m2: float = 1.0, n: float = 1.0) -> float:
    return math.pi * n * w0_mm**2 / (m2 * wavelength_nm * NM)


def gaussian_radius_from_q(q: complex, wavelength_nm: float, m2: float = 1.0, n: float = 1.0) -> float:
    """w(q) = sqrt(M^2 lambda |q|^2 / (pi n Im q)) — the definition 1/q = 1/R - i M^2 lambda/(pi n w^2)."""
    return math.sqrt(m2 * wavelength_nm * NM * abs(q) ** 2 / (math.pi * n * q.imag))


def abcd_apply(matrix: np.ndarray, q: complex) -> complex:
    return (matrix[0, 0] * q + matrix[0, 1]) / (matrix[1, 0] * q + matrix[1, 1])


def free(length: float) -> np.ndarray:
    return np.array([[1.0, length], [0.0, 1.0]])


def thin(f: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def stated_splitter(component):
    """A splitter component whose phases are stated constant over wavelength -- those of the default lossless splitter,
    r_s = i sqrt(R), r_p = -i sqrt(R), t = sqrt(T) -- so its group delay is known (none): two beams that take it in
    reflection and in transmission keep a known relative envelope delay (``optics/interference/coherence.py``)."""
    from os_optics.contracts.optics import InterfacePhases

    phases = InterfacePhases(reflection_s_rad=(math.pi / 2.0,), reflection_p_rad=(3.0 * math.pi / 2.0,),
                             transmission_s_rad=(0.0,), transmission_p_rad=(0.0,))
    surfaces = tuple(s.model_copy(update={"interaction": s.interaction.model_copy(update={"phases": phases})})
                     if s.interaction.kind == "beam_splitter" else s for s in component.surfaces)
    return component.model_copy(update={"surfaces": surfaces})
