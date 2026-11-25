"""Why a Project's optics cannot be exported as a linear optical network (structured, never a partial network)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

#: The ``error.code`` of a refused export in the JSON API envelope.
NETWORK_NOT_EXPORTABLE = "NETWORK_NOT_EXPORTABLE"


class NetworkErrorCode(StrEnum):
    """Stable codes of ``NetworkExportError.problems``.

    * ``OPTIONS_INVALID``: an option names an unknown source or plane, or has a bad value.
    * ``SOURCES_MISSING`` / ``OUTPUTS_MISSING``: nothing to export from / to.
    * ``MIXED_WAVELENGTHS``: the sources emit several wavelengths; choose one with ``wavelength_nm``.
    * ``WAVELENGTH_NOT_EMITTED``: no exported source emits the chosen wavelength.
    * ``OUTPUT_NOT_TERMINAL``: a beam crosses an output plane and meets further optics.
    * ``OUTSIDE_ACCEPTANCE``: a beam crosses an output plane outside the plane's acceptance cone (the plane does not count it).
    * ``TRACE_INCOMPLETE``: the trace did not follow every beam (trace budget, interaction limit, out-of-model termination).
    * ``NETWORK_NOT_CLOSED``: the power ledger of a probe trace does not close.
    * ``CAVITY_NOT_SUPPORTED``: light circulates in a resonator (an infinite series of round trips).
    * ``PARTIAL_MODE_OVERLAP``: two beams on one output plane are neither one mode nor separate modes.
    * ``PHASE_UNDEFINED_IDEAL_LENS``: a path crosses an ideal or tabulated thin lens, whose phase is not modelled.
    * ``PHASE_UNDEFINED_FARADAY_ROTATOR``: a path crosses a Faraday rotator, whose model states its rotation and
      transmittance and no optical path through its body.
    * ``PHASE_UNDEFINED_GRATING``: a path crosses a diffraction grating, whose model states its order directions and at most
      its order powers, and no phase of an order.
    * ``PHASE_UNDEFINED_ATTENUATOR``: a path crosses or is reflected by a neutral-density attenuator, a thin plane whose
      model states powers only (the optical path through its substrate and the phase of its reflection are not stated).
    * ``PHASE_UNDEFINED_SAMPLE``: a path crosses a complex-transmission sample, whose phase varies across a beam (no one
      coefficient of a mode).
    * ``MODE_UNDEFINED``: a beam has no valid Gaussian mode (aberration dominated, high NA, twisted astigmatism ...).
    * ``MODE_TRUNCATED``: an aperture cuts more than the Engine's truncation warning fraction (1e-3) of a beam that reaches an
      output, or a detector's active area cuts an output beam (it is no longer one mode).
    * ``POLARIZING_NORMAL_INCIDENCE``: a coating whose model is polarization-dependent at normal incidence (R_s != R_p or
      T_s != T_p there, e.g. a constant PBS coating) is met within 1 degree of normal incidence, where its s axis is set by a
      misalignment or the Engine's internal transverse axis.
    * ``NETWORK_NOT_PASSIVE``: an element's stated amplitudes are not those of a passive two-port (a splitter, or a
      refracting interface with stated phases and a traced reflection, with R + T + 2 sqrt(R T) |cos(phi_r - phi_t)| > 1),
      a singular value of T exceeds 1 by more than the stated tolerance (gain), or a source whose paths lose no light does
      not keep all of it in T.
    * ``OUT_OF_MODEL_SCOPE``: the Engine or a published model's validity reports the case outside its model.
    * ``CLASSICAL_LIMIT_MISMATCH``: the network's |T J|^2 disagrees with the Engine's own coherent superposition.
    * ``ENGINE_INPUT_INVALID``: the Engine cannot lower the input (an invalid scene).
    * ``NUMERICAL_FAILURE``: an arithmetic, value or linear-algebra error while computing the network (non-finite or extreme
      numbers).
    """

    OPTIONS_INVALID = "OPTIONS_INVALID"
    SOURCES_MISSING = "SOURCES_MISSING"
    OUTPUTS_MISSING = "OUTPUTS_MISSING"
    MIXED_WAVELENGTHS = "MIXED_WAVELENGTHS"
    WAVELENGTH_NOT_EMITTED = "WAVELENGTH_NOT_EMITTED"
    OUTPUT_NOT_TERMINAL = "OUTPUT_NOT_TERMINAL"
    OUTSIDE_ACCEPTANCE = "OUTSIDE_ACCEPTANCE"
    TRACE_INCOMPLETE = "TRACE_INCOMPLETE"
    NETWORK_NOT_CLOSED = "NETWORK_NOT_CLOSED"
    CAVITY_NOT_SUPPORTED = "CAVITY_NOT_SUPPORTED"
    PARTIAL_MODE_OVERLAP = "PARTIAL_MODE_OVERLAP"
    PHASE_UNDEFINED_IDEAL_LENS = "PHASE_UNDEFINED_IDEAL_LENS"
    PHASE_UNDEFINED_ATTENUATOR = "PHASE_UNDEFINED_ATTENUATOR"
    PHASE_UNDEFINED_FARADAY_ROTATOR = "PHASE_UNDEFINED_FARADAY_ROTATOR"
    PHASE_UNDEFINED_GRATING = "PHASE_UNDEFINED_GRATING"
    PHASE_UNDEFINED_SAMPLE = "PHASE_UNDEFINED_SAMPLE"
    MODE_UNDEFINED = "MODE_UNDEFINED"
    MODE_TRUNCATED = "MODE_TRUNCATED"
    POLARIZING_NORMAL_INCIDENCE = "POLARIZING_NORMAL_INCIDENCE"
    NETWORK_NOT_PASSIVE = "NETWORK_NOT_PASSIVE"
    OUT_OF_MODEL_SCOPE = "OUT_OF_MODEL_SCOPE"
    CLASSICAL_LIMIT_MISMATCH = "CLASSICAL_LIMIT_MISMATCH"
    ENGINE_INPUT_INVALID = "ENGINE_INPUT_INVALID"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


#: Location prefixes of a problem that concerns one output plane / one source.
OBSERVATION_PREFIX, SOURCE_PREFIX = "observation:", "source:"


@dataclass(frozen=True)
class NetworkProblem:
    """One reason an export is refused. ``observation_id`` / ``source_id``: the one output plane or source the refusal
    concerns (taken from a ``observation:`` / ``source:`` location when not given), so the export can be repeated without
    it through the ``observation_ids`` / ``source_ids`` options."""

    code: NetworkErrorCode
    message: str
    location: str | None = None
    observation_id: str | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        location = self.location or ""
        if self.observation_id is None and location.startswith(OBSERVATION_PREFIX):
            object.__setattr__(self, "observation_id", location[len(OBSERVATION_PREFIX):])
        if self.source_id is None and location.startswith(SOURCE_PREFIX):
            object.__setattr__(self, "source_id", location[len(SOURCE_PREFIX):])

    def as_dict(self) -> dict[str, str]:
        data = {"code": self.code.value, "message": self.message}
        if self.location is not None:
            data["location"] = self.location
        if self.observation_id is not None:
            data["observation_id"] = self.observation_id
        if self.source_id is not None:
            data["source_id"] = self.source_id
        return data


class NetworkExportError(ValueError):
    """The Project's optics are not exportable as a linear optical network; ``problems`` lists every reason found."""

    def __init__(self, problems) -> None:
        self.problems: tuple[NetworkProblem, ...] = tuple(problems)
        super().__init__("; ".join(f"{p.code.value}{f' at {p.location}' if p.location else ''}: {p.message}" for p in self.problems))

    @property
    def codes(self) -> tuple[NetworkErrorCode, ...]:
        return tuple(p.code for p in self.problems)

    def as_dict(self) -> dict[str, object]:
        return {"code": NETWORK_NOT_EXPORTABLE, "problems": [p.as_dict() for p in self.problems]}
