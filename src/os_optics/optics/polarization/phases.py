"""Stated interface phases (``contracts.optics.InterfacePhases``) evaluated at a vacuum wavelength; no extrapolation.

A tabulated phase is interpolated linearly between neighbouring wavelengths (``np.interp``: plain ``+ - x /``, the same
bits in every runtime). At a wavelength outside the table (beyond the coating range slack ``tabulated_range_tol_nm``) the
interface states no phase: ``PhaseEvaluator.at`` gives phases that are all ``None`` (the interface's own phase applies, as
without stated phases) and an ``OUT_OF_MODEL_SCOPE`` ``COATING_OUT_OF_RANGE`` issue.

The amplitude helpers below (``coated_interface``, ``mirror_reflection``, ``mirror_transmission``,
``splitter_amplitudes``) form an interface's complex s / p amplitudes from its coating's power coefficients, its own phase
convention and the stated phases. The tracer and the polarization router (``routing.py``) both use them: one definition
of the physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...config import NumericalConfig
from ...contracts.optics import InterfacePhases
from ...errors import ErrorCode, Status, ValidationIssue, make_issue
from ...math.exact_trig import cis


def _phase_factor(z):
    """``exp(i arg z)`` as ``z / |z|`` (1 where ``z`` is 0), from ``+ - x / sqrt`` only (no libm / NumPy complex exp)."""

    z = np.asarray(z, dtype=np.complex128)
    magnitude = np.sqrt(z.real * z.real + z.imag * z.imag)
    safe = np.where(magnitude > 0.0, magnitude, 1.0)
    return np.where(magnitude > 0.0, z / safe, 1.0 + 0.0j)


@dataclass(frozen=True)
class PhaseValues:
    """The stated phases (rad) at one wavelength; ``None`` where the interface states none (its own phase applies)."""

    r_s: float | None = None
    r_p: float | None = None
    t_s: float | None = None
    t_p: float | None = None


class PhaseEvaluator:
    def __init__(self, model: InterfacePhases, cfg: NumericalConfig) -> None:
        self._model = model
        self._cfg = cfg
        self._nodes = None if model.wavelength_nm is None else np.array(model.wavelength_nm, dtype=np.float64)
        self._columns = {name: None if getattr(model, name) is None else np.array(getattr(model, name), dtype=np.float64)
                         for name in ("reflection_s_rad", "reflection_p_rad", "transmission_s_rad", "transmission_p_rad")}

    @property
    def model(self) -> InterfacePhases:
        return self._model

    def covers(self, wavelength_nm: float) -> bool:
        """Whether the phases are stated at ``wavelength_nm``: constant phases everywhere, a table within its wavelengths."""

        nodes = self._nodes
        if nodes is None:
            return True
        tol = self._cfg.tabulated_range_tol_nm
        return float(nodes[0]) - tol <= wavelength_nm <= float(nodes[-1]) + tol

    def at(self, wavelength_nm: float, *, location: str) -> tuple[PhaseValues, ValidationIssue | None]:
        """The phases at ``wavelength_nm``. Outside a tabulated phase's wavelengths: no stated phase (every value ``None``)
        and the issue that says so."""

        nodes = self._nodes
        if nodes is None:
            values = {name: None if column is None else float(column[0]) for name, column in self._columns.items()}
        else:
            lo, hi = float(nodes[0]), float(nodes[-1])
            if not self.covers(wavelength_nm):
                return PhaseValues(), make_issue(
                    ErrorCode.COATING_OUT_OF_RANGE, Status.OUT_OF_MODEL_SCOPE,
                    "the coating's stated phases do not cover this wavelength: the interface's own phases are used, and the "
                    "phase of this pass is not stated",
                    location=location, wavelength_nm=wavelength_nm, valid_wavelength_nm=[lo, hi],
                    suggested_action="restrict the operating wavelength or state the phases over it")
            wl = min(max(wavelength_nm, lo), hi)
            values = {name: None if column is None else float(np.interp(wl, nodes, column)) for name, column in self._columns.items()}
        return PhaseValues(r_s=values["reflection_s_rad"], r_p=values["reflection_p_rad"], t_s=values["transmission_s_rad"],
                           t_p=values["transmission_p_rad"]), None


def coated_interface(rs, rp, ts, tp, bare_t_s, bare_t_p, bare_r_s, bare_r_p, phases: PhaseValues | None):
    """A coated refracting interface's amplitudes ``(t_s, t_p, r_s, r_p)``: the coating's power coefficients as magnitudes,
    with the stated phases where the coating states them and the bare Fresnel amplitudes' phases elsewhere."""

    t_s = np.sqrt(ts) * (_phase_factor(bare_t_s) if phases is None or phases.t_s is None else cis(phases.t_s))
    t_p = np.sqrt(tp) * (_phase_factor(bare_t_p) if phases is None or phases.t_p is None else cis(phases.t_p))
    r_s = np.sqrt(rs) * (_phase_factor(bare_r_s) if phases is None or phases.r_s is None else cis(phases.r_s))
    r_p = np.sqrt(rp) * (_phase_factor(bare_r_p) if phases is None or phases.r_p is None else cis(phases.r_p))
    return t_s, t_p, r_s, r_p


def mirror_reflection(amp_s, amp_p, phase_s: float, phase_p: float, phases: PhaseValues | None):
    """A mirror's reflection amplitudes ``(r_s, r_p)``: magnitudes ``amp_s`` / ``amp_p`` with the stated reflection phases,
    else the mirror's own ``phase_s`` / ``phase_p``."""

    r_s = amp_s * (cis(phase_s) if phases is None or phases.r_s is None else cis(phases.r_s))
    r_p = amp_p * (cis(phase_p) if phases is None or phases.r_p is None else cis(phases.r_p))
    return r_s, r_p


def mirror_transmission(amp_s, amp_p, phases: PhaseValues | None):
    """A coated mirror's transmission amplitudes ``(t_s, t_p)`` (the light its coating lets through): magnitudes with the
    stated transmission phases, else no phase."""

    if phases is not None and phases.t_s is not None:
        return amp_s * cis(phases.t_s), amp_p * cis(phases.t_p)
    return amp_s, amp_p


def splitter_amplitudes(rs, rp, ts, tp, reflection_phase: float, transmission_phase: float, phases: PhaseValues | None):
    """A beam splitter's amplitudes ``(t_s, t_p, r_s, r_p)`` from its coating's power coefficients: the stated phases, else
    ``t = sqrt(T) exp(i transmission_phase)`` and the physical reflection ``r_s = sqrt(R_s) exp(i reflection_phase)``,
    ``r_p = -sqrt(R_p) exp(i reflection_phase)`` in the ``(s, d_out x s)`` triad."""

    if phases is None or phases.t_s is None:
        t_s = np.sqrt(ts) * cis(transmission_phase)
        t_p = np.sqrt(tp) * cis(transmission_phase)
    else:
        t_s = np.sqrt(ts) * cis(phases.t_s)
        t_p = np.sqrt(tp) * cis(phases.t_p)
    if phases is None or phases.r_s is None:
        r_s = np.sqrt(rs) * cis(reflection_phase)
        r_p = -np.sqrt(rp) * cis(reflection_phase)
    else:
        r_s = np.sqrt(rs) * cis(phases.r_s)
        r_p = np.sqrt(rp) * cis(phases.r_p)
    return t_s, t_p, r_s, r_p
