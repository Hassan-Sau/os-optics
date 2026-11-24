"""What two beams' sources say about the coherence of their fields beyond one line's model: pulse trains, continuous spectra
and partially polarized sources sharing a coherence group (``SourceCoherence``), and the coherence model of a pair
(``pair_model``).

The line model. Two fields of one line of a source of declared coherence length ``L_c`` have the degree of coherence of the
Gaussian spectrum of that length (``optics/interference/coherence.py``, ``LineCoherence``): relative to the fields' phase
difference ``phi`` at the line, complex where their paths' group-delay dispersions differ (its phase shifts the fringes) and a
real number otherwise; the combined power is ``P_a + P_b + 2 sqrt(P_a P_b) Re[gamma exp(i phi)]`` as for a continuous spectrum.

A continuous spectrum. A source that states a spectral density is traced as quadrature lines (``optics/interference/spectrum.py``),
each a narrow Gaussian line whose width is the bin width: its coherence length ``(2 sqrt(ln 2) / pi) lambda^2 / w`` is a
property of the discretization, not of the light, and grows with the number of lines. Two beams of one such source interfere
with the coherence of the whole spectrum instead (``SpectrumCoherence``): each line is taken as monochromatic within its
line, carrying the beams' relative spectral phase ``phi(k_line) + D (k - k_line) + B (k - k_line)^2 / 2`` (their group-path
difference ``D`` and group-delay-dispersion difference ``B`` at the line), and the coherent quantities of a line's pair
are its share of the whole spectrum's -- the spectrum's complex degree of coherence ``gamma`` at that relative phase
(``spectrum.pair_coherence``: an integral of the declared density, independent of the number of lines):

* the coherence factor and the fringe visibilities: ``|gamma|`` (the contrast of the source's fringes at this pair's delay,
  with this line's beam balance, overlap and polarization);
* the coherence length and margin: the spectrum's coherence length (``SourceSpectrum.coherence_length_mm``; none where
  ``|gamma|`` stays above ``1/e``);
* the combined power and the peak irradiance: ``P_a + P_b + 2 sqrt(P_a P_b) Re[gamma exp(i phi)]`` and its maximum over the
  fringes, the line's share of the source's (they sum over the lines to the source's when the beams' balance, overlap and
  polarization are the same at every line, exactly for a dispersion-free path difference);
* a coating's group-delay bound ``B`` moves ``gamma`` by at most ``B`` times the spectrum's mean ``|k - k_line|``
  (``spectrum.mean_wavenumber_offset``), as it does for a line.

Two beams of different sources of one coherence group of which one states a continuous spectrum have a mutual coherence the
sources do not state, unless both state the same spectral density: it is not computed (``MIXED_SPECTRA``). A pulse train's
revival is the spectrum's ``|gamma|`` at the delay's distance to the nearest non-zero multiple of the period length.

Partially polarized sources sharing a coherence group. A partially polarized source line of power ``L`` and degree of
polarization ``P`` is a polarized part ``P L`` along ``J`` and an unpolarized part ``(1 - P) L``; it is traced as two
mutually incoherent components, ``(1 + P) L / 2`` along ``J`` (component 0, in the source's coherence group) and
``(1 - P) L / 2`` along ``J_perp`` (component 1, in a group of its own; ``optics/polarization/partial.py``). Another source of
the same coherence group is mutually coherent with the polarized part only: the unpolarized part's field along any direction
has no fixed phase relation to a polarized field (its two components along any pair of orthogonal directions have a random
relative phase), so the time average of its product with another source's field is zero. Of component 0's field
``sqrt((1 + P) L / 2) J`` the part coherent with another source is ``sqrt(P L) J``: two beams of different sources in one
coherence group have the degree of coherence of the line times ``polarized_share`` of each beam -- ``sqrt(2 P / (1 + P))``
for component 0 of a partially polarized source, 0 for its component 1, 1 for a fully polarized source. An unpolarized
source (``P = 0``) is then coherent with no other source, whatever basis its decomposition takes: every result is the same
when it is turned about its own axis. Two beams of one source keep the line's coherence (each component with itself).

Arithmetic: ``+ - x /``, ``sqrt``, the line's libm-free exponential and ``cis`` and the spectrum module's libm-free integrals (identical
bits in every runtime).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .coherence import (coherence_spread, complex_degree_of_coherence, degree_of_coherence, largest_within_length,
                        period_length_mm, revival_coherence, revival_coherence_within, revival_distance)
from .spectrum import mean_wavenumber_offset, pair_coherence

if TYPE_CHECKING:
    from ...contracts.components import SpectralDensity

#: ``pair_model``'s answer for two beams of different sources of one group whose spectra differ, one continuous.
MIXED_SPECTRA = "MIXED_SPECTRA"


@dataclass(frozen=True)
class LineCoherence:
    """The Gaussian-spectrum model of a line of coherence length ``length`` (``optics/interference/coherence.py``).
    ``assumed``: a line of the pair states no coherence and is taken as monochromatic -- ``length`` is infinite where neither
    line states one (the degree of coherence is exactly 1 at every delay), else the other line's stated length. The
    unstated line's own length may be shorter (``open_length``): the degree of coherence at the stated length is then the
    upper end only without a group-delay-dispersion difference -- with one, a shorter length can give a larger ``|gamma|``,
    and the upper end is ``top`` (``coherence.largest_within_length``), while the phase of ``gamma`` moves with the length
    (from its value at the stated length towards ``+-pi/4``), so a coherent power's range takes every phase of the cross term
    up to that end; behind coating group-delay bounds, whose change of the degree of coherence ``2 B / (sqrt(pi) L)`` grows
    without limit as ``L`` shrinks, it is the line's 1."""

    length: float
    assumed: bool = False

    @property
    def open_length(self) -> bool:
        """One line states its coherence length and the other none: the pair's own length is any up to ``length``."""
        return self.assumed and math.isfinite(self.length)

    def gamma(self, delay_mm: float, dispersion_mm2: float) -> complex | float:
        """The complex degree of coherence relative to the fields' phase difference at the line
        (``coherence.complex_degree_of_coherence``): a real number where the paths' group-delay dispersions do not differ."""
        return complex_degree_of_coherence(delay_mm, dispersion_mm2, self.length)

    def magnitude(self, delay_mm: float, dispersion_mm2: float) -> float:
        """``|gamma|`` (``coherence.degree_of_coherence``): what the coherence factor, the visibilities and the peak irradiance
        read."""
        return degree_of_coherence(delay_mm, dispersion_mm2, self.length)

    def top(self, delay_mm: float, dispersion_mm2: float) -> float:
        """The largest ``|gamma|`` at this delay and dispersion over every length the model allows: ``magnitude``, or over
        every length up to ``length`` where it is open (``open_length``)."""
        if self.open_length:
            return largest_within_length(delay_mm, dispersion_mm2, self.length)
        return self.magnitude(delay_mm, dispersion_mm2)

    def spread(self, bound_mm: float) -> float:
        return coherence_spread(bound_mm, self.length)

    def revival(self, delay_mm: float, dispersion_mm2: float, periods_mm: tuple[float, ...]) -> float | None:
        return revival_coherence(delay_mm, dispersion_mm2, self.length, periods_mm, self.open_length)

    def revival_within(self, delay_mm: float, bound_mm: float, dispersion_mm2: float, periods_mm: tuple[float, ...]) -> float | None:
        """The largest revival coherence at any relative delay within ``bound_mm`` of ``delay_mm`` (and over every
        length up to ``length`` where it is open)."""
        return revival_coherence_within(delay_mm, bound_mm, dispersion_mm2, self.length, periods_mm, self.open_length)


@dataclass(frozen=True)
class SpectrumCoherence:
    """The coherence of a continuous spectrum at the line ``reference_nm`` (module docstring); ``length``: the spectrum's
    coherence length (``None``: ``|gamma|`` stays above ``1/e``)."""

    density: SpectralDensity
    reference_nm: float
    length: float | None

    def gamma(self, delay_mm: float, dispersion_mm2: float) -> complex | None:
        return pair_coherence(self.density, self.reference_nm, delay_mm, dispersion_mm2)

    def spread(self, bound_mm: float) -> float:
        return bound_mm * mean_wavenumber_offset(self.density, self.reference_nm)

    def revival(self, delay_mm: float, dispersion_mm2: float, periods_mm: tuple[float, ...]) -> float | None:
        best = 0.0
        for period in periods_mm:
            value = pair_coherence(self.density, self.reference_nm, revival_distance(delay_mm, period), dispersion_mm2)
            if value is None:
                return None
            magnitude = abs_complex(value)
            if magnitude > best:
                best = magnitude
        return best

    def revival_within(self, delay_mm: float, bound_mm: float, dispersion_mm2: float, periods_mm: tuple[float, ...]) -> float | None:
        """A spectrum's ``|gamma|`` need not fall with the delay: over an interval of delays its revival is not bounded by one
        value (``None``), unless the interval is a point."""
        return self.revival(delay_mm, dispersion_mm2, periods_mm) if bound_mm == 0.0 else None


def abs_complex(value: complex | float) -> float:
    """``|value|`` with ``sqrt`` alone (the same bits in every runtime)."""

    if isinstance(value, complex):
        return math.sqrt(value.real * value.real + value.imag * value.imag)
    return abs(value)


def in_phase(gamma: complex | float, cos_phi: float, sin_phi: float) -> float:
    """``Re[gamma exp(i phi)]``: ``gamma cos(phi)`` for a real degree of coherence."""

    if isinstance(gamma, complex):
        return gamma.real * cos_phi - gamma.imag * sin_phi
    return gamma * cos_phi


@dataclass(frozen=True)
class SourceCoherence:
    """Per source id what its coherence reads (module docstring): ``periods`` the pulse trains' period lengths ``c / f_rep``
    (mm), ``spectra`` the continuous spectra, ``lengths`` their coherence lengths (``SourceSpectrum.coherence_length_mm``;
    ``None``: not reached), ``polarization`` the degree of polarization of partially polarized sources (below 1)."""

    periods: Mapping[str, float] = field(default_factory=dict)
    spectra: Mapping[str, Any] = field(default_factory=dict)
    lengths: Mapping[str, float | None] = field(default_factory=dict)
    polarization: Mapping[str, float] = field(default_factory=dict)

    def trains(self, *sources: str) -> tuple[float, ...]:
        """The period lengths of the pulse trains among ``sources`` (each once, in id order)."""

        return tuple(self.periods[x] for x in sorted(set(sources)) if x in self.periods)

    def polarized_share(self, source_a: str, component_a: int | None, source_b: str, component_b: int | None) -> float:
        """The factor of the degree of coherence of two beams for the partially polarized sources among them (module
        docstring): 1 for two beams of one source."""

        if source_a == source_b:
            return 1.0
        return self._share(source_a, component_a) * self._share(source_b, component_b)

    def _share(self, source: str, component: int | None) -> float:
        p = self.polarization.get(source)
        if p is None or component is None:
            return 1.0
        if component != 0:
            return 0.0
        return math.sqrt(2.0 * p / (1.0 + p))

    def spectrum(self, source_a: str, source_b: str) -> Any:
        """The continuous spectrum two beams of these sources interfere with: ``None`` (neither states one: the line
        model), the density, or ``MIXED_SPECTRA`` (different sources whose spectra differ, one continuous)."""

        a, b = self.spectra.get(source_a), self.spectra.get(source_b)
        if a is None and b is None:
            return None
        if source_a == source_b or (a is not None and b is not None and a == b):
            return a
        return MIXED_SPECTRA


#: No source states anything beyond its lines.
NO_SOURCE_COHERENCE = SourceCoherence()


def pair_model(sources: SourceCoherence | None, source_a: str, source_b: str, wavelength_nm: float,
               lengths: Sequence[float]) -> LineCoherence | SpectrumCoherence | str | None:
    """The coherence model of two beams of one coherence group at the line ``wavelength_nm``: the spectrum's
    (``SpectrumCoherence``) when their source states a continuous spectrum, ``MIXED_SPECTRA`` (module docstring), else the line
    model of the smaller of the lines' stated coherence lengths ``lengths`` (one per line that states one). Where only one
    line states a length, the other is taken as monochromatic and the model is that length, ``assumed`` (the unstated line's
    own length may be shorter: ``LineCoherence.top`` is the upper end). Where neither line states one, the line is taken as monochromatic
    (``LineCoherence(inf, assumed=True)``), except for a pulse train among the sources: its pulses' overlap needs a stated
    coherence, so ``None`` (not declared)."""

    spectrum = None if sources is None else sources.spectrum(source_a, source_b)
    if spectrum == MIXED_SPECTRA:
        return MIXED_SPECTRA
    if spectrum is not None:
        return SpectrumCoherence(spectrum, wavelength_nm, sources.lengths.get(source_a) if sources is not None else None)
    if lengths:
        return LineCoherence(min(lengths), assumed=len(lengths) < 2)
    if sources is not None and sources.trains(source_a, source_b):
        return None
    return LineCoherence(math.inf, assumed=True)


def source_coherence(components: Sequence[Any], spectra: Sequence[Any] = ()) -> SourceCoherence:
    """The ``SourceCoherence`` of an Engine Project's components (their ``source``); ``spectra``: the run's ``source_spectra``
    (their coherence lengths)."""

    periods: dict[str, float] = {}
    densities: dict[str, Any] = {}
    polarization: dict[str, float] = {}
    for component in components:
        source = getattr(component, "source", None)
        if source is None:
            continue
        if source.pulse is not None:
            periods[source.source_id] = period_length_mm(source.pulse.repetition_rate_hz)
        if source.spectral_density is not None:
            densities[source.source_id] = source.spectral_density
        p = source.degree_of_polarization
        if p is not None and p < 1.0:
            polarization[source.source_id] = float(p)
    lengths = {s.source_id: s.coherence_length_mm for s in spectra if s.source_id in densities}
    return SourceCoherence(periods, densities, lengths, polarization)


def request_source_coherence(request: Mapping[str, Any] | None) -> SourceCoherence:
    """The ``SourceCoherence`` of a run's stored Engine request (a JSON object; what the results layer reads): pulse trains,
    continuous spectra and partially polarized sources (no coherence lengths: the results layer reads none)."""

    from ...contracts.components import SpectralDensity

    periods: dict[str, float] = {}
    densities: dict[str, Any] = {}
    polarization: dict[str, float] = {}
    project = (request or {}).get("project") if isinstance(request, Mapping) else None
    for component in (project or {}).get("components", ()) if isinstance(project, Mapping) else ():
        source = component.get("source") if isinstance(component, Mapping) else None
        if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str):
            continue
        source_id = source["source_id"]
        pulse = source.get("pulse")
        rate = pulse.get("repetition_rate_hz") if isinstance(pulse, Mapping) else None
        if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0.0:
            periods[source_id] = period_length_mm(float(rate))
        if isinstance(source.get("spectral_density"), Mapping):
            densities[source_id] = SpectralDensity.model_validate(source["spectral_density"])
        p = source.get("degree_of_polarization")
        if isinstance(p, (int, float)) and not isinstance(p, bool) and p < 1.0:
            polarization[source_id] = float(p)
    return SourceCoherence(periods, densities, {}, polarization)


__all__ = ["MIXED_SPECTRA", "NO_SOURCE_COHERENCE", "LineCoherence", "SourceCoherence", "SpectrumCoherence", "abs_complex", "in_phase",
           "pair_model", "request_source_coherence", "source_coherence"]
