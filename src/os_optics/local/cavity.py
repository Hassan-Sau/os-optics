"""Cavity analysis of placed mirrors: stability, the fundamental Gaussian eigenmode, Gouy phase, free spectral range,
transverse-mode spacing, finesse and linewidth, and the mode matching of a stated beam (an ``AnalysisResultV1``, analysis
``CAVITY``).

``cavity_analysis(project, request)`` takes a linear (standing-wave) cavity of a Project: ``components`` lists its elements
once each, from one end mirror to the other, in the order the light meets them on its way from the first to the last. The
elements are placed as ``os_optics.local.paraxial`` places them (the same prescriptions, matrices, unfolded axis and refusals):
the end mirrors are reflectors at normal incidence (a curved mirror is; a plane end mirror must be normal to the axis), plane
fold mirrors at any tilt and transmissive elements may lie between them. The light starts at the first mirror, met on its
reflecting side. ``request``:

* ``components``: the instance ids, first end mirror ... last end mirror;
* ``wavelength_nm``: the vacuum wavelength;
* ``axis`` (optional): ``"X"`` / ``"Y"``, the section of a cavity with cylindrical surfaces (as in the paraxial analysis);
* ``reflectances`` (optional): ``{instance_id: R}``, a mirror's power reflectance at this wavelength and angle of incidence,
  ``0 <= R < 1``, stated instead of its coating's;
* ``input_beam`` (optional): a fundamental (TEM00) Gaussian beam at the reference plane, travelling into the cavity:
  ``{"waist_radius_mm", "waist_radius_y_mm"?, "waist_after_reference_mm"}`` (the waist's signed distance from the reference
  plane along the light, positive inside the cavity; its x / y are the section axes). Its mode matching to the eigenmode is
  reported;
* ``polarization`` (optional; needed with a crystal in the cavity): ``"ORDINARY"`` or ``"EXTRAORDINARY"``, the eigen-polarization
  of the cavity's crystals the analysis is for (below); or ``"S"`` / ``"P"`` for a cavity whose tilted mirrors share one plane of
  incidence (Reflectances, below);
* ``transmittances`` (optional): ``{end mirror: T}``, stated instead of the coating's (with its reflectance stated as well);
* ``round_trip_loss`` (optional): the power fraction lost per round trip besides the mirrors' reflectances (intracavity elements);
* ``injection`` (optional): light fed into the cavity through the first listed mirror (below): ``{"power_w" | "source_id", "beam":
  "EIGENMODE" | "INPUT_BEAM" | "SOURCE", "detuning_Hz"? | "length_offset_nm"?, "mode_order_max"? (default 10), "scan"?: {"from_Hz",
  "to_Hz", "points"}, "source_linewidth_fraction_max"? (default 0.01: the largest share of the cavity's linewidth a source's linewidth
  may have and count as one frequency)}``;
* ``targets`` (optional): ``[{"target_id", "quantity_id", "target_value"?, "tolerance_abs"?, "tolerance_rel"?, "minimum"?, "maximum"?}]``,
  assessed with the Project's targets of a cavity metric (``validation.scope.CAVITY_METRICS``) in ``arrays.targets``;
* ``aperture_over_mode_min`` (optional, default 3): the least clear aperture radius over the mode radius (and over an injected
  beam's radius at the input mirror) at which an aperture counts as not reaching the beam;
* ``coaxial_tolerance`` (optional): ``{"lateral_mm"?, "tilt_rad"?}``, how far an element may lie off the axis and how far its axis
  (a plane end mirror's normal, an injected source's direction, a transmissive element's axis) may be tilted and still be
  analysed on the nominal axis, as in the paraxial analysis (``paraxial.CoaxialTolerance``). A curved mirror is analysed only
  at normal incidence, whatever ``tilt_rad`` says (``PARAXIAL_CURVED_REFLECTOR_TILTED``). ``arrays.elements`` states each element's measured ``decentre_mm`` and ``tilt_rad``;
  a stated tolerance is named by the eigenmode quantities' ``value_basis``.

Limits. The paraxial limits of the eigenmode are the Project's numerical settings (``gaussian_scope_divergence_rad``,
``gaussian_max_w_over_radius``), the level below which light not resolved leaves a power ``KNOWN`` its ``negligible_power_fraction``;
the request holds ``aperture_over_mode_min``, ``coaxial_tolerance`` and ``injection.source_linewidth_fraction_max``. ``settings.limits`` records the values in
force under the names a validity row or a ``value_basis`` uses as ``setting``.

Crystals. A uniaxial crystal (a Project-declared ``birefringence``) between the mirrors has two indices: light polarized
across its optic axis meets ``n_o``, light polarized along it ``n_e``, so the cavity has two families of modes with different
round-trip group paths, free spectral ranges and eigenmodes. Their linear polarizations are eigenmodes of the round trip
where every crystal's optic axis lies in its faces (perpendicular to the axis there, within ``CRYSTAL_AXIS_TOLERANCE``) and the
crystals' axes are parallel in the section frame; ``polarization`` then chooses one family. The ordinary wave is isotropic:
its plate of thickness ``t`` has the reduced length ``t / n_o`` and group path ``n_g,o t``. The extraordinary wave of a crystal
whose axis lies in its face diffracts anisotropically: its wave-vector surface ``k_c^2 / n_o^2 + k_perp^2 / n_e^2 = k0^2`` (``k_c``
along the optic axis) gives ``k_z ~ n_e k0 - (n_e / n_o^2) k_c^2 / (2 k0) - k_t^2 / (2 n_e k0)``, the reduced length
``t n_e / n_o^2`` in the section along the optic axis and ``t / n_e`` across it, and the group path ``n_g,e t``; its eigenmode is
computed where the optic axis lies along a section axis and every element holding a crystal is free of power (a pure
translation in both sections), else its eigenmode quantities are outside the model (``CRYSTAL_AXIS_OBLIQUE_TO_SECTIONS``,
``CRYSTAL_IN_POWERED_ELEMENT``) while its group path and free spectral range stand. Without ``polarization``
(``POLARIZATION_NOT_STATED``), or where the crystals have no common linear eigen-polarization (``CRYSTAL_EIGENMODES_UNDEFINED``:
an optic axis out of a face, crystals turned against each other), every quantity that depends on a crystal's index -- the
stability, the eigenmode, the round-trip group path, the free spectral range, the transverse-mode spacing, the linewidth and
the mode matching -- is outside the model; the cavity length, the reflectances, the round-trip power fraction and the finesse
stand.

Round trip. The reference plane is the first mirror's vertex, the light leaving it towards the next element. With ``P`` the
forward pass from there to the last mirror's vertex (the free gaps and the elements between), ``M1`` / ``M2`` the end mirrors'
matrices, and ``P_r`` the pass back (``((D, B), (C, A))`` of ``P``: the same elements met in reverse order), the round trip is
``M1 P_r M2 P``. Its half trace ``m = (A + D) / 2`` decides stability: ``|m| < 1`` stable, ``|m| > 1`` unstable, ``|m|`` within
``MARGINAL_HALF_TRACE`` of 1 marginal (plane-plane, concentric, hemispherical: no Gaussian eigenmode). ``g1 g2 = (1 + m) / 2``
(``g_i = 1 - L / R_i`` of two bare mirrors). The eigenmode ``q`` reproduces itself: ``1 / q = (D - A) / (2 B) - i sqrt(1 - m^2) / |B|``,
the round-trip Gouy phase ``theta`` has ``cos theta = m`` and ``sin theta`` of the sign of ``B`` (in ``[0, 2 pi)``). A symmetric
cavity (equal end mirrors, a forward pass equal to its reverse) takes its eigenmode from the half round trip ``H = M2 P``
(``theta = 2 theta_H``): the symmetric confocal cavity's round trip is ``-I``, which every beam reproduces, and its half round
trip gives its one mode, ``w0^2 = lambda L / (2 pi)``. Every number is on the unfolded axis.

Resonances. ``FSR = c / L_rt``, ``L_rt`` the round-trip group path: twice the free gaps plus, through a transmissive element,
``n_g t`` of each medium (``os_optics.optics.materials.group``). An element whose model states no thickness (an ideal or tabulated thin
lens, a polarization element, an attenuator, a Faraday rotator: ``ZERO_THICKNESS_KINDS``) adds none, as its paraxial matrix takes it
at one plane: the group path, the free spectral range and what reads them (the transverse-mode spacing, the linewidth, a detuning
from a length offset) carry that number in state ``UNKNOWN`` with a ``value_basis`` of kind ``ESTIMATE`` that names the elements
(``THIN_ELEMENT_PATH``); a stated ``detuning_Hz`` other than 0 is then a phase the estimate gives, so the powers keep their bounds
over every detuning with the value at that phase beside them.

Reflectances. A mirror's reflectance is the request's (``reflectances``) or its coating's at the wavelength and its angle of incidence.
Outside the wavelength bands or the angle range the coating's record states, the value is the nearest published one, held -- the
number a run traces with --, in state ``UNKNOWN`` with a ``value_basis`` of kind ``ESTIMATE`` (``HELD_AT_NEAREST_STATED_POINT``; its
``validity`` rows state the wavelength and the angle against the record's ranges); the round-trip power fraction, the finesse and the
linewidth inherit it, and the injection waits for a stated coefficient. A fold mirror at an angle whose coating reflects s- and
p-polarized light differently has ``reflectance_s.<id>`` and ``reflectance_p.<id>``; ``polarization`` ``"S"`` (perpendicular to the plane
of incidence) or ``"P"`` (in it) chooses one where they are eigen-polarizations of the round trip: the tilted mirrors share one plane
of incidence and the cavity holds no crystal, polarization element or Faraday rotator. A source then feeds that polarization its
share of the power (``polarization_family_power_fraction``). A reflectance stated without its transmittance leaves the transmittance
``BOUNDED`` in ``[0, 1 - R]``. The transverse-mode spacing ``FSR theta / (2 pi)`` (from ``TEM_mnq`` to ``m + n + 1``,
modulo the FSR). The round-trip power fraction ``rho`` is the product of the reflectances met in one round trip (an end mirror
once, a fold mirror twice); with ``g = sqrt(rho)``, finesse ``pi sqrt(g) / (1 - g)`` (equal mirrors: ``pi sqrt(R) / (1 - R)``)
and the Airy linewidth (full width at half maximum) ``(2 FSR / pi) asin((1 - g) / (2 sqrt(g)))``, defined while that argument is
at most 1. Losses other than the mirrors' (intracavity elements, clipping, an unstable or marginal cavity) only lower ``rho``:
the finesse is then a proven upper bound and the linewidth a proven lower bound (``BOUNDED``); the value beside the interval is
the mirrors' number, and its ``value_basis`` names the side it bounds and the losses taken as zero (``NO_UNSTATED_LOSS``, at the
elements or with the eigenmode's reason).

Values and bounds. A ``BOUNDED`` quantity carries the value computed on its stated basis beside its proven interval, with a
``value_basis`` (``contracts/basis.py``) that says what the number is: an upper or lower bound, or an estimate. Only a ``KNOWN``
value or a proven interval decides a target (``arrays.targets``); the row carries the value and its basis either way.

Mode matching. The power fraction of a TEM00 beam coupled into the fundamental mode, the product over the two section axes of
``2 sqrt(Im q_a Im q_b) / |q_a - conj(q_b)|`` at the reference plane.

Injection. Light of one frequency fed through the first listed mirror (its transmittance the coating's or stated) builds up in each
transverse eigenmode by the Airy function of its round-trip phase (``os_optics.local.cavity_build_up``): the injected, circulating,
transmitted (through the last mirror), reflected and lost powers, the build-up, transmission and reflection, on resonance and at the
detuning, and a power budget whose channels add up to the injected power. ``EIGENMODE`` injects ``power_w`` in a beam matched to the
fundamental mode; ``INPUT_BEAM`` the request's ``input_beam``; ``SOURCE`` the beam of the Project's source ``source_id``. A beam that is
not the eigenmode is split into the cavity's Hermite-Gauss modes up to the resolved order (the stated ``mode_order_max``, and no higher
than the apertures and the paraxial limits admit), each resonating at its Gouy-shifted phase; the light in the modes not resolved is
bounded for any phase and loss (a beam cut by the input mirror's aperture: ``INPUT_BEAM_CLIPPED``, below). With the detuning and the losses stated, each power's value is the resolved modes' sum: ``KNOWN``
where the light not resolved is at most ``negligible_power_fraction`` of the injected power (the Project's numerical setting; the
proven interval stays beside the value and ``convergence.resolution`` is the most that light can add), else the lower bound of a
``BOUNDED`` quantity (``value_basis``: ``LOWER_BOUND``, omitted ``UNRESOLVED_MODES`` with the most it can add). Where the losses of
an element between the mirrors are not stated (no ``round_trip_loss``), the values are those without that loss: the build-up, the
transmission and the matching reflectances on resonance are upper bounds, the reflection on resonance a lower bound for an
under-coupled cavity and an estimate for an over-coupled one, the powers at the detuning estimates (``NO_UNSTATED_LOSS``), each
inside an interval that holds for any such loss. The coupling regime on resonance (under-, over- or critically
coupled) and the input reflectances that cancel the reflection are in ``arrays.injection.coupling``.

Detuning. ``detuning_Hz`` is the source's frequency less that of the nearest TEM00 resonance; ``length_offset_nm`` is the cavity's length
less the length of that resonance (a round-trip phase of ``4 pi n_phase offset / lambda_vac``, where ``n_phase`` is the
surrounding medium's phase index). The nominal positions of a Project do not fix the detuning: the round-trip phase turns by
``2 pi`` per half wavelength *in the surrounding medium* of length, far below the precision of a placement. Without either,
the detuning is not known (``DETUNING_NOT_STATED``): the powers are ``BOUNDED`` over every detuning (the sum over the resolved modes,
each at its Gouy-shifted phase, by branch and bound over the detuning: ``cavity_build_up.swept_extremes``; plus the bounds of the light
not resolved) and carry no value (``missing``: ``injection.detuning_Hz``), and the resolved modes' channels with TEM00 on resonance
are the quantities ``at_tem00_resonance.<channel>`` (lower bounds while light is not resolved; their modes in
``arrays.injection.at_tem00_resonance``, ``not_resolved_W`` the power they leave out).

A source of the Project (``source_id``) feeds the cavity from the axis behind the first mirror. Its light is traced from the source to
the first mirror (``os_optics.validation.pipeline``): the power arriving there is the injected power, and the source's own Gaussian beam,
carried through the first mirror's substrate (the paraxial matrix of its spherical surfaces) to the reference plane, is the injected
beam: its mode matching is reported and used. How the light reaches the mirror is a condition of the injection quantities alone: the
cavity's own quantities (stability, eigenmode, resonances, finesse, reflectances, the resonant quantities and the coupling) are those of
the same request without the source. Where a source is not a single-line Gaussian beam, does not lie on the axis behind the first
mirror, meets anything else on its way (a stop, a lens, a window), or its beam is not carried through the mirror's substrate, the beam
at the mirror is not the source's (``CAVITY_SOURCE_UNSUPPORTED``): the mode matching and the mode fractions are ``OUT_OF_MODEL_SCOPE``,
the injected power is the traced arriving power (a stop or a lens the chief ray passes changes the beam; the trace carries its power)
and the seven powers are ``BOUNDED`` from zero by it as light not resolved. Where the light does not reach the mirror, or part of the
beam meets an element its chief ray misses or another of the elements' model conditions fails on its path (``local.validity``, from
the trace's pupil rays: the chief ray's power counts the whole beam there), the arriving power is not known either: the injected
power and the seven powers are ``OUT_OF_MODEL_SCOPE``. ``arrays.injection.traced_arrival`` records the emitted and arriving power, the
path, the elements met and the model conditions found; ``arrays.injection.source_problem`` the code and the message. Refused: a source
the Project does not hold, a ``wavelength_nm`` that is not its line.
An ``input_beam`` stated with a source must be that beam. With a crystal in the cavity the source's polarization is split onto the
crystals' eigen-polarizations: the extraordinary family receives the power fraction polarized along the optic axis, the ordinary family
the rest (``polarization_family_power_fraction``); the other family's power is ``other_polarization_W`` in the power budget. A stated
``power_w`` is the power in the family analysed. The trace follows single passes and does not compute the build-up: a beam meeting the
back of a transmitting mirror ends there (``OUT_OF_MODEL_SCOPE``, cause ``MIRROR_TRANSMISSION_NOT_TRACED``), and a run reports a target with
a cavity metric ``OUT_OF_MODEL_SCOPE``. A Project target with a cavity metric whose ``unit`` is not the metric's
(``validation.scope.CAVITY_METRIC_UNITS``) is not assessed (``INDETERMINATE``, ``TARGET_UNIT_MISMATCH``).

Three groups of injection quantities, each with its own inputs. The resonant quantities (``resonant_*``, the two matching
reflectances) and ``arrays.injection.coupling`` read the mirrors' coefficients alone: no condition of the source or the beam
touches them, and a coefficient that is not known withholds only the quantities that read it (``missing`` names the request field
that states it). ``detuning_Hz`` is the stated detuning. The seven powers read the source and the beam as well; where one of their
conditions fails they are ``BOUNDED`` by the bound that still holds, with the condition as their ``reason`` and its numbers in the
injection model's ``validity``:

* a source that is not of one frequency -- pulsed, a linewidth above ``injection.source_linewidth_fraction_max`` of the cavity's, or a
  stated linewidth beside a cavity linewidth that is not known (``SOURCE_NOT_SINGLE_FREQUENCY``): the cavity is linear and every
  spectral component sits at some detuning, so the time-averaged powers lie in the bounds over every detuning (the coefficients and
  the Gouy phase at the stated wavelength);
* a transverse content the model does not decompose -- a source with ``M^2 > 1`` (``SOURCE_NOT_FUNDAMENTAL_MODE``), an injected beam
  whose radius at the input mirror is above ``1 / APERTURE_OVER_MODE_MIN`` of its clear radius (``INPUT_BEAM_CLIPPED``): all the light is
  light not resolved, bounded from zero (how much of it enters is not known) to the bound for any phase and loss.

A validity condition the eigenmode fails. Where the round-trip matrix has an eigenmode whose far-field angle or radius over a
mirror's radius of curvature passes the paraxial limits (``MODE_OUTSIDE_PARAXIAL_LIMITS``), or which a clear aperture reaches
(``MODE_CLIPPED_BY_APERTURE``), the eigenmode quantities -- Gouy phase, mode radii, waists, transverse-mode spacing, mode matching --
carry the matrix's values in state ``OUT_OF_MODEL_SCOPE`` with a ``value_basis`` of kind ``ESTIMATE`` whose ``validity`` rows state
the failed condition (observed value, limit, the setting that holds it, the element); the mode radius at an element also states
that element's own clear aperture over its mode radius. Such a value decides no target. A clipped mode adds a loss no input states:
the finesse, the linewidth and the resonant quantities are bounds for any loss; no mode of the injected light is resolved, so the
seven powers are bounded from zero. A limit the caller set and the case meets is named the same way beside a ``KNOWN`` value
(kind ``MODEL``).

Out of the model: the eigenmode of an unstable or marginal cavity (``OUT_OF_MODEL_SCOPE`` without a value; a plane-plane Fabry-Perot
etalon is marginal) and with it the injection quantities, and a tilted or offset input beam.
Refused: a cavity shorter than ``MIN_LENGTH_WAVELENGTHS`` wavelengths (``CAVITY_TOO_SHORT``) and a detuning at or above the light's
frequency.

Execution. The analysis has no budget and no early stop: every document it returns has ``execution`` ``COMPLETED``. A quantity that is
not known or outside the model says so in its own state, reason and message.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

import numpy as np

from ..analysis import completed, document, settings_hash
from ..config import DEFAULT_NUMERICAL_CONFIG, NumericalConfig
from ..contracts.analysis import (
    AnalysisResultV1,
    Convergence,
    ConvergenceMethod,
    Interval,
    IntervalKind,
    ModelBasis,
    Quantity,
    QuantityOrigin,
    QuantityScope,
    ValidityCheck,
)
from ..contracts.basis import Assumed, Omitted, ValueBasis, merge_bases
from ..errors.status import AssumedInput, OmittedPart, QuantityState, RequirementAssessment, ValueKind
from ..math.exact_trig import acos_deg, asin, atan2
from ..optics.materials.coatings import CoatingEvaluator
from ..contracts.materials import GRADIENT_MEDIA, UniaxialMaterial
from ..optics.materials.group import group_index
from ..optics.materials.index import build_evaluator, law_material
from ..optics.rays.tracer import BLOCKED_BACK, OBSERVATION_PREFIX
from ..validation.scope import CAVITY_METRIC_UNITS, CAVITY_METRICS
from .catalog import load_project
from .cavity_build_up import (CHANNELS, MAX_MODE_ORDER, MAX_SCAN_POINTS, TARGET_KEYS, Coupler, judge, lg_fractions, mode_fractions, steady_state,
                              sum_left, swept_extremes, unresolved_bounds, unresolved_fraction)
from .compile import AIR, VACUUM, CompiledElement, Matrix, _media, _multiply, reflecting_surface, surrounding_index
from .lowering import source_polarization
from .paraxial import (DEFAULT_TILT_RAD, LATERAL_TOLERANCE_MM, NM_TO_MM, REQUEST_INVALID, SECTIONS, ParaxialError, _angle, _axis_path, _dot, _in_frame,
                       _lateral, _lay_out, _local_axes, _number, _Placed, _refuse, coaxial_tolerance)

CAVITY_ANALYSIS = "CAVITY"
CAVITY_ANALYSIS_VERSION = "1"
class CavityErrorCode(StrEnum):
    """The codes of ``CavityError`` problems beside the paraxial placement's (``ParaxialErrorCode``), and of the refused
    analysis's envelope (stable values; a problem holds the value as a plain string)."""

    #: The envelope code of a refused cavity analysis (``json_api.cavity_analysis``).
    CAVITY_ANALYSIS_INVALID = "CAVITY_ANALYSIS_INVALID"
    #: An end of the list that is not a mirror, or a unit with a mirror inside the cavity.
    CAVITY_ELEMENT_UNSUPPORTED = "CAVITY_ELEMENT_UNSUPPORTED"
    #: An end mirror that does not send the light back along its path (a tilted plane end mirror).
    CAVITY_NOT_CLOSED = "CAVITY_NOT_CLOSED"
    #: A source named by the injection that is not a single-line Gaussian beam on the cavity's axis behind the first mirror, or
    #: whose light meets something else on its way to the first mirror: the reason of the injection quantities that need its
    #: beam (and the refusal code of a source the Project does not hold).
    CAVITY_SOURCE_UNSUPPORTED = "CAVITY_SOURCE_UNSUPPORTED"
    #: A cavity shorter than ``MIN_LENGTH_WAVELENGTHS`` wavelengths: outside the paraxial Gaussian and Airy model.
    CAVITY_TOO_SHORT = "CAVITY_TOO_SHORT"


CAVITY_ANALYSIS_INVALID = CavityErrorCode.CAVITY_ANALYSIS_INVALID.value
CAVITY_ELEMENT_UNSUPPORTED = CavityErrorCode.CAVITY_ELEMENT_UNSUPPORTED.value
CAVITY_NOT_CLOSED = CavityErrorCode.CAVITY_NOT_CLOSED.value
CAVITY_SOURCE_UNSUPPORTED = CavityErrorCode.CAVITY_SOURCE_UNSUPPORTED.value
CAVITY_TOO_SHORT = CavityErrorCode.CAVITY_TOO_SHORT.value
#: The shortest cavity (end mirror vertex to end mirror vertex along the unfolded axis, in wavelengths) the analysis takes.
MIN_LENGTH_WAVELENGTHS = 10.0
REQUEST_KEYS = frozenset({"components", "wavelength_nm", "axis", "reflectances", "transmittances", "round_trip_loss", "input_beam", "polarization",
                          "configuration_id",
                          "injection", "targets", "aperture_over_mode_min", "coaxial_tolerance"})
#: The eigen-polarizations of the cavity's crystals a request may choose (module docstring).
POLARIZATIONS = ("ORDINARY", "EXTRAORDINARY")
#: The polarizations a request may choose for a cavity whose tilted mirrors share one plane of incidence: perpendicular to that plane
#: (``S``) or in it (``P``).
MIRROR_POLARIZATIONS = ("S", "P")
#: Models the paraxial matrix takes at one plane (``compile.paraxial_matrix``): they state no thickness, so they add no group path.
ZERO_THICKNESS_KINDS = frozenset({"IDEAL_THIN_LENS", "TABULATED_THIN_LENS", "POLARIZATION_ELEMENT", "NEUTRAL_DENSITY", "FARADAY_ROTATOR"})
#: Elements that change the polarization between the mirrors: s and p are then not eigen-polarizations of the round trip.
POLARIZATION_CHANGING_KINDS = frozenset({"POLARIZATION_ELEMENT", "FARADAY_ROTATOR"})
INPUT_BEAM_KEYS = frozenset({"waist_radius_mm", "waist_radius_y_mm", "waist_after_reference_mm"})
INJECTION_KEYS = frozenset({"power_w", "source_id", "beam", "detuning_Hz", "length_offset_nm", "mode_order_max", "scan", "source_linewidth_fraction_max"})
#: The injected beam: the cavity's fundamental eigenmode itself, the request's ``input_beam``, or the beam of the source ``source_id``.
INJECTED_BEAMS = ("EIGENMODE", "INPUT_BEAM", "SOURCE")
SCAN_KEYS = frozenset({"from_Hz", "to_Hz", "points"})
#: The transverse order resolved when the injection states none.
DEFAULT_MODE_ORDER = 10

#: Reason codes of quantities outside the model or not known.
CAVITY_UNSTABLE = "CAVITY_UNSTABLE"
CAVITY_MARGINALLY_STABLE = "CAVITY_MARGINALLY_STABLE"
MODE_OUTSIDE_PARAXIAL_LIMITS = "MODE_OUTSIDE_PARAXIAL_LIMITS"
MODE_CLIPPED_BY_APERTURE = "MODE_CLIPPED_BY_APERTURE"
REFLECTANCE_NOT_STATED = "REFLECTANCE_NOT_STATED"
REFLECTANCE_OUTSIDE_COATING_DATA = "REFLECTANCE_OUTSIDE_COATING_DATA"
REFLECTANCE_DEPENDS_ON_POLARIZATION = "REFLECTANCE_DEPENDS_ON_POLARIZATION"
OPTICAL_THICKNESS_NOT_STATED = "OPTICAL_THICKNESS_NOT_STATED"
GROUP_INDEX_UNAVAILABLE = "GROUP_INDEX_UNAVAILABLE"
LOSSES_NOT_MODELED = "LOSSES_NOT_MODELED"
NO_HALF_MAXIMUM = "NO_HALF_MAXIMUM"
INPUT_SECTION_HAS_NO_MODE = "INPUT_SECTION_HAS_NO_MODE"
POLARIZATION_NOT_STATED = "POLARIZATION_NOT_STATED"
CRYSTAL_EIGENMODES_UNDEFINED = "CRYSTAL_EIGENMODES_UNDEFINED"
CRYSTAL_AXIS_OBLIQUE_TO_SECTIONS = "CRYSTAL_AXIS_OBLIQUE_TO_SECTIONS"
CRYSTAL_IN_POWERED_ELEMENT = "CRYSTAL_IN_POWERED_ELEMENT"
TRANSMITTANCE_NOT_STATED = "TRANSMITTANCE_NOT_STATED"
TRANSMITTANCE_DEPENDS_ON_POLARIZATION = "TRANSMITTANCE_DEPENDS_ON_POLARIZATION"
SOURCE_NOT_SINGLE_FREQUENCY = "SOURCE_NOT_SINGLE_FREQUENCY"
#: A source whose ``M^2`` is above 1: a higher-order or multimode transverse beam, not a TEM00 Gaussian beam.
SOURCE_NOT_FUNDAMENTAL_MODE = "SOURCE_NOT_FUNDAMENTAL_MODE"
INPUT_BEAM_NOT_STATED = "INPUT_BEAM_NOT_STATED"
#: The detuning is neither stated nor fixed by the Project (module docstring, Detuning).
DETUNING_NOT_STATED = "DETUNING_NOT_STATED"
#: The injected beam's radius at the reference plane is above ``1 / APERTURE_OVER_MODE_MIN`` of the input mirror's clear radius.
INPUT_BEAM_CLIPPED = "INPUT_BEAM_CLIPPED"
#: A target whose ``unit`` is not its metric's (``validation.scope.CAVITY_METRIC_UNITS``).
TARGET_UNIT_MISMATCH = "TARGET_UNIT_MISMATCH"
#: An ``input_beam`` stated with a source equals the source's beam at the reference plane within this, relative to the waist radius
#: and, for the waist position, to the larger of 1 mm and the Rayleigh range.
INPUT_BEAM_MATCH_REL = 1e-6
#: The two coupling terms ``sqrt(R1)`` and ``(1 - A1) b`` count as equal (critical coupling) within this.
CRITICAL_COUPLING_ABS = 1e-12
#: A source counts as single-frequency for the Airy response while its linewidth is at most this fraction of the cavity's (the default of
#: the request's ``injection.source_linewidth_fraction_max``).
SOURCE_LINEWIDTH_FRACTION = 0.01
#: sqrt(ln 2): the coherence length of a Gaussian spectrum, L_c = (2 sqrt(ln 2) / pi) lambda^2 / linewidth.
SQRT_LN2 = 0.8325546111576977
#: The source's wavelength equals the request's within this (relative).
WAVELENGTH_MATCH_REL = 1e-12
#: An optic axis lies in a crystal's face (and two crystals' axes are parallel, or one along a section axis) within this
#: (the cosine of its angle to the light, the sine of the angle between two axes).
CRYSTAL_AXIS_TOLERANCE = 1e-9

#: A half trace within this of +-1 is marginal stability: the eigenmode is not a Gaussian beam of finite size.
MARGINAL_HALF_TRACE = 1e-9
#: Two matrix entries count as equal (a symmetric cavity) within this, relative to max(1, |entry|).
SYMMETRY_REL = 1e-12
#: The clear aperture radius over the mode radius at every element must be at least this (a Gaussian loses exp(-18), about
#: 1.5e-8 of its power, outside 3 radii): the hard-edged aperture then leaves the Gaussian eigenmode. The default of the request's
#: ``aperture_over_mode_min``.
APERTURE_OVER_MODE_MIN = 3.0
#: Two reflectances (s, p) count as one value within this.
REFLECTANCE_EQUAL_ABS = 1e-12
#: Vacuum speed of light (mm/s), applied to the round-trip group path.
C_MM_PER_S = 299792458000.0
TWO_PI = 2.0 * math.pi
MODEL_EIGENMODE = "CAVITY_GAUSSIAN_EIGENMODE"
MODEL_RESONANCE = "CAVITY_FABRY_PEROT_RESONANCE"
MODEL_INJECTION = "CAVITY_INJECTED_BUILD_UP"
FINESSE_NAME = "finesse pi sqrt(g) / (1 - g), g = sqrt(round-trip power fraction)"
STABLE, UNSTABLE, MARGINAL = "STABLE", "UNSTABLE", "MARGINAL"
ANALYTIC = Convergence(method=ConvergenceMethod.ANALYTIC)


class CavityError(ParaxialError):
    """A cavity request the analysis cannot answer: ``problems`` (each ``{"code", "message", "location"}``)."""

    def as_dict(self) -> dict[str, Any]:
        return {"code": CAVITY_ANALYSIS_INVALID, "problems": list(self.problems)}


def _cavity_refuse(code: str, message: str, location: str | None = None) -> CavityError:
    return CavityError([{"code": code, "message": message, "location": location}])


# ------------------------------------------------------------------------------------------------ request


def _signed(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _refuse(REQUEST_INVALID, f"{name} is a number, not {value!r}", name)
    try:
        number = float(Decimal(str(value)))
    except ArithmeticError:
        raise _refuse(REQUEST_INVALID, f"{name} is a number, not {value!r}", name) from None
    if not math.isfinite(number):
        raise _refuse(REQUEST_INVALID, f"{name} must be finite, not {value!r}", name)
    return number


def _request(request: Any) -> tuple[list[str], float, str | None, dict[str, float], dict[str, float] | None, str | None]:
    if not isinstance(request, Mapping):
        raise _refuse(REQUEST_INVALID, "a request is a JSON object with components and wavelength_nm")
    unknown = sorted(set(request) - REQUEST_KEYS)
    if unknown:
        raise _refuse(REQUEST_INVALID, f"unknown request fields {unknown}; a request takes {sorted(REQUEST_KEYS)}")
    ids = request.get("components")
    if not isinstance(ids, (list, tuple)) or len(ids) < 2 or not all(isinstance(i, str) for i in ids):
        raise _refuse(REQUEST_INVALID, "components lists the cavity's elements once each, from one end mirror to the other (at least two)", "components")
    wavelength_nm = _number(request.get("wavelength_nm"), "wavelength_nm")
    axis = request.get("axis")
    if axis not in (None, *SECTIONS):
        raise _refuse(REQUEST_INVALID, f"axis is 'X', 'Y' or absent, not {axis!r}", "axis")
    stated = request.get("reflectances") or {}
    if not isinstance(stated, Mapping):
        raise _refuse(REQUEST_INVALID, "reflectances is an object {instance_id: R}", "reflectances")
    reflectances = {}
    for key, value in stated.items():
        r = _signed(value, f"reflectances.{key}")
        if r < 0.0 or r >= 1.0:
            raise _refuse(REQUEST_INVALID, f"reflectances.{key} is a power reflectance from 0 to below 1, not {value!r}", "reflectances")
        reflectances[str(key)] = r
    beam = request.get("input_beam")
    input_beam = None
    if beam is not None:
        if not isinstance(beam, Mapping) or set(beam) - INPUT_BEAM_KEYS or "waist_radius_mm" not in beam or "waist_after_reference_mm" not in beam:
            raise _refuse(REQUEST_INVALID, f"input_beam is an object with waist_radius_mm and waist_after_reference_mm (and optionally "
                          f"waist_radius_y_mm); it takes {sorted(INPUT_BEAM_KEYS)}", "input_beam")
        w_x = _number(beam["waist_radius_mm"], "input_beam.waist_radius_mm")
        w_y = w_x if beam.get("waist_radius_y_mm") is None else _number(beam["waist_radius_y_mm"], "input_beam.waist_radius_y_mm")
        input_beam = {"waist_radius_mm": w_x, "waist_radius_y_mm": w_y, "waist_after_reference_mm": _signed(beam["waist_after_reference_mm"],
                                                                                                          "input_beam.waist_after_reference_mm")}
    polarization = request.get("polarization")
    if polarization not in (None, *POLARIZATIONS, *MIRROR_POLARIZATIONS):
        raise _refuse(REQUEST_INVALID, "polarization is 'ORDINARY' / 'EXTRAORDINARY' (a cavity with a crystal), 'S' / 'P' (a cavity whose tilted "
                      f"mirrors share one plane of incidence) or absent, not {polarization!r}", "polarization")
    return list(ids), wavelength_nm, axis, reflectances, input_beam, polarization


def _fraction(value: Any, name: str, *, below_one: bool) -> float:
    number = _signed(value, name)
    if number < 0.0 or number > 1.0 or (below_one and number == 1.0):
        raise _refuse(REQUEST_INVALID, f"{name} is a power fraction from 0 to {'below ' if below_one else ''}1, not {value!r}", name)
    return number


@dataclass(frozen=True)
class _Extra:
    """The request's injection fields (module docstring): ``transmittances``, ``round_trip_loss``, ``injection`` and ``targets``."""

    transmittances: dict[str, float]
    round_trip_loss: float | None
    injection: dict[str, Any] | None
    targets: tuple[dict[str, Any], ...]
    aperture_over_mode_min: float


def _extra_request(request: Mapping[str, Any], wavelength_nm: float) -> _Extra:
    stated = request.get("transmittances") or {}
    if not isinstance(stated, Mapping):
        raise _refuse(REQUEST_INVALID, "transmittances is an object {instance_id: T}", "transmittances")
    transmittances = {str(key): _fraction(value, f"transmittances.{key}", below_one=False) for key, value in stated.items()}
    loss = request.get("round_trip_loss")
    round_trip_loss = None if loss is None else _fraction(loss, "round_trip_loss", below_one=True)
    injection = None
    given = request.get("injection")
    if given is not None:
        if not isinstance(given, Mapping) or set(given) - INJECTION_KEYS:
            raise _refuse(REQUEST_INVALID, f"injection is an object; it takes {sorted(INJECTION_KEYS)}", "injection")
        beam = given.get("beam")
        if beam not in INJECTED_BEAMS:
            raise _refuse(REQUEST_INVALID, f"injection.beam is 'EIGENMODE' (power_w in a beam matched to the fundamental mode), 'INPUT_BEAM' (the "
                          f"request's input_beam) or 'SOURCE' (the beam of the source source_id), not {beam!r}", "injection.beam")
        if (given.get("power_w") is None) == (given.get("source_id") is None):
            raise _refuse(REQUEST_INVALID, "injection states the injected power: power_w, or source_id (a source of the Project, whose light "
                          "arriving at the first mirror it takes)", "injection")
        if given.get("source_id") is not None and not isinstance(given["source_id"], str):
            raise _refuse(REQUEST_INVALID, "injection.source_id is a component instance id", "injection.source_id")
        if beam == "SOURCE" and given.get("source_id") is None:
            raise _refuse(REQUEST_INVALID, "injection.beam SOURCE is the beam of the source source_id: name the source", "injection.beam")
        if beam == "EIGENMODE" and given.get("source_id") is not None:
            raise _refuse(REQUEST_INVALID, "a source's beam is its own, not the cavity's eigenmode: inject it with beam SOURCE (its mode matching "
                          "is computed and used), or state power_w for a beam matched to the fundamental mode", "injection.beam")
        if given.get("detuning_Hz") is not None and given.get("length_offset_nm") is not None:
            raise _refuse(REQUEST_INVALID, "injection states the detuning once: detuning_Hz (from the nearest TEM00 resonance), or length_offset_nm "
                          "(the cavity's length less the length of that resonance)", "injection")
        order = given.get("mode_order_max", DEFAULT_MODE_ORDER)
        if isinstance(order, bool) or not isinstance(order, int) or not 0 <= order <= MAX_MODE_ORDER:
            raise _refuse(REQUEST_INVALID, f"injection.mode_order_max is an integer from 0 to {MAX_MODE_ORDER}, not {order!r}", "injection.mode_order_max")
        scan = given.get("scan")
        if scan is not None:
            if not isinstance(scan, Mapping) or set(scan) != SCAN_KEYS:
                raise _refuse(REQUEST_INVALID, f"injection.scan is an object with {sorted(SCAN_KEYS)}", "injection.scan")
            points = scan["points"]
            if isinstance(points, bool) or not isinstance(points, int) or not 2 <= points <= MAX_SCAN_POINTS:
                raise _refuse(REQUEST_INVALID, f"injection.scan.points is an integer from 2 to {MAX_SCAN_POINTS}, not {points!r}", "injection.scan.points")
            low, high = _signed(scan["from_Hz"], "injection.scan.from_Hz"), _signed(scan["to_Hz"], "injection.scan.to_Hz")
            if not high > low:
                raise _refuse(REQUEST_INVALID, "injection.scan.to_Hz is above from_Hz", "injection.scan")
            scan = {"from_Hz": low, "to_Hz": high, "points": points}
        detuning = None if given.get("detuning_Hz") is None else _signed(given["detuning_Hz"], "injection.detuning_Hz")
        carrier = C_MM_PER_S / (wavelength_nm * NM_TO_MM)
        for name, value in (("injection.detuning_Hz", detuning), ("injection.scan.from_Hz", None if scan is None else scan["from_Hz"]),
                            ("injection.scan.to_Hz", None if scan is None else scan["to_Hz"])):
            if value is not None and abs(value) >= carrier:
                raise _refuse(REQUEST_INVALID, f"{name} {value:g} Hz is not below the light's frequency c / lambda = {carrier:.6g} Hz: a detuning is a "
                              "frequency offset from a resonance, below the light's frequency (the coatings and the Gouy phase are taken at the stated "
                              "wavelength)", name)
        fraction = given.get("source_linewidth_fraction_max")
        injection = {"beam": beam, "mode_order_max": order, "scan": scan,
                     "source_linewidth_fraction_max": SOURCE_LINEWIDTH_FRACTION if fraction is None else
                     _number(fraction, "injection.source_linewidth_fraction_max"),
                     "power_w": None if given.get("power_w") is None else _number(given["power_w"], "injection.power_w"),
                     "source_id": given.get("source_id"),
                     "detuning_Hz": detuning,
                     "length_offset_nm": None if given.get("length_offset_nm") is None else _signed(given["length_offset_nm"], "injection.length_offset_nm")}
    targets = request.get("targets") or []
    if not isinstance(targets, (list, tuple)):
        raise _refuse(REQUEST_INVALID, "targets is a list of {target_id, quantity_id, target_value?, tolerance_abs?, tolerance_rel?, minimum?, maximum?}", "targets")
    checked = []
    for k, target in enumerate(targets):
        where = f"targets[{k}]"
        if not isinstance(target, Mapping) or set(target) - TARGET_KEYS or not isinstance(target.get("target_id"), str) \
                or not isinstance(target.get("quantity_id"), str):
            raise _refuse(REQUEST_INVALID, f"{where} is an object with target_id and quantity_id (strings) and target_value / minimum / maximum; it takes "
                          f"{sorted(TARGET_KEYS)}", where)
        entry = {"target_id": target["target_id"], "quantity_id": target["quantity_id"]}
        for key in ("target_value", "minimum", "maximum"):
            if target.get(key) is not None:
                entry[key] = _signed(target[key], f"{where}.{key}")
        for key in ("tolerance_abs", "tolerance_rel"):
            if target.get(key) is not None:
                entry[key] = _signed(target[key], f"{where}.{key}")
                if entry[key] < 0.0:
                    raise _refuse(REQUEST_INVALID, f"{where}.{key} is not negative", where)
        if not any(key in entry for key in ("target_value", "minimum", "maximum")):
            raise _refuse(REQUEST_INVALID, f"{where} needs target_value and/or minimum / maximum", where)
        checked.append(entry)
    names = [t["target_id"] for t in checked]
    if len(set(names)) != len(names):
        raise _refuse(REQUEST_INVALID, "targets repeats a target_id", "targets")
    ratio = request.get("aperture_over_mode_min")
    return _Extra(transmittances, round_trip_loss, injection, tuple(checked),
                  APERTURE_OVER_MODE_MIN if ratio is None else _number(ratio, "aperture_over_mode_min"))


# ------------------------------------------------------------------------------------------------ matrices and modes


def _free(length: float) -> Matrix:
    return ((1.0, length), (0.0, 1.0))


def _reverse(matrix: Matrix) -> Matrix:
    """The same elements met in reverse order (reduced matrices in air on both sides)."""

    (a, b), (c, d) = matrix
    return ((d, b), (c, a))


def _close(x: float, y: float) -> bool:
    return abs(x - y) <= SYMMETRY_REL * max(1.0, abs(x), abs(y))


def _same(m: Matrix, n: Matrix) -> bool:
    return all(_close(m[i][j], n[i][j]) for i in (0, 1) for j in (0, 1))


def _through(matrix: Matrix, q: complex) -> complex:
    (a, b), (c, d) = matrix
    return (a * q + b) / (c * q + d)


def _modulus(z: complex) -> float:
    return math.sqrt(z.real * z.real + z.imag * z.imag)


def _eigen(matrix: Matrix) -> tuple[str, float, complex | None, float | None]:
    """``(stability, half trace, q, round-trip Gouy phase in [0, 2 pi))`` of a round-trip matrix (module docstring)."""

    (a, b), (_c, d) = matrix
    m = (a + d) / 2.0
    if abs(m) > 1.0:
        return (MARGINAL if abs(m) - 1.0 <= MARGINAL_HALF_TRACE else UNSTABLE), m, None, None
    if 1.0 - abs(m) <= MARGINAL_HALF_TRACE:
        return MARGINAL, m, None, None
    s = math.sqrt((1.0 - m) * (1.0 + m))
    inverse_q = complex((d - a) / (2.0 * b), -s / abs(b))
    theta = float(atan2(s if b > 0.0 else -s, m))
    return STABLE, m, 1.0 / inverse_q, theta + TWO_PI if theta < 0.0 else theta


def _radius(q: complex, wavelength_mm: float) -> float:
    inverse = 1.0 / q
    return math.sqrt(-wavelength_mm / (math.pi * inverse.imag))


def _overlap_1d(q_a: complex, q_b: complex) -> float:
    return 2.0 * math.sqrt(q_a.imag * q_b.imag) / _modulus(q_a - q_b.conjugate())


# ------------------------------------------------------------------------------------------------ apertures and paths


def _available(radius: Any, half_widths: Any, center: Any) -> float | None:
    if radius is None and half_widths is None:
        return None
    reach = float(Decimal(radius)) if radius is not None else min(float(Decimal(v)) for v in half_widths)
    if center is not None:
        cx, cy = (float(Decimal(v)) for v in center)
        reach -= math.sqrt(cx * cx + cy * cy)
    return reach


def _clear_radius(element: CompiledElement) -> float | None:
    """The radius about the axis inside every opening of the element (mm), ``None`` when it states none."""

    if element.is_compound:
        radii = [_clear_radius(member) for member in element.members]
        return None if any(r is None for r in radii) else min(r for r in radii if r is not None)
    model = element.model
    if model.model_kind == "APERTURE_STOP":
        if model.obstruction_radius_mm is not None:
            return 0.0
        return _available(model.opening_radius_mm, model.opening_half_widths_mm, model.opening_center_mm)
    surfaces = getattr(model, "surfaces", None)
    if surfaces and hasattr(surfaces[0], "vertex_z_mm"):
        radii = [_available(s.clear_radius_mm, s.clear_half_widths_mm, s.aperture_center_mm) for s in surfaces]
        known = [r for r in radii if r is not None]
        return min(known) if known else None
    clear = getattr(model, "clear_radius_mm", None)
    return None if clear is None else float(Decimal(clear))


def _glass_path(element: CompiledElement, wavelength_nm: float, polarization: str | None = None, *,
                phase: bool = False, cfg: NumericalConfig, surrounding: float,
                thin: list[str] | None = None) -> tuple[float | None, str | None, str]:
    """The path the glass adds beyond its span in the surrounding air, with a reason and message if unavailable.

    ``surrounding`` is the ambient group index, or the phase index when ``phase`` is true. A vacuum gap has index 1.
    A crystal uses the index of the ``polarization`` wave. A model with no stated thickness adds no path and is named
    in ``thin`` (``ZERO_THICKNESS_KINDS``)."""

    if element.is_compound:
        extra = 0.0
        for member in element.members:
            part, reason, message = _glass_path(member, wavelength_nm, polarization, phase=phase, cfg=cfg, surrounding=surrounding, thin=thin)
            if part is None:
                return None, reason, message
            extra += part
        return extra, None, ""
    model = element.model
    if model.model_kind == "APERTURE_STOP":
        return 0.0, None, ""
    surfaces = getattr(model, "surfaces", None)
    if model.model_kind in ZERO_THICKNESS_KINDS:
        if thin is not None:
            thin.append(element.instance_id)
        return 0.0, None, ""
    if model.model_kind != "SURFACE_SEQUENCE" or not surfaces:
        return None, OPTICAL_THICKNESS_NOT_STATED, (f"{element.instance_id!r} is a {model.model_kind} element: its model states no thickness or "
                                                    "medium, so its group path is not known")
    materials = {m.material_id: m for m in element.materials}
    extra = 0.0
    for i in range(1, len(surfaces)):
        medium = model.media[i - 1]
        if medium == AIR:
            continue
        thickness = float(Decimal(surfaces[i].vertex_z_mm) - Decimal(surfaces[i - 1].vertex_z_mm))
        if medium == VACUUM:
            extra += (1.0 - surrounding) * thickness
            continue
        material = materials[element.engine_material_id(medium)]
        if isinstance(material, UniaxialMaterial):
            if polarization is None:
                return None, POLARIZATION_NOT_STATED, f"{element.instance_id!r} holds the crystal {medium}: state the polarization"
            material = law_material(material, material.ordinary if polarization == "ORDINARY" else material.extraordinary)
        if phase:
            n_g, _issue = build_evaluator(material, cfg).index(wavelength_nm)
        else:
            n_g = group_index(material, wavelength_nm, cfg)
        if n_g is None:
            return None, GROUP_INDEX_UNAVAILABLE, f"no {'refractive' if phase else 'group'} index of {medium} in {element.instance_id!r} at {wavelength_nm:g} nm"
        extra += (n_g - surrounding) * thickness
    return extra, None, ""


# ------------------------------------------------------------------------------------------------ crystals


@dataclass(frozen=True)
class _Cut:
    """One crystal medium of an element between the mirrors: where it is in ``placed``, its element's name, thickness (mm),
    material, ``|c . d|`` (the cosine of its optic axis ``c`` to the light) and the optic axis in the section frame."""

    item: int
    name: str
    thickness: float
    material: UniaxialMaterial
    across: float
    section_axis: tuple[float, float]


@dataclass(frozen=True)
class _Crystals:
    """What the cavity's crystals do to the analysis (module docstring): ``reason``/``message`` withhold every quantity that
    depends on a crystal's index; ``mode_reason``/``mode_message`` the eigenmode alone; ``extra``: per item of ``placed`` and
    section the change of its reduced length for the chosen wave (mm)."""

    cuts: tuple[_Cut, ...] = ()
    reason: str | None = None
    message: str = ""
    mode_reason: str | None = None
    mode_message: str = ""
    extra: dict[int, dict[str, float]] = field(default_factory=dict)


def _indices(material: UniaxialMaterial, wavelength_nm: float, cfg: NumericalConfig) -> tuple[float | None, float | None]:
    out = []
    for law in (material.ordinary, material.extraordinary):
        value, _issue = build_evaluator(law_material(material, law), cfg).index(wavelength_nm)
        out.append(value)
    return out[0], out[1]


def _cuts(project: Any, layout: Any) -> list[_Cut]:
    """Every crystal medium of the transmissive elements between the end mirrors (compound members included)."""

    components = {c.instance_id: c for c in load_project(project).components}
    placed = layout.placed
    found: list[_Cut] = []
    for k in range(1, len(placed) - 1):
        item = placed[k]
        if item.reflector:
            continue
        frame = _local_axes(components[item.instance_id].nominal_pose.quaternion_wxyz)
        for part in (item.members or (item,)):
            element = part.element
            model = element.model
            surfaces = getattr(model, "surfaces", None)
            if model.model_kind != "SURFACE_SEQUENCE" or not surfaces:
                continue
            materials = {m.material_id: m for m in element.materials}
            stretch = layout.walk.stretches[part.stretch]
            for i in range(1, len(surfaces)):
                material = materials.get(element.engine_material_id(model.media[i - 1]))
                if not isinstance(material, UniaxialMaterial):
                    continue
                axis = material.optic_axis if material.optic_axis_component is None else _in_frame(frame, material.optic_axis)
                d = part.direction_in
                across = abs(axis[0] * d[0] + axis[1] * d[1] + axis[2] * d[2])
                sx = axis[0] * stretch.section_x[0] + axis[1] * stretch.section_x[1] + axis[2] * stretch.section_x[2]
                sy = axis[0] * stretch.section_y[0] + axis[1] * stretch.section_y[1] + axis[2] * stretch.section_y[2]
                length = math.sqrt(sx * sx + sy * sy)
                thickness = float(Decimal(surfaces[i].vertex_z_mm) - Decimal(surfaces[i - 1].vertex_z_mm))
                found.append(_Cut(k, _who_cut(item, part), thickness, material, across,
                                  (sx / length, sy / length) if length > 0.0 else (0.0, 0.0)))
    return found


def _who_cut(item: _Placed, part: _Placed) -> str:
    return item.instance_id if part is item else f"{item.instance_id} (member {part.element.member_id})"


def _crystal_state(project: Any, layout: Any, wavelength_nm: float, polarization: str | None) -> _Crystals:
    """The analysis of the cavity's crystals (module docstring)."""

    cuts = tuple(_cuts(project, layout))
    if not cuts:
        return _Crystals()
    cfg = layout.lowered.request.configuration.numerical
    names = ", ".join(sorted({c.name for c in cuts}))
    tilted = [c for c in cuts if c.across > CRYSTAL_AXIS_TOLERANCE]
    first = cuts[0].section_axis
    turned = [c for c in cuts if abs(c.section_axis[0] * first[1] - c.section_axis[1] * first[0]) > CRYSTAL_AXIS_TOLERANCE]
    if tilted or turned:
        why = (f"the optic axis of {tilted[0].name} is not in its faces (cosine {tilted[0].across:.6g} to the light): its "
               "extraordinary wave walks off the axis" if tilted else
               f"the optic axes of {cuts[0].name} and {turned[0].name} are not parallel across the light: the crystals' "
               "eigen-polarizations differ")
        return _Crystals(cuts, CRYSTAL_EIGENMODES_UNDEFINED,
                         f"{why}, so no linear polarization is an eigenmode of the round trip and the quantities that depend "
                         "on the crystals' indices are outside the model")
    if polarization is None:
        n_o, n_e = _indices(cuts[0].material, wavelength_nm, cfg)
        values = "" if n_o is None or n_e is None else f" (n_o = {n_o:.6g}, n_e = {n_e:.6g} at {wavelength_nm:g} nm)"
        return _Crystals(cuts, POLARIZATION_NOT_STATED,
                         f"the cavity holds the crystal {names}{values}: light polarized across its optic axis (ORDINARY) and "
                         "along it (EXTRAORDINARY) has different round-trip group paths, free spectral ranges and eigenmodes; "
                         "state polarization in the request")
    if polarization == "ORDINARY":
        return _Crystals(cuts)
    along = "X" if abs(first[0]) >= abs(first[1]) else "Y"
    if min(abs(first[0]), abs(first[1])) > CRYSTAL_AXIS_TOLERANCE:
        return _Crystals(cuts, mode_reason=CRYSTAL_AXIS_OBLIQUE_TO_SECTIONS,
                         mode_message=f"the optic axis of {names} lies at an angle to the section axes: the extraordinary wave "
                         "diffracts differently along and across it, which turns the eigenmode's axes off the sections")
    extra: dict[int, dict[str, float]] = {}
    n_air = surrounding_index(layout.lowered.air, wavelength_nm)
    for cut in cuts:
        n_o, n_e = _indices(cut.material, wavelength_nm, cfg)
        if n_o is None or n_e is None:
            return _Crystals(cuts, GROUP_INDEX_UNAVAILABLE, f"no refractive index of the crystal in {cut.name} at {wavelength_nm:g} nm")
        shift = extra.setdefault(cut.item, {"X": 0.0, "Y": 0.0})
        for s in SECTIONS:
            effective = n_o * n_o / n_e if s == along else n_e
            shift[s] += cut.thickness * n_air * (1.0 / effective - 1.0 / n_o)
    for k in extra:
        for s in SECTIONS:
            (a, _b), (c, d) = layout.placed[k].matrices[s]
            if not (a == 1.0 and c == 0.0 and d == 1.0):
                return _Crystals(cuts, mode_reason=CRYSTAL_IN_POWERED_ELEMENT,
                                 mode_message=f"{layout.placed[k].instance_id!r} holds a crystal and has optical power: the "
                                 "extraordinary wave's matrix of an element with curved surfaces is not formed")
    return _Crystals(cuts, extra=extra)


# ------------------------------------------------------------------------------------------------ reflectances


@dataclass(frozen=True)
class _Coefficient:
    """A mirror's power reflectance or transmittance at its angle of incidence: ``value`` (``None``: not known, with ``reason`` and
    ``message``); ``s`` / ``p``: the coating's values per polarization where they differ; ``entry``: what a value held outside the
    coating's stated data is (with ``reason``); ``missing``: the request field that states it; ``interval``: proven bounds of a
    coefficient that is not stated; ``note``: where the coating's values were read."""

    value: float | None
    origin: QuantityOrigin
    reason: str | None
    message: str
    s: float | None = None
    p: float | None = None
    entry: ValueBasis | None = None
    missing: tuple[str, ...] = ()
    interval: Interval | None = None
    note: str = ""


def _transmission(item: _Placed, stated_r: Mapping[str, float], stated_t: Mapping[str, float], wavelength_nm: float, cfg: NumericalConfig) -> _Coefficient:
    """The power transmittance of an end mirror of the cavity at its angle of incidence: stated, else its coating's. A reflectance
    stated in the request without its transmittance leaves ``0 <= T <= 1 - R`` (no value is invented)."""

    field = f"transmittances.{item.instance_id}"
    if item.instance_id in stated_t:
        return _Coefficient(stated_t[item.instance_id], QuantityOrigin.EXTERNALLY_CHARACTERIZED, None, "stated in the request")
    if item.instance_id in stated_r:
        return _Coefficient(None, QuantityOrigin.MODELED, TRANSMITTANCE_NOT_STATED,
                            f"the reflectance of {item.instance_id!r} is stated in the request: state its transmittance in transmittances as well "
                            f"(it is at most 1 - R = {1.0 - stated_r[item.instance_id]:.12g})", missing=(field,),
                            interval=Interval(lower=0.0, upper=1.0 - stated_r[item.instance_id], kind=IntervalKind.PROVEN))
    values, aoi, reason, message, held = _coating_values(item, wavelength_nm, cfg)
    if values is None:
        return _Coefficient(None, QuantityOrigin.MODELED, reason, message.replace("reflectance", "transmittance").replace("reflectances", "transmittances"),
                            missing=(field,))
    t_s, t_p = float(values.ts[0]), float(values.tp[0])
    note = f"its coating at {wavelength_nm:g} nm and {aoi:.6g} deg"
    if abs(t_s - t_p) > REFLECTANCE_EQUAL_ABS:
        return _Coefficient(None, QuantityOrigin.MODELED, TRANSMITTANCE_DEPENDS_ON_POLARIZATION,
                            f"{item.instance_id!r} transmits T_s = {t_s:.6g} and T_p = {t_p:.6g} at {aoi:.6g} deg: state T in transmittances",
                            missing=(field,))
    if held is not None:
        return _Coefficient(t_s, QuantityOrigin.MODELED, REFLECTANCE_OUTSIDE_COATING_DATA, message.replace("reflectance", "transmittance")
                            .replace("reflectances", "transmittances"), entry=held, missing=(field,), note=note)
    return _Coefficient(t_s, QuantityOrigin.MODELED, None, note, note=note)


def _coating_values(item: _Placed, wavelength_nm: float, cfg: NumericalConfig) -> tuple[Any, float, str | None, str, ValueBasis | None]:
    """``(coefficients or None, angle of incidence (deg), reason, message, what a value outside the coating's stated data is)`` of a
    mirror's coating at its angle of incidence. Outside the wavelength bands or the angle range the coating's record states, the
    compiled coating holds its nearest stated value (the number a run traces with): it is returned with an ``ESTIMATE`` entry that
    states the wavelength and the angle against the record's ranges. ``None`` only where the compiled coating has no value."""

    element = item.element
    surface = reflecting_surface(element)
    engine = next((s for s in element.surfaces if s.surface_id == surface.surface_id), None) if surface is not None else None
    coating = getattr(getattr(engine, "interaction", None), "coating", None)
    if coating is None:
        return None, 0.0, REFLECTANCE_NOT_STATED, f"{item.instance_id!r} states no coating reflectance: state it in reflectances", None
    turn = (item.direction_out[0] - item.direction_in[0], item.direction_out[1] - item.direction_in[1], item.direction_out[2] - item.direction_in[2])
    length = math.sqrt(turn[0] * turn[0] + turn[1] * turn[1] + turn[2] * turn[2])
    cosine = abs(turn[0] * item.direction_in[0] + turn[1] * item.direction_in[1] + turn[2] * item.direction_in[2]) / length
    aoi = float(acos_deg(cosine))
    evaluator = CoatingEvaluator(coating, cfg)
    declared = coating.declared_validity
    outside = (f"the coating of {item.instance_id!r} has no published data at {wavelength_nm:g} nm and {aoi:.6g} deg: state its reflectance in "
               "reflectances")
    if not bool(evaluator.in_range(wavelength_nm, np.array([aoi]))[0]):
        return None, aoi, REFLECTANCE_OUTSIDE_COATING_DATA, outside, None
    values = evaluator.coefficients(wavelength_nm, np.array([aoi]), location=item.instance_id)
    if declared is None:
        return values, aoi, None, "", None
    bands = list(declared.wavelength_bands_nm)
    in_band = any(low - cfg.tabulated_range_tol_nm <= wavelength_nm <= high + cfg.tabulated_range_tol_nm for low, high in bands)
    in_angle = declared.min_deg - declared.slack_deg <= aoi <= declared.max_deg + declared.slack_deg
    if in_band and in_angle:
        return values, aoi, None, "", None
    # (the band the wavelength is nearest to: the one whose stated value the compiled coating holds)
    nearest = min(bands, key=lambda band: 0.0 if band[0] <= wavelength_nm <= band[1] else min(abs(wavelength_nm - band[0]), abs(wavelength_nm - band[1])))
    where = (item.instance_id,)
    ranges = (ValidityCheck(parameter="coating_wavelength_nm", unit="nm", minimum=float(nearest[0]), maximum=float(nearest[1]),
                            observed=wavelength_nm, within=in_band, at=where),
              ValidityCheck(parameter="coating_angle_of_incidence_deg", unit="deg", minimum=float(declared.min_deg),
                            maximum=float(declared.max_deg), observed=aoi, within=in_angle, at=where))
    held = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(Assumed(input=AssumedInput.HELD_AT_NEAREST_STATED_POINT, at=where, code=REFLECTANCE_OUTSIDE_COATING_DATA),),
                      validity=tuple(check for check in ranges if check.within is False))
    return values, aoi, REFLECTANCE_OUTSIDE_COATING_DATA, (
        f"the coating of {item.instance_id!r} has no published data at {wavelength_nm:g} nm and {aoi:.6g} deg: the value is its nearest published "
        "one, held; state its reflectance in reflectances"), held


def _reflection(item: _Placed, stated: Mapping[str, float], wavelength_nm: float, cfg: NumericalConfig, polarization: str | None,
                eigen: bool) -> _Coefficient:
    """The power reflectance of a mirror of the cavity at its angle of incidence: stated, else its coating's. Where the coating
    reflects s- and p-polarized light differently (a fold mirror at an angle) the value is the one of ``polarization`` (``"S"`` /
    ``"P"``: an eigen-polarization of this cavity's round trip, ``cavity_analysis``), and not known without it."""

    field = f"reflectances.{item.instance_id}"
    if item.instance_id in stated:
        return _Coefficient(stated[item.instance_id], QuantityOrigin.EXTERNALLY_CHARACTERIZED, None, "stated in the request")
    values, aoi, reason, message, held = _coating_values(item, wavelength_nm, cfg)
    if values is None:
        return _Coefficient(None, QuantityOrigin.MODELED, reason, message, missing=(field,))
    r_s, r_p = float(values.rs[0]), float(values.rp[0])
    note = f"its coating at {wavelength_nm:g} nm and {aoi:.6g} deg"
    outside = {} if held is None else {"entry": held, "missing": (field,)}
    if abs(r_s - r_p) > REFLECTANCE_EQUAL_ABS:
        if polarization in MIRROR_POLARIZATIONS:
            return _Coefficient(r_s if polarization == "S" else r_p, QuantityOrigin.MODELED, reason, message or f"{note}, {polarization.lower()}-polarized",
                                s=r_s, p=r_p, note=note, **outside)
        how = ("state polarization 'S' or 'P' (the eigen-polarizations of this cavity's round trip), or R in reflectances" if eigen else
               "s and p are not eigen-polarizations of this cavity's round trip (its tilted mirrors do not share one plane of incidence, or it "
               "holds a crystal, a polarization element or a Faraday rotator); state R in reflectances")
        return _Coefficient(None, QuantityOrigin.MODELED, REFLECTANCE_DEPENDS_ON_POLARIZATION,
                            f"{item.instance_id!r} reflects R_s = {r_s:.6g} and R_p = {r_p:.6g} at {aoi:.6g} deg: the round trip depends on the "
                            f"polarization; {how}", s=r_s, p=r_p, missing=("polarization", field) if eigen else (field,), note=note, entry=held)
    return _Coefficient(r_s, QuantityOrigin.MODELED, reason, message or note, note=note, **outside)


# ------------------------------------------------------------------------------------------------ the analysis


class _Quantities:
    def __init__(self) -> None:
        self.items: list[Quantity] = []

    def add(self, quantity_id: str, name: str, unit: str | None, value: float | None, *, basis: str, state: QuantityState | None = None,
            reason: str | None = None, message: str = "", origin: QuantityOrigin = QuantityOrigin.MODELED, interval: Interval | None = None,
            source: str | None = None, depends_on: Sequence[str] = (), value_basis: ValueBasis | None = None, missing: Sequence[str] = (),
            resolution: float | None = None) -> None:
        """One quantity. A ``KNOWN`` / ``KNOWN_ZERO`` quantity holds its value; a ``BOUNDED`` one its interval and, where one was
        computed, the value beside it; any other state a value only with the ``value_basis`` that says what the number is.
        ``resolution``: the proven most the true value differs from a ``KNOWN`` value that leaves out light below the negligible level."""

        if state is None:
            state = QuantityState.KNOWN if value is not None else QuantityState.UNKNOWN
        valued = state in (QuantityState.KNOWN, QuantityState.KNOWN_ZERO)
        held = value is not None and (valued or state is QuantityState.BOUNDED or value_basis is not None)
        convergence = ANALYTIC if valued or interval is not None else None
        if convergence is not None and resolution is not None:
            convergence = Convergence(method=ConvergenceMethod.ANALYTIC, resolution=resolution)
        self.items.append(Quantity(quantity_id=quantity_id, name=name, unit=unit, state=state, origin=origin, value=value if held else None,
                                   interval=interval, basis=basis, convergence=convergence,
                                   reason=None if valued else reason, message=message, source=source, missing=tuple(missing),
                                   depends_on=tuple(depends_on), scope=QuantityScope(), value_basis=value_basis if held else None))


def cavity_analysis(project: Any, request: Any, *, catalog: Any = None, run_result_hash: str | None = None,  # noqa: C901
                    engine_build_id: str | None = None) -> AnalysisResultV1:
    """The cavity analysis (module docstring). Raises ``CavityError`` for a request it cannot answer and ``LocalRunError`` for a
    Project the runtime cannot lower. ``run_result_hash``: the run this analysis is read from (``os_optics.results``), recorded in
    its identity; ``engine_build_id``: the Engine build id the caller holds (``os_optics.analysis.document``)."""

    try:
        ids, wavelength_nm, axis, stated, input_beam, polarization = _request(request)
        more = _extra_request(request, wavelength_nm)
        configuration_id = request.get("configuration_id")
        if configuration_id is not None:
            # (one configuration of a Project that declares several: analysed as its own Project, configured(id))
            from .configurations import configured_project

            project = configured_project(project, configuration_id)
        tolerance = coaxial_tolerance(request.get("coaxial_tolerance"))
        layout = _lay_out(project, ids, wavelength_nm, axis, source_id=None, catalog=catalog, cavity=True, tolerance=tolerance)
    except CavityError:
        raise
    except ParaxialError as exc:
        raise CavityError(exc.problems) from None
    placed = layout.placed
    for k in (0, len(placed) - 1):
        item = placed[k]
        if not item.reflector or item.element.is_compound:
            raise _cavity_refuse(CAVITY_ELEMENT_UNSUPPORTED, f"{item.instance_id!r} is an end of the cavity, so it is a mirror (a single element, not a unit): "
                                 "list the elements from one end mirror to the other", f"component {item.instance_id}")
        back = (-item.direction_in[0], -item.direction_in[1], -item.direction_in[2])
        returned = _angle(item.direction_out, back) if _dot(item.direction_out, back) > 0.0 else math.pi / 2.0
        # (a mirror tilted by t returns the light 2 t off its path; without a stated tolerance: the direction cosine, as the placement)
        open_ = (_dot(item.direction_in, item.direction_out) > SYMMETRY_REL - 1.0 if tolerance.tilt_rad is None else
                 _dot(item.direction_out, back) <= 0.0 or returned > 2.0 * tolerance.tilt_rad)
        if open_:
            raise _cavity_refuse(CAVITY_NOT_CLOSED, f"the end mirror {item.instance_id!r} does not send the light back along its path (it is not normal to "
                                 f"the axis: the light returns {returned:.6g} rad off its path, and {tolerance.tilt_text()} for the mirror's tilt, "
                                 "half of that): a linear cavity's end mirrors meet the light at normal incidence", f"component {item.instance_id}")
    for item in placed[1:-1]:
        if item.element.is_compound and item.reflector:
            raise _cavity_refuse(CAVITY_ELEMENT_UNSUPPORTED, f"{item.instance_id!r} is a unit with a mirror inside the cavity: list its members as separate "
                                 "components", f"component {item.instance_id}")
    for key in stated:
        if key not in ids or not next(p for p in placed if p.instance_id == key).reflector:
            raise _cavity_refuse(REQUEST_INVALID, f"reflectances names {key!r}, which is not a mirror of the cavity", "reflectances")
    for key in more.transmittances:
        if key not in (placed[0].instance_id, placed[-1].instance_id):
            raise _cavity_refuse(REQUEST_INVALID, f"transmittances names {key!r}, which is not an end mirror of the cavity", "transmittances")
        if key in stated and stated[key] + more.transmittances[key] > 1.0:
            raise _cavity_refuse(REQUEST_INVALID, f"{key!r}: R + T = {stated[key] + more.transmittances[key]:.12g} is above 1", "transmittances")
    if more.injection is not None and more.injection["beam"] == "INPUT_BEAM" and input_beam is None:
        raise _cavity_refuse(REQUEST_INVALID, "injection.beam INPUT_BEAM is the request's input_beam: state input_beam", "injection.beam")
    phase_index = surrounding_index(layout.lowered.air, wavelength_nm)
    wavelength_mm = wavelength_nm * NM_TO_MM / phase_index
    gaps = layout.gaps
    cavity_length = gaps[0] if len(placed) == 2 else placed[-1].entrance_mm - placed[0].exit_mm
    if cavity_length < MIN_LENGTH_WAVELENGTHS * wavelength_mm:
        raise _cavity_refuse(CAVITY_TOO_SHORT, f"the cavity is {cavity_length:.6g} mm long, {cavity_length / wavelength_mm:.6g} wavelengths: the paraxial "
                             f"Gaussian eigenmode and the Airy resonances need at least {MIN_LENGTH_WAVELENGTHS:g} wavelengths", "components")
    source = None if more.injection is None or more.injection["source_id"] is None else _injected_source(project, more.injection["source_id"], layout,
                                                                                                        wavelength_nm, wavelength_mm)
    q_input = None if input_beam is None else _input_q(input_beam, wavelength_mm)
    if source is not None:
        if q_input is not None:
            _check_stated_beam(q_input, source, wavelength_mm)
        q_input = source["q"]
    #: What stands between the source's beam and the cavity (``_injected_source``): the beam-dependent quantities say so.
    unsupported = None if source is None else source["problem"]
    section = axis or "X"
    cfg = layout.lowered.request.configuration.numerical
    crystal_polarization = polarization if polarization in POLARIZATIONS else None
    crystals = _crystal_state(project, layout, wavelength_nm, crystal_polarization)
    # s and p of the tilted mirrors are eigen-polarizations of the round trip where those mirrors share one plane of incidence and
    # nothing between the mirrors changes the polarization
    plane_normal = _common_plane_normal(placed)
    eigen = plane_normal is not None and plane_normal != (0.0, 0.0, 0.0) and not crystals.cuts and not _changes_polarization(placed)
    if polarization in MIRROR_POLARIZATIONS and crystals.cuts:
        raise _cavity_refuse(REQUEST_INVALID, f"polarization {polarization!r} names a polarization relative to the tilted mirrors' plane of incidence; "
                             "this cavity holds a crystal, whose eigen-polarizations are 'ORDINARY' and 'EXTRAORDINARY'", "polarization")
    if polarization in MIRROR_POLARIZATIONS and plane_normal == (0.0, 0.0, 0.0):
        raise _cavity_refuse(REQUEST_INVALID, f"polarization {polarization!r} names a polarization relative to the tilted mirrors' plane of incidence; "
                             "this cavity has no tilted mirror, and its mirrors at normal incidence do not tell s from p", "polarization")
    mirror_polarization = polarization if polarization in MIRROR_POLARIZATIONS and eigen else None
    matrices = [dict(item.matrices) for item in placed]
    for k, shift in crystals.extra.items():
        for s in SECTIONS:
            (a, b), (c, d) = matrices[k][s]
            matrices[k][s] = ((a, b + shift[s]), (c, d))

    def pass_matrix(s: str) -> Matrix:
        matrix = _free(gaps[0])
        for k in range(1, len(placed) - 1):
            matrix = _multiply(_free(gaps[k]), _multiply(matrices[k][s], matrix))
        return matrix

    modes: dict[str, dict[str, Any]] = {}
    for s in SECTIONS:
        forward = pass_matrix(s)
        m1, m2 = placed[0].matrices[s], placed[-1].matrices[s]
        half = _multiply(m2, forward)
        round_trip = _multiply(m1, _multiply(_reverse(forward), half))
        symmetric = _same(m1, m2) and _close(forward[0][0], forward[1][1])
        stability, m_rt, q, theta = _eigen(round_trip)
        checked = m_rt
        if symmetric:
            stability, checked, q, theta_h = _eigen(half)
            theta = None if theta_h is None else (2.0 * theta_h) % TWO_PI
        modes[s] = {"round_trip": round_trip, "half_trace": m_rt, "checked": checked, "stability": stability, "q": q, "theta": theta, "symmetric": symmetric}
    mode = modes[section]
    stability, m, q0, theta = mode["stability"], mode["half_trace"], mode["q"], mode["theta"]

    # the eigenmode along the forward pass
    radii: dict[str, float] = {}
    waists: list[dict[str, Any]] = []
    if q0 is not None:
        q = q0
        radii[placed[0].instance_id] = _radius(q, wavelength_mm)
        for k in range(1, len(placed)):
            start, before = placed[k - 1], q
            q = q + gaps[k - 1]
            distance = -before.real
            waists.append({"from": start.instance_id, "to": placed[k].instance_id, "radius": math.sqrt(wavelength_mm * before.imag / math.pi),
                           "along": start.exit_mm + distance, "inside": -LATERAL_SLACK_MM <= distance <= gaps[k - 1] + LATERAL_SLACK_MM,
                           "point": [start.exit_point[i] + start.direction_out[i] * distance for i in range(3)]})
            entrance = _radius(q, wavelength_mm)
            q = _through(matrices[k][section], q) if k < len(placed) - 1 else q
            radii[placed[k].instance_id] = max(entrance, _radius(q, wavelength_mm))

    # conditions of the paraxial Gaussian eigenmode: their limits are the Project's numerical settings and the request's
    # aperture_over_mode_min
    aperture_min = more.aperture_over_mode_min
    crystal_reason = crystals.reason or crystals.mode_reason
    checks = [ValidityCheck(parameter="half_round_trip_half_trace" if mode["symmetric"] else "round_trip_half_trace", minimum=MARGINAL_HALF_TRACE - 1.0,
                            maximum=1.0 - MARGINAL_HALF_TRACE, observed=None if crystal_reason else mode["checked"],
                            within=None if crystal_reason else stability == STABLE)]
    reason, why = None, ""
    if crystal_reason:
        # (the crystals' index enters every number of the eigenmode: none is stated for an undetermined polarization)
        reason, why = crystal_reason, crystals.message if crystals.reason else crystals.mode_message
        q0, theta, radii, waists = None, None, {}, []
    elif q0 is None:
        reason = CAVITY_UNSTABLE if stability == UNSTABLE else CAVITY_MARGINALLY_STABLE
        why = (f"the round trip's half trace (A + D) / 2 = {m:.12g}: the cavity is {'unstable' if stability == UNSTABLE else 'marginally stable'} "
               f"(g1 g2 = {(1.0 + m) / 2.0:.12g}), so it has no Gaussian eigenmode of finite size")
    #: The eigenmode exists: its quantities have the round-trip matrix's values, whatever the validity conditions below say.
    mode_exists = q0 is not None
    divergence = max((wavelength_mm / (math.pi * w["radius"]) for w in waists), default=None)
    checks.append(_condition("far_field_half_angle_rad", divergence, maximum=cfg.gaussian_scope_divergence_rad,
                             within=None if divergence is None else divergence <= cfg.gaussian_scope_divergence_rad,
                             setting="gaussian_scope_divergence_rad", default=DEFAULT_NUMERICAL_CONFIG.gaussian_scope_divergence_rad,
                             code=MODE_OUTSIDE_PARAXIAL_LIMITS))
    curvature: dict[str, float] = {}
    for item in placed:
        surface = reflecting_surface(item.element) if item.reflector else None
        if surface is not None and surface.radius_mm is not None and item.instance_id in radii:
            curvature[item.instance_id] = radii[item.instance_id] / abs(float(Decimal(surface.radius_mm)))
    worst_curvature = max(curvature.values(), default=None)
    checks.append(_condition("mode_radius_over_mirror_radius", worst_curvature, maximum=cfg.gaussian_max_w_over_radius,
                             within=None if worst_curvature is None else worst_curvature <= cfg.gaussian_max_w_over_radius,
                             setting="gaussian_max_w_over_radius", default=DEFAULT_NUMERICAL_CONFIG.gaussian_max_w_over_radius,
                             at=() if not curvature else (max(curvature, key=lambda i: curvature[i]),), code=MODE_OUTSIDE_PARAXIAL_LIMITS))
    clear = {item.instance_id: _clear_radius(item.element) for item in placed}
    ratios = {i: clear[i] / radii[i] for i in radii if clear[i] is not None}
    worst_ratio = min(ratios.values(), default=None)
    clipped = worst_ratio is not None and worst_ratio < aperture_min
    #: The elements whose clear aperture reaches the mode: their edges add a loss no input states.
    clipped_at = tuple(i for i in ratios if ratios[i] < aperture_min)
    narrow = min(ratios, key=lambda i: ratios[i]) if ratios else None
    checks.append(_condition("clear_aperture_over_mode_radius", worst_ratio, minimum=aperture_min,
                             within=None if worst_ratio is None or (len(ratios) < len(radii) and not clipped) else not clipped,
                             setting="aperture_over_mode_min", default=APERTURE_OVER_MODE_MIN, at=() if narrow is None else (narrow,),
                             code=MODE_CLIPPED_BY_APERTURE))
    # what was measured of the elements against the nominal axis, and the tolerance they were accepted within
    parts = [part for item in placed for part in (item.members or (item,))]
    decentred = max((part for part in parts if part.decentre_mm is not None), key=lambda part: part.decentre_mm, default=None)
    tilted = max((part for part in parts if part.tilt_rad is not None), key=lambda part: part.tilt_rad, default=None)
    checks.append(_condition("element_decentre_mm", None if decentred is None else decentred.decentre_mm, maximum=tolerance.lateral_mm,
                             within=None if decentred is None else True, setting="coaxial_tolerance.lateral_mm", default=LATERAL_TOLERANCE_MM,
                             at=() if decentred is None else (decentred.instance_id,)))
    checks.append(_condition("element_tilt_rad", None if tilted is None else tilted.tilt_rad, maximum=tolerance.as_dict()["tilt_rad"],
                             within=None if tilted is None else True, setting="coaxial_tolerance.tilt_rad", default=DEFAULT_TILT_RAD,
                             at=() if tilted is None else (tilted.instance_id,)))
    if reason is None:
        if divergence is not None and divergence > cfg.gaussian_scope_divergence_rad or worst_curvature is not None and worst_curvature > cfg.gaussian_max_w_over_radius:
            reason, why = MODE_OUTSIDE_PARAXIAL_LIMITS, (f"the eigenmode leaves the paraxial Gaussian model: far-field half angle {divergence:.6g} rad "
                                                         f"(limit {cfg.gaussian_scope_divergence_rad:g}), mode radius over mirror radius "
                                                         f"{worst_curvature or 0.0:.6g} (limit {cfg.gaussian_max_w_over_radius:g})")
        elif clipped:
            reason, why = MODE_CLIPPED_BY_APERTURE, (f"the clear aperture of {narrow!r} is {ratios[narrow]:.6g} mode radii (at least "
                                                     f"{aperture_min:g}): its edge changes the mode and adds loss")
    # what the eigenmode's numbers are: the paraxial round-trip matrix's values. A condition that fails makes them estimates that
    # name it (state OUT_OF_MODEL_SCOPE); a limit the caller set, met, is named with the value it produced (state KNOWN).
    failed = tuple(c for c in checks[1:] if c.within is False)
    restated = tuple(c for c in checks[1:] if c.within is True and c.setting is not None)
    mode_entry = None
    if mode_exists and failed:
        mode_entry = ValueBasis(kind=ValueKind.ESTIMATE, validity=(*failed, *restated))
    elif mode_exists and restated:
        mode_entry = ValueBasis(kind=ValueKind.MODEL, validity=restated)

    def at_element(instance_id: str) -> ValueBasis | None:
        """The entry of the mode radius at one element: the eigenmode's, and beside a failed condition that element's own clear
        aperture over its mode radius."""

        own = ratios.get(instance_id)
        if mode_entry is None or mode_entry.kind is ValueKind.MODEL or own is None or instance_id == narrow:
            return mode_entry
        return ValueBasis(kind=ValueKind.ESTIMATE, validity=(*mode_entry.validity, ValidityCheck(
            parameter="clear_aperture_over_mode_radius", minimum=aperture_min, observed=own, within=own >= aperture_min,
            setting="aperture_over_mode_min", at=(instance_id,))))

    mode_state = QuantityState.KNOWN if reason is None else QuantityState.OUT_OF_MODEL_SCOPE

    models = [
        ModelBasis(model_id=MODEL_EIGENMODE, name="fundamental Gaussian eigenmode of the paraxial round-trip matrix", version=CAVITY_ANALYSIS_VERSION,
                   assumptions=("paraxial ray-transfer matrices of the elements' prescriptions at the stated wavelength, on the unfolded axis",
                                "a linear cavity: the light returns from each end mirror along its path through the Project's ambient medium",
                                "a symmetric cavity takes its eigenmode from the half round trip",
                                "apertures do not reach the mode: the clear aperture radius is at least aperture_over_mode_min mode radii at "
                                "every element (3 unless the request states it)",
                                "a fundamental mode with a round-trip Gouy phase theta: TEM_mnq lies (m + n) theta / (2 pi) free spectral ranges above TEM_00q"),
                   validity=tuple(checks)),
        ModelBasis(model_id=MODEL_RESONANCE, name="Fabry-Perot resonance of the round trip (Airy function)", version=CAVITY_ANALYSIS_VERSION,
                   assumptions=("resonance where the round-trip phase is a multiple of 2 pi; FSR = c / round-trip group path",
                                "the round-trip power fraction is the product of the mirrors' reflectances at their angles of incidence; other losses "
                                "(intracavity elements, clipping, an unstable cavity) only lower it",
                                "reflection phases are equal for every longitudinal mode: they shift all resonances alike")),
    ]
    if mirror_polarization is not None:
        models = [model.model_copy(update={"assumptions": (*model.assumptions,
                                                            f"the light is {mirror_polarization.lower()}-polarized with respect to the tilted "
                                                            "mirrors' common plane of incidence, an eigen-polarization of the round trip")})
                  for model in models]
    if crystals.cuts:
        models = [model.model_copy(update={"assumptions": (*model.assumptions,
                                                            "a crystal's two eigen-polarizations (across and along its optic axis, which lies in "
                                                            "its faces) are the cavity's two families of modes; the analysis is for the one the "
                                                            "request states, with that wave's index, group index and diffraction")})
                  for model in models]
    out = _Quantities()
    out.add("cavity_length_mm", "length of the cavity along the unfolded axis, end mirror vertex to end mirror vertex", "mm", cavity_length, basis=MODEL_EIGENMODE)
    if crystal_reason:
        withheld = {"basis": MODEL_EIGENMODE, "state": QuantityState.OUT_OF_MODEL_SCOPE, "reason": reason, "message": why}
        out.add("round_trip_half_trace", "stability parameter (A + D) / 2 of the round-trip matrix", None, None, **withheld)
        out.add("g1g2", "stability product g1 g2 = (1 + (A + D) / 2) / 2 (stable between 0 and 1)", None, None, **withheld)
    else:
        out.add("round_trip_half_trace", "stability parameter (A + D) / 2 of the round-trip matrix", None, m, basis=MODEL_EIGENMODE)
        out.add("g1g2", "stability product g1 g2 = (1 + (A + D) / 2) / 2 (stable between 0 and 1)", None, (1.0 + m) / 2.0, basis=MODEL_EIGENMODE)
    if len(placed) == 2:
        for k, item in ((1, placed[0]), (2, placed[1])):
            power = item.matrices[section][1][0]
            out.add(f"g{k}", f"g{k} = 1 - L / R{k} of {item.instance_id}", None, 1.0 + cavity_length * power / 2.0, basis=MODEL_EIGENMODE)

    mode_value = (lambda v: v) if mode_exists else (lambda v: None)
    mode_kwargs = {"basis": MODEL_EIGENMODE, "state": mode_state, "reason": reason, "message": why, "value_basis": mode_entry}
    out.add("round_trip_gouy_phase_rad", "Gouy phase of the fundamental mode per round trip, in [0, 2 pi)", "rad",
            mode_value(theta) if theta is not None else None, **mode_kwargs)
    for item in placed:
        r = radii.get(item.instance_id)
        out.add(f"mode_radius_mm.{item.instance_id}", f"1/e^2 radius of the fundamental mode at {item.instance_id} (the larger of its entrance and exit)", "mm",
                mode_value(r) if r is not None else None, **{**mode_kwargs, "value_basis": at_element(item.instance_id)})
    for k, waist in enumerate(waists or [{"from": placed[j].instance_id, "to": placed[j + 1].instance_id} for j in range(len(placed) - 1)], start=1):
        where = f"between {waist['from']} and {waist['to']}"
        out.add(f"gap_{k}.waist_radius_mm", f"waist radius of the fundamental mode's beam {where}", "mm",
                mode_value(waist.get("radius")) if waist.get("radius") is not None else None, **mode_kwargs)
        out.add(f"gap_{k}.waist_along_axis_mm", f"waist position {where} along the unfolded axis from the first mirror's vertex (a waist outside "
                f"the gap is that beam's virtual waist)", "mm", mode_value(waist.get("along")) if waist.get("along") is not None else None, **mode_kwargs)

    # round-trip group path and free spectral range
    # (in the environment's air: its group index along the free gaps and the elements' spans, n_g - n_g,air through their media)
    air_group = _air_group_index(layout.lowered.air, wavelength_nm, cfg)
    path, path_reason, path_message = 2.0 * _sum(gaps) * air_group, None, ""
    path_state = None
    if crystals.reason:
        path, path_reason, path_message, path_state = None, crystals.reason, crystals.message, QuantityState.OUT_OF_MODEL_SCOPE
    thin: list[str] = []  # (elements whose model states no thickness: their path is the thin element's, none)
    for item in placed[1:-1] if path is not None else ():
        if item.reflector:
            continue
        found: list[str] = []
        extra, path_reason, path_message = _glass_path(item.element, wavelength_nm, crystal_polarization, cfg=cfg, surrounding=air_group, thin=found)
        if extra is None:
            path = None
            break
        if found:
            thin.append(item.instance_id)
        path += 2.0 * ((item.exit_mm - item.entrance_mm) * air_group + extra)
    #: What the group path and its readers are where an element states no thickness: the model's number with that element's
    #: substrate path taken as none.
    path_entry = None
    if path is not None and thin:
        path_entry = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(Assumed(input=AssumedInput.THIN_ELEMENT_PATH, at=tuple(thin), code=OPTICAL_THICKNESS_NOT_STATED),))
        path_state, path_reason = QuantityState.UNKNOWN, OPTICAL_THICKNESS_NOT_STATED
        path_message = (f"{', '.join(repr(name) for name in thin)}: the model states no thickness or medium, so the group path counts none through it "
                        "(twice the free gaps and the other elements' n_g t)")
    out.add("round_trip_group_path_mm", "round-trip group path: twice the free gaps plus n_g t through every element", "mm", path, basis=MODEL_RESONANCE,
            state=path_state, reason=path_reason, message=path_message, value_basis=path_entry)
    fsr = None if path is None else C_MM_PER_S / path
    out.add("free_spectral_range_Hz", "free spectral range c / round-trip group path", "Hz", fsr, basis=MODEL_RESONANCE, state=path_state, reason=path_reason,
            message=path_message, value_basis=path_entry, depends_on=("round_trip_group_path_mm",))
    spacing = None if fsr is None or theta is None else fsr * theta / TWO_PI
    spacing_state = mode_state if mode_state is not QuantityState.KNOWN else (QuantityState.KNOWN if spacing is not None and path_entry is None
                                                                             else QuantityState.UNKNOWN)
    out.add("transverse_mode_spacing_Hz", "frequency spacing from TEM_mnq to m + n + 1: FSR theta / (2 pi), modulo the FSR", "Hz", spacing,
            basis=MODEL_EIGENMODE, state=spacing_state, reason=reason or path_reason, message=why or path_message,
            value_basis=mode_entry if path_entry is None else merge_bases(mode_entry, path_entry, monotone=None),
            depends_on=("free_spectral_range_Hz", "round_trip_gouy_phase_rad"))

    # reflectances, finesse, linewidth
    rho, rho_reason, rho_message = 1.0, None, ""
    last = len(placed) - 1
    reflectance_of: dict[str, tuple[float | None, str | None, str]] = {}
    held: list[ValueBasis] = []  # (coefficients held outside their coating's stated data: the readers' values are estimates)
    for k, item in enumerate(placed):
        if not item.reflector:
            continue
        found_r = _reflection(item, stated, wavelength_nm, cfg, mirror_polarization, eigen)
        r = found_r.value
        count = 1 if k in (0, last) else 2
        external = found_r.origin is QuantityOrigin.EXTERNALLY_CHARACTERIZED
        if found_r.s is not None and found_r.p is not None:
            for letter, value in (("s", found_r.s), ("p", found_r.p)):
                out.add(f"reflectance_{letter}.{item.instance_id}", f"power reflectance of {item.instance_id} for {letter}-polarized light at its angle of "
                        "incidence", None, value, basis=MODEL_RESONANCE, state=None if found_r.entry is None else QuantityState.UNKNOWN,
                        reason=None if found_r.entry is None else REFLECTANCE_OUTSIDE_COATING_DATA, message=found_r.note, value_basis=found_r.entry)
        out.add(f"reflectance.{item.instance_id}", f"power reflectance of {item.instance_id} (met {count} time{'s' if count > 1 else ''} per round trip)", None, r,
                basis=MODEL_RESONANCE, origin=found_r.origin, state=None if found_r.entry is None or r is None else QuantityState.UNKNOWN,
                reason=found_r.reason, message=found_r.message, missing=found_r.missing, value_basis=found_r.entry,
                source="the request's reflectances" if external else None)
        # (the injection waits for a coefficient that is not known or is held outside its data: its readers name the request field)
        reflectance_of[item.instance_id] = (r if found_r.entry is None else None, found_r.reason, found_r.message)
        if r is None:
            rho, rho_reason, rho_message = None, found_r.reason, found_r.message
        elif rho is not None:
            rho *= r if count == 1 else r * r
            if found_r.entry is not None:
                held.append(found_r.entry)
                rho_reason, rho_message = found_r.reason, found_r.message
    rho_entry = merge_bases(*held, monotone=None) if held and rho is not None else None
    out.add("round_trip_power_fraction", "power fraction left after one round trip by the mirrors' reflectances", None, rho, basis=MODEL_RESONANCE,
            state=None if rho_entry is None else QuantityState.UNKNOWN, reason=rho_reason, message=rho_message, value_basis=rho_entry)
    # losses the mirror-limited finesse and linewidth leave out: the value is then a bound (the loss assumed zero is named in its basis)
    unmodeled = [f"the cavity's eigenmode is outside the model ({reason})"] if reason is not None else []
    loss_free = [Assumed(input=AssumedInput.NO_UNSTATED_LOSS, at=clipped_at if reason == MODE_CLIPPED_BY_APERTURE else (), code=reason)] \
        if reason is not None else []
    loss_missing: tuple[str, ...] = ()
    if more.round_trip_loss is not None:
        out.add("round_trip_loss", "power fraction lost per round trip besides the mirrors' reflectances (intracavity elements, scatter)", None,
                more.round_trip_loss, basis=MODEL_RESONANCE, origin=QuantityOrigin.EXTERNALLY_CHARACTERIZED, source="the request's round_trip_loss",
                message="stated in the request")
        rho = None if rho is None else rho * (1.0 - more.round_trip_loss)
    else:
        lossy = [item.instance_id for item in placed[1:-1] if not item.reflector]
        unmodeled += [f"the losses of {name!r} are not evaluated" for name in lossy]
        if lossy:
            loss_free.append(Assumed(input=AssumedInput.NO_UNSTATED_LOSS, at=tuple(lossy)))
            loss_missing = ("round_trip_loss",)

    def resonance(quantity_id: str, name: str, unit: str | None, value: float, side: ValueKind, estimates: Sequence[ValueBasis | None],
                  depends: tuple[str, ...]) -> None:
        """The finesse or the linewidth from the round-trip power fraction (and the free spectral range): the value; with losses that
        are not evaluated a bound on ``side`` beside its proven interval; with an input that is itself an estimate (a coefficient held
        outside its coating's data, a group path without an element's thickness) an estimate that names what its inputs name."""

        inherited = merge_bases(*estimates, monotone=None)
        if inherited is not None:
            from_path = path_entry is not None and any(entry is path_entry for entry in estimates)
            why_estimate = "; ".join(x for x in (rho_message if rho_entry is not None else "", path_message if from_path else "") if x)
            out.add(quantity_id, name, unit, value, basis=MODEL_RESONANCE, state=QuantityState.UNKNOWN,
                    reason=rho_reason if rho_entry is not None else path_reason, message=why_estimate, missing=loss_missing,
                    value_basis=ValueBasis(kind=ValueKind.ESTIMATE, assumed=(*inherited.assumed, *loss_free), validity=inherited.validity),
                    depends_on=depends)
        elif unmodeled:
            upper = side is ValueKind.UPPER_BOUND
            out.add(quantity_id, name, unit, value, basis=MODEL_RESONANCE, state=QuantityState.BOUNDED,
                    interval=Interval(upper=value, kind=IntervalKind.PROVEN) if upper else Interval(lower=value, kind=IntervalKind.PROVEN),
                    value_basis=ValueBasis(kind=side, assumed=tuple(loss_free)), missing=loss_missing,
                    message=(f"an upper bound: {'; '.join(unmodeled)}, and further loss only lowers it" if upper else
                             f"a lower bound: {'; '.join(unmodeled)}, and further loss only widens it"), depends_on=depends)
        else:
            out.add(quantity_id, name, unit, value, basis=MODEL_RESONANCE, depends_on=depends)

    if rho is None:
        out.add("finesse", FINESSE_NAME, None, None, basis=MODEL_RESONANCE, reason=rho_reason,
                message=rho_message, depends_on=("round_trip_power_fraction",))
        out.add("linewidth_fwhm_Hz", "full width at half maximum of a resonance (Airy function)", "Hz", None, basis=MODEL_RESONANCE, reason=rho_reason,
                message=rho_message, depends_on=("round_trip_power_fraction", "free_spectral_range_Hz"))
    else:
        g = math.sqrt(rho)
        finesse = math.pi * math.sqrt(g) / (1.0 - g)
        x = math.inf if g == 0.0 else (1.0 - g) / (2.0 * math.sqrt(g))  # (no light returns: no half maximum)
        resonance("finesse", FINESSE_NAME, None, finesse, ValueKind.UPPER_BOUND, (rho_entry,), ("round_trip_power_fraction",))
        if x > 1.0:
            out.add("linewidth_fwhm_Hz", "full width at half maximum of a resonance (Airy function)", "Hz", None, basis=MODEL_RESONANCE,
                    state=QuantityState.NOT_APPLICABLE, reason=NO_HALF_MAXIMUM,
                    message=f"the round-trip power fraction {rho:.6g} is too low for the resonances to fall to half their maximum between them",
                    depends_on=("round_trip_power_fraction",))
        elif fsr is None:
            out.add("linewidth_fwhm_Hz", "full width at half maximum of a resonance (Airy function)", "Hz", None, basis=MODEL_RESONANCE,
                    state=path_state, reason=path_reason, message=path_message, depends_on=("round_trip_power_fraction", "free_spectral_range_Hz"))
        else:
            resonance("linewidth_fwhm_Hz", "full width at half maximum of a resonance (Airy function)", "Hz", 2.0 * fsr / math.pi * float(asin(x)),
                      ValueKind.LOWER_BOUND, (rho_entry, path_entry), ("round_trip_power_fraction", "free_spectral_range_Hz"))

    if unsupported is not None:
        out.add("mode_matching_efficiency", "power fraction of the input beam coupled into the fundamental mode", None, None, basis=MODEL_EIGENMODE,
                state=QuantityState.OUT_OF_MODEL_SCOPE, reason=unsupported[0], message=unsupported[1])
    elif q_input is not None:
        q_in = q_input
        missing = [s for s in SECTIONS if modes[s]["q"] is None]
        if not mode_exists:
            out.add("mode_matching_efficiency", "power fraction of the input beam coupled into the fundamental mode", None, None, **mode_kwargs)
        elif missing:
            out.add("mode_matching_efficiency", "power fraction of the input beam coupled into the fundamental mode", None, None, basis=MODEL_EIGENMODE,
                    state=QuantityState.OUT_OF_MODEL_SCOPE, reason=INPUT_SECTION_HAS_NO_MODE,
                    message=f"section {missing[0]} of the cavity has no Gaussian eigenmode")
        else:
            efficiency = _overlap_1d(q_in["X"], modes["X"]["q"]) * _overlap_1d(q_in["Y"], modes["Y"]["q"])
            out.add("mode_matching_efficiency", "power fraction of the input beam coupled into the fundamental mode", None, efficiency, **mode_kwargs)

    injected = None
    if more.injection is not None:
        ctx = _Context(placed=placed, gaps=gaps, matrices=matrices, modes=modes, wavelength_nm=wavelength_nm, wavelength_mm=wavelength_mm,
                       phase_index=phase_index,
                       mode_exists=mode_exists, reason=reason, why=why, clipped_at=clipped_at, cfg=cfg, aperture_min=aperture_min, fsr=fsr, path_reason=path_reason, path_message=path_message,
                       polarization=crystal_polarization, family_axis=_family_axis(layout, crystals, mirror_polarization, plane_normal),
                       fsr_entry=path_entry, q_input=q_input, stated=stated, extra=more, source=source, clear=clear, crystals=crystals, walk=layout.walk,
                       reflectance_of=reflectance_of, linewidth=_known(out, "linewidth_fwhm_Hz"),
                       linewidth_lower=_proven_lower(out, "linewidth_fwhm_Hz"),
                       negligible=cfg.negligible_power_fraction)
        injected, conditions = _inject(ctx, out)
        models.append(_INJECTION_MODEL.model_copy(update={"validity": conditions}) if conditions else _INJECTION_MODEL)

    settings = {"components": list(ids), "wavelength_nm": wavelength_nm, "axis": axis, "reflectances": dict(sorted(stated.items())),
                "input_beam": input_beam, "catalog_version": layout.lowered.catalog_version}
    if configuration_id is not None:
        settings["configuration_id"] = configuration_id
    if polarization is not None:
        settings["polarization"] = polarization
    if more.transmittances:
        settings["transmittances"] = dict(sorted(more.transmittances.items()))
    if more.round_trip_loss is not None:
        settings["round_trip_loss"] = more.round_trip_loss
    # the limits and levels in force that a quantity's value_basis or a model's validity may name (their ``setting``): the Project's
    # numerical settings the analysis reads and the request's own limits
    settings["limits"] = {"gaussian_scope_divergence_rad": cfg.gaussian_scope_divergence_rad, "gaussian_max_w_over_radius": cfg.gaussian_max_w_over_radius,
                          "aperture_over_mode_min": aperture_min, "coaxial_tolerance.lateral_mm": tolerance.lateral_mm,
                          "coaxial_tolerance.tilt_rad": tolerance.as_dict()["tilt_rad"]}
    if more.injection is not None:
        settings["injection"] = {key: value for key, value in more.injection.items() if value is not None and key != "source_linewidth_fraction_max"}
        settings["limits"].update({"negligible_power_fraction": cfg.negligible_power_fraction,
                                   "injection.source_linewidth_fraction_max": more.injection["source_linewidth_fraction_max"]})
    arrays = {
        "round_trip_abcd": [list(row) for row in mode["round_trip"]],
        "eigenmode_from": "HALF_ROUND_TRIP" if mode["symmetric"] else "ROUND_TRIP",
        "stability": stability,
        "reference_plane": {"component": placed[0].instance_id, "point_mm": list(placed[0].exit_point), "direction": list(placed[0].direction_out)},
        "section": section, "section_frame": {"reference": layout.frame_reference, "x": list(layout.frame_x), "y": list(layout.frame_y)},
        "elements": [{"instance_id": item.instance_id, "reflector": item.reflector, "entrance_vertex_along_axis_mm": item.entrance_mm,
                      "exit_vertex_along_axis_mm": item.exit_mm, "entrance_vertex_mm": list(item.entrance_point), "clear_radius_mm": clear[item.instance_id],
                      "decentre_mm": item.decentre_mm, "tilt_rad": item.tilt_rad,
                      "reflections_per_round_trip": (1 if k in (0, len(placed) - 1) else 2) if item.reflector else 0} for k, item in enumerate(placed)],
        "waists": [{"from": w["from"], "to": w["to"], "inside_gap": w["inside"], "point_mm": w["point"]} for w in waists],
        "axis_path": _axis_path(layout.walk, list(placed))[1:-1],
    }
    if injected is not None:
        arrays["injection"] = injected
    targets = _targets(project, more.targets, ids, wavelength_nm, out.items)
    if targets:
        arrays["targets"] = targets
    # (the analysis has no budget and no early stop: an answered request completed; a quantity that is not known or outside the model
    # says so in its own state, reason and message)
    execution = completed()
    requirements = None if not targets else settings_hash({"targets": [{k: v for k, v in t.items() if k in REQUIREMENT_FIELDS} for t in targets]})
    quantities = out.items
    if configuration_id is not None:  # (every quantity is that configuration's)
        quantities = [q.model_copy(update={"scope": q.scope.model_copy(update={"configuration_id": configuration_id})}) for q in quantities]
    return document(CAVITY_ANALYSIS, CAVITY_ANALYSIS_VERSION, settings, execution, engine_input_hash=layout.lowered.engine_input_hash,
                    run_result_hash=run_result_hash, targets_hash=requirements, models=models, quantities=quantities, arrays=arrays,
                    engine_build_id=engine_build_id)


# ------------------------------------------------------------------------------------------------ injection


_INJECTION_MODEL = ModelBasis(
    model_id=MODEL_INJECTION, name="steady-state resonant build-up of light injected through the first mirror (Airy functions per transverse eigenmode)",
    version=CAVITY_ANALYSIS_VERSION,
    assumptions=(
        "a single-frequency source injected through the first listed mirror, coaxial with the cavity's axis; the light leaving through the last "
        "listed mirror is the transmitted light, the light leaving back through the first the reflected light",
        "a source that is not of one frequency (pulsed, or its linewidth above the stated fraction of the cavity's) has its time-averaged "
        "powers inside the bounds over every detuning: each spectral component sits at some detuning, with the coefficients and the Gouy "
        "phase of the stated wavelength",
        "a beam whose transverse content is not decomposed (M^2 above 1, a beam the input mirror's aperture cuts) is light not resolved, "
        "bounded from zero",
        "the input mirror reflects the field with -sqrt(R) from outside and +sqrt(R) from inside and transmits sqrt(T) either way (Stokes "
        "relations); its loss A = 1 - R - T is taken from each incident power with no interference term",
        "the detuning is the light's frequency less that of the nearest TEM00 resonance, stated (detuning_Hz, or the cavity's length less the "
        "length of that resonance, length_offset_nm): a Project's nominal positions do not fix it; without it the powers are bounded over every "
        "detuning",
        "the stated round_trip_loss is shared equally by the two passes; fold mirrors lose 1 - R at each of their two reflections per round trip",
        "the injected beam's transverse content in the cavity's Hermite-Gauss eigenbasis at the reference plane: HG_mn resonates at round-trip "
        "phase delta_00 - m theta_x - n theta_y, theta the round-trip Gouy phase of each section; a coaxial beam has only even orders",
        "a source's light reaches the first mirror with nothing else on its way (its trace from the source to the mirror); its Gaussian beam "
        "is carried through the first mirror's substrate by the paraxial matrix of its spherical surfaces; a beam that meets anything else is "
        "light not resolved, its power the traced arriving power",
        "with crystals in the cavity, a source's polarization divides its power between the crystals' two eigen-polarizations: along the optic "
        "axis (extraordinary) and across it (ordinary)",
        "a mode of order N = m + n is resolved while the clear apertures are at least aperture_over_mode_min sqrt(N + 1) mode radii (3 "
        "classical radii of the order unless the request states another ratio), "
        "its far-field half angle sqrt(N + 1) theta_0 and its radius over each mirror's radius of curvature stay inside the paraxial limits",
        "the light in modes not resolved is bounded for every round-trip phase and any additional loss",
    ))


@dataclass(frozen=True)
class _Context:
    placed: Any
    gaps: Any
    matrices: Any
    modes: Any
    wavelength_nm: float
    wavelength_mm: float
    phase_index: float
    #: The eigenmode exists; ``reason`` / ``why``: the validity condition it fails, if any; ``clipped_at``: the elements whose clear
    #: aperture reaches it.
    mode_exists: bool
    reason: str | None
    why: str
    clipped_at: tuple[str, ...]
    #: The Project's lowered numerical configuration and the request's ``aperture_over_mode_min``.
    cfg: NumericalConfig
    aperture_min: float
    fsr: float | None
    #: What the free spectral range is where an element of the cavity states no thickness (an estimate), else ``None``.
    fsr_entry: ValueBasis | None
    path_reason: str | None
    path_message: str
    polarization: str | None
    #: The polarization family analysed: ``(name, axis in the section frame, along it?)`` -- a crystal's optic axis (extraordinary
    #: along it) or the normal of the tilted mirrors' plane of incidence (S along it) --, ``None`` where the analysis is for any.
    family_axis: tuple[str, tuple[float, float], bool] | None
    q_input: dict[str, complex] | None
    stated: Mapping[str, float]
    extra: _Extra
    source: dict[str, Any] | None
    clear: Mapping[str, float | None]
    crystals: _Crystals
    walk: Any
    reflectance_of: Mapping[str, tuple[float | None, str | None, str]]
    #: The cavity's linewidth where it is known, and the proven least it can be (its value, or the lower end of its interval).
    linewidth: float | None
    linewidth_lower: float | None
    #: The share of the injected power below which light left out changes no state (``NumericalConfig.negligible_power_fraction``
    #: of the Project's lowered configuration).
    negligible: float


def _known(out: _Quantities, quantity_id: str) -> float | None:
    """The value of a ``KNOWN`` quantity (a bounded quantity's value beside its interval is not the quantity's value)."""

    quantity = next((q for q in out.items if q.quantity_id == quantity_id), None)
    return None if quantity is None or quantity.state not in (QuantityState.KNOWN, QuantityState.KNOWN_ZERO) else quantity.value


def _proven_lower(out: _Quantities, quantity_id: str) -> float | None:
    """The proven least a quantity can be: its ``KNOWN`` value, or the lower end of its ``PROVEN`` interval."""

    quantity = next((q for q in out.items if q.quantity_id == quantity_id), None)
    if quantity is None:
        return None
    if quantity.state in (QuantityState.KNOWN, QuantityState.KNOWN_ZERO):
        return quantity.value
    interval = quantity.interval
    return interval.lower if interval is not None and interval.kind is IntervalKind.PROVEN else None


def _input_q(beam: Mapping[str, float], wavelength_mm: float) -> dict[str, complex]:
    """The complex beam parameter ``q = z - z_waist + i z_R`` of the request's ``input_beam`` at the reference plane, per section."""

    return {"X": complex(-beam["waist_after_reference_mm"], math.pi * beam["waist_radius_mm"] ** 2 / wavelength_mm),
            "Y": complex(-beam["waist_after_reference_mm"], math.pi * beam["waist_radius_y_mm"] ** 2 / wavelength_mm)}


def _waist_of(q: complex, wavelength_mm: float) -> tuple[float, float]:
    """``(waist radius, waist position after the plane)`` of the beam of parameter ``q`` at a plane (mm)."""

    return math.sqrt(wavelength_mm * q.imag / math.pi), -q.real


def _check_stated_beam(q_stated: Mapping[str, complex], source: Mapping[str, Any], wavelength_mm: float) -> None:
    """Refuse an ``input_beam`` that is not the beam of the injected source at the reference plane (``INPUT_BEAM_MATCH_REL``)."""

    if source["q"] is None:
        return
    for s in SECTIONS:
        w_stated, z_stated = _waist_of(q_stated[s], wavelength_mm)
        w_source, z_source = _waist_of(source["q"][s], wavelength_mm)
        if abs(w_stated - w_source) > INPUT_BEAM_MATCH_REL * w_source or \
                abs(z_stated - z_source) > INPUT_BEAM_MATCH_REL * max(1.0, source["q"][s].imag):
            raise _cavity_refuse(REQUEST_INVALID, f"input_beam is not the beam of source {source['source_id']!r} at the reference plane: in section {s} it "
                                 f"states a waist of {w_stated:.9g} mm {z_stated:.9g} mm after the plane, the source's beam has {w_source:.9g} mm "
                                 f"{z_source:.9g} mm after it; leave input_beam out (the source's beam is used) or state power_w for another beam",
                                 "input_beam")


def _air_group_index(air: Any, wavelength_nm: float, cfg: NumericalConfig) -> float:
    """The group index of the environment's air (``LoweredProject.air``) at a vacuum wavelength (1 for vacuum)."""

    value = group_index(air, wavelength_nm, cfg)
    return 1.0 if value is None else value


def _substrate_matrix(element: CompiledElement, wavelength_nm: float, source_id: str, cfg: NumericalConfig, air: Any) -> Matrix:
    """The reduced paraxial matrix of the first mirror crossed from behind: from the plane of its first vertex through every surface
    (the reflector's coating transmitting) to the plane of the reflector's vertex, in air on both sides. Spherical and plane surfaces of
    homogeneous media; refused otherwise."""

    model = element.model
    where = "injection.source_id"
    for surface in model.surfaces:
        if surface.has_section_powers or surface.is_cylindrical or surface.is_freeform or surface.is_axicon:
            raise _cavity_refuse(CAVITY_SOURCE_UNSUPPORTED, f"surface {surface.surface_id} of {element.instance_id!r} is not spherical or plane: the beam of "
                                 f"source {source_id!r} through it to the cavity is not formed; state power_w with input_beam (the beam inside the "
                                 "cavity) instead", where)
    evaluators = {m.material_id: m for m in element.materials}
    n_air = surrounding_index(air, wavelength_nm)  # (indices relative to the environment's air, as compile.paraxial_matrix)

    def index(medium: str) -> float:
        if medium == AIR:
            return 1.0
        if medium == VACUUM:
            return 1.0 / n_air
        material = evaluators[element.engine_material_id(medium)]
        if isinstance(material, (UniaxialMaterial, *GRADIENT_MEDIA)):
            raise _cavity_refuse(CAVITY_SOURCE_UNSUPPORTED, f"the substrate of {element.instance_id!r} is {medium}, which is not a homogeneous isotropic "
                                 f"medium: the beam of source {source_id!r} through it is not formed; state power_w with input_beam instead", where)
        value, _issue = build_evaluator(material, cfg).index(wavelength_nm)
        if value is None:
            raise _cavity_refuse(CAVITY_SOURCE_UNSUPPORTED, f"no refractive index of {medium} in {element.instance_id!r} at {wavelength_nm:g} nm", where)
        return value / n_air

    matrix: Matrix = ((1.0, 0.0), (0.0, 1.0))
    surfaces = model.surfaces
    for i, surface in enumerate(surfaces):
        before, after = _media(model, i)
        if i > 0:
            gap = float(Decimal(surface.vertex_z_mm) - Decimal(surfaces[i - 1].vertex_z_mm))
            matrix = _multiply(((1.0, gap / index(before)), (0.0, 1.0)), matrix)
        if surface.radius_mm is not None:
            power = (index(after) - index(before)) / float(Decimal(surface.radius_mm))
            if power != 0.0:
                matrix = _multiply(((1.0, 0.0), (-power, 1.0)), matrix)
    return matrix


def _traced_arrival(layout: Any, source_id: str, first: _Placed, emitted: float) -> tuple[float | None, dict[str, Any], str | None]:
    """``(power arriving at the back of the first mirror or None, the record of that arrival, what stands between the source's beam and
    the mirror or None)`` from the trace of the source alone (``os_optics.validation.ray_paths``: the run's trace with its pupil rays).

    The power is the traced one where every root path of the source ends on the back of the mirror and no model condition of the
    elements fails on the way: a stop or a lens the chief ray passes changes the beam, and the trace carries its power. It is not
    known where the light does not reach the mirror, or where part of the beam meets an element its chief ray misses or another
    model condition fails (``local.validity``: e.g. ``BEAM_PARTIALLY_INTERCEPTED`` -- the chief ray's power counts that light). The
    beam at the mirror is the source's only when nothing but the mirror's own surfaces and observation planes lies on the way."""

    from dataclasses import replace

    from ..validation.ray_paths import NOT_FOLLOWED, validate_with_ray_paths
    from .validity import BEAM_PARTIALLY_INTERCEPTED, element_poses, element_scope_issues

    request = layout.lowered.request
    project = request.project
    if project.control_spec is None:  # (the other sources' light is not needed: traced alone)
        kept = tuple(c for c in project.components if c.instance_id == source_id or (c.source is None and c.emitter is None))
        request = request.model_copy(update={"project": project.model_copy(update={"components": kept, "targets": ()})})
    result, ray_paths = validate_with_ray_paths(request)
    roots = [path for path in result.paths if path.source_id == source_id and path.parent_path_id is None]
    own = f"{first.instance_id}/"
    reached = [path for path in roots if path.termination == BLOCKED_BACK and (path.termination_surface_id or "").startswith(own)
               and path.final_power_W is not None]
    met: list[str] = []
    for path in roots:
        for event in path.lineage:
            if not event.startswith(own) and not event.startswith(OBSERVATION_PREFIX) and event not in met:
                met.append(event)
    arriving = sum_left([path.final_power_W for path in reached])
    arrives = bool(roots) and len(reached) == len(roots)
    ends = sorted({f"{path.termination} at {path.termination_surface_id}" for path in roots if path not in reached}) or (["no traced path"] if not roots else [])
    record = {"emitted_W": emitted, "arriving_W": arriving, "path_ids": [path.path_id for path in reached],
              "lineage": list(reached[0].lineage) if reached else [], "met": list(met)}
    if ends:
        record["not_arriving"] = ends
    found: list[str] = []
    if arrives:
        # the elements' model conditions on the source's own paths (its pupil rays: part of the beam meeting an element beside the
        # chief ray, light leaving an element through its edge, ...): the chief ray's power and beam stand for the beam only then
        root_ids = {path.path_id for path in roots}
        own_rays = tuple(rp for rp in ray_paths.paths if rp.path_id in root_ids)
        issues = element_scope_issues(result.model_copy(update={"paths": tuple(roots)}), layout.lowered.elements,
                                      ray_paths=replace(ray_paths, paths=own_rays, sampled=None, scattered=None), poses=element_poses(layout.lowered),
                                      tabulated_domains=False)
        if issues:
            met_by: dict[str, int] = {}
            samples = 0
            for rp in own_rays:
                samples += rp.pupil_ray_count
                for ray in rp.pupil_rays:
                    if ray.end == NOT_FOLLOWED and ray.counted_with_chief and ray.next_surface_id is not None \
                            and not ray.next_surface_id.startswith(OBSERVATION_PREFIX):
                        met_by[ray.next_surface_id] = met_by.get(ray.next_surface_id, 0) + 1
            for issue in issues:
                text = f"{issue.code} at {issue.instance_id}" + (f" ({issue.surface_id})" if issue.surface_id else "")
                if issue.code == BEAM_PARTIALLY_INTERCEPTED and samples:
                    count = sum(n for surface, n in met_by.items() if surface.startswith(f"{issue.instance_id}/"))
                    if count:
                        text += (f": {count} of its {samples} equal-power pupil samples meet it while the chief ray does not, about "
                                 f"{count / samples:.3g} of the power ({arriving * count / samples:.6g} W)")
                found.append(text)
            record["model_conditions"] = [{"code": str(issue.code), "instance_id": issue.instance_id, "surface_id": issue.surface_id} for issue in issues]
    parts = []
    named = ", ".join(event.replace(":", " (") + ")" for event in met)
    if met and arrives:
        parts.append(f"the light of source {source_id!r} meets {named} before the input mirror {first.instance_id!r} (the trace delivers "
                     f"{arriving:.6g} W of its {emitted:.6g} W to {first.instance_id!r}): the beam arriving there is no longer the source's")
    if not arrives:
        parts.append(f"the traced light of source {source_id!r} does not reach the back of {first.instance_id!r} ({'; '.join(ends)}"
                     + (f"; it meets {named}" if met else "") + ")")
    if found:
        parts.append(f"the beam of source {source_id!r} on its way to the input mirror {first.instance_id!r} is outside the elements' model: "
                     f"{'; '.join(found)}. The trace's chief ray delivers {arriving:.6g} W of its {emitted:.6g} W, which counts the whole "
                     "beam; the beam that arrives is not the source's Gaussian beam, so neither its power nor its mode matching is known")
    if not parts:
        return arriving, record, None
    problem = "; ".join(parts) + ". State the power and the beam that arrive (injection power_w with beam INPUT_BEAM and input_beam)"
    return (arriving if arrives and not found else None), record, problem


def _injected_source(project: Any, source_id: str, layout: Any, wavelength_nm: float, wavelength_mm: float) -> dict[str, Any]:
    """The Project's source that feeds the cavity through its first mirror (module docstring): its arriving power, linewidth, kind, beam
    at the reference plane (``q`` per section) and polarization in the section frame, its light traced to the mirror.

    A request that contradicts the Project is refused: a source the Project does not hold, a ``wavelength_nm`` that is not the source's
    line. Everything that concerns how the light reaches the mirror is a ``problem`` ``(code, message)`` of the record instead (the
    injection quantities that need the beam then say so; the cavity's own quantities stand): a source that is not a single-line
    Gaussian beam, one that is not on the axis behind the first mirror, something met on the way, light that does not arrive, a model
    condition failing on the way, a substrate or astigmatic axes the beam is not carried through. ``power_w`` is ``None`` where the
    trace does not give the arriving power."""

    first = layout.placed[0]
    component = next((c for c in load_project(project).components if c.instance_id == source_id), None)
    if component is None or component.source_model is None:
        raise _cavity_refuse(CAVITY_SOURCE_UNSUPPORTED, f"no source {source_id!r} in the Project", "injection.source_id")
    model = component.source_model
    blank = {"source_id": source_id, "power_w": None, "emitted_w": None if model.power_w is None else float(model.power_w), "linewidth_Hz": None,
             "pulsed": False, "m2": 1.0, "q": None, "field": None, "degree_of_polarization": 1.0,
             "arrival": {"emitted_W": None if model.power_w is None else float(model.power_w), "arriving_W": None, "path_ids": [], "lineage": [], "met": []}}
    if model.model_kind != "GAUSSIAN_BEAM" or model.wavelength_nm is None or model.power_w is None:
        return {**blank, "problem": (CAVITY_SOURCE_UNSUPPORTED, f"source {source_id!r} is not a single-line Gaussian beam (wavelength_nm and power_w): "
                                     "the injection takes one line; state power_w instead for one line of it")}
    if abs(float(model.wavelength_nm) - wavelength_nm) > WAVELENGTH_MATCH_REL * wavelength_nm:
        raise _cavity_refuse(REQUEST_INVALID, f"wavelength_nm {wavelength_nm:g} differs from the line of source {source_id!r} "
                             f"({float(model.wavelength_nm):g} nm)", "wavelength_nm")
    origin = tuple(float(v) for v in component.nominal_pose.position_mm)
    axes = _local_axes(component.nominal_pose.quaternion_wxyz)
    direction = axes[2]
    into = first.direction_out
    along, lateral = _lateral(origin, first.exit_point, into)
    problems: list[str] = []
    if layout.tolerance.differ(direction, into) or layout.tolerance.off(lateral) or along >= 0.0:
        where = f"{-along:.6g} mm behind the mirror" if along < 0.0 else f"{along:.6g} mm in front of its reflecting side, inside the cavity"
        problems.append(f"source {source_id!r} does not feed the cavity through {first.instance_id!r}: it lies "
                        f"{lateral:.6g} mm off the cavity's axis, {where}, emitting at a cosine {_dot(direction, into):.12g} to the light "
                        f"entering the cavity (it must lie on the axis behind the first listed mirror and emit along it; "
                        f"{layout.tolerance.lateral_text()}, {layout.tolerance.tilt_text()})")
    width_hz = None
    if model.linewidth_nm is not None:
        width_hz = C_MM_PER_S * float(model.linewidth_nm) / (wavelength_nm * wavelength_nm * NM_TO_MM)
    elif model.coherence_length_mm is not None:
        width_hz = C_MM_PER_S * (2.0 * SQRT_LN2 / math.pi) / float(model.coherence_length_mm)
    m2 = max(float(model.m2), float(model.m2_y if model.m2_y is not None else model.m2))
    arriving, arrival, on_the_way = _traced_arrival(layout, source_id, first, float(model.power_w))
    if on_the_way is not None:
        problems.append(on_the_way)
    stretch = layout.walk.stretches[first.stretch + 1]  # (the light entering the cavity)
    section_x, section_y = stretch.section_x, stretch.section_y
    # The source's waist in the surrounding medium is carried through the first mirror to its reflector's vertex.
    waist_z = along + float(model.waist_position_mm) * _dot(direction, into)
    surfaces = first.element.model.surfaces
    back_z = float(Decimal(surfaces[0].vertex_z_mm) - Decimal(surfaces[-1].vertex_z_mm))
    radii = (float(model.waist_radius_mm), float(model.waist_radius_y_mm if model.waist_radius_y_mm is not None else model.waist_radius_mm))
    if radii[0] != radii[1]:
        along_x = _dot(axes[0], section_x)
        if abs(along_x) <= CRYSTAL_AXIS_TOLERANCE:
            radii = (radii[1], radii[0])
        elif 1.0 - abs(along_x) > CRYSTAL_AXIS_TOLERANCE:
            problems.append(f"source {source_id!r} is an astigmatic beam whose axes lie at an angle to the cavity's section axes: its beam "
                            "in the cavity's Hermite-Gauss basis is not formed; turn the source or state power_w with input_beam")
    q = None
    if m2 <= 1.0 and not problems:
        try:
            matrix = _substrate_matrix(first.element, wavelength_nm, source_id, layout.lowered.request.configuration.numerical, layout.lowered.air)
        except CavityError as exc:  # (a substrate the beam is not carried through: the beam inside the cavity is not formed)
            problems.append(exc.problems[0]["message"])
        else:
            q = {s: _through(matrix, complex(back_z - waist_z, math.pi * w * w / wavelength_mm)) for s, w in zip(SECTIONS, radii)}
    jones, dop = source_polarization(model.polarization)
    (ex_re, ex_im), (ey_re, ey_im) = jones.ex, jones.ey
    ex, ey = complex(ex_re, ex_im), complex(ey_re, ey_im)
    field = (ex * _dot(axes[0], section_x) + ey * _dot(axes[1], section_x), ex * _dot(axes[0], section_y) + ey * _dot(axes[1], section_y))
    return {"source_id": source_id, "power_w": arriving, "emitted_w": float(model.power_w), "linewidth_Hz": width_hz,
            "pulsed": model.pulse is not None, "m2": m2, "q": q, "field": field, "degree_of_polarization": 1.0 if dop is None else dop,
            "arrival": arrival, "problem": (CAVITY_SOURCE_UNSUPPORTED, "; ".join(problems)) if problems else None}


def _power(z: complex) -> float:
    return z.real * z.real + z.imag * z.imag


def _family_fraction(source: Mapping[str, Any], axis: tuple[float, float], along_axis: bool) -> float:
    """The power fraction of the source's light in the eigen-polarization analysed: of the polarized part the share along ``axis`` (a
    unit vector in the section frame: a crystal's optic axis for ``EXTRAORDINARY``, the normal of the tilted mirrors' plane of incidence
    for ``S``) or across it, and half of the unpolarized part."""

    cx, cy = axis
    fx, fy = source["field"]
    along = fx * cx + fy * cy
    across = fy * cx - fx * cy
    total = _power(fx) + _power(fy)
    share = (_power(along) if along_axis else _power(across)) / total
    dop = source["degree_of_polarization"]
    return dop * share + (1.0 - dop) * 0.5


def _common_plane_normal(placed: Sequence[_Placed]) -> tuple[float, float, float] | None:
    """The unit normal the planes of incidence of the cavity's tilted mirrors share (``direction_in x direction_out`` of each, parallel
    within ``CRYSTAL_AXIS_TOLERANCE``); ``(0, 0, 0)`` where no mirror is tilted; ``None`` where the planes differ."""

    normal: tuple[float, float, float] | None = None
    for item in placed:
        if not item.reflector:
            continue
        a, b = item.direction_in, item.direction_out
        cross = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
        length = math.sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2])
        if length <= CRYSTAL_AXIS_TOLERANCE:
            continue  # (normal incidence)
        unit = (cross[0] / length, cross[1] / length, cross[2] / length)
        if normal is None:
            normal = unit
            continue
        turned = (normal[1] * unit[2] - normal[2] * unit[1], normal[2] * unit[0] - normal[0] * unit[2], normal[0] * unit[1] - normal[1] * unit[0])
        if math.sqrt(turned[0] * turned[0] + turned[1] * turned[1] + turned[2] * turned[2]) > CRYSTAL_AXIS_TOLERANCE:
            return None
    return (0.0, 0.0, 0.0) if normal is None else normal


def _changes_polarization(placed: Sequence[_Placed]) -> bool:
    """Whether an element between the end mirrors is a polarization element or a Faraday rotator (compound members included)."""

    for item in placed[1:-1]:
        for part in (item.members or (item,)):
            if part.element.model.model_kind in POLARIZATION_CHANGING_KINDS:
                return True
    return False


def _family_axis(layout: Any, crystals: _Crystals, mirror_polarization: str | None,
                 plane_normal: tuple[float, float, float] | None) -> tuple[str, tuple[float, float], bool] | None:
    """The polarization family the analysis is for (``_Context.family_axis``)."""

    if crystals.cuts:
        return None  # (the crystals' family: ``_Context.polarization`` and their optic axis)
    if mirror_polarization is None or plane_normal is None:
        return None
    stretch = layout.walk.stretches[layout.placed[0].stretch + 1]  # (the light entering the cavity)
    cx, cy = _dot(plane_normal, stretch.section_x), _dot(plane_normal, stretch.section_y)
    length = math.sqrt(cx * cx + cy * cy)
    return mirror_polarization, (cx / length, cy / length), mirror_polarization == "S"


def _section_profile(ctx: _Context, s: str) -> tuple[dict[str, float], float]:
    """Per element the mode radius of section ``s`` (the larger of its entrance and exit) and the smallest Rayleigh range of the mode
    along the forward pass."""

    placed, q = ctx.placed, ctx.modes[s]["q"]
    radii = {placed[0].instance_id: _radius(q, ctx.wavelength_mm)}
    rayleigh = []
    for k in range(1, len(placed)):
        rayleigh.append(q.imag)
        q = q + ctx.gaps[k - 1]
        entrance = _radius(q, ctx.wavelength_mm)
        q = _through(ctx.matrices[k][s], q) if k < len(placed) - 1 else q
        radii[placed[k].instance_id] = max(entrance, _radius(q, ctx.wavelength_mm))
    return radii, min(rayleigh)


def _order_limit(ctx: _Context) -> tuple[int, str]:
    """The highest transverse order ``N`` the model resolves (module docstring of ``cavity_build_up``; -1: none) and what limits it."""

    cfg = ctx.cfg
    widest: dict[str, float] = {}
    theta0 = 0.0
    for s in SECTIONS:
        radii, rayleigh = _section_profile(ctx, s)
        for key, value in radii.items():
            widest[key] = max(widest.get(key, 0.0), value)
        theta0 = max(theta0, math.sqrt(ctx.wavelength_mm / (math.pi * rayleigh)))
    limits = [((cfg.gaussian_scope_divergence_rad / theta0) ** 2 - 1.0, "the far-field half angle sqrt(N + 1) theta_0")]
    for item in ctx.placed:
        clear = ctx.clear.get(item.instance_id)
        if clear is not None:
            limits.append(((clear / (ctx.aperture_min * widest[item.instance_id])) ** 2 - 1.0, f"the clear aperture of {item.instance_id!r}"))
        surface = reflecting_surface(item.element) if item.reflector else None
        if surface is not None and surface.radius_mm is not None:
            ratio = widest[item.instance_id] / abs(float(Decimal(surface.radius_mm)))
            limits.append(((cfg.gaussian_max_w_over_radius / ratio) ** 2 - 1.0, f"the mode radius over the radius of curvature of {item.instance_id!r}"))
    value, what = min(limits, key=lambda pair: pair[0])
    return (-1 if value < 0.0 else int(min(value, float(MAX_MODE_ORDER)))), what


INJECTION_QUANTITIES = (
    ("circulating_power_W", "power of the field circulating in the cavity: the forward field leaving the input mirror", "W"),
    ("transmitted_power_W", "power transmitted through the last mirror", "W"),
    ("reflected_power_W", "power leaving back through the input mirror: its prompt reflection and the leakage of the stored field", "W"),
    ("lost_power_W", "power lost in the mirrors (their absorption and scatter), at the fold mirrors and inside the cavity", "W"),
    ("power_build_up", "circulating power over the injected power (all of it, whatever share the polarization family carries)", None),
    ("transmission", "transmitted power over injected power", None),
    ("reflection", "reflected power over injected power", None),
)
RESONANT_QUANTITIES = (
    ("resonant_power_build_up", "build-up of a mode-matched beam on resonance, per unit power in it: T1 / (1 - sqrt(R1 R2) F)^2", None),
    ("resonant_transmission", "transmission of a mode-matched beam on resonance: T1 T2 F / (1 - sqrt(R1 R2) F)^2", None),
    ("resonant_reflection", "reflection of a mode-matched beam on resonance", None),
    ("impedance_matching_reflectance", "input-mirror reflectance that cancels the reflection on resonance with a lossless input mirror: R2 F^2", None),
    ("input_loss_matching_reflectance", "input-mirror reflectance that cancels the reflection on resonance with the input mirror's own loss A1: "
                                        "(1 - A1)^2 R2 F^2", None),
)
#: The quantities whose interval cannot exceed the power the analysis follows (a ratio: that power over the injected power).
_CAPPED = frozenset({"transmitted_power_W", "reflected_power_W", "lost_power_W", "transmission", "reflection"})
_DETUNING = ("detuning_Hz", "detuning of the source from the nearest TEM00 resonance", "Hz")
#: The channels with TEM00 on resonance reported as quantities where the detuning is not stated (``at_tem00_resonance.<channel>``).
_RESONANCE_CHANNELS = (
    ("circulating_W", "power of the field circulating in the cavity"),
    ("at_output_mirror_W", "power arriving at the last mirror"),
    ("transmitted_W", "power transmitted through the last mirror"),
    ("reflected_W", "power leaving back through the input mirror"),
    ("input_mirror_loss_W", "power lost in the input mirror"),
    ("output_mirror_loss_W", "power lost in the last mirror"),
    ("inside_loss_W", "power lost at the fold mirrors and inside the cavity"),
)
_LOSS_CHANNELS = ("input_mirror_loss_W", "output_mirror_loss_W", "inside_loss_W")


def _channels(coupler: Coupler, content: Sequence[tuple[int, int, float]], power: float, delta: float, theta_x: float,
              theta_y: float) -> list[dict[str, float]]:
    """``steady_state`` of each resolved mode ``(m, n, fraction)`` at the TEM00 round-trip phase ``delta``."""

    return [steady_state(coupler, fraction * power, delta - (m * theta_x + n * theta_y)) for m, n, fraction in content]


def _inject(ctx: _Context, out: _Quantities) -> tuple[dict[str, Any], tuple[ValidityCheck, ...]]:  # noqa: C901 - one flat sequence of model checks
    """The injection quantities (``cavity_build_up``) into ``out``; returns the injection arrays and the validity conditions checked for
    the injected light. Three groups, each with its own inputs: the resonant quantities and the coupling read the mirrors' coefficients
    alone; the detuning is the stated one; the seven powers read the source and the beam as well, and a failed source or beam condition
    selects the proven bound that still holds."""

    inj = ctx.extra.injection
    assert inj is not None
    placed = ctx.placed
    first, last = placed[0], placed[-1]
    source = ctx.source
    power = source["power_w"] if source is not None else inj["power_w"]
    unsupported = None if source is None else source["problem"]
    if source is not None and power is None:
        assert unsupported is not None
        out.add("injected_power_W", "power injected into the cavity through its first mirror (arriving at it from outside)", "W", None,
                basis=MODEL_INJECTION, state=QuantityState.OUT_OF_MODEL_SCOPE, reason=unsupported[0], message=unsupported[1],
                source=f"source {source['source_id']}")
    elif source is not None:
        arrival = source["arrival"]
        out.add("injected_power_W", "power injected into the cavity through its first mirror (arriving at it from outside)", "W", power,
                basis=MODEL_INJECTION, source=f"source {source['source_id']}",
                message=f"traced from source {source['source_id']} to {first.instance_id}: {arrival['arriving_W']:.9g} W of its {arrival['emitted_W']:.9g} W arrive")
    else:
        out.add("injected_power_W", "power injected into the cavity through its first mirror (arriving at it from outside)", "W", power,
                basis=MODEL_INJECTION, origin=QuantityOrigin.EXTERNALLY_CHARACTERIZED, source="the request's injection.power_w")
    record: dict[str, Any] = {"beam": inj["beam"], "power_W": power, "input_mirror": first.instance_id, "output_mirror": last.instance_id,
                              "source_id": None if source is None else source["source_id"]}
    if source is not None:
        record["traced_arrival"] = source["arrival"]
        if unsupported is not None:
            record["source_problem"] = {"code": unsupported[0], "message": unsupported[1]}
        if source["q"] is not None:
            beam = {}
            for s, axis_name in (("X", "x"), ("Y", "y")):
                waist, after = _waist_of(source["q"][s], ctx.wavelength_mm)
                beam[f"waist_radius_{axis_name}_mm"], beam[f"waist_after_reference_{axis_name}_mm"] = waist, after
            record["source_beam"] = beam
    t_values = {}
    for item in (first, last):
        found_t = _transmission(item, ctx.stated, ctx.extra.transmittances, ctx.wavelength_nm, ctx.cfg)
        external = found_t.origin is QuantityOrigin.EXTERNALLY_CHARACTERIZED
        state = None
        if found_t.interval is not None:
            state = QuantityState.BOUNDED  # (a stated reflectance leaves 0 <= T <= 1 - R)
        elif found_t.entry is not None:
            state = QuantityState.UNKNOWN  # (held outside the coating's stated data: an estimate)
        out.add(f"transmittance.{item.instance_id}", f"power transmittance of {item.instance_id}", None, found_t.value, basis=MODEL_INJECTION,
                origin=found_t.origin, state=state, interval=found_t.interval, reason=found_t.reason, message=found_t.message,
                missing=found_t.missing, value_basis=found_t.entry, source="the request's transmittances" if external else None)
        # (the powers wait for a transmittance that is not known or is held outside its data)
        t_values[item.instance_id] = (found_t.value if found_t.entry is None else None, found_t.reason, found_t.message)

    checks: list[ValidityCheck] = []

    def withhold(state: QuantityState, reason: str | None, message: str) -> tuple[dict[str, Any], tuple[ValidityCheck, ...]]:
        for quantity_id, name, unit in (*RESONANT_QUANTITIES, *INJECTION_QUANTITIES, _DETUNING):
            out.add(quantity_id, name, unit, None, basis=MODEL_INJECTION, state=state, reason=reason, message=message)
        return {**record, "reason": reason, "message": message}, ()

    # no eigenmode: nothing for the light to build up in
    if not ctx.mode_exists:
        return withhold(QuantityState.OUT_OF_MODEL_SCOPE, ctx.reason, ctx.why)
    missing = [s for s in SECTIONS if ctx.modes[s]["q"] is None]
    if missing:
        return withhold(QuantityState.OUT_OF_MODEL_SCOPE, INPUT_SECTION_HAS_NO_MODE, f"section {missing[0]} of the cavity has no Gaussian eigenmode")

    # ---- the mirrors' coefficients: a quantity waits only for the ones it reads (the request field that states a missing one is named)
    values: dict[str, float] = {}
    lacking: dict[str, tuple[str | None, str, str]] = {}
    for role, item in (("R1", first), ("R2", last)):
        r, r_reason, r_message = ctx.reflectance_of[item.instance_id]
        if r is None:
            lacking[role] = (r_reason, r_message, f"reflectances.{item.instance_id}")
        else:
            values[role] = r
    folds = 1.0
    for item in placed[1:-1]:
        if item.reflector:
            r, r_reason, r_message = ctx.reflectance_of[item.instance_id]
            if r is None:
                lacking.setdefault("folds", (r_reason, r_message, f"reflectances.{item.instance_id}"))
            else:
                folds *= r
    for role, item in (("T1", first), ("T2", last)):
        t, t_reason, t_message = t_values[item.instance_id]
        if t is None:
            lacking[role] = (t_reason, t_message, f"transmittances.{item.instance_id}")
        else:
            values[role] = t
    for r_role, t_role, item in (("R1", "T1", first), ("R2", "T2", last)):
        if r_role in values and t_role in values and values[r_role] + values[t_role] > 1.0 + REFLECTANCE_EQUAL_ABS:
            del values[t_role]
            lacking[t_role] = (TRANSMITTANCE_NOT_STATED, f"{item.instance_id!r}: R + T is above 1; state both in reflectances and transmittances",
                               f"transmittances.{item.instance_id}")

    def first_lacking(*roles: str) -> tuple[str | None, str, str] | None:
        return next((lacking[role] for role in ("R1", "folds", "R2", "T1", "T2") if role in roles and role in lacking), None)

    # losses no input states: those of the elements between the mirrors without a round_trip_loss, and the edges of the apertures
    # that reach the mode (whatever round_trip_loss says: it states the elements' loss, not the mode's clipping)
    loss = ctx.extra.round_trip_loss
    lossy = [item.instance_id for item in placed[1:-1] if not item.reflector] if loss is None else []
    losses_known = not lossy and not ctx.clipped_at
    passing = folds * math.sqrt(1.0 - (loss or 0.0))
    loss_message = "; ".join(x for x in (
        f"the losses of {', '.join(repr(i) for i in lossy)} are not stated (round_trip_loss): bounded for any loss" if lossy else "",
        f"the clear aperture of {', '.join(repr(i) for i in ctx.clipped_at)} reaches the mode and adds a loss that is not stated: bounded for any loss"
        if ctx.clipped_at else "") if x)
    loss_missing = ("round_trip_loss",) if lossy else ()

    fractions = (("higher_order_mode_power_fraction", "power fraction of the injected beam in the resolved higher-order modes HG_mn, m + n > 0"),
                 ("unresolved_mode_power_fraction", "power fraction of the injected beam in modes that are not resolved"))
    if power is None:
        # the trace does not give the power that arrives: nothing bounds the light in the cavity; the mirrors' quantities stand below
        assert unsupported is not None
        record.update(reason=unsupported[0], message=unsupported[1])
    # the power in the eigen-polarization family analysed (a crystal in the cavity: the source's polarization divides it)
    family = 1.0
    if source is not None and source["field"] is not None and ctx.crystals.cuts and ctx.polarization is not None:
        family = _family_fraction(source, ctx.crystals.cuts[0].section_axis, ctx.polarization == "EXTRAORDINARY")
        out.add("polarization_family_power_fraction", f"power fraction of the source's light in the cavity's {ctx.polarization} eigen-polarization "
                "(along the crystals' optic axis: extraordinary; across it: ordinary)", None, family, basis=MODEL_INJECTION,
                depends_on=("injected_power_W",))
    elif source is not None and source["field"] is not None and ctx.family_axis is not None:
        named, axis, along_axis = ctx.family_axis
        family = _family_fraction(source, axis, along_axis)
        out.add("polarization_family_power_fraction", f"power fraction of the source's light in the cavity's {named} eigen-polarization "
                "(perpendicular to the tilted mirrors' plane of incidence: S; in it: P)", None, family, basis=MODEL_INJECTION,
                depends_on=("injected_power_W",))
    followed = None if power is None else power * family
    record["polarization_family_power_fraction"] = family

    # ---- on resonance, a mode-matched beam, and the coupling on resonance: the mirrors alone
    resonant: dict[str, float] = {}
    spans: dict[str, tuple[Interval, ValueBasis]] = {}
    no_loss = (*((Assumed(input=AssumedInput.NO_UNSTATED_LOSS, at=tuple(lossy)),) if lossy else ()),
               *((Assumed(input=AssumedInput.NO_UNSTATED_LOSS, at=ctx.clipped_at, code=MODE_CLIPPED_BY_APERTURE),) if ctx.clipped_at else ()))
    upper = ValueBasis(kind=ValueKind.UPPER_BOUND, assumed=no_loss) if not losses_known else None

    def falls_with_loss(quantity_id: str, value: float, least: float) -> None:
        resonant[quantity_id] = value
        if upper is not None:
            spans[quantity_id] = (Interval(lower=min(least, value), upper=value, kind=IntervalKind.PROVEN), upper)

    if first_lacking("folds", "R2") is None:
        returned = math.sqrt(values["R2"]) * passing
        falls_with_loss("impedance_matching_reflectance", returned * returned, 0.0)
    if first_lacking("R1", "folds", "R2", "T1") is None:
        mirrors = Coupler(r1=values["R1"], t1=values["T1"], r2=values["R2"], t2=values.get("T2", 0.0), passing=passing)
        a, b = mirrors.gain, mirrors.b
        enhancement = 1.0 / ((1.0 - a) * (1.0 - a))
        keep = 1.0 - mirrors.a1
        on = math.sqrt(mirrors.r1) - keep * b
        falls_with_loss("resonant_power_build_up", mirrors.t1 * enhancement, mirrors.t1)
        if "T2" in values:
            falls_with_loss("resonant_transmission", mirrors.t1 * mirrors.t2 * passing * enhancement, 0.0)
        falls_with_loss("input_loss_matching_reflectance", keep * keep * b * b, 0.0)
        reflection = on * on * enhancement if mirrors.t1 > 0.0 else mirrors.r1
        resonant["resonant_reflection"] = reflection
        if not losses_known:
            # the field reflected on resonance, (r1 - (1 - A1) b') / (1 - r1 b'), falls from r1 (b' = 0, nothing returns) to its value
            # without the unstated loss: the reflection lies between that value's square and R1 while the field stays positive
            # (under-coupled), and between 0 and the larger of the two once it changes sign
            under = on >= 0.0 or mirrors.t1 == 0.0
            spans["resonant_reflection"] = (Interval(lower=reflection if under else 0.0, upper=max(mirrors.r1, reflection), kind=IntervalKind.PROVEN),
                                            ValueBasis(kind=ValueKind.LOWER_BOUND if under else ValueKind.ESTIMATE, assumed=no_loss))
        if losses_known:
            regime = None
            if mirrors.t1 > 0.0:
                regime = "CRITICAL" if abs(on) <= CRITICAL_COUPLING_ABS else ("UNDER_COUPLED" if on > 0.0 else "OVER_COUPLED")
            record["coupling"] = {"regime": regime, "input_field_reflection": math.sqrt(mirrors.r1), "returned_field_leak": keep * b,
                                  "matching_reflectance_lossless_input": b * b, "matching_reflectance_at_input_loss": keep * keep * b * b}
    reads = {"resonant_power_build_up": ("R1", "folds", "R2", "T1"), "resonant_transmission": ("R1", "folds", "R2", "T1", "T2"),
             "resonant_reflection": ("R1", "folds", "R2", "T1"), "impedance_matching_reflectance": ("folds", "R2"),
             "input_loss_matching_reflectance": ("R1", "folds", "R2", "T1")}
    for quantity_id, name, unit in RESONANT_QUANTITIES:
        gap = first_lacking(*reads[quantity_id])
        if gap is not None:
            out.add(quantity_id, name, unit, None, basis=MODEL_INJECTION, reason=gap[0], message=gap[1], missing=(gap[2],))
        elif losses_known:
            out.add(quantity_id, name, unit, resonant[quantity_id], basis=MODEL_INJECTION)
        else:
            # (the value without the unstated loss, inside the interval that holds for any such loss, 0 <= F' <= F)
            interval, entry = spans[quantity_id]
            out.add(quantity_id, name, unit, resonant[quantity_id], basis=MODEL_INJECTION, state=QuantityState.BOUNDED, interval=interval,
                    value_basis=entry, reason=LOSSES_NOT_MODELED, message=loss_message, missing=loss_missing)

    # ---- the detuning of TEM00: stated, or from the stated length offset
    theta_x, theta_y = ctx.modes["X"]["theta"], ctx.modes["Y"]["theta"]
    delta, detuning, detuning_reason, detuning_message = None, None, None, ""
    #: The round-trip phase of a stated detuning where the free spectral range is an estimate: the powers' values are estimates at
    #: it, inside their bounds over every detuning.
    estimated_phase = None
    detuning_entry = None
    if inj["detuning_Hz"] is not None:
        detuning = inj["detuning_Hz"]
        if ctx.fsr is None:
            detuning_reason, detuning_message = ctx.path_reason, ctx.path_message
        elif ctx.fsr_entry is not None and detuning != 0.0:
            estimated_phase = TWO_PI * detuning / ctx.fsr
            detuning_reason, detuning_message = ctx.path_reason, ctx.path_message
        else:
            delta = TWO_PI * detuning / ctx.fsr
        record["detuning_from"] = "STATED"
    elif inj["length_offset_nm"] is not None:
        waves = 2.0 * ctx.phase_index * inj["length_offset_nm"] / ctx.wavelength_nm  # (phase path of the round-trip displacement, in vacuum wavelengths)
        delta = TWO_PI * (waves - round(waves))
        detuning = None if ctx.fsr is None else delta * ctx.fsr / TWO_PI
        if detuning is None or ctx.fsr_entry is not None:
            detuning_reason, detuning_message, detuning_entry = ctx.path_reason, ctx.path_message, ctx.fsr_entry
        record["detuning_from"] = "LENGTH_OFFSET"
    else:
        record["detuning_from"] = "NOT_STATED"
        detuning_reason = DETUNING_NOT_STATED
        detuning_message = (f"the detuning is not stated, and a Project's positions do not fix it: the round-trip phase turns by 2 pi per "
                            f"{ctx.wavelength_nm / (2.0 * ctx.phase_index):g} nm of cavity length in the surrounding medium; state "
                            "injection.detuning_Hz or injection.length_offset_nm. The powers "
                            "are bounded over every detuning")
    stated = inj["detuning_Hz"] is not None
    out.add(*_DETUNING, detuning, basis=MODEL_INJECTION, origin=QuantityOrigin.EXTERNALLY_CHARACTERIZED if stated else QuantityOrigin.MODELED,
            state=None if detuning_entry is None or detuning is None else QuantityState.UNKNOWN, value_basis=detuning_entry,
            reason=detuning_reason, message=detuning_message or ("stated in the request" if stated else
                                                                 "from the stated length offset from the TEM00 resonance"))
    record["round_trip_phase_rad"] = delta

    # ---- the source and the beam: a failed condition selects the proven bound that still holds
    # not a single frequency: every spectral component sits at some detuning and the cavity is linear, so the time-averaged powers lie
    # in the bounds over every detuning
    fraction_max = inj["source_linewidth_fraction_max"]
    spectral = ""
    if source is not None and source["pulsed"]:
        spectral = ("the source is pulsed: its spectrum is a comb of lines, each at its own detuning, not the single frequency the Airy response "
                    "is for")
    elif source is not None and source["linewidth_Hz"] is not None:
        # (single frequency is proven where the source's linewidth is within the fraction of the least the cavity's can be)
        least = ctx.linewidth_lower
        narrow = least is not None and source["linewidth_Hz"] <= fraction_max * least
        checks.append(_condition("source_linewidth_over_cavity_linewidth", None if not least else source["linewidth_Hz"] / least, maximum=fraction_max,
                                 within=None if least is None else narrow, setting="injection.source_linewidth_fraction_max",
                                 default=SOURCE_LINEWIDTH_FRACTION, code=SOURCE_NOT_SINGLE_FREQUENCY))
        if not narrow and ctx.linewidth is None:
            spectral = ("the source states a linewidth and the cavity's linewidth is not known: the Airy response of a single frequency may not "
                        "apply")
        elif not narrow:
            spectral = (f"the source's linewidth {source['linewidth_Hz']:.6g} Hz is above {fraction_max:g} of the cavity's "
                        f"{ctx.linewidth:.6g} Hz: the transmitted power is the Airy function averaged over its spectrum")
    # a transverse content the model does not decompose: all the light is 'not resolved', and nothing says how much of it enters
    whole: list[tuple[str, str]] = []
    if ctx.reason is not None:
        whole.append((ctx.reason, ctx.why))  # (the eigenmode fails a validity condition: no mode of it is resolved)
    if source is not None and source["m2"] > 1.0:
        checks.append(_condition("source_m2", source["m2"], maximum=1.0, within=False, code=SOURCE_NOT_FUNDAMENTAL_MODE))
        whole.append((SOURCE_NOT_FUNDAMENTAL_MODE,
                      f"the source's M^2 is {source['m2']:g}: a higher-order or multimode transverse beam, whose content in the cavity's "
                      "Hermite-Gauss modes this model does not decompose (it takes a TEM00 Gaussian beam, M^2 = 1)"))
    if ctx.q_input is not None and inj["beam"] != "EIGENMODE":
        widest = max(_radius(ctx.q_input[s], ctx.wavelength_mm) for s in SECTIONS)
        clear = ctx.clear.get(first.instance_id)
        if clear is not None:
            cut = clear < ctx.aperture_min * widest
            checks.append(_condition("input_mirror_clear_radius_over_beam_radius", clear / widest, minimum=ctx.aperture_min, within=not cut,
                                     setting="aperture_over_mode_min", default=APERTURE_OVER_MODE_MIN, at=(first.instance_id,),
                                     code=INPUT_BEAM_CLIPPED))
            if cut:
                whole.append((INPUT_BEAM_CLIPPED,
                              f"the injected beam's radius at {first.instance_id!r} is {widest:.6g} mm and its clear radius {clear:.6g} mm is "
                              f"{clear / widest:.6g} beam radii (at least {ctx.aperture_min:g}): the mirror's edge cuts the beam, whose "
                              "transverse content then is not the Gaussian beam's"))
    if unsupported is not None:
        whole.insert(0, unsupported)  # (the beam at the mirror is not the source's: nothing of it is decomposed)
    cause = whole[0][0] if whole else (SOURCE_NOT_SINGLE_FREQUENCY if spectral else None)
    if followed is None:
        assert unsupported is not None
        for quantity_id, name in fractions:
            out.add(quantity_id, name, None, None, basis=MODEL_INJECTION, state=QuantityState.OUT_OF_MODEL_SCOPE, reason=unsupported[0], message=unsupported[1])
        for quantity_id, name, unit in INJECTION_QUANTITIES:
            out.add(quantity_id, name, unit, None, basis=MODEL_INJECTION, state=QuantityState.OUT_OF_MODEL_SCOPE, reason=unsupported[0],
                    message=unsupported[1], depends_on=("injected_power_W",))
        return record, tuple(checks)

    # ---- the transverse content
    order_limit, limited_by = _order_limit(ctx)
    resolved_order = -1 if whole else min(inj["mode_order_max"], order_limit)
    record.update(resolved_order=resolved_order, order_limit=order_limit, order_limited_by=limited_by)
    etas = None
    if inj["beam"] == "EIGENMODE":
        etas = (1.0, 1.0)
        content = [(0, 0, 1.0)] if resolved_order >= 0 else []
        unresolved = 0.0 if resolved_order >= 0 else 1.0
    elif ctx.q_input is not None:
        q_in = ctx.q_input
        etas = (_overlap_1d(q_in["X"], ctx.modes["X"]["q"]), _overlap_1d(q_in["Y"], ctx.modes["Y"]["q"]))
        content = mode_fractions(etas[0], etas[1], resolved_order) if resolved_order >= 0 else []
        unresolved = unresolved_fraction(etas[0], etas[1], resolved_order)
    else:
        content, unresolved = [], 1.0
    higher = sum_left([f for m, n, f in content if m + n > 0])
    if unsupported is not None:
        # (the beam that arrives is not known: neither is its content in the cavity's modes; its power is bounded as light not resolved)
        for quantity_id, name in fractions:
            out.add(quantity_id, name, None, None, basis=MODEL_INJECTION, state=QuantityState.OUT_OF_MODEL_SCOPE, reason=unsupported[0], message=unsupported[1])
    else:
        out.add("higher_order_mode_power_fraction", fractions[0][1], None, higher, basis=MODEL_INJECTION)
        out.add("unresolved_mode_power_fraction", f"power fraction of the injected beam in modes above order {resolved_order} (not resolved)", None, unresolved,
                basis=MODEL_INJECTION)
    fsr = ctx.fsr
    sides = ("injected_power_W", "detuning_Hz", "unresolved_mode_power_fraction")
    gap = first_lacking("R1", "folds", "R2", "T1", "T2")
    coupler = None if gap is not None else Coupler(r1=values["R1"], t1=values["T1"], r2=values["R2"], t2=values["T2"], passing=passing)
    #: The TEM00 round-trip phase the powers are evaluated at: none where the light is not of one frequency.
    at_phase = None if spectral else delta
    evaluated = coupler is not None and at_phase is not None and losses_known
    states = _channels(coupler, content, followed, at_phase, theta_x, theta_y) if evaluated else []
    rows = []
    for k, (m, n, fraction) in enumerate(content):
        shift = m * theta_x + n * theta_y
        row: dict[str, Any] = {"m": m, "n": n, "order": m + n, "power_fraction": fraction, "gouy_shift_rad": shift % TWO_PI,
                               "offset_from_tem00_Hz": None if fsr is None else (shift % TWO_PI) * fsr / TWO_PI}
        if evaluated:
            row["round_trip_phase_rad"] = at_phase - shift
            row.update(states[k])
        rows.append(row)
    record["modes"] = rows
    if whole:
        record["not_decomposed"] = [code for code, _message in whole]
    if inj["beam"] != "EIGENMODE" and etas is not None:
        record["eta_x"], record["eta_y"] = etas
        if abs(etas[0] - etas[1]) <= SYMMETRY_REL and _close(theta_x, theta_y) and resolved_order >= 0:
            fractions = lg_fractions(etas[0] * etas[1], resolved_order // 2)
            record["lg_modes"] = [{"p": p, "l": 0, "order": 2 * p, "power_fraction": f,
                                   "offset_from_tem00_Hz": None if fsr is None else ((2 * p * theta_x) % TWO_PI) * fsr / TWO_PI}
                                  for p, f in enumerate(fractions)]
    scan = inj["scan"]
    if coupler is None:
        # a coefficient the powers read is not known: they wait for it; the content above and the quantities that do not read it stand
        assert gap is not None
        for quantity_id, name, unit in INJECTION_QUANTITIES:
            out.add(quantity_id, name, unit, None, basis=MODEL_INJECTION, reason=gap[0], message=gap[1], missing=(gap[2],), depends_on=sides)
        if scan is not None:
            record["scan"] = {"reason": gap[0], "message": gap[1]}
        return record, tuple(checks)
    a, b = coupler.gain, coupler.b
    record["coupler"] = {"R1": coupler.r1, "T1": coupler.t1, "A1": coupler.a1, "R2": coupler.r2, "T2": coupler.t2, "A2": coupler.a2,
                         "pass_power_fraction": passing, "round_trip_field_gain": a, "losses_known": losses_known}

    # the resolved modes' channels: at the detuning, or bounded over every detuning (an Airy function is monotonic in sin^2 of half
    # its phase; the sum over the modes by branch and bound, cavity_build_up.swept_extremes)
    keys = ("circulating_W", "transmitted_W", "reflected_W", "lost_W")

    low = {key: 0.0 for key in keys}
    high = {key: 0.0 for key in keys}
    sweep = at_phase is None and losses_known

    def summed(found: Sequence[Mapping[str, float]]) -> dict[str, float]:
        totals = {key: sum_left([state[key] for state in found]) for key in keys[:3]}
        totals["lost_W"] = (sum_left([state["input_mirror_loss_W"] for state in found]) + sum_left([state["output_mirror_loss_W"] for state in found])
                            + sum_left([state["inside_loss_W"] for state in found]))
        return totals

    #: The resolved modes' channels at the detuning: the value of each power (with unstated losses: the value without them).
    resolved: dict[str, float] | None = None
    if evaluated:
        resolved = summed(states)
        for key in keys:
            low[key] = high[key] = resolved[key]
    elif at_phase is not None and content:
        resolved = summed(_channels(coupler, content, followed, at_phase, theta_x, theta_y))
    elif estimated_phase is not None and content and not spectral:
        resolved = summed(_channels(coupler, content, followed, estimated_phase, theta_x, theta_y))
    if not content:
        resolved = None  # (nothing resolved: no number beside the bounds)
    if sweep:
        # (the sum over the modes at each detuning: branch and bound over the TEM00 round-trip phase, cavity_build_up.swept_extremes)
        powers = [f * followed for _m, _n, f in content]
        shifts = [m * theta_x + n * theta_y for m, n, _f in content]
        for key in keys:
            low[key], high[key] = swept_extremes(coupler, powers, shifts, key)
        if not spectral:
            at_resonance = _channels(coupler, content, followed, 0.0, theta_x, theta_y)
            record["at_tem00_resonance"] = {**{key: sum_left([state[key] for state in at_resonance]) for key in CHANNELS},
                                            "not_resolved_W": followed * unresolved,
                                            "modes": [{"m": m, "n": n, **state} for (m, n, _f), state in zip(content, at_resonance)]}
    bounded_power = followed * (unresolved if losses_known else 1.0)
    bounds = unresolved_bounds(coupler, bounded_power)
    negligible_tail = losses_known and bounded_power <= ctx.negligible * power

    def with_tail(quantity_id: str, name: str, unit: str | None, value: float, lo: float, hi: float, depends: tuple[str, ...],
                  message: str) -> None:
        """A power whose resolved modes carry ``value`` and whose true value, with the light not resolved, lies in ``[lo, hi]``
        (that light only adds). Light not resolved below the negligible level leaves the value ``KNOWN``, the interval beside it
        and the most it can add as the value's resolution; above it the value is the lower bound of a ``BOUNDED`` quantity."""

        most = max(0.0, hi - value)
        interval = Interval(lower=lo, upper=hi, kind=IntervalKind.PROVEN)
        if negligible_tail:
            set_by_caller = ctx.negligible != DEFAULT_NUMERICAL_CONFIG.negligible_power_fraction
            entry = None if not set_by_caller else ValueBasis(kind=ValueKind.MODEL, omitted=(
                Omitted(part=OmittedPart.BELOW_NEGLIGIBLE_LEVEL, upper=most, setting="negligible_power_fraction"),))
            out.add(quantity_id, name, unit, value, basis=MODEL_INJECTION, interval=interval, resolution=max(most, value - lo), value_basis=entry,
                    depends_on=depends)
        else:
            out.add(quantity_id, name, unit, value, basis=MODEL_INJECTION, state=QuantityState.BOUNDED, interval=interval,
                    value_basis=ValueBasis(kind=ValueKind.LOWER_BOUND, omitted=(Omitted(part=OmittedPart.UNRESOLVED_MODES, upper=most),)),
                    message=message, depends_on=depends)

    if "at_tem00_resonance" in record:
        # the channels with TEM00 on resonance as quantities: the resolved modes' sums, and with them the light not resolved at any
        # round-trip phase (every loss channel rises with that light's circulating power)
        tail = followed * unresolved
        beside = unresolved_bounds(coupler, tail)
        c_lo, c_hi = beside["circulating_W"]
        beside.update({
            "at_output_mirror_W": (passing * c_lo, passing * c_hi),
            "input_mirror_loss_W": (coupler.a1 * (tail + b * b * c_lo), coupler.a1 * (tail + b * b * c_hi)),
            "output_mirror_loss_W": (coupler.a2 * passing * c_lo, coupler.a2 * passing * c_hi),
            "inside_loss_W": ((1.0 - passing) * (c_lo + coupler.r2 * passing * c_lo), (1.0 - passing) * (c_hi + coupler.r2 * passing * c_hi))})
        for key, what in _RESONANCE_CHANNELS:
            value = record["at_tem00_resonance"][key]
            name = f"{what} with TEM00 on resonance"
            depends = ("injected_power_W", "unresolved_mode_power_fraction")
            if tail == 0.0:
                out.add(f"at_tem00_resonance.{key}", name, "W", value, basis=MODEL_INJECTION, depends_on=depends)
            elif not content:
                out.add(f"at_tem00_resonance.{key}", name, "W", None, basis=MODEL_INJECTION, state=QuantityState.BOUNDED,
                        interval=Interval(lower=0.0 if whole else beside[key][0], upper=beside[key][1], kind=IntervalKind.PROVEN), reason=cause,
                        message="; ".join(message for _code, message in whole) or f"no mode is resolved ({limited_by})", depends_on=depends)
            else:
                with_tail(f"at_tem00_resonance.{key}", name, "W", value, value + beside[key][0], value + beside[key][1], depends,
                          f"a lower bound: {unresolved:.6g} of the injected power is in modes above order {resolved_order}, which only add")

    why_bounded = "; ".join(x for x in (
        *(message for _code, message in whole),
        f"{unresolved:.6g} of the injected power is in modes above order {resolved_order} ({limited_by} limits the order to {order_limit})"
        if unresolved > 0.0 and losses_known and not whole else "",
        "" if delta is not None else (
            f"the stated detuning's round-trip phase rests on a free spectral range that is an estimate ({detuning_reason}): bounded over every "
            "detuning" if estimated_phase is not None else f"the detuning is not known ({detuning_reason}): bounded over every detuning"),
        f"{spectral}: bounded over every detuning" if spectral else "",
        loss_message) if x)
    not_stated = (*(("injection.detuning_Hz",) if detuning_reason == DETUNING_NOT_STATED and not spectral else ()),
                  *loss_missing)

    def report(quantity_id: str, name: str, unit: str | None, lo: float, hi: float, value: float | None, tail_most: float,
               depends: tuple[str, ...]) -> None:
        """One injected power: its value at the detuning beside its proven interval ``[lo, hi]``. No single detuning, or no resolved
        mode: no value exists. Unstated losses: the value without them, an estimate inside the interval that holds for any loss."""

        interval = Interval(lower=lo, upper=hi, kind=IntervalKind.PROVEN)
        if lo == hi and bounded_power == 0.0:
            out.add(quantity_id, name, unit, lo, basis=MODEL_INJECTION, depends_on=depends)
        elif value is None:
            out.add(quantity_id, name, unit, None, basis=MODEL_INJECTION, state=QuantityState.BOUNDED, interval=interval, reason=cause,
                    message=f"proven bounds: {why_bounded}", missing=not_stated, depends_on=depends)
        elif losses_known and estimated_phase is None:
            with_tail(quantity_id, name, unit, value, lo, hi, depends, f"proven bounds: {why_bounded}")
        else:
            # (the value without the unstated loss, or at the phase an estimated free spectral range gives the stated detuning)
            omitted = (Omitted(part=OmittedPart.UNRESOLVED_MODES, upper=tail_most),) if unresolved > 0.0 else ()
            phase_assumed = ctx.fsr_entry.assumed if estimated_phase is not None and ctx.fsr_entry is not None else ()
            out.add(quantity_id, name, unit, value, basis=MODEL_INJECTION, state=QuantityState.BOUNDED, interval=interval,
                    value_basis=ValueBasis(kind=ValueKind.ESTIMATE, omitted=omitted, assumed=(*phase_assumed, *no_loss)),
                    message=f"proven bounds: {why_bounded}", missing=not_stated, depends_on=depends)

    extra_lost = (max(0.0, bounded_power - bounds["transmitted_W"][1] - bounds["reflected_W"][1]),
                  bounded_power - bounds["transmitted_W"][0] - bounds["reflected_W"][0])
    for quantity_id, name, unit in INJECTION_QUANTITIES:
        key = {"circulating_power_W": "circulating_W", "transmitted_power_W": "transmitted_W", "reflected_power_W": "reflected_W",
               "lost_power_W": "lost_W", "power_build_up": "circulating_W", "transmission": "transmitted_W", "reflection": "reflected_W"}[quantity_id]
        add_low, add_high = extra_lost if key == "lost_W" else bounds[key]
        lo, hi = low[key] + add_low, high[key] + add_high
        if bounded_power != 0.0 or sweep:
            lo = max(0.0, lo)
            if quantity_id in _CAPPED:
                hi = min(followed, hi)  # (a channel of the light followed never carries more than it)
        if whole:
            lo = 0.0  # (how much of a beam that is not decomposed enters the cavity is not known)
        scale = 1.0 if unit == "W" else 1.0 / power
        # (with unstated losses: the most the light not resolved can add to the value, for any loss)
        tail_most = (followed * unresolved if key == "lost_W" else unresolved_bounds(coupler, followed * unresolved)[key][1]) * scale
        report(quantity_id, name, unit, lo * scale, hi * scale, None if resolved is None else resolved[key] * scale, tail_most, sides)
    budget = {"injected_W": power, "reflected_W": low["reflected_W"] if evaluated else 0.0, "transmitted_W": low["transmitted_W"] if evaluated else 0.0}
    for key in _LOSS_CHANNELS:
        budget[key] = sum_left([state[key] for state in states]) if evaluated else 0.0
    budget["not_resolved_W"] = bounded_power if evaluated else followed
    budget["other_polarization_W"] = power - followed
    budget["closure_residual_W"] = power - sum_left([budget[key] for key in ("reflected_W", "transmitted_W", *_LOSS_CHANNELS, "not_resolved_W",
                                                                              "other_polarization_W")])
    record["power_budget"] = budget
    record["not_resolved_bounds"] = {key: list(value) for key, value in bounds.items()}

    if scan is not None:
        if fsr is None or not losses_known:
            record["scan"] = {"reason": ctx.path_reason if fsr is None else LOSSES_NOT_MODELED,
                              "message": ctx.path_message if fsr is None else loss_message}
        else:
            grid, sums = [], {"circulating_W": [], "transmitted_W": [], "reflected_W": []}
            for k in range(scan["points"]):
                nu = scan["from_Hz"] + (scan["to_Hz"] - scan["from_Hz"]) * k / (scan["points"] - 1)
                phase = TWO_PI * nu / fsr
                at = [steady_state(coupler, f * followed, phase - m * theta_x - n * theta_y) for m, n, f in content]
                grid.append(nu)
                for key in sums:
                    sums[key].append(sum_left([state[key] for state in at]))
            if ctx.fsr_entry is not None:  # (the scan's phases come from a free spectral range that is an estimate)
                record["scan_free_spectral_range"] = {"Hz": fsr, "value_basis": ctx.fsr_entry.model_dump(mode="json")}
            record["scan"] = {"detuning_Hz": grid, **sums, "not_resolved_W": followed * unresolved,
                              "not_resolved_bounds": {key: list(value) for key, value in unresolved_bounds(coupler, followed * unresolved).items()}}
    return record, tuple(checks)


def _condition(parameter: str, observed: float | None, *, within: bool | None, minimum: float | None = None, maximum: float | None = None,
               setting: str | None = None, default: float | None = None, at: Sequence[str] = (), code: str | None = None) -> ValidityCheck:
    """One validity condition of the analysis as a ``ValidityCheck``. The caller setting that holds its limit and where it was checked
    are named where the case is outside the limit or the limit in force is not ``default``; ``code`` where it is outside."""

    limit = minimum if minimum is not None else maximum
    named = within is False or (setting is not None and limit != default)
    return ValidityCheck(parameter=parameter, minimum=minimum, maximum=maximum, observed=observed, within=within,
                         setting=setting if named else None, at=tuple(at) if named else (), code=code if within is False else None)


# ------------------------------------------------------------------------------------------------ targets

#: The fields of a target row that are thresholds (the analysis's requirements identity).
REQUIREMENT_FIELDS = ("target_id", "quantity_id", "origin", "minimum", "maximum")


def _targets(project: Any, stated: Sequence[Mapping[str, Any]], ids: Sequence[str], wavelength_nm: float, quantities: Sequence[Quantity]) -> list[dict[str, Any]]:
    """The assessment of the request's targets and of the Project's targets with a cavity metric (``validation.scope.CAVITY_METRICS``)."""

    by_id = {q.quantity_id: q for q in quantities}
    rows = [{**judge(target, by_id.get(target["quantity_id"])), "origin": "REQUEST"} for target in stated]
    for target in load_project(project).targets:
        quantity_id = CAVITY_METRICS.get(target.metric)
        if quantity_id is None:
            continue
        entry = {"target_id": target.target_id, "quantity_id": quantity_id, "target_value": target.target_value, "tolerance_abs": target.tolerance_abs,
                 "tolerance_rel": target.tolerance_rel, "minimum": target.minimum, "maximum": target.maximum}
        why = None
        unit = CAVITY_METRIC_UNITS[target.metric]
        if target.unit is not None and target.unit != unit:
            rows.append({"target_id": target.target_id, "quantity_id": quantity_id, "metric": target.metric, "origin": "PROJECT", "metric_unit": unit,
                         "assessment": RequirementAssessment.INDETERMINATE.value, "reason": TARGET_UNIT_MISMATCH,
                         "message": f"the target declares unit {target.unit!r}; metric {target.metric} is in {unit!r}, so the cavity analysis gives "
                                    f"it no value. Set unit to {unit!r} and give its bounds in {unit}."})
            continue
        if target.configuration_ids:
            why = "the target names configurations; the cavity analysis reads the Project's nominal layout"
        elif target.device_instance_id is not None and target.device_instance_id not in ids:
            why = f"the target's component {target.device_instance_id!r} is not part of this cavity"
        elif target.wavelength_nm is not None and abs(target.wavelength_nm - wavelength_nm) > WAVELENGTH_MATCH_REL * wavelength_nm:
            why = f"the target is for {target.wavelength_nm:g} nm, the analysis for {wavelength_nm:g} nm"
        if why is not None:
            rows.append({"target_id": target.target_id, "quantity_id": quantity_id, "metric": target.metric, "origin": "PROJECT",
                         "assessment": RequirementAssessment.NOT_APPLICABLE.value, "message": why})
            continue
        rows.append({**judge(entry, by_id.get(quantity_id)), "metric": target.metric, "metric_unit": unit, "origin": "PROJECT"})
    return rows


#: A waist counts as inside its gap within this of the gap's ends (mm).
LATERAL_SLACK_MM = 1e-9


def _sum(values: Sequence[float]) -> float:
    """Left-to-right sum (the same order in every runtime)."""

    total = 0.0
    for value in values:
        total += value
    return total


__all__ = ["CAVITY_ANALYSIS", "CAVITY_ANALYSIS_INVALID", "CAVITY_ANALYSIS_VERSION", "CavityError", "cavity_analysis"]
