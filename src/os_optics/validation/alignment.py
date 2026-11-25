"""Geometric pair metrics: how well two beams -- of the same or of different wavelengths -- coincide on an observation plane.

They read only the chief rays and the paraxial Gaussian intensity envelopes of two arrivals on one plane (no phase, no
coherence, no polarization): co-alignment and overlap of beams that are combined but do not interfere (a two-colour
pump-probe overlap, a fluorescence excitation and a reference laser). They are computed on demand for the target, outside
the same-wavelength interference pairs, and never refused for a wavelength difference.

Operands: ``a`` is the arrival of ``source_id`` at ``wavelength_nm`` (and ``parameters.path_id_a`` /
``crossing_index_a``); ``b`` the arrival of ``source_id_b`` (default: ``source_id``) at ``wavelength_nm_b`` (default:
``wavelength_nm``) (and ``path_id_b`` / ``crossing_index_b``). Each must select exactly one beam, and not the same one.
Plane coordinates ``(u, v)`` are the observation plane's; ``c_a``, ``c_b`` the chief-ray hit points; ``S_j`` the
footprint matrix of beam ``j`` on the plane (``I_j ∝ exp(-2 (x - c_j)^T S_j (x - c_j))``, ``S = b_x b_x^T / w_x^2 +
b_y b_y^T / w_y^2`` with the beam axes projected onto the plane):

* ``centroid_separation_mm`` = ``|c_a - c_b|``;
* ``normalized_centroid_separation`` = ``|c_a - c_b| / sqrt(r_a^2 + r_b^2)``, ``r_j = 1 / sqrt(e^T S_j e)`` the 1/e^2
  footprint radius of beam ``j`` along the separation direction ``e`` (0 for coincident centres);
* ``angular_separation_rad`` = the angle between the chief-ray directions, ``atan2(|d_a x d_b|, d_a . d_b)``;
* ``geometric_overlap_ratio`` = ``∫ sqrt(I_a I_b) dA / sqrt(P_a P_b)`` over the plane (1 for identical footprints,
  ``exp(-d^2 / (2 w^2))`` for two round beams of radius ``w`` whose centres are ``d`` apart at normal incidence);
* ``waist_position_separation_mm`` = ``|(W_a - W_b) . m|`` with ``W_j = c_j - z_j d_j`` the waist position of beam ``j``
  (``z_j`` the mean of its x and y waist distances) and ``m`` the unit mean direction of the two beams: how far apart
  along the beams the two foci lie.

The last three need valid Gaussian states of both beams (else ``OUT_OF_MODEL_SCOPE`` with the state's code).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import NumericalConfig
from ..contracts.basis import ValidityCheck
from ..contracts.project import Target
from ..contracts.results import ArrivalResult, ObservationResult
from ..errors import ErrorCode, Status

#: metric -> (unit, needs valid Gaussian states)
ALIGNMENT_METRICS: dict[str, tuple[str, bool]] = {
    "centroid_separation_mm": ("mm", False),
    "normalized_centroid_separation": ("1", True),
    "angular_separation_rad": ("rad", False),
    "geometric_overlap_ratio": ("1", True),
    "waist_position_separation_mm": ("mm", True),
}


@dataclass(frozen=True)
class AlignmentValue:
    value: float | None
    status: Status = Status.PASS
    codes: tuple[str, ...] = ()
    message: str = ""
    empty: bool = False  # an operand selects no beam: the caller decides (empty selection, incomplete trace, not reached)
    validity: tuple[ValidityCheck, ...] = ()  # the failed accuracy check of a Gaussian state the metric needs


def _matches(a: ArrivalResult, source: str | None, wavelength: float | None, path: object, crossing: object, tol: float) -> bool:
    return ((source is None or a.beam.source_id == source)
            and (wavelength is None or abs(a.beam.wavelength_nm - wavelength) <= tol)
            and (path is None or a.beam.path_id == path)
            and (crossing is None or a.crossing_index == crossing))


def _footprint_matrix(a: ArrivalResult, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    g = a.beam.gaussian
    assert g is not None and g.e1 is not None and g.e2 is not None and g.beam_radius_x_mm and g.beam_radius_y_mm
    bx = np.array([np.dot(g.e1, u), np.dot(g.e1, v)])
    by = np.array([np.dot(g.e2, u), np.dot(g.e2, v)])
    return np.outer(bx, bx) / g.beam_radius_x_mm**2 + np.outer(by, by) / g.beam_radius_y_mm**2


def _gaussian_problem(a: ArrivalResult) -> str | None:
    g = a.beam.gaussian
    if a.chief_ray_obstructed:
        return ErrorCode.GAUSSIAN_CENTRAL_OBSTRUCTION.value
    if g is None or not g.valid:
        return (g.reason if g is not None and g.reason else None) or ErrorCode.OUT_OF_MODEL_SCOPE.value
    return None


def alignment_operands(t: Target, obs: ObservationResult, cfg: NumericalConfig) -> tuple[list[ArrivalResult], list[ArrivalResult]]:
    """The arrivals on ``obs`` that match the target's operand a and its operand b (module docstring)."""

    tol = cfg.interference_wavelength_tol_nm
    p = t.parameters
    source_b = t.source_id_b if t.source_id_b is not None else t.source_id
    wavelength_b = t.wavelength_nm_b if t.wavelength_nm_b is not None else t.wavelength_nm
    a_beams = [a for a in obs.arrivals if _matches(a, t.source_id, t.wavelength_nm, p.get("path_id_a"), p.get("crossing_index_a"), tol)]
    b_beams = [a for a in obs.arrivals if _matches(a, source_b, wavelength_b, p.get("path_id_b"), p.get("crossing_index_b"), tol)]
    return a_beams, b_beams


def measure_alignment(t: Target, obs: ObservationResult | None, cfg: NumericalConfig) -> AlignmentValue:
    """The geometric pair metric ``t.metric`` on ``t.observation_id`` (module docstring)."""

    needs_gaussian = ALIGNMENT_METRICS[t.metric][1]
    if obs is None:
        return AlignmentValue(None, Status.UNKNOWN, (ErrorCode.UNKNOWN_REFERENCE.value,), f"unknown observation {t.observation_id!r}")
    a_beams, b_beams = alignment_operands(t, obs, cfg)
    if not a_beams or not b_beams:
        return AlignmentValue(None, empty=True)
    if len(a_beams) > 1 or len(b_beams) > 1:
        return AlignmentValue(None, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                              f"{len(a_beams)} beams match operand a and {len(b_beams)} operand b; select one each with source_id / "
                              "source_id_b, wavelength_nm / wavelength_nm_b or parameters path_id_a / path_id_b")
    a, b = a_beams[0], b_beams[0]
    if (a.beam.path_id, a.crossing_index) == (b.beam.path_id, b.crossing_index):
        return AlignmentValue(None, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                              "both operands select the same beam; give the second one with source_id_b, wavelength_nm_b or "
                              "parameters.path_id_b")
    if needs_gaussian:
        for beam in (a, b):
            code = _gaussian_problem(beam)
            if code is not None:
                g = beam.beam.gaussian
                return AlignmentValue(None, Status.OUT_OF_MODEL_SCOPE, (code,),
                                      f"beam {beam.beam.path_id} has no valid Gaussian state on the plane: {t.metric} is not available",
                                      validity=g.validity if g is not None and not beam.chief_ray_obstructed else ())
    da, db = np.array(a.beam.direction), np.array(b.beam.direction)
    if t.metric == "angular_separation_rad":
        return AlignmentValue(math.atan2(float(np.linalg.norm(np.cross(da, db))), float(da @ db)))
    ca, cb = np.array(a.uv_mm), np.array(b.uv_mm)
    delta = ca - cb
    distance = float(np.hypot(delta[0], delta[1]))
    if t.metric == "centroid_separation_mm":
        return AlignmentValue(distance)
    if t.metric == "waist_position_separation_mm":
        ga, gb = a.beam.gaussian, b.beam.gaussian
        assert ga is not None and gb is not None
        za = 0.5 * (ga.waist_distance_x_mm + ga.waist_distance_y_mm)  # type: ignore[operator]
        zb = 0.5 * (gb.waist_distance_x_mm + gb.waist_distance_y_mm)  # type: ignore[operator]
        wa = np.array(a.beam.position_mm) - za * da
        wb = np.array(b.beam.position_mm) - zb * db
        mean = da + db
        norm = float(np.linalg.norm(mean))
        if norm <= cfg.zero_vector_tol:
            return AlignmentValue(None, Status.UNKNOWN, (ErrorCode.NUMERICAL_FAILURE.value,),
                                  "the two beams travel in opposite directions: their foci have no common axis")
        return AlignmentValue(abs(float((wa - wb) @ (mean / norm))))
    assert obs.plane is not None  # (every observation result carries its plane)
    u, v = np.array(obs.plane.u_axis), np.array(obs.plane.v_axis)
    sa, sb = _footprint_matrix(a, u, v), _footprint_matrix(b, u, v)
    if t.metric == "normalized_centroid_separation":
        if distance == 0.0:
            return AlignmentValue(0.0)
        e = delta / distance
        ra2, rb2 = 1.0 / float(e @ sa @ e), 1.0 / float(e @ sb @ e)
        return AlignmentValue(distance / math.sqrt(ra2 + rb2))
    m = sa + sb
    det_m, det_a, det_b = float(np.linalg.det(m)), float(np.linalg.det(sa)), float(np.linalg.det(sb))
    if det_m <= cfg.interference_det_tol or det_a <= 0.0 or det_b <= 0.0:
        return AlignmentValue(None, Status.UNKNOWN, (ErrorCode.NUMERICAL_FAILURE.value,),
                              "a beam crosses the plane at grazing incidence: its footprint is unbounded")
    rhs = sa @ ca + sb @ cb
    const = float(ca @ sa @ ca + cb @ sb @ cb - rhs @ np.linalg.solve(m, rhs))
    ratio = 2.0 * (det_a * det_b) ** 0.25 / math.sqrt(det_m) * math.exp(-max(const, 0.0))
    return AlignmentValue(min(1.0, ratio))
