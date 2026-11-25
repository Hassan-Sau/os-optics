"""Transverse-plane handoff: the beams at an observation plane and an element's outline in that plane's frame, for external
wave-optics, full-wave or CAD tools.

``handoff(...)`` reads what the Engine computed at one observation plane -- per arriving beam its power, wavelength,
medium, chief-ray hit point and direction, complex q per Gaussian axis with waist, curvature and M^2, Jones vector and
basis, phase and optical path length -- and states the convention under which a TEM00 field follows from it. Every
vector is given in the plane's own frame (u, v, n). With ``element`` it adds the clear aperture of each surface of that
component, transformed into the same (u, v, n) frame. It exports no sampled field: the field-construction convention
says how a client builds one when the beam is a TEM00 Gaussian.

Each beam carries a ``validity`` block: whether a TEM00 field may be constructed from it, what an upstream aperture
truncation means for its numbers (the transmitted power is exact; q, spot size and irradiance remain those of the
untruncated Gaussian, so a diffraction-accurate field downstream needs an external propagation from a plane before the
aperture), how much each aperture cut (``truncated_upstream[].power_fraction``, their sum ``truncated_power_fraction``) with
the intensity ripple a hard edge cutting that share makes (``field_intensity_ripple``), and whether its chief-ray phase and optical path length are defined: behind an ideal or tabulated thin lens, a Faraday rotator
or a diffraction grating they are not (the lens states a focal length, not the optical path through it), so ``phase_rad`` and
``optical_path_length_mm`` are null there and ``validity.phase_defined`` is false. A beam whose power is out of model scope
(``ArrivalResult.power_out_of_scope_at``: behind a crystal whose second-harmonic generation is out of model scope there, a variable
attenuator or transmission map varying across it, the side of a gradient-index rod, an absorbing plane or element part of it meets
beside its chief ray) carries the chief ray's power, not the light's: its ``power_W`` is null, ``validity.beam_parameters``
``POWER_OUT_OF_MODEL_SCOPE`` with ``validity.power_out_of_scope_at``, and no field is constructed from it.

Model scope. A beam whose light a model-scope finding of the run reaches (an element used outside its published model:
``model_scope_issues``, its ``affected_arrivals``) keeps its numbers and its ``beam_parameters``; ``validity.model_scope`` lists
each such finding (``code``, ``instance_id``, the kinds of quantity it touches, its validity checks of this beam's light) and
the beam's ``value_basis`` carries an ``ESTIMATE`` with those checks on each number of a kind the finding touches: ``power_W``
(``POWER``), the chief ray's position and direction (``GEOMETRY``), the Jones vector (``POLARIZATION``), the phase and optical
path length, and the Gaussian state under ``gaussian.value_basis["*"]`` (``GEOMETRY``). A field, map or detector reading
built from such a beam carries them on (``results.pupil_field``, ``results.focal_field``, ``results.irradiance``).

Truncation rule. A beam is ``TRUNCATED_GAUSSIAN`` (no TEM00 field) when an aperture on its way cut more than the run's
``gaussian_clip_warn_fraction`` of its power: the share above which the trace itself warns (``GAUSSIAN_TRUNCATION``). A
smaller cut leaves the beam ``TEM00`` and is stated in numbers: per aperture the share ``f`` of the arriving power it
removed, and ``field_intensity_ripple``, the sum of ``2 sqrt(f) + f``: the largest relative change of the on-axis intensity
behind a centred round hard edge that cuts ``f`` of a Gaussian beam (the edge wave has ``sqrt(f)`` of the axial amplitude:
1 % at ``f = 2.5e-5``, an opening of 2.3 beam radii; 6.4 % at ``f = 1e-3``), and to first order the relative drop of the
focal peak against the untruncated beam of the same power. ``numerical_settings`` states the limits of the run the
document was read from.
"""

from __future__ import annotations

import math
from typing import Any

from ..config import NumericalConfig
from ..contracts.request import ValidationRequest
from ..contracts.results import ArrivalResult, ValidationResult
from ..errors import ErrorCode, Status
from ..optics import apertures

HANDOFF_SCHEMA = "opensci.transverse_handoff"
#: 1.1: a beam is TEM00 unless an aperture cut more than ``gaussian_clip_warn_fraction``; each aperture's cut share
#: (``truncated_upstream[].power_fraction``, ``warned``), ``truncated_power_fraction``, ``field_intensity_ripple``,
#: ``numerical_settings``, and the export columns of the two sums. 1.2 adds ``validity.model_scope``, a beam's
#: ``value_basis`` and ``gaussian.value_basis`` (present when a model-scope finding reaches the beam), and the export
#: columns ``model_scope`` and ``value_basis``.
HANDOFF_SCHEMA_VERSION = "1.2"
#: Power-ledger sink of the Gaussian power a clear aperture the chief ray passes cuts off.
APERTURE_TRUNCATION = "APERTURE_TRUNCATION"
#: The run's settings a handoff document states (``numerical_settings``): the share its truncation rule applies and the
#: Gaussian limit a field stage built on the document reads (``results.focal_field``).
HANDOFF_SETTINGS = ("gaussian_clip_warn_fraction", "gaussian_scope_divergence_rad")
#: |n| and 1 - |n . N| below this make an element surface coplanar with the observation plane (mm; dimensionless).
COPLANAR_TOLERANCE = 1e-9
#: Error codes (``ResultsError``).
ELEMENT_NOT_FOUND = "ELEMENT_NOT_FOUND"
REQUEST_NOT_AVAILABLE = "REQUEST_NOT_AVAILABLE"

FIELD_CONVENTION = {
    "time_dependence": "exp(-i w t); propagation along the beam direction is exp(+i k z), k = 2 pi n / lambda0 (lambda0 the vacuum "
                       "wavelength, n = medium_index)",
    "q": "q_x_mm = [z, z_R] per Gaussian axis: z = waist_distance_x_mm, the signed distance from the waist to the plane along the beam "
         "(positive: the waist lies upstream, the beam diverges), z_R the Rayleigh range in the medium. 1 / q = 1 / R - i lambda / (pi "
         "w^2), lambda = lambda0 / n; through a paraxial system q' = (A q + B) / (C q + D)",
    "tem00_field": "E(x, y) = sqrt(2 P / (pi w_x w_y)) exp(-x^2 / w_x^2 - y^2 / w_y^2) exp(+i k (x^2 / (2 R_x) + y^2 / (2 R_y))) "
                   "(J_1 e1 + J_2 e2) / |J|, equivalently exp(+i k x^2 / (2 conj(q_x))) per axis: x, y measured from the chief-ray hit "
                   "point along gaussian.axis_x / axis_y in the plane perpendicular to the beam; w the 1/e^2 radii (beam_radius_*), R "
                   "the curvature radii (curvature_radius_*: positive when the beam diverges; a flat wavefront when null)",
    "normalization": "|E|^2 in W/mm^2 integrates to P = power_W over the transverse plane",
    "phase_reference": "the field phase at the chief ray is phase_rad (null without a single coherent reference: then 0); the Gouy phase "
                       "is omitted. Behind an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating phase_rad and optical_path_length_mm are null "
                       "(validity.phase_defined false): the lens model states no optical path, so a field built there has an "
                       "undetermined constant phase and no phase relation to other beams",
    "polarization": "Jones vector (J_1, J_2) in the transported basis (polarization.basis_e1, basis_e2), e1 x e2 = beam direction",
    "applicability": "TEM00 only (M^2 = 1 on both axes, Gaussian state valid, no aperture upstream that cut more than the run's "
                     "gaussian_clip_warn_fraction of the beam: validity.tem00_field_applicable). A smaller cut is stated per beam: "
                     "validity.truncated_power_fraction, and validity.field_intensity_ripple, the sum of 2 sqrt(f) + f over the "
                     "apertures: the largest relative change of the on-axis intensity behind a centred round hard edge cutting "
                     "the share f. M^2 > 1 alone does not determine the modal content or the coherence of a beam: no field "
                     "follows from it. Intensity or power alone does not determine a complex field",
    "tilted_plane": "the field is defined in the beam's transverse plane at the chief-ray hit point; on a plane the beam crosses at "
                    "aoi_deg != 0 a client evaluates it at the plane points from their coordinates along axis_x, axis_y and the beam "
                    "direction (all given in the plane's (u, v, n) frame)",
    "units": "mm, W, rad, nm (wavelength)",
}
ELEMENT_CONVENTION = ("Each surface's clear aperture in its own tangent plane at the vertex (plane surfaces: the plane itself): centre, "
                      "surface axes (u_s, v_s, normal) and corners as components along the observation plane's (u, v, n). in_plane: the "
                      "surface lies in the observation plane, so the outline is exact in (u, v). A curved surface's rim lies at its sag "
                      "depth (rim_center_uvn_mm for a centred round aperture of a rotationally symmetric surface).")


def _floats(vector: Any) -> list[float]:
    return [float(c) for c in vector]


def _dot(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class _Frame:
    """The observation plane's frame: origin and unit axes u, v, n."""

    def __init__(self, origin: Any, u: Any, v: Any, n: Any) -> None:
        self.origin, self.u, self.v, self.n = _floats(origin), _floats(u), _floats(v), _floats(n)

    def point(self, p: Any) -> list[float]:
        d = [float(p[i]) - self.origin[i] for i in range(len(self.origin))]
        return [_dot(d, self.u), _dot(d, self.v), _dot(d, self.n)]

    def vector(self, d: Any) -> list[float]:
        d = _floats(d)
        return [_dot(d, self.u), _dot(d, self.v), _dot(d, self.n)]


def _ancestry(result: ValidationResult, path_id: str) -> list[str]:
    parents = {p.path_id: p.parent_path_id for p in result.paths}
    chain, current = [], path_id
    while current is not None and current not in chain:
        chain.append(current)
        current = parents.get(current)
    return chain


def _arriving_powers(result: ValidationResult, path_id: str | None, surface_id: str | None) -> list[float | None]:
    """The power of path ``path_id`` each time it arrives at ``surface_id`` (its segments that end there, in order)."""

    path = next((p for p in result.paths if p.path_id == path_id), None)
    return [] if path is None else [segment.power_end_W for segment in path.segments if segment.end_surface_id == surface_id]


def _truncations(result: ValidationResult, arrival: ArrivalResult, warn_fraction: float, warned_surfaces: frozenset[str]) -> list[dict[str, Any]]:
    """Where an aperture removed Gaussian power from the beam before it reached the plane: power-ledger entries of the path or a
    path it branched from, at surfaces of the arrival's lineage (its interactions from the source up to the plane).

    Each row states the share of the power arriving at the aperture that it removed -- ``power_fraction``, the entry over the
    path's power arriving at that surface on that pass -- and ``warned``: the share is above ``warn_fraction`` (the run's
    ``gaussian_clip_warn_fraction``, the trace's own rule). A path that meets the aperture once, or is cut at every pass,
    pairs each entry with its pass. Where the pass is not determined (a path that meets the aperture several times and is
    cut at some of them only) or the arriving power is not recorded, the row states ``power_fraction_upper`` -- the entry over
    the smallest power the path brings to that surface, or over the entry plus the beam's power at the plane: no smaller
    than the share -- and ``warned`` follows the trace's ``GAUSSIAN_TRUNCATION`` finding at that surface."""

    budget = result.power_budget
    if budget is None:
        return []
    chain = set(_ancestry(result, arrival.beam.path_id))
    upstream = {entry.rpartition(":")[0] for entry in arrival.lineage}
    cuts = [e for e in budget.entries if e.kind == APERTURE_TRUNCATION and e.path_id in chain and e.surface_id in upstream and e.power_W > 0.0]
    counts: dict[tuple[str | None, str | None], int] = {}
    for e in cuts:
        counts[(e.path_id, e.surface_id)] = counts.get((e.path_id, e.surface_id), 0) + 1
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[str | None, str | None], int] = {}
    for e in cuts:
        key = (e.path_id, e.surface_id)
        index = seen.get(key, 0)
        seen[key] = index + 1
        arriving = _arriving_powers(result, e.path_id, e.surface_id)
        incident = arriving[0] if len(arriving) == 1 else arriving[index] if len(arriving) == counts[key] else None
        row: dict[str, Any] = {"surface_id": e.surface_id, "path_id": e.path_id, "power_W": e.power_W}
        if incident is not None and incident > 0.0:
            row["power_fraction"] = min(e.power_W / incident, 1.0)
            row["warned"] = row["power_fraction"] > warn_fraction
        else:
            known = [power for power in arriving if power is not None and power > 0.0]
            power = arrival.beam.power_W
            if known:
                row["power_fraction_upper"] = min(e.power_W / min(known), 1.0)
            elif power is not None:
                row["power_fraction_upper"] = e.power_W / (e.power_W + power)
            row["warned"] = e.surface_id in warned_surfaces and row.get("power_fraction_upper", 1.0) > warn_fraction
        rows.append(row)
    return rows


def edge_ripple(power_fraction: float) -> float:
    """The largest relative change of the on-axis intensity behind a centred round hard edge that cuts ``power_fraction`` of a
    Gaussian beam: the edge wave has ``sqrt(f)`` of the axial amplitude, so the intensity lies within ``(1 +- sqrt(f))^2``."""

    return 2.0 * math.sqrt(power_fraction) + power_fraction


#: Why ``phase_rad`` / ``optical_path_length_mm`` are null behind an ideal or tabulated thin lens, a Faraday rotator or a grating.
PHASE_UNDEFINED_STATEMENT = ("This beam passed an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating, whose model states no optical "
                             "path: its chief-ray phase and optical path length are not defined (null). q, power and Jones vector "
                             "stay valid; a field built from them has an undetermined constant phase.")


#: The numbers of a handoff beam a model-scope finding qualifies, each with the kinds of quantity it is read as.
_BEAM_KINDS = {"power_W": ("POWER",), "chief_ray.uv_mm": ("GEOMETRY",), "chief_ray.point_uvn_mm": ("GEOMETRY",),
               "chief_ray.direction_uvn": ("GEOMETRY",), "chief_ray.aoi_deg": ("GEOMETRY",), "polarization.jones": ("POLARIZATION",),
               "phase_rad": ("GEOMETRY", "POLARIZATION"), "optical_path_length_mm": ("GEOMETRY",)}


def model_scope(findings: Any, plane: str, light: set[tuple[str, str, int]], lines: set[tuple[str, str, int]] | None = None,
                wavelength_nm: float | None = None) -> list[dict[str, Any]]:
    """The model-scope findings ``findings`` (their rows) that reach the arrivals ``light`` or the sampled lines ``lines`` on
    ``plane``, each as ``{code, instance_id, touches, validity}``: the kinds of quantity it touches and its checks of that
    light (``results.basis.scope_checks``; ``wavelength_nm``: the lines' wavelength, which a finding of an earlier Engine
    that lists no sampled line is matched by)."""

    from ..local.validity import finding_quantity_kinds
    from .basis import _finding_row, scope_checks

    out = []
    for row in map(_finding_row, findings or ()):
        if plane not in (row.get("affected_observation_ids") or ()):
            continue
        checks = scope_checks([row], plane, light, lines=lines, unlocalised=bool(lines), wavelength_nm=wavelength_nm)
        if checks:
            out.append({"code": row["code"], "instance_id": row["instance_id"],
                        "touches": sorted(kind.value for kind in finding_quantity_kinds(row["code"])),
                        "validity": [check.model_dump(mode="json") for check in checks]})
    return out


def merged_scope(scopes: Any) -> list[dict[str, Any]]:
    """Several ``model_scope`` lists as one: a finding (code and element) once, with every check its entries state."""

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for scope in scopes:
        for finding in scope or ():
            key = (finding["code"], finding["instance_id"])
            into = merged.setdefault(key, {**finding, "touches": list(finding["touches"]), "validity": []})
            into["touches"] = sorted(set(into["touches"]) | set(finding["touches"]))
            into["validity"] += [c for c in finding["validity"] if c not in into["validity"]]
    return list(merged.values())


def handoff_scope(handoff: Any, labels: Any) -> list[dict[str, Any]]:
    """The model-scope findings of the beams ``labels`` (path ids) of a handoff document (their ``validity.model_scope``)."""

    beams = handoff.get("beams") if isinstance(handoff, dict) else None
    wanted = {str(label) for label in labels}
    return merged_scope((beam.get("validity") or {}).get("model_scope") for beam in beams or ()
                        if isinstance(beam, dict) and str(beam.get("path_id")) in wanted)


#: The numbers of a map document a model-scope finding on its light qualifies (``mark_document``).
MAP_KEYS = ("incoherent_W_per_mm2", "coherent_W_per_mm2", "grid_power_incoherent_W", "grid_power_coherent_W")
#: Every kind of quantity: what a map, a field or a detector reading of light reads.
_EVERY_KIND = ("GEOMETRY", "POLARIZATION", "POWER")


def _merged_entry(entry: dict[str, Any] | None, scope_entry: dict[str, Any]) -> dict[str, Any]:
    from ..contracts.basis import ValueBasis, merge_bases

    if entry is None:
        return scope_entry
    return merge_bases(ValueBasis.model_validate(entry), ValueBasis.model_validate(scope_entry), monotone=None).model_dump(mode="json")


#: Dict-quantity statuses a model-scope finding on the light does not qualify: no number, or a number stated from outside.
_UNMARKED_STATUSES = ("UNAVAILABLE", "EXTERNALLY_CHARACTERIZED")


def mark_quantities(quantities: dict[str, Any], scope: list[dict[str, Any]], skip: Any = ()) -> None:
    """Dict quantities (``results.quantity``) of light the findings ``scope`` reach: each the model computed (not
    ``UNAVAILABLE``, not a stated ``EXTERNALLY_CHARACTERIZED`` one, not in ``skip``: a number that does not read the light)
    gets their ``ESTIMATE`` merged into its ``value_basis``."""

    entry = _scope_entry(scope, _EVERY_KIND)
    if entry is None:
        return
    for name, quantity in quantities.items():
        if name in skip or not isinstance(quantity, dict) or quantity.get("status") in _UNMARKED_STATUSES:
            continue
        quantity["value_basis"] = _merged_entry(quantity.get("value_basis"), entry)


def mark_document(doc: dict[str, Any], scope: list[dict[str, Any]], skip: Any = ()) -> dict[str, Any]:
    """A map document (an irradiance map, a pupil field) built from light the findings ``scope`` reach: ``model_scope`` lists
    them, each of its maps and grid powers (``MAP_KEYS``) and its dict quantities carry their ``ESTIMATE`` in
    ``value_basis`` (``skip``: quantities that do not read the light). A detector reading of the document carries them on
    (``results.detector``). ``doc`` without ``scope`` is returned as it is."""

    entry = _scope_entry(scope, _EVERY_KIND) if scope else None
    if entry is None:
        return doc
    doc["model_scope"] = scope
    bases = dict(doc.get("value_basis") or {})
    for key in MAP_KEYS:
        if doc.get(key) is not None:
            bases[key] = _merged_entry(bases.get(key), entry)
    if bases:
        doc["value_basis"] = bases
    if isinstance(doc.get("quantities"), dict):
        mark_quantities(doc["quantities"], scope, skip)
    return doc


def _scope_entry(scope: list[dict[str, Any]], kinds: tuple[str, ...] | frozenset[str]) -> dict[str, Any] | None:
    """The ``ESTIMATE`` (JSON form) with the checks of the findings of ``scope`` that touch one of ``kinds``, or ``None``."""

    from ..contracts.basis import ValidityCheck, ValueBasis
    from ..errors.status import ValueKind

    checks = [ValidityCheck.model_validate(c) for f in scope if set(f["touches"]) & set(kinds) for c in f["validity"]]
    unique = [c for i, c in enumerate(checks) if c not in checks[:i]]
    return ValueBasis(kind=ValueKind.ESTIMATE, validity=tuple(unique)).model_dump(mode="json") if unique else None


def scope_marks(scope: list[dict[str, Any]]) -> list[tuple[str, frozenset[Any], tuple[Any, ...]]]:
    """A ``model_scope`` list as ``(code, kinds, checks)``: what an analysis built from that light marks its quantities with
    (``results.imaging.outside_model``)."""

    from ..contracts.basis import ValidityCheck
    from ..validation.quantity_kinds import QuantityKind

    return [(f["code"], frozenset(QuantityKind(k) for k in f.get("touches") or ()),
             tuple(ValidityCheck.model_validate(c) for c in f.get("validity") or ()))
            for f in scope or ()]


def _validity(arrival: ArrivalResult, truncations: list[dict[str, Any]], issues: list[dict[str, Any]]) -> dict[str, Any]:
    g = arrival.beam.gaussian
    valid = g is not None and g.valid
    m2_one = valid and g.m2_x == 1.0 and g.m2_y == 1.0  # type: ignore[union-attr]
    # (the trace's own judgement: a cut above the run's warning share, or the beam's Gaussian state says so)
    truncated = any(t["warned"] for t in truncations) or (g is not None and g.reason == ErrorCode.GAUSSIAN_TRUNCATION.value)
    fractions = [t.get("power_fraction") for t in truncations]
    cut = None if any(f is None for f in fractions) else math.fsum(fractions)  # type: ignore[arg-type]
    ripple = None if cut is None else math.fsum(edge_ripple(f) for f in fractions)  # type: ignore[arg-type]
    if arrival.power_out_of_scope_at:
        state = "POWER_OUT_OF_MODEL_SCOPE"
        statement = (f"The power of this beam is out of model scope behind {', '.join(arrival.power_out_of_scope_at)}: it is its chief ray's "
                     "power, not the light's, so power_W is withheld (null) and no field is constructed from it.")
    elif arrival.power_taken_before_at:
        state = "POWER_TAKEN_BEFORE_PLANE"
        statement = (f"{', '.join(arrival.power_taken_before_at)} take part of this beam before the plane: its power lies only between "
                     "the beam's less power_taken_before_upper_W and the beam's, and where the light was taken is not known, so "
                     "power_W is withheld (null) and no field is constructed from it.")
    elif not valid:
        state = "UNAVAILABLE"
        statement = (f"No Gaussian state here ({g.reason if g is not None else 'none'}): the beam parameters are withheld; the power and "
                     "the chief ray are the Engine's values.")
    elif truncated:
        state = "TRUNCATED_GAUSSIAN"
        statement = ("An aperture upstream removed part of this beam: power_W is exact (the Gaussian power the aperture passes), while q, "
                     "beam radius and irradiance are those of the untruncated Gaussian, not of the diffracted field. For a "
                     "diffraction-accurate field, start the external propagation at a plane before that aperture.")
    elif not m2_one:
        state = "EMBEDDED_GAUSSIAN"
        statement = ("M^2 > 1: q describes the beam's embedded Gaussian (second-moment widths). M^2 alone does not determine the modal "
                     "content or the coherence of the beam, so no field is constructed from it.")
    elif truncations:
        state = "TEM00"
        statement = ("TEM00 Gaussian: q, power and Jones vector define the paraxial field under the stated convention. No aperture "
                     "upstream cut more than the run's gaussian_clip_warn_fraction of it; what they cut is in truncated_upstream, "
                     "truncated_power_fraction and field_intensity_ripple. Downstream hard apertures diffract it; the Engine's "
                     "Gaussian numbers there do not contain that diffraction.")
    else:
        state = "TEM00"
        statement = ("Unclipped TEM00 Gaussian: q, power and Jones vector define the paraxial field under the stated convention. "
                     "Downstream hard apertures diffract it; the Engine's Gaussian numbers there do not contain that diffraction.")
    phase_defined = arrival.ray_opl_defined
    if not phase_defined:
        statement = f"{statement} {PHASE_UNDEFINED_STATEMENT}"
    scope = {"power_out_of_scope_at": list(arrival.power_out_of_scope_at)} if arrival.power_out_of_scope_at else {}
    if arrival.power_taken_before_at:
        scope["power_taken_before_at"] = list(arrival.power_taken_before_at)
    return {"beam_parameters": state, "tem00_field_applicable": state == "TEM00", "phase_defined": phase_defined, **scope,
            "gaussian_status": None if g is None else g.status.value, "gaussian_reason": None if g is None else g.reason,
            "truncated_upstream": truncations, "truncated_power_fraction": cut, "field_intensity_ripple": ripple,
            "issues": issues, "statement": statement}


def _beam(arrival: ArrivalResult, frame: _Frame, result: ValidationResult, issues_by_surface: dict[str, list[dict[str, Any]]],
          warn_fraction: float, scope: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    beam, g = arrival.beam, arrival.beam.gaussian
    warned = frozenset(surface for surface, found in issues_by_surface.items()
                       if any(issue["code"] == ErrorCode.GAUSSIAN_TRUNCATION.value for issue in found))
    truncations = _truncations(result, arrival, warn_fraction, warned)
    issues = [issue for t in truncations for issue in issues_by_surface.get(t["surface_id"] or "", [])]
    gaussian: dict[str, Any] | None = None
    if g is not None and g.valid:
        gaussian = {
            "q_x_mm": _floats(g.q_x_mm), "q_y_mm": _floats(g.q_y_mm), "m2_x": g.m2_x, "m2_y": g.m2_y,  # type: ignore[arg-type]
            "beam_radius_x_mm": g.beam_radius_x_mm, "beam_radius_y_mm": g.beam_radius_y_mm,
            "waist_radius_x_mm": g.waist_radius_x_mm, "waist_radius_y_mm": g.waist_radius_y_mm,
            "waist_distance_x_mm": g.waist_distance_x_mm, "waist_distance_y_mm": g.waist_distance_y_mm,
            "rayleigh_range_x_mm": g.rayleigh_range_x_mm, "rayleigh_range_y_mm": g.rayleigh_range_y_mm,
            "curvature_radius_x_mm": g.curvature_radius_x_mm, "curvature_radius_y_mm": g.curvature_radius_y_mm,
            "far_field_half_angle_x_rad": g.far_field_half_angle_x_rad, "far_field_half_angle_y_rad": g.far_field_half_angle_y_rad,
            "axis_x_uvn": frame.vector(g.e1), "axis_y_uvn": frame.vector(g.e2),  # type: ignore[arg-type]
        }
    doc = {
        "path_id": beam.path_id, "source_id": beam.source_id, "spectral_index": beam.spectral_index, "wavelength_nm": beam.wavelength_nm,
        "crossing_index": arrival.crossing_index, "medium_index": None if g is None else g.medium_index,
        "power_W": None if arrival.power_out_of_scope_at or arrival.power_taken_before_at else beam.power_W,
        "chief_ray": {"uv_mm": _floats(arrival.uv_mm), "point_uvn_mm": frame.point(beam.position_mm),
                      "direction_uvn": frame.vector(beam.direction), "aoi_deg": arrival.aoi_deg},
        "gaussian": gaussian,
        "polarization": {"jones": None if beam.polarization is None else [_floats(beam.polarization.ex), _floats(beam.polarization.ey)],
                         "basis_e1_uvn": frame.vector(beam.basis.e1), "basis_e2_uvn": frame.vector(beam.basis.e2)},
        "phase_rad": beam.phase_rad if arrival.ray_opl_defined else None,
        "optical_path_length_mm": beam.optical_path_length_mm if arrival.ray_opl_defined else None,
        "validity": _validity(arrival, truncations, issues),
    }
    if scope:
        # (a model-scope finding reaches this beam: its numbers stay, each of a kind the finding touches marked)
        doc["validity"]["model_scope"] = scope
        codes = ", ".join(sorted({f"{f['code']} at {f['instance_id']}" for f in scope}))
        doc["validity"]["statement"] += (f" A published model used outside its validity reaches this beam ({codes}): each number "
                                         "of a kind that finding touches is that model's number, an ESTIMATE in value_basis.")
        bases = {}
        for key, kinds in _BEAM_KINDS.items():
            head, _dot, tail = key.partition(".")
            value = doc[head][tail] if tail else doc[head]
            if value is not None and (entry := _scope_entry(scope, kinds)) is not None:
                bases[key] = entry
        if bases:
            doc["value_basis"] = bases
        if gaussian is not None and (entry := _scope_entry(scope, ("GEOMETRY",))) is not None:
            gaussian["value_basis"] = {"*": entry}
    return doc


def _sag(surface: Any, rho: float) -> float:
    c, k = surface.curvature, surface.conic
    total = c * rho * rho / (1.0 + math.sqrt(max(0.0, 1.0 - (1.0 + k) * c * c * rho * rho)))
    power = rho * rho * rho * rho
    for coefficient in surface.coefficients:
        total += coefficient * power
        power *= rho * rho
    return total


def _surface_outline(surface: Any, frame: _Frame) -> dict[str, Any]:
    ap = surface.aperture
    if surface.geometry == "sphere":
        normal = _floats(surface.cap_axis)
        u_axis, v_axis = _floats(surface.u_axis), _floats(surface.v_axis)
        radius = float(surface.radius)
        vertex = [float(surface.center[i]) + radius * normal[i] for i in range(len(normal))]
        rim_depth = radius - math.sqrt(max(0.0, radius * radius - surface.aperture_radius * surface.aperture_radius))
        outline: dict[str, Any] = {"shape": "circle", "radius_mm": float(surface.aperture_radius), "obstruction_radius_mm": 0.0}
        center = vertex
        rim = [vertex[i] - rim_depth * normal[i] for i in range(len(normal))]
    else:
        normal, u_axis, v_axis = _floats(surface.normal), _floats(surface.u_axis), _floats(surface.v_axis)
        origin = _floats(surface.origin)
        if ap.shape == "none":
            return {"surface_id": surface.surface_id, "geometry": surface.geometry, "shape": "unbounded",
                    "origin_uvn_mm": frame.point(origin), "normal_uvn": frame.vector(normal)}
        center = [origin[i] + ap.offset_u * u_axis[i] + ap.offset_v * v_axis[i] for i in range(len(origin))]
        if ap.shape == "circle" and not ap.holes:
            outline = {"shape": "circle", "radius_mm": ap.p1, "obstruction_radius_mm": ap.obstruction}
        elif ap.shape == "rectangle" and not ap.holes:
            outline = {"shape": "rectangle", "half_width_mm": ap.p1, "half_height_mm": ap.p2}
        else:  # (its sizes, and its boundary rings below)
            outline = {key: value for key, value in apertures.describe(ap).items() if key != "center_offset_mm"}
        rim = None
        if (surface.geometry == "sag" and ap.shape == "circle" and ap.offset_u == 0.0 and ap.offset_v == 0.0 and not surface.cylindrical
                and surface.freeform is None):
            depth = _sag(surface, ap.p1)
            rim = [origin[i] + depth * normal[i] for i in range(len(origin))]
        elif surface.geometry == "plane":
            rim = center
    doc: dict[str, Any] = {"surface_id": surface.surface_id, "geometry": surface.geometry, **outline,
                           "center_uvn_mm": frame.point(center), "u_s_uvn": frame.vector(u_axis), "v_s_uvn": frame.vector(v_axis),
                           "normal_uvn": frame.vector(normal),
                           "rim_center_uvn_mm": None if rim is None else frame.point(rim)}
    if surface.geometry != "sphere" and (ap.shape not in ("circle", "rectangle") or ap.holes):
        # the outer boundary, then each hole's, as points in the plane (corners exactly, round parts as rings)
        doc["boundary_uvn_mm"] = [[frame.point([origin[i] + pu * u_axis[i] + pv * v_axis[i] for i in range(len(origin))]) for pu, pv in ring]
                                  for ring in apertures.rings(ap)]
    if outline["shape"] == "rectangle":
        doc["corners_uvn_mm"] = [frame.point([center[i] + su * ap.p1 * u_axis[i] + sv * ap.p2 * v_axis[i] for i in range(len(center))])
                                 for su, sv in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))]
    n_uvn = doc["normal_uvn"]
    doc["in_plane"] = (surface.geometry == "plane" and abs(doc["center_uvn_mm"][2]) <= COPLANAR_TOLERANCE
                       and 1.0 - abs(n_uvn[2]) <= COPLANAR_TOLERANCE)
    return doc


def element_outline(request: dict[str, Any] | None, element: str, frame: _Frame, cfg: NumericalConfig) -> dict[str, Any]:
    """The clear apertures of every surface of component ``element`` in the plane's (u, v, n) frame."""

    from ..optics.materials.library import MaterialLibrary
    from ..scene.lower import lower_scene
    from .model import ResultsError

    if request is None:
        raise ResultsError(REQUEST_NOT_AVAILABLE, "this run carries no Engine request (engine_request.json): an element outline needs it")
    parsed = ValidationRequest.model_validate(request)
    scene, _poses = lower_scene(parsed.project, cfg, MaterialLibrary(parsed.project.materials, cfg))
    surfaces = [s for s in scene.surfaces if s.owner_id == element and not s.surface_id.startswith("observation/")]
    if not surfaces:
        known = sorted({c.instance_id for c in parsed.project.components if c.surfaces})
        raise ResultsError(ELEMENT_NOT_FOUND, f"no optical element {element!r} in this run; elements: {known}")
    return {"instance_id": element, "convention": ELEMENT_CONVENTION, "surfaces": [_surface_outline(s, frame) for s in surfaces]}


def handoff(result: ValidationResult, request: dict[str, Any] | None, monitor: str, *, path: str | None = None,
            crossing_index: int | None = None, element: str | None = None, cfg: NumericalConfig | None = None,
            findings: Any = ()) -> dict[str, Any]:
    """The handoff document of observation plane ``monitor`` (module docstring): its frame, the arriving beams (one ``path`` /
    ``crossing_index`` when given) and, with ``element``, that component's clear apertures in the plane's frame.
    ``findings``: the run's model-scope findings (their rows), carried by the beams they reach (module docstring)."""

    from .model import NOT_COMPUTED, QUERY_REQUIRES, ResultsError, not_computed_message

    scope = result.evaluation
    if scope is not None:
        # (a narrowed Engine evaluation: the handoff needs the readings, the ledger's truncations and the bounds' issues, as
        # RunResults.handoff does -- never a document whose validity statements are incomplete)
        left_out = {n.capability for n in scope.not_computed}
        missing = [c for c in QUERY_REQUIRES["handoff"] if c in left_out]
        if missing:
            raise ResultsError(NOT_COMPUTED, not_computed_message("handoff", missing, engine_result=True))
    observation = next((o for o in result.observations if o.observation_id == monitor), None)
    if observation is None or observation.plane is None:
        known = [o.observation_id for o in result.observations]
        raise ResultsError("MONITOR_NOT_FOUND", f"no observation plane {monitor!r} in this run; monitors: {known}")
    plane = observation.plane
    frame = _Frame(plane.origin_mm, plane.u_axis, plane.v_axis, plane.normal)
    issues_by_surface: dict[str, list[dict[str, Any]]] = {}
    prefix = "surface:"
    for issue in result.issues:
        if issue.location and issue.location.startswith(prefix) and issue.status is not Status.PASS:
            issues_by_surface.setdefault(issue.location[len(prefix):], []).append(
                {"code": issue.code.value, "status": issue.status.value, "location": issue.location})
    arrivals = [a for a in observation.arrivals if path in (None, a.beam.path_id) and crossing_index in (None, a.crossing_index)]
    cfg = cfg or NumericalConfig()
    from .basis import _finding_row

    named = [row for row in map(_finding_row, findings or ()) if monitor in (row.get("affected_observation_ids") or ())]
    return {
        "schema": HANDOFF_SCHEMA, "schema_version": HANDOFF_SCHEMA_VERSION, "observation_id": monitor,
        "plane": {"origin_mm": _floats(plane.origin_mm), "u_axis": _floats(plane.u_axis), "v_axis": _floats(plane.v_axis),
                  "normal": _floats(plane.normal), "medium": plane.medium,
                  "aperture": None if plane.aperture is None else plane.aperture.model_dump(mode="json")},
        "beams": [_beam(a, frame, result, issues_by_surface, cfg.gaussian_clip_warn_fraction,
                        model_scope(named, monitor, {(monitor, a.beam.path_id, a.crossing_index)}) if named else None)
                  for a in arrivals],
        "numerical_settings": {name: getattr(cfg, name) for name in HANDOFF_SETTINGS},
        "element": None if element is None else element_outline(request, element, frame, cfg),
        "conventions": dict(FIELD_CONVENTION),
        "scope": "Engine values only: paraxial Gaussian beam parameters, chief-ray geometry, power and polarization at the plane. No "
                 "sampled field is exported; diffraction at hard apertures is not part of these numbers.",
    }
