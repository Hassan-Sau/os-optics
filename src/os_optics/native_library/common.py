"""Shared pieces of native classes: the build result, coatings, frames, outlines and the component template.

Conventions of every produced model (the public catalog's own, ``OpticalFrameV1``):

* a transmissive element (lens, plate, polarizer, waveplate, stop): local +Z is the optical axis and the direction
  light travels; the origin is the first surface vertex (or the element's plane);
* a reflector or a plate splitter: local +Z is the normal of the coated face pointing toward the incoming light; the
  origin is the centre of that face (``FACE_NORMAL``);
* a compound unit: local +Z is the unit's optical axis; members sit at their poses in the unit frame.

A component is placed by its ``nominal_pose`` (``component_template`` places it at the origin, unrotated): rotating it
about local Z turns a polarizer's transmission axis, a waveplate's fast axis or a slit's long side.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from .._vendor.shared_contracts.design_catalog import (
    AngleRangeV1,
    BoxGeometryV1,
    CoatingBandV1,
    CoatingKind,
    CylinderGeometryV1,
    FactStatus,
    FrameOrigin,
    FrameReferenceAxis,
    OpticalFrameV1,
    OpticalSymmetry,
    PublicCoatingV1,
)
from .._vendor.shared_contracts.optical_design import (
    OpticalComponentInstanceV1,
    ProjectModelMaterialV1,
    ProjectOpticalModelV1,
    check_optical_models,
)
from .errors import NativeComponentError, NativeErrorCode, NativeProblem, infeasible, invalid
from .materials import MaterialResolver
from .numbers import LENGTH_QUANTUM_MM, ONE, ZERO, ceil_to, floor_to, text, to_decimal

#: The angle-of-incidence range of a declared coating when none is stated: an idealized coating acts the same at every
#: angle (a run flags light outside a narrower declared range as outside the model).
IDEAL_AOI_DEG = (Decimal(0), Decimal(90))
#: The most characters of a coating id (``CoatingId``).
MAX_COATING_ID = 96
COATING_DIGEST_LENGTH = 10
#: The most characters of a model id (``ModelId``).
MAX_MODEL_ID = 64
DECLARED_COATING_KEYS = frozenset({"reflectance", "valid_aoi_deg", "coating_id", "display_name"})
ANGULAR_COATING_KEYS = frozenset({"angular_spectrum", "valid_aoi_deg", "coating_id", "display_name", "basis"})
#: Drawn thickness of a thin element (stop plate, Jones element, ideal lens, uncoated-substrate splitter) when none is given.
THIN_ELEMENT_THICKNESS_MM = Decimal(1)
#: Clear aperture of an element as a fraction of its outer diameter when none is given.
DEFAULT_CLEAR_FRACTION = Decimal("0.9")
COATING_ABBREVIATIONS = {
    CoatingKind.ANTI_REFLECTION: "AR",
    CoatingKind.METALLIC_REFLECTOR: "METAL",
    CoatingKind.DIELECTRIC_HIGH_REFLECTOR: "HR",
    CoatingKind.BEAMSPLITTER: "BS",
}


@dataclass(frozen=True)
class NativeBuild:
    """What a native class produces: the model a component uses (``optical_model``), the member models of a compound
    unit (``member_models``, empty otherwise), the materials and coatings they use, a ready component instance using the
    model (``component_template``) and the design report (``design``: the computed values and the equations used).

    ``as_dict`` writes ``model_materials`` / ``model_coatings`` as ``null`` when the build uses none, like the Project fields
    they go into (which hold at least one entry or are absent): each can be copied into a Project as it is."""

    class_id: str
    optical_model: ProjectOpticalModelV1
    member_models: tuple[ProjectOpticalModelV1, ...]
    model_materials: tuple[ProjectModelMaterialV1, ...]
    model_coatings: tuple[PublicCoatingV1, ...]
    component_template: OpticalComponentInstanceV1
    design: dict[str, Any] = field(default_factory=dict)

    @property
    def models(self) -> tuple[ProjectOpticalModelV1, ...]:
        return (*self.member_models, self.optical_model)

    def as_dict(self) -> dict[str, Any]:
        def dump(model: Any) -> dict[str, Any]:
            return model.model_dump(mode="json", by_alias=True, exclude_none=True)

        return {
            "class_id": self.class_id,
            "optical_model": dump(self.optical_model),
            "member_models": [dump(m) for m in self.member_models],
            "model_materials": [dump(m) for m in self.model_materials] or None,
            "model_coatings": [dump(c) for c in self.model_coatings] or None,
            "component_template": dump(self.component_template),
            "design": self.design,
        }


@dataclass
class BuildContext:
    """Per-request state: the material / coating sources (one catalog load at most)."""

    resolver: MaterialResolver


def unique_by(items, key: str) -> tuple:
    """``items`` without repeats of the same id (the first kept); ``NATIVE_DESIGN_INFEASIBLE`` for one id with two
    different contents."""

    seen: dict[str, Any] = {}
    for item in items:
        ident = getattr(item, key)
        if ident in seen and seen[ident] != item:
            raise infeasible(f"two different definitions share the id {ident!r}; give them distinct ids")
        seen.setdefault(ident, item)
    return tuple(seen.values())


def finish(class_id: str, model: ProjectOpticalModelV1, *, members: tuple[ProjectOpticalModelV1, ...] = (),
           materials=(), coatings=(), role: str, values: Mapping[str, Any], design: dict[str, Any],
           selected_parameters: list[dict[str, Any]] | None = None) -> NativeBuild:
    """Check the produced models with the Project contract's own cross-reference rules and wrap them
    (``selected_parameters``: entries of the component template, e.g. an attenuator's position)."""

    materials_t = tuple(sorted(unique_by(materials, "material_id"), key=lambda m: m.material_id))
    coatings_t = tuple(sorted(unique_by(coatings, "coating_id"), key=lambda c: c.coating_id))
    try:
        check_optical_models((*members, model), materials_t, coatings_t)
    except ValueError as exc:
        raise infeasible(str(exc)) from None
    instance_id = values.get("instance_id") or model.model_id
    name = model.display_name or model.model_id
    try:
        template = OpticalComponentInstanceV1.model_validate({
            "instance_id": instance_id, "optical_role": role,
            "custom_part": {"custom_part_key": model.model_id, "description": f"{name} (native component {class_id})"},
            "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "optical_model_id": model.model_id, **({"selected_parameters": selected_parameters} if selected_parameters else {}),
        })
    except ValidationError as exc:  # pragma: no cover - every field above is checked by the library
        raise invalid(f"component template: {exc.errors(include_url=False)[0]['msg']}", "instance_id") from None
    return NativeBuild(class_id, model, tuple(members), materials_t, coatings_t, template, design)


def optical_model(model_id: str, display_name: str | None, physics: Any, *, frame: OpticalFrameV1 | None = None,
                  geometry: Any = None, parameter: str | None = None, optic_axes: list[dict[str, Any]] | None = None) -> ProjectOpticalModelV1:
    """A ``ProjectOpticalModelV1`` (with the optic axes of its crystals, when it has any); a contract refusal (surfaces
    meeting, an aperture beyond the outline, ...) is ``NATIVE_DESIGN_INFEASIBLE`` (about ``parameter`` when the caller knows
    which one decides it)."""

    try:
        return ProjectOpticalModelV1.model_validate({
            "model_id": model_id, "display_name": display_name, "model": physics, "frame": frame,
            "design_geometry": geometry, "optic_axes": optic_axes,
        })
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        where = ".".join(str(p) for p in first["loc"])
        message = str(first["msg"]).removeprefix("Value error, ")
        raise NativeComponentError([NativeProblem(NativeErrorCode.DESIGN_INFEASIBLE,
                                                  f"the element is not physically valid ({where}: {message})", parameter)]) from None


def transmissive_frame(rotational: bool = True) -> OpticalFrameV1:
    return OpticalFrameV1(origin=FrameOrigin.FIRST_SURFACE_VERTEX,
                          symmetry=OpticalSymmetry.ROTATIONAL_ABOUT_LOCAL_Z if rotational else OpticalSymmetry.NONE)


def face_frame(split: bool = False, rotational: bool = True) -> OpticalFrameV1:
    return OpticalFrameV1(origin=FrameOrigin.REFLECTING_FACE_CENTER, reference_axis=FrameReferenceAxis.FACE_NORMAL,
                          nominal_propagation="SPLIT_ABOUT_+Z" if split else "REFLECTED_ABOUT_+Z",
                          symmetry=OpticalSymmetry.ROTATIONAL_ABOUT_LOCAL_Z if rotational else OpticalSymmetry.NONE)


def cylinder(diameter: Decimal, z_min: Decimal, z_max: Decimal) -> CylinderGeometryV1:
    """A nominal cylindrical body about local Z, its bounds rounded outward to ``LENGTH_QUANTUM_MM``."""

    return CylinderGeometryV1(diameter_mm=text(ceil_to(diameter, LENGTH_QUANTUM_MM)), z_min_mm=text(floor_to(z_min, LENGTH_QUANTUM_MM)),
                              z_max_mm=text(ceil_to(z_max, LENGTH_QUANTUM_MM)), status=FactStatus.NOMINAL)


def box(half_x: Decimal, half_y: Decimal, z_min: Decimal, z_max: Decimal) -> BoxGeometryV1:
    half_z = (z_max - z_min) / 2
    center = (z_max + z_min) / 2
    return BoxGeometryV1(center_mm=("0", "0", text(center, LENGTH_QUANTUM_MM)),
                         half_extents_mm=(text(ceil_to(half_x, LENGTH_QUANTUM_MM)), text(ceil_to(half_y, LENGTH_QUANTUM_MM)),
                                          text(ceil_to(half_z, LENGTH_QUANTUM_MM))), status=FactStatus.NOMINAL)


def band(low: Decimal, high: Decimal, status: FactStatus = FactStatus.DERIVED) -> dict[str, str]:
    return {"min_nm": text(low), "max_nm": text(high), "status": status.value}


def _digest(document: object) -> str:
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:COATING_DIGEST_LENGTH].upper()


def declared_coating(kind: CoatingKind, low: Decimal, high: Decimal, reflectance: Decimal, transmittance: Decimal, *,
                     polarized: tuple[Decimal, Decimal, Decimal, Decimal] | None = None, aoi: tuple[Decimal, Decimal] = IDEAL_AOI_DEG,
                     coating_id: str | None = None, display_name: str | None = None, basis: str, parameter: str) -> PublicCoatingV1:
    """A coating whose R / T the user declares: one band over ``[low, high]`` nm, optionally per polarization
    (``polarized`` = R_s, R_p, T_s, T_p; R / T are then their means)."""

    entry: dict[str, Any] = {"min_nm": text(low), "max_nm": text(high), "reflectance": text(reflectance),
                             "transmittance": text(transmittance), "status": FactStatus.DESIGN_BASIS.value}
    if polarized is not None:
        r_s, r_p, t_s, t_p = polarized
        entry.update(reflectance_s=text(r_s), reflectance_p=text(r_p), transmittance_s=text(t_s), transmittance_p=text(t_p))
    abbreviation = COATING_ABBREVIATIONS.get(kind, kind.value)
    if coating_id is None:
        coating_id = (f"NATIVE-{abbreviation}_{text(low)}-{text(high)}R{text(reflectance)}T{text(transmittance)}"
                      f"_AOI{text(aoi[0])}-{text(aoi[1])}")
        if polarized is not None or len(coating_id) > MAX_COATING_ID:
            coating_id = f"NATIVE-{abbreviation}-{_digest({'band': entry, 'aoi': [text(a) for a in aoi]})}"
    document = {
        "coating_id": coating_id, "kind": kind.value,
        "display_name": display_name or f"Declared {kind.value.replace('_', ' ').lower()} R={text(reflectance)}",
        "bands": [entry],
        "valid_aoi_deg": {"min_deg": text(aoi[0]), "max_deg": text(aoi[1]), "status": FactStatus.DESIGN_BASIS.value},
        "basis": basis,
    }
    try:
        return PublicCoatingV1.model_validate(document)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        raise invalid(f"not a valid coating: {'.'.join(str(p) for p in first['loc'])}: {first['msg']}", parameter) from None


def aoi_range(raw: Any, parameter: str) -> tuple[Decimal, Decimal]:
    if raw is None:
        return IDEAL_AOI_DEG
    if isinstance(raw, str):
        raw = raw.split(",")
    values = [to_decimal(v) for v in raw] if isinstance(raw, (list, tuple)) and len(raw) == 2 else None
    if values is None or any(v is None for v in values) or not ZERO <= values[0] <= values[1] <= IDEAL_AOI_DEG[1]:  # type: ignore[operator]
        raise invalid("valid_aoi_deg must be [min_deg, max_deg] with 0 <= min_deg <= max_deg <= 90", parameter)
    return values[0], values[1]  # type: ignore[return-value]


def angular_coating(spec: Mapping[str, Any], parameter: str) -> PublicCoatingV1:
    """A declared angle-resolved anti-reflection coating: ``{"angular_spectrum": {wavelength_nm, aoi_deg, reflectance_s,
    reflectance_p, transmittance_s, transmittance_p}, "valid_aoi_deg"?: [a, b], "coating_id"?, "display_name"?, "basis"?}``
    (``CoatingAngularSpectrumV1``: s / p power fractions per wavelength row and angle column, read in the incident medium,
    interpolated bilinearly, never extrapolated). ``valid_aoi_deg`` defaults to the table's angle span."""

    if set(spec) - ANGULAR_COATING_KEYS:
        raise invalid(f"an angle-resolved coating takes {sorted(ANGULAR_COATING_KEYS)}", parameter)
    table = spec["angular_spectrum"]
    if not isinstance(table, Mapping):
        raise invalid("angular_spectrum is an object (CoatingAngularSpectrumV1)", parameter)
    table = {"status": FactStatus.DESIGN_BASIS.value, **table}
    angles = table.get("aoi_deg") or ()
    raw = spec.get("valid_aoi_deg")
    low, high = aoi_range(raw, parameter) if raw is not None else (to_decimal(angles[0]) if angles else None,
                                                                   to_decimal(angles[-1]) if angles else None)
    if low is None or high is None:
        raise invalid("angular_spectrum.aoi_deg lists the table's angles", parameter)
    coating_id = spec.get("coating_id") or f"NATIVE-AR-AOI-{_digest({'table': table})}"
    document = {"coating_id": coating_id, "kind": CoatingKind.ANTI_REFLECTION.value,
                "display_name": spec.get("display_name") or "Declared angle-resolved anti-reflection coating", "angular_spectrum": table,
                "valid_aoi_deg": {"min_deg": text(low), "max_deg": text(high), "status": FactStatus.DESIGN_BASIS.value},
                "aoi_reference": "INCIDENT_MEDIUM", "basis": spec.get("basis") or "Declared angle-resolved anti-reflection coating"}
    try:
        return PublicCoatingV1.model_validate(document)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        raise invalid(f"not a valid coating: {'.'.join(str(p) for p in first['loc'])}: {first['msg']}", parameter) from None


def resolve_ar_coating(spec: Any, low: Decimal, high: Decimal, context: BuildContext, parameter: str) -> PublicCoatingV1:
    """An anti-reflection coating for refracting surfaces: a public catalog coating id (copied verbatim),
    ``{"reflectance": R, "valid_aoi_deg": [a, b]}`` -- a declared lossless coating (T = 1 - R) over the element's band --, or
    ``{"angular_spectrum": {...}}``: a declared table over wavelength and angle of incidence (``angular_coating``)."""

    if isinstance(spec, str):
        catalog = context.resolver.catalog(parameter)
        coating = catalog.coating(spec)
        if coating is None:
            raise NativeComponentError([NativeProblem(NativeErrorCode.CATALOG_DATA_UNAVAILABLE,
                                                      f"public catalog {catalog.catalog_version} has no coating {spec!r}", parameter)])
        return coating
    if isinstance(spec, Mapping) and "angular_spectrum" in spec:
        return angular_coating(spec, parameter)
    if not isinstance(spec, Mapping) or "reflectance" not in spec or set(spec) - DECLARED_COATING_KEYS:
        raise invalid('expected a catalog coating id, {"reflectance": R, "valid_aoi_deg": [min, max]} or {"angular_spectrum": {...}}',
                      parameter)
    reflectance = to_decimal(spec["reflectance"])
    if reflectance is None or not ZERO <= reflectance < ONE:
        raise invalid("reflectance must lie in [0, 1)", parameter)
    return declared_coating(CoatingKind.ANTI_REFLECTION, low, high, reflectance, ONE - reflectance,
                            aoi=aoi_range(spec.get("valid_aoi_deg"), parameter), coating_id=spec.get("coating_id"),
                            display_name=spec.get("display_name"), basis="Declared anti-reflection coating (lossless: T = 1 - R)",
                            parameter=parameter)


def default_model_id(prefix: str, *parts: object) -> str:
    """A readable model id: ``prefix`` and the key values (``-`` as ``m``, ``.`` as ``p``), at most 64 characters."""

    pieces = [prefix]
    for part in parts:
        if isinstance(part, Decimal):
            part = text(part).replace("-", "m").replace(".", "p")
        pieces.append(str(part))
    model_id = "-".join(pieces)
    model_id = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in model_id)
    return model_id[:MAX_MODEL_ID]


__all__ = [
    "BuildContext",
    "DEFAULT_CLEAR_FRACTION",
    "IDEAL_AOI_DEG",
    "NativeBuild",
    "THIN_ELEMENT_THICKNESS_MM",
    "aoi_range",
    "band",
    "box",
    "cylinder",
    "declared_coating",
    "default_model_id",
    "face_frame",
    "finish",
    "optical_model",
    "resolve_ar_coating",
    "transmissive_frame",
]
