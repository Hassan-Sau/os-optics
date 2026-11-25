"""Structured, fail-closed errors of the local pure-optical workflow.

The workflow never guesses: an input that cannot be simulated exactly as given by the Project and the catalog
raises ``LocalRunError`` listing every problem found, each with a stable ``LocalRunErrorCode``; no physics value is
ever assumed in its place. A Project that *can* be lowered always runs; an Engine outcome such as ``FAIL`` or
``OUT_OF_MODEL_SCOPE`` is then reported as-is in the result and the local validation record, never raised and never
upgraded to ``PASS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class LocalRunErrorCode(StrEnum):
    """Why the local runtime refuses an input (stable values).

    Catalog / model: ``CATALOG_INVALID`` (not a valid ``PublicOpticalCatalogV2``), ``CATALOG_NOT_AVAILABLE`` (no
    usable catalog release on this device -- or not the one the Project pins -- and none could be obtained; run
    ``os-optics catalog sync`` or ``os-optics catalog import <file>``; when releases are stored, none is
    the active one and this Engine reads none of them, the message carries ``CATALOG_ACTIVE_UNKNOWN`` and also names
    ``os-optics catalog use <catalog_version>``), ``ENGINE_UPGRADE_REQUIRED`` (the
    release is written in a catalog schema this Engine version cannot read), ``CATALOG_QUERY_INVALID`` (a catalog browsing query with an unknown category / form, a refused focal-length filter, ``details.reason``, or a bad paging value),
    ``OPTICAL_SKU_NOT_PUBLISHED``, ``CATALOG_VERSION_MISMATCH``, ``SIMULATION_MODEL_AMBIGUOUS``,
    ``SIMULATION_MODEL_UNSUPPORTED`` (a model kind, a model or coating feature, a source feature or a power rating this
    runtime does not simulate), ``PRODUCT_ROLE_MISMATCH`` (the product's category may not play the component's
    ``optical_role``; ``PublicOpticalProductV2.design_roles``), ``SURFACE_COATING_REQUIRED`` (a splitting surface with the same
    medium on both sides that states no coating; a reflecting surface without one is an ideal reflector). ``EXAMPLE_UNAVAILABLE``: the catalog holds no product an example design
    needs.

    Targets: ``TARGET_INVALID`` (a metric the local runtime does not evaluate, or an invalid target), and
    ``TARGET_PARAMETER_INVALID`` (location ``target <target_id> parameters.<key>``): a ``parameters`` key the target's
    metric does not take, or a value of the wrong type (``validation/target_parameters.py``). A Project's target with such a
    problem (or with ``UNKNOWN_SOURCE_REFERENCE``) does not refuse the Project: the check lists it as a warning and the run
    reports that target ``UNKNOWN`` with the code (``lowering.target_problems``).

    Declared optical models (``ProjectDocumentV2.optical_models``): ``OPTICAL_MODEL_INVALID`` (location ``component
    <instance_id>`` or ``... member <member_id>``): a declared model the runtime cannot place truthfully -- its stated
    frame contradicts the model (other than in its symmetry flag, which the model decides), no frame follows from it, or
    its stated convex-polyhedron design geometry does not contain it (a cylinder or box is extended to hold it);
    ``OPTICAL_MODEL_ROLE_MISMATCH``: the component's ``optical_role`` is not one the model kind can play
    (``lowering.declared_model_roles``). ``CUSTOM_PART_UNSUPPORTED``: a user-defined optical element without an
    ``optical_model_id``. ``BIREFRINGENT_MEDIUM_UNSUPPORTED``: an observation plane whose ``medium`` is a declared crystal
    (a ``model_materials`` entry with ``birefringence``): light is traced in a crystal only between the crystal's own faces,
    and is observed outside crystals (location ``observation <observation_id>``, ``details.material_id``).
    ``GRADIENT_INDEX_UNSUPPORTED``: the same for an observation plane whose ``medium`` is a declared gradient-index material
    (``gradient``).

    A wavelength or an angle outside a published model's validity is not a refusal: the Project runs, the run lists it in
    ``model_scope_issues`` (for a tabulated thin lens also as an Engine issue of status ``OUT_OF_MODEL_SCOPE``), and its status is at least ``OUT_OF_MODEL_SCOPE`` (``FAIL`` when a target also fails).

    Optical solids (``os_optics.optical_solids``, checked on the compiled optical bodies before any ray is traced;
    location ``component <instance_id>[ member <member_id>][ surface <surface_id>]``): ``SAG_DOMAIN_INSIDE_OUTLINE`` (a
    surface profile is not defined everywhere inside the outline it is drawn and traced over), ``SURFACES_INTERSECT`` (two
    consecutive surfaces of an element cross somewhere over its outline: a negative edge or centre thickness),
    ``MEDIUM_BOUNDARY_INCONSISTENT`` (the media on the two sides of consecutive surfaces do not chain),
    ``OPTICAL_BODIES_OVERLAP`` (two placed optical bodies share volume), ``OPTICAL_BODIES_TOUCH`` (two bodies touch where
    light would cross from one into the other -- on traced surfaces inside their clear apertures, e.g. a lens's flat back
    against a window, or two vertices on one axis: the trace cannot pass two surfaces at one point; merge them into one
    sequence with a cemented surface or separate them; a contact anywhere else is allowed) and
    ``MEMBER_OUTSIDE_ENVELOPE`` (a member of a compound unit reaches outside the unit's envelope). These are optical
    solids, not lab mechanics: mounts, posts and boards are never required.

    Sample emission (``FLUORESCENCE`` / ``RAMAN`` sources): ``SAMPLE_EXCITATION_NOT_KNOWN`` (location ``component
    <instance_id>``): the power of the excitation wavelength arriving at the sample's plane is not a known quantity in the trace
    of the Project's other sources (light of unknown fate may reach it, part of it is bounded only, the trace of the
    excitation failed, or the trace with the samples' emission delivers a different excitation), so the emission it derives is
    not known either; ``details``: ``observation_id``, ``wavelength_nm``, ``quantity_state``.

    Configurations (``ProjectDocumentV2.configurations``): ``CONFIGURATION_NOT_SELECTED`` (a Project that declares
    configurations given to an evaluation of one state -- a paraxial or network analysis, a workspace run, a single run:
    evaluate ``ProjectDocumentV2.configured(id)``, or run every configuration with ``run_configured_project`` /
    ``os-optics project run``), ``CONFIGURATION_UNKNOWN`` (a configuration selection names an id the Project does not
    declare, repeats one, or the Project declares no configurations).

    Run directory (``json_api.run_project`` with ``output_dir``; nothing is written): ``RUN_DIRECTORY_OCCUPIED`` (a file
    there the Engine did not write, at a name the run writes: ``artifacts.RunDirectoryOccupiedError``; move it, or write
    the run into another directory) and ``RUN_DIRECTORY_LINK`` (the directory, or a name the run writes, is a link:
    ``artifacts.RunDirectoryLinkError``).
    """

    PROJECT_INVALID = "PROJECT_INVALID"
    CONFIGURATION_NOT_SELECTED = "CONFIGURATION_NOT_SELECTED"
    CONFIGURATION_UNKNOWN = "CONFIGURATION_UNKNOWN"
    CATALOG_INVALID = "CATALOG_INVALID"
    CATALOG_NOT_AVAILABLE = "CATALOG_NOT_AVAILABLE"
    ENGINE_UPGRADE_REQUIRED = "ENGINE_UPGRADE_REQUIRED"
    CATALOG_QUERY_INVALID = "CATALOG_QUERY_INVALID"
    OPTICAL_SKU_NOT_PUBLISHED = "OPTICAL_SKU_NOT_PUBLISHED"
    CATALOG_VERSION_MISMATCH = "CATALOG_VERSION_MISMATCH"
    SIMULATION_MODEL_AMBIGUOUS = "SIMULATION_MODEL_AMBIGUOUS"
    SIMULATION_MODEL_UNSUPPORTED = "SIMULATION_MODEL_UNSUPPORTED"
    PRODUCT_ROLE_MISMATCH = "PRODUCT_ROLE_MISMATCH"
    SURFACE_COATING_REQUIRED = "SURFACE_COATING_REQUIRED"
    EXAMPLE_UNAVAILABLE = "EXAMPLE_UNAVAILABLE"
    CUSTOM_PART_UNSUPPORTED = "CUSTOM_PART_UNSUPPORTED"
    SELECTED_PARAMETERS_UNSUPPORTED = "SELECTED_PARAMETERS_UNSUPPORTED"
    SOURCE_MISSING = "SOURCE_MISSING"
    SOURCE_MODEL_MISSING = "SOURCE_MODEL_MISSING"
    SOURCE_PRODUCT_UNSUPPORTED = "SOURCE_PRODUCT_UNSUPPORTED"
    UNKNOWN_SOURCE_REFERENCE = "UNKNOWN_SOURCE_REFERENCE"
    TARGET_INVALID = "TARGET_INVALID"
    TARGET_PARAMETER_INVALID = "TARGET_PARAMETER_INVALID"
    ENGINE_INPUT_INVALID = "ENGINE_INPUT_INVALID"
    ENGINE_RUN_INCOMPLETE = "ENGINE_RUN_INCOMPLETE"
    BINDING_INCONSISTENT = "BINDING_INCONSISTENT"
    OPTICAL_VIEW_BINDING_FAILED = "OPTICAL_VIEW_BINDING_FAILED"
    SAG_DOMAIN_INSIDE_OUTLINE = "SAG_DOMAIN_INSIDE_OUTLINE"
    SURFACES_INTERSECT = "SURFACES_INTERSECT"
    MEDIUM_BOUNDARY_INCONSISTENT = "MEDIUM_BOUNDARY_INCONSISTENT"
    OPTICAL_BODIES_OVERLAP = "OPTICAL_BODIES_OVERLAP"
    OPTICAL_BODIES_TOUCH = "OPTICAL_BODIES_TOUCH"
    MEMBER_OUTSIDE_ENVELOPE = "MEMBER_OUTSIDE_ENVELOPE"
    OPTICAL_MODEL_INVALID = "OPTICAL_MODEL_INVALID"
    OPTICAL_MODEL_ROLE_MISMATCH = "OPTICAL_MODEL_ROLE_MISMATCH"
    BIREFRINGENT_MEDIUM_UNSUPPORTED = "BIREFRINGENT_MEDIUM_UNSUPPORTED"
    SAMPLE_EXCITATION_NOT_KNOWN = "SAMPLE_EXCITATION_NOT_KNOWN"
    GRADIENT_INDEX_UNSUPPORTED = "GRADIENT_INDEX_UNSUPPORTED"
    RUN_DIRECTORY_OCCUPIED = "RUN_DIRECTORY_OCCUPIED"
    RUN_DIRECTORY_LINK = "RUN_DIRECTORY_LINK"


@dataclass(frozen=True)
class LocalRunProblem:
    """One reason: its stable ``code``, a factual ``message``, where (``location``) and, for an optical-solid finding,
    ``details``: the structured fields of the finding (``instance_id``, ``member_id``, ``surface_id``, the other body's
    ``other_instance_id`` / ``other_member_id`` / ``other_surface_id``, ``position_mm`` (design frame) or
    ``local_position_mm`` (element frame), ``value_mm`` -- an overlap depth, a thickness -- and ``limit_mm``); for a refused
    focal-length filter, ``details.reason`` (``FocalQueryRefusal``)."""

    code: LocalRunErrorCode
    message: str
    location: str | None = None
    details: dict[str, Any] | None = field(default=None, compare=False)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.location is not None:
            data["location"] = self.location
        if self.details:
            data["details"] = dict(self.details)
        return data


class LocalRunError(ValueError):
    """The Project cannot be simulated as given. ``problems`` lists every reason found; ``optical_solids``: the whole
    optical-solid report (``OpticalSolidsReport.as_dict()``) when the refusal comes from the solid check."""

    def __init__(self, problems: list[LocalRunProblem] | tuple[LocalRunProblem, ...], *, optical_solids: dict[str, Any] | None = None) -> None:
        self.problems: tuple[LocalRunProblem, ...] = tuple(problems)
        self.optical_solids = optical_solids
        super().__init__("; ".join(f"{p.code.value}{f' at {p.location}' if p.location else ''}: {p.message}" for p in self.problems))

    @property
    def codes(self) -> tuple[LocalRunErrorCode, ...]:
        return tuple(p.code for p in self.problems)

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"status": "ERROR", "problems": [p.as_dict() for p in self.problems]}
        if self.optical_solids is not None:
            data["optical_solids"] = self.optical_solids
        return data


def fail(code: LocalRunErrorCode, message: str, location: str | None = None) -> LocalRunError:
    return LocalRunError([LocalRunProblem(code, message, location)])
