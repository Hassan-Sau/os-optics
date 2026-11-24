"""Every element kind is classified at every place that decides per kind.

The kinds come from their definitions: the Engine interactions (``contracts.optics.Interaction``), the interactions of the
lowered scene (``scene/lower.py``), the Project model kinds (``ProjectSimulationModelV1``) and the Optical Simulation View
scene (``OPTIC_MODEL_KINDS``, ``OpticSurfaceRoleV1``). Each per-kind table or set must name every kind exactly once, so a
kind added to a contract fails here until each place states what it does with it -- instead of falling through to a
default (no returned light, no ledger sink, no phase refusal, an unrotated member, a surface sequence). A place this test
reads but does not own (``optics/rays/tracer.py``) is read from its source.

The complete headless kind tables are verified against the Engine's own contracts.
"""

from __future__ import annotations

import inspect
import re
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

from os_optics._vendor.shared_contracts.design_catalog import PolarizationElementKind
from os_optics._vendor.shared_contracts.design_catalog.optical_catalog import MODEL_KINDS_BY_CATEGORY
from os_optics._vendor.shared_contracts.optical_design import FRAME_FREE_KINDS, ProjectSimulationModelV1
from os_optics._vendor.shared_contracts.optical_view import OpticBodyGeometryV1, OpticSurfaceRoleV1
from os_optics._vendor.shared_contracts.optical_view import scene as scene_contract
from os_optics.contracts.optics import Interaction
from os_optics.local import compile as compile_module
from os_optics.local import lowering
from os_optics.local.project_ops import PROJECT_OPS
from os_optics.network import export as network_export
from os_optics.network.errors import NetworkErrorCode
from os_optics.optical_view import builder, glb, optic_mesh, section
from os_optics.optics.polarization import routing
from os_optics.optics.rays import tracer
from os_optics.scene import lower
from os_optics.scene.model import InteractionG
from os_optics.validation import resonance
from os_optics.viewer import palette

SRC = Path(lower.__file__).resolve().parents[1]


def _union_members(annotated) -> tuple[type, ...]:
    return typing.get_args(typing.get_args(annotated)[0])


def _literal(model: type, field: str) -> str:
    [value] = typing.get_args(model.model_fields[field].annotation)
    return value


#: Engine interaction kinds (``Interaction.kind``) and their classes.
INTERACTIONS = {_literal(cls, "kind"): cls for cls in _union_members(Interaction)}
#: Interaction kinds of the lowered scene (``InteractionG.kind``): what ``scene/lower.py`` builds -- a tabulated thin lens
#: is lowered to ``thin_lens`` with its table; observation planes are ``observe``.
LOWERED = frozenset(re.findall(r'InteractionG\("(\w+)"', (SRC / "scene" / "lower.py").read_text(encoding="utf-8")))
#: Project model kinds and versions (``ProjectSimulationModelV1``).
MODELS = {_literal(cls, "model_kind"): _literal(cls, "model_version") for cls in _union_members(ProjectSimulationModelV1)}
MODEL_KINDS = frozenset(MODELS)
#: Model kinds a catalog product may publish (``design_catalog.optical_catalog``).
CATALOG_KINDS = frozenset().union(*MODEL_KINDS_BY_CATEGORY.values())
#: Catalog-only unit kinds: a catalog product's model of two or more bodies (a polarizing prism's ``JOINED_BODIES``),
#: compiled like a ``COMPOUND`` unit of members and drawn as the ``OPTICAL_UNIT`` of its bodies; a Project never declares one
#: (it joins members of a ``COMPOUND`` unit), so it is no Project model kind and no scene body kind.
CATALOG_UNIT_KINDS = frozenset({"JOINED_BODIES"})
#: Optic-body model kinds and surface roles of the Optical Simulation View scene.
BODY_KINDS = frozenset(scene_contract.OPTIC_MODEL_KINDS)
SURFACE_ROLES = frozenset(role.value for role in OpticSurfaceRoleV1)


def _partition(*parts: frozenset[str] | set[str]) -> frozenset[str]:
    """The union of ``parts``, which must not overlap."""

    union: set[str] = set()
    for part in parts:
        assert not union & set(part), f"{sorted(union & set(part))} classified twice"
        union |= set(part)
    return frozenset(union)


# ------------------------------------------------------------------------------------------------ the vocabularies


def test_the_vocabularies_are_what_this_test_was_written_for():
    """A changed vocabulary is the moment to walk the extension checklist (every per-kind table and its documentation);
    then update these lists."""

    assert set(INTERACTIONS) == {"refract", "mirror", "thin_lens", "tabulated_thin_lens", "beam_splitter", "polarizer", "retarder",
                                 "absorber", "aperture_stop", "neutral_density", "faraday_rotator", "diffraction_grating",
                                 "complex_transmission"}
    assert LOWERED == {"refract", "mirror", "thin_lens", "beam_splitter", "polarizer", "retarder", "absorber", "aperture_stop",
                       "neutral_density", "faraday_rotator", "diffraction_grating", "complex_transmission", "observe"}
    assert MODEL_KINDS == {"SURFACE_SEQUENCE", "SURFACE_SET", "POLARIZATION_ELEMENT", "NEUTRAL_DENSITY", "APERTURE_STOP",
                           "IDEAL_THIN_LENS", "TABULATED_THIN_LENS", "FARADAY_ROTATOR", "DIFFRACTION_GRATING", "COMPOUND", "FACET_SOLID",
                           "NONLINEAR_CRYSTAL"}
    assert CATALOG_KINDS == {"SURFACE_SEQUENCE", "SURFACE_SET", "POLARIZATION_ELEMENT", "NEUTRAL_DENSITY", "JOINED_BODIES"}
    assert SURFACE_ROLES == {"REFRACT", "REFLECT", "SPLIT", "POLARIZATION_ELEMENT", "APERTURE_STOP", "THIN_LENS", "ATTENUATOR", "FARADAY_ROTATOR", "GRATING"}


def test_every_engine_interaction_is_lowered():
    source = inspect.getsource(lower._lower_interaction)
    for kind, cls in INTERACTIONS.items():
        assert f"co.{cls.__name__}" in source, f"_lower_interaction does not lower {kind}"
    assert LOWERED - {"observe"} == (set(INTERACTIONS) - {"tabulated_thin_lens"})


# ------------------------------------------------------------------------------------------------ trace and checks


def test_the_tracer_has_a_handler_for_every_lowered_kind():
    # (read from the source: the handler table is built inside Tracer._interact)
    found = re.search(r"handler = \{(.*?)\}\[inter\.kind\]", inspect.getsource(tracer), re.DOTALL)
    assert found is not None, "the tracer's handler table moved: point this test at it"
    assert set(re.findall(r'"(\w+)":', found.group(1))) == LOWERED


#: Lowered kinds whose interaction books no loss sink of its own: an absorber ends the path (its power is the path's
#: ABSORBED termination), an observation plane passes or absorbs the light as an observation.
LOSS_LEDGER_EXEMPT = frozenset({"absorber", "observe"})


def test_every_lowered_kind_has_a_loss_sink_or_is_exempt():
    # tracer._record_interaction_loss books a kind outside INTERACTION_LOSS_SINKS nowhere: the classification is checked here
    assert _partition(set(tracer.INTERACTION_LOSS_SINKS), LOSS_LEDGER_EXEMPT) == LOWERED


def test_resonance_classifies_every_lowered_kind():
    assert _partition(resonance.REFLECTING, resonance.NON_RETURNING) == LOWERED
    assert resonance.BENDING <= resonance.NON_RETURNING


def test_resonance_refuses_an_unclassified_kind():
    surface = SimpleNamespace(interaction=InteractionG(kind="grating"), surface_id="G/S1")
    with pytest.raises(KeyError, match="grating"):
        resonance._coefficients(surface, resonance.EITHER, 0.0, 0.1, None, 633.0)  # type: ignore[arg-type]


def test_polarization_routing_classifies_every_lowered_kind():
    assert _partition(routing.FOLLOWED_INTERACTIONS, routing.NOT_FOLLOWED_INTERACTIONS) == LOWERED - {"observe"}


def test_the_network_export_states_the_phase_of_every_lowered_kind():
    assert set(network_export.INTERACTION_PHASE) == LOWERED
    source = inspect.getsource(network_export)
    for kind, code in network_export.INTERACTION_PHASE.items():
        if code is not None:
            assert isinstance(code, NetworkErrorCode)
            # the table and at least one refusal that raises it
            assert source.count(f"NetworkErrorCode.{code.name}") >= 2, f"{kind}: {code} is never refused"


# ------------------------------------------------------------------------------------------------ compilation


def test_a_placed_member_turns_every_interaction_axis():
    assert _partition(set(compile_module.ORIENTED_INTERACTION_AXES), compile_module.AXIS_FREE_INTERACTIONS) == set(INTERACTIONS)
    for kind, cls in INTERACTIONS.items():
        local = {name for name in cls.model_fields if name.endswith("_local")}
        stated = {compile_module.ORIENTED_INTERACTION_AXES[kind]} if kind in compile_module.ORIENTED_INTERACTION_AXES else set()
        assert local == stated, f"{kind}: element-frame fields {sorted(local)}, turned {sorted(stated)}"


def test_a_placed_member_refuses_an_unclassified_interaction():
    with pytest.raises(KeyError, match="grating"):
        compile_module._placed_interaction(SimpleNamespace(kind="grating"), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


def test_every_model_kind_is_compiled_and_declared():
    # (every Project model kind and every catalog-only unit kind has a compiler; a catalog unit is a located compound)
    assert not CATALOG_UNIT_KINDS & MODEL_KINDS
    assert set(compile_module.MODEL_COMPILERS) == MODEL_KINDS | CATALOG_UNIT_KINDS
    assert CATALOG_UNIT_KINDS <= compile_module._LOCATED_COMPILERS
    assert compile_module.DECLARED_MODELS == set(MODELS.items())
    # (the runtime compiles every kind a catalog product may publish: the scene's catalog body kinds and the catalog units)
    supported = {kind for kind, _ in compile_module.SUPPORTED_MODELS}
    assert supported == set(scene_contract.CATALOG_OPTIC_MODEL_KINDS) | CATALOG_UNIT_KINDS == CATALOG_KINDS
    assert {m for m in compile_module.SUPPORTED_MODELS if m[0] not in CATALOG_UNIT_KINDS} <= compile_module.DECLARED_MODELS
    assert not {m for m in compile_module.DECLARED_MODELS if m[0] in CATALOG_UNIT_KINDS}
    assert "JoinedBodiesModelV1" in inspect.getsource(builder.optic_body)  # (drawn as the OPTICAL_UNIT of its bodies)
    assert FRAME_FREE_KINDS <= MODEL_KINDS


def test_every_declared_model_kind_has_its_roles():
    by_element = {kind.value for kind in PolarizationElementKind}
    assert set(lowering.DECLARED_MODEL_ROLES) == (MODEL_KINDS - {"POLARIZATION_ELEMENT"}) | by_element


def test_add_optical_model_names_every_model_kind():
    summary = PROJECT_OPS["add_optical_model"]["summary"]
    assert {kind for kind in MODEL_KINDS if re.search(rf"\b{kind}\b", summary)} == MODEL_KINDS


# ------------------------------------------------------------------------------------------------ scene and views


def test_the_scene_contract_names_every_body_kind_and_plane_role():
    # a compound unit is an OPTICAL_UNIT of member bodies; a kind compiled as another kind's physics is drawn as that kind
    # (a nonlinear crystal: its slab, a SURFACE_SEQUENCE body)
    assert BODY_KINDS == MODEL_KINDS - {"COMPOUND"} - set(compile_module.COMPILED_AS)
    assert set(compile_module.COMPILED_AS.values()) <= BODY_KINDS
    assert set(typing.get_args(OpticBodyGeometryV1.model_fields["model_kind"].annotation)) == BODY_KINDS
    assert set(scene_contract.CATALOG_OPTIC_MODEL_KINDS) <= BODY_KINDS
    planes = scene_contract._PLANE_ROLE_BY_KIND
    # (a facet solid's body is its facets: it carries no surface, OpticFacetsV1)
    assert set(planes) == BODY_KINDS - {"SURFACE_SEQUENCE", "SURFACE_SET", "FACET_SOLID"}
    assert _partition(set(planes.values()), scene_contract._SEQUENCE_ROLES) == SURFACE_ROLES
    assert set(scene_contract._PLANE_SURFACE_IDS) == set(planes.values())


def test_every_body_kind_is_drawn():
    assert set(optic_mesh.BODY_DRAWINGS) == BODY_KINDS
    assert set(section.SECTION_DRAWINGS) == BODY_KINDS
    with pytest.raises(KeyError, match="GRATING"):
        builder._body({}, SimpleNamespace(model_kind="GRATING"), lambda surface: None)


def test_every_surface_role_has_a_shown_material():
    assert set(optic_mesh.ROLE_MATERIALS) == SURFACE_ROLES
    materials = {m for m in optic_mesh.ROLE_MATERIALS.values() if m is not None} | set(optic_mesh.REFRACT_MATERIALS)
    assert materials <= set(glb.STATIC_MATERIALS)
    assert materials <= set(palette.IDENTITY_MATERIALS)  # a compound unit's legend names its members by them


def test_the_local_viewer_labels_every_surface_role_in_both_languages():
    script = (SRC / "viewer" / "assets" / "viewer.js").read_text(encoding="utf-8")
    english, chinese = script.split('"zh-CN": {', 1)
    for role in SURFACE_ROLES:
        assert re.search(rf"\b{role}: \"", english), f"en label of {role}"
        assert re.search(rf"\b{role}: \"", chinese.split("\n  };", 1)[0]), f"zh-CN label of {role}"
