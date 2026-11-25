"""Project-declared optical models (``ProjectDocumentV2.optical_models``) on the single compile path.

* A declared copy of a catalog product's model (``declared_copy``) compiles to the product's own Engine surfaces --
  only the material namespace differs (``project:`` instead of ``catalog:``) -- and traces identically: every model kind
  of the fixture catalog, the schema-3.0 catalog constructs and one product of every form of the seed catalog.
* Existing Projects keep their payload / osc1 hashes; Engine input hashes change only because unstated bulk absorption
  is no longer serialized as an explicit lossless assumption. The old request hashes remain pinned against exactly that delta.
* A Project of declared models only needs no catalog.
* Frames and design bodies are stated or derived by rule, and refused when they contradict the model.
* Refusals: a custom optic without a model, a model in a role its kind cannot play, the legacy ``materials``.
"""

from __future__ import annotations

import json
import math
import re

import pytest

from os_optics._vendor.shared_contracts.common import DECIMAL_STRING_PATTERN
from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics._version import SCHEMA_VERSION
from os_optics.manifest import hash_exact
from os_optics.local.compile import environment_air
from os_optics._vendor.shared_contracts.optical_design.optical_model import DEFAULT_ENVIRONMENT
from os_optics.local import (
    EXAMPLE_KINDS,
    LocalRunError,
    LocalRunErrorCode,
    compile_element,
    declared_copy,
    example_project_of_kind,
    json_api,
    load_public_catalog,
    lower_project,
    run_optical_project,
)

from catalog3_products import all_products, catalog3_document
from native_classes_b import CLASSES_B
from catalog_store_helpers import REAL_CATALOG
from declared_helpers import (
    compound_model,
    cylinder,
    declared_component,
    declared_project,
    nbk7,
    objective_model,
    sequence,
    stop_model,
    surface,
    thin_lens_model,
    with_models,
    without_namespace,
)
from local_helpers import ALONG_X, catalog_document, element, plane, project, source

#: ``(project osc1 hash, historical Engine input hash)`` of every reference design, recorded before declared models
#: existed (engine 0.6.1 + the declared-model contract, ``ed8312b0``), against the fixture and seed catalogs.
#: Reconstruct only the formerly serialized, implicit alpha=0 for the historical Engine hash below; the actual
#: Engine request omits alpha when loss is not stated, so its content hash correctly differs.
#: The results schema version the hashes below were recorded with.
PINNED_SCHEMA_VERSION = "1.3.0"
PINNED_NUMERICAL_CONFIG_VERSION = "1.4.0"
#: The seed catalog release the ``real:`` hashes were recorded against: a later seed release moves only the products'
#: ``catalog_version`` pins of these designs (their products are published unchanged), which the test puts back.
PINNED_SEED_VERSION = "2026-10-06.3"
#: Designs whose example now carries other targets than it was released with (the interferometer's visibility and
#: coherence margin depend on a group delay its catalog coatings do not state): an existing Project of that design keeps
#: its released targets, and those are what its pins were recorded with.
RELEASED_TARGETS = {
    "two_beam_interference": [
        {"target_id": "fringe_period", "metric": "fringe_period_nm", "observation_id": "crossing_plane",
         "target_value": 780.0 / math.sqrt(2.0), "tolerance_rel": 1e-3, "unit": "nm"},
        {"target_id": "fringe_visibility", "metric": "fringe_visibility", "observation_id": "crossing_plane", "minimum": 0.9, "unit": "1"},
        {"target_id": "coherence_margin", "metric": "coherence_margin", "observation_id": "crossing_plane", "minimum": 0.9, "unit": "1"},
    ],
}
#: (the examples that place their elements by a paraxial design -- focus, folded, beam_expander, folded_expander,
#: relay_4f -- design them in the air they operate in, n_air(20 C, 101325 Pa) ~ 1.00027, since the Engine evaluates indices
#: relative to the environment's air: their computed positions moved by ~3e-4 of the focal distances, so their Project and
#: Engine-input pins were re-recorded. The other designs place nothing by refraction: their pins are unchanged, with the
#: request's explicit ``air`` material put back below.)
EXAMPLE_HASHES = {
    "fixture:beam_expander": ("7af100d482681255918792cede3b818045ef6ff836ec61ec625f122b1133ee24", "f1d30666913ef261b1986c33866ed7c06a253326ac6da856b57f0f052238f524"),
    "fixture:dichroic_combiner": ("6d9b6d74452a22ad6cca7577ffd19f7970374597b4e2c3ac6e1097c7c640acab", "b2cdc3e1b080e3fdd163c81bffe1807248b831283961d0041e35b8dfe07ddbe0"),
    "fixture:dichroic_routing": ("297ebe62b2a57f43db2bc435ca88a26d2b95a714a148fb0d26923b6d352e15cb", "c67c05d65fea8509dd62d21518006832e3e1a35c953eb19f28eb36aba387eff7"),
    "fixture:focus": ("40261960b40e669326419423e8223d363845ea698e4a55ec8fb7a7424a191ed9", "f827959d56392f0939eb2d8623dbaab3fd23bedc6287d15f89f8e3fb69f25db0"),
    "fixture:folded": ("71ad10d51053b40548c50326c2167449280ef59ff51ceb620a0f68b753d96bf7", "3f62fd778eef60a42aa44786fa14025519c6c7f2b319ce074a061bf5aee49fa1"),
    "fixture:folded_expander": ("b8402ffd13a607f96bf35da83ea4b528cb97008f521e4a2f162a99629ed3ff0e", "af116f05dd8312193a9657e8703a60d8fe5d6d69bc5834b09f95093725fe8d78"),
    "fixture:polarization": ("d5a3126c194142dd8cd62ed65ebc4e11ce4dbdc4ebd9325db3195ef85021ebf9", "562a956050183e903b3551abb552804d282c79c7ef551ce1b2a1b2e0a7355e14"),
    "fixture:relay_4f": ("f43c6018874dbf5ac842fdd1773264ceeaa0a848d8a5b91835a1d20533e2ac1b", "35d4cabc5d150c8ab7458983a560857f97a5748f20dfb06dd8e999db7442e351"),
    "fixture:two_beam_interference": ("3087b89b271c0875455c799ebb56e46c985599d071fa2ddc13af953bd7632885", "45d99a6ddc033aba57ac2bffa678e959cdcd4bdf33855fd47fc3369ea86a01b8"),
    "fixture:two_color_dichroic": ("32dd0ada77529363802b6e0d34620f20110473de37fb6fa90a02d2d296864300", "1016a442d0d753b6ac94a23cd54b79f91a1b41e4fe3a003b8504787f8ab4db36"),
    "real:beam_expander": ("210a204f2989b1f5cdccd48f6d851ae27db955597512e7580cbfa844c7a7f368", "96a2e5f151124a080ed1ebb5790f13ffb87f74134895f5e7bbd8165ec1ee0db2"),
    "real:dichroic_combiner": ("d9083c352556f173dde150d0e6ba01744d9483f52e0627df73f16702d8dd0796", "ba7996e6746e6a33ff260ffaa933620fd678d6d6d2c25d525e8df5e5031e0d75"),
    "real:dichroic_routing": ("f3d4bf6a1eb73fc42cdebe391feb074e23f3e1ed9e433146f137b04e62033531", "fa608cca406d5e5913aa0c3bc53594944de023afe3c1420d9e2e6eedf265c478"),
    "real:focus": ("dffbca2f2947493ac349c9a5035dbfc237032d6158bd842c02a98fa3d5c18b2a", "1f770d0735b7b2acd07039acd0f40b178bcec886000851227608f541d113450c"),
    "real:folded": ("27f247a7383a4fa57d623fa631e432560358ae0a55c5435ffcb59d3dafc3d7a2", "fb94faf393ed6d199e7d51b8b57da5fb8811b1ff85ae50223d37df4b004ba755"),
    "real:folded_expander": ("a5582a7683fe3d9838e3d2f3a62ace4bd2ae396b8b8d547f7ce55245c1dcd5bf", "296a8b42e5afdb6abf1b72f02e08bd49046c56f8f1b86c353df344dfec7d928e"),
    "real:polarization": ("370a8522de762cf698afcbf8852a67beaf001872fdb9d120d83fc370b308b803", "bc6bc6974dff5f72b2d515a66bd174c496708f25f56a9e2a8d3105bb0f75a0c3"),
    "real:relay_4f": ("c3aa1f47d80c34e11cad2586ce4045d32684329fd15f674ef7b7a0cfd5feef1d", "57f79443109d9383d54e653412f4c7e10560016f780886d3d44037635cf016c1"),
    "real:two_beam_interference": ("17414758f0e77e5608792dbaf4558833e52ca8cb6541a5ca43ba58432e6c50ef", "ad8b83cb60480cef43d64d52bbba91d7d2ebe6ea39d864aef7ff3966369cd157"),
    "real:two_color_dichroic": ("b53d281244121b349ab920ed70ae80e45dc33af2d5e9220275485f1cd5a21577", "cee65b6b41d101245ebedbe5dc7d0ae13b0a5ddce26ad85dc89b13232483102a"),
}


def _result_text(run) -> str:
    """The entire Engine result without its manifest, neutralizing only material namespaces.

    Unknown bulk loss stays unknown in both copies: keep its warning and every
    bounded-power/target value, but compare its material location by the same
    namespace rule as surfaces and Engine materials.
    """

    result = run.result.model_copy(update={"manifest": None}).model_dump(mode="json")
    for issue in result["issues"]:
        if issue["code"] == "BULK_LOSS_NOT_ESTABLISHED" and issue["location"].startswith("material:project:"):
            issue["location"] = issue["location"].replace("material:project:", "material:catalog:", 1)
    return without_namespace(result)


def _assert_equivalent(document: dict, catalog_source) -> None:
    """The catalog design and its declared copy: the same Engine surfaces up to the material namespace, and the same trace."""

    catalog_run = run_optical_project(document, catalog=catalog_source)
    declared_run = run_optical_project(declared_project(document, catalog_source))
    for a, b in zip(catalog_run.request.project.components, declared_run.request.project.components, strict=True):
        assert without_namespace([s.model_dump(mode="json") for s in a.surfaces]) == without_namespace([s.model_dump(mode="json") for s in b.surfaces])
    assert without_namespace(catalog_run.request.project.model_dump(mode="json")["materials"]) == \
        without_namespace(declared_run.request.project.model_dump(mode="json")["materials"])
    assert _result_text(catalog_run) == _result_text(declared_run)
    assert declared_run.status == catalog_run.status
    assert [(i.instance_id, i.code, i.surface_id) for i in declared_run.model_scope_issues] == \
        [(i.instance_id, i.code, i.surface_id) for i in catalog_run.model_scope_issues]
    assert declared_run.simulated_parts == () and declared_run.catalog_version is None


# ------------------------------------------------------------------------------------- catalog == declared


@pytest.mark.parametrize("kind", EXAMPLE_KINDS)
def test_every_reference_design_traces_identically_with_declared_copies(kind):
    document = example_project_of_kind(kind).payload()
    _assert_equivalent(document, catalog_document())


@pytest.mark.parametrize("sku", [p["optical_sku"] for p in all_products()])
def test_every_schema_3_construct_compiles_and_traces_identically_when_declared(sku):
    """Cube splitters, prisms, a wedge sampler, a rectangular window, an off-axis parabolic mirror, a cylindrical lens and a
    typical-spectrum bandpass filter: a beam along the element's local +Z from 100 mm before it, an observation plane
    100 mm after (whatever the element does to the beam, the declared copy does the same)."""

    document = catalog3_document()
    product = next(p for p in document["products"] if p["optical_sku"] == sku)
    role = sorted(load_public_catalog(document).get(sku).design_roles)[0]
    component = {"instance_id": "E", "optical_role": role, "product": {"optical_sku": sku}, "retained_dofs": [],
                 "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}
    design = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=633.0, waist_mm=0.5), component],
                     [{"observation_id": "after", "pose": {"position_mm": [100.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}])
    assert product["simulation_models"]
    _assert_equivalent(design, document)


def _seed_catalog():
    return load_public_catalog(REAL_CATALOG)


def test_every_seed_catalog_product_compiles_to_the_same_surfaces_when_declared():
    catalog = _seed_catalog()
    from os_optics.local import ProjectResolver, compile_declared

    class _Project:  # the declared model library of one product
        def __init__(self, model, materials, coatings):
            self.optical_models, self.model_materials, self.model_coatings = (model,), materials, coatings

    for product in catalog.products:
        model, materials, coatings = declared_copy(product, catalog, model_id="copy")
        declared = compile_declared("X", model, ProjectResolver(_Project(model, materials, coatings)))
        published = compile_element("X", product, catalog)
        assert without_namespace([s.model_dump(mode="json") for s in declared.surfaces]) == \
            without_namespace([s.model_dump(mode="json") for s in published.surfaces]), product.optical_sku
        assert without_namespace([m.model_dump(mode="json") for m in declared.materials]) == \
            without_namespace([m.model_dump(mode="json") for m in published.materials]), product.optical_sku
        assert declared.identity.frame == product.frame and declared.identity.design_geometry == product.design_geometry


@pytest.mark.parametrize("kind", ["focus", "relay_4f", "dichroic_routing", "polarization", "two_beam_interference"])
def test_seed_catalog_reference_designs_trace_identically_with_declared_copies(kind, real_catalog_store):
    """The reference designs built from the seed catalog (achromats, aspheres, dichroics, waveplates, splitters, as the
    seed release offers them) and their declared copies."""

    document = example_project_of_kind(kind).payload()
    _assert_equivalent(document, REAL_CATALOG)


# ------------------------------------------------------------------------------------------ hash stability


def _without_group_delay_bounds(node, removed=None):
    """``node`` without its ``group_delay_bound_fs`` entries; each removed value is appended to ``removed``."""
    if isinstance(node, dict):
        if "group_delay_bound_fs" in node and removed is not None:
            removed.append(node["group_delay_bound_fs"])
        return {k: _without_group_delay_bounds(v, removed) for k, v in node.items() if k != "group_delay_bound_fs"}
    if isinstance(node, list):
        return [_without_group_delay_bounds(v, removed) for v in node]
    return node


def _without_declared_validity(value):
    """``value`` without the coatings' ``declared_validity`` (the pins predate the field)."""
    if isinstance(value, dict):
        return {k: _without_declared_validity(v) for k, v in value.items() if k != "declared_validity"}
    return [_without_declared_validity(v) for v in value] if isinstance(value, list) else value


def _historical_implicit_zero_loss(request: dict) -> tuple[dict, tuple[str, ...]]:
    """Restore exactly the pre-absence-preserving serializer's implicit alpha=0.

    This is a comparison with the historical request pin, NOT the request sent to
    the Engine: an unstated loss remains absent in the live request and its trace
    must retain the unknown-loss bounds.
    """

    old = json.loads(json.dumps(request))
    omitted = []
    for material in old["project"]["materials"]:
        if "absorption_coefficient_per_mm" not in material:
            omitted.append(material["material_id"])
            material["absorption_coefficient_per_mm"] = 0.0
    return old, tuple(omitted)


@pytest.mark.parametrize("key", sorted(EXAMPLE_HASHES))
def test_existing_projects_keep_project_hashes_and_pin_the_implicit_loss_delta(key, real_catalog_store, fixture_catalog_store, monkeypatch):
    label, kind = key.split(":")
    if label == "fixture":
        monkeypatch.setenv("OPENSCI_CATALOG_DIR", str(fixture_catalog_store))
    project_ = example_project_of_kind(kind)
    if kind in RELEASED_TARGETS:  # (an existing Project of that design, with the targets it was released with)
        payload = project_.payload()
        payload["targets"] = json.loads(json.dumps(RELEASED_TARGETS[kind]))
        project_ = type(project_).model_validate(payload)
    lowered = lower_project(project_)
    # The pins precede this release. Put back the historical schema stamp (1.3.0 before 0.7.0), numerical defaults,
    # air and later catalog fields below; the remaining input delta is precisely omitted implicit alpha=0.
    request = lowered.request.model_dump(mode="json")
    assert request["project"]["project_schema_version"] == SCHEMA_VERSION
    request["project"]["project_schema_version"] = PINNED_SCHEMA_VERSION
    # a compiled coating carries its record (validation/coating_use.py: declared_validity); the pins predate the field
    request = _without_declared_validity(request)
    # the run's declared reach (NumericalConfig.laboratory_reach_mm, config 1.5.0); the pins predate the field
    numerical = request["configuration"]["numerical"]
    del numerical["laboratory_reach_mm"]
    # the crystal tolerances (config 1.6.0); the pins predate the fields
    del numerical["birefringent_interface_warn_aoi_deg"], numerical["birefringent_gaussian_rel_tol"]
    # the return-light analysis budget (config 1.6.0); the pins predate the field
    numerical.pop("max_return_analysis_work", None)
    # the second-harmonic tolerances (config 1.6.0, within 0.13.0); the pins predate the fields
    for name in ("nonlinear_depletion_rel_tol", "nonlinear_focusing_rel_tol", "nonlinear_quadrature_rel_tol",
                 "nonlinear_harmonic_gaussian_rel_tol", "nonlinear_beam_quality_tol", "nonlinear_linewidth_rel_tol"):
        del numerical[name]
    # the caller settings (config 1.7.0): the negligible level, the rating windows, the design-point tolerances and the
    # scatter-origin and mesh-patch budgets; the pins predate the fields
    for name in ("negligible_power_fraction", "max_scatter_origins", "max_mesh_patches", "rating_wavelength_tol_rel",
                 "rating_pulse_duration_tol_rel", "rating_repetition_rate_tol_rel", "rating_beam_diameter_tol_rel",
                 "rating_polarization_fraction_tol", "rating_unlocated_light_share", "design_point_wavelength_rel",
                 "design_point_retarder_wavelength_rel", "design_point_angle_metallic_reflector_deg",
                 "design_point_angle_dielectric_high_reflector_deg", "design_point_angle_anti_reflection_deg",
                 "design_point_angle_beamsplitter_deg", "design_point_interference_shift_fraction",
                 "design_point_interference_angle_max_deg"):
        del numerical[name]
    numerical["config_version"] = PINNED_NUMERICAL_CONFIG_VERSION
    # the environment's air: a native Project operates in the air of its environment -- these state none, so air at
    # 20 C and 101325 Pa -- which the request states as the Engine material ``air`` (it replaced the Engine's unit-index
    # built-in). It is the only new material and changes nothing else in the request; the pins predate it.
    air = environment_air(DEFAULT_ENVIRONMENT).model_dump(mode="json")
    assert request["project"]["materials"][0] == air
    request["project"]["materials"] = request["project"]["materials"][1:]
    document = project_.payload()
    if label == "real":  # with the seed release they were recorded against put back, too
        seed = load_public_catalog(REAL_CATALOG).catalog_version
        assert {c["product"]["catalog_version"] for c in document["components"] if c.get("product")} == {seed}
        document = json.loads(json.dumps(document).replace(f'"{seed}"', f'"{PINNED_SEED_VERSION}"'))
        request = json.loads(json.dumps(request).replace(f'"{seed}"', f'"{PINNED_SEED_VERSION}"'))
        # catalog schema 3.2 (seed 2026-10-28.1) states a group-delay bound on high-reflector, beamsplitter, metallic and
        # anti-reflection coatings; the Engine request carries it on those coatings. The pins predate the field.
        removed: list = []
        request = _without_group_delay_bounds(request, removed)
        # only the generator's class bounds (high reflector / beamsplitter 1000, metallic 100, anti-reflection 50 fs)
        assert set(removed) <= {1000.0, 100.0, 50.0}, removed
    historical, omitted = _historical_implicit_zero_loss(request)
    # These references contain catalog media whose bulk loss was not stated. An
    # explicit zero here would recreate the old digest but falsely make their
    # power known: pin the omission itself, not just its reconstructed digest.
    references_with_unstated_loss = {
        "fixture:beam_expander", "fixture:dichroic_combiner", "fixture:dichroic_routing", "fixture:focus",
        "fixture:folded", "fixture:folded_expander", "fixture:relay_4f", "fixture:two_beam_interference",
        "fixture:two_color_dichroic", "real:beam_expander", "real:focus", "real:folded",
        "real:folded_expander", "real:relay_4f", "real:two_beam_interference",
    }
    assert bool(omitted) == (key in references_with_unstated_loss), (key, omitted)
    assert type(project_).model_validate(document).content_hash() == EXAMPLE_HASHES[key][0]
    assert hash_exact(historical) == EXAMPLE_HASHES[key][1]
    if omitted:
        assert hash_exact(request) != EXAMPLE_HASHES[key][1]  # a changed input hash is necessary to preserve unknown loss
    else:
        assert hash_exact(request) == EXAMPLE_HASHES[key][1]
    assert lowered.engine_input_hash == hash_exact(lowered.request.model_dump(mode="json"))
    assert "optical_models" not in project_.payload() and "design_intent" not in project_.payload()


# ----------------------------------------------------------------------------------------- catalog-free Projects


def _objective_design() -> dict:
    document = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8), declared_component("OBJ", "lens", "objective")],
                       [plane("focus", (80.0, 0.0, 0.0), radius_mm=20.0)])
    from declared_helpers import flint

    return with_models(document, [objective_model()], [nbk7(), flint()])


def test_a_project_of_declared_models_needs_no_catalog(empty_catalog_store):
    with pytest.raises(LocalRunError) as refused:
        load_public_catalog()
    assert refused.value.codes == (LocalRunErrorCode.CATALOG_NOT_AVAILABLE,)
    run = run_optical_project(_objective_design())
    assert run.catalog_version is None and run.simulated_parts == () and run.binding_problems() == ()
    assert run.optical_view.catalog_version is None
    checked = json_api.check_project(_objective_design())
    assert checked["ok"] and checked["catalog_version"] is None and checked["simulated_parts"] == []
    assert checked["declared_models"] == [{"instance_id": "OBJ", "optical_model_id": "objective", "model_kind": "SURFACE_SEQUENCE",
                                          "engine_surfaces": ["S1", "S2", "S3", "S4", "S5"],
                                          "engine_materials": ["project:FLINT", "project:N-BK7"]}]
    assert checked["design_intent"] == "LAB_ASSEMBLY"


def _as_json_numbers(node, key: str | None = None):
    """Every canonical decimal string of a declared model or material written as a JSON number (an integer where it is
    one); ``model_version`` stays the literal it is."""

    if isinstance(node, dict):
        return {k: _as_json_numbers(v, k) for k, v in node.items()}
    if isinstance(node, list):
        return [_as_json_numbers(v, key) for v in node]
    if isinstance(node, str) and key != "model_version" and re.fullmatch(DECIMAL_STRING_PATTERN, node):
        return int(node) if "." not in node else float(node)
    return node


def test_a_declared_model_written_with_json_numbers_is_the_same_project_and_the_same_run():
    """The objective and its glasses with every decimal written as a JSON number: check_project returns the canonical
    document and hash, and the run is the same to the last bit."""

    canonical = _objective_design()
    written = {**canonical, "optical_models": _as_json_numbers(canonical["optical_models"]),
               "model_materials": _as_json_numbers(canonical["model_materials"])}
    assert written != canonical and json.dumps(written).count('"') < json.dumps(canonical).count('"')
    checked, expected = json_api.check_project(written), json_api.check_project(canonical)
    assert checked["ok"] and checked["project"] == expected["project"] == ProjectDocumentV2.model_validate(canonical).payload()
    assert (checked["project_hash"], checked["engine_input_hash"]) == (expected["project_hash"], expected["engine_input_hash"])
    assert _result_text(run_optical_project(written)) == _result_text(run_optical_project(canonical))


def test_design_intent_is_recorded_and_bound_by_the_project_hash():
    document = _objective_design()
    plain = run_optical_project(document)
    system = run_optical_project({**document, "design_intent": "OPTICAL_SYSTEM"})
    assert plain.summary()["design_intent"] == "LAB_ASSEMBLY" and system.summary()["design_intent"] == "OPTICAL_SYSTEM"
    assert system.local_validation.project.project_hash != plain.local_validation.project.project_hash
    assert system.engine_input_hash == plain.engine_input_hash  # physics does not read the intent


def test_a_declared_material_never_replaces_a_catalog_material_of_the_same_name():
    """The Project declares its own N-BK7 (a constant index here) next to the catalog's lens of N-BK7: two Engine
    materials, ``catalog:N-BK7`` and ``project:N-BK7``, each used by its own element."""

    own = {**nbk7(), "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "1.5"}}
    window = {"model_id": "slab", "model": sequence([surface(1, "0"), surface(2, "5")], ["N-BK7"])}
    document = project([source(), element("L1", "lens", "OS-OPT-TEST-LENS-PCX-F100"), declared_component("W", "window", "slab", (50.0, 0.0, 0.0))],
                       [plane("P", (150.0, 0.0, 0.0))])
    lowered = lower_project(with_models(document, [window], [own]), catalog_document())
    # (and the environment's air, ``air``: here the test Projects' vacuum, the constant index 1)
    assert [m.material_id for m in lowered.request.project.materials] == ["air", "catalog:N-BK7", "project:N-BK7"]
    assert lowered.request.project.materials[2].kind == "constant" and lowered.simulated_parts[0].instance_id == "L1"


# ---------------------------------------------------------------------------------------- frames and bodies


def _lower_one(model: dict, role: str = "lens", materials=None) -> object:
    document = project([source(), declared_component("E", role, model["model_id"])], [plane("P", (100.0, 0.0, 0.0))])
    return lower_project(with_models(document, [model], materials if materials is not None else [nbk7()])).elements[0]


def test_frames_and_bodies_are_derived_by_rule():
    lens = _lower_one({"model_id": "lens", "model": sequence([surface(1, "0", radius="50"), surface(2, "4")], ["N-BK7"])})
    assert lens.identity.frame.origin.value == "FIRST_SURFACE_VERTEX" and not lens.identity.is_reflector
    body = lens.identity.design_geometry
    # a cylinder of the edge diameter from the first vertex to the second (the sphere's sag at the edge is inside)
    assert (body.kind, body.diameter_mm, body.z_min_mm, body.z_max_mm, body.status.value) == ("CYLINDER", "20", "0", "4", "NOMINAL")


def test_a_reflector_takes_the_face_normal_frame_and_a_derived_shell_body():
    coating = next(c for c in catalog_document()["coatings"] if c["coating_id"].startswith("DIELECTRIC_HIGH_REFLECTOR"))
    model = {"model_id": "mirror", "model": sequence([surface(1, "0", radius="200", traced="REFLECTED", coating_id=coating["coating_id"])], [])}
    document = project([source(), declared_component("M", "mirror", "mirror")], [plane("P", (-100.0, 0.0, 0.0))])
    (element_,) = lower_project(with_models(document, [model], coatings=[coating])).elements
    frame = element_.identity.frame
    assert (frame.reference_axis.value, frame.nominal_propagation, frame.origin.value) == ("FACE_NORMAL", "REFLECTED_ABOUT_+Z", "REFLECTING_FACE_CENTER")
    body = element_.identity.design_geometry  # the face's sag over the outline: 10^2 / (200 + sqrt(200^2 - 10^2))
    assert body.z_min_mm == "0" and float(body.z_max_mm) == pytest.approx(100.0 / (200.0 + (200.0**2 - 100.0) ** 0.5), abs=1e-9)


def test_a_thin_element_has_no_derived_body():
    stop = _lower_one(stop_model(opening_radius_mm="2", edge_radius_mm="10"), role="aperture")
    assert stop.identity.design_geometry is None and stop.identity.frame.symmetry.value == "ROTATIONAL_ABOUT_LOCAL_Z"
    slit = _lower_one(stop_model(opening_half_widths_mm=["0.05", "3"], edge_radius_mm="10"), role="aperture")
    assert slit.identity.frame.symmetry.value == "NONE"


@pytest.mark.parametrize("model, message", [
    ({"model_id": "m", "model": sequence([surface(1, "5", radius="50"), surface(2, "9")], ["N-BK7"])}, "state its frame"),
    ({"model_id": "m", "model": sequence([surface(1, "0", radius="50"), surface(2, "4")], ["N-BK7"]),
      "frame": {"origin": "REFLECTING_FACE_CENTER", "nominal_propagation": "REFLECTED_ABOUT_+Z", "reference_axis": "FACE_NORMAL",
                "symmetry": "ROTATIONAL_ABOUT_LOCAL_Z"}}, "face-normal frame's coated face"),
])
def test_contradicting_frames_and_bodies_are_refused(model, message):
    document = project([source(), declared_component("E", "lens", "m")], [plane("P", (100.0, 0.0, 0.0))])
    with pytest.raises(LocalRunError) as refused:
        lower_project(with_models(document, [model], [nbk7()]))
    (problem,) = refused.value.problems
    assert problem.code is LocalRunErrorCode.OPTICAL_MODEL_INVALID and message in problem.message and problem.location == "component E"


def test_a_stated_symmetry_flag_and_a_small_stated_body_are_replaced_by_what_the_model_states():
    """The model decides: a stated symmetry flag that differs is replaced by the derived one, and a stated body that does not
    hold the surfaces is extended to the smallest body of its kind that does. The check states both; the Engine request (and
    so the traced result) is that of the Project that states the derived facts."""

    lens = sequence([surface(1, "0", radius="50"), surface(2, "4")], ["N-BK7"])
    document = project([source(), declared_component("E", "lens", "m")], [plane("P", (100.0, 0.0, 0.0))])
    stated = with_models(document, [{"model_id": "m", "model": lens, "frame": {"origin": "FIRST_SURFACE_VERTEX", "symmetry": "NONE"},
                                     "design_geometry": cylinder("20", "0", "3")}], [nbk7()])
    plain = with_models(document, [{"model_id": "m", "model": lens}], [nbk7()])
    lowered, derived = lower_project(stated), lower_project(plain)
    (element_,) = lowered.elements
    assert element_.identity.frame == derived.elements[0].identity.frame
    assert element_.identity.frame.symmetry.value == "ROTATIONAL_ABOUT_LOCAL_Z"
    body = element_.identity.design_geometry
    assert (body.kind, body.diameter_mm, body.z_min_mm, body.z_max_mm) == ("CYLINDER", "20", "0", "4")
    assert lowered.engine_input_hash == derived.engine_input_hash
    assert [(n["code"], n["model_id"]) for n in lowered.model_notices] == [("FRAME_SYMMETRY_DERIVED", "m"), ("DESIGN_GEOMETRY_EXTENDED", "m")]
    assert (lowered.model_notices[0]["stated"], lowered.model_notices[0]["derived"]) == ("NONE", "ROTATIONAL_ABOUT_LOCAL_Z")
    warnings = json_api.check_project(stated)["warnings"]
    (symmetry,) = [w for w in warnings if w["code"] == "FRAME_SYMMETRY_DERIVED"]
    assert symmetry["location"] == "optical model m" and symmetry["details"] == {"model_id": "m", "stated": "NONE",
                                                                                 "derived": "ROTATIONAL_ABOUT_LOCAL_Z"}
    assert not derived.model_notices
    # the view draws the body that is used: that of the Project without the stated frame and body
    (drawn,) = [e for e in run_optical_project(stated).optical_view.elements if e.element_id == "E"]
    (plain_drawn,) = [e for e in run_optical_project(plain).optical_view.elements if e.element_id == "E"]
    assert drawn.geometry == plain_drawn.geometry


# --------------------------------------------------------------------------------------------- refusals


def test_a_custom_optic_without_a_model_names_the_declared_model_route():
    document = project([source(), {**declared_component("L1", "lens", "x"), "optical_model_id": None}], [plane("P", (100.0, 0.0, 0.0))])
    document["components"][1].pop("optical_model_id")
    with pytest.raises(LocalRunError) as refused:
        lower_project(document, catalog_document())
    (problem,) = refused.value.problems
    assert problem.code is LocalRunErrorCode.CUSTOM_PART_UNSUPPORTED and "optical_models" in problem.message and "optical_model_id" in problem.message


@pytest.mark.parametrize("model, role, ok", [
    (stop_model(opening_radius_mm="2", edge_radius_mm="10"), "aperture", True),
    (stop_model(opening_radius_mm="2", edge_radius_mm="10"), "lens", False),
    (thin_lens_model(), "lens", True),
    (thin_lens_model(), "mirror", False),
    (thin_lens_model(), "other_optical", True),
    (stop_model(opening_radius_mm="2", edge_radius_mm="10"), "window", False),
    ({"model_id": "m", "model": sequence([surface(1, "0", radius="50"), surface(2, "4")], ["N-BK7"])}, "prism", True),
    ({"model_id": "m", "model": sequence([surface(1, "0", radius="50"), surface(2, "4")], ["N-BK7"])}, "aperture", True),
])
def test_a_declared_model_plays_the_roles_its_kind_can(model, role, ok):
    document = with_models(project([source(), declared_component("E", role, model["model_id"])], [plane("P", (100.0, 0.0, 0.0))]),
                           [model], [nbk7()] if model["model"]["model_kind"] == "SURFACE_SEQUENCE" else None)
    if ok:
        assert lower_project(document).elements[0].optical_model_id == model["model_id"]
        return
    with pytest.raises(LocalRunError) as refused:
        lower_project(document)
    assert refused.value.codes == (LocalRunErrorCode.OPTICAL_MODEL_ROLE_MISMATCH,)


def test_the_projects_materials_list_is_not_simulation_input():
    """A declared model's media are ``model_materials``; the Project's ``materials`` list is passed over and listed, and
    the request is the one of the Project without it."""

    document = {**_objective_design(), "materials": [{"material_id": "N-BK7", "kind": "GLASS"}]}
    lowered = lower_project(document)
    assert lowered.inputs_not_simulated == ({"location": "materials", "field": "materials", "key": "N-BK7"},)
    assert lowered.engine_input_hash == lower_project(_objective_design()).engine_input_hash


@pytest.mark.parametrize("mutate, fragment", [
    (lambda d: d["components"][1].update(optical_model_id="missing"), "unknown optical model"),
    (lambda d: d.update(model_materials=None), "not declared in model_materials"),
    (lambda d: d["components"][0].update(optical_model_id="objective"), "has no optical_model_id"),
])
def test_unresolved_and_misplaced_declarations_are_invalid_projects(mutate, fragment):
    document = _objective_design()
    mutate(document)
    with pytest.raises(LocalRunError) as refused:
        lower_project(document)
    assert refused.value.codes == (LocalRunErrorCode.PROJECT_INVALID,) and fragment in str(refused.value)


# ---------------------------------------------------------------------------------------------- json api / cli


def test_capabilities_name_the_project_features():
    features = json_api.capabilities()["project_features"]
    assert features == ["design_intent", "compute_budget", "numerical_settings", "optical_models", "compound_units", "aperture_stops", "ideal_thin_lens", "tabulated_thin_lens",
                        "faraday_rotators", "diffraction_gratings", "axicon_surfaces", "acylinder_surfaces", "freeform_surfaces", "neutral_density_attenuators", "neutral_density_coated_annulus", "attenuator_positions",
                        "power_ratings", "pulsed_sources", "geometric_pair_metrics", "polarization_resolved_coatings", "angle_resolved_coatings",
                        "coating_phases", "polarization_transmittance_bands", "elliptical_sources", "source_total_pulses", "extended_emitters",
                        "rating_value_definition", "rating_test_protocol", "partially_polarized_sources", "retained_dof_limits", "aperture_shapes", "surface_holes", "lens_arrays",
                        "configurations", "uniaxial_crystals", "continuous_spectra", "facet_solids", "sampled_intensity_sources",
                        "sampled_field_sources", "speckle_sources", "nonlinear_crystals", "sample_emission", "detection",
                        "compound_joints", "gradient_index_media", "axial_gradient_media"]
    assert "exposures" in json_api.capabilities()["results"]["queries"]


def test_catalog_model_returns_an_insertable_declared_copy():
    reply = json_api.catalog_model("OS-OPT-TEST-LENS-PCX-F100", model_id="pcx")
    assert reply["ok"] and reply["optical_model"]["model_id"] == "pcx" and [m["material_id"] for m in reply["model_materials"]] == ["N-BK7"]
    document = project([source(), declared_component("L1", "lens", "pcx")], [plane("P", (100.0, 0.0, 0.0))])
    document = with_models(document, [reply["optical_model"]], reply["model_materials"], reply["model_coatings"])
    catalog_design = project([source(), element("L1", "lens", "OS-OPT-TEST-LENS-PCX-F100")], [plane("P", (100.0, 0.0, 0.0))])
    assert _result_text(run_optical_project(document)) == _result_text(run_optical_project(catalog_design))
    assert json_api.catalog_model("OS-OPT-NOT-A-PRODUCT")["error"]["problems"][0]["code"] == "OPTICAL_SKU_NOT_PUBLISHED"
    assert json_api.catalog_model("OS-OPT-TEST-LENS-PCX-F100", model_id="has space")["error"]["problems"][0]["code"] == "PROJECT_INVALID"


def test_the_command_line_prints_the_declared_copy(capsys):
    from os_optics.cli import main

    assert main(["project", "model-from-catalog", "OS-OPT-TEST-MIR-FLAT", "--model-id", "flat"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["optical_model"]["model_id"] == "flat" and printed["optical_model"]["frame"]["reference_axis"] == "FACE_NORMAL"
    assert printed["model_coatings"] and printed["model_materials"] is None  # null like the Project field when none
    assert main(["project", "model-from-catalog", "OS-OPT-NOT-A-PRODUCT"]) == 65


# ---------------------------------------------------------------------- refusals and frames


def _lens(model_id: str = "lens", *, r1: str = "52", clear: str = "9", edge: str = "10") -> dict:
    return {"model_id": model_id, "model": sequence([surface(1, "0", radius=r1, clear=clear), surface(2, "4.5", clear=clear)], ["N-BK7"], edge=edge)}


def _document(models: list[dict], components: list[dict], materials: list[dict] | None = None) -> dict:
    return with_models(project([source(), *components], [plane("P", (150.0, 0.0, 0.0))]), models, [nbk7()] if materials is None else materials)


def _refusal(document: dict) -> dict:
    answer = json_api.check_project(document)
    assert answer["ok"] is False
    (problem,) = answer["error"]["problems"]
    return problem


def test_a_refused_declared_model_names_its_item_and_surface_without_derived_clauses():
    """m10: pydantic's derived "Tuple should have at least 1 item" clause is dropped when the item's own error is there;
    the location names the model (and the surface); the contract names the surface whose aperture leaves the outline."""

    outline = _refusal(_document([_lens(edge="8")], [declared_component("L", "lens", "lens")]))
    assert outline["location"] == "optical model lens" and "Tuple should have" not in outline["message"]
    assert "surface S1: its clear aperture (reaching 9 mm from the axis) exceeds the edge radius 8 mm" in outline["message"]
    assert "Value error" not in outline["message"]
    zero = _refusal(_document([_lens(r1="0")], [declared_component("L", "lens", "lens")]))
    assert zero["location"] == "optical model lens surface S1" and "radius_mm must be non-zero" in zero["message"]
    # an exponent is read as the number it spells: 5.2e1 is 52, and 1e-30 is refused for what it is, not for its spelling
    lens = [declared_component("L", "lens", "lens")]
    spelled = json_api.check_project(_document([_lens(r1="5.2e1")], lens))
    assert spelled["ok"] and spelled["project_hash"] == json_api.check_project(_document([_lens()], lens))["project_hash"]
    exponent = _refusal(_document([_lens(r1="1e-30")], lens))
    assert "|radius_mm| must be at least clear_radius_mm" in exponent["message"] and exponent["location"] == "optical model lens surface S1"
    index = _refusal(_document([_lens()], [declared_component("L", "lens", "lens")],
                               [{**nbk7(), "dispersion": {"kind": "CONSTANT_INDEX", "refractive_index": "0.5"}}]))
    assert index["location"] == "material N-BK7" and "refractive index 0.5 is below 1" in index["message"]


def test_a_declared_model_no_component_uses_is_still_checked():
    """n5: check_project compiles every declared model; an unusable one is refused even when nothing places it."""

    unusable = {"model_id": "bad", "model": sequence([surface(1, "5", radius="52"), surface(2, "9")], ["N-BK7"])}
    problem = _refusal(_document([_lens(), unusable], [declared_component("L", "lens", "lens")]))
    assert (problem["code"], problem["location"]) == ("OPTICAL_MODEL_INVALID", "optical model bad") and "state its frame" in problem["message"]
    member = compound_model("unit", [("L1", "lens", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))], cylinder("20", "-1", "6"))
    assert json_api.check_project(_document([_lens(), member], [declared_component("U", "lens", "unit")]))["ok"] is True
    assert json_api.check_project(_document([_lens(), member], [declared_component("L", "lens", "lens")]))["ok"] is True  # unplaced, valid


@pytest.mark.parametrize("model", [stop_model(opening_radius_mm="2", edge_radius_mm="5", plane_z_mm="5"), thin_lens_model()])
def test_a_stop_or_thin_lens_frame_is_its_plane(model):
    """n5: the element lies in its frame's z = 0 plane: a plane at z != 0 is refused by the contract (place the component
    there instead) rather than derived as a frame whose origin is not on the element."""

    role = "aperture" if model["model"]["model_kind"] == "APERTURE_STOP" else "lens"
    if model["model"]["model_kind"] == "IDEAL_THIN_LENS":
        model = {**model, "model": {**model["model"], "plane_z_mm": "2"}}
    off_plane = _refusal(_document([model], [declared_component("E", role, model["model_id"])], []))
    assert "plane_z_mm is 0" in off_plane["message"], off_plane
    on_plane = {**model, "model": {**model["model"], "plane_z_mm": "0"}}
    assert json_api.check_project(_document([on_plane], [declared_component("E", role, model["model_id"])], []))["ok"] is True


def test_a_declared_model_and_material_named_in_any_script_check_run_and_are_written(tmp_path):
    """Names an author writes in Chinese, with a degree, micro or accented letter, go through check_project, a run and its
    run directory; NFC and NFD spellings are one Project; the native library takes such a display name."""

    import unicodedata

    name = "物镜 f=50 mm, 30° ± 5 µm, Lentille à focale"
    document = _objective_design()
    document["optical_models"][0]["display_name"] = unicodedata.normalize("NFD", name)
    document["model_materials"][0]["display_name"] = "玻璃 N-BK7"
    checked = json_api.check_project(document)
    assert checked["ok"] and checked["project"]["optical_models"][0]["display_name"] == unicodedata.normalize("NFC", name)
    nfc = json.loads(json.dumps(document))
    nfc["optical_models"][0]["display_name"] = unicodedata.normalize("NFC", name)
    assert json_api.check_project(nfc)["project_hash"] == checked["project_hash"]
    plain = run_optical_project(_objective_design())
    run = run_optical_project(document, output_dir=tmp_path / "run")
    assert [o.total_power_W for o in run.result.observations] == [o.total_power_W for o in plain.result.observations]
    assert name in (tmp_path / "run" / "project.json").read_text(encoding="utf-8")
    built = json_api.build_native_model("lens.tabulated_thin", {**CLASSES_B["lens.tabulated_thin"], "model_id": "ml", "display_name": "超透镜 Ø10 mm"})
    assert built["ok"], built
    refused = json_api.build_native_model("lens.tabulated_thin", {**CLASSES_B["lens.tabulated_thin"], "model_id": "ml", "display_name": "bad\u0007name"})
    assert refused["ok"] is False and "control character" in json.dumps(refused["error"])


TYPICAL_BANDPASS = {"coating_id": "BP_TYPICAL", "kind": "BANDPASS", "display_name": "Typical bandpass curve",
                    "spectrum": {"wavelength_nm": ["400", "500", "550", "600", "700"], "transmittance": ["0.001", "0.4", "0.855", "0.3", "0.6"],
                                 "reflectance": ["0.95", "0.55", "0.1", "0.65", "0.35"], "status": "TYPICAL"},
                    "valid_aoi_deg": {"min_deg": "0", "max_deg": "5", "status": "SPECIFIED"}, "basis": "synthetic typical curve"}


def test_a_declared_model_that_leaves_out_a_label_that_follows_from_it_runs_and_reports_the_label():
    """A typical-curve coating makes the model a ``TYPICAL_SPECTRAL_CURVE`` model whether or not it lists the label: the
    Project with and without the label runs to the same light, and the summary reports the label for both."""

    def design(listed: bool) -> dict:
        document = _objective_design()
        model = document["optical_models"][0]["model"]
        model["surfaces"][0]["coating_id"] = "BP_TYPICAL"
        model["valid_band_nm"] = {"min_nm": "400", "max_nm": "700", "status": "SPECIFIED"}
        model["approximations"] = sorted({*model["approximations"], "TYPICAL_SPECTRAL_CURVE"} if listed else
                                         set(model["approximations"]) - {"TYPICAL_SPECTRAL_CURVE"})
        document["model_coatings"] = [TYPICAL_BANDPASS]
        return document

    listed, omitted = run_optical_project(design(True)), run_optical_project(design(False))
    assert [o.total_power_W for o in omitted.result.observations] == [o.total_power_W for o in listed.result.observations]
    rows = [next(r for r in run.summary()["element_models"] if r["instance_id"] == "OBJ") for run in (listed, omitted)]
    assert rows[0]["approximations"] == rows[1]["approximations"] and "TYPICAL_SPECTRAL_CURVE" in rows[1]["approximations"]
    assert rows[0]["approximations"] == sorted(rows[0]["approximations"])
