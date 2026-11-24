"""Data boundary: the Optical Simulation View holds only design-stage optical content.

* The artifacts (scene JSON, GLB JSON chunk, manifest) contain no mechanical or supplier vocabulary, and identities a
  Project or an Engine request carries outside the published catalog never reach them. (The check of the artifacts
  against the identities of the catalog's private realizations needs that private data and runs where it is kept.)
* The only element bodies are public catalog design geometry or neutral view placeholders; no node is anything but an
  optical element, an observation plane, a beam path or the optical-axis markers.
* A Project's free text never reaches the view, and an Engine request that carries a component the Project does not
  show (e.g. a mechanical blocker) is refused rather than drawn around.
* The view package never imports mechanical / commercial / supplier code.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
from optical_view_reference import build_scene, folded_project, polarization_project
from optical_view_reference import engine_request, engine_result, engine_run, project, ray_paths, scene  # noqa: F401  (pytest fixtures)

from catalog_store_helpers import REAL_CATALOG
from pydantic import ValidationError

from os_optics import validate
from os_optics._vendor.shared_contracts.optical_view import SceneElementV1
from os_optics.contracts import AbsorberInteraction, CircularAperture, EngineeringComponentSnapshot, PlaneSurface, Pose3D
from os_optics._vendor.shared_contracts.optical_design.design_content import CustomPartRefV1
from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.optical_view import OpticalViewBindingError, build_optical_simulation_scene, optical_view_files, parse_glb

PKG_ROOT = Path(__file__).resolve().parents[1]
VIEW_SRC = PKG_ROOT / "src" / "os_optics" / "optical_view"
# Synthetic stand-ins for identities that are not part of the published catalog (a vendor, its part numbers): a Project's
# free text or an Engine-only component naming them must not carry them into the view.
FOREIGN_TOKENS = ("ACME-OPTICS", "XQ-412B", "ZK-75")
MECHANICAL_WORDS = ("mount", "mounts", "post", "posts", "holder", "holders", "base", "bases", "breadboard", "breadboards", "stage",
                    "stages", "fastener", "fasteners", "screw", "screws", "bracket", "supplier", "suppliers", "supplierpart", "sku_supplier")


@pytest.fixture(scope="module")
def catalog():
    """The pinned public catalog release."""
    from os_optics.local import load_public_catalog

    return load_public_catalog(REAL_CATALOG)


def _pattern(token: str) -> re.Pattern[str]:
    """Separator-insensitive and alphanumerically bounded: 'XQ-412B' also matches 'xq412b' / 'XQ 412 B', never inside a
    longer alphanumeric run (e.g. a hex digest)."""
    chars = [re.escape(c) for c in token.lower() if c.isalnum()]
    return re.compile(r"(?<![a-z0-9])" + r"[\W_]*".join(chars) + r"(?![a-z0-9])")


def _public_texts(scene) -> dict[str, str]:
    files = optical_view_files(scene)
    gltf, _ = parse_glb(files["optical_scene.glb"])
    return {
        "optical_scene.json": files["optical_scene.json"].decode("utf-8"),
        "optical_scene.glb (JSON chunk)": json.dumps(gltf, sort_keys=True),
    }


@pytest.fixture(scope="module")
def multi_element_scenes(catalog):
    return {"folded": build_scene(folded_project(catalog), catalog)[0], "polarization": build_scene(polarization_project(catalog), catalog)[0]}


@pytest.mark.parametrize("word", MECHANICAL_WORDS)
def test_no_mechanical_or_supplier_vocabulary_in_the_multi_element_artifacts(multi_element_scenes, word):
    """The two-branch design (lenses, mirror, window, splitter, filter) and the polarization design: no mechanical or
    supplier vocabulary in any artifact."""
    for design, design_scene in multi_element_scenes.items():
        for name, text in _public_texts(design_scene).items():
            text = text.lower().replace("design-stage", "")
            assert not re.search(rf"(?<![a-z0-9]){word}(?![a-z0-9])", text), f"{design} {name} contains {word!r}"


@pytest.mark.parametrize("word", MECHANICAL_WORDS)
def test_no_mechanical_or_supplier_vocabulary_in_any_artifact(scene, word):
    for name, text in _public_texts(scene).items():
        text = text.lower().replace("design-stage", "")  # the view notice's "design-stage optical system" is not a stage
        assert not re.search(rf"(?<![a-z0-9]){word}(?![a-z0-9])", text), f"{name} contains {word!r}"


def test_only_optical_nodes_and_neutral_bodies(scene):
    gltf, _ = parse_glb(optical_view_files(scene)["optical_scene.glb"])
    root = gltf["nodes"][gltf["scenes"][0]["nodes"][0]]
    kinds = {gltf["nodes"][i]["extras"]["kind"] for i in root["children"]}
    assert kinds <= {"OPTICAL_ELEMENT", "TARGET_PLANE", "OBSERVATION_PLANE", "BEAM_PATH", "OPTICAL_AXIS_MARKERS"}
    # root + camera + a flat list, plus one child node per optical surface of an element: no hidden sub-assemblies
    surfaces = [c for i in root["children"] for c in gltf["nodes"][i].get("children", ())]
    assert all(gltf["nodes"][c]["extras"]["kind"] == "OPTICAL_SURFACE" and not gltf["nodes"][c].get("children") for c in surfaces)
    assert all(gltf["nodes"][i]["extras"]["kind"] == "OPTICAL_ELEMENT" for i in root["children"] if gltf["nodes"][i].get("children"))
    assert len(gltf["nodes"]) == 2 + len(root["children"]) + len(surfaces)
    for element in scene.elements:
        assert element.geometry.basis.value in {"PUBLIC_CATALOG_DESIGN_GEOMETRY", "VIEW_PLACEHOLDER"}
        assert element.geometry.status.value in {"NOMINAL", "PLACEHOLDER"}


def test_project_free_text_never_reaches_the_view(project, catalog, engine_request, engine_result):
    marker = "ACME-OPTICS XQ-412B kinematic mount on ZK-75 post"
    components = tuple(
        c.model_copy(update={"custom_part": CustomPartRefV1(custom_part_key="user-collimated-laser", description=marker)})
        if c.custom_part is not None else c
        for c in project.components
    )
    noisy = ProjectDocumentV2.model_validate(project.model_copy(update={"components": components}).payload())
    scene = build_optical_simulation_scene(project=noisy, catalog=catalog, engine_request=engine_request, engine_result=engine_result)
    for text in _public_texts(scene).values():
        low = text.lower()
        assert "kinematic" not in low and not [token for token in FOREIGN_TOKENS if _pattern(token).search(low)]


def test_an_engine_component_the_project_does_not_show_is_refused(project, catalog, engine_request):
    blocker = EngineeringComponentSnapshot(
        instance_id="XQ-412B", role="dump", pose=Pose3D(position_mm=(3.0, 0.0, 0.0)),
        surfaces=(PlaneSurface(surface_id="ring", origin_mm=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                               aperture=CircularAperture(radius_mm=15.0, obstruction_radius_mm=12.7), interaction=AbsorberInteraction()),),
    )
    request = engine_request.model_copy(update={"project": engine_request.project.model_copy(
        update={"components": (*engine_request.project.components, blocker)})})
    with pytest.raises(OpticalViewBindingError, match="Engine only: \\['XQ-412B'\\]"):
        build_optical_simulation_scene(project=project, catalog=catalog, engine_request=request, engine_result=validate(request))


def test_the_contract_cannot_carry_mechanical_fields(scene):
    element = scene.elements[1].model_dump(mode="json")
    for field in ("mount_sku", "supplier_sku", "supplier", "post", "breadboard"):
        with pytest.raises(ValidationError):
            SceneElementV1.model_validate({**element, field: "x"})
    with pytest.raises(ValidationError):
        SceneElementV1.model_validate({**element, "optical_role": "mount"})
    with pytest.raises(ValidationError):
        SceneElementV1.model_validate({**element, "optical_sku": "ZK-75A"})


FORBIDDEN_IMPORT_PREFIXES = (
    "opensci_data", "opensci_manufacturing_compiler", "opensci_product_data", "sqlalchemy", "psycopg", "fastapi", "requests", "httpx",
    "shared_contracts.assembly", "shared_contracts.commerce", "shared_contracts.fulfillment", "shared_contracts.agent_catalog",
    "shared_contracts.manufacturing_ir", "shared_contracts.manufacturing_route", "shared_contracts.compiler_manifest",
    "numpy", "trimesh", "matplotlib", "shared_contracts", "opensci_optical_view",
)
# Relative imports the view may use: the vendored public contracts, Engine contracts / hashing / ray paths, the Engine's
# public-model compilation (the one source of truth the view binds catalog elements to), itself.
ALLOWED_RELATIVE_PREFIXES = (
    "_vendor.shared_contracts.common", "_vendor.shared_contracts.design_catalog",
    "_vendor.shared_contracts.optical_design", "_vendor.shared_contracts.optical_view", "_version", "contracts", "manifest",
    "validation.ray_paths", "local.compile", "optics.materials.index",
    # the design-point tolerances the run compiled its elements under (the view compiles them the same way)
    "local.design_point",
    # the aperture regions (an outline's shape and holes) and their lowering from the contract: geometry of the optic itself
    "optics.apertures", "scene.model", "scene.lower",
    # the run's monitor (progress and cancellation checkpoints while the view is drawn and written)
    "run_control",
)


@pytest.mark.parametrize("path", sorted(VIEW_SRC.rglob("*.py")), ids=lambda p: p.name)
def test_view_never_imports_mechanical_commercial_or_supplier_code(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else [])
        for name in names:
            assert not name.startswith(FORBIDDEN_IMPORT_PREFIXES), f"{path.name} imports {name}"
        if isinstance(node, ast.ImportFrom) and node.level == 2:  # from ..x import y: outside the view package
            assert (node.module or "").startswith(ALLOWED_RELATIVE_PREFIXES), f"{path.name} imports ..{node.module}"


def test_the_view_uses_only_the_supported_ray_path_api():
    """No production view code reaches into Engine tracer internals (the earlier subclass seam is gone)."""
    for path in sorted(VIEW_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for internal in ("optics.rays", "Tracer", "_RecordingTracer", "lower_scene", "_root_packet", "_spawn", "_interact", "_finish"):
            assert internal not in text, f"{path.name} mentions Engine tracer internal {internal!r}"
