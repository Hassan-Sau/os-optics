"""Wheel build, clean-environment install (native) and the REAL Pyodide integration + native/Pyodide parity tests.

These tests never skip silently: a missing prerequisite is a failure with an explicit message, unless the operator sets
OPENSCI_SKIP_PYODIDE=1 (must not be set in CI). Marks: ``install`` and ``pyodide``."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from os_optics import SCHEMA_VERSION, __version__, validate_json
from os_optics._version import MIN_PYTHON, NUMERICAL_CONFIG_VERSION
from os_optics.compare import compare_results
from os_optics.manifest import engine_build_id

from catalog_store_helpers import REAL_CATALOG
from generate_fixtures import REQ_DIR

ROOT = Path(__file__).resolve().parents[1]
PYODIDE_DIR = Path(__file__).resolve().parent / "pyodide"
PYODIDE_PIN = "314.0."  # Pyodide 314.0.x stable series
NAMES = sorted(p.stem for p in REQ_DIR.glob("*.json"))
ASSEMBLY_PACKAGE = ROOT / "tests/fixtures/assembly_preview/reference.assembly-preview.zip"
DRAWING_PACKAGE = ROOT / "tests/fixtures/assembly_drawing/reference.assembly-preview.zip"
DRAWING_DETAIL = ROOT / "tests/fixtures/assembly_drawing/reference.assembly-detail.json"
#: v0.13: four support chains on one breadboard (an A2 sheet, merged / ordinate dimensions, stacked balloons, text columns).
MULTI_DRAWING_PACKAGE = ROOT / "tests/fixtures/assembly_drawing/four_lenses.assembly-preview.zip"
MULTI_DRAWING_DETAIL = ROOT / "tests/fixtures/assembly_drawing/four_lenses.assembly-detail.json"
#: a customer-supplied optical component drawn at its design position (table, placement step).
CUSTOMER_DRAWING_PACKAGE = ROOT / "tests/fixtures/assembly_drawing/customer_supplied.assembly-preview.zip"
CUSTOMER_DRAWING_DETAIL = ROOT / "tests/fixtures/assembly_drawing/customer_supplied.assembly-detail.json"


@pytest.fixture(scope="session")
def wheel(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("dist")
    proc = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(out), str(ROOT)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    wheels = list(out.glob("os_optics-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


# ---- wheel content ---------------------------------------------------------------------------------------------------------------
def test_wheel_is_pure_python_and_declares_the_dependency_policy(wheel):
    assert wheel.name == f"os_optics-{__version__}-py3-none-any.whl"  # py3-none-any: no compiled code, installs anywhere
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        assert not any(n.endswith((".so", ".pyd", ".dll", ".dylib", ".wasm")) for n in names)
        assert "os_optics/py.typed" in names and "os_optics/cli.py" in names
        assert not any(n.startswith(("tests/", "docs/")) for n in names)
        meta = z.read(next(n for n in names if n.endswith("METADATA"))).decode()
        wheel_meta = z.read(next(n for n in names if n.endswith("/WHEEL"))).decode()
        entry = z.read(next(n for n in names if n.endswith("entry_points.txt"))).decode()
    assert "Root-Is-Purelib: true" in wheel_meta and "Tag: py3-none-any" in wheel_meta
    assert "Requires-Python: >={}.{}".format(*MIN_PYTHON) in meta
    requires = sorted(line.split(":", 1)[1].strip() for line in meta.splitlines() if line.startswith("Requires-Dist") and "extra ==" not in line)
    assert [r.split("<")[0].split(">")[0].split("=")[0] for r in requires] == ["numpy", "pydantic", "scipy", "shapely"]
    assert all("<" in r and ">=" in r for r in requires)  # lower bound = tested minimum, upper bound = next major
    assert "os-optics = os_optics.cli:main" in entry


def test_wheel_metadata_names_headless_ordering_and_only_compatible_kit_viewing(tmp_path):
    """The shipped PyPI description must not send customers to the retired hosted editor."""
    builder = shutil.which("uv")
    assert builder is not None, "uv is required to build the candidate wheel"
    built = subprocess.run([builder, "build", "--wheel", "--out-dir", str(tmp_path), str(ROOT)],
                           capture_output=True, text=True, timeout=600)
    assert built.returncode == 0, built.stdout + built.stderr
    wheels = list(tmp_path.glob(f"os_optics-{__version__}-py3-none-any.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        metadata = archive.read(next(name for name in archive.namelist()
                                     if name.endswith(".dist-info/METADATA"))).decode("utf-8")
    prose = " ".join(metadata.split())
    for fact in (
        "Your Agent edits a Project through the CLI or Python API",
        "os-optics project update",
        "os-optics project evaluate",
        "os-optics order create RESOLUTION_ID",
        "This wheel includes `os-optics view RUN`",
        "If a published `os-optics-kit` wheel includes a local Project/run-follow viewer and declares compatibility",
        "use the kit's documented command and install it with normal dependency resolution",
    ):
        assert fact in prose, fact
    for retired in ("Designer", "Mode B", "updateProject", "window.OpenSciAgent", "os-optics-kit view"):
        assert retired not in prose, retired


@pytest.mark.install
def test_clean_environment_install_from_the_wheel_needs_no_compiler(wheel, tmp_path):
    """`pip install` into a fresh venv with --only-binary=:all: (no compiler / CMake / system library can be involved), then run
    the 405 nm / 500 nm golden case through the CLI and the Python API."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=300)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    proc = subprocess.run([str(py), "-m", "pip", "install", "--only-binary=:all:", "--no-cache-dir", str(wheel)], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    cli = subprocess.run([str(py), "-m", "os_optics", "validate", str(REQ_DIR / "interference_405_500.json")], capture_output=True, text=True, timeout=300)
    assert cli.returncode == 0, cli.stderr
    doc = json.loads(cli.stdout)
    assert doc["status"] == "PASS" and doc["interference"][0]["fringe_period_nm"] == pytest.approx(500.0, abs=1e-6)
    api = subprocess.run([str(py), "-c",
                          "import os_optics as e; from os_optics import validate_json; "
                          "print(e.__version__, e.SCHEMA_VERSION, e.NUMERICAL_CONFIG_VERSION)"], capture_output=True, text=True, timeout=300)
    assert api.stdout.split() == [__version__, SCHEMA_VERSION, NUMERICAL_CONFIG_VERSION]
    # the installed copy is byte-identical to the source tree: same build id, and the same result as the in-process engine
    ident = subprocess.run([str(py), "-c", "from os_optics.manifest import engine_build_id; print(engine_build_id())"], capture_output=True, text=True, timeout=300)
    assert ident.stdout.strip() == engine_build_id()
    assert compare_results(cli.stdout, validate_json((REQ_DIR / "interference_405_500.json").read_text(encoding="utf-8"))) == []
    # an installed package must not need dev-only files
    listing = subprocess.run([str(py), "-m", "pip", "list", "--format=json"], capture_output=True, text=True, timeout=300)
    installed = {p["name"].lower() for p in json.loads(listing.stdout)}
    assert {"numpy", "scipy", "pydantic", "shapely", "os-optics"} <= installed
    assert not installed & {"torch", "tensorflow", "sqlalchemy", "requests", "httpx", "fastapi", "trimesh", "networkx"}


# ---- real Pyodide -----------------------------------------------------------------------------------------------------------------
SQRT_HALF = 0.7071067811865476
BEAM_ALONG_Y = [0.0, 0.0, SQRT_HALF, SQRT_HALF]  # (Rz 90 deg) * example roll: local +Z -> design +Y, local +Y -> design +Z


def json_api_cases() -> dict[str, dict]:
    """Projects the browser Designer runs through os_optics.local.json_api: bounded UNKNOWN on public glass, FAIL,
    OUT_OF_MODEL_SCOPE (a wavelength just beyond the lens model's band), a refusal (a lens placed as a mirror), the reference design rotated by 90 deg about
    Z (beam along +Y) with edited DOF intent and target bounds, and every other reference design kind (a mirror fold, a
    two-lens beam expander, a 4f relay of one product used twice, a folded expander, dichroic wavelength routing through
    tabulated coatings, a waveplate + polarizer chain, a one-laser two-arm interferometer with a declared linewidth), and
    that interferometer without its linewidth (UNKNOWN: its visibility / coherence targets need it), Projects of
    declared optical models (``_declared_cases``) and one with an elliptical source, an angle-resolved coating with
    tabulated phases and banded polarization elements."""
    from decimal import Decimal

    from os_optics.local import EXAMPLE_KINDS, example_project, example_project_of_kind, load_public_catalog

    reference = example_project(REAL_CATALOG).payload()
    failing = json.loads(json.dumps(reference))
    failing["components"][1]["nominal_pose"]["position_mm"] = [5.0, 0.0, 0.0]
    lens = load_public_catalog(REAL_CATALOG).get(reference["components"][1]["product"]["optical_sku"])
    band_oos = json.loads(json.dumps(reference))
    band_oos["components"][0]["source_model"]["wavelength_nm"] = float(Decimal(lens.simulation_models[0].valid_band_nm.max_nm) + 1)
    band_oos["targets"] = [{"target_id": "power", "metric": "power_W", "observation_id": "focal_plane", "minimum": 0.0, "unit": "W"}]
    refused = json.loads(json.dumps(reference))
    refused["components"][1]["optical_role"] = "mirror"
    rotated = json.loads(json.dumps(reference))
    for pose in [c["nominal_pose"] for c in rotated["components"]] + [o["pose"] for o in rotated["observations"]]:
        x, y, z = pose["position_mm"]
        pose["position_mm"] = [-y, x, z]
        assert pose["quaternion_wxyz"] == [0.5, 0.5, 0.5, 0.5]
        pose["quaternion_wxyz"] = BEAM_ALONG_Y
    rotated["components"][1]["retained_dofs"] = [{"axis": "Tz", "control": "NUMERICAL"}, {"axis": "Rx", "control": "MANUAL"}]
    rotated["targets"][1]["maximum"] = 0.04  # the focus spot is ~25 um
    # the interferometer without its declared linewidth -- the coherence-length target is UNKNOWN in both runtimes
    undeclared = example_project_of_kind("two_beam_interference", REAL_CATALOG).payload()
    undeclared["components"][0]["source_model"].pop("linewidth_nm")
    # sampled rays of an extended emitter through a lens, with a beam metric that has no value for them
    # a lens whose retained DOFs state limits, assessed by dof_* targets, beside an out-of-scope metric target
    limits = json.loads(json.dumps(reference))
    limits["components"][1]["retained_dofs"] = [
        {"axis": "Tz", "control": "NUMERICAL", "travel": {"min": -1.5, "max": 2.5}, "hard_limits": {"min": -2.0, "max": 3.0},
         "repeatability": 0.003, "drift": {"per_kelvin": 0.0011, "per_hour": 0.0007}, "unit": "mm"},
        {"axis": "Rx", "control": "MANUAL"}]
    lens_id = limits["components"][1]["instance_id"]
    limits["targets"] += [
        {"target_id": "dof_travel", "metric": "dof_travel_margin_mm", "device_instance_id": lens_id, "minimum": 0.0,
         "parameters": [{"key": "axis", "value": "Tz"}, {"key": "required_max", "value": 0.7}, {"key": "required_min", "value": -0.3}]},
        {"target_id": "dof_drift", "metric": "dof_drift_mm", "device_instance_id": lens_id, "maximum": 0.01,
         "parameters": [{"key": "axis", "value": "Tz"}, {"key": "duration_h", "value": 3.3}, {"key": "temperature_span_K", "value": 7.1}]},
        {"target_id": "dof_tilt", "metric": "dof_travel_margin_deg", "device_instance_id": lens_id, "minimum": 0.0,
         "parameters": [{"key": "axis", "value": "Rx"}, {"key": "required_max", "value": 0.2}]},
        {"target_id": "lock_in", "metric": "spectral_resolution_nm", "observation_id": "focal_plane", "maximum": 1.0}]
    emitter = json.loads(json.dumps(reference))
    emitter["components"][0]["source_model"] = {
        "model_kind": "EXTENDED_EMITTER", "wavelength_nm": 550.0, "power_w": 0.1, "emitting_surface": {"shape": "RECTANGLE", "width_mm": 2.0, "height_mm": 1.0},
        "angular_law": "LAMBERTIAN", "sampling": {"sample_count": 256, "seed": 3}}
    return {"reference": reference, "extended_emitter": emitter, "retained_dof_limits": limits, "lens_shift_fail": failing, "band_oos": band_oos, "refused_role": refused, "rotated_beam_y": rotated,
            **{kind: example_project_of_kind(kind, REAL_CATALOG).payload() for kind in EXAMPLE_KINDS if kind != "focus"},  # focus = reference
            "interference_coherence_undeclared": undeclared,
            **_beam_scope_cases(reference), **_declared_cases(reference), "native_splitter_cavity": _native_splitter_cavity(),
            "tail_outside_acceptance": _tail_project(10.0), "tail_inside_acceptance": _tail_project(30.0),
            "coatings_phases_polarization": _coating_and_polarization_project(),
            "rated_oblique_window_rotary_nd": _rated_oblique_rotary_case(), "rated_untraced_transmission": _rated_untraced_case(),
            "rated_test_beam_and_pulse_count": _rated_conditions_case(), "rated_left_out_light_only": _rated_left_out_only_case(),
            "rated_window_stack": _rated_window_stack_case(),
            "rated_pulse_counts_and_value_readings": _rated_counts_case(),
            "coherent_elliptical_interference": _coherent_elliptical_case(), **_oblique_cases(), **_line_focus_cases(),
            **_tabulated_scope_cases(), "interference_bounded_coatings": _bounded_coatings_case(), **_faraday_cases(), **_grating_cases(), "facet_solid_corner_cube": _facet_solid_case(),
            "weak_signal_chain": _weak_signal_chain_case(),
            **_michelson_cases(), **_result_semantics_cases(), **_aperture_shape_cases(), "independent_sources": _independent_sources_case(),
            **_power_interval_cases(), "polarization_families": _polarization_families_case(),
            # the 35 / 70 / 105 mm zoom as one Project with configurations, its 105 mm configuration limited to two
            # interactions per path (PARTIAL beside the two complete configurations)
            "zoom_configurations": _zoom_configurations_case(), "ghost_singlet": _ghost_singlet_case(),
            **_scatter_and_sample_cases(), **_sampled_source_cases(),
            "crystal_displacer_continuous_spectrum": _crystal_spectrum_case(), "solid_fold_pbs_cube": _solid_fold_case(),
            "shg_d_tensor_turned": _shg_turned_case(), **_gradient_rod_cases()}


def _gradient_rod_cases() -> dict:
    """An axial gradient-index rod with spherical end faces, and a radial rod whose round body has an end face slanted by
    8 deg, each on a bench turned about an oblique axis: the axial legs' quadrature, the face crossings by bisection and the
    Gaussian transfer on the faces' tangent planes give the same bits in the browser as natively
    (``test_gradient_index_axial_faces``)."""
    from test_gradient_index_axial_faces import axial_material, radial_material, rod_project, sequence_rod, slanted_rod

    axial = rod_project(sequence_rod("AGRIN", "10", "8", "-12", edge="2", clear="1.9"), axial_material(), start=(0.4, -0.2), out_at=14.0)
    slanted = rod_project(slanted_rod("GRIN", 8.0), radial_material(), start=(0.0, 0.3))
    return {"grin_axial_curved_turned": _turned(axial), "grin_slanted_face_turned": _turned(slanted)}


def _zoom_configurations_case() -> dict:
    from test_project_configurations import _budget_limited

    return _budget_limited()


def _ghost_singlet_case() -> dict:
    """A biconvex N-BK7 singlet focusing on its image plane, whose ghost-path analysis (every face's reflections up to two
    bounces, ``test_ghosts``) the harness also runs: ``run_results`` kind ``ghosts``."""
    from test_ghosts import _paraxial_ghost_focus, _prescription
    from test_imaging import _lens_project, _nbk7

    bfl, _ghost = _paraxial_ghost_focus(_nbk7(587.5618), 40.0, -40.0, 5.0)
    return _lens_project(_prescription(40.0, -40.0, 5.0, bfl), waist_mm=0.5)


def _sampled_source_cases() -> dict:
    """A Gaussian intensity map emitting into a 5-degree cone, and a tilted sampled field whose coherent propagation the
    harness also reads (``run_results`` kind ``field``, ``test_sampled_sources``)."""
    from local_helpers import plane, project
    from test_sampled_sources import _component, _gaussian_rows, _intensity, _tilted_field

    intensity = _intensity(_gaussian_rows(16, 0.1, 0.5), 0.1, law="UNIFORM_CONE", cone_half_angle_deg=5.0, rays=256)
    cases = {"sampled_intensity_map": project([_component(intensity)], [plane("P", (10.0, 0.0, 0.0), radius_mm=5.0)]),
             "field_sampled_tilt": project([_component(_tilted_field(0.1, n=32, pitch=0.01, w=0.08))], [plane("P", (5.0, 0.0, 0.0), radius_mm=5.0)])}
    # the same two designs turned about design Z (47 and 30 degrees): the samples' placement in the design frame is no longer an
    # exact product of 0 and 1, so their arithmetic must be runtime-identical (no fused or BLAS kernels)
    cases["sampled_intensity_map_turned"] = _turned_about_z(cases["sampled_intensity_map"], 47.0)
    cases["field_sampled_tilt_turned"] = _turned_about_z(cases["field_sampled_tilt"], 30.0)
    return cases


def _turned_about_z(document: dict, degrees: float) -> dict:
    """``document`` with every component and observation turned about design Z through the origin by ``degrees``."""
    import copy

    from local_helpers import quat_multiply, turn_about_z

    turned = copy.deepcopy(document)
    half = math.radians(degrees) / 2.0
    turn = turn_about_z(math.cos(half), math.sin(half))
    c, s = math.cos(2.0 * half), math.sin(2.0 * half)
    for pose in [c_["nominal_pose"] for c_ in turned["components"]] + [o["pose"] for o in turned["observations"]]:
        x, y, z = pose["position_mm"]
        pose["position_mm"] = [c * x - s * y, s * x + c * y, z]
        pose["quaternion_wxyz"] = list(quat_multiply(turn, pose["quaternion_wxyz"]))
    return turned


def _scatter_and_sample_cases() -> dict:
    """Scattered light and thin samples: a plate of a scattering glass whose front face scatters (traced scatter rays, bulk
    scattering booked and not followed: ``test_surface_and_bulk_scatter``), and an aperture stop holding a sample with a
    linear amplitude map and a Zernike phase (``test_complex_transmission_sample``)."""
    from decimal import Decimal

    from test_complex_transmission_sample import _sample_document
    from test_surface_and_bulk_scatter import _milky_plate_document

    xs = [i - 5 for i in range(11)]
    sample = {"amplitude": "0.95", "phase_terms": [{"kind": "ZERNIKE", "indexing": "NOLL", "normalization": "RMS_ORTHONORMAL",
                                                    "normalization_radius_mm": "8", "zernike_indices": [4, 11], "zernike_values_rad": ["-3", "0.2"]}],
              "sampled_map": {"nodes_x": 11, "nodes_y": 11, "pitch_mm": ["1", "1"], "amplitude": [str((Decimal(50) + x) / 100) for _y in xs for x in xs]}}
    return {"scattering_plate": _milky_plate_document(scatter={"bsdf": "COSINE_POWER", "exponent": "2", "fraction": "0.05", "side": "REFLECT",
                                                               "sample_count": 256}),
            "complex_transmission_sample": _sample_document(sample)}


def _aperture_shape_cases() -> dict:
    """Aperture regions: a triangular stop (the boundary-integral Gaussian power), a window with a hole a beam passes
    through, and a lens array whose cells focus three beams (``test_declared_aperture_shapes``)."""
    from test_declared_aperture_shapes import _array_project, _array_unit, _window
    from declared_helpers import declared_component, nbk7, stop_model, with_models
    from local_helpers import ALONG_X, project, source

    stop = stop_model(opening_radius_mm="1.2", edge_radius_mm="12.7",
                      opening_shape={"kind": "POLYGON", "vertices_mm": [["-0.9", "-0.6"], ["1", "-0.5"], ["0", "1.1"]]})
    after = [{"observation_id": "after", "pose": {"position_mm": [30.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}]
    triangle = with_models(project([source(position=(-50.0, 0.1, 0.0), wavelength_nm=632.8, waist_mm=0.25), declared_component("A", "aperture", "stop")],
                                   after), [stop])
    holed = with_models(project([source(position=(-50.0, 4.0, 0.0), wavelength_nm=632.8, waist_mm=0.3), declared_component("W", "window", "window")],
                                after), [_window([{"center_mm": ["4", "0"], "radius_mm": "2"}])], [nbk7()])
    return {"triangle_stop": triangle, "window_hole": holed, "lens_array": _array_project([-2.5, 0.0, 2.5], _array_unit())}


def _independent_sources_case() -> dict:
    """Four independent lasers of one wavelength on one plane, a fringe-visibility target on two of them: the plane's
    six pairs are counted (``incoherent_pairs``), the target's pair alone is analysed (``test_scale_outputs``)."""
    from local_helpers import plane, project, source

    lasers = [source(f"L{i}", position=(-200.0, 3.0 * i, 0.0), waist_mm=0.5) for i in range(4)]
    return project(lasers, [plane("P", (0.0, 4.5, 0.0), radius_mm=20.0)],
                   [{"target_id": "vis", "metric": "fringe_visibility", "observation_id": "P", "source_id": "L1",
                     "source_id_b": "L2", "maximum": 0.5}])


def _solid_fold_case() -> dict:
    """The real catalog's 532 nm polarizing cube lit by a 15 um waist (a 5.2 deg cone): its internal folds (faces and
    diagonal) are bounded by their certified closed walks (``validation/solid_folds.py``, ``test_solid_folds``), which the
    plane's power upper bound and the cube's exposure bounds read -- the same branch-and-bound and sums in the browser as
    natively."""
    from catalog_store_helpers import catalog_version_of
    from local_helpers import plane, project, source

    laser = source(position=(-62.7, 0.0, 0.0), wavelength_nm=532.0, waist_mm=0.015, power_w=1.0, waist_position_mm=50.0)
    cube = {"instance_id": "P", "optical_role": "splitter",
            "product": {"optical_sku": "OS-OPT-PBS-CUBE-A25P4-532-ER3000-TP95RS99P5-UVFS-CA20P32",
                        "catalog_version": catalog_version_of(REAL_CATALOG)},
            "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]}}
    return project([laser, cube], [plane("far", (60.0, 0.0, 0.0), radius_mm=20.0)],
                   [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "P", "maximum": 1}])


def _power_interval_cases() -> dict:
    """The geometric pupil-sample power basis (an F/1.2 objective whose 12 deg field a later clear aperture cuts after
    the stop truncated it severely: ``test_geometric_power_basis``) and a share taken before a plane (a monitor 0.5 mm
    before the folded reference design's mirror reads the beam within the share the mirror meets first:
    ``test_taken_before_interval``)."""
    from test_geometric_power_basis import _fast_project

    from os_optics.local import example_project_of_kind

    folded = example_project_of_kind("folded", REAL_CATALOG).payload()
    folded["observations"].append({"observation_id": "near_fold", "pose": {"position_mm": [-0.5, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
                                   "aperture": {"shape": "circle", "radius_mm": 10.0}})
    folded["targets"].append({"target_id": "near_fold_power", "metric": "power_W", "observation_id": "near_fold", "minimum": 0.9e-3, "unit": "W"})
    # a monitor 0.3 mm after the same fold: the share the mirror sends beyond it bounded by the energy flow's lines at their
    # own crossing of the mirror's plane (``test_start_and_certain_shares``): the 0.5 mW minimum is met on the interval
    after = example_project_of_kind("folded", REAL_CATALOG).payload()
    after["observations"].append({"observation_id": "after_fold", "pose": {"position_mm": [0.0, 0.3, 0.0], "quaternion_wxyz": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]},
                                  "aperture": {"shape": "circle", "radius_mm": 10.0}})
    after["targets"].append({"target_id": "after_fold_power", "metric": "power_W", "observation_id": "after_fold", "minimum": 0.5e-3, "unit": "W"})
    # rays launched at a 20 um waist do not follow its diffraction: a later stop's power is withheld; a stop-filled beam's
    # share on a small absorbing active area comes from its pupil samples (``test_geometric_basis_soundness``)
    from declared_helpers import declared_component, stop_model, with_models
    from local_helpers import ALONG_X, plane, project, source

    stops = [declared_component("A", "aperture", "S1", position=(0.0, 0.0, 0.0), quaternion=ALONG_X),
             declared_component("B", "aperture", "S2", position=(30.0, 0.0, 0.0), quaternion=ALONG_X)]
    withheld = with_models(project([source(position=(-20.0, 0.0, 0.0), waist_mm=0.02), *stops],
                                   [plane("OUT", (150.0, 0.0, 0.0), radius_mm=40.0)]),
                           [stop_model("S1", opening_radius_mm="0.2", edge_radius_mm="20"),
                            stop_model("S2", opening_radius_mm="0.5", edge_radius_mm="20")])
    detector = plane("DET", (50.0, 0.0, 0.0), radius_mm=0.2)
    detector["absorbing"] = True
    area = with_models(project([source(position=(-20.0, 0.0, 0.0), waist_mm=1.0), *stops], [detector],
                               [{"target_id": "det_power", "metric": "power_W", "observation_id": "DET", "minimum": 0.3e-3, "unit": "W"}]),
                       [stop_model("S1", opening_radius_mm="1", edge_radius_mm="20"),
                        stop_model("S2", opening_radius_mm="0.8", edge_radius_mm="20")])
    # the F/1.4 objective's fields launched from the axis: their chief rays fall beyond the stop's opening or a lens's, and a
    # passing pupil ray carries each field's reference (``test_promoted_reference``), on a Project's 16 x 24 pupil sampling
    from test_promoted_reference import _fast_project as _vignetted_project

    return {"geometric_power_basis": _fast_project("fast_f1p2.csv"), "taken_before_fold": folded, "taken_after_fold": after,
            "geometric_basis_withheld": withheld, "geometric_area_share": area,
            "promoted_reference": _vignetted_project("fast_f1p4.csv", (12.0, 13.0, 14.0), False, 16, 24)}


def _bounded_coatings_case() -> dict:
    """The reference interferometer of declared copies of the test catalog's parts whose coatings state a group-delay
    bound: its visibility and degree of coherence are stated within the bound, and a target on the coherence factor is
    judged on the interval (``test_bounded_coating_coherence``)."""
    from declared_helpers import declared_project
    from local_helpers import bounded_catalog_document

    from os_optics.local import example_project_of_kind

    catalog = bounded_catalog_document()
    document = example_project_of_kind("two_beam_interference", catalog).payload()
    document["targets"] += [
        {"target_id": "visibility", "metric": "fringe_visibility", "observation_id": "crossing_plane", "minimum": 0.5, "unit": "1"},
        {"target_id": "coherence", "metric": "coherence_factor", "observation_id": "crossing_plane", "minimum": 0.9999, "unit": "1"},
    ]
    return declared_project(document, catalog)


def _polarization_families_case() -> dict:
    """A dual-wavelength retarder (stated per wavelength) turned by its rotation parameter, then a circular polarizer, with a
    power target behind (``test_polarization_families``)."""
    from local_helpers import source
    from test_polarization_families import _rotatable, _with_angle, chain, circular, tabulated_retarder

    plate = _rotatable(tabulated_retarder([("532", "1"), ("1064", "0.5")], design=("1064", "0.5")))
    document = _with_angle(chain(source(wavelength_nm=1064.0), ("W", "waveplate", plate, 0.0), ("C", "polarizer", circular(), 0.0)), "W", 22.5)
    document["targets"] = [{"target_id": "power", "metric": "power_W", "observation_id": "OUT", "minimum": 0.0, "unit": "W"}]
    return document


def _crystal_spectrum_case() -> dict:
    """A calcite beam displacer declared in the Project (a crystal material and the element's optic axis), lit by a partially
    polarized source of a continuous Gaussian spectrum (8 quadrature lines; its coherence reported in source_spectra), with a
    power target behind (``test_project_crystals``, ``test_continuous_spectrum``)."""
    from test_project_crystals import displacer, partial, scene

    laser = partial(0.4)
    del laser["source_model"]["wavelength_nm"]
    laser["source_model"]["spectral_density"] = {"shape": {"kind": "GAUSSIAN", "center_nm": 780.0, "fwhm_nm": 20.0}, "line_count": 8}
    document = scene(laser, ("BD", displacer(), 0.0))
    document["targets"] = [{"target_id": "power", "metric": "power_W", "observation_id": "OUT", "minimum": 0.0, "unit": "W"}]
    return document


def _shg_turned_case() -> dict:
    """The BBO type I doubler of ``test_nonlinear_crystal_project`` with a d tensor (d22, d31 test values; crystal X off
    the plane of the wave normal and the optic axis), the input polarized 13 deg off the ordinary axis, and the whole bench
    turned about an oblique axis: the contraction with the traced fields and the overlap integral run on oblique vectors, and
    the harmonic's power and the efficiency are graded behind the crystal."""
    from test_nonlinear_crystal_project import _design, _theta

    theta = _theta()
    document = _design([{"target_id": "eta", "metric": "shg_conversion_efficiency", "device_instance_id": "X", "minimum": 1e-8},
                        {"target_id": "p532", "metric": "power_W", "observation_id": "out", "wavelength_nm": 532.0, "minimum": 0.0,
                         "unit": "W"}])
    model = document["optical_models"][0]["model"]
    del model["d_eff_pm_per_V"]
    model["d_tensor_pm_per_V"] = [["0", "0", "0", "0", "0.08", "-2.2"], ["-2.2", "2.2", "0", "0.08", "0", "0"],
                                  ["0.08", "0.08", "0", "0", "0", "0"]]
    a = 0.96
    b = math.sqrt(1.0 - a * a)
    model["crystal_x_axis"] = [repr(b * math.cos(theta)), repr(a), repr(-b * math.sin(theta))]
    document["components"][0]["source_model"]["polarization"]["angle_deg"] = 13.0
    return _turned(document)


def _assume_test_only_lossless_materials(document: dict) -> dict:
    """Isolate coating/phase physics on declared Project copies, never on public catalog materials.

    The seed catalog gives glass index data but no bulk-loss evidence. Explicit zero absorption here is a test hypothesis,
    not a claimed measurement or an alteration to the installed release.
    """
    assert document.get("model_materials"), "Only declared Project materials may assume zero bulk loss"
    for material in document["model_materials"]:
        assert "absorption_coefficient_per_mm" not in material and "bulk_loss" not in material
        material["absorption_coefficient_per_mm"] = "0"
    return document


def _michelson_cases() -> dict[str, dict]:
    """Explicitly lossless declared copies isolate the exact open/stated-phase power experiment; a separate copy of the
    real catalog Michelson retains bounded/UNKNOWN power (``test_interferometer_open_phase``)."""
    from declared_helpers import declared_project
    from michelson_helpers import expected, michelson, stated_phase_michelson

    catalog = json.loads(REAL_CATALOG.read_text(encoding="utf-8"))
    dark = expected(catalog).bright_extra_mm() + 780e-6 / 4.0
    # The partially polarized experiment remains on public parts: its route geometry and delay survive, while unstated
    # substrate absorption withholds numeric power and the aggregate status.
    partial = michelson(catalog, targets=[
        {"target_id": "dop", "metric": "degree_of_polarization", "observation_id": "out", "minimum": 0.3, "unit": "1",
         "parameters": [{"key": "path_id", "value": "SRC#0.0"}]},
        {"target_id": "delay", "metric": "time_delay_ps", "observation_id": "out", "minimum": -20.0, "unit": "ps",
         "parameters": [{"key": "path_id_a", "value": "SRC#0:p1.0"}, {"key": "path_id_b", "value": "SRC#0:p1.1"}]},
        {"target_id": "arrival", "metric": "group_delay_ps", "observation_id": "out", "minimum": 0.0, "unit": "ps",
         "parameters": [{"key": "path_id", "value": "SRC#0.1"}]}])
    partial["components"][0]["source_model"]["polarization"] = {
        "kind": "PARTIAL", "degree_of_polarization": 0.4, "polarized": {"kind": "ELLIPTICAL", "orientation_deg": 30.0, "ellipticity_deg": -12.0}}
    return {"michelson_open_phase": _assume_test_only_lossless_materials(declared_project(michelson(catalog), catalog)),
            "michelson_stated_phases": _assume_test_only_lossless_materials(stated_phase_michelson(catalog, dark)),
            "michelson_public_unstated_bulk": michelson(catalog), "michelson_partially_polarized_delays": partial}


def _result_semantics_cases() -> dict[str, dict]:
    """Partial-beam residuals and ring returns at plane level. ``tail_energy_flow``: the recorded tail, a beam of 20 um waist that leaves the scene past a 5 mm
    disc parallel to it, 2 mm off axis (``test_tail_dark_planes_and_surface_coincidence``). ``ghost_on_a_plane``: a tilted window's
    ghost, on a plane no traced beam reaches. ``coincident_normal_mirror`` and ``coincident_splitter``: a plane through a surface's
    chief-ray hit point (the arriving and the returning beam; the arriving beam alone, with its literal crossing bound).
    ``coating_use_refused``: the beam expander of declared copies of its catalog lens, rated: ghosts fill the lenses, and each
    face's coating record is refused from the glass (``test_coating_use``)."""
    from declared_helpers import declared_project, nbk7, sequence, surface, with_models
    from local_helpers import AGAINST_X, ALONG_X, ALONG_Y, FOLD_X_TO_Y, element, plane, project, quat_multiply, source
    from michelson_helpers import BS_PLATE, FLAT_MIRROR_R95
    from os_optics.local import example_project_of_kind

    catalog = json.loads(REAL_CATALOG.read_text(encoding="utf-8"))
    version = catalog["catalog_version"]
    tail = project([source(position=(0.0, 0.0, 0.0), waist_mm=0.02)], [plane("T", (90.0, 2.0, 0.0), ALONG_Y, radius_mm=5.0)],
                   [{"target_id": "side", "metric": "power_W", "observation_id": "T", "maximum": 1e-7, "unit": "W"}])
    half = math.radians(2.5)
    tilted = quat_multiply((math.cos(half), 0.0, 0.0, math.sin(half)), ALONG_X)  # (the window's axis turned 5 deg about Z)
    ghost = with_models(project([source(position=(0.0, 0.0, 0.0), waist_mm=0.5),
                                 {"instance_id": "W", "optical_role": "window", "custom_part": {"custom_part_key": "own-w", "description": "Uncoated window."},
                                  "nominal_pose": {"position_mm": [50.0, 0.0, 0.0], "quaternion_wxyz": list(tilted)}, "optical_model_id": "win"}],
                                [plane("T", (20.0, -30.0 * math.tan(math.radians(10.0)), 0.0), ALONG_X, radius_mm=4.0)],
                                [{"target_id": "ghost", "metric": "power_W", "observation_id": "T", "maximum": 1e-3, "unit": "W"}]),
                        [{"model_id": "win", "model": sequence([surface(1, "0", clear="12"), surface(2, "3", clear="12")], ["N-BK7"], edge="13")}], [nbk7()])
    mirror = project([source(position=(-100.0, 0.0, 0.0), waist_mm=0.5), element("M", "mirror", FLAT_MIRROR_R95, (0.0, 0.0, 0.0), AGAINST_X, catalog_version=version)],
                     [plane("T", (0.0, 0.0, 0.0), ALONG_X)],
                     [{"target_id": "arrival", "metric": "power_W", "observation_id": "T", "minimum": 9e-4, "unit": "W"}])
    splitter = project([source(position=(-100.0, 0.0, 0.0), waist_mm=0.5, polarization_deg=90.0), element("BS", "splitter", BS_PLATE, (0.0, 0.0, 0.0), FOLD_X_TO_Y, catalog_version=version)],
                       [plane("T", (0.0, 0.0, 0.0), ALONG_X)],
                       [{"target_id": "arrival", "metric": "power_W", "observation_id": "T", "minimum": 9e-4, "unit": "W"}])
    refused = _assume_test_only_lossless_materials(
        declared_project(example_project_of_kind("beam_expander", REAL_CATALOG).payload(), catalog))
    rating = {"regime": "CW", "value": "10000", "unit": "W/cm2", "wavelength_nm": "780", "value_definition": "PEAK", "beam_diameter_mm": "50",
              "beam_diameter_definition": "1/E2", "polarization_condition": "linear", "status": "DECLARED", "basis": "fixture data"}
    for model in refused["optical_models"]:
        model["power_ratings"] = [rating]
    return {"tail_energy_flow": tail, "ghost_on_a_plane": ghost, "coincident_normal_mirror": mirror, "coincident_splitter": splitter, "coating_use_refused": refused}


def _faraday_cases() -> dict[str, dict]:
    """An isolator (polarizer, Faraday rotator, analyzer, one source from each end), a rotator met by a line its table does
    not cover and one in front of a detector, whose network is refused (``test_faraday_rotator``)."""
    from local_helpers import ALONG_X
    from test_faraday_rotator import LAMBDA_NM, _single, isolator, rotator_model

    detector = _single(rotator_model(), LAMBDA_NM)
    detector["observations"][0].update(absorbing=True, medium="air")
    return {"faraday_isolator": isolator(rotator_model(), ALONG_X), "faraday_outside_table": _single(rotator_model(), 1200.0),
            "faraday_before_a_detector": detector}


def _facet_solid_case() -> dict:
    """A declared N-BK7 corner cube (FACET_SOLID) retroreflecting a beam (``test_facet_solid_project``): its facets traced as planes."""
    from test_facet_solid_project import _corner_cube_design

    return _corner_cube_design()


def _weak_signal_chain_case() -> dict:
    """Laser, fluorescence of a sample excited by it, collection lens, detector with its noise, chopper and lock-in
    (``test_weak_signal_chain``): the sample's emission derived from a first trace, the detection targets graded."""
    from test_weak_signal_chain import CHAIN_TARGETS, chain_detection, chain_project, fluorescence

    return chain_project(fluorescence(absorbance=1e-4, sampling={"sample_count": 1024, "seed": 1}), chain_detection(), CHAIN_TARGETS[:3])


def _grating_cases() -> dict[str, dict]:
    """A grating without a response (its orders' powers not known, their positions at the closed form), one with a stated table, and one
    in front of a detector, whose network is refused (``test_diffraction_grating``)."""
    from declared_helpers import declared_component, with_models
    from local_helpers import ALONG_X, plane, project
    from test_diffraction_grating import NORMAL_ROWS, grating, laser, table

    def case(model: dict, *, absorbing: bool = False) -> dict:
        screen = plane("SCREEN", (120.0, 0.0, 0.0), ALONG_X, radius_mm=300.0)
        if absorbing:
            screen.update(absorbing=True, medium="air")
        return with_models(project([laser(550.0, polarization_deg=90.0), declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)], [screen]), [model],
                           design_intent="OPTICAL_SYSTEM")

    return {"grating_without_response": case(grating(mode="TRANSMISSION", density="300", orders=(-2, -1, 0, 1, 2))),
            "grating_with_response": case(grating(mode="TRANSMISSION", density="300", response=table(NORMAL_ROWS))),
            "grating_before_a_detector": case(grating(mode="TRANSMISSION", density="300", orders=(0,)), absorbing=True)}


def _tabulated_scope_cases() -> dict[str, dict]:
    """A tabulated thin lens used outside its domain (450 nm below its table, 8 deg beyond its 2 deg field
    angle) leaves the spot behind it INDETERMINATE (``applicability_issue_codes``); a plane before the lens and a plane only
    a beam beside it reaches stay decided (``test_tabulated_lens_applicability``)."""
    from test_tabulated_lens_applicability import _model, _tilted_project, _waist, beside_and_before_project
    from test_tabulated_thin_lens import ttl_project

    return {"tabulated_lens_outside_band": ttl_project(((450.0, 0.001),), model=_model(), targets=[_waist()]),
            "tabulated_lens_beyond_field_angle": _tilted_project(8.0),
            "tabulated_lens_beside_and_before": beside_and_before_project()}


#: A turn of a whole bench about an oblique axis (1, 2, 3) by 23 deg: every surface, beam and plane oblique to the design axes.
OBLIQUE_TURN = (math.cos(math.radians(11.5)), *(math.sin(math.radians(11.5)) * c / math.sqrt(14.0) for c in (1.0, 2.0, 3.0)))


def _turned(document: dict, turn: tuple[float, float, float, float] = OBLIQUE_TURN) -> dict:
    """``document`` with every component and free observation plane turned by ``turn`` about the design origin."""
    from local_helpers import quat_multiply

    w, x, y, z = turn
    rot = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
           [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
           [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    out = json.loads(json.dumps(document))
    poses = [c["nominal_pose"] for c in out["components"]] + [o["pose"] for o in out["observations"] if not o.get("device_instance_id")]
    for pose in poses:
        p = pose["position_mm"]
        pose["position_mm"] = [sum(rot[i][j] * p[j] for j in range(3)) for i in range(3)]
        pose["quaternion_wxyz"] = list(quat_multiply(turn, pose["quaternion_wxyz"]))
    return out


def _oblique_cases() -> dict[str, dict]:
    """PV-F1: beams crossing observation planes that are not square to a design axis. A beam 3 deg off +X to a plane 150 mm
    away tilted 10 deg further (no optics: the ray / plane intersection and k0 OPL alone); the catalog wedge prism with two
    lines and a plane 150 mm behind it facing the mean exit direction (the physics evidence's c5_wedge); and the oblique
    window / rotary attenuator, the coating + polarization chain and the elliptical two-arm interferometer turned as a
    whole by 23 deg about (1, 2, 3)."""
    from local_helpers import plane, project, source
    from test_coating_tables_and_phases import _quaternion

    beam, tilt = math.radians(3.0), math.radians(13.0)
    d = (math.cos(beam), -math.sin(beam), 0.0)
    facing = (math.cos(tilt), -math.sin(tilt), 0.0)
    free = project([source(position=(0.0, 0.0, 0.0), wavelength_nm=633.0, waist_mm=0.5,
                           quaternion=_quaternion(x_axis=(-d[1], d[0], 0.0), z_axis=d))],
                   [plane("P", tuple(150.0 * c for c in d), _quaternion(x_axis=(-facing[1], facing[0], 0.0), z_axis=facing))],
                   [{"target_id": "phase", "metric": "phase_rad", "observation_id": "P", "minimum": -4.0, "maximum": 4.0, "unit": "rad"},
                    {"target_id": "power", "metric": "power_W", "observation_id": "P", "minimum": 0.0, "unit": "W"}])
    return {"oblique_plane_free_beam": free, "wedge_prism_two_lines": _wedge_case(),
            "turned_rated_oblique_window_rotary_nd": _turned(_rated_oblique_rotary_case()),
            "turned_coatings_phases_polarization": _turned(_coating_and_polarization_project()),
            "turned_coherent_elliptical_interference": _turned(_coherent_elliptical_case())}


def _line_focus_cases() -> dict[str, dict]:
    """The catalog's rectangular cylindrical lens focusing a 0.5 mm beam to a line, captured by an off-centre pinhole, a
    turned slit and a wider pinhole at the focus (the truncation integrals of strongly astigmatic footprints: closed-form
    erf chord / edge integrals, principal-frame disk, turned rectangle), and that bench turned about an oblique axis."""
    from test_gaussian_truncation_astigmatic import line_focus_project

    return {"cylinder_line_focus": line_focus_project(), "turned_cylinder_line_focus": _turned(line_focus_project())}


def _wedge_case() -> dict:
    """The catalog wedge prism (N-BK7, entrance at normal incidence, exit face tilted) with a 633 nm + 532 nm source and a
    plane 150 mm behind the exit point facing the mean of the two exit directions (pose from the physics evidence's
    c5_wedge: the first Python / Pyodide result-hash difference, PV-F1)."""
    from os_optics.local import load_public_catalog

    catalog = load_public_catalog(REAL_CATALOG)
    sku = "OS-OPT-PRISM-WEDGE-D25P4-W11P367-NBK7-UNC-350-2000-T5P555-CA22P86"
    along_x = [0.5, 0.5, 0.5, 0.5]
    laser = {"instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser-src", "description": "two-line laser (fixture)"},
             "nominal_pose": {"position_mm": [-50.0, 0.0, 0.0], "quaternion_wxyz": along_x},
             "source_model": {"model_kind": "GAUSSIAN_BEAM", "waist_radius_mm": 0.5, "waist_position_mm": 0.0, "m2": 1.0,
                              "polarization": {"kind": "LINEAR", "angle_deg": 0.0},
                              "spectral_lines": [{"wavelength_nm": 633.0, "power_w": 0.001}, {"wavelength_nm": 532.0, "power_w": 0.001}]}}
    prism = {"instance_id": "WP", "optical_role": "prism", "product": {"optical_sku": sku, "catalog_version": catalog.catalog_version},
             "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": along_x}}
    screen = {"observation_id": "P", "aperture": {"shape": "circle", "radius_mm": 12.5},
              "pose": {"position_mm": [151.80004623807028, -15.765260392002356, 0.0],
                       "quaternion_wxyz": [0.525619095911355, 0.525619095911355, 0.472995312887273, 0.472995312887273]}}
    targets = [{"target_id": f"power_{w}", "metric": "power_W", "observation_id": "P", "wavelength_nm": float(w), "minimum": 0.0, "unit": "W"}
               for w in (633, 532)]
    return {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM",
            "components": [laser, prism], "observations": [screen], "targets": targets}


def _coherent_elliptical_case() -> dict:
    """The two-arm interferometer with an elliptical source (complex Jones vectors, phase-shifted arms): coherent power,
    polarization overlap, visibility and phase difference from FMA-free complex dot products."""
    from os_optics.local import example_project_of_kind

    document = example_project_of_kind("two_beam_interference", REAL_CATALOG).payload()
    document["components"][0]["source_model"]["polarization"] = {"kind": "ELLIPTICAL", "orientation_deg": 20.0, "ellipticity_deg": -30.0}
    plane_id = document["observations"][0]["observation_id"]
    for target in document["targets"]:
        if target["metric"] == "polarization_overlap":
            target["minimum"] = 0.2  # (the arms' polarizations overlap by about 0.31)
        if target["metric"] == "intensity_balance":
            target["minimum"] = 0.0  # (the splitter's s and p reflectances differ)
    document["targets"] += [{"target_id": "coherent", "metric": "coherent_power_W", "observation_id": plane_id, "minimum": 0.0, "unit": "W"},
                            {"target_id": "overlap", "metric": "polarization_overlap", "observation_id": plane_id, "minimum": 0.0},
                            {"target_id": "phase", "metric": "phase_difference_rad", "observation_id": plane_id, "minimum": -4.0,
                             "maximum": 4.0}]
    return document


def _rated_oblique_rotary_case() -> dict:
    """Exposures at oblique incidence and a turned rotary attenuator (libm-free AOI, footprint, azimuth and 10^-OD): a beam
    3 mm off the wheel's centre (azimuth 90 deg) through a LINEAR_ROTARY ND turned 35 deg, then a CW-rated N-BK7 window tilted 30 deg (fixture
    rating, its value stated as a peak irradiance for linear light, with a test beam three times the beam's 1/e^2 diameter at
    the window -- larger than its footprint on either face: a threshold at least the rated value) with a utilization and a
    power target."""
    from declared_helpers import declared_component, nbk7, sequence, surface, with_models
    from local_helpers import plane, project, source
    from test_coating_tables_and_phases import _quaternion

    nd = declared_component("ND", "filter", "wheel", (-60.0, 0.0, 0.0))
    nd["selected_parameters"] = [{"semantic_key": "nd_rotation_deg", "value_kind": "DECIMAL", "value": "35", "unit_code": "deg"}]
    tilt = math.radians(30.0)
    window = declared_component("W", "window", "win", (20.0, 0.0, 0.0),
                                quaternion=_quaternion(x_axis=(0.0, 1.0, 0.0), z_axis=(math.cos(tilt), 0.0, math.sin(tilt))))
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0},
               {"target_id": "p", "metric": "power_W", "observation_id": "det", "minimum": 0.0001}]
    document = project([source(position=(-150.0, 0.0, 3.0), waist_mm=0.3, power_w=0.5), nd, window],
                       [plane("det", (80.0, 0.0, 3.0), radius_mm=20.0)], targets)
    wheel = {"model_id": "wheel", "model": {
        "model_kind": "NEUTRAL_DENSITY", "model_version": "1", "fidelity": "IDEAL_THIN_ELEMENT", "status": "SPECIFIED",
        "law": {"kind": "LINEAR_ROTARY", "od_start": "0.1", "od_end": "2.1", "arc_start_deg": "0", "arc_end_deg": "300"},
        "clear_radius_mm": "20", "edge_radius_mm": "22", "valid_band_nm": {"min_nm": "400", "max_nm": "1100", "status": "SPECIFIED"},
        "position_parameter": "nd_rotation_deg", "approximations": ["THIN_ELEMENT"]}}
    win = {"model_id": "win", "model": sequence([surface(1, "0"), surface(2, "3")], ["N-BK7"]),
           "power_ratings": [{"regime": "CW", "value": "500", "unit": "W/cm2", "wavelength_nm": "780", "value_definition": "PEAK",
                              "beam_diameter_mm": f"{3.0 * _beam_mm(0.3, 170.0, 780.0):.6f}", "beam_diameter_definition": "1/E2",
                              "polarization_condition": "linear", "status": "DECLARED", "basis": "fixture data"}]}
    return with_models(document, [wheel, win], [nbk7()])


def _beam_mm(waist_mm: float, path_mm: float, wavelength_nm: float) -> float:
    """A fixture source's 1/e^2 diameter ``path_mm`` of air from its waist: 2 w0 sqrt(1 + (z / z_R)^2)."""
    return 2.0 * waist_mm * math.sqrt(1.0 + (path_mm / (math.pi * waist_mm ** 2 / (wavelength_nm * 1e-6))) ** 2)


def _rated_conditions_case() -> dict:
    """Pulsed light (10 ns, 10 Hz, a 2 mm beam) on an N-BK7 window with ratings that state a test beam or a pulse count
    (fixture data, values stated as peaks): a test beam smaller than the beam (a threshold at most the rated value), one
    larger (at least), one whose unstated definition leaves the beam between its readings, one over 100 pulses without a
    protocol, with a utilization target. None states pulses per tested site, so none is verified: not decided."""
    from declared_helpers import declared_component, nbk7, sequence, surface, with_models
    from local_helpers import plane, project, source

    laser = source(position=(-200.0, 0.0, 0.0), waist_mm=1.0, power_w=0.5)
    laser["source_model"]["pulse"] = {"repetition_rate_hz": 10.0, "pulse_duration_s": 1e-8, "duration_definition": "FWHM"}
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    document = project([laser, declared_component("W", "window", "win", (0.0, 0.0, 0.0))], [plane("det", (60.0, 0.0, 0.0), radius_mm=20.0)],
                       targets)

    def rating(**extra) -> dict:
        return {"regime": "PULSED", "value": "5", "unit": "J/cm2", "wavelength_nm": "780", "pulse_duration_s": "0.00000001",
                "repetition_rate_hz": "10", "value_definition": "PEAK", "status": "DECLARED", "basis": "fixture data", **extra}

    win = {"model_id": "win", "model": sequence([surface(1, "0"), surface(2, "3")], ["N-BK7"]),
           "power_ratings": [rating(beam_diameter_mm="0.05", beam_diameter_definition="NOT_STATED"),
                             rating(value="0.5", beam_diameter_mm="5", beam_diameter_definition="1/E2"),
                             rating(beam_diameter_mm="1.5"), rating(pulse_count=100)]}
    return with_models(document, [win], [nbk7()])


def _rated_counts_case() -> dict:
    """Two pulsed sources (10 ns, 10 Hz) through one N-BK7 window, one that keeps pulsing (UNLIMITED) and one that sends 60
    pulses (from 60 to unboundedly many per site), with fixture ratings: undecidable -- 100 pulses with neither a protocol
    nor a value definition (not verifiable), 100-on-1 of a peak value (pulses per site undetermined), a typical value --
    and decidable -- a single pulse of a peak value and 50-on-1 of a value stated as the total over the test beam's 1/e^2
    area (threshold twice it), both for linear light with the beams' own 1/e^2 diameter at the window as their test beam,
    at most rated and exceeded by each pulse: the per-site pulse counts and the rated-value readings in both runtimes."""
    from declared_helpers import declared_component, nbk7, sequence, surface, with_models
    from local_helpers import plane, project, source

    pulse = {"repetition_rate_hz": 10.0, "pulse_duration_s": 1e-8, "duration_definition": "FWHM"}
    a = source("A", position=(-200.0, 0.0, 0.0), waist_mm=1.0, power_w=0.5)
    a["source_model"]["pulse"] = dict(pulse, total_pulses="UNLIMITED")
    b = source("B", position=(-200.0, 2.0, 0.0), waist_mm=1.0, power_w=0.5)
    b["source_model"]["pulse"] = dict(pulse, total_pulses=60)
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}]
    document = project([a, b, declared_component("W", "window", "win", (0.0, 0.0, 0.0))], [plane("det", (60.0, 0.0, 0.0), radius_mm=20.0)],
                       targets)
    # (both beams' 1/e^2 diameter at the window, 200 mm from their 1 mm waists; both linear)
    stated = {"beam_diameter_mm": f"{_beam_mm(1.0, 200.0, 780.0):.6f}", "beam_diameter_definition": "1/E2",
              "polarization_condition": "linear"}

    def rating(**extra) -> dict:
        return {"regime": "PULSED", "value": "5", "unit": "J/cm2", "wavelength_nm": "780", "pulse_duration_s": "0.00000001",
                "repetition_rate_hz": "10", "status": "DECLARED", "basis": "fixture data", **extra}

    win = {"model_id": "win", "model": sequence([surface(1, "0"), surface(2, "3")], ["N-BK7"]),
           "power_ratings": [rating(pulse_count=100), rating(pulse_count=100, test_protocol="S_ON_1", value_definition="PEAK"),
                             rating(pulse_count=1, value="0.5", value_definition="PEAK", **stated), rating(status="TYPICAL"),
                             rating(value="0.05", pulse_count=50, test_protocol="S_ON_1", value_definition="TOTAL_PER_1E2_AREA",
                                    **stated)]}
    return with_models(document, [win], [nbk7()])


def _rated_left_out_only_case() -> dict:
    """A pulsed beam (10 ns, 10 Hz) crossing an uncoated N-BK7 window and, 60 mm beside it, a second window rated for that
    line (fixture data, a peak value) that only the light left out as negligible reaches: its conditions are compared with
    that light's line and pulse train, and its exposure is bounded by that light's diffraction limit, within the value; the
    rating states no pulse count (nor a test beam or polarization), so it is not verified and the target is not decided."""
    from declared_helpers import declared_component, nbk7, sequence, surface, with_models
    from local_helpers import plane, project, source

    laser = source(position=(-200.0, 0.0, 0.0), waist_mm=1.0, power_w=0.5)
    laser["source_model"]["pulse"] = {"repetition_rate_hz": 10.0, "pulse_duration_s": 1e-8, "duration_definition": "FWHM"}
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "R", "maximum": 1.0}]
    document = project([laser, declared_component("W", "window", "win", (0.0, 0.0, 0.0)),
                        declared_component("R", "window", "rated", (0.0, 60.0, 0.0))],
                       [plane("det", (60.0, 0.0, 0.0), radius_mm=20.0)], targets)
    plate = sequence([surface(1, "0"), surface(2, "3")], ["N-BK7"])
    rating = {"regime": "PULSED", "value": "1000000", "unit": "J/cm2", "wavelength_nm": "780", "pulse_duration_s": "0.00000001",
              "repetition_rate_hz": "10", "value_definition": "PEAK", "status": "DECLARED", "basis": "fixture data"}
    return with_models(document, [{"model_id": "win", "model": plate}, {"model_id": "rated", "model": plate, "power_ratings": [rating]}],
                       [nbk7()])


def _rated_window_stack_case() -> dict:
    """Three uncoated N-BK7 windows in a row, the last one rated (fixture data, a peak value): the stack's faces
    reflect more than the sufficient all-orders series allows (A > 1), so its
    left-out light is bounded by the alternating series of its reflections (``validation/closed_returns.py``), in the
    browser runtime exactly as in Python."""
    from declared_helpers import declared_component, nbk7, sequence, surface, with_models
    from local_helpers import plane, project, source

    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W3", "maximum": 1.0}]
    document = project([source(position=(-200.0, 0.0, 0.0), waist_mm=1.0, power_w=0.5),
                        declared_component("W1", "window", "win", (0.0, 0.0, 0.0)),
                        declared_component("W2", "window", "win", (30.0, 0.0, 0.0)),
                        declared_component("W3", "window", "rated", (60.0, 0.0, 0.0))],
                       [plane("det", (120.0, 0.0, 0.0), radius_mm=20.0)], targets)
    plate = sequence([surface(1, "0"), surface(2, "3")], ["N-BK7"])
    rating = {"regime": "CW", "value": "1000000000000000", "unit": "W/cm2", "wavelength_nm": "780", "value_definition": "PEAK",
              "status": "DECLARED", "basis": "fixture data"}
    return with_models(document, [{"model_id": "win", "model": plate}, {"model_id": "rated", "model": plate, "power_ratings": [rating]}],
                       [nbk7()])


def _rated_untraced_case() -> dict:
    """A rated window reached only by a flat mirror coating's transmission (light the trace does not follow, followed to the
    surfaces by probe traces): its utilization is bounded, not a known 0 (UNKNOWN)."""
    from test_exposure_bounds import _mirror_window

    return _mirror_window("0.5", "0.5")


def _coating_and_polarization_project() -> dict:
    """An elliptical source, an angle-resolved coating with tabulated phases, a banded waveplate and polarizer."""
    from test_polarization_extensions import coating_and_polarization_project

    return coating_and_polarization_project()


def _tail_project(half_angle_deg: float) -> dict:
    """A 1 mW, w0 1 mm beam 2.2 mm beside a 0.5 mm detector at 40 mm whose normal is tilted 25 deg, with a 1 nW maximum: the
    Gaussian tail on the area arrives at 25 deg -- outside a 10 deg acceptance cone (a known zero, PASS), inside a 30 deg one
    (it may arrive, UNKNOWN)."""
    from local_helpers import project, source

    half = math.radians(25.0) / 2  # the plane frame (local +Z along +X) turned 25 deg about design Z
    w1, x1, y1, z1 = (0.5, 0.5, 0.5, 0.5)
    w2, z2 = math.cos(half), math.sin(half)
    quaternion = [float(format(v, ".15g")) for v in (w2 * w1 - z2 * z1, w2 * x1 - z2 * y1, w2 * y1 + z2 * x1, w2 * z1 + z2 * w1)]
    detector = {"observation_id": "DET", "pose": {"position_mm": [40.0, 0.0, 0.0], "quaternion_wxyz": quaternion},
                "aperture": {"shape": "circle", "radius_mm": 0.5}, "absorbing": True, "acceptance_half_angle_deg": half_angle_deg, "medium": "air"}
    return project([source(position=(0.0, 2.2, 0.0), wavelength_nm=633.0, waist_mm=1.0, power_w=0.001)], [detector],
                   [{"target_id": "det_max", "metric": "power_W", "observation_id": "DET", "maximum": 1e-9, "unit": "W"}])


def _native_splitter_cavity() -> dict:
    """A Fabry-Perot of two native plate splitters (R = 0.7): deep branches get compacted path ids, the run
    and its scene bind, and the network export refuses with a network code."""
    from test_path_ids import native_splitter_cavity

    return native_splitter_cavity("0.7")


def _declared_cases(reference: dict) -> dict[str, dict]:
    """Declared optical models in the browser runtime exactly as in Python: the reference design with its lens as a
    declared copy (no catalog component left), an air-spaced objective of one declared model, the same objective as a
    ``COMPOUND`` unit with an aperture stop, an ideal thin lens with an off-axis slit (OUT_OF_MODEL_SCOPE: a severe
    truncation), a plano-axicon (OUT_OF_MODEL_SCOPE: the Gaussian beam is undefined behind a cone; its power is traced)
    and a tabulated thin lens (WARNING: its unmodelled response may reach the plane)."""

    from declared_helpers import (
        compound_model,
        cylinder,
        declared_component,
        declared_project,
        doublet_model,
        flint,
        nbk7,
        objective_model,
        plano_convex_model,
        stop_model,
        thin_lens_model,
        with_models,
    )

    copied = declared_project(reference, REAL_CATALOG)
    base = json.loads(json.dumps(reference))
    base["components"] = base["components"][:1]
    base["targets"] = [{"target_id": "power", "metric": "power_W", "observation_id": "focal_plane", "minimum": 0.0003, "unit": "W"}]
    objective = json.loads(json.dumps(base))
    objective["components"].append(declared_component("OBJ", "lens", "objective"))
    objective = with_models(objective, [objective_model()], [nbk7(), flint()], design_intent="OPTICAL_SYSTEM")
    unit = json.loads(json.dumps(base))
    unit["components"].append(declared_component("OBJ", "lens", "unit"))
    unit = with_models(unit, [doublet_model(), plano_convex_model(), stop_model(opening_radius_mm="6", edge_radius_mm="10"),
                              compound_model("unit", [("L1", "doublet", (0, 0, 0), (1, 0, 0, 0)), ("STOP", "stop", (0, 0, 14.0), (1, 0, 0, 0)),
                                                      ("L2", "pcx", (0, 0, 25.0), (0.0, 1.0, 0.0, 0.0))], cylinder("22", "-1", "30"))],
                       [nbk7(), flint()], design_intent="OPTICAL_SYSTEM")
    thin = json.loads(json.dumps(base))
    thin["components"] += [declared_component("T", "lens", "thin"), declared_component("S", "aperture", "slit", (20.0, 0.0, 0.0))]
    thin = with_models(thin, [thin_lens_model(focal_length_mm="80"), stop_model("slit", opening_half_widths_mm=["0.2", "3"], opening_center_mm=["0.1", "0"],
                                                                               edge_half_widths_mm=["10", "10"])])
    from declared_helpers import sequence, surface

    axicon = json.loads(json.dumps(base))
    axicon["components"].append(declared_component("AX", "lens", "ax"))
    axicon = with_models(axicon, [{"model_id": "ax", "model": sequence([surface(1, "0"), surface(2, "5", linear_sag_coefficient="-0.2")], ["N-BK7"])}],
                         [nbk7()])
    tabulated = json.loads(json.dumps(base))
    tabulated["components"].append(declared_component("ML", "lens", "ml"))
    evidence = {"method": "fixture response", "tools": [{"name": "fixture", "version": "1"}], "source_data_sha256": "0" * 64,
                "fit_model": "fixture", "validation_summary": "fixture", "date": "2026-10-15"}
    tabulated = with_models(tabulated, [{"model_id": "ml", "model": {
        "model_kind": "TABULATED_THIN_LENS", "model_version": "1", "status": "DERIVED", "evidence": evidence,
        "rows": [{"wavelength_nm": "760", "focal_length_mm": "103", "transmittance": "0.85"},
                 {"wavelength_nm": "800", "focal_length_mm": "98", "transmittance": "0.9"}],
        "focal_interpolation": "LINEAR_IN_OPTICAL_POWER", "transmittance_interpolation": "LINEAR", "clear_radius_mm": "10",
        "edge_radius_mm": "12.7", "max_field_angle_deg": "5", "max_beam_radius_mm": "2", "validated_polarizations": ["ANY"],
        "supported_metrics": [{"metric": "transmittance", "abs_error": "0.05", "basis": "fixture"}]}}])
    return {"declared_copy": copied, "declared_objective": objective, "declared_compound_unit": unit, "declared_thin_lens_slit": thin,
            "declared_axicon": axicon, "declared_tabulated_thin_lens": tabulated, "declared_attenuator_ratings": _attenuator_ratings_case()}


def _attenuator_ratings_case() -> dict:
    """A pulsed two-line source (a single shot) through a position-dependent attenuator (gradient shifted by its position
    parameter) and a thin lens with a declared single-pulse power rating of a peak value (fixture data: the source's rate,
    the beam's 1/e^2 diameter at the lens, linear light), with a utilization and a two-colour overlap target."""

    from declared_helpers import declared_component, thin_lens_model, with_models
    from local_helpers import plane, project, source

    laser = source(position=(-100.0, 0.0, 0.0), waist_mm=0.5)
    model = laser["source_model"]
    del model["wavelength_nm"], model["power_w"]
    model["spectral_lines"] = [{"wavelength_nm": 532.0, "power_w": 0.002}, {"wavelength_nm": 540.0, "power_w": 0.001}]
    model["pulse"] = {"repetition_rate_hz": 1000.0, "pulse_duration_s": 1e-8, "duration_definition": "FWHM", "total_pulses": 1}
    nd = declared_component("ND", "filter", "nd", (-50.0, 0.0, 0.0))
    nd["selected_parameters"] = [{"semantic_key": "nd_offset_mm", "value_kind": "DECIMAL", "value": "2", "unit_code": "mm"}]
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "T", "maximum": 1.0},
               {"target_id": "overlap", "metric": "geometric_overlap_ratio", "observation_id": "det", "wavelength_nm": 532.0,
                "wavelength_nm_b": 540.0, "minimum": 0.9}]
    document = project([laser, nd, declared_component("T", "lens", "thin")], [plane("det", (60.0, 0.0, 0.0), radius_mm=20.0)], targets)
    lens = thin_lens_model("thin", "100")
    lens["power_ratings"] = [{"regime": "PULSED", "value": "5", "unit": "J/cm2", "wavelength_nm": "532", "pulse_duration_s": "0.00000001",
                              "pulse_count": 1, "value_definition": "PEAK", "repetition_rate_hz": "1000",
                              "beam_diameter_mm": f"{_beam_mm(0.5, 100.0, 532.0):.6f}", "beam_diameter_definition": "1/E2",
                              "polarization_condition": "linear", "status": "DECLARED", "basis": "fixture data"}]
    attenuator = {"model_id": "nd", "model": {
        "model_kind": "NEUTRAL_DENSITY", "model_version": "1", "fidelity": "IDEAL_THIN_ELEMENT", "status": "SPECIFIED",
        "law": {"kind": "LINEAR_GRADIENT", "od_start": "0", "od_end": "1", "x_start_mm": "-10", "x_end_mm": "10"},
        "clear_radius_mm": "12", "edge_radius_mm": "12.7", "valid_band_nm": {"min_nm": "400", "max_nm": "700", "status": "SPECIFIED"},
        "position_parameter": "nd_offset_mm", "approximations": ["THIN_ELEMENT"]}}
    return with_models(document, [lens, attenuator])


def _beam_scope_cases(reference: dict) -> dict[str, dict]:
    """The v0.14 physics review's reproducers, built from the seed catalog release by query (no generated SKU is pinned): a
    window half over the beam its chief ray misses (R1), a plano-convex lens turned flat side first focusing an
    aberration-dominated spot (R2) and light leaving a window turned 70 deg through its side wall (R3). All three are
    OUT_OF_MODEL_SCOPE, in the browser (Pyodide) runtime exactly as natively."""
    from decimal import Decimal

    from os_optics.local import load_public_catalog

    from local_helpers import quat_multiply

    catalog = load_public_catalog(REAL_CATALOG)
    ordered = sorted(catalog.products, key=lambda product: product.optical_sku)

    def covers(product, nm: int) -> bool:
        band = product.simulation_models[0].valid_band_nm
        return Decimal(band.min_nm) <= nm <= Decimal(band.max_nm)

    def uncoated(product) -> bool:
        return all(surface.coating_id is None for surface in product.simulation_models[0].surfaces)

    along_x, against_x = [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5]
    window = next(p for p in ordered if p.category.value == "WINDOW" and uncoated(p) and covers(p, 633) and covers(p, 780))
    model = window.simulation_models[0]
    edge, clear = float(model.edge_radius_mm), float(model.clear_radius_mm)
    thickness = float(Decimal(model.surfaces[-1].vertex_z_mm) - Decimal(model.surfaces[0].vertex_z_mm))

    def with_window(pose_position, quaternion, source_position, waist_mm, wavelength_nm) -> dict:
        document = json.loads(json.dumps(reference))
        document["components"][0]["nominal_pose"]["position_mm"] = source_position
        document["components"][0]["source_model"].update(waist_radius_mm=waist_mm, wavelength_nm=wavelength_nm)
        document["components"].append({"instance_id": "W", "optical_role": "window",
                                       "product": {"optical_sku": window.optical_sku, "catalog_version": catalog.catalog_version},
                                       "nominal_pose": {"position_mm": pose_position, "quaternion_wxyz": quaternion}})
        return document

    # R1: the chief ray passes 0.01 mm beside the window's edge; about half of the 1 mm beam goes through it
    partial = with_window([-100.0, edge + 0.01, 0.0], along_x, [-200.0, 0.0, 0.0], 1.0, 780.0)
    # R3: turned 70 deg about the vertical, entered inside its clear aperture close enough to the edge that the refracted
    # ray (any index 1.4 - 1.8) reaches the plane of the back face beyond the edge
    tilt = 70.0
    shortest_shift = thickness * math.tan(math.asin(math.sin(math.radians(tilt)) / 1.8))
    entry = (edge - shortest_shift + clear) / 2.0
    assert edge - shortest_shift < entry < clear
    half = math.radians(tilt) / 2.0
    tilted = list(quat_multiply(along_x, (math.cos(half), 0.0, math.sin(half), 0.0)))
    edge_exit = with_window([0.0, 0.0, 0.0], tilted, [-100.0, -entry * math.cos(math.radians(tilt)), 0.0], 0.1, 633.0)
    edge_exit["components"] = [c for c in edge_exit["components"] if c["instance_id"] != "L1"]
    edge_exit["observations"][0]["pose"]["position_mm"] = [100.0, 0.0, 0.0]
    edge_exit["observations"][0]["aperture"]["radius_mm"] = 200.0
    # no target: the Local Validation record carries target values at full precision, and this beam's power (after a
    # Gaussian truncation on the 70 deg face) differs in its last bits between the runtimes
    edge_exit["targets"] = []
    # R2: the shortest plano-convex lens (curved face = its first surface) turned flat side first, a beam at ~f/5.5
    lens = min((p for p in ordered if p.form.value == "PLANO_CONVEX_LENS" and covers(p, 780) and p.simulation_models[0].surfaces[0].radius_mm),
               key=lambda p: (Decimal(p.specifications.focal_length_mm.value), p.optical_sku))
    focal = float(Decimal(lens.specifications.focal_length_mm.value))
    aberrated = json.loads(json.dumps(reference))
    aberrated["components"][0]["source_model"]["waist_radius_mm"] = min(0.09 * focal, 0.3 * float(lens.simulation_models[0].clear_radius_mm))
    aberrated["components"][1]["product"]["optical_sku"] = lens.optical_sku
    aberrated["components"][1]["nominal_pose"]["quaternion_wxyz"] = against_x
    aberrated["observations"][0]["pose"]["position_mm"] = [3.0 * focal, 0.0, 0.0]
    aberrated["targets"] = [{"target_id": "spot", "metric": "beam_radius_mm", "observation_id": "focal_plane", "maximum": 10.0, "unit": "mm"},
                            {"target_id": "power", "metric": "power_W", "observation_id": "focal_plane", "minimum": 0.0, "unit": "W"}]
    return {"partial_interception": partial, "aberration_dominated": aberrated, "edge_exit": edge_exit,
            "double_ghost": _double_ghost(reference, window, catalog)}


def _double_ghost(reference: dict, window, catalog) -> dict:
    """Round-3 review F1: the uncoated window turned 10 deg in the beam, its ghost onto a second one turned 15 deg to it, and
    that window's reflection of the ghost onto an absorbing detector no traced beam reaches: the probe traces follow the
    ghost of the ghost there (UNKNOWN with its power), identically in the browser runtime."""

    def unit(v):
        return [c / math.sqrt(sum(x * x for x in v)) for c in v]

    def reflect(d, n):
        k = 2.0 * sum(a * b for a, b in zip(d, n))
        return [a - k * b for a, b in zip(d, n)]

    def facing(n):  # the quaternion that turns the frame's +z onto n
        w, x, y, z = 1.0 + n[2], -n[1], n[0], 0.0
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        return [w / norm, x / norm, y / norm, z / norm]

    t1, t2 = math.radians(10.0), math.radians(15.0)
    n1 = [math.cos(t1), math.sin(t1), 0.0]
    r1 = reflect([1.0, 0.0, 0.0], n1)
    w2 = [50.0 + 60.0 * r1[0], 60.0 * r1[1], 0.0]
    n2 = unit([-math.cos(t2) * r1[0] + math.sin(t2) * r1[1], -math.sin(t2) * r1[0] - math.cos(t2) * r1[1], 0.0])
    r2 = reflect(r1, n2)
    document = json.loads(json.dumps(reference))
    source = document["components"][0]
    source["nominal_pose"]["position_mm"] = [0.0, 0.0, 0.0]
    source["source_model"].update(waist_radius_mm=0.5, wavelength_nm=633.0)
    product = {"optical_sku": window.optical_sku, "catalog_version": catalog.catalog_version}
    document["components"] = [source] + [
        {"instance_id": iid, "optical_role": "window", "product": product, "nominal_pose": {"position_mm": pos, "quaternion_wxyz": facing(n)}}
        for iid, pos, n in (("W1", [50.0, 0.0, 0.0], n1), ("W2", w2, n2))]
    plane = document["observations"][0]
    plane["pose"] = {"position_mm": [w2[0] + 60.0 * r2[0], w2[1] + 60.0 * r2[1], 0.0], "quaternion_wxyz": facing(r2)}
    plane["aperture"] = {"shape": "circle", "radius_mm": 8.0, "center_offset_mm": [0.0, 0.0], "obstruction_radius_mm": 0.0}
    plane["absorbing"] = True
    document["observations"] = [plane]
    document["targets"] = [{"target_id": "ghost_max", "metric": "power_W", "observation_id": plane["observation_id"],
                            "maximum": 1e-9, "unit": "W"}]
    return document


def network_cases() -> dict[str, dict]:
    """Engine requests exported as linear optical networks: the frozen mzi_window / hwp_pbs / interference_405_500
    requests, a Michelson and a two-source Hong-Ou-Mandel splitter, a Michelson with both ports and sub-micron positions
    (per-beam reference phases), and refusals (mixed wavelengths, gain, a PBS at normal incidence)."""
    from os_optics import ValidationRequest
    from os_optics import builders as B
    from os_optics.contracts import ProjectSnapshot

    from test_network_accuracy_and_scope import _hom_with_phase, _michelson_both_ports
    from test_network_export import _hom, _michelson

    cases = {name: json.loads((REQ_DIR / f"{name}.json").read_text(encoding="utf-8")) for name in ("mzi_window", "hwp_pbs", "interference_405_500",
                                                                                                    "multi_laser")}
    michelson = _michelson()
    cases["michelson"] = ValidationRequest(project=michelson.model_copy(update={"observations": michelson.observations[:1]})).model_dump(mode="json")
    cases["hom"] = ValidationRequest(project=_hom(45.0)).model_dump(mode="json")
    cases["michelson_both_ports"] = ValidationRequest(project=_michelson_both_ports(80.0003, 80.2501234, 50.0004, 70.0007)).model_dump(mode="json")
    cases["gain"] = ValidationRequest(project=_hom_with_phase(0.0)).model_dump(mode="json")
    cases["pbs_normal_incidence"] = ValidationRequest(project=ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3),
                    B.pbs("P", position=(50, 0, 20), normal=(-1, 0, 0))),
        observations=(B.observation_plane("fwd", position=(100, 0, 20), normal=(1, 0, 0)),))).model_dump(mode="json")
    return cases


def _field_calls() -> dict[str, list]:
    """The pupil field and detector stages in the browser runtime: the worked chain's spot, a ramp on pixels, a binary grating, a spot array, independent tones, a mask read from
    its file (its bytes as a JSON list of integers), the refusal above the scalar limit and a detector with a parameter left out. A transform is not bitwise the same across runtimes:
    these compare within 1e-9."""
    import io

    import numpy as np

    from test_modulator_chain import CAMERA, slm

    def grid(pixels: int, pitch: float, **more: object) -> dict:
        return {"wavelength_nm": 852.0, "focal_length_mm": 200.0, "sampling": {"pixel_count": [pixels, pixels], "pitch_mm": [pitch, pitch], **more}}

    spot = slm(1e-9, half=(0.1, 0.1), fft=2048)
    ramp = {**grid(128, 0.0125, fill_factor=1.0, fft_size=[512, 512]), "illumination": {"kind": "UNIFORM", "power_w": 1e-3},
            "mask": {"kind": "BLAZED_RAMP", "sin_theta": [8.52e-3, 0.0]}, "output": {"half_extent_mm": [2.0, 0.1], "include_field": True}}
    grating = {**grid(128, 0.0125, fill_factor=0.9, fft_size=[512, 512]), "illumination": {"kind": "UNIFORM", "power_w": 1e-3},
               "mask": {"kind": "BINARY_GRATING", "period_mm": 0.2, "axis": "X"}, "output": {"half_extent_mm": [2.0, 0.1]}}
    array = {**grid(64, 0.1, fft_size=[256, 256]), "aperture": {"shape": "CIRCLE", "diameter_mm": 6.0}, "illumination": {"kind": "UNIFORM", "power_w": 1e-3},
             "mask": {"kind": "SPOT_ARRAY", "spots": [{"u_mm": u, "v_mm": v} for u in (-0.3, 0.3) for v in (-0.3, 0.3)], "iterations": 10}}
    tones = {**grid(128, 0.05, fft_size=[512, 512]), "illumination": {"kind": "BEAMS", "composition": "INCOHERENT", "beams": [
        {"label": f"tone {i}", "power_w": 1e-3, "radius_mm": [1.2, 1.2], "direction_cosines": [i * 1e-3, 0.0]} for i in range(3)]}}
    above = {**grid(256, 0.2, fft_size=[512, 512]), "focal_length_mm": 100.0, "aperture": {"shape": "CIRCLE", "diameter_mm": 50.0}, "illumination": {"kind": "UNIFORM", "power_w": 1e-3},
             "output": {"half_extent_mm": [0.1, 0.1]}}
    buffer = io.BytesIO()
    np.savez(buffer, phase_rad=np.full((8, 8), 0.7), amplitude=np.full((8, 8), 0.5))
    data = buffer.getvalue()
    from_file = {**grid(8, 0.1, fft_size=[32, 32]), "illumination": {"kind": "UNIFORM", "power_w": 1e-3}, "output": {"include_field": True},
                 "mask": {"kind": "FILE", "file_name": "mask.npz", "sha256": hashlib.sha256(data).hexdigest(), "source": {"kind": "CUSTOMER_STATEMENT", "reference": "test mask"}}}
    field = json.loads(json.dumps(json_api_field(spot)))
    from local_helpers import plane, run
    from os_optics.results import load_run
    from test_acousto_optic import aod, bench, table

    handoff = load_run(run(bench(aod(efficiency=table()), [plane("PUPIL", (5.0, 0.0, 0.0), radius_mm=40.0)]))).handoff("PUPIL")
    tones_only = sorted(handoff["beams"], key=lambda beam: beam["chief_ray"]["uv_mm"][0])[1:4]
    deflector = {**grid(256, 0.01, fft_size=[1024, 1024]), "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT", "path_ids": [beam["path_id"] for beam in tones_only]},
                 "output": {"center_uv_mm": [7.0, 0.0], "half_extent_mm": [2.0, 0.5]}}
    return {"field_deflector_tones": ["pupil_field", [deflector, None, json.loads(json.dumps(handoff))]], "field_from_file": ["pupil_field", [from_file, {"mask.npz": list(data)}]], "field_spot": ["pupil_field", [spot]], "field_ramp": ["pupil_field", [ramp]], "field_grating": ["pupil_field", [grating]], "field_array": ["pupil_field", [array]],
            "field_tones": ["pupil_field", [tones]], "field_above_limit": ["pupil_field", [above]],
            "field_refused": ["pupil_field", [{"wavelength_nm": 852.0}]],
            "field_detector": ["detector_response", [field, {**CAMERA, "pixel_count": [9, 9]}]],
            "field_detector_partial": ["detector_response", [field, {"pixel_pitch_mm": [0.0065, 0.0065], "fill_factor": 1.0, "pixel_count": [9, 9]}]],
            "field_detector_refused": ["detector_response", [field, {"bit_depth": 0}]],
            **_focal_calls()}


def _focal_calls() -> dict[str, list]:
    """The vector focal field in the browser runtime: a rotationally symmetric pupil (Bessel reduction), a tilted one (grid transform) with a window and an
    invalid request. Quadratures and transforms compare within 1e-9."""

    def focus(**more: object) -> dict:
        body = {"wavelength_nm": 852.0, "objective": {"model": "IDEAL_APLANATIC", "numerical_aperture": 0.65, "focal_length_mm": 20.0},
                "illumination": {"kind": "GAUSSIAN", "power_w": 1e-3, "radius_mm": 10.0}, "polarization": {"kind": "LINEAR", "angle_deg": 0.0},
                "output": {"samples": 9}}
        body.update(more)
        return body

    tilted = focus(illumination={"kind": "GAUSSIAN", "power_w": 1e-3, "radius_mm": 10.0, "direction_cosines": [0.002, 0.0]},
                   objective={"model": "IDEAL_APLANATIC", "numerical_aperture": 0.5, "focal_length_mm": 20.0, "window": {"thickness_mm": 0.5, "refractive_index": 1.45}},
                   sampling={"theta_nodes": 24, "phi_nodes": 24, "grid_cells": 48}, polarization={"kind": "CIRCULAR", "handedness": "RIGHT"})
    return {"field_focal_symmetric": ["focal_field", [focus()]], "field_focal_tilted_window": ["focal_field", [tilted]],
            "field_focal_refused": ["focal_field", [{"wavelength_nm": 852.0}]]}


def json_api_field(request: dict) -> dict:
    from os_optics.local import json_api

    return json_api.pupil_field(request)


#: Calls whose replies hold transforms (``_field_calls``): compared within 1e-9, not bit for bit.
FIELD_CALLS_RELATIVE_TOLERANCE = 1e-9


def _close(browser: object, native: object, where: str) -> None:
    """The same JSON document up to the transforms' rounding: numbers within ``FIELD_CALLS_RELATIVE_TOLERANCE`` of the largest of their list or their object (a residual of 1e-16
    or a clipped power of 1e-20 beside a power of 1e-3 is rounding, whatever its own digits)."""
    if isinstance(native, dict):
        assert isinstance(browser, dict) and set(browser) == set(native), where
        scale = max((abs(v) for v in native.values() if isinstance(v, float)), default=0.0)
        for key in native:
            if key == "parseval_relative_residual":  # a pure number beside powers: rounding in both runtimes
                assert max(browser[key], native[key]) < FIELD_CALLS_RELATIVE_TOLERANCE, f"{where}.{key}"
            elif isinstance(native[key], float):
                assert isinstance(browser[key], (int, float)) and abs(browser[key] - native[key]) <= FIELD_CALLS_RELATIVE_TOLERANCE * scale, f"{where}.{key}"
            else:
                _close(browser[key], native[key], f"{where}.{key}")
    elif isinstance(native, list):
        assert isinstance(browser, list) and len(browser) == len(native), where
        scale = max((abs(v) for row in native for v in (row if isinstance(row, list) else [row]) if isinstance(v, float)), default=0.0)
        for index, (b, n) in enumerate(zip(browser, native)):
            if isinstance(n, float) and isinstance(b, (int, float)):
                assert abs(b - n) <= FIELD_CALLS_RELATIVE_TOLERANCE * max(abs(n), scale), f"{where}[{index}]"
            elif isinstance(n, list) and all(isinstance(v, float) for v in n):
                assert len(b) == len(n), where
                assert all(abs(x - y) <= FIELD_CALLS_RELATIVE_TOLERANCE * max(abs(y), scale) for x, y in zip(b, n)), f"{where}[{index}]"
            else:
                _close(b, n, f"{where}[{index}]")
    elif isinstance(native, float):
        assert isinstance(browser, (int, float)) and abs(browser - native) <= FIELD_CALLS_RELATIVE_TOLERANCE * abs(native), where
    else:
        assert browser == native, where


def json_api_calls() -> dict[str, list]:
    """json_api entry points other than Project runs that a browser host / Agent calls, as
    ``{name: [function, args]}``: both prescription fixtures (mm and inch) and two refusals, every native component class
    built with real parameters plus two more designs and a refusal, ``native_components``, the pupil field and detector stages (``_field_calls``), exact sections of an imported
    unit (XZ with its SVG, YZ) and of a native unit, a catalog cube as a declared model, and ``inspect_optical_solids`` /
    ``check_project`` of placed declared lenses: two coincident (INVALID) and tilted near neighbours."""
    from local_helpers import project, plane, source
    from declared_helpers import declared_component, nbk7, sequence, surface, with_models
    from test_native_library import ALL_CLASSES

    fixtures = Path(__file__).resolve().parent / "fixtures" / "prescriptions"
    cooke = (fixtures / "cooke_triplet.csv").read_text(encoding="utf-8")
    inch = (fixtures / "doublet_singlet_inch.csv").read_text(encoding="utf-8")
    calls: dict[str, list] = {
        "import_cooke": ["import_prescription", [cooke, {"model_id": "cooke"}]],
        "import_inch": ["import_prescription", [inch, {"model_id": "ds"}]],
        "import_refused_label": ["import_prescription", ["surface,radius,thickness,material,semi_diameter\nFl\u00e4che1,50,5,N-BK7,10\n"
                                                         "2,-50,5,,10\n", {}]],
        "import_refused_units": ["import_prescription", [inch, {"units": "mm"}]],
        "native_components": ["native_components", []],
        **_field_calls(),
    }
    native = json.loads(json.dumps(ALL_CLASSES))  # the native library's own cases, with a coating the seed catalog publishes
    native["plate.window"]["ar_coating"] = "AR_400-1100R0.01T0.99_AOI0-10"
    for class_id, parameters in native.items():
        calls[f"native_{class_id}"] = ["build_native_model", [class_id, parameters]]
    calls["native_achromat_shape_-2"] = ["build_native_model", ["lens.achromat", {**native["lens.achromat"], "crown_shape_factor": "-2"}]]
    calls["native_singlet_1064"] = ["build_native_model", ["lens.singlet", {"focal_length_mm": "25", "material": "N-SF11",
                                                                            "design_wavelength_nm": "1064", "diameter_mm": "12.7"}]]
    calls["native_refused"] = ["build_native_model", ["lens.achromat", {**native["lens.achromat"], "focal_length_mm": "40"}]]
    src = source(position=(-100.0, 0.0, 0.0), wavelength_nm=632.8, waist_mm=0.5)
    screen = [plane("scr", (100.0, 0.0, 0.0), radius_mm=100.0)]
    from os_optics.local import json_api

    assert json_api.install_catalog(REAL_CATALOG.read_bytes())["ok"] is True
    imported = json_api.import_prescription(cooke, {"model_id": "cooke", "into": project([src], screen), "instance_id": "OBJ"})["project"]
    calls["native_into_project"] = ["build_native_model", ["aperture.circular", native["aperture.circular"],
                                                           {"into": project([src], screen), "instance_id": "STOP", "position_mm": [50, 0, 0]}]]
    calls["section_import_xz_svg"] = ["optical_section", [imported, "OBJ", "XZ", True]]
    calls["section_import_yz"] = ["optical_section", [imported, "OBJ", "YZ", False]]
    relay = json_api.build_native_model("compound.relay_4f", native["compound.relay_4f"])
    relay_project = with_models(project([src, {**relay["component_template"], "instance_id": "RELAY",
                                               "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]}}],
                                        screen), [relay["optical_model"], *relay["member_models"]], relay["model_materials"], relay["model_coatings"])
    calls["section_native_relay"] = ["optical_section", [relay_project, "RELAY", "XZ", True]]
    # the layout of a design without a run (elements, planes, nominal beam axes and its GLB bytes): REVIEW2 D-4
    calls["layout_folded"] = ["project_layout", [json_api.example_project_document(folded=True)["project"]]]
    calls["layout_relay"] = ["project_layout", [relay_project]]
    calls["catalog_model_cube"] = ["catalog_model", ["OS-OPT-BS-CUBE-A10-R10T90-700-1100-NBK7-CA8"]]
    meniscus = {"model_id": "men", "model": sequence([surface(1, "0", radius="20", clear="9"), surface(2, "3", radius="25", clear="9")], ["N-BK7"])}
    biconvex = {"model_id": "bic", "model": sequence([surface(1, "0", radius="30", clear="9"), surface(2, "5", radius="-30", clear="9")], ["N-BK7"])}
    along_x = (0.5, 0.5, 0.5, 0.5)
    coincident = with_models(project([src, declared_component("M", "lens", "bic", (0, 0, 0), along_x),
                                      declared_component("B", "lens", "bic", (1.0, 0.0, 0.0), along_x)], screen), [biconvex], [nbk7()])
    calls["solids_invalid"] = ["inspect_optical_solids", [coincident]]
    calls["check_solids_invalid"] = ["check_project", [coincident]]
    for index, (dx, degrees) in enumerate(((5.5, 20.0), (6.5, 25.0), (4.0, 40.0))):
        half = math.radians(degrees) / 2  # the lens frame's turn, then a tilt about the lens's own X axis
        w1, x1, y1, z1 = along_x
        w2, x2, y2, z2 = (math.cos(half), math.sin(half), 0.0, 0.0)
        quaternion = tuple(float(format(v, ".15g")) for v in (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                                               w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2))
        tilted = with_models(project([src, declared_component("M", "lens", "men", (0, 0, 0), along_x),
                                      declared_component("B", "lens", "bic", (dx, 0.3, 0.0), quaternion)], screen), [meniscus, biconvex], [nbk7()])
        calls[f"solids_tilted_{index}"] = ["inspect_optical_solids", [tilted]]
        calls[f"check_tilted_{index}"] = ["check_project", [tilted]]
    # a convex mirror's substrate holds its face (default) or is refused (a given thickness below the face's depth)
    convex = {"diameter_mm": "25.4", "reflectance": "1", "valid_band_nm": ["400", "1000"]}
    calls["native_convex_mirror"] = ["build_native_model", ["mirror.curved", {**convex, "focal_length_mm": "-6.5"}]]
    calls["native_convex_mirror_refused"] = ["build_native_model", ["mirror.curved", {**convex, "focal_length_mm": "-100",
                                                                                      "substrate_thickness_mm": "0.1"}]]
    # the network export checks the published models on the light it exports only: an unselected 800 nm source beside a
    # 600-700 nm retarder does not refuse the 633 nm export; selected, it does
    from test_network_project import retarder_two_source_project

    two_sources = retarder_two_source_project(("GOOD", 633, 0), ("OTHER", 800, 3))
    calls["network_selected_in_band"] = ["linear_optical_network", [two_sources, {"source_ids": ["GOOD"], "wavelength_nm": 633}]]
    calls["network_selected_out_of_band"] = ["linear_optical_network", [two_sources, {"source_ids": ["OTHER"], "wavelength_nm": 800}]]
    # the interop capability catalog (whole, one entry, an unknown id) and the paraxial analysis of declared lenses (BB1-02's
    # afocal pair, a collimation with a source, a refusal)
    from test_paraxial_analysis import _bb102_project

    calls["interop_catalog"] = ["interop_catalog", []]
    calls["interop_entry"] = ["interop_catalog", ["metasurface-virtual-element"]]
    calls["interop_unknown"] = ["interop_catalog", ["nope"]]
    bb102 = _bb102_project(200.0)
    calls["paraxial_afocal"] = ["paraxial_analysis", [bb102, {"components": ["L1", "L2"], "wavelength_nm": 780, "solve": "afocal"}]]
    calls["paraxial_collimate"] = ["paraxial_analysis", [bb102, {"components": ["L1"], "wavelength_nm": "780", "solve": "collimate",
                                                                 "source_id": "SRC"}]]
    calls["paraxial_refused"] = ["paraxial_analysis", [bb102, {"components": ["L9"], "wavelength_nm": 780}]]
    sourceless = {**bb102, "components": [c for c in bb102["components"] if c["optical_role"] != "source"], "targets": []}
    calls["paraxial_afocal_no_source"] = ["paraxial_analysis", [sourceless, {"components": ["L1", "L2"], "wavelength_nm": 780,
                                                                             "solve": "afocal"}]]
    # the cavity analysis: a stable two-mirror cavity with a stated input beam, an unstable one, a lens between plane mirrors, a refusal
    from declared_helpers import plano_convex_model
    from test_cavity_analysis import _lens_cavity, two_mirror_cavity

    calls["cavity_stable"] = ["cavity_analysis", [two_mirror_cavity("200", "300", 120.0), {"components": ["M1", "M2"], "wavelength_nm": 1064,
                                                                                           "input_beam": {"waist_radius_mm": 0.2, "waist_after_reference_mm": 40}}]]
    calls["cavity_unstable"] = ["cavity_analysis", [two_mirror_cavity("50", "50", 120.0), {"components": ["M1", "M2"], "wavelength_nm": 1064}]]
    calls["cavity_lens"] = ["cavity_analysis", [_lens_cavity(plano_convex_model("pcx"), 40.0, 130.0), {"components": ["M1", "L", "M2"], "wavelength_nm": 1064}]]
    calls["cavity_refused"] = ["cavity_analysis", [two_mirror_cavity("200", "300", 120.0), {"components": ["M1", "M1"], "wavelength_nm": 1064}]]
    # light injected through the first mirror: a mismatched beam split into Hermite-Gauss modes, bounded over every detuning (none stated), a scan
    calls["cavity_injected"] = ["cavity_analysis", [two_mirror_cavity("200", "300", 120.0), {
        "components": ["M1", "M2"], "wavelength_nm": 1064, "reflectances": {"M1": 0.98, "M2": 0.995}, "transmittances": {"M1": 0.019, "M2": 0.004},
        "input_beam": {"waist_radius_mm": 0.2, "waist_after_reference_mm": 40},
        "injection": {"power_w": 1.5, "beam": "INPUT_BEAM", "mode_order_max": 8, "scan": {"from_Hz": -3e8, "to_Hz": 3e8, "points": 31}},
        "targets": [{"target_id": "t", "quantity_id": "transmission", "minimum": 0.01}]}]]
    # a Project source traced to the input mirror, its own beam injected at a length offset from the resonance
    from test_cavity_injection import _fed as fed_cavity

    calls["cavity_source"] = ["cavity_analysis", [fed_cavity(), {"components": ["M1", "M2"], "wavelength_nm": 1064.0,
                                                                 "injection": {"source_id": "S", "beam": "SOURCE", "length_offset_nm": 50.0}}]]
    calls.update(composition_calls())
    calls.update(project_update_calls())
    calls.update(evaluation_calls())
    return calls


#: The selective-evaluation calls that are refusals (``evaluation_calls``).
EVALUATION_REFUSED = {"evaluate_excluded_refused", "evaluate_unknown_target_refused"}


def evaluation_calls() -> dict[str, list]:
    """Selective evaluations of a Project (``evaluate_project``) as a JSON host makes them: the default readings, the
    requirements of the example Project, one selected target of two, a source outside a declared lens's published band
    (its target undecided, the model-scope finding), the scene of a view alone with every pupil ray, and two refusals (an
    exclusion a requested capability needs, an unknown target). Native and browser replies are compared byte for byte, the
    readings, targets, findings, identity hashes and scene included."""
    from os_optics.local import json_api

    focus = json_api.example_project_document()["project"]
    lens = json.loads((Path(__file__).resolve().parent / "fixtures" / "ideal_lens_band.project.json").read_text(encoding="utf-8"))
    lens["components"][0]["source_model"]["wavelength_nm"] = 1200
    lens["targets"] = [{"target_id": "spot", "metric": "spot_rms_mm", "observation_id": "early", "maximum": 1.0, "unit": "mm"}]
    return {
        "evaluate_readings": ["evaluate_project", [focus]],
        "evaluate_requirements": ["evaluate_project", [focus, ["readings", "requirements"]]],
        "evaluate_one_target": ["evaluate_project", [focus, None, None, ["spot_radius"]]],
        "evaluate_out_of_band": ["evaluate_project", [lens, ["readings", "requirements"]]],
        "evaluate_view": ["evaluate_project", [focus, ["view"], None, None, {"pupil_ray_selection": "ALL"}]],
        "evaluate_excluded_refused": ["evaluate_project", [focus, ["record"], ["untraced_light"]]],
        "evaluate_unknown_target_refused": ["evaluate_project", [focus, None, None, ["nope"]]],
    }


#: Calls on the prepared evaluation handles of one runtime, in order (run by the harness in Pyodide and by
#: ``test_prepared_evaluation_handles_in_pyodide_are_identical_to_native`` natively): a bound of two, two handles, an
#: evaluation, a candidate, a view, a third handle evicting the least recently used, a release and the refusals that
#: follow. Each handle id (random) is replaced by its name, so the two runtimes' results compare as JSON text.
PREPARED_SEQUENCE = """
import json


def sequence(json_api, documents):
    focus, moved = documents["focus"], documents["moved"]
    out = {"bound": json_api.evaluation_handles(2)}
    out["prepare_a"] = json_api.prepare_evaluation(focus)
    out["prepare_b"] = json_api.prepare_evaluation(focus)
    if not (out["prepare_a"]["ok"] and out["prepare_b"]["ok"]):
        return out  # (the comparison names the refusal)
    a, b = out["prepare_a"]["handle"], out["prepare_b"]["handle"]
    out["requirements"] = json_api.evaluate_prepared(a, ["readings", "requirements"])
    out["fresh_requirements"] = json_api.evaluate_project(focus, ["readings", "requirements"])
    out["again"] = json_api.evaluate_prepared(a, ["readings", "requirements"])
    out["candidate"] = json_api.evaluate_prepared(a, ["readings"], None, None, None, moved)
    out["fresh_candidate"] = json_api.evaluate_project(moved, ["readings"])
    out["view"] = json_api.evaluate_prepared(a, ["view"], None, None, {"pupil_ray_selection": "ALL"})
    out["fresh_view"] = json_api.evaluate_project(moved, ["view"], None, None, {"pupil_ray_selection": "ALL"})
    out["prepare_c"] = json_api.prepare_evaluation(focus)
    out["evicted"] = json_api.evaluate_prepared(b)
    out["cancel_idle"] = json_api.cancel_evaluation(a)
    out["release"] = json_api.release_evaluation(a)
    out["released"] = json_api.evaluate_prepared(a)
    out["listing"] = json_api.evaluation_handles()
    json_api.release_evaluation(out["prepare_c"]["handle"])
    json_api.evaluation_handles(4)
    text = json.dumps(out, sort_keys=True)
    for name, handle in (("<A>", a), ("<B>", b), ("<C>", out["prepare_c"]["handle"])):
        text = text.replace(handle, name)
    return json.loads(text)
"""


def prepared_documents() -> dict[str, dict]:
    """The Projects of ``PREPARED_SEQUENCE``: the example Project (on the catalog the harness installs) and the same with its
    lens moved 0.4 mm along the beam."""
    from os_optics.local import example_project

    focus = example_project(REAL_CATALOG).payload()
    moved = json.loads(json.dumps(focus))
    next(c for c in moved["components"] if c["optical_role"] == "lens")["nominal_pose"]["position_mm"][0] += 0.4
    return {"focus": focus, "moved": moved}


#: The Project update / freshness calls that are refusals (``project_update_calls``).
PROJECT_UPDATE_REFUSED = {"update_project_op_refused", "update_project_rejected", "update_project_changed", "update_project_transform_refused"}


def project_update_calls() -> dict[str, list]:
    """Engine 0.9.0's calls the Designer makes for its Project edits (``update_project``: field operations, a rigid
    transform, a retried add and every refusal), the advisory warnings of ``check_project`` and ``run_freshness`` of a
    Local Validation record; on a catalog-free Project of a declared lens."""
    from declared_helpers import declared_component, nbk7, plano_convex_model, with_models
    from local_helpers import plane, project, source
    from os_optics.local import json_api

    doc = with_models(project([source(), declared_component("L1", "lens", "pcx")], [plane("out", (100.0, 0.0, 0.0))],
                              [{"target_id": "T", "metric": "beam_radius_mm", "observation_id": "out", "maximum": 5.0, "unit": "mm"}]),
                      [plano_convex_model("pcx")], [nbk7()])
    current = json_api.check_project(doc)["project_hash"]
    fields = [{"op": "set_source_model", "instance_id": "SRC", "changes": {"wavelength_nm": 1500.0}},
              {"op": "set_target", "target_id": "T", "changes": {"unit": "um"}},
              {"op": "set_retained_dof", "instance_id": "L1", "axis": "Rz", "control": "MANUAL"}]
    transform = {"op": "transform", "selector": {"ids": ["L1"]}, "rotation": {"axis": [0, 1, 0], "angle_deg": 37}, "translation_mm": [0, 0, 2.5],
                 "frame": {"component": "L1"}, "pivot": "selection_centroid"}
    extra = {**doc["observations"][0], "observation_id": "extra"}
    with_extra = json_api.update_project(doc, [{"op": "add_observation", "observation": extra}])["project"]
    record = json_api.run_project(doc)["local_validation"]
    moved = json_api.update_project(doc, [transform])["project"]
    return {
        "update_project_fields": ["update_project", [doc, fields, {"expected_project_hash": current}]],
        "update_project_transform": ["update_project", [doc, [transform, {"op": "set_design_intent", "design_intent": "OPTICAL_SYSTEM"}]]],
        "update_project_retry_add": ["update_project", [with_extra, [{"op": "add_observation", "observation": extra}]]],
        "update_project_op_refused": ["update_project", [doc, [fields[0], {"op": "remove_target", "target_id": "nope"}]]],
        "update_project_rejected": ["update_project", [doc, [{"op": "set_source_model", "instance_id": "SRC", "changes": {"waist_radius_mm": -1}}]]],
        "update_project_changed": ["update_project", [doc, fields, {"expected_project_hash": "0" * 64}]],
        "update_project_transform_refused": ["update_project", [doc, [{**transform, "selector": {"ids": ["NOPE"]}}]]],
        "check_project_warnings": ["check_project", [json_api.update_project(doc, fields)["project"]]],
        "run_freshness_current": ["run_freshness", [doc, record]],
        "run_freshness_historical": ["run_freshness", [moved, record]],
    }


#: The composition calls that are refusals (``json_api_calls``).
COMPOSITION_REFUSED = {"ws_import_stale_base", "ws_replace_deleted_refused", "ws_replace_enclosing_after_nested_deletion_refused",
                       "ws_replace_nested_with_lacked_members_added_refused", "ws_replace_nested_reserved_own_member_refused",
                       "ws_replace_defining_a_used_definition_differently_refused"}


def composition_calls() -> dict[str, list]:
    """The composition calls with real modules: init with a module descriptor, imports at an identity, a rotated and
    an arbitrary placement, coordinated rotations (a quarter turn, 37 degrees about a tilted axis in a group frame with a
    pivot, a quaternion in a component frame, a 270 degree turn without op_id), selection, groups, a batch, replacements
    (also of a composed module's instance), export / re-import of a composed workspace, status, history, a recorded
    validation and two refusals. Every reply carries the resulting workspace, so
    byte-identical replies mean byte-identical Projects and workspace digests in both runtimes."""

    import math as _math

    from composition_helpers import FOLD_DESCRIPTOR, LENS_DESCRIPTOR, fold_module, lens_module, project_of, screen, system_with_laser

    from os_optics.local import json_api

    def ok(reply):
        assert reply["ok"], reply
        return reply

    fold_ws = ok(json_api.workspace_init(fold_module(), {"module": FOLD_DESCRIPTOR}))["workspace"]
    fold = ok(json_api.workspace_export_module(fold_ws, FOLD_DESCRIPTOR))["module"]
    lens50 = ok(json_api.workspace_export_module(ok(json_api.workspace_init(lens_module("50"), {"module": LENS_DESCRIPTOR}))["workspace"],
                                                 LENS_DESCRIPTOR))["module"]
    lens60 = ok(json_api.workspace_export_module(ok(json_api.workspace_init(lens_module("60"), {"module": LENS_DESCRIPTOR}))["workspace"],
                                                 LENS_DESCRIPTOR))["module"]
    parent = system_with_laser()
    rebind = {"rebind": {"SRC": "LASER"}}
    half = _math.sqrt(0.5)
    calls: dict[str, list] = {"ws_init": ["workspace_init", [fold_module(), {"module": FOLD_DESCRIPTOR}]],
                              "ws_export": ["workspace_export_module", [fold_ws, FOLD_DESCRIPTOR]]}
    first = {"instance_key": "fa", "fixtures": rebind, "op_id": "imp-fa"}
    calls["ws_import"] = ["workspace_import", [parent, fold, first]]
    ws = ok(json_api.workspace_import(parent, fold, first))["workspace"]
    second = {"instance_key": "fb", "fixtures": rebind, "op_id": "imp-fb",
              "placement": {"position_mm": [0.0, 200.0, 0.0], "quaternion_wxyz": [half, 0.0, 0.0, half]}}
    calls["ws_import_quarter_turn"] = ["workspace_import", [ws, fold, second]]
    ws = ok(json_api.workspace_import(ws, fold, second))["workspace"]
    tilted = {"instance_key": "foc", "fixtures": rebind, "op_id": "imp-foc",
              "placement": {"position_mm": [0.3, 120.7, -0.2], "quaternion_wxyz": [0.7064, 0.0123, -0.0217, 0.7072]}}
    tilted["placement"]["quaternion_wxyz"] = [c / _math.sqrt(sum(v * v for v in tilted["placement"]["quaternion_wxyz"]))
                                              for c in tilted["placement"]["quaternion_wxyz"]]
    calls["ws_import_arbitrary_placement"] = ["workspace_import", [ws, lens50, tilted]]
    ws = ok(json_api.workspace_import(ws, lens50, tilted))["workspace"]
    calls["ws_import_retry"] = ["workspace_import", [ws, lens50, tilted]]
    calls["ws_import_stale_base"] = ["workspace_import", [ws, lens50, {**tilted, "op_id": "imp-again", "instance_key": "fd",
                                                                       "expected_base": "0" * 64}]]
    calls["ws_select"] = ["workspace_select", [ws, {"instances": ["fa", "fb"], "groups": ["fb"], "ids": ["LASER", "foc_L1"]}]]
    group = {"group_id": "g", "members": {"instances": ["fb"]}, "frame": {"instance": "fb"}, "op_id": "grp"}
    calls["ws_group_create"] = ["workspace_group_create", [ws, group]]
    ws = ok(json_api.workspace_group_create(ws, group))["workspace"]
    rotate = {"selector": {"groups": ["g"]}, "rotation": {"axis": [0.2, 0.3, 1.0], "angle_deg": 37.0}, "translation_mm": [1.0, 2.0, -0.5],
              "frame": {"group": "g"}, "pivot": {"point_mm": [3.0, -4.0, 1.5]}, "op_id": "rot37"}
    calls["ws_transform_37deg_group_frame"] = ["workspace_transform", [ws, rotate]]
    calls["ws_transform_quarter_turn_centroid"] = ["workspace_transform", [ws, {"selector": {"instances": ["fb"]}, "op_id": "q",
                                                                                "rotation": {"axis": [0, 0, 1], "angle_deg": -90},
                                                                                "pivot": "selection_centroid"}]]
    calls["ws_transform_quaternion_component_frame"] = ["workspace_transform", [ws, {
        "selector": {"ids": ["foc_L1", "foc_FOCUS"]}, "rotation": {"quaternion_wxyz": [0.9998477, 0.0174524, 0.0, 0.0]},
        "frame": {"component": "foc_L1"}, "pivot": "frame_origin", "op_id": "tip"}]]
    calls["ws_transform_overlap"] = ["workspace_transform", [ws, {"selector": {"instances": ["fb"]}, "translation_mm": [0.0, -200.0, 0.0]}]]
    moved = ok(json_api.workspace_transform(ws, rotate))["workspace"]
    calls["ws_batch"] = ["workspace_batch", [moved, {"op_id": "b", "operations": [
        {"kind": "group_add", "request": {"group_id": "g", "members": {"ids": ["foc_L1"]}}},
        {"kind": "transform", "request": {"selector": {"groups": ["g"]}, "rotation": {"axis": [1, 1, 1], "angle_deg": 0.25}}},
        {"kind": "group_dissolve", "request": {"group_id": "fa"}}]}]]
    calls["ws_replace"] = ["workspace_replace_instance", [moved, lens60, {"instance_key": "foc", "fixtures": rebind, "op_id": "rep"}]]
    composed = ok(json_api.workspace_export_module(moved, {"module_id": "composed", "title": "Composed layout", "fixtures": [
        {"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]
    host = system_with_laser((-100.0, 0.0, 400.0))
    composed_import = {"instance_key": "sys", "fixtures": {"rebind": {"LASER": "LASER"}},
                       "placement": {"position_mm": [0.0, 0.0, 400.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}
    calls["ws_import_composed"] = ["workspace_import", [host, composed, composed_import]]
    # replacing an instance of a composed module replaces its nested instances (parent_instance)
    sys_ws = ok(json_api.workspace_import(host, composed, composed_import))["workspace"]
    calls["ws_replace_composed"] = ["workspace_replace_instance", [sys_ws, composed, {"instance_key": "sys", "fixtures": {"rebind": {"LASER": "LASER"}}}]]
    # no op_id: always applied, recorded under an id generated from the journal; a turn whose quaternion is written with w >= 0
    calls["ws_transform_without_op_id"] = ["workspace_transform", [moved, {"selector": {"instances": ["fb"]}, "rotation": {"axis": [0, 0, 1], "angle_deg": 270},
                                                                           "pivot": "selection_centroid"}]]
    calls["ws_status"] = ["workspace_status", [moved]]
    calls["ws_history"] = ["workspace_history", [moved, {"op_id": "rot37"}]]
    record = json_api.run_project(moved["project"])["local_validation"]
    calls["ws_record_validation"] = ["workspace_record_validation", [moved, {"local_validation": record}]]
    # a member deleted from the Project stays an edit of its instance: status, the refused and the discarding
    # replacement, and the member added back
    without_mirror = {**ws["project"], "components": [c for c in ws["project"]["components"] if c["instance_id"] != "fa_M1"]}
    calls["ws_update_project_deleting_a_member"] = ["workspace_update_project", [ws, {"project": without_mirror, "op_id": "del"}]]
    deleted = ok(json_api.workspace_update_project(ws, {"project": without_mirror, "op_id": "del"}))["workspace"]
    calls["ws_status_after_deletion"] = ["workspace_status", [deleted]]
    calls["ws_replace_deleted_refused"] = ["workspace_replace_instance", [deleted, fold, {"instance_key": "fa", "fixtures": rebind}]]
    calls["ws_replace_deleted_discarded"] = ["workspace_replace_instance", [deleted, fold, {"instance_key": "fa", "fixtures": rebind,
                                                                                            "discard_edits": True, "op_id": "back"}]]
    calls["ws_update_project_re_adding"] = ["workspace_update_project", [deleted, {"project": ws["project"], "op_id": "undo"}]]
    # the rest of the instance moved as a whole, then the member added back: an edit (its digest no longer matches)
    shift = {"selector": {"instances": ["fa"]}, "translation_mm": [0.0, 0.0, -30.0], "op_id": "mv"}
    calls["ws_transform_after_deletion"] = ["workspace_transform", [deleted, shift]]
    shifted = ok(json_api.workspace_transform(deleted, shift))["workspace"]
    mirror = next(c for c in ws["project"]["components"] if c["instance_id"] == "fa_M1")
    readd = {**shifted["project"], "components": [*shifted["project"]["components"], mirror]}
    calls["ws_update_project_re_adding_after_move"] = ["workspace_update_project", [shifted, {"project": readd, "op_id": "readd"}]]
    # an edited member brought back onto its import digest by a move of the whole instance: still an edit
    lowered = ok(json_api.workspace_transform(ws, {"selector": {"ids": ["fa_M1"]}, "translation_mm": [0.0, 0.0, -10.0], "op_id": "low"}))["workspace"]
    calls["ws_transform_edited_member_back"] = ["workspace_transform", [lowered, {"selector": {"instances": ["fa"]}, "translation_mm": [0.0, 0.0, 10.0],
                                                                                  "op_id": "up"}]]
    # a retained test source deleted: no port names it as its beam reference
    retained = ok(json_api.workspace_import(parent, fold, {"instance_key": "fr", "fixtures": {"policy": "retain_all"},
                                                           "placement": {"position_mm": [0.0, 0.0, 50.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    no_source = {**retained["project"], "components": [c for c in retained["project"]["components"] if c["instance_id"] != "fr_SRC"],
                 "targets": [t for t in retained["project"]["targets"] if t["target_id"] != "fr_T_OUT"]}
    calls["ws_status_deleted_beam_reference"] = ["workspace_status", [ok(json_api.workspace_update_project(retained, {"project": no_source}))["workspace"]]]
    # a module whose component and observation plane share an id: one digest each (with its kind)
    same_id = fold_module()
    same_id["observations"].append(screen("M1", (0.0, 30.0, 0.0), radius_mm=10.0))
    same_id_module = ok(json_api.workspace_export_module(ok(json_api.workspace_init(same_id, {"module": FOLD_DESCRIPTOR}))["workspace"], {}))["module"]
    calls["ws_import_same_id_members"] = ["workspace_import", [parent, same_id_module, {"instance_key": "fs", "fixtures": rebind, "op_id": "imp-fs"}]]
    same_ws = ok(json_api.workspace_import(parent, same_id_module, {"instance_key": "fs", "fixtures": rebind}))["workspace"]
    no_plane = {**same_ws["project"], "observations": [o for o in same_ws["project"]["observations"] if o["observation_id"] != "fs_M1"]}
    calls["ws_update_project_deleting_a_same_id_member"] = ["workspace_update_project", [same_ws, {"project": no_plane, "op_id": "del-fs"}]]
    # a nested instance replaced on its own by a revision adding a plane: its enclosing instance records the plane as
    # added; the plane deleted or edited is an edit of the enclosing instance too (refused, then discarded)
    extra = fold_module()
    extra["observations"].append(screen("EXTRA", (0.0, 70.0, 0.0), (half, -half, 0.0, 0.0), radius_mm=15.0))
    fold_r3 = ok(json_api.workspace_export_module(ok(json_api.workspace_init(extra, {"module": {**FOLD_DESCRIPTOR, "revision": "r3"}}))["workspace"],
                                                  {}))["module"]
    nested_ws = ok(json_api.workspace_replace_instance(sys_ws, fold_r3, {"instance_key": "sys_fa", "fixtures": rebind}))["workspace"]
    calls["ws_replace_nested_adding_a_member"] = ["workspace_replace_instance", [sys_ws, fold_r3, {"instance_key": "sys_fa", "fixtures": rebind,
                                                                                                   "op_id": "nested-r3"}]]
    calls["ws_status_nested_added_member"] = ["workspace_status", [nested_ws]]
    calls["ws_transform_enclosing_after_nested_replacement"] = ["workspace_transform", [nested_ws, {"selector": {"instances": ["sys"]},
                                                                                                    "translation_mm": [0.0, 0.0, 10.0], "op_id": "mv-sys"}]]
    no_extra = {**nested_ws["project"], "observations": [o for o in nested_ws["project"]["observations"] if o["observation_id"] != "sys_fa_EXTRA"]}
    nested_deleted = ok(json_api.workspace_update_project(nested_ws, {"project": no_extra}))["workspace"]
    enclosing = {"instance_key": "sys", "fixtures": {"rebind": {"LASER": "LASER"}}}
    calls["ws_replace_enclosing_after_nested_deletion_refused"] = ["workspace_replace_instance", [nested_deleted, composed, enclosing]]
    calls["ws_replace_enclosing_after_nested_deletion_discarded"] = ["workspace_replace_instance", [nested_deleted, composed, {
        **enclosing, "discard_edits": True, "op_id": "discard-sys"}]]
    edited_extra = json.loads(json.dumps(nested_ws["project"]))
    next(o for o in edited_extra["observations"] if o["observation_id"] == "sys_fa_EXTRA")["aperture"]["radius_mm"] = 9.0
    nested_edited = ok(json_api.workspace_update_project(nested_ws, {"project": edited_extra}))["workspace"]
    calls["ws_replace_enclosing_after_nested_edit_discarded"] = ["workspace_replace_instance", [nested_edited, composed, {
        **enclosing, "discard_edits": True}]]
    # a nested revision lacking the monitor plane and its target: the nested record keeps them as lacked; added back they
    # are members added to it (its replacement refused, then discarded)
    no_monitor = fold_module()
    no_monitor["observations"] = [o for o in no_monitor["observations"] if o["observation_id"] != "MON"]
    no_monitor["targets"] = [t for t in no_monitor["targets"] if t["target_id"] != "T_MON_CENTER"]
    fold_rm = ok(json_api.workspace_export_module(ok(json_api.workspace_init(no_monitor, {"module": {**FOLD_DESCRIPTOR, "revision": "rm",
                                                                                                      "requirements": []}}))["workspace"], {}))["module"]
    calls["ws_replace_nested_lacking_members"] = ["workspace_replace_instance", [sys_ws, fold_rm, {"instance_key": "sys_fa", "fixtures": rebind,
                                                                                                  "op_id": "nested-rm"}]]
    lacked_ws = ok(json_api.workspace_replace_instance(sys_ws, fold_rm, {"instance_key": "sys_fa", "fixtures": rebind}))["workspace"]
    monitor = [o for o in sys_ws["project"]["observations"] if o["observation_id"] == "sys_fa_MON"]
    monitor_target = [t for t in sys_ws["project"]["targets"] if t["target_id"] == "sys_fa_T_MON_CENTER"]
    back_project = {**lacked_ws["project"], "observations": [*lacked_ws["project"]["observations"], *monitor],
                    "targets": [*lacked_ws["project"]["targets"], *monitor_target]}
    calls["ws_update_project_adding_back_lacked_members"] = ["workspace_update_project", [lacked_ws, {"project": back_project, "op_id": "add-mon"}]]
    lacked_back = ok(json_api.workspace_update_project(lacked_ws, {"project": back_project}))["workspace"]
    calls["ws_replace_nested_with_lacked_members_added_refused"] = ["workspace_replace_instance", [lacked_back, fold, {"instance_key": "sys_fa",
                                                                                                                     "fixtures": rebind}]]
    calls["ws_replace_nested_with_lacked_members_added_discarded"] = ["workspace_replace_instance", [lacked_back, fold, {
        "instance_key": "sys_fa", "fixtures": rebind, "discard_edits": True, "op_id": "back-r1"}]]
    # the enclosing instance rotated as a whole, then an unedited nested instance replaced by its own revision: its
    # members keep their exact content
    turned = ok(json_api.workspace_transform(sys_ws, {"selector": {"instances": ["sys"]}, "rotation": {"axis": [1, 1, 0], "angle_deg": 37.0},
                                                      "frame": {"instance": "sys"}}))["workspace"]
    calls["ws_replace_nested_same_revision_after_rotation"] = ["workspace_replace_instance", [turned, fold, {"instance_key": "sys_fb",
                                                                                                            "fixtures": rebind, "op_id": "same-fb"}]]
    # a deleted plane of a composed module whose id lies in its nested instance's namespace stays reserved
    own = ok(json_api.workspace_import(system_with_laser(), fold, {"instance_key": "a", "fixtures": rebind}))["workspace"]
    own_project = json.loads(json.dumps(own["project"]))
    own_project["observations"].append(screen("a_Z", (0.0, 70.0, 0.0), (half, -half, 0.0, 0.0), radius_mm=15.0))
    own = ok(json_api.workspace_update_project(own, {"project": own_project}))["workspace"]
    own_module = ok(json_api.workspace_export_module(own, {"module_id": "own", "title": "Fold and a plane", "fixtures": [
        {"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]
    z_ws = ok(json_api.workspace_import(host, own_module, {"instance_key": "z", "fixtures": {"rebind": {"LASER": "LASER"}}}))["workspace"]
    z_ws = ok(json_api.workspace_update_project(z_ws, {"project": {**z_ws["project"], "observations": [
        o for o in z_ws["project"]["observations"] if o["observation_id"] != "z_a_Z"]}}))["workspace"]
    with_z = fold_module()
    with_z["observations"].append(screen("Z", (0.0, 70.0, 0.0), (half, -half, 0.0, 0.0), radius_mm=15.0))
    fold_rz = ok(json_api.workspace_export_module(ok(json_api.workspace_init(with_z, {"module": {**FOLD_DESCRIPTOR, "revision": "rz"}}))["workspace"],
                                                  {}))["module"]
    calls["ws_replace_nested_reserved_own_member_refused"] = ["workspace_replace_instance", [z_ws, fold_rz, {"instance_key": "z_a", "fixtures": rebind,
                                                                                                           "discard_edits": True}]]
    # a revision changing values far below 1 (pulse duration, power, a power target's minimum) reaches the Project; after
    # a rotation the plane the revision has unchanged keeps its exact turned content
    from local_helpers import source as gaussian_source

    def pulsed(revision, duration_s, power_w, minimum_w):
        src = gaussian_source("SRC", (-50.0, 0.0, 0.0), wavelength_nm=1064.0, waist_mm=0.5, power_w=power_w)
        src["source_model"]["pulse"] = {"repetition_rate_hz": 80000000.0, "pulse_duration_s": duration_s, "duration_definition": "FWHM"}
        project = project_of([src], [screen("MON", (50.0, 0.0, 0.0), radius_mm=10.0)],
                             [{"target_id": "T_P", "metric": "power_W", "observation_id": "MON", "minimum": minimum_w, "source_id": "SRC"}])
        pulsed_ws = ok(json_api.workspace_init(project, {"module": {"module_id": "pulsed", "title": "Pulsed source", "revision": revision}}))["workspace"]
        return ok(json_api.workspace_export_module(pulsed_ws, {}))["module"]

    pulsed_r1, pulsed_r2 = pulsed("r1", 1e-13, 1e-10, 1e-11), pulsed("r2", 1e-12, 9e-10, 5e-10)
    p_ws = ok(json_api.workspace_import(parent, pulsed_r1, {"instance_key": "p", "placement": {"position_mm": [0.0, 0.0, 300.0],
                                                                                              "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}))["workspace"]
    calls["ws_replace_small_value_revision"] = ["workspace_replace_instance", [p_ws, pulsed_r2, {"instance_key": "p", "op_id": "p-r2"}]]
    p_turned = ok(json_api.workspace_transform(p_ws, {"selector": {"instances": ["p"]}, "rotation": {"axis": [1, 1, 0], "angle_deg": 37.0},
                                                      "frame": {"instance": "p"}}))["workspace"]
    calls["ws_replace_small_value_revision_after_rotation"] = ["workspace_replace_instance", [p_turned, pulsed_r2, {"instance_key": "p"}]]
    # a nested instance moved on its own by 0.2 um (an edit of the enclosing instance), discarded exactly
    nudged = ok(json_api.workspace_transform(sys_ws, {"selector": {"instances": ["sys_fa"]}, "translation_mm": [0.0, 0.0, 2e-7]}))["workspace"]
    calls["ws_replace_discarding_a_tiny_move"] = ["workspace_replace_instance", [nudged, composed, {**enclosing, "discard_edits": True}]]
    # three levels: the middle instance replaced by a revision whose fold lacks the monitor plane; the fold, unedited,
    # back at its own revision
    nest_ws = ok(json_api.workspace_import(system_with_laser(), composed, {"instance_key": "c", "fixtures": {"rebind": {"LASER": "LASER"}}}))["workspace"]
    nest = ok(json_api.workspace_export_module(nest_ws, {"module_id": "nest", "title": "Nest", "fixtures": [
        {"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]
    d_ws = ok(json_api.workspace_import(host, nest, {**composed_import, "instance_key": "d"}))["workspace"]
    lacking_fa = ok(json_api.workspace_replace_instance(moved, fold_rm, {"instance_key": "fa", "fixtures": rebind}))["workspace"]
    composed_rm = ok(json_api.workspace_export_module(lacking_fa, {"module_id": "composed", "title": "Composed layout", "revision": "c3",
                                                                   "fixtures": [{"object_kind": "COMPONENT", "object_id": "LASER",
                                                                                 "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]
    middle = {"instance_key": "d_c", "fixtures": {"rebind": {"LASER": "LASER"}}}
    calls["ws_replace_middle_lacking_nested_members"] = ["workspace_replace_instance", [d_ws, composed_rm, {**middle, "op_id": "d-c3"}]]
    d_lacking = ok(json_api.workspace_replace_instance(d_ws, composed_rm, middle))["workspace"]
    calls["ws_replace_nested_back_at_its_revision"] = ["workspace_replace_instance", [d_lacking, fold, {"instance_key": "d_c_fa", "fixtures": rebind}]]
    # a definition of the instance that the system's own mirror uses: kept with the instance, reported
    shared = json.loads(json.dumps(ws["project"]))
    own_mirror = json.loads(json.dumps(next(c for c in shared["components"] if c["instance_id"] == "fa_M1")))
    own_mirror["instance_id"] = "U1"
    own_mirror["nominal_pose"]["position_mm"] = [0.0, 0.0, -500.0]
    shared["components"].append(own_mirror)
    next(c for c in shared["model_coatings"] if c["coating_id"].startswith("FA_"))["bands"][0]["reflectance"] = "0.9"
    shared_ws = ok(json_api.workspace_update_project(ws, {"project": shared}))["workspace"]
    fold_rb = ok(json_api.workspace_export_module(ok(json_api.workspace_init(fold_module(mirror_model_id="fold-mirror-b", reflectance="0.95"), {
        "module": {**FOLD_DESCRIPTOR, "revision": "rb"}}))["workspace"], {}))["module"]
    calls["ws_status_kept_definitions"] = ["workspace_status", [shared_ws]]
    calls["ws_replace_keeping_definitions_others_use"] = ["workspace_replace_instance", [shared_ws, fold_rb, {"instance_key": "fa", "fixtures": rebind,
                                                                                                            "op_id": "fa-rb"}]]
    # the same without the coating edit: replacing the instance by its own revision changes nothing (its model and coating
    # are the ones the Project has); with the edit, the revision defines the coating differently (refused)
    used = json.loads(json.dumps(ws["project"]))
    used["components"].append(own_mirror)
    used_ws = ok(json_api.workspace_update_project(ws, {"project": used}))["workspace"]
    calls["ws_replace_by_its_revision_while_definitions_are_used_outside"] = ["workspace_replace_instance", [used_ws, fold, {
        "instance_key": "fa", "fixtures": rebind, "op_id": "fa-r1"}]]
    calls["ws_replace_defining_a_used_definition_differently_refused"] = ["workspace_replace_instance", [shared_ws, fold, {
        "instance_key": "fa", "fixtures": rebind, "discard_edits": True}]]
    # the kept definitions once the system's mirror is gone: additions to the instance
    kept_ws = ok(json_api.workspace_replace_instance(used_ws, fold_rb, {"instance_key": "fa", "fixtures": rebind}))["workspace"]
    unused = {**kept_ws["project"], "components": [c for c in kept_ws["project"]["components"] if c["instance_id"] != "U1"]}
    calls["ws_update_project_leaving_kept_definitions_unused"] = ["workspace_update_project", [kept_ws, {"project": unused, "op_id": "drop-u1"}]]
    # a composed module exported while its instance keeps definitions for the module's own mirror: imported, the nested
    # instance keeps them the same way, replacing it by its own revision changes nothing and its enclosing instance
    # stays unedited
    bench = ok(json_api.workspace_export_module(kept_ws, {"module_id": "bench", "title": "Bench", "revision": "x1", "fixtures": [
        {"object_kind": "COMPONENT", "object_id": "LASER", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]
    bench_import = {"instance_key": "k", "fixtures": {"rebind": {"LASER": "LASER"}},
                    "placement": {"position_mm": [0.0, 0.0, 600.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}
    calls["ws_import_module_keeping_nested_definitions"] = ["workspace_import", [parent, bench, {**bench_import, "op_id": "bench"}]]
    k_ws = ok(json_api.workspace_import(parent, bench, bench_import))["workspace"]
    calls["ws_replace_nested_keeping_definitions_by_its_revision"] = ["workspace_replace_instance", [k_ws, fold_rb, {
        "instance_key": "k_fa", "fixtures": rebind, "op_id": "k-fa-rb"}]]
    k_same = ok(json_api.workspace_replace_instance(k_ws, fold_rb, {"instance_key": "k_fa", "fixtures": rebind}))["workspace"]
    calls["ws_status_after_nested_replacement_keeping_definitions"] = ["workspace_status", [k_same]]
    # a nested instance whose members were all deleted turns with its enclosing instance; replaced, its members are
    # where the turn puts them
    fa_ids = {e["parent_id"] for e in next(i for i in sys_ws["instances"] if i["instance_key"] == "sys_fa")["id_map"]
              if e["kind"] in ("COMPONENT", "OBSERVATION", "TARGET")}
    emptied = {**sys_ws["project"], **{name: [x for x in sys_ws["project"][name] if x[ident] not in fa_ids]
                                       for name, ident in (("components", "instance_id"), ("observations", "observation_id"),
                                                           ("targets", "target_id"))}}
    emptied_ws = ok(json_api.workspace_update_project(sys_ws, {"project": emptied}))["workspace"]
    turn_sys = {"selector": {"instances": ["sys"]}, "rotation": {"axis": [1, 1, 0], "angle_deg": 37.0}, "translation_mm": [0.0, 0.0, 10.0],
                "frame": {"instance": "sys"}, "op_id": "turn-sys"}
    calls["ws_transform_enclosing_with_an_emptied_nested_instance"] = ["workspace_transform", [emptied_ws, turn_sys]]
    turned_emptied = ok(json_api.workspace_transform(emptied_ws, turn_sys))["workspace"]
    calls["ws_replace_emptied_nested_after_enclosing_turn"] = ["workspace_replace_instance", [turned_emptied, fold, {
        "instance_key": "sys_fa", "fixtures": rebind, "discard_edits": True}]]
    # after the enclosing instance turned as a whole: a nested instance replaced by a revision moving its mirror (the
    # enclosing record names the object and placement it had), back at its revision, and an edit discarded at it
    sys_turned = ok(json_api.workspace_transform(sys_ws, {"selector": {"instances": ["sys"]}, "rotation": {"axis": [1, 1, 0], "angle_deg": 37.0},
                                                          "frame": {"instance": "sys"}}))["workspace"]
    shifted = fold_module()
    next(c for c in shifted["components"] if c["instance_id"] == "M1")["nominal_pose"]["position_mm"][1] += 1.0
    fold_r2m = ok(json_api.workspace_export_module(ok(json_api.workspace_init(shifted, {"module": {**FOLD_DESCRIPTOR, "revision": "r2m"}}))["workspace"],
                                                   {}))["module"]
    calls["ws_replace_nested_changing_a_member_after_a_turn"] = ["workspace_replace_instance", [sys_turned, fold_r2m, {
        "instance_key": "sys_fb", "fixtures": rebind, "op_id": "fb-r2m"}]]
    changed = ok(json_api.workspace_replace_instance(sys_turned, fold_r2m, {"instance_key": "sys_fb", "fixtures": rebind}))["workspace"]
    calls["ws_status_nested_revision_named"] = ["workspace_status", [changed]]
    calls["ws_replace_nested_back_after_a_turn"] = ["workspace_replace_instance", [changed, fold, {"instance_key": "sys_fb", "fixtures": rebind,
                                                                                                   "op_id": "fb-back"}]]
    nudged = json.loads(json.dumps(sys_turned["project"]))
    next(c for c in nudged["components"] if c["instance_id"] == "sys_fb_M1")["nominal_pose"]["position_mm"][2] += 1.0
    nudged_ws = ok(json_api.workspace_update_project(sys_turned, {"project": nudged}))["workspace"]
    calls["ws_replace_nested_discarding_after_a_turn"] = ["workspace_replace_instance", [nudged_ws, fold, {
        "instance_key": "sys_fb", "fixtures": rebind, "discard_edits": True, "op_id": "fb-discard"}]]
    # definitions sys and sys_fa keep for the system's own mirror (sys replaced by a revision whose fold fa lacks them),
    # left once the mirror is gone: replacing sys_fa with discard_edits removes them from the Project and from every
    # record, so sys_fa takes its first revision again
    moved_rb = ok(json_api.workspace_replace_instance(moved, fold_rb, {"instance_key": "fa", "fixtures": rebind}))["workspace"]
    composed_rb = ok(json_api.workspace_export_module(moved_rb, {"module_id": "composed", "title": "Composed layout", "revision": "cb",
                                                                 "fixtures": [{"object_kind": "COMPONENT", "object_id": "LASER",
                                                                               "role": "TEST_SOURCE", "default_action": "REMOVE"}]}))["module"]
    sys_used = json.loads(json.dumps(sys_ws["project"]))
    sys_mirror = json.loads(json.dumps(next(c for c in sys_used["components"] if c["instance_id"] == "sys_fa_M1")))
    sys_mirror["instance_id"] = "U1"
    sys_mirror["nominal_pose"]["position_mm"] = [0.0, 0.0, -500.0]
    sys_used["components"].append(sys_mirror)
    sys_kept = ok(json_api.workspace_replace_instance(ok(json_api.workspace_update_project(sys_ws, {"project": sys_used}))["workspace"], composed_rb,
                                                      enclosing))["workspace"]
    sys_left = ok(json_api.workspace_update_project(sys_kept, {"project": {**sys_kept["project"], "components": [
        c for c in sys_kept["project"]["components"] if c["instance_id"] != "U1"]}}))["workspace"]
    calls["ws_replace_nested_releasing_kept_definitions"] = ["workspace_replace_instance", [sys_left, fold_rb, {
        "instance_key": "sys_fa", "fixtures": rebind, "discard_edits": True, "op_id": "fa-release"}]]
    released = ok(json_api.workspace_replace_instance(sys_left, fold_rb, {"instance_key": "sys_fa", "fixtures": rebind, "discard_edits": True}))["workspace"]
    calls["ws_replace_nested_after_kept_definitions_left"] = ["workspace_replace_instance", [released, fold, {
        "instance_key": "sys_fa", "fixtures": rebind, "op_id": "fa-r1-again"}]]
    return calls


#: The sections of the harness (tests/pyodide/run_parity.mjs) a test may read, with the files each writes. Loading Pyodide,
#: installing the wheel and the catalog, and the environment and timings report (``_report.json``) always run.
HARNESS_FILES = {"fixtures": ("<request>.json",), "local_workflow": ("_local_workflow.json",),
                 "json_api_cases": ("_json_api.json",), "network_cases": ("_network.json",),
                 "json_api_calls": ("_json_api_calls.json",), "assembly": ("_assembly.json",),
                 "drawings": ("_drawing.json", "_drawing_2.json", "_drawing_3.json"), "layout": ("_layout.json",),
                 "prepared": ("_prepared.json",)}
#: The harness sections each Pyodide test reads. ``pyodide_run`` runs only the sections its session's selected tests read,
#: so a CI shard of the Pyodide job (.github/workflows/ci.yml, engine-pyodide) pays only for its own; a test missing here
#: makes the session run every section (test_every_pyodide_test_declares_the_harness_sections_it_reads keeps it complete).
HARNESS_SECTIONS: dict[str, frozenset[str]] = {name: frozenset(sections) for name, sections in {
    "test_pyodide_environment_matches_the_target_stack": (),
    "test_native_and_pyodide_results_agree_within_declared_tolerance": ("fixtures",),
    "test_pyodide_leaves_targets_behind_an_out_of_domain_tabulated_lens_undecided": ("json_api_cases",),
    "test_pyodide_reads_the_faraday_isolator_as_derived_by_hand": ("json_api_cases",),
    "test_pyodide_reads_the_grating_cases_as_derived_by_hand": ("json_api_cases",),
    "test_pyodide_out_of_scope_results_are_identical_in_kind": ("fixtures",),
    "test_pyodide_benchmark_warm_time_is_recorded_and_within_ci_guardrail": ("fixtures",),  # _report.json's fixture timings
    "test_the_browser_runtime_installs_the_catalog_it_is_given": (),
    "test_the_mode_a_design_workflow_runs_in_pyodide": ("local_workflow",),
    "test_the_browser_json_api_is_semantically_identical_to_native": ("json_api_cases",),
    "test_the_browser_reports_the_plane_states_the_marker_and_the_withheld_bound": ("json_api_cases",),
    "test_the_browser_reports_the_open_phase_range_and_the_stated_phase_value": ("json_api_cases",),
    "test_the_other_json_api_entry_points_in_pyodide_are_identical_to_native": ("json_api_calls",),
    "test_the_network_export_in_pyodide_has_the_native_contract_hash": ("network_cases",),
    "test_the_assembly_check_in_pyodide_is_identical_to_native": ("assembly",),
    "test_the_assembly_drawing_in_pyodide_is_byte_identical_to_native": ("drawings",),
    "test_the_multi_chain_assembly_drawing_in_pyodide_is_byte_identical_to_native": ("drawings",),
    "test_a_drawing_with_a_customer_supplied_component_is_byte_identical_in_pyodide": ("drawings",),
    "test_the_optical_layout_drawing_in_pyodide_is_byte_identical_to_native": ("layout",),
    "test_prepared_evaluation_handles_in_pyodide_are_identical_to_native": ("prepared",),
}.items()}
#: The json_api cases (``json_api_cases()``) the harness runs for each reader of its json_api section: the semantic comparison
#: reads one part of them (``json_api_part``), the other readers the cases named here, through ``_browser_cases``, which
#: gives them no other case. The cases are the bulk of the harness (about two thirds of the job), so the parts are what the
#: CI shards balance (.github/ci-shard-weights/engine-pyodide.json).
JSON_API_CASES_READ: dict[str, tuple[str, ...]] = {
    "test_pyodide_leaves_targets_behind_an_out_of_domain_tabulated_lens_undecided": (
        "tabulated_lens_outside_band", "tabulated_lens_beyond_field_angle", "tabulated_lens_beside_and_before"),
    "test_pyodide_reads_the_faraday_isolator_as_derived_by_hand": (
        "faraday_isolator", "faraday_before_a_detector", "faraday_outside_table"),
    "test_pyodide_reads_the_grating_cases_as_derived_by_hand": (
        "grating_without_response", "grating_with_response", "grating_before_a_detector"),
    "test_the_browser_reports_the_plane_states_the_marker_and_the_withheld_bound": (
        "tail_energy_flow", "ghost_on_a_plane", "coating_use_refused", "coincident_normal_mirror", "coincident_splitter"),
    "test_the_browser_reports_the_open_phase_range_and_the_stated_phase_value": (
        "michelson_public_unstated_bulk", "michelson_open_phase", "michelson_stated_phases"),
}
#: The semantic comparison of the json_api cases runs as this many tests, each over every JSON_API_PARTS-th case.
JSON_API_PARTS = 6
SEMANTIC_TEST = "test_the_browser_json_api_is_semantically_identical_to_native"


def json_api_part(names, part: int) -> list[str]:
    """Part ``part`` (0-based) of the json_api case names: every JSON_API_PARTS-th, in ``json_api_cases()`` order."""
    return list(names)[part::JSON_API_PARTS]


def _json_api_cases_read_by(item, names) -> set[str]:
    """The json_api cases one selected test reads (all of them for a Pyodide test missing from the tables)."""
    test = getattr(item, "originalname", item.name)
    if test == SEMANTIC_TEST:
        return set(json_api_part(names, item.callspec.params["part"]))
    if test in JSON_API_CASES_READ:
        return set(JSON_API_CASES_READ[test])
    return set() if test in HARNESS_SECTIONS else set(names)


def _harness_sections_read_by(items) -> frozenset[str]:
    """The harness sections the session's selected tests read (every section for a Pyodide test missing from the table)."""
    sections: set[str] = set()
    for item in items:
        if "pyodide_run" in getattr(item, "fixturenames", ()):
            sections |= HARNESS_SECTIONS.get(getattr(item, "originalname", item.name), frozenset(HARNESS_FILES))
    return frozenset(sections)


def _browser_cases(pyodide_run: Path, test: str) -> dict[str, dict]:
    """The Pyodide replies of the json_api cases ``test`` declares in JSON_API_CASES_READ, and no other."""
    cases = json.loads((pyodide_run / "_json_api.json").read_text(encoding="utf-8"))["cases"]
    return {name: cases[name] for name in JSON_API_CASES_READ[test]}


def test_every_pyodide_test_declares_the_harness_sections_it_reads():
    """HARNESS_SECTIONS names every Pyodide test, and each entry covers every harness file the test's body reads: a test that
    read a section it does not declare would find no file in a session (a CI shard) that selects no other reader of it. A
    reader of the json_api section reads it whole only as the parted semantic comparison, the others through
    ``_browser_cases`` under their own name."""
    import inspect
    import re

    tests = {name: fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)}
    readers = {name for name, fn in tests.items() if "pyodide_run" in inspect.signature(fn).parameters}
    marked = {name for name, fn in tests.items() if any(mark.name == "pyodide" for mark in getattr(fn, "pytestmark", ()))}
    assert readers == marked == set(HARNESS_SECTIONS)
    section_of = {file: section for section, files in HARNESS_FILES.items() for file in files if not file.startswith("<")}
    for name in readers:
        assert HARNESS_SECTIONS[name] <= set(HARNESS_FILES), name
        source = inspect.getsource(tests[name])
        read = set(re.findall(r'pyodide_run / f?"([^"]+)"', source))
        if "_browser_cases(pyodide_run" in source:
            assert f'_browser_cases(pyodide_run, "{name}")' in source and name in JSON_API_CASES_READ, name
            read.add("_json_api.json")
        assert read, name
        for file in read - {"_report.json"}:  # the environment, catalog and timings report is always written
            assert section_of.get(file, "fixtures") in HARNESS_SECTIONS[name], (name, file)
        if "_json_api.json" in read:
            assert (name in JSON_API_CASES_READ) != (name == SEMANTIC_TEST), name
            assert ('pyodide_run / "_json_api.json"' in source) == (name == SEMANTIC_TEST), name
    assert set(JSON_API_CASES_READ) <= readers


def test_the_harness_runs_only_the_sections_and_cases_the_selected_tests_read():
    class Item:
        def __init__(self, name: str, fixtures=("pyodide_run",), part=None):
            self.originalname, self.name, self.fixturenames = name, name if part is None else f"{name}[{part}]", fixtures
            self.callspec = type("CallSpec", (), {"params": {"part": part}})()

    assert _harness_sections_read_by([Item("test_the_browser_runtime_installs_the_catalog_it_is_given")]) == frozenset()
    assert _harness_sections_read_by([Item("test_the_network_export_in_pyodide_has_the_native_contract_hash"),
                                      Item("test_the_assembly_check_in_pyodide_is_identical_to_native"),
                                      Item("test_wheel_is_pure_python_and_declares_the_dependency_policy", ("wheel",))]) == {
        "network_cases", "assembly"}
    assert _harness_sections_read_by([Item("test_a_pyodide_test_not_in_the_table")]) == set(HARNESS_FILES)
    names = [f"case{index}" for index in range(2 * JSON_API_PARTS + 1)]
    assert _json_api_cases_read_by(Item(SEMANTIC_TEST, part=1), names) == {"case1", f"case{JSON_API_PARTS + 1}"}
    assert _json_api_cases_read_by(Item("test_pyodide_reads_the_faraday_isolator_as_derived_by_hand"), names) == {
        "faraday_isolator", "faraday_before_a_detector", "faraday_outside_table"}
    assert _json_api_cases_read_by(Item("test_the_mode_a_design_workflow_runs_in_pyodide"), names) == set()
    assert _json_api_cases_read_by(Item("test_a_pyodide_test_not_in_the_table"), names) == set(names)
    # the parts of the semantic comparison cover every case exactly once
    parts = [json_api_part(names, part) for part in range(JSON_API_PARTS)]
    assert sorted(name for part in parts for name in part) == sorted(names) and all(parts)


@pytest.fixture(scope="session")
def pyodide_run(wheel, tmp_path_factory, request):
    if os.environ.get("OPENSCI_SKIP_PYODIDE") == "1":
        pytest.skip("OPENSCI_SKIP_PYODIDE=1 was set explicitly (CI must not set this)")
    node = shutil.which("node")
    assert node, "Pyodide tests need Node.js on PATH (install Node >= 20) — or set OPENSCI_SKIP_PYODIDE=1 to skip explicitly"
    if not (PYODIDE_DIR / "node_modules" / "pyodide").exists():
        npm = shutil.which("npm")
        assert npm, "npm is required to install the pinned pyodide package (tests/pyodide/package-lock.json)"
        proc = subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=PYODIDE_DIR, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path_factory.mktemp("pyodide_out")
    items = request.session.items
    sections = _harness_sections_read_by(items)
    args = [node, "run_parity.mjs", "--wheel", str(wheel), "--requests", str(REQ_DIR), "--out", str(out), "--warm-iterations", "3",
            "--catalog", str(REAL_CATALOG)]
    if skipped := [section for section in ("fixtures", "local_workflow") if section not in sections]:
        args += ["--skip", ",".join(skipped)]
    if "json_api_cases" in sections:
        documents = json_api_cases()
        read = set().union(*(_json_api_cases_read_by(item, documents) for item in items
                             if "pyodide_run" in getattr(item, "fixturenames", ())))
        assert read <= set(documents), sorted(read - set(documents))
        cases = out / "_json_api_cases.json"
        cases.write_text(json.dumps({name: document for name, document in documents.items() if name in read}), encoding="utf-8")
        args += ["--json-api-cases", str(cases)]
    if "network_cases" in sections:
        network = out / "_network_cases.json"
        network.write_text(json.dumps(network_cases()), encoding="utf-8")
        args += ["--network-cases", str(network)]
    if "json_api_calls" in sections:
        calls = out / "_json_api_call_cases.json"
        calls.write_text(json.dumps(json_api_calls()), encoding="utf-8")
        args += ["--json-api-calls", str(calls)]
    if "prepared" in sections:
        prepared = out / "_prepared_sequence.json"
        prepared.write_text(json.dumps({"script": PREPARED_SEQUENCE, "documents": prepared_documents()}), encoding="utf-8")
        args += ["--prepared-sequence", str(prepared)]
    if "assembly" in sections:
        args += ["--assembly-package", str(ASSEMBLY_PACKAGE)]
    if "drawings" in sections:
        args += ["--drawing-package", str(DRAWING_PACKAGE), "--drawing-detail", str(DRAWING_DETAIL),
                 "--drawing-package-2", str(MULTI_DRAWING_PACKAGE), "--drawing-detail-2", str(MULTI_DRAWING_DETAIL),
                 "--drawing-package-3", str(CUSTOMER_DRAWING_PACKAGE), "--drawing-detail-3", str(CUSTOMER_DRAWING_DETAIL)]
    if "layout" in sections:
        from layout_helpers import layout_package

        layout = out / "_optical_layout_package.json"
        layout.write_text(json.dumps(layout_package().model_dump(mode="json", by_alias=True), ensure_ascii=False), encoding="utf-8")
        args += ["--layout-package", str(layout)]
    proc = subprocess.run(args, cwd=PYODIDE_DIR, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, f"Pyodide harness failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return out


@pytest.mark.pyodide
def test_pyodide_environment_matches_the_target_stack(pyodide_run):
    report = json.loads((pyodide_run / "_report.json").read_text(encoding="utf-8"))
    env = report["environment"]
    assert report["pyodide_version"].startswith(PYODIDE_PIN)
    assert env["python"].startswith("3.14") and env["platform"] == "emscripten"
    assert env["os_optics"] == __version__ == env["engine_dist_version"]
    assert env["engine_build_id"] == engine_build_id()  # the wheel installed in Pyodide is the same source as native
    for key in ("numpy", "scipy", "pydantic", "shapely"):
        assert env[key]  # provided by the Pyodide distribution itself


@pytest.mark.pyodide
@pytest.mark.parametrize("name", NAMES)
def test_native_and_pyodide_results_agree_within_declared_tolerance(name, pyodide_run):
    text = (REQ_DIR / f"{name}.json").read_text(encoding="utf-8")
    native = validate_json(text)
    pyodide = (pyodide_run / f"{name}.json").read_text(encoding="utf-8")
    diffs = compare_results(native, pyodide)
    assert diffs == [], "\n".join(str(d) for d in diffs[:10])
    n, p = json.loads(native), json.loads(pyodide)
    assert n["status"] == p["status"]  # semantic status identical
    assert n["manifest"]["input_hash"] == p["manifest"]["input_hash"]  # canonical input hash identical
    assert [t["status"] for t in n["targets"]] == [t["status"] for t in p["targets"]]
    assert [i["code"] for i in n["issues"]] == [i["code"] for i in p["issues"]]  # warnings / OUT_OF_MODEL_SCOPE never change with runtime


@pytest.mark.pyodide
def test_pyodide_leaves_targets_behind_an_out_of_domain_tabulated_lens_undecided(pyodide_run):
    """The browser runtime's own reading of the scientific-result acceptance cases (native and Pyodide are also compared in full above)."""

    cases = _browser_cases(pyodide_run, "test_pyodide_leaves_targets_behind_an_out_of_domain_tabulated_lens_undecided")
    expected = {"tabulated_lens_outside_band": {"spot": ("INDETERMINATE", ["MODEL_WAVELENGTH_OUTSIDE_BAND"])},
                "tabulated_lens_beyond_field_angle": {"spot": ("INDETERMINATE", ["MODEL_FIELD_ANGLE_EXCEEDED"])},
                "tabulated_lens_beside_and_before": {"spot": ("INDETERMINATE", ["MODEL_WAVELENGTH_OUTSIDE_BAND"]),
                                                     "before": ("MET", []), "beside": ("MET", [])}}
    for name, targets in expected.items():
        rows = {t["target_id"]: t for t in cases[name]["summary"]["targets"]}
        assert {k: (rows[k]["assessment"], rows[k]["applicability_issue_codes"]) for k in targets} == targets, name
        assert all(rows[k]["status"] == "PASS" for k in targets), name


@pytest.mark.pyodide
def test_pyodide_reads_the_faraday_isolator_as_derived_by_hand(pyodide_run):
    """The browser runtime's own numbers for both launches, and its refusals: the line outside the table has no value; a
    network through the rotator is refused (native and Pyodide are also compared in full above)."""
    from test_faraday_rotator import POWER_W, TF, _hand_ports

    cases = _browser_cases(pyodide_run, "test_pyodide_reads_the_faraday_isolator_as_derived_by_hand")
    lines = {line["source_id"]: line for line in cases["faraday_isolator"]["results"]["power_budget"]["result"]["lines"]}
    forward, reverse = _hand_ports(45.0, TF)
    assert lines["FWD"]["observed_W"]["OUT"] == pytest.approx(POWER_W * forward, rel=1e-12)
    assert lines["REV"]["observed_W"]["IN"] == pytest.approx(POWER_W * reverse, rel=1e-12)
    assert {p["code"] for p in cases["faraday_before_a_detector"]["network"]["error"]["problems"]} == {"PHASE_UNDEFINED_FARADAY_ROTATOR"}
    outside = cases["faraday_outside_table"]
    assert outside["status"] == "OUT_OF_MODEL_SCOPE"
    (target,) = outside["summary"]["targets"]
    assert target["value"] is None and target["assessment"] == "INDETERMINATE"


@pytest.mark.pyodide
def test_pyodide_reads_the_grating_cases_as_derived_by_hand(pyodide_run):
    """The browser runtime's own numbers: the order positions of the closed form with no power value, the powers of the stated table,
    the network refusal (native and Pyodide are also compared in full above)."""
    from test_diffraction_grating import POWER_W, theta_m

    cases = _browser_cases(pyodide_run, "test_pyodide_reads_the_grating_cases_as_derived_by_hand")
    unknown = cases["grating_without_response"]
    assert unknown["status"] == "UNKNOWN"
    (screen,) = unknown["summary"]["observations"]
    assert screen["quantity_state"] == "UNKNOWN" and all(a["power_W"] is None for a in screen["arrivals"])
    assert len(screen["arrivals"]) == 5
    positions = sorted(a["position_mm"][1] for a in screen["arrivals"])
    expected = sorted(120.0 * math.tan(math.radians(theta_m(0.0, m, 550.0, 300.0))) for m in (-2, -1, 0, 1, 2))
    assert positions == pytest.approx(expected, abs=1e-6)  # (the summary's positions are quantized to the hash's digits)
    (line,) = unknown["results"]["power_budget"]["result"]["lines"]
    assert line["sinks"] == {"UNMODELLED_RESPONSE": pytest.approx(POWER_W, rel=1e-12)}
    (stated,) = cases["grating_with_response"]["summary"]["observations"]
    assert sorted(a["power_W"] for a in stated["arrivals"]) == pytest.approx(sorted(POWER_W * v for v in (0.15, 0.2, 0.45)), rel=1e-9)
    assert {p["code"] for p in cases["grating_before_a_detector"]["network"]["error"]["problems"]} == {"PHASE_UNDEFINED_GRATING"}


@pytest.mark.pyodide
def test_pyodide_out_of_scope_results_are_identical_in_kind(pyodide_run):
    p = json.loads((pyodide_run / "out_of_scope_targets.json").read_text(encoding="utf-8"))
    assert p["status"] == "OUT_OF_MODEL_SCOPE" and {t["status"] for t in p["targets"]} == {"OUT_OF_MODEL_SCOPE", "UNKNOWN"}


def _pyodide_timings_markdown(report: dict) -> str:
    """The Node/Pyodide parity run's load laps and warm benchmark as Markdown (the CI job summary)."""
    bench = report["fixtures"]["benchmark_standard"]
    lines = ["### Pyodide parity timings (engine-pyodide, Node)", "",
             f"Pyodide {report['pyodide_version']}. Load laps run in sequence; these measure the retained "
             "headless Engine runtime in Node, not a browser page.", "", "| lap | ms |", "|---|---:|"]
    lines += [f"| {label} | {ms} |" for label, ms in report["timings_ms"].items()]
    lines += [f"| benchmark_standard warm median | {bench['warm_median_ms']} |",
              f"| benchmark_standard warm max (limit 5000) | {bench['warm_max_ms']} |", ""]
    return "\n".join(lines)


@pytest.mark.pyodide
def test_pyodide_benchmark_warm_time_is_recorded_and_within_ci_guardrail(pyodide_run):
    report = json.loads((pyodide_run / "_report.json").read_text(encoding="utf-8"))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(_pyodide_timings_markdown(report) + "\n")
    bench = report["fixtures"]["benchmark_standard"]
    assert bench["warm_median_ms"] > 0
    # This CI check is a regression guard, not a controlled performance benchmark. Three warm samples cannot
    # establish P95; runner scheduling alone has produced 2.0-2.4 s samples, and the reference workload grows with the
    # Engine's results (exposures, bounds on untraced light, runtime-independent arithmetic: about 2.7 s on the CI runner
    # at engine 0.8.0). The ceiling is 5 s so that only gross slowdowns fail this check; the
    # timings stay recorded in the report for trend checks.
    assert bench["warm_max_ms"] <= 5000.0, bench
    assert "pyodide_core_load" in report["timings_ms"] and "engine_wheel_install" in report["timings_ms"]


@pytest.mark.pyodide
def test_the_browser_runtime_installs_the_catalog_it_is_given(pyodide_run):
    """The wheel bundles no catalog and Pyodide never fetches one: before the host installs a release nothing is
    available; ``json_api.install_catalog`` then makes the given release the active one."""

    catalog = json.loads((pyodide_run / "_report.json").read_text(encoding="utf-8"))["catalog"]
    assert catalog["before"]["available"] is False and catalog["before"]["active"] is None
    digest = hashlib.sha256(REAL_CATALOG.read_bytes()).hexdigest()
    assert catalog["install"]["ok"] is True and catalog["install"]["sha256"] == digest
    assert catalog["after"]["active"]["source"] == "installed" and catalog["after"]["active"]["sha256"] == digest


@pytest.mark.pyodide
def test_the_mode_a_design_workflow_runs_in_pyodide(pyodide_run):
    """Same Project semantics in the browser runtime: one run gives the record, the view and a verifiable run directory."""
    from os_optics.local import example_project, install_catalog, run_optical_project

    install_catalog(REAL_CATALOG.read_bytes())  # the same catalog, installed the way the browser runtime installs it

    browser = json.loads((pyodide_run / "_local_workflow.json").read_text(encoding="utf-8"))
    native = run_optical_project(example_project())
    record, scene = browser["record"], browser["scene"]
    assert browser["status"] == native.status.value == "UNKNOWN"  # public glass states no bulk transmission
    assert record["validation_status"] == record["simulation"]["status"] == "UNKNOWN"
    assert {t["target_id"]: t["status"] for t in record["simulation"]["target_outcomes"]} == {
        "focus_position": "PASS", "spot_radius": "PASS"}  # geometry remains decidable
    assert any(o["quantity_state"] == "UNKNOWN" for o in record["simulation"]["observation_quantities"])
    assert browser["verify"] == []
    assert browser["files"] == sorted(["manifest.json", "project.json", "engine_request.json", "simulation_result.json", "ray_data.json",
                                       "local_validation.json", "optical_scene.json", "optical_scene.glb"])
    assert record["runtime"]["runtime_kind"] == "BROWSER_PYTHON"
    native_record = native.local_validation.payload()
    assert record["project"] == native_record["project"] and record["input_hash"] == native_record["input_hash"]
    assert [(o["target_id"], o["status"]) for o in record["simulation"]["target_outcomes"]] == [
        (o["target_id"], o["status"]) for o in native_record["simulation"]["target_outcomes"]]
    assert scene["engine_run"]["input_hash"] == record["input_hash"] and scene["engine_run"]["result_hash"] == record["result_hash"]
    assert [e["element_id"] for e in scene["elements"]] == [e.element_id for e in native.optical_view.elements]
    assert len(scene["beam_paths"]) == len(native.optical_view.beam_paths)


def _close_documents(a, b, rel: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """Equal JSON documents except floats within ``rel`` / ``abs_tol`` (cross-runtime last-ulp noise)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close_documents(a[k], b[k], rel, abs_tol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_close_documents(x, y, rel, abs_tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))
    return a == b


def _without_runtime_kind(record: dict) -> dict:
    record = json.loads(json.dumps(record))
    record["runtime"].pop("runtime_kind")
    return record


def _left_fold(iterable, /, start=0):
    total = start
    for value in iterable:
        total = total + value
    return total


def _correctly_rounded(iterable, /, start=0):
    values = list(iterable)
    return math.fsum(values) if start == 0 and values and all(isinstance(v, float) for v in values) else _left_fold(values, start)


@pytest.mark.parametrize("name", ["declared_objective", "turned_coatings_phases_polarization"])
def test_run_files_do_not_depend_on_how_the_interpreter_sums_floats(name, monkeypatch, tmp_path):
    """The built-in ``sum`` folds on Python 3.11 and compensates from 3.12, so a run that reports it differs between a native 3.11
    and Pyodide (NUMERICAL_CONVENTIONS section 4). The parity test below runs on 3.14 alone and cannot see that. Two other sums
    in its place must leave every run file unchanged on any interpreter: these two cases are the ones that moved."""
    import builtins

    from os_optics.local import json_api

    document = json_api_cases()[name]
    files = {}
    for label, implementation in (("left_fold", _left_fold), ("correctly_rounded", _correctly_rounded)):
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "sum", implementation)
            files[label] = json_api.run_project(document, output_dir=str(tmp_path / label))["files"]
    assert files["left_fold"] == files["correctly_rounded"]


def _assert_configurations_identical(got: dict, want: dict, name: str) -> None:
    """A run of a Project with configurations in Pyodide and natively: the same input hashes, executions and comparison;
    per configuration the same summary, scene and record apart from ``runtime_kind``; every file identical but the records,
    the manifests (their digests of the records) and the result's runtime environment record."""
    import hashlib

    from os_optics.local import json_api

    assert got["verify"] == want["verify"] == [], name
    for key in ("project_hash", "engine_input_hash", "execution", "comparison"):
        assert got[key] == want[key], (name, key)
    assert [c["configuration_id"] for c in got["configurations"]] == [c["configuration_id"] for c in want["configurations"]]
    for mine, native in zip(got["configurations"], want["configurations"], strict=True):
        assert mine["status"] == native["status"] and mine["execution"] == native["execution"], name
        summary, native_summary = dict(mine["summary"]), dict(native["summary"])
        assert (summary.pop("runtime_kind"), native_summary.pop("runtime_kind")) == ("BROWSER_PYTHON", "PYTHON_LOCAL")
        assert _without_runtime_kind(summary.pop("local_validation")) == _without_runtime_kind(native_summary.pop("local_validation"))
        assert summary == native_summary, (name, mine["configuration_id"])
        assert mine["optical_scene"] == native["optical_scene"], name
        assert _without_runtime_kind(mine["local_validation"]) == _without_runtime_kind(native["local_validation"]), name
    native_files = {k: hashlib.sha256(v).hexdigest() for k, v in want["files"].items()}
    assert set(got["files"]) == set(native_files), name
    differing = sorted(k for k in native_files if got["files"][k]["sha256"] != native_files[k])
    allowed = {"manifest.json"} | {f"{c['directory']}/{f}" for c in want["configurations"]
                                   for f in ("local_validation.json", "manifest.json", "simulation_result.json")}
    assert {"manifest.json", "comparison.json"} - set(differing) == {"comparison.json"} and set(differing) <= allowed, (name, differing)
    native_results = {q: json_api.run_results(want["files"], {"kind": q}) for q in ("monitors", "paths", "power_budget", "exposures")}
    native_results["rays"] = json_api.run_results(want["files"], {"kind": "rays", "limit": 4})
    assert got["results"] == json.loads(json.dumps(native_results)), name  # the top directory holds no run: the same refusals


@pytest.mark.pyodide
@pytest.mark.parametrize("part", range(JSON_API_PARTS), ids=lambda part: f"part{part + 1}of{JSON_API_PARTS}")
def test_the_browser_json_api_is_semantically_identical_to_native(pyodide_run, tmp_path, part):
    """The browser (Pyodide) runtime calls os_optics.local.json_api. For every case the Pyodide reply equals the native reply;
    the only runtime-dependent value is ``runtime_kind`` (PYTHON_LOCAL vs BROWSER_PYTHON), which the Local Validation
    record carries and therefore local_validation.json's bytes and the manifest's digest of that file.

    simulation_result.json is the full-precision Engine result. It carries the runtime environment it ran in
    (``manifest.runtime``: platform, Python / numpy / pydantic versions); apart from that record it is identical bit for bit,
    as are ray_data.json and every other run file (PV-F1: the trace, the Gaussian and polarization states, the phases and
    the pair metrics use no fused multiply-add and no libm; NUMERICAL_CONVENTIONS section 4).

    The linear optical network export of every case is the native one too (acceptance 3 of the network export, as for the Engine requests in
    ``test_the_network_export_in_pyodide_has_the_native_contract_hash``). The cases run as JSON_API_PARTS tests, ``part``
    taking every JSON_API_PARTS-th case (``json_api_part``), so that CI shards can share them."""
    import base64
    import hashlib

    from os_optics.local import json_api

    assert json_api.install_catalog(REAL_CATALOG.read_bytes())["ok"] is True  # as the browser runtime installs it
    browser = json.loads((pyodide_run / "_json_api.json").read_text(encoding="utf-8"))
    caps = dict(browser["capabilities"])
    assert caps.pop("runtime_kind") == "BROWSER_PYTHON"
    native_caps = json_api.capabilities()
    assert native_caps.pop("runtime_kind") == "PYTHON_LOCAL"
    assert caps == native_caps  # same engine build id, catalog, metrics, roles
    statuses = {}
    documents = json_api_cases()
    names = json_api_part(documents, part)
    for name in names:
        document, got = documents[name], browser["cases"][name]
        assert got.pop("network") == json_api.linear_optical_network(document), name
        want = json_api.run_project(document, output_dir=str(tmp_path / name))
        assert got["check"] == json_api.check_project(document), name
        if want["ok"] and "configurations" in want:  # a Project with configurations: each configuration as a run
            _assert_configurations_identical(got, want, name)
            statuses[name] = "CONFIGURATIONS"
            continue
        if not want["ok"]:
            got.pop("files")
            got.pop("check")
            assert got == want, name  # identical refusal: same codes, messages and locations
            statuses[name] = "REFUSED"
            continue
        statuses[name] = want["status"]
        if name in {"declared_objective", "declared_compound_unit"}:
            execution = want["summary"]["local_validation"]["simulation"]["outcome"]["execution"]
            # (the probes' budget is an auxiliary analysis's: recorded per phase, the execution COMPLETED)
            assert execution["outcome"] == "COMPLETED"
            assert execution["budgets"]["exhausted_by_phase"] == {"untraced_light": ["max_trace_steps"]}
        assert got["verify"] == want["verify"] == [], name
        assert got["status"] == want["status"], name
        summary, native_summary = dict(got["summary"]), dict(want["summary"])
        assert (summary.pop("runtime_kind"), native_summary.pop("runtime_kind")) == ("BROWSER_PYTHON", "PYTHON_LOCAL")
        assert _without_runtime_kind(summary.pop("local_validation")) == _without_runtime_kind(native_summary.pop("local_validation"))
        # project hash, catalog binding, Engine input AND result hash, target values (bit for bit), observed beams,
        # retained DOFs, Engine build id and the scene content hash
        assert summary == native_summary, name
        assert got["optical_scene"] == want["optical_scene"], name
        assert _without_runtime_kind(got["local_validation"]) == _without_runtime_kind(want["local_validation"]), name
        native_files = {k: hashlib.sha256(v).hexdigest() for k, v in want["files"].items()}
        differing = sorted(k for k in native_files if got["files"][k]["sha256"] != native_files[k])
        # only the runtime record differs: runtime_kind (local_validation.json, the manifest's digest of it) and the runtime
        # environment in simulation_result.json's manifest; every full-precision float is the same bit for bit
        assert {"local_validation.json", "manifest.json"} <= set(differing) <= {"local_validation.json", "manifest.json",
                                                                               "simulation_result.json"}, (name, differing)
        native_results = {q: json_api.run_results(want["files"], {"kind": q}) for q in ("monitors", "paths", "power_budget", "exposures")}
        native_results["rays"] = json_api.run_results(want["files"], {"kind": "rays", "limit": 4})
        monitors = native_results["monitors"]
        if monitors["ok"] and monitors["result"]:  # the transverse-plane handoff of the first monitor
            native_results["handoff"] = json_api.run_results(want["files"], {"kind": "handoff",
                                                                             "monitor": monitors["result"][0]["observation_id"]})
        if name.startswith("ghost_"):  # the ghost-path analysis, as the harness asks for it
            native_results["ghosts"] = json_api.run_results(want["files"], {"kind": "ghosts", "monitor": "image"})
        if name.startswith("field_"):  # the field propagation of a sampled field, as the harness asks for it
            native_results["field"] = json_api.run_results(want["files"], {"kind": "field", "monitor": "P"})
        assert _close_documents(got.pop("results"), json.loads(json.dumps(native_results))), name
        browser_result = base64.b64decode(got["files"]["simulation_result.json"]["base64"]).decode()
        native_result = want["files"]["simulation_result.json"].decode()
        assert compare_results(native_result, browser_result) == [], name
        browser_doc, native_doc = json.loads(browser_result), json.loads(native_result)
        browser_doc["manifest"].pop("runtime"), native_doc["manifest"].pop("runtime")
        assert browser_doc == native_doc, name  # bit for bit apart from the runtime environment record
        browser_manifest, native_manifest_result = json.loads(browser_result)["manifest"], json.loads(native_result)["manifest"]
        assert browser_manifest["result_hash"] == native_manifest_result["result_hash"] == want["summary"]["engine_result_hash"], name
        assert browser_manifest["runtime"]["platform"] == "emscripten"
        for m in (browser_manifest, native_manifest_result):
            m.pop("runtime")  # the runtime environment record (platform, python, numpy, pydantic)
        assert browser_manifest == native_manifest_result, name
        for other in ("project.json", "engine_request.json", "optical_scene.json", "optical_scene.glb"):
            assert got["files"][other]["sha256"] == native_files[other], (name, other)
        browser_record = json.loads(base64.b64decode(got["files"]["local_validation.json"]["base64"]))
        assert browser_record["runtime"]["runtime_kind"] == "BROWSER_PYTHON"
        assert _without_runtime_kind(browser_record) == _without_runtime_kind(json.loads(want["files"]["local_validation.json"]))
        manifest, native_manifest = got["manifest"], want["manifest"]
        for m in (manifest, native_manifest):
            for entry in m["files"]:  # the digests of exactly the files shown above to differ
                if entry["name"] in differing and entry["name"] != "manifest.json":
                    entry.pop("sha256")
                    entry.pop("bytes")  # "BROWSER_PYTHON" / "PYTHON_LOCAL" and the float noise change lengths
        assert manifest == native_manifest, name
    # (the extended emitter: no sampled ray is counted on the focal plane, a sampled zero; its beam-metric targets describe
    # one beam, which an extended emitter's independent rays are not: out of model scope, undecided)
    expected = {"reference": "UNKNOWN", "extended_emitter": "OUT_OF_MODEL_SCOPE",
                # a Project with configurations (each configuration compared as a run above)
                "zoom_configurations": "CONFIGURATIONS",
                # stated DOF limits (a drift UNMET) beside an out-of-scope metric target
                "retained_dof_limits": "FAIL",
                # four independent lasers on one plane: their pairs counted, the named pair's visibility 0
                "independent_sources": "PASS",
                "lens_shift_fail": "FAIL", "band_oos": "OUT_OF_MODEL_SCOPE", "refused_role": "REFUSED",
                "rotated_beam_y": "UNKNOWN", "folded": "UNKNOWN", "beam_expander": "UNKNOWN", "relay_4f": "UNKNOWN", "folded_expander": "UNKNOWN",
                "dichroic_routing": "PASS", "polarization": "PASS", "two_color_dichroic": "PASS", "dichroic_combiner": "PASS",
                # the interferometer's arms cross the catalog splitter's and fold mirrors' coatings, whose group delay the
                # catalog does not state: its visibility and coherence margin have no value, and its targets are those
                # it decides (fringe period, intensity balance, polarization overlap, coherence length)
                "two_beam_interference": "UNKNOWN",
                "interference_coherence_undeclared": "UNKNOWN", "partial_interception": "OUT_OF_MODEL_SCOPE",
                "aberration_dominated": "OUT_OF_MODEL_SCOPE", "edge_exit": "OUT_OF_MODEL_SCOPE",
                # Observation evidence now follows omitted light independently of optional thresholds.
                # These multi-surface assemblies exhaust the auxiliary trace-step budget: recorded per phase, a notice
                "declared_copy": "UNKNOWN", "declared_objective": "PASS", "declared_compound_unit": "PASS",
                "declared_thin_lens_slit": "OUT_OF_MODEL_SCOPE", "declared_axicon": "OUT_OF_MODEL_SCOPE",
                "declared_tabulated_thin_lens": "PASS", "declared_attenuator_ratings": "PASS",
                # the cavity's reflected branches count their splits: stopped at max_interactions_per_path (UNKNOWN)
                "native_splitter_cavity": "UNKNOWN",
                # every field of the F/1.2 objective arrives (the stop's severe truncation keeps the aggregate out of
                # scope); the monitor near the fold reads an interval and its minimum is not decided
                "geometric_power_basis": "OUT_OF_MODEL_SCOPE", "taken_before_fold": "UNKNOWN", "taken_after_fold": "UNKNOWN",
                # rays that do not follow a small beam's diffraction: the stop's power withheld; a small detector area
                # detects its pupil samples' share (the stop's severe truncation keeps both aggregates out of scope)
                "geometric_basis_withheld": "OUT_OF_MODEL_SCOPE", "geometric_area_share": "OUT_OF_MODEL_SCOPE",
                # fields whose chief rays fall beyond an opening go on (CLIPPED is a WARNING; the stop's severe
                # truncation keeps the aggregate out of scope)
                "promoted_reference": "OUT_OF_MODEL_SCOPE",
                # a singlet whose ghost paths the harness also reads (run_results kind "ghosts")
                "ghost_singlet": "UNKNOWN",
                # reached only by a ghost of a ghost: not a known zero
                "double_ghost": "UNKNOWN",
                # a beam's tail on a detector whose acceptance cone rejects it is a known zero: its target is met and the
                # known-dark plane is PASS (its finding a notice); inside the cone it may arrive
                "tail_outside_acceptance": "PASS", "tail_inside_acceptance": "UNKNOWN",
                # elliptical source, angle-resolved coating with tabulated phases, banded waveplate and polarizer
                "coatings_phases_polarization": "PASS",
                # oblique exposures and a turned rotary attenuator; a rated window only untraced light reaches
                "rated_oblique_window_rotary_nd": "PASS", "rated_untraced_transmission": "UNKNOWN",
                # ratings stating a test beam (smaller, larger, between the readings) or a pulse count, none of them
                # pulses per tested site: not verified, not decided
                "rated_test_beam_and_pulse_count": "UNKNOWN",
                # a rated window only the light left out as negligible of a pulsed line reaches: within the value,
                # but the rating states no pulse count, so it is not verified
                "rated_left_out_light_only": "UNKNOWN",
                # three bare windows bounded by the alternating series of their reflections; the light left out
                # as negligible, of unknown size and state, leaves the rating's conditions not verified
                "rated_window_stack": "UNKNOWN",
                # per-site pulse counts of two trains and the rated-value readings: a single-pulse peak value and a
                # 50-on-1 value stated as the total over the 1/e^2 area (threshold twice it), both at most rated,
                # are exceeded by each pulse; the others are not verified
                "rated_pulse_counts_and_value_readings": "FAIL",
                # an elliptical source in that interferometer: overlap and phase difference of complex fields (its
                # coherent power UNKNOWN: the arms' degree of coherence is not known through the coatings)
                "coherent_elliptical_interference": "UNKNOWN",
                # PV-F1: oblique observation planes, the wedge prism, whole benches turned about an oblique axis
                "oblique_plane_free_beam": "PASS", "wedge_prism_two_lines": "UNKNOWN",
                "turned_rated_oblique_window_rotary_nd": "PASS", "turned_coatings_phases_polarization": "PASS",
                "turned_coherent_elliptical_interference": "UNKNOWN",
                # a line focus partly captured by a pinhole / slit (INCOMPLETE_ENVELOPE notices)
                "cylinder_line_focus": "UNKNOWN", "turned_cylinder_line_focus": "UNKNOWN",
                # a tabulated thin lens outside its table's wavelengths / beyond its field angle (the spot
                # behind it INDETERMINATE, a plane before it and one only a beam beside it reaches still decided)
                "faraday_isolator": "PASS", "faraday_outside_table": "OUT_OF_MODEL_SCOPE", "faraday_before_a_detector": "PASS",
                "grating_without_response": "UNKNOWN", "grating_with_response": "PASS", "grating_before_a_detector": "UNKNOWN",
                "tabulated_lens_outside_band": "OUT_OF_MODEL_SCOPE", "tabulated_lens_beyond_field_angle": "OUT_OF_MODEL_SCOPE",
                "tabulated_lens_beside_and_before": "OUT_OF_MODEL_SCOPE",
                # coatings with group-delay bounds: the visibility target is met over its whole range, the
                # coherence-factor minimum 0.9999 lies inside the degree of coherence's range (not decided)
                "interference_bounded_coatings": "UNKNOWN",
                # lossless test-only Project copies isolate coating-phase physics (open UNKNOWN; stated
                # dark fringe FAIL). Untouched public parts leave coherent power UNKNOWN from bulk loss.
                "michelson_open_phase": "UNKNOWN", "michelson_stated_phases": "FAIL", "michelson_public_unstated_bulk": "UNKNOWN",
                # Public parts' route geometry/delay stay available, but the unstated glass absorption withholds
                # their numeric power and hence the overall status.
                "michelson_partially_polarized_delays": "UNKNOWN",
                # a plane the tail or a ghost reaches is not decided on its bound; a plane through a hit point reads
                # the arrival, and says so. The test-only lossless copy retains its plane's power bound
                # while a refused coating use withholds surface exposure bounds.
                "tail_energy_flow": "UNKNOWN", "ghost_on_a_plane": "PASS", "coincident_normal_mirror": "PASS",
                "coincident_splitter": "WARNING", "coating_use_refused": "PASS",
                "triangle_stop": "PASS", "window_hole": "PASS", "lens_array": "PASS",
                # a dual-wavelength retarder turned by its rotation parameter, then a circular polarizer
                "polarization_families": "PASS",
                # (the scattered light the run does not trace: a notice, its bound at the planes)
                "scattering_plate": "PASS", "complex_transmission_sample": "OUT_OF_MODEL_SCOPE",
                # a Project-declared calcite displacer lit by a partially polarized continuous spectrum
                "crystal_displacer_continuous_spectrum": "PASS",
                # a catalog polarizing cube's internal folds bounded (the plane's bound reads them); the
                # cube's rating does not state the light's polarization: not verified, not decided
                "solid_fold_pbs_cube": "UNKNOWN",
                # a corner cube declared as a facet solid: its six faces traced as planes
                "facet_solid_corner_cube": "PASS",
                # a fluorescence sample excited by a chopped laser, read by a detector and a lock-in
                "weak_signal_chain": "PASS",
                # a sampled intensity map and a sampled field (its field propagation read too)
                "sampled_intensity_map": "PASS", "field_sampled_tilt": "PASS",
                # a d-tensor BBO doubler on a bench turned about an oblique axis: both targets met; the harmonic
                # walks off its Gaussian state (HARMONIC_BEAM_NOT_GAUSSIAN, its power known)
                "shg_d_tensor_turned": "OUT_OF_MODEL_SCOPE",
                # an axial gradient rod with spherical faces and a radial rod with a slanted face, turned
                "grin_axial_curved_turned": "PASS", "grin_slanted_face_turned": "PASS",
                # the same two turned about design Z (runtime-identical placement of the samples)
                "sampled_intensity_map_turned": "PASS", "field_sampled_tilt_turned": "PASS"}
    assert set(expected) == set(documents)  # every case has its expected status (each part checks its own)
    assert statuses == {name: expected[name] for name in names}


@pytest.mark.pyodide
def test_the_browser_reports_the_plane_states_the_marker_and_the_withheld_bound(pyodide_run):
    """Partial-beam residuals and ring returns, as the browser reads them (compared with native bit for bit in
    ``test_the_browser_json_api_is_semantically_identical_to_native``). The tail's flux by independent integration is 2.71e-7 W
    (``test_tail_dark_planes_and_surface_coincidence``)."""
    cases = _browser_cases(pyodide_run, "test_the_browser_reports_the_plane_states_the_marker_and_the_withheld_bound")

    def summary(name: str) -> dict:
        return cases[name]["summary"]

    # (a plane only followed light reaches: BOUNDED from 0 with no value; a maximum at or above the bound is met on it, one
    # below it is not decided)
    for name, target, assessment in (("tail_energy_flow", "side", "INDETERMINATE"), ("ghost_on_a_plane", "ghost", "MET")):
        (row,) = summary(name)["observations"]
        assert row["quantity_state"] == "BOUNDED" and row["total_power_W"] is None and row["untraced_power_W"] > 0.0
        assert row["power_lower_bound_W"] == 0.0 and row["power_upper_bound_W"] is not None and row["status"] == "PASS"
        (outcome,) = summary(name)["targets"]
        assert (outcome["target_id"], outcome["assessment"], outcome["value"]) == (target, assessment, None)
    (row,) = summary("ghost_on_a_plane")["observations"]
    assert row["power_upper_bound_W"] <= 1e-3
    (row,) = summary("tail_energy_flow")["observations"]
    assert 2.71e-7 <= row["power_upper_bound_W"] <= 2.76e-7

    (row,) = summary("coating_use_refused")["observations"]
    assert row["quantity_state"] == "BOUNDED" and row["power_upper_bound_W"] is not None  # a refusal leaves a plane's bound on the record's value
    (row,) = summary("coincident_normal_mirror")["observations"]
    assert [(a["crossing_index"], a["surface_coincidence"]["surface_id"], a["surface_coincidence"]["role"]) for a in row["arrivals"]] == [
        (0, "M/S1", "ARRIVING"), (1, "M/S1", "RETURNING")]
    assert "SURFACE_COINCIDENT_ARRIVAL" in summary("coincident_normal_mirror")["targets"][0]["issue_codes"]
    (row,) = summary("coincident_splitter")["observations"]
    (arrival,) = row["arrivals"]
    assert arrival["power_W"] == pytest.approx(1e-3) and arrival["surface_coincidence"]["role"] == "ARRIVING"
    assert arrival["surface_coincidence"]["surface_event"] == "REFLECTIVE"  # (a splitter: the chief ray leaves on the side it came from)
    assert arrival["surface_coincidence"]["crossing_power_upper_bound_W"] == pytest.approx(0.5e-3, rel=1e-3)  # half the beam at 45 deg

    refused = [i for i in summary("coating_use_refused")["issues"] if i["code"] == "EXPOSURE_BOUND_WITHHELD"
               and i["numerical_context"]["cause"] == "COATING_USE_REFUSED"]
    assert {i["location"] for i in refused} == {"surface:L1/S1", "surface:L1/S2", "surface:L2/S1", "surface:L2/S2"}
    assert {i["status"] for i in refused} == {"NOT_APPLICABLE"}


@pytest.mark.pyodide
def test_the_browser_reports_the_open_phase_range_and_the_stated_phase_value(pyodide_run):
    """The open phase of the incoherent sum, as the browser reads them: the range and its statement where a coating states no phase, the value where they state
    theirs (compared with native bit for bit in ``test_the_browser_json_api_is_semantically_identical_to_native``)."""
    import base64

    from michelson_helpers import expected

    want = expected(json.loads(REAL_CATALOG.read_text(encoding="utf-8")))
    cases = _browser_cases(pyodide_run, "test_the_browser_reports_the_open_phase_range_and_the_stated_phase_value")

    def parts(name: str) -> tuple[dict, dict, dict]:
        (row,) = cases[name]["summary"]["observations"]
        document = json.loads(base64.b64decode(cases[name]["files"]["simulation_result.json"]["base64"]))
        (pair,) = document["interference"]
        return row, row["superpositions"][0], pair

    public = cases["michelson_public_unstated_bulk"]
    (public_row,) = public["summary"]["observations"]
    public_result = json.loads(base64.b64decode(public["files"]["simulation_result.json"]["base64"]))
    assert public["status"] == "UNKNOWN" and public_row["quantity_state"] == "UNKNOWN"
    assert public_row["total_power_W"] is None and public_row.get("superpositions", []) == []
    assert public_result["interference"] == []
    assert "BULK_LOSS_NOT_ESTABLISHED" in {i["code"] for i in public["summary"]["issues"]}
    assert len(public_row["arrivals"]) == 2
    public_bounds = sorted(a["power_upper_bound_W"] for a in public_row["arrivals"])
    assert public_bounds == pytest.approx(sorted((want.power_t_W, want.power_r_W)), rel=1e-9)

    row, superposition, pair = parts("michelson_open_phase")
    assert row["total_power_W"] == pytest.approx(want.sum_W, rel=1e-9)
    # (the coherent sum at the trace's default interface phases: an estimate whose entry names the passes that state no
    # phase, inside the range over every phase)
    assert "PHASE_UNDEFINED" not in superposition["issue_codes"] and superposition.get("phase_statement") is None
    assert superposition["total_power_W"] == pytest.approx(pair["combined_power_collinear_W"], rel=1e-8, abs=1e-12)
    assert superposition["total_power_min_W"] <= superposition["total_power_W"] <= superposition["total_power_max_W"]
    entry = superposition["value_basis"]["total_power_W"]
    assert entry["kind"] == "ESTIMATE"
    assert {a["input"]: a.get("at") for a in entry["assumed"]}["UNSTATED_PHASE_AS_INTERFACE"] == sorted(want.open_passes)
    assert superposition["total_power_min_W"] == pytest.approx(want.sum_W - want.largest_cross_term_W, abs=1e-10)
    assert superposition["total_power_max_W"] == pytest.approx(want.sum_W + want.largest_cross_term_W, rel=1e-8)
    assert pair["combined_power_collinear_min_W"] == pytest.approx(want.sum_W - want.largest_cross_term_W, abs=1e-10)
    assert pair["combined_power_collinear_max_W"] == pytest.approx(want.sum_W + want.largest_cross_term_W, rel=1e-8)
    assert pair.get("phase_statement") is None and pair["value_basis"]["combined_power_collinear_W"]["kind"] == "ESTIMATE"

    row, superposition, pair = parts("michelson_stated_phases")
    assert row["total_power_W"] == pytest.approx(want.sum_W, rel=1e-9)
    assert pair["combined_power_collinear_W"] < 2e-3 * want.sum_W and "phase_statement" not in pair
    assert superposition["total_power_W"] == pytest.approx(pair["combined_power_collinear_W"], rel=1e-5) and "phase_statement" not in superposition


@pytest.mark.pyodide
def test_the_other_json_api_entry_points_in_pyodide_are_identical_to_native(pyodide_run):
    """Prescription import, the native component library, exact sections, catalog_model and the
    optical-solid inspection give the same reply in the browser runtime as in native Python (byte-identical JSON)."""
    from os_optics.local import json_api

    assert json_api.install_catalog(REAL_CATALOG.read_bytes())["ok"] is True
    browser = json.loads((pyodide_run / "_json_api_calls.json").read_text(encoding="utf-8"))
    calls = json_api_calls()
    assert set(browser) == set(calls)
    def digest(value):  # (a reply's bytes -- a layout's GLB -- by their SHA-256, as the browser side writes them)
        if isinstance(value, (bytes, bytearray)):
            return {"sha256": hashlib.sha256(bytes(value)).hexdigest(), "bytes": len(value)}
        raise TypeError(type(value).__name__)

    for name, (function, args) in calls.items():
        native = json.loads(json.dumps(getattr(json_api, function)(*args), default=digest))
        if name.startswith("field_"):
            _close(browser[name], native, name)
        else:
            assert browser[name] == native, name
    # the browser runtime's own reading of the worked chain (test_modulator_chain derives the numbers)
    assert browser["field_ramp"]["status"] == "OK" and browser["field_above_limit"]["status"] == "PARTIAL"
    assert browser["field_deflector_tones"]["status"] == "OK" and browser["field_deflector_tones"]["spots"][0]["u_mm"] == pytest.approx(6.55385, rel=1e-5)
    assert browser["field_from_file"]["power"]["after_mask_W"] == pytest.approx(0.25e-3, rel=1e-9)
    assert browser["field_above_limit"]["quantities"]["focal_irradiance"]["reason"] == "NA_ABOVE_SCALAR_LIMIT"
    assert browser["field_tones"]["composition"] == "INCOHERENT_SUM" and browser["field_tones"]["array"]["spot_count"] == 3
    assert browser["field_spot"]["power"]["window_W"] + browser["field_spot"]["power"]["higher_orders_W"] == pytest.approx(1e-9, rel=1e-9)
    camera = browser["field_detector"]
    assert camera["status"] == "OK" and camera["unavailable_quantities"] == [] and camera["quantities"]["snr"]["values"][4][4] == pytest.approx(134.5, rel=0.02)
    assert "photons" in browser["field_detector_partial"]["unavailable_quantities"]
    refused = {"import_refused_label", "import_refused_units", "native_refused", "check_solids_invalid", "native_convex_mirror_refused", "field_refused", "field_detector_refused", "field_focal_refused",
               "network_selected_out_of_band", "interop_unknown", "paraxial_refused", "cavity_refused", *COMPOSITION_REFUSED, *PROJECT_UPDATE_REFUSED,
               *EVALUATION_REFUSED}
    assert not any(browser[name]["ok"] for name in refused)
    # (a transform that makes two bodies overlap is applied, and its check names the overlap)
    check = browser["ws_transform_overlap"]["result"]["check"]
    assert (check["ok"], [p["code"] for p in check["new_problems"]]) == (False, ["OPTICAL_BODIES_OVERLAP"])
    assert all(browser[name]["ok"] for name in calls if name not in refused and not name.startswith("check_tilted"))
    assert browser["solids_invalid"]["status"] == "INVALID" and browser["section_import_xz_svg"]["svg"].startswith("<svg")
    assert browser["import_refused_label"]["error"]["problems"][0]["code"] == "VALUE_INVALID"
    assert browser["import_refused_units"]["error"]["problems"][0]["code"] == "UNITS_CONFLICT"
    # (an answered analysis completed, the unstable cavity too: its quantities outside the model say so themselves)
    assert browser["cavity_stable"]["execution"]["outcome"] == "COMPLETED" and browser["cavity_unstable"]["execution"]["outcome"] == "COMPLETED"
    assert browser["cavity_refused"]["error"]["code"] == "CAVITY_ANALYSIS_INVALID"
    assert browser["cavity_injected"]["arrays"]["injection"]["power_budget"]["injected_W"] == 1.5
    assert browser["cavity_source"]["arrays"]["injection"]["traced_arrival"]["arriving_W"] == 2.0
    # (the browser runtime's selective evaluations: what each computed, decided and refused)
    assert browser["evaluate_readings"]["targets"] == [] and browser["evaluate_readings"]["monitors"][0]["quantity_state"] == "NOT_COMPUTED"
    assert {t["target_id"]: t["assessment"] for t in browser["evaluate_requirements"]["targets"]} == {"focus_position": "MET", "spot_radius": "MET"}
    assert [t["target_id"] for t in browser["evaluate_one_target"]["evaluation"]["targets_not_evaluated"]] == ["focus_position"]
    assert [t["assessment"] for t in browser["evaluate_out_of_band"]["targets"]] == ["INDETERMINATE"]
    assert [i["code"] for i in browser["evaluate_out_of_band"]["model_scope_issues"]] == ["MODEL_WAVELENGTH_OUTSIDE_BAND"]
    view = browser["evaluate_view"]
    assert view["optical_scene"]["engine_run"]["result_hash"]["digest"] == view["identity"]["result_hash"] and not view["evaluation"]["complete"]
    assert browser["evaluate_excluded_refused"]["error"]["code"] == "EXCLUDED_CAPABILITY_REQUIRED"
    assert browser["evaluate_unknown_target_refused"]["error"]["code"] == "UNKNOWN_TARGET"


@pytest.mark.pyodide
def test_prepared_evaluation_handles_in_pyodide_are_identical_to_native(pyodide_run):
    """The prepared evaluation handles live in the module state of the browser runtime between calls, as in native Python:
    the same sequence of calls gives the same results in both (handle ids masked), and in the browser a reply through a
    handle is the fresh ``evaluate_project`` reply with the handle's keys, the least recently used handle is evicted with
    its notice, and an evicted or released handle is refused with its reason."""
    from os_optics.local import json_api
    from os_optics.local.prepared import PREPARED_EVALUATIONS

    assert json_api.install_catalog(REAL_CATALOG.read_bytes())["ok"] is True
    browser = json.loads((pyodide_run / "_prepared.json").read_text(encoding="utf-8"))
    namespace: dict = {}
    exec(PREPARED_SEQUENCE, namespace)
    PREPARED_EVALUATIONS.clear()
    try:
        native = namespace["sequence"](json_api, prepared_documents())
    finally:
        PREPARED_EVALUATIONS.clear()
    assert json.dumps(browser, sort_keys=True) == json.dumps(native, sort_keys=True)

    def without_handle(reply: dict) -> dict:
        return {k: v for k, v in reply.items() if k not in {"handle", "reuse", "notices"}}

    assert without_handle(browser["requirements"]) == browser["fresh_requirements"]
    assert without_handle(browser["again"]) == browser["fresh_requirements"] and "beam_trace" in browser["again"]["reuse"]["reused"]
    assert without_handle(browser["candidate"]) == browser["fresh_candidate"]
    assert without_handle(browser["view"]) == browser["fresh_view"] and "optical_scene" in browser["view"]
    assert [(n["code"], n["handle"]) for n in browser["prepare_c"]["notices"]] == [("EVALUATION_HANDLE_EVICTED", "<B>")]
    assert browser["evicted"]["error"]["problems"][0]["context"] == {"handle": "<B>", "reason": "EVICTED"}
    assert browser["released"]["error"]["problems"][0]["context"] == {"handle": "<A>", "reason": "RELEASED"}
    assert browser["cancel_idle"]["cancelled"] is False and [row["handle"] for row in browser["listing"]["handles"]] == ["<C>"]


@pytest.mark.pyodide
def test_the_network_export_in_pyodide_has_the_native_contract_hash(pyodide_run):
    """Acceptance 3 of the network export: the same input exported in Pyodide and in native Python gives the identical document (every number
    quantized as declared), hence the identical contract hash -- for Engine requests here, and for every json_api Project case in
    ``test_the_browser_json_api_is_semantically_identical_to_native``."""
    from os_optics.local import json_api

    assert json_api.install_catalog(REAL_CATALOG.read_bytes())["ok"] is True
    browser = json.loads((pyodide_run / "_network.json").read_text(encoding="utf-8"))
    codes = {}
    for name, document in network_cases().items():
        native = json_api.linear_optical_network(document)
        assert browser[name] == native, name
        codes[name] = native["contract_hash"] if native["ok"] else native["error"]["problems"][0]["code"]
    assert codes["multi_laser"] == "MIXED_WAVELENGTHS" and all(len(codes[n]) == 64 for n in ("mzi_window", "hwp_pbs", "michelson", "hom",
                                                                                           "michelson_both_ports"))
    assert (codes["gain"], codes["pbs_normal_incidence"]) == ("NETWORK_NOT_PASSIVE", "POLARIZING_NORMAL_INCIDENCE")


def _masked_assembly_result(result: dict) -> dict:
    result = json.loads(json.dumps(result))
    result["checker"].pop("runtime_kind")
    result.pop("result_hash")
    return result


@pytest.mark.pyodide
def test_the_assembly_check_in_pyodide_is_identical_to_native(pyodide_run):
    """The local assembly layout / collision check (json_api.validate_assembly_package) of the reference Assembly
    Preview Package in Pyodide, given the archive as bytes, as a JS Uint8Array (a memoryview) and as a file map.

    The checker is pure Python arithmetic (+, -, *, /, sqrt: IEEE-754 correctly rounded on both runtimes), so the whole
    result equals the native one bit for bit; the only runtime-dependent values are ``checker.runtime_kind``
    (BROWSER_PYTHON vs PYTHON_LOCAL) and therefore ``result_hash``."""
    import base64

    from os_optics._vendor.shared_contracts.assembly_preview import AssemblyLocalValidationV1, json_file_bytes
    from os_optics.local import json_api

    browser = json.loads((pyodide_run / "_assembly.json").read_text(encoding="utf-8"))
    native = json_api.validate_assembly_package(ASSEMBLY_PACKAGE.read_bytes())
    assert native["ok"] and native["status"] == "PASS"
    native_result = native["result"]
    assert native_result["checker"]["runtime_kind"] == "PYTHON_LOCAL"
    assert sorted(browser) == ["bytes", "file_map", "memoryview"]
    for label, reply in browser.items():
        assert reply["ok"] is True, (label, reply)
        result = reply["result"]
        assert result["checker"]["runtime_kind"] == "BROWSER_PYTHON", label
        assert reply["status"] == result["status"] == native["status"] == "PASS", label
        assert result["summary"]["interference"] == native_result["summary"]["interference"] == "NO_INTERFERENCE"
        assert [(c["check"], c["status"]) for c in result["checks"]] == [(c["check"], c["status"]) for c in native_result["checks"]], label
        assert [(i["code"], i["severity"]) for i in result["issues"]] == [(i["code"], i["severity"]) for i in native_result["issues"]], label
        assert [(p["part_ref"], p["status"]) for p in result["parts"]] == [(p["part_ref"], p["status"]) for p in native_result["parts"]], label
        assert result["checker"]["engine_build_id"] == native_result["checker"]["engine_build_id"] == engine_build_id()
        # everything else, including every float, is identical (JSON round-trips floats exactly)
        assert _masked_assembly_result(result) == _masked_assembly_result(native_result), label
        assert result["result_hash"] != native_result["result_hash"]  # it covers runtime_kind
        browser_model = AssemblyLocalValidationV1.model_validate(result)  # result_hash verifies against the browser document
        # the canonical file bytes differ only in runtime_kind and result_hash
        data = base64.b64decode(reply["result_file"]["base64"])
        assert data == json_file_bytes(browser_model.payload()), label
        swapped = data.replace(b'"runtime_kind": "BROWSER_PYTHON"', b'"runtime_kind": "PYTHON_LOCAL"')
        swapped = swapped.replace(result["result_hash"].encode(), native_result["result_hash"].encode())
        assert swapped == native["result_file"], label
    report = json.loads((pyodide_run / "_report.json").read_text(encoding="utf-8"))
    assert "assembly_validation" in report["timings_ms"]


@pytest.mark.pyodide
def test_the_assembly_drawing_in_pyodide_is_byte_identical_to_native(pyodide_run):
    """The local Detailed Assembly Drawing (json_api.generate_assembly_drawing) of the reference pair in Pyodide.

    The generator is pure Python arithmetic (+, -, *, /, sqrt), literal circle / view tables and fixed-decimal formatting,
    so the SVG and PDF bytes equal the native ones exactly. The result and manifest differ only in the runtime kind of
    the local assembly validation they record (BROWSER_PYTHON vs PYTHON_LOCAL) and therefore in their hashes."""
    import base64

    from os_optics._vendor.shared_contracts.assembly_detail import AssemblyDrawingResultV1, verify_drawing_outputs
    from os_optics.local import json_api

    browser = json.loads((pyodide_run / "_drawing.json").read_text(encoding="utf-8"))
    native = json_api.generate_assembly_drawing(DRAWING_PACKAGE.read_bytes(), DRAWING_DETAIL.read_bytes())
    assert browser["ok"] is True and native["ok"] is True
    assert browser["drawing_status"] == native["drawing_status"] == "DETAILED_ASSEMBLY_DRAWING" and browser["delivery_ready"] is True
    for name in ("assembly_drawing.svg", "assembly_drawing.pdf"):
        assert browser["files"][name]["sha256"] == hashlib.sha256(native["files"][name]).hexdigest(), name
    assert browser["result"]["validation"]["runtime_kind"] == "BROWSER_PYTHON"
    assert native["result"]["validation"]["runtime_kind"] == "PYTHON_LOCAL"

    def masked(doc: dict) -> dict:
        doc = json.loads(json.dumps(doc))
        doc["validation"].pop("runtime_kind")
        doc["validation"].pop("result_hash")
        doc.pop("result_hash")
        return doc

    assert masked(browser["result"]) == masked(native["result"])
    assert browser["result"]["generator"]["engine_build_id"] == engine_build_id()
    AssemblyDrawingResultV1.model_validate(browser["result"])
    files = {k: base64.b64decode(v["base64"]) for k, v in browser["files"].items()}
    assert verify_drawing_outputs(files) == ()
    report = json.loads((pyodide_run / "_report.json").read_text(encoding="utf-8"))
    assert "assembly_drawing" in report["timings_ms"] and len(browser["ms"]) == 2


@pytest.mark.pyodide
def test_the_multi_chain_assembly_drawing_in_pyodide_is_byte_identical_to_native(pyodide_run):
    """The same for the four-chain pair: the sheet choice, merged and ordinate dimensions, stacked balloons and text
    columns are pure arithmetic too, so its A2 SVG and PDF equal the native bytes."""

    browser = json.loads((pyodide_run / "_drawing_2.json").read_text(encoding="utf-8"))
    from os_optics.local import json_api

    native = json_api.generate_assembly_drawing(MULTI_DRAWING_PACKAGE.read_bytes(), MULTI_DRAWING_DETAIL.read_bytes())
    assert browser["ok"] is True and native["ok"] is True and browser["delivery_ready"] is True
    assert browser["result"]["sheet"] == native["result"]["sheet"] == "A2_LANDSCAPE_594x420_MM"
    for name in ("assembly_drawing.svg", "assembly_drawing.pdf"):
        assert browser["files"][name]["sha256"] == hashlib.sha256(native["files"][name]).hexdigest(), name


@pytest.mark.pyodide
def test_a_drawing_with_a_customer_supplied_component_is_byte_identical_in_pyodide(pyodide_run):
    """The customer-supplied component's dashed design-position mark, its table row and its placement step
    are the same arithmetic and templates in the browser runtime: the SVG and PDF equal the native bytes."""

    browser = json.loads((pyodide_run / "_drawing_3.json").read_text(encoding="utf-8"))
    from os_optics.local import json_api

    native = json_api.generate_assembly_drawing(CUSTOMER_DRAWING_PACKAGE.read_bytes(), CUSTOMER_DRAWING_DETAIL.read_bytes(),
                                                {"locale": "zh-CN"})  # the Chinese sheet: CJK widths, wrapping and PDF text
    assert browser["ok"] is True and native["ok"] is True and browser["delivery_ready"] is True
    assert browser["locale"] == native["locale"] == "zh-CN"
    rows = browser["result"]["customer_supplied_components"]
    assert [r["design_instance_id"] for r in rows] == [r["design_instance_id"] for r in native["result"]["customer_supplied_components"]]
    assert "P1" in [r["design_instance_id"] for r in rows]
    for name in ("assembly_drawing.svg", "assembly_drawing.pdf"):
        assert browser["files"][name]["sha256"] == hashlib.sha256(native["files"][name]).hexdigest(), name


@pytest.mark.pyodide
def test_the_optical_layout_drawing_in_pyodide_is_byte_identical_to_native(pyodide_run):
    """The Optical Layout Drawing (json_api.generate_optical_layout, zh-CN) of the synthetic folded layout:
    hull, angle and text arithmetic give the same SVG, PDF and manifest bytes in the browser runtime."""

    from layout_helpers import layout_package

    from os_optics.local import json_api

    browser = json.loads((pyodide_run / "_layout.json").read_text(encoding="utf-8"))
    native = json_api.generate_optical_layout(layout_package().model_dump(mode="json", by_alias=True), "zh-CN")
    assert browser["ok"] is True and native["ok"] is True and browser["locale"] == "zh-CN"
    for name in ("optical_layout.svg", "optical_layout.pdf", "optical_layout_manifest.json"):
        assert browser["files"][name]["sha256"] == hashlib.sha256(native["files"][name]).hexdigest(), name
