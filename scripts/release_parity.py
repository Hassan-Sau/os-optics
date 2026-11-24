"""Release-time parity checks of an INSTALLED os-optics: the recorded retained-DOF, plate and 32-line-source scenarios.

The same checks run against the repository's sources (``tests/test_release_parity_script.py``), against a candidate wheel
or sdist installed into a clean virtual environment, and against the release published on PyPI, so the published build can
be shown to behave as the source did. Every expected number is a closed form computed here (Fresnel and Beer-Lambert, the
optical path through a plate, the grating equation, the error function of a Gaussian spectrum, the narrow-band coherence
length, the stated DOF limits) -- never a number read from an Engine.

Uses nothing but the installed package and the standard library (no repository import, no test helper). Run it with the
Python of the environment under test::

    python -m venv parity && parity/bin/python -m pip install os-optics==0.13.0      # or a wheel / sdist file
    parity/bin/python -I scripts/release_parity.py --expect-version 0.13.0 --out parity.json

``--catalog`` is the catalog document the retained-DOF Project's flat mirror comes from (default: the repository's frozen test catalog
``tests/fixtures/catalog/test_optical_catalog.v2.json``, release 2026-01-01.1, next to this script). The script checks that
``os_optics`` is imported from the environment's site-packages (``--allow-source`` skips that, for the in-repository test),
prints one JSON evidence document (also written to ``--out``) and exits 0 only if every check passed.

The scenarios:

* retained_dof: a catalog flat mirror folding a beam, with retained ``Tx``, ``Rx`` and ``Ry``. Without limits the Project keeps
  the hash the published 0.12.0 gives it; stated travel / hard limits / repeatability / drift are accepted, recorded per DOF
  in the run (record, manifest, ``results describe``) and assessed by the ``dof_*`` targets (MET / UNMET / INDETERMINATE,
  values from the stated limits); contradictory limits and the field names a recorded probe tried are refused; the
  requirements exit code reads the DOF targets.
* plate: the recorded homogeneous plate (index 1.6, 0.01 / mm, 5 mm, 632.8 nm, ``plate.window``): 0.852612768013 mW
  and an optical path of 203 mm; a thin complex-transmission sample behind it (amplitude 0.9, phase 0.5 rad) scales the
  power by 0.81 and adds 0.5 lambda / (2 pi); a sampled linear phase map steers the beam by the grating equation; the
  recorded ``transmission_phase_map`` field and an amplitude above 1 stay refused.
* spectral_lines: the recorded 32-line source (500-700 nm, 1 mW / 32 per line) stays discrete (32 entries, no spectral density
  in the Engine request, labelled DISCRETE beside a continuous source) and keeps its 0.12.0 hash; a continuous Gaussian
  spectrum is traced as its quadrature lines with the error-function bin powers and the spectrum's coherence length; the
  time-domain inputs are refused with the static approximation named.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
import traceback
from importlib import metadata
from pathlib import Path

#: Project content hashes (``load_project(...).content_hash()``) of the Projects that use none of the 0.13.0 fields, as the
#: published os-optics 0.12.0 computes them (recorded 2026-11-07 by this script in a clean environment with
#: ``os-optics==0.12.0`` from PyPI). A build that adds fields must keep these.
HASHES_0_12_0 = {
    "retained_dof_control": "c1c500cefdd0cb83d83dc8f3b4025b593cae700fbadc91a688c91daa3d5506e6",
    "plate": "399f1278cf30cd401a4ba47b625481045ef3e6166228f11a76054b9af43bd5b2",
    "spectral_lines_32": "976ade6c1b2a1e211bf511bb31c6f373dff5052357e0adf41dabc40e4902d2f3",
}

POWER_W = 1e-3
DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "catalog" / "test_optical_catalog.v2.json"
TEST_CATALOG_VERSION = "2026-01-01.1"
FLAT_MIRROR = "OS-OPT-TEST-MIR-FLAT"
SQRT_HALF = math.sqrt(0.5)
#: Local +Z -> design +X (local +Y -> design +Z); local +Z -> design +Y; a face normal that folds +X into +Y.
ALONG_X = (0.5, 0.5, 0.5, 0.5)
ALONG_Y = (0.0, 0.0, SQRT_HALF, SQRT_HALF)
FOLD_A = math.sqrt((2.0 - math.sqrt(2.0)) / 8.0)
FOLD_B = math.sqrt((2.0 + math.sqrt(2.0)) / 8.0)
FOLD_X_TO_Y = (FOLD_A, FOLD_A, -FOLD_B, -FOLD_B)


# ------------------------------------------------------------------------------------------------------ Project documents
def source(instance_id: str = "SRC", position=(-200.0, 0.0, 0.0), *, wavelength_nm: float = 780.0, waist_mm: float = 1.0) -> dict:
    return {"instance_id": instance_id, "optical_role": "source",
            "custom_part": {"custom_part_key": "test-laser", "description": "Test laser."},
            "nominal_pose": {"position_mm": list(position), "quaternion_wxyz": list(ALONG_X)},
            "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": wavelength_nm, "power_w": POWER_W, "waist_radius_mm": waist_mm,
                             "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}


def plane(observation_id: str, position, quaternion=ALONG_X, *, radius_mm: float = 12.5) -> dict:
    return {"observation_id": observation_id, "pose": {"position_mm": list(position), "quaternion_wxyz": list(quaternion)},
            "aperture": {"shape": "circle", "radius_mm": radius_mm}}


def project(components: list[dict], observations: list[dict], targets: list[dict] | None = None) -> dict:
    return {"schema": "opensci.project", "project_schema_version": "2.0", "components": components, "observations": observations,
            "targets": targets or []}


def declared(instance_id: str, role: str, model_id: str, position) -> dict:
    return {"instance_id": instance_id, "optical_role": role, "custom_part": {"custom_part_key": f"own-{instance_id.lower()}",
                                                                              "description": "My own optic."},
            "optical_model_id": model_id, "nominal_pose": {"position_mm": list(position), "quaternion_wxyz": list(ALONG_X)}}


# retained_dof: the stated limits (closed forms of the dof_* metrics below use exactly these numbers)
TX = {"axis": "Tx", "travel": {"min": -2.0, "max": 3.0}, "hard_limits": {"min": -2.5, "max": 3.5}, "repeatability": 0.002,
      "drift": {"per_kelvin": 0.0015, "per_hour": 0.0004}, "unit": "mm"}
RX = {"axis": "Rx", "travel": {"min": -4.0, "max": 4.0}, "repeatability": 0.01, "unit": "deg"}
RY = {"axis": "Ry"}
BARE = ({"axis": "Tx"}, {"axis": "Rx"}, {"axis": "Ry"})


def retained_dof_project(dofs=(TX, RX, RY), targets=()) -> dict:
    mirror = {"instance_id": "M1", "optical_role": "mirror", "product": {"optical_sku": FLAT_MIRROR, "catalog_version": TEST_CATALOG_VERSION},
              "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(FOLD_X_TO_Y)},
              "retained_dofs": [copy.deepcopy(d) for d in dofs]}
    return project([source(), mirror], [plane("OUT", (0.0, 100.0, 0.0), ALONG_Y, radius_mm=20.0)], list(targets))


def dof_target(target_id: str, metric: str, axis: str, *, device: str = "M1", **comparison) -> dict:
    parameters = comparison.pop("parameters", {})
    return {"target_id": target_id, "metric": metric, "device_instance_id": device, **comparison,
            "parameters": [{"key": k, "value": v} for k, v in {"axis": axis, **parameters}.items()]}


# plate: the recorded plate (plate.window parameters), and the thin sample in an aperture stop
PLATE_FIELDS = {"material": {"constant_index": "1.6", "valid_band_nm": ["400", "1100"], "absorption_coefficient_per_mm": "0.01"},
                "thickness_mm": "5", "diameter_mm": "25.4", "model_id": "plate"}
PLATE_WAVELENGTH_NM = 632.8


def plate_project(plate_model: dict, *, plate_at: float = 0.0, sample: dict | None = None, plane_at: float = 100.0) -> dict:
    components = [source(position=(-100.0, 0.0, 0.0), wavelength_nm=PLATE_WAVELENGTH_NM, waist_mm=0.5),
                  declared("PLATE", "window", "plate", (plate_at, 0.0, 0.0))]
    models = [plate_model["optical_model"]]
    if sample is not None:
        components.append(declared("CELL", "other_optical", "cell", (0.0, 0.0, 0.0)))
        models.append(stop_with_sample(sample))
    doc = project(components, [plane("D", (plane_at, 0.0, 0.0), radius_mm=5.0)])
    doc["optical_models"] = models
    doc["model_materials"] = plate_model["model_materials"]
    if plate_model.get("model_coatings"):
        doc["model_coatings"] = plate_model["model_coatings"]
    return doc


def stop_with_sample(sample: dict) -> dict:
    return {"model_id": "cell", "model": {"model_kind": "APERTURE_STOP", "model_version": "1", "status": "SPECIFIED",
                                          "opening_half_widths_mm": ["5", "5"], "edge_half_widths_mm": ["10", "10"], "sample": sample}}


def sample_only(sample: dict) -> dict:
    doc = project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=PLATE_WAVELENGTH_NM, waist_mm=0.5),
                   declared("CELL", "other_optical", "cell", (0.0, 0.0, 0.0))], [plane("D", (50.0, 0.0, 0.0), radius_mm=5.0)])
    doc["optical_models"] = [stop_with_sample(sample)]
    return doc


# spectral_lines: the recorded 32-line source, and a continuous Gaussian spectrum
def spectral_lines_project() -> dict:
    laser = source()
    del laser["source_model"]["wavelength_nm"], laser["source_model"]["power_w"]
    laser["source_model"]["spectral_lines"] = [{"wavelength_nm": 500.0 + 200.0 * k / 31, "power_w": POWER_W / 32} for k in range(32)]
    return laser


def continuous(spectral_density: dict, instance_id: str = "SRC", position=(-200.0, 0.0, 0.0)) -> dict:
    laser = source(instance_id, position)
    del laser["source_model"]["wavelength_nm"]
    laser["source_model"]["spectral_density"] = spectral_density
    return laser


def free_space(*sources: dict) -> dict:
    return project(list(sources), [plane("OUT", (100.0, 0.0, 0.0))])


# --------------------------------------------------------------------------------------------------------------- checks
class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, scenario: str, name: str, ok: bool, **detail) -> bool:
        self.rows.append({"scenario": scenario, "check": name, "ok": bool(ok), **({"detail": detail} if detail else {})})
        return bool(ok)

    def guard(self, scenario: str, name: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - a failure is evidence, the other checks still run
            self.check(scenario, name, False, error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stderr)
            return None


def close(a, b, *, rel: float = 0.0, abs_: float = 0.0) -> bool:
    return a is not None and b is not None and abs(a - b) <= max(rel * abs(b), abs_)


def cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "OS_OPTICS_STATUS_CONTRACT")}
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([sys.executable, "-m", "os_optics", *args], cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                          timeout=600)


def problems_of(reply: dict) -> list[dict]:
    return [] if reply.get("ok") else list(reply.get("error", {}).get("problems", []))


def project_hash(document: dict) -> str:
    from os_optics.local import load_project

    return load_project(document).content_hash()


def check_retained_dof(c: Checks, catalog: dict, work: Path, hashes: dict) -> None:
    from os_optics.local import json_api, run_optical_project

    scenario = "retained_dof"
    bare = retained_dof_project(BARE)
    hashes["retained_dof_control"] = c.guard(scenario, "control Project hash", lambda: project_hash(bare))
    if HASHES_0_12_0["retained_dof_control"] is not None and hashes["retained_dof_control"] is not None:
        c.check(scenario, "control Project without limits keeps its 0.12.0 hash", hashes["retained_dof_control"] == HASHES_0_12_0["retained_dof_control"],
                got=hashes["retained_dof_control"], published_0_12_0=HASHES_0_12_0["retained_dof_control"])

    def control_run():
        run = run_optical_project(bare, catalog=catalog)
        records = [d.model_dump(mode="json", exclude_none=True) for d in run.local_validation.retained_dofs]
        want = [{"instance_id": "M1", "control": "MANUAL", "axis": a} for a in ("Tx", "Rx", "Ry")]
        c.check(scenario, "control run records [Tx, Rx, Ry] without limits", records == want, records=records)
        return run

    control = c.guard(scenario, "control run", control_run)

    reply = c.guard(scenario, "check_project with stated limits", lambda: json_api.check_project(retained_dof_project(), catalog=catalog))
    if reply is not None:
        c.check(scenario, "stated travel, hard limits, repeatability and drift are accepted", reply.get("ok") is True,
                problems=problems_of(reply))

    def limits_run():
        out = work / "retained_dof_run"
        run = run_optical_project(retained_dof_project(), catalog=catalog, output_dir=out)
        want = [{"instance_id": "M1", "control": "MANUAL", **d} for d in (TX, RX, RY)]
        records = [d.model_dump(mode="json", exclude_none=True) for d in run.local_validation.retained_dofs]
        c.check(scenario, "the Local Validation record holds the stated limits per DOF", records == want, records=records)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        c.check(scenario, "manifest.json holds the stated limits per DOF", manifest.get("retained_dofs") == want, manifest=manifest.get("retained_dofs"))
        if control is not None:
            c.check(scenario, "stated limits change no optical result", run.result.observations == control.result.observations
                    and run.result.paths == control.result.paths)
        described = cli(["results", "describe", str(out)], work)
        dofs = json.loads(described.stdout).get("retained_dofs") if described.returncode == 0 else None
        c.check(scenario, "results describe lists the DOFs with their travel", described.returncode == 0 and dofs is not None
                and [d["axis"] for d in dofs] == ["Tx", "Rx", "Ry"] and dofs[0].get("travel") == {"min": -2.0, "max": 3.0},
                exit=described.returncode, retained_dofs=dofs, stderr=described.stderr[-400:])

    c.guard(scenario, "run with stated limits", limits_run)

    refused = [  # (DOF entry, a word of the refusal)
        ({"axis": "Tx", "travel": {"min": -1.0, "max": 1.0}, "unit": "deg"}, "is in mm, not deg"),
        ({"axis": "Rx", "travel": {"min": 0.5, "max": 1.0}}, "travel.min"),
        ({"axis": "Tx", "travel": {"min": -1.0, "max": 1.0}, "hard_limits": {"min": -0.5, "max": 2.0}}, "must contain the travel"),
        ({"axis": "Tx", "drift": {"per_kelvin": 0.001}}, "state travel with them"),
        ({"axis": "Tx", "travel": {"min": -1.0, "max": 1.0}, "repeatability": 0.0}, "repeatability"),
        # the field names a recorded probe tried stay unknown
        ({"axis": "Tx", "travel_min_mm": -1.0}, "travel_min_mm"),
        ({"axis": "Tx", "drift_um_per_K": 1.0}, "drift_um_per_K"),
        ({"axis": "Tx", "hard_limit_mm": 1.0}, "hard_limit_mm"),
    ]
    for dof, word in refused:
        reply = c.guard(scenario, f"refusal of {dof}", lambda dof=dof: json_api.check_project(retained_dof_project((dof,)), catalog=catalog))
        if reply is not None:
            found = problems_of(reply)
            c.check(scenario, f"contradictory or unknown DOF limit refused ({word})", reply.get("ok") is False and len(found) == 1
                    and found[0]["code"] == "PROJECT_INVALID" and word in found[0]["message"], problems=found)
    for where in ("component", "project"):
        doc = retained_dof_project()
        (doc["components"][1] if where == "component" else doc)["mechanical_limits"] = {"travel_mm": 1.0}
        reply = c.guard(scenario, f"{where} mechanical_limits", lambda doc=doc: json_api.check_project(doc, catalog=catalog))
        if reply is not None:
            c.check(scenario, f"{where} mechanical_limits stays an unknown field", reply.get("ok") is False
                    and any(p["code"] == "PROJECT_INVALID" and "mechanical_limits" in p["message"] for p in problems_of(reply)),
                    problems=problems_of(reply))

    targets = [
        dof_target("travel_inside", "dof_travel_margin_mm", "Tx", minimum=0.0, parameters={"required_min": -0.5, "required_max": 1.0}),
        dof_target("travel_beyond", "dof_travel_margin_mm", "Tx", minimum=0.0, parameters={"required_max": 3.25}),
        dof_target("tilt_inside", "dof_travel_margin_deg", "Rx", minimum=1.0, parameters={"required_min": -2.0, "required_max": 2.0}),
        dof_target("drift", "dof_drift_mm", "Tx", maximum=0.02, parameters={"temperature_span_K": 10.0, "duration_h": 8.0}),
        dof_target("drift_tight", "dof_drift_mm", "Tx", maximum=0.01, parameters={"temperature_span_K": 10.0}),
        dof_target("repeat", "dof_repeatability_deg", "Rx", maximum=0.005),
        dof_target("no_travel", "dof_travel_margin_deg", "Ry", minimum=0.0, parameters={"required_max": 0.1}),
        dof_target("no_drift", "dof_drift_deg", "Rx", maximum=1.0, parameters={"duration_h": 1.0}),
    ]
    # closed forms: travel margin min(travel_max - required_max, required_min - travel_min); drift per_kelvin dT + per_hour dt
    expected = {
        "travel_inside": ("MET", min(3.0 - 1.0, -0.5 - (-2.0)), "mm"),
        "travel_beyond": ("UNMET", 3.0 - 3.25, "mm"),
        "tilt_inside": ("MET", min(4.0 - 2.0, -2.0 - (-4.0)), "deg"),
        "drift": ("MET", 0.0015 * 10.0 + 0.0004 * 8.0, "mm"),
        "drift_tight": ("UNMET", 0.0015 * 10.0, "mm"),
        "repeat": ("UNMET", 0.01, "deg"),
        "no_travel": ("INDETERMINATE", None, "deg"),
        "no_drift": ("INDETERMINATE", None, "deg"),
    }

    def assessed():
        result = run_optical_project(retained_dof_project(targets=targets), catalog=catalog).result
        by_id = {t.target_id: t for t in result.targets}
        for target_id, (assessment, value, unit) in expected.items():
            t = by_id[target_id]
            got_value = t.value
            ok = t.assessment.value == assessment and t.unit == unit and (got_value is None if value is None else close(got_value, value, abs_=1e-15))
            if value is None:
                ok = ok and tuple(t.issue_codes) == ("MECHANICAL_LIMIT_NOT_DECLARED",)
            c.check(scenario, f"target {target_id}: {assessment} {value} {unit}", ok, assessment=t.assessment.value, value=got_value, unit=t.unit,
                    issue_codes=list(t.issue_codes))
        req = result.outcome.requirements
        c.check(scenario, "requirements record lists met / unmet / indeterminate", set(req.met) == {"travel_inside", "tilt_inside", "drift"}
                and set(req.unmet) == {"travel_beyond", "drift_tight", "repeat"} and set(req.indeterminate) == {"no_travel", "no_drift"},
                met=list(req.met), unmet=list(req.unmet), indeterminate=list(req.indeterminate))
        c.check(scenario, "the plane's power is judged by none of this", result.observations[0].quantity_state.value == "KNOWN",
                quantity_state=result.observations[0].quantity_state.value)

    c.guard(scenario, "dof_* targets", assessed)

    catalog_file = work / "catalog.json"
    catalog_file.write_text(json.dumps(catalog), encoding="utf-8")
    met = dof_target("travel", "dof_travel_margin_mm", "Tx", minimum=0.0, parameters={"required_min": -1.0, "required_max": 1.0})
    undecided = dof_target("tilt", "dof_travel_margin_deg", "Ry", minimum=0.0, parameters={"required_max": 0.5})
    unmet = dof_target("far", "dof_travel_margin_mm", "Tx", minimum=0.0, parameters={"required_max": 3.5})
    for name, chosen, want in (("met", [met], 0), ("met + unmet", [met, unmet], 1), ("met + undecided", [met, undecided], 2)):
        path = work / f"retained_dof_{name.replace(' + ', '_')}.json"
        path.write_text(json.dumps(retained_dof_project(targets=chosen)), encoding="utf-8")
        proc = cli(["project", "run", str(path), "--catalog", str(catalog_file)], work)
        summary = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {}
        c.check(scenario, f"project run exit code with {name} DOF targets is {want} (requirements contract)", proc.returncode == want
                and summary.get("exit_contract") == "requirements", exit=proc.returncode, exit_contract=summary.get("exit_contract"),
                stderr=proc.stderr[-400:])


def check_plate(c: Checks, work: Path, hashes: dict) -> None:
    from os_optics.local import json_api, run_optical_project

    scenario = "plate"
    built = c.guard(scenario, "plate.window native model", lambda: json_api.build_native_model("plate.window", copy.deepcopy(PLATE_FIELDS)))
    if not built or not built.get("ok"):
        c.check(scenario, "plate.window builds", False, reply=built)
        return
    lam_mm = PLATE_WAVELENGTH_NM * 1e-6
    reflectance = ((1.6 - 1.0) / (1.6 + 1.0)) ** 2
    plate_power = POWER_W * (1.0 - reflectance) ** 2 * math.exp(-0.01 * 5.0)  # two uncoated faces, Beer-Lambert over 5 mm
    c.check(scenario, "independent plate power is the recorded 0.852612768013 mW", close(plate_power, 0.852612768013e-3, rel=1e-11),
            independent_W=plate_power)
    # (since os-optics 0.14.0 a Project that states no environment operates in air at 20 C and 101325 Pa: the plate refracts and
    # reflects by its index relative to that air, the free path counts n_air, a phase gradient steers by lambda / n_air)
    from os_optics._vendor.shared_contracts.design_catalog.material_physics import air_index

    n_air = air_index(PLATE_WAVELENGTH_NM)
    relative = 1.6 / n_air
    plate_power_air = POWER_W * (1.0 - ((relative - 1.0) / (relative + 1.0)) ** 2) ** 2 * math.exp(-0.01 * 5.0)
    plate_opl_air = 195.0 * n_air + 1.6 * 5.0
    plate = plate_project(built)
    hashes["plate"] = c.guard(scenario, "plate Project hash", lambda: project_hash(plate))
    if HASHES_0_12_0["plate"] is not None and hashes["plate"] is not None:
        c.check(scenario, "the plate Project keeps its 0.12.0 hash", hashes["plate"] == HASHES_0_12_0["plate"],
                got=hashes["plate"], published_0_12_0=HASHES_0_12_0["plate"])

    def arrival(doc):
        result = run_optical_project(doc, catalog=None).result
        (a,) = result.observations[0].arrivals
        return a, result

    def plate_case():
        a, result = arrival(plate)
        # waist at the source (x = -100), plane at x = 100: 195 mm of air + 1.6 x 5 mm of plate
        c.check(scenario, "recorded plate: power and optical path", close(a.beam.power_W, plate_power_air, rel=1e-12)
                and close(a.beam.optical_path_length_mm, plate_opl_air, abs_=1e-9),
                power_W=a.beam.power_W, independent_W=plate_power_air, opl_mm=a.beam.optical_path_length_mm, independent_opl_mm=plate_opl_air,
                execution=result.outcome.execution.outcome.value)
        return a

    plain = c.guard(scenario, "recorded plate run", plate_case)

    def sample_case():
        before, _ = arrival(plate_project(built, plate_at=-50.0))
        a, _ = arrival(plate_project(built, plate_at=-50.0, sample={"amplitude": "0.9", "phase_rad": "0.5"}))
        c.check(scenario, "plate + thin sample (0.9, 0.5 rad): power x 0.81, path + 0.5 lambda / 2 pi",
                close(a.beam.power_W, plate_power_air * 0.81, rel=1e-12)
                and close(a.beam.optical_path_length_mm - before.beam.optical_path_length_mm, 0.5 * lam_mm / (2.0 * math.pi), abs_=1e-12)
                and close(before.beam.optical_path_length_mm, plate_opl_air, abs_=1e-9),
                power_W=a.beam.power_W, independent_W=plate_power_air * 0.81,
                path_change_mm=a.beam.optical_path_length_mm - before.beam.optical_path_length_mm,
                independent_path_change_mm=0.5 * lam_mm / (2.0 * math.pi))

    c.guard(scenario, "plate + sample", sample_case)

    def map_case():
        nodes, slope = 11, 30.0
        xs = [(i - (nodes - 1) / 2) for i in range(nodes)]
        sample = {"sampled_map": {"nodes_x": nodes, "nodes_y": nodes, "pitch_mm": ["1", "1"],
                                  "phase_rad": [str(int(slope * x)) for _y in xs for x in xs]}}
        a, _ = arrival(sample_only(sample))
        turned = math.hypot(a.beam.direction[1], a.beam.direction[2])
        c.check(scenario, "sampled linear phase map steers by the grating equation", close(turned, lam_mm * slope / (2.0 * math.pi * n_air), abs_=1e-12),
                sin_turned=turned, independent=lam_mm * slope / (2.0 * math.pi * n_air))

    c.guard(scenario, "sampled map", map_case)

    def refusals():
        doc = copy.deepcopy(plate)
        doc["components"][1]["transmission_phase_map"] = {"nodes": 2}
        found = problems_of(json_api.check_project(doc))
        c.check(scenario, "the recorded transmission_phase_map field stays refused", any(p["code"] == "PROJECT_INVALID"
                and "transmission_phase_map" in p["message"] for p in found), problems=found)
        found = problems_of(json_api.check_project(sample_only({"amplitude": "1.2"})))
        c.check(scenario, "a sample amplitude above 1 is refused", bool(found) and any("amplitude" in p["message"] for p in found), problems=found)

    c.guard(scenario, "refusals", refusals)


def check_spectral_lines(c: Checks, work: Path, hashes: dict) -> None:
    from os_optics.local import json_api, lower_project, run_optical_project

    scenario = "spectral_lines"
    lines = free_space(spectral_lines_project())
    hashes["spectral_lines_32"] = c.guard(scenario, "32-line Project hash", lambda: project_hash(lines))
    if HASHES_0_12_0["spectral_lines_32"] is not None and hashes["spectral_lines_32"] is not None:
        c.check(scenario, "the 32-line Project keeps its 0.12.0 hash", hashes["spectral_lines_32"] == HASHES_0_12_0["spectral_lines_32"],
                got=hashes["spectral_lines_32"], published_0_12_0=HASHES_0_12_0["spectral_lines_32"])

    def discrete():
        request = lower_project(lines, catalog=None).request
        c.check(scenario, "the 32-line Engine request states no spectral density", "spectral_density" not in json.dumps(request.model_dump(mode="json")))
        result = run_optical_project(lines, catalog=None).result
        (obs,) = result.observations
        want = [500.0 + 200.0 * k / 31 for k in range(32)]
        got = sorted(p.wavelength_nm for p in obs.power_by_wavelength)
        c.check(scenario, "32 discrete lines arrive, 1 mW / 32 each, 1 mW in all", len(obs.power_by_wavelength) == 32
                and all(close(g, w, abs_=1e-9) for g, w in zip(got, want))
                and all(close(p.power_W, POWER_W / 32, rel=1e-12) for p in obs.power_by_wavelength)
                and close(obs.total_power_W, POWER_W, rel=1e-12), lines=len(obs.power_by_wavelength), total_W=obs.total_power_W)
        c.check(scenario, "the recorded 32-line run: aggregate PASS, execution COMPLETED, no continuous report",
                result.status.value == "PASS" and result.outcome.execution.outcome.value == "COMPLETED" and not result.source_spectra,
                status=result.status.value, execution=result.outcome.execution.outcome.value)

    c.guard(scenario, "32-line run", discrete)

    gaussian = {"shape": {"kind": "GAUSSIAN", "center_nm": 800.0, "fwhm_nm": 20.0}, "line_count": 16}

    def beside():
        white = continuous(copy.deepcopy(gaussian) | {"line_count": 8}, "WHITE", (-200.0, 5.0, 0.0))
        result = run_optical_project(free_space(spectral_lines_project(), white), catalog=None).result
        labels = {s.source_id: (s.basis, s.line_count) for s in result.source_spectra}
        c.check(scenario, "beside a continuous source the 32-line source is labelled DISCRETE",
                labels == {"SRC": ("DISCRETE", 32), "WHITE": ("SAMPLED_CONTINUOUS", 8)}, labels={k: list(v) for k, v in labels.items()})

    c.guard(scenario, "discrete beside continuous", beside)

    def spectrum():
        result = run_optical_project(free_space(continuous(copy.deepcopy(gaussian))), catalog=None).result
        (obs,) = result.observations
        # default band: centre +- 1.5 FWHM = 770 .. 830 nm, 16 equal bins, each with its error-function share of the power
        edges = [770.0 + 60.0 * k / 16 for k in range(17)]
        a = 2.0 * math.sqrt(math.log(2.0)) / 20.0
        shares = [math.erf(a * (e1 - 800.0)) - math.erf(a * (e0 - 800.0)) for e0, e1 in zip(edges, edges[1:])]
        want = [(0.5 * (e0 + e1), POWER_W * s / sum(shares)) for e0, e1, s in zip(edges, edges[1:], shares)]
        got = sorted((p.wavelength_nm, p.power_W) for p in obs.power_by_wavelength)
        c.check(scenario, "continuous Gaussian spectrum: 16 bin-centre lines with error-function powers", len(got) == 16
                and all(close(g[0], w[0], abs_=1e-9) and close(g[1], w[1], rel=1e-10) for g, w in zip(got, want)),
                lines=len(got), worst_power_rel=max((abs(g[1] - w[1]) / w[1] for g, w in zip(got, want)), default=None))
        (report,) = result.source_spectra
        narrow_band = 2.0 * math.sqrt(math.log(2.0)) / math.pi * 800.0 ** 2 / 20.0 * 1e-6  # mm
        c.check(scenario, "coherence length from the spectrum = (2 sqrt(ln 2) / pi) lambda^2 / FWHM to 2e-3",
                report.basis == "SAMPLED_CONTINUOUS" and close(report.coherence_length_mm, narrow_band, rel=2e-3),
                coherence_length_mm=report.coherence_length_mm, narrow_band_mm=narrow_band, basis=report.basis)

    c.guard(scenario, "continuous spectrum", spectrum)

    def static():
        for field in ("vibration", "time_series", "dynamic_feedback"):
            doc = copy.deepcopy(lines)
            doc[field] = {"amplitude_um": 1.0}
            found = problems_of(json_api.check_project(doc))
            c.check(scenario, f"{field} is refused with the static approximation named", len(found) == 1 and found[0]["code"] == "PROJECT_INVALID"
                    and "static approximation" in found[0]["message"], problems=found)

    c.guard(scenario, "time-domain refusals", static)


# ----------------------------------------------------------------------------------------------------------- the run
def installation(c: Checks, allow_source: bool, expect_version: str | None) -> dict:
    import os_optics

    module_file = Path(os_optics.__file__).resolve()
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    in_site = purelib in module_file.parents
    if not allow_source:
        c.check("install", "os_optics is imported from this environment's site-packages", in_site, module=str(module_file))
    version = metadata.version("os-optics")
    if expect_version is not None:
        c.check("install", f"installed os-optics is {expect_version}", version == expect_version, installed=version)
    build_id = None
    try:
        from os_optics.manifest.build import engine_build_id

        build_id = engine_build_id()
    except Exception as exc:  # noqa: BLE001 - recorded, not a check
        build_id = f"unavailable: {exc}"
    return {"distribution": "os-optics", "version": version, "engine_build_id": build_id, "module": str(module_file),
            "site_packages": in_site, "python": sys.version.split()[0], "platform": platform.platform()}


def run(catalog_path: Path, *, allow_source: bool = False, expect_version: str | None = None, workdir: Path | None = None) -> dict:
    c = Checks()
    hashes: dict[str, str | None] = {}
    env = installation(c, allow_source, expect_version)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="release-parity-") as tmp:
        work = Path(workdir) if workdir is not None else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        # a catalog store of its own, automatic checks off: nothing here contacts the catalog service (the retained-DOF
        # scenario passes its catalog document explicitly; the plate and 32-line scenarios use no catalog product)
        isolated = {"OPENSCI_CATALOG_DIR": str(work / "catalog-store"), "OPENSCI_CATALOG_CHECK_INTERVAL": "0"}
        saved = {name: os.environ.get(name) for name in isolated}
        os.environ.update(isolated)
        try:
            for name, fn in (("retained_dof_project", lambda: check_retained_dof(c, catalog, work, hashes)), ("plate_project", lambda: check_plate(c, work, hashes)),
                             ("spectral_lines_project", lambda: check_spectral_lines(c, work, hashes))):
                c.guard(name, f"{name} scenario", fn)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    failed = [r for r in c.rows if not r["ok"]]
    return {"evidence": "opensci.release_parity/1", "installation": env, "catalog": str(catalog_path), "project_hashes": hashes,
            "checks": c.rows, "counts": {"checks": len(c.rows), "passed": len(c.rows) - len(failed), "failed": len(failed)},
            "ok": not failed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="catalog document of the retained-DOF scenario's flat mirror (default: %(default)s)")
    parser.add_argument("--expect-version", help="the os-optics version that must be installed (e.g. 0.13.0)")
    parser.add_argument("--allow-source", action="store_true", help="do not require os_optics from site-packages (repository test)")
    parser.add_argument("--workdir", type=Path, help="keep the run directories and Project files here (default: a temporary directory)")
    parser.add_argument("--out", type=Path, help="also write the evidence JSON here")
    args = parser.parse_args(argv)
    evidence = run(args.catalog, allow_source=args.allow_source, expect_version=args.expect_version, workdir=args.workdir)
    text = json.dumps(evidence, indent=1, sort_keys=False)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0 if evidence["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
