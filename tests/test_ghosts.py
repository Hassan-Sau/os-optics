"""Ghost-path analysis of a run's image plane (``os_optics.results.ghosts``, ``os-optics results ghosts``): analytic references and
the A1 stand-in.

References computed here, independently of the Engine:

* an uncoated N-BK7 plate at normal incidence: Fresnel ``R = ((n - 1) / (n + 1))^2``, primary ``T^2``, the one two-bounce ghost
  ``T R R T`` (relative ``R^2``), the back reflections ``R`` and ``T R T``, and every ghost beyond two bounces born from ``T R^3``;
* two windows with a stated 0.5 % AR coating: ``N (N - 1) / 2 = 6`` two-bounce ghosts, ghost ``(i, j)`` (reflected at face ``i``,
  then at the earlier face ``j``) relative ``R^2 T^(2 (i - j - 1))``;
* a biconvex singlet: the paraxial ghost focus of the folded path (refract S1, reflect S2, reflect S1, refract S2) by a y-u trace with
  signed indices (``n' u' = n u - y c (n' - n)``, a reflection ``n' = -n``);
* completeness: the bound of a two-bounce analysis against the exhaustive four-bounce enumeration of the same case, and against
  the light a cut trace budget leaves out.

These Fresnel and ghost power references assume explicit alpha=0 in test-only declared material copies; they do not
infer known transmission for a catalog glass without bulk-loss evidence. The A1 double-Gauss acceptance case is not in
the repository; the stand-in is ``tests/fixtures/prescriptions/double_gauss_6.csv``
(the same stand-in the imaging analysis uses), on axis and at a 5 deg field.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from os_optics.cli import main as cli_main
from os_optics.errors.status import QuantityState
from os_optics.local import json_api, run_optical_project
from os_optics.results import GhostError, ResultsError, load_run
from os_optics.validation.scope import OUT_OF_SCOPE_METRICS

from declared_helpers import declared_component, with_models
from local_helpers import ALONG_X, plane, project, source
from test_imaging import NBK7, _lens_project, _nbk7

ENGINE = Path(__file__).resolve().parents[1]
WL = 587.5618
LINE = "SRC#0."  # (a line's ids carry the source and its spectral index)


def _ghost_project(text: str, **kwargs) -> dict:
    """A test-only ideal-loss copy of the prescription for Fresnel/ghost power references."""
    document = _lens_project(text, **kwargs)
    for glass in document["model_materials"]:
        assert "absorption_coefficient_per_mm" not in glass and "bulk_loss" not in glass
        glass["absorption_coefficient_per_mm"] = "0"  # explicitly assumed alpha=0, not catalog transmission evidence
    return document


def _prescription(r1: float | None, r2: float | None, thickness: float, back: float, semi: float = 10.0) -> str:
    def radius(r: float | None) -> str:
        return "inf" if r is None else f"{r:.9f}"

    return "\n".join([
        "#! format = csv-surface-table/1", "#! name = test element", "#! units = mm", "[surfaces]",
        "surface,type,comment,radius,thickness,material,semi_diameter,mech_semi_diameter,conic,a4,a6,a8,cylinder_axis,stop",
        "OBJ,STANDARD,object at infinity,inf,inf,,,,,,,,,",
        f"1,STANDARD,front,{radius(r1)},{thickness},N-BK7,{semi},{semi + 0.5},,,,,,yes",
        f"2,STANDARD,rear,{radius(r2)},{back:.9f},,{semi},{semi + 0.5},,,,,,",
        "IMA,STANDARD,image plane,,,,,,,,,,,", "", "[materials]", "name,kind,b1,b2,b3,c1,c2,c3,min_nm,max_nm,display_name",
        "N-BK7,SELLMEIER," + ",".join(str(v) for v in NBK7) + ",300,2500,N-BK7 (fixture Sellmeier)", ""])


def _rows(doc, line: str = LINE) -> list[dict]:
    return doc.arrays[line + "ghosts"]


def _row(doc, surfaces: list[str], line: str = LINE) -> dict:
    (row,) = [r for r in _rows(doc, line) if r["bounce_surfaces"] == surfaces]
    return row


def _q(doc, row: dict, name: str, line: str = LINE):
    return doc.quantity(f"{line}{row['path_id']}.{name}")


def _value(doc, row: dict, name: str, line: str = LINE) -> float:
    q = _q(doc, row, name, line)
    assert q.state is QuantityState.KNOWN, (q.quantity_id, q.state, q.reason)
    return float(q.value)


def _bound(doc, line: str = LINE) -> float:
    q = doc.quantity(line + "unlisted_power_upper_relative_to_line")
    assert q.state is QuantityState.BOUNDED and q.interval.kind.value == "PROVEN"
    return float(q.interval.upper)


# ------------------------------------------------------------------------------------------------- an uncoated plate: Fresnel


@pytest.fixture(scope="module")
def plate():
    return load_run(run_optical_project(_ghost_project(_prescription(None, None, 5.0, 20.0), waist_mm=1.0)))


def test_an_uncoated_plate_gives_the_fresnel_double_reflection(plate):
    doc = plate.ghosts("image")
    n = _nbk7(WL)
    r = ((n - 1.0) / (n + 1.0)) ** 2
    t = 1.0 - r
    assert doc.analysis == "GHOST_PATHS" and doc.execution.complete
    assert doc.quantity(LINE + "primary_power_relative_to_line").value == pytest.approx(t * t, rel=1e-8)
    ghost = _row(doc, ["LENS/L1-S2", "LENS/L1-S1"])
    assert ghost["direction"] == "FORWARD" and ghost["bounce_order"] == 2 and ghost["reaches_image_plane"]
    assert _value(doc, ghost, "relative_power") == pytest.approx(r * r, rel=1e-8)
    assert _value(doc, ghost, "power_relative_to_line") == pytest.approx(t * r * r * t, rel=1e-8)
    assert _value(doc, ghost, "reflected_power_relative_to_line") == pytest.approx(t * r * r, rel=1e-8)
    # a collimated ghost has no focus; its Gaussian beam has a waist, and the same spot as the primary
    focus = _q(doc, ghost, "focus_offset_mm")
    assert focus.state is QuantityState.NOT_APPLICABLE and focus.reason == "RAY_FOCUS_UNDEFINED"
    assert _value(doc, ghost, "spot_rms_radius_um") == pytest.approx(doc.quantity(LINE + "primary_spot_rms_radius_um").value, rel=1e-6)
    assert _value(doc, ghost, "irradiance_ratio") == pytest.approx(r * r, rel=1e-3)
    assert _q(doc, ghost, "irradiance_ratio").basis == "gaussian_peak_irradiance"
    # the back reflections: R from the front face, T R T from the rear face
    front, rear = _row(doc, ["LENS/L1-S1"]), _row(doc, ["LENS/L1-S2"])
    assert front["direction"] == rear["direction"] == "BACKWARD" and not front["reaches_image_plane"]
    assert _value(doc, front, "end_power_relative_to_line") == pytest.approx(r, rel=1e-8)
    assert _value(doc, rear, "end_power_relative_to_line") == pytest.approx(t * r * t, rel=1e-8)
    # everything beyond two bounces is born from the two-bounce ghost's reflection at the rear face: T R^3, a proven bound; it
    # takes each face's reflectance and transmittance at their largest over the beam's plane-wave spectrum (far-field half angle
    # 1.9e-4 rad here), which lifts the chief ray's T R^3 by a few 1e-6
    assert doc.completeness.enumerated == 3 and doc.completeness.proven
    assert t * r ** 3 <= _bound(doc) <= t * r ** 3 * (1.0 + 1e-5)
    assert {item.reason for item in doc.completeness.not_followed} == {"BOUNCE_LIMIT"}
    assert t * r ** 3 * 1e-3 * (1.0 - 1e-9) <= doc.completeness.unenumerated_power_upper_W <= t * r ** 3 * 1e-3 * (1.0 + 1e-5)


def test_the_two_bounce_bound_covers_the_exhaustive_four_bounce_enumeration(plate):
    two = plate.ghosts("image")
    four = plate.ghosts("image", {"max_bounce_order": 4})
    n = _nbk7(WL)
    r = ((n - 1.0) / (n + 1.0)) ** 2
    t = 1.0 - r
    beyond = [row for row in _rows(four) if row["bounce_order"] > 2]
    assert sorted(row["bounce_order"] for row in beyond) == [3, 4]
    # the light beyond two bounces: the order-3 back reflection leaving the plate and the order-4 ghost on the image plane
    order3 = next(row for row in beyond if row["bounce_order"] == 3)
    order4 = next(row for row in beyond if row["bounce_order"] == 4)
    assert _value(four, order3, "end_power_relative_to_line") == pytest.approx(t * r ** 3 * t, rel=1e-8)
    assert _value(four, order4, "power_relative_to_line") == pytest.approx(t * r ** 4 * t, rel=1e-8)
    births = math.fsum(_value(four, row, "reflected_power_relative_to_line") for row in beyond if row["bounce_order"] == 3)
    # the bound is the power born beyond the bounce order, with each face's coefficients at their largest over the beam's angles
    assert births <= _bound(two) <= births * (1.0 + 1e-5)
    reached = math.fsum([_value(four, order3, "end_power_relative_to_line"), _value(four, order4, "power_relative_to_line"), _bound(four)])
    assert reached <= _bound(two) * (1.0 + 1e-12)
    assert t * r ** 5 <= _bound(four) <= t * r ** 5 * (1.0 + 1e-5)


# ------------------------------------------------------------------------------------------------- coated windows: N (N - 1) / 2


COATING_R = 0.005


@pytest.fixture(scope="module")
def windows():
    window = json_api.build_native_model("plate.window", {"material": {"constant_index": "1.5", "valid_band_nm": ["400", "900"], "absorption_coefficient_per_mm": "0"}, "thickness_mm": "5",
                                                          "diameter_mm": "25", "ar_coating": {"reflectance": str(COATING_R), "valid_aoi_deg": [0, 10]},
                                                          "model_id": "w"})
    assert window["ok"] is True, window
    document = with_models(project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=632.8, waist_mm=1.0),
                                    declared_component("W1", "window", "w", (0.0, 0.0, 0.0), ALONG_X),
                                    declared_component("W2", "window", "w", (20.0, 0.0, 0.0), ALONG_X)],
                                   [plane("image", (60.0, 0.0, 0.0), ALONG_X, radius_mm=30.0)]),
                           [window["optical_model"]], window["model_materials"], window["model_coatings"], design_intent="OPTICAL_SYSTEM")
    return load_run(run_optical_project(document, catalog=None))


def test_coated_faces_give_every_pair_with_the_coating_reflectance(windows):
    doc = windows.ghosts("image")
    line = next(key[:-len("ghosts")] for key in doc.arrays if key.endswith("ghosts"))
    faces = ["W1/S1", "W1/S2", "W2/S1", "W2/S2"]
    forward = [row for row in _rows(doc, line) if row["direction"] == "FORWARD"]
    assert len(forward) == len(faces) * (len(faces) - 1) // 2
    t = 1.0 - COATING_R
    for row in forward:
        later, earlier = (faces.index(s) for s in row["bounce_surfaces"])
        assert later > earlier
        assert _value(doc, row, "relative_power", line) == pytest.approx(COATING_R ** 2 * t ** (2 * (later - earlier - 1)), rel=1e-9)
    backward = [row for row in _rows(doc, line) if row["direction"] == "BACKWARD"]
    for row in backward:
        (face,) = (faces.index(s) for s in row["bounce_surfaces"])
        assert _value(doc, row, "end_power_relative_to_line", line) == pytest.approx(COATING_R * t ** (2 * face), rel=1e-9)
    assert doc.completeness.proven and doc.completeness.enumerated == len(faces) * (len(faces) + 1) // 2


def test_a_cut_trace_budget_is_partial_and_its_bound_covers_what_it_left_out(windows):
    full = windows.ghosts("image")
    cut = windows.ghosts("image", {"max_paths": 4})
    line = next(key[:-len("ghosts")] for key in full.arrays if key.endswith("ghosts"))
    assert not cut.execution.complete and cut.execution.termination.value == "BUDGET_EXHAUSTED"
    assert "max_paths" in cut.execution.budgets.exhausted
    assert any(item.reason == "PATH_BUDGET" for item in cut.completeness.not_followed)
    listed = {row["path_id"] for row in _rows(cut, line)}
    missing = [row for row in _rows(full, line) if row["path_id"] not in listed]
    assert missing
    # every ghost the cut trace does not list carries at most the cut trace's bound
    left = math.fsum(_value(full, row, "reflected_power_relative_to_line", line) for row in missing if row["bounce_order"] == 1)
    assert left <= _bound(cut, line)
    assert _bound(cut, line) >= _bound(full, line)


# ------------------------------------------------------------------------------------------------- a singlet: the ghost focus


def _paraxial_ghost_focus(n: float, r1: float, r2: float, thickness: float) -> tuple[float, float]:
    """(back focal length, ghost focus behind the rear vertex) of a singlet in air: refract S1, reflect S2, reflect S1, refract S2."""

    c1, c2, h = 1.0 / r1, 1.0 / r2, 1.0
    u = -h * c1 * (n - 1.0) / n
    y = h + u * thickness
    bfl = -y / (n * u - y * c2 * (1.0 - n))
    y, u = h, -h * c1 * (n - 1.0) / n
    y += u * thickness
    u = -u - 2.0 * y * c2  # reflect at S2: n' = -n
    y -= u * thickness
    u = -u - 2.0 * y * c1  # reflect at S1, back to +n
    y += u * thickness
    u = n * u - y * c2 * (1.0 - n)
    return bfl, -y / u


def test_a_two_surface_ghost_focuses_where_paraxial_imaging_of_the_folded_path_puts_it():
    n = _nbk7(WL)
    r1, r2, thickness = 40.0, -40.0, 5.0
    bfl, ghost_focus = _paraxial_ghost_focus(n, r1, r2, thickness)
    results = load_run(run_optical_project(_ghost_project(_prescription(r1, r2, thickness, bfl), waist_mm=0.5)))
    doc = results.ghosts("image")
    ghost = _row(doc, ["LENS/L1-S2", "LENS/L1-S1"])
    expected = ghost_focus - bfl  # behind the image plane positive
    assert expected < -30.0  # this ghost focuses just behind the lens, far before the image plane
    assert _value(doc, ghost, "focus_offset_mm") == pytest.approx(expected, abs=0.02)
    for axis in ("x", "y"):
        assert _value(doc, ghost, f"waist_offset_{axis}_mm") == pytest.approx(expected, abs=0.02)
    assert _q(doc, ghost, "focus_offset_mm").convergence.converged
    # a defocused ghost: its spot on the image plane is the cone it diverges in over the distance from its focus
    assert _value(doc, ghost, "spot_rms_radius_um") > 100.0 * doc.quantity(LINE + "primary_spot_rms_radius_um").value
    assert 0.0 < _value(doc, ghost, "irradiance_ratio") < _value(doc, ghost, "relative_power")


# ------------------------------------------------------------------------------------------------- the A1 stand-in


@pytest.fixture(scope="module")
def double_gauss():
    text = (ENGINE / "tests" / "fixtures" / "prescriptions" / "double_gauss_6.csv").read_text(encoding="utf-8")
    results = load_run(run_optical_project(_ghost_project(text, waist_mm=3.0, field_deg=(5.0,))))
    return results, results.ghosts("image")


def test_a1_like_double_gauss_lists_its_strongest_ghosts_with_focus_order_and_relative_power(double_gauss):
    results, doc = double_gauss
    # the steep inner faces leave most ghost beams without a valid Gaussian state: there the reflectance is bounded over the
    # beam's pupil rays only, so the bound is not proven (NOMINAL) and each line counts those ghost-surface meetings
    assert doc.execution.complete and not doc.completeness.proven
    surfaces = 10  # the double Gauss's refracting faces (two of them cemented)
    for field in ("SRC", "FIELD1"):
        line = f"{field}#0."
        rows = _rows(doc, line)
        assert all(row["bounce_order"] == 2 for row in rows if row["direction"] == "FORWARD")
        # a forward ghost an aperture clips on its way, or that leaves beside the image plane, is listed with its end, not dropped
        assert all(row["termination"] in ("CLIPPED", "ESCAPED") for row in rows if row["direction"] == "FORWARD" and not row["reaches_image_plane"])
        forward = [row for row in rows if row["direction"] == "FORWARD" and row["reaches_image_plane"]]
        assert 20 <= len(forward) <= surfaces * (surfaces - 1) // 2
        # listed strongest first; the strongest is a product of two air-glass reflections, a few 0.1 % of the primary
        powers = [_value(doc, row, "relative_power", line) for row in forward]
        assert powers == sorted(powers, reverse=True)
        assert 1e-3 < powers[0] < 1e-2
        for row in forward:
            focus = _q(doc, row, "focus_offset_mm", line)
            if row.get("reference_promoted_at"):
                # a ghost whose chief ray fell beside a clear aperture goes on as its pupil rays that passed: with fewer than
                # MIN_FOCUS_RAYS of them on the plane its focus is not given, and says why
                assert focus.state is QuantityState.KNOWN or (focus.state is QuantityState.UNKNOWN and focus.reason == "TOO_FEW_RAYS")
                continue
            assert focus.state is QuantityState.KNOWN and focus.unit == "mm"
            for axis in ("x", "y"):
                waist = _q(doc, row, f"waist_offset_{axis}_mm", line)
                # a ghost whose paraxial Gaussian beam is not valid keeps its ray focus and states why its waist is not given
                assert waist.state is QuantityState.KNOWN or (waist.state is QuantityState.OUT_OF_MODEL_SCOPE and waist.reason.startswith("GAUSSIAN_"))
        bound = doc.quantity(line + "unlisted_power_upper_relative_to_primary")
        assert bound.state is QuantityState.BOUNDED and bound.interval.kind.value == "NOMINAL" and bound.interval.upper >= bound.value > 0.0
        assert doc.arrays[line + "completeness"]["ghost_surface_meetings_with_sampled_angles"] > 0
    assert any(_q(doc, row, "waist_offset_x_mm", LINE).state is QuantityState.OUT_OF_MODEL_SCOPE for row in _rows(doc, LINE)
               if row["reaches_image_plane"])
    assert "PASS" not in json.dumps(doc.model_dump(mode="json", by_alias=True))
    # the analysis changes no run: the primary light of the ghost trace is the run's own on the plane
    observation = results._observation("image")
    run_power = sum(a.beam.power_W for a in observation.arrivals if a.beam.source_id == "SRC")
    line_W = doc.arrays[f"{LINE}completeness"]["line_input_W"]
    assert doc.quantity(f"{LINE}primary_power_relative_to_line").value == pytest.approx(run_power / line_W, rel=1e-8)


# ------------------------------------------------------------------------------------------------- the surfaces


def test_requests_refusals_and_the_surfaces(plate, tmp_path, capsys):
    with pytest.raises(GhostError) as exc:
        plate.ghosts("image", {"max_bounce_order": 5})
    assert exc.value.code == "GHOSTS_INVALID"
    with pytest.raises(ResultsError):
        plate.ghosts("nowhere")
    unknown = plate.ghosts("image", {"fields": ["SRC", "NOPE"]})
    # a field that is not a beam source is its own NOT_APPLICABLE row; the enumeration of the others ran to its end
    assert unknown.execution.complete and unknown.execution.outcome.value == "COMPLETED"
    assert unknown.quantity("NOPE.field").state.value == "NOT_APPLICABLE" and unknown.quantity("NOPE.field").reason == "FIELD_NOT_A_BEAM_SOURCE"
    assert "ghost_analysis" not in OUT_OF_SCOPE_METRICS
    run_dir = tmp_path / "run"
    run_optical_project(_ghost_project(_prescription(None, None, 5.0, 20.0), waist_mm=1.0), output_dir=str(run_dir))
    before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    request = tmp_path / "ghosts.json"
    request.write_text(json.dumps({"max_bounce_order": 2}), encoding="utf-8")
    assert cli_main(["results", "ghosts", str(run_dir), "--monitor", "image", "--request", str(request)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "GHOST_PATHS" and out["execution"]["outcome"] == "COMPLETED"
    request.write_text(json.dumps({"max_bounce_order": 0}), encoding="utf-8")
    assert cli_main(["results", "ghosts", str(run_dir), "--monitor", "image", "--request", str(request)]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "GHOSTS_INVALID"
    request.write_text(json.dumps({"fields": ["NOPE"]}), encoding="utf-8")
    assert cli_main(["results", "ghosts", str(run_dir), "--monitor", "image", "--request", str(request)]) == 0
    answered = json.loads(capsys.readouterr().out)
    assert answered["execution"]["outcome"] == "COMPLETED"
    assert [(q["quantity_id"], q["state"], q["reason"]) for q in answered["quantities"]] == [("NOPE.field", "NOT_APPLICABLE", "FIELD_NOT_A_BEAM_SOURCE")]
    # a budget that runs out is what makes the analysis partial, and the command's exit code 2
    request.write_text(json.dumps({"max_bounce_order": 2, "max_paths": 1}), encoding="utf-8")
    assert cli_main(["results", "ghosts", str(run_dir), "--monitor", "image", "--request", str(request)]) == 2
    stopped = json.loads(capsys.readouterr().out)["execution"]
    assert stopped["outcome"] == "PARTIAL" and stopped["termination"] == "BUDGET_EXHAUSTED" and "max_paths" in stopped["budgets"]["exhausted"]
    files = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    assert files == before  # the analysis writes nothing into the run
    reply = json_api.run_results(files, {"kind": "ghosts", "monitor": "image", "request": {"max_bounce_order": 2}})
    assert reply["ok"] is True and reply["result"]["analysis"] == "GHOST_PATHS"
    assert reply["result"]["quantities"] == out["quantities"]
    assert json_api.capabilities()["field_analyses"]["ghost_paths"]["schema_kind"] == "ghost-request"
    assert json_api.json_schema("ghost-request")["schema"]["title"] == "GhostRequestV1"
    assert cli_main(["schema", "ghost-request"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "GhostRequestV1"


# ------------------------------------------------------------------------------------------------- model scope


AOI_OUTSIDE = "COATING_AOI_OUTSIDE_RANGE"
#: Where a ghost lands: a coefficient used outside its range leaves these as they are.
LANDING = ("focus_offset_mm", "waist_offset_x_mm", "waist_offset_y_mm", "spot_rms_radius_um", "primary_spot_rms_radius_um")


def _tilted_windows(tmp_path):
    """The coated windows with W2 turned 15 deg about z (its coating is published for 0-10 deg) and moved 6 mm up: SRC
    (y = 0) crosses both windows, FREE (y = -9) crosses W1 and passes beside W2. A plane ``early`` between the windows
    receives the light the primary carries before W2 and the ghosts W2 sends back."""

    from local_helpers import quat_multiply, turn_about_z

    window = json_api.build_native_model("plate.window", {"material": {"constant_index": "1.5", "valid_band_nm": ["400", "900"]}, "thickness_mm": "5",
                                                          "diameter_mm": "25", "ar_coating": {"reflectance": str(COATING_R), "valid_aoi_deg": [0, 10]},
                                                          "model_id": "w"})
    turned = quat_multiply(turn_about_z(math.cos(math.radians(7.5)), math.sin(math.radians(7.5))), ALONG_X)
    document = with_models(project([source(position=(-50.0, 0.0, 0.0), wavelength_nm=632.8, waist_mm=1.0),
                                    source("FREE", position=(-50.0, -9.0, 0.0), wavelength_nm=632.8, waist_mm=1.0),
                                    declared_component("W1", "window", "w", (0.0, 0.0, 0.0), ALONG_X),
                                    declared_component("W2", "window", "w", (20.0, 6.0, 0.0), turned)],
                                   [plane("image", (60.0, 0.0, 0.0), ALONG_X, radius_mm=30.0),
                                    plane("early", (10.0, 0.0, 0.0), ALONG_X, radius_mm=30.0)]),
                           [window["optical_model"]], window["model_materials"], window["model_coatings"], design_intent="OPTICAL_SYSTEM")
    run_dir = tmp_path / "tilted"
    run_optical_project(document, catalog=None, output_dir=str(run_dir))
    return run_dir


def _checks(quantity) -> list[tuple[str, tuple[str, ...]]]:
    basis = quantity.value_basis
    return sorted((c.code, tuple(c.at)) for c in (basis.validity if basis is not None else ()) or ())


def test_ghost_powers_of_light_a_coating_finding_reaches_read_out_of_model_scope_with_its_check(tmp_path, capsys):
    """W2's coating met at 15 deg (published 0-10 deg): the run's findings at W2 list SRC's arrival on ``image``. Every
    ghost-analysis number of SRC's line that reads a power -- the primary's, each ghost's (those that bounce at W2 and those
    that only pass it), the bound on the ghosts not listed -- reads OUT_OF_MODEL_SCOPE, keeps its number and carries the
    findings' ESTIMATE; where the ghosts land stays KNOWN with no basis. FREE's line, which passes beside W2, stays KNOWN with
    no basis. RunResults.ghosts, the JSON query and the CLI give the same document."""

    run_dir = _tilted_windows(tmp_path)
    results = load_run(str(run_dir))
    findings = [f for f in results.targets()["model_scope_issues"] if f["code"] == AOI_OUTSIDE]
    assert findings and {f["instance_id"] for f in findings} == {"W2"}
    assert all({a["path_id"] for a in f["affected_arrivals"]} == {"SRC#0"} for f in findings)
    doc = results.ghosts("image")
    w2 = {("W2/S1",), ("W2/S2",)}
    expected = sorted((AOI_OUTSIDE, at) for at in w2)
    primary = doc.quantity(LINE + "primary_power_relative_to_line")
    assert primary.state is QuantityState.OUT_OF_MODEL_SCOPE and primary.value is not None and primary.reason == AOI_OUTSIDE
    assert primary.value_basis.kind.value == "ESTIMATE" and _checks(primary) == expected
    bounce = _row(doc, ["W2/S2", "W2/S1"])
    relative = _q(doc, bounce, "relative_power")
    assert relative.state is QuantityState.OUT_OF_MODEL_SCOPE and relative.value == pytest.approx(COATING_R ** 2, rel=1e-4)
    assert _checks(relative) == expected
    passing = _row(doc, ["W1/S2", "W1/S1"])
    assert _q(doc, passing, "relative_power").state is QuantityState.OUT_OF_MODEL_SCOPE
    for row in _rows(doc):
        for qid in row["quantities"]:
            q = doc.quantity(qid)
            if qid.rsplit(".", 1)[-1] in LANDING:
                assert q.value_basis is None and q.state is not QuantityState.OUT_OF_MODEL_SCOPE or q.value is None, qid
            elif q.value is not None:
                assert q.state is QuantityState.OUT_OF_MODEL_SCOPE and _checks(q) == expected, qid
    bound = doc.quantity(LINE + "unlisted_power_upper_relative_to_line")
    assert bound.state is QuantityState.OUT_OF_MODEL_SCOPE and bound.interval is not None and _checks(bound) == expected
    assert doc.quantity(LINE + "primary_spot_rms_radius_um").state is QuantityState.KNOWN
    assert {(f["code"], f["instance_id"]) for f in doc.arrays["model_scope"]} == {(AOI_OUTSIDE, "W2")}

    free = "FREE#0."
    assert all("W2" not in s for row in _rows(doc, free) for s in row["bounce_surfaces"])
    free_quantities = [q for q in doc.quantities if q.quantity_id.startswith(free)]
    assert free_quantities and all(q.value_basis is None and q.state in (QuantityState.KNOWN, QuantityState.BOUNDED, QuantityState.NOT_APPLICABLE)
                                   for q in free_quantities)

    expected_json = doc.model_dump(mode="json", by_alias=True)
    files = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    reply = json_api.run_results(files, {"kind": "ghosts", "monitor": "image"})
    assert reply["ok"] is True and reply["result"]["quantities"] == expected_json["quantities"]
    assert reply["result"]["arrays"]["model_scope"] == expected_json["arrays"]["model_scope"]
    capsys.readouterr()
    assert cli_main(["results", "ghosts", str(run_dir), "--monitor", "image"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["quantities"] == expected_json["quantities"] and out["arrays"]["model_scope"] == expected_json["arrays"]["model_scope"]


def test_ghosts_of_coated_windows_used_inside_their_range_carry_no_model_scope(windows):
    """The untilted windows: no finding; every number reads KNOWN (the bound BOUNDED) with no value_basis, and the document
    lists no model scope."""

    doc = windows.ghosts("image")
    numbers = [q for q in doc.quantities if q.value is not None]
    assert numbers and all(q.state in (QuantityState.KNOWN, QuantityState.BOUNDED) and q.value_basis is None for q in numbers)
    assert "model_scope" not in doc.arrays


def test_a_ghost_that_bounces_at_an_element_a_finding_names_carries_its_check_on_a_plane_before_that_element(tmp_path):
    """On ``early``, between W1 and W2, SRC's primary light has not met W2 yet: the findings at W2 do not list it, and the
    primary and the ghosts that only bounce at W1 stay KNOWN with no basis. A ghost that bounces at W2 reflects W2's
    coating at 15 deg: its powers read OUT_OF_MODEL_SCOPE with W2's checks, its landing stays KNOWN."""

    doc = load_run(str(_tilted_windows(tmp_path))).ghosts("early")
    expected = sorted((AOI_OUTSIDE, at) for at in (("W2/S1",), ("W2/S2",)))
    primary = doc.quantity(LINE + "primary_power_relative_to_line")
    assert primary.state is QuantityState.KNOWN and primary.value_basis is None
    for row in _rows(doc):
        at_w2 = any(s.startswith("W2/") for s in row["bounce_surfaces"])
        for qid in row["quantities"]:
            q = doc.quantity(qid)
            if q.value is None:
                continue
            if at_w2 and qid.rsplit(".", 1)[-1] not in LANDING:
                assert q.state is QuantityState.OUT_OF_MODEL_SCOPE and _checks(q) == expected, qid
            else:
                assert q.state is QuantityState.KNOWN and q.value_basis is None, qid
    assert any(any(s.startswith("W2/") for s in row["bounce_surfaces"]) and row["reaches_image_plane"] for row in _rows(doc))
    assert any(not any(s.startswith("W2/") for s in row["bounce_surfaces"]) and row["reaches_image_plane"] for row in _rows(doc))
    assert {(f["code"], f["instance_id"]) for f in doc.arrays["model_scope"]} == {(AOI_OUTSIDE, "W2")}
