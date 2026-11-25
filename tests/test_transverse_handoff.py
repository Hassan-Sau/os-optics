"""Transverse-plane handoff export (``RunResults.handoff``, export kind ``handoff``, ``json_api.run_results`` kind ``handoff``,
``os-optics results handoff`` / ``export handoff``).

Independent references: the stated TEM00 field-construction convention is checked by building the sampled field from the
exported q at one plane with NumPy, propagating it with an angular-spectrum (FFT) propagator written here and comparing
its second-moment radius with the Engine's beam at a second plane (a wrong curvature sign would make the beam converge);
the normalization by integrating |E|^2; the transmitted power after a stop with the closed form P (1 - exp(-2 a^2 / w^2));
the element outline with a hand-computed frame change.
"""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stdout

import numpy as np
import pytest

from os_optics.cli import main as cli_main
from os_optics.local import json_api, run_optical_project
from os_optics.results import load_run

from declared_helpers import declared_component, sequence, stop_model, surface, with_models
from local_helpers import AGAINST_X, ALONG_X, catalog_document, plane, project, quat_multiply, source, turn_about_z

NM = 1e-6
SQRT_HALF = math.sqrt(0.5)


def _stop_project(opening: dict, *, plane_x: float = 99.0, waist_mm: float = 1.0, wavelength_nm: float = 532.0, m2: float = 1.0,
                  plane_quaternion=ALONG_X) -> dict:
    src = source(position=(0.0, 0.0, 0.0), waist_mm=waist_mm, wavelength_nm=wavelength_nm)
    src["source_model"]["m2"] = m2
    doc = project([src, declared_component("STOP", "aperture", "stop", (100.0, 0.0, 0.0), ALONG_X)],
                  [plane("pupil", (plane_x, 0.0, 0.0), plane_quaternion, radius_mm=20.0), plane("after", (300.0, 0.0, 0.0), radius_mm=40.0)])
    return with_models(doc, [stop_model("stop", edge_radius_mm="12.7", **opening)])


def _results(document: dict):
    return load_run(run_optical_project(document))


# ------------------------------------------------------------------------------------------------ field convention


def _angular_spectrum(field: np.ndarray, dx: float, wavelength_mm: float, distance_mm: float) -> np.ndarray:
    n = field.shape[0]
    f = np.fft.fftfreq(n, d=dx)
    fx, fy = np.meshgrid(f, f, indexing="ij")
    kz = 2 * np.pi * np.sqrt(np.maximum(0.0, 1.0 / wavelength_mm**2 - fx**2 - fy**2))
    return np.fft.ifft2(np.fft.fft2(field) * np.exp(1j * kz * distance_mm))


def test_the_stated_tem00_convention_rebuilds_the_engines_beam():
    """A 50 um waist at 532 nm (z_R = 14.8 mm): the field built from the handoff at 30 mm by the stated convention, propagated 30 mm
    with an independent angular-spectrum propagator, has the Engine's 1/e^2 radius at 60 mm (second moments), and |E|^2
    integrates to the beam power."""

    lam, w0 = 532.0, 0.05
    doc = project([source(position=(0.0, 0.0, 0.0), waist_mm=w0, wavelength_nm=lam)],
                  [plane("a", (30.0, 0.0, 0.0), radius_mm=10.0), plane("b", (60.0, 0.0, 0.0), radius_mm=10.0)])
    results = _results(doc)
    beam_a = results.handoff("a")["beams"][0]
    beam_b = results.handoff("b")["beams"][0]
    assert beam_a["validity"]["beam_parameters"] == "TEM00" and beam_a["validity"]["tem00_field_applicable"] is True
    g = beam_a["gaussian"]
    assert g["q_x_mm"][0] == pytest.approx(30.0) and g["q_x_mm"][1] == pytest.approx(math.pi * w0**2 / (lam * NM), rel=1e-12)
    assert g["curvature_radius_x_mm"] > 0  # diverging after the waist

    n, half = 1024, 1.6
    x = (np.arange(n) - n / 2) * (2 * half / n)
    dx = x[1] - x[0]
    xx, yy = np.meshgrid(x, x, indexing="ij")
    k = 2 * np.pi * beam_a["medium_index"] / (lam * NM)
    wx, wy, rx, ry, power = g["beam_radius_x_mm"], g["beam_radius_y_mm"], g["curvature_radius_x_mm"], g["curvature_radius_y_mm"], beam_a["power_W"]
    field = (np.sqrt(2 * power / (np.pi * wx * wy)) * np.exp(-xx**2 / wx**2 - yy**2 / wy**2)
             * np.exp(1j * k * (xx**2 / (2 * rx) + yy**2 / (2 * ry))))
    # the convention's equivalent form: exp(+i k x^2 / (2 conj(q)))
    q = complex(*g["q_x_mm"])
    assert np.exp(1j * k * x**2 / (2 * q.conjugate())) * np.exp(1j * k * x**2 / (2 * q.conjugate())).conj() == pytest.approx(
        np.exp(-2 * x**2 / wx**2), abs=1e-12)
    assert float(np.sum(np.abs(field) ** 2) * dx * dx) == pytest.approx(power, rel=1e-9)
    propagated = _angular_spectrum(field, dx, lam * NM, 30.0)
    intensity = np.abs(propagated) ** 2
    w_numeric = 2 * math.sqrt(float(np.sum(intensity * xx**2) / np.sum(intensity)))
    assert w_numeric == pytest.approx(beam_b["gaussian"]["beam_radius_x_mm"], rel=2e-4)
    # the opposite curvature sign (a converging wavefront) would not reproduce the Engine's beam
    wrong = _angular_spectrum(field.conj(), dx, lam * NM, 30.0)
    w_wrong = 2 * math.sqrt(float(np.sum(np.abs(wrong) ** 2 * xx**2) / np.sum(np.abs(wrong) ** 2)))
    assert abs(w_wrong - beam_b["gaussian"]["beam_radius_x_mm"]) > 0.1 * beam_b["gaussian"]["beam_radius_x_mm"]
    conventions = results.handoff("a")["conventions"]
    assert "exp(+i k x^2 / (2 conj(q_x)))" in conventions["tem00_field"] and "Gouy phase is omitted" in conventions["phase_reference"]


# ------------------------------------------------------------------------------------------------ validity


def test_truncation_keeps_the_power_exact_and_marks_the_downstream_beam():
    a = 1.2
    results = _results(_stop_project({"opening_radius_mm": str(a)}))
    before = results.handoff("pupil")["beams"][0]
    assert before["validity"]["beam_parameters"] == "TEM00"  # the stop is downstream of this plane
    w_stop = math.sqrt(1 + (100.0 / (math.pi * 1.0**2 / (532.0 * NM))) ** 2)  # 1/e^2 radius of the 1 mm waist 100 mm downstream
    after = results.handoff("after")["beams"][0]
    validity = after["validity"]
    assert validity["beam_parameters"] == "TRUNCATED_GAUSSIAN" and validity["tem00_field_applicable"] is False
    assert [t["surface_id"] for t in validity["truncated_upstream"]] == ["STOP/stop"]
    assert [i["code"] for i in validity["issues"]] == ["GAUSSIAN_TRUNCATION"]
    assert "power_W is exact" in validity["statement"]
    expected = 0.001 * (1 - math.exp(-2 * a**2 / w_stop**2))
    assert after["power_W"] == pytest.approx(expected, rel=1e-6)
    # the share the stop cut, against the closed form exp(-2 a^2 / w^2) of a centred round opening: far above the run's
    # warning share, so the trace's own judgement stands
    cut = math.exp(-2 * a**2 / w_stop**2)
    (row,) = validity["truncated_upstream"]
    assert row["power_fraction"] == pytest.approx(cut, rel=1e-6) and row["warned"] is True and cut > 1e-3
    assert validity["truncated_power_fraction"] == row["power_fraction"]
    assert validity["field_intensity_ripple"] == pytest.approx(2 * math.sqrt(cut) + cut, rel=1e-6)
    rows = results.table("handoff", monitor="after")[0]
    assert rows[0]["beam_parameters"] == "TRUNCATED_GAUSSIAN" and rows[0]["truncated_at"] == ["STOP/stop"]
    assert rows[0]["truncated_power_fraction"] == row["power_fraction"] and rows[0]["field_intensity_ripple"] == validity["field_intensity_ripple"]


W_STOP = math.sqrt(1 + (100.0 / (math.pi * 1.0**2 / (532.0 * NM))) ** 2)  # 1/e^2 radius of the 1 mm waist at the stop, mm


def _cut_share(opening_mm: float, radius_mm: float = W_STOP) -> float:
    """Closed form: the share of a Gaussian beam's power beyond a centred round opening."""

    return math.exp(-2 * opening_mm**2 / radius_mm**2)


def test_a_cut_below_the_runs_warning_share_keeps_tem00_and_is_stated_in_numbers():
    """A stop of 2.5 beam radii cuts exp(-12.5) = 3.7e-6 of the beam, far below the trace's warning share (1e-3): the beam stays
    TEM00, and the cut is stated as numbers. Independent references: the closed form of the cut share, and the ripple figure
    against a numerical Fresnel integral of the truncated beam on its axis."""

    a = 2.5
    results = _results(_stop_project({"opening_radius_mm": str(a)}))
    document = results.handoff("after")
    validity = document["beams"][0]["validity"]
    cut = _cut_share(a)
    assert 1e-6 < cut < 1e-5
    assert validity["beam_parameters"] == "TEM00" and validity["tem00_field_applicable"] is True
    (row,) = validity["truncated_upstream"]
    assert row["surface_id"] == "STOP/stop" and row["warned"] is False
    # (the ledger books the share to 1e-12 of the arriving power)
    assert row["power_fraction"] == pytest.approx(cut, abs=2e-12, rel=1e-6)
    assert validity["truncated_power_fraction"] == row["power_fraction"]
    ripple = validity["field_intensity_ripple"]
    assert ripple == pytest.approx(2 * math.sqrt(cut) + cut, rel=1e-6)
    assert "gaussian_clip_warn_fraction" in validity["statement"] and "Unclipped" not in validity["statement"]
    assert document["schema_version"] == "1.2"
    assert document["numerical_settings"] == {"gaussian_clip_warn_fraction": 1e-3, "gaussian_scope_divergence_rad": 0.3}
    assert [i["code"] for i in validity["issues"]] == []  # the trace did not warn here

    # the ripple figure: |U(0, z)|^2 of the hard-cut Gaussian over that of the whole one, by a numerical Fresnel integral
    # U(0, z) ~ Integral_0^a exp(-r^2 / w^2) exp(i pi r^2 / (lambda z)) 2 r dr, over distances where the edge wave turns
    # through every phase; the largest deviation from 1 reaches the stated figure and never exceeds it
    lam = 532.0 * NM
    r = np.linspace(0.0, a, 400001)
    whole = np.linspace(0.0, 8.0, 1280001)

    def on_axis(radii: np.ndarray, z: float) -> complex:
        values = np.exp(-radii**2 / W_STOP**2) * np.exp(1j * np.pi * radii**2 / (lam * z)) * 2 * radii
        return complex(np.sum(0.5 * (values[1:] + values[:-1])) * (radii[1] - radii[0]))

    deviations = [abs(abs(on_axis(r, z)) ** 2 / abs(on_axis(whole, z)) ** 2 - 1.0) for z in np.linspace(2000.0, 12000.0, 41)]
    assert max(deviations) <= ripple * (1 + 1e-3)
    assert max(deviations) >= 0.9 * ripple
    # the focal peak of the cut beam against the untruncated beam of the same power, (1 - sqrt f) / (1 + sqrt f): within it too
    cut_amplitude = float(np.sum(np.exp(-r**2 / W_STOP**2) * 2 * r) * (r[1] - r[0]))
    whole_amplitude = float(np.sum(np.exp(-whole**2 / W_STOP**2) * 2 * whole) * (whole[1] - whole[0]))
    focal_ratio = (cut_amplitude / whole_amplitude) ** 2 / (1 - cut)
    assert abs(focal_ratio - 1.0) <= ripple and abs(focal_ratio - 1.0) == pytest.approx(2 * math.sqrt(cut), rel=2e-2)


def test_the_truncation_rule_is_the_runs_own_warning_share():
    """A stop of 1.85 beam radii cuts 1.07e-3 of the beam: just above the default warning share, so the beam is
    TRUNCATED_GAUSSIAN. A Project that sets gaussian_clip_warn_fraction to 2e-3 reads the same beam as TEM00 with the same cut
    share, and the document states the limit in force."""

    a = 1.85
    cut = _cut_share(a)
    assert 1e-3 < cut < 1.2e-3
    default = _results(_stop_project({"opening_radius_mm": str(a)})).handoff("after")["beams"][0]["validity"]
    assert default["beam_parameters"] == "TRUNCATED_GAUSSIAN" and default["tem00_field_applicable"] is False
    assert default["truncated_upstream"][0]["warned"] is True
    assert default["truncated_upstream"][0]["power_fraction"] == pytest.approx(cut, rel=1e-6)
    assert [i["code"] for i in default["issues"]] == ["GAUSSIAN_TRUNCATION"]
    doc = _stop_project({"opening_radius_mm": str(a)})
    doc["numerical_settings"] = {"gaussian_clip_warn_fraction": 2e-3}
    raised = _results(doc).handoff("after")
    validity = raised["beams"][0]["validity"]
    assert raised["numerical_settings"]["gaussian_clip_warn_fraction"] == 2e-3
    assert validity["beam_parameters"] == "TEM00" and validity["tem00_field_applicable"] is True
    assert validity["truncated_upstream"][0]["warned"] is False
    assert validity["truncated_upstream"][0]["power_fraction"] == pytest.approx(cut, rel=1e-6)
    assert validity["field_intensity_ripple"] == pytest.approx(2 * math.sqrt(cut) + cut, rel=1e-6)


def test_a_beam_is_judged_by_its_own_cut_not_by_another_beams_warning_at_the_same_stop():
    """Two beams through one stop of 2.5 mm: the 1 mm beam loses 3.7e-6, the 1.6 mm beam 7.6e-3. The trace warns at the stop
    for the wide beam only; the narrow beam's handoff stays TEM00 although a GAUSSIAN_TRUNCATION finding stands at the surface
    it passed."""

    doc = _stop_project({"opening_radius_mm": "2.5"})
    wide = source("WIDE", position=(0.0, 0.0, 0.0), waist_mm=1.6, wavelength_nm=532.0)
    doc["components"].insert(1, wide)
    beams = {beam["source_id"]: beam for beam in _results(doc).handoff("after")["beams"]}
    narrow_validity, wide_validity = beams["SRC"]["validity"], beams["WIDE"]["validity"]
    w_wide = math.sqrt(1.6**2 + (100.0 * 532.0 * NM / (math.pi * 1.6)) ** 2)
    assert wide_validity["beam_parameters"] == "TRUNCATED_GAUSSIAN"
    assert wide_validity["truncated_upstream"][0]["power_fraction"] == pytest.approx(_cut_share(2.5, w_wide), rel=1e-6)
    assert [i["code"] for i in wide_validity["issues"]] == ["GAUSSIAN_TRUNCATION"]
    assert narrow_validity["beam_parameters"] == "TEM00" and narrow_validity["tem00_field_applicable"] is True
    assert narrow_validity["truncated_upstream"][0]["warned"] is False
    assert narrow_validity["truncated_upstream"][0]["power_fraction"] == pytest.approx(_cut_share(2.5), abs=2e-12, rel=1e-6)
    # the finding of the wide beam stands at the surface the narrow beam passed, and is listed with it
    assert [i["code"] for i in narrow_validity["issues"]] == ["GAUSSIAN_TRUNCATION"]


def test_each_pass_through_a_stop_states_its_own_cut_share():
    """The beam passes a 2.5 mm stop at 100 mm, a mirror at 200 mm returns it through the stop: two cuts on one path at one
    surface. Each row's share is that pass's own -- exp(-2 a^2 / w^2) with the beam radius at the stop after 100 mm and after
    300 mm (closed forms) -- so the second is taken against the power arriving the second time, not the first."""

    coating = next(c for c in catalog_document()["coatings"] if c["kind"] == "DIELECTRIC_HIGH_REFLECTOR")
    mirror = {"model_id": "m", "model": sequence([surface(1, "0", clear="12", traced="REFLECTED", coating_id=coating["coating_id"])], [], edge="12.7")}
    doc = project([source(position=(0.0, 0.0, 0.0), waist_mm=1.0, wavelength_nm=532.0),
                   declared_component("STOP", "aperture", "stop", (100.0, 0.0, 0.0), ALONG_X),
                   declared_component("M", "mirror", "m", (200.0, 0.0, 0.0), AGAINST_X)],
                  [plane("back", (50.0, 0.0, 0.0), radius_mm=20.0)])
    doc = with_models(doc, [stop_model("stop", edge_radius_mm="12.7", opening_radius_mm="2.5"), mirror], None, [coating])
    results = _results(doc)
    (returned,) = [beam for beam in results.handoff("back")["beams"] if beam["crossing_index"] == 1]
    rows = returned["validity"]["truncated_upstream"]
    rayleigh = math.pi * 1.0**2 / (532.0 * NM)
    shares = [math.exp(-2 * 2.5**2 / (1 + (z / rayleigh) ** 2)) for z in (100.0, 300.0)]
    assert [row["surface_id"] for row in rows] == ["STOP/stop", "STOP/stop"] and len({row["path_id"] for row in rows}) == 1
    assert [row["power_fraction"] for row in rows] == pytest.approx(shares, rel=1e-6, abs=2e-12)
    assert returned["validity"]["truncated_power_fraction"] == pytest.approx(sum(shares), rel=1e-6, abs=4e-12)
    assert all(row["warned"] is False for row in rows) and returned["validity"]["beam_parameters"] == "TEM00"
    # the first crossing of the plane, before the stop, has no cut upstream
    (outgoing,) = [beam for beam in results.handoff("back")["beams"] if beam["crossing_index"] == 0]
    assert outgoing["validity"]["truncated_upstream"] == [] and outgoing["validity"]["truncated_power_fraction"] == 0.0


def test_a_withheld_gaussian_and_an_embedded_gaussian_are_stated():
    severe = _results(_stop_project({"opening_radius_mm": "0.5"})).handoff("after")["beams"][0]
    assert severe["gaussian"] is None and severe["validity"]["beam_parameters"] == "UNAVAILABLE"
    assert severe["power_W"] > 0
    multimode = _results(_stop_project({"opening_radius_mm": "8"}, m2=1.5)).handoff("pupil")["beams"][0]
    assert multimode["validity"]["beam_parameters"] == "EMBEDDED_GAUSSIAN"
    assert "M^2 alone does not determine" in multimode["validity"]["statement"]


# ------------------------------------------------------------------------------------------------ element outline


def test_element_outline_is_transformed_into_the_plane_frame():
    """A decentred slit (local centre (0.4, -0.3) mm, half widths 0.8 x 0.5 mm) 1 mm behind a plane turned 90 degrees about the
    beam: in that plane u = design +Z, v = -design Y, so the slit centre is at (u, v, n) = (-0.3, -0.4, 1)."""

    turned = quat_multiply(ALONG_X, turn_about_z(SQRT_HALF, SQRT_HALF))
    doc = _stop_project({"opening_half_widths_mm": ["0.8", "0.5"], "opening_center_mm": ["0.4", "-0.3"]}, plane_quaternion=turned)
    doc["components"][0]["source_model"]["waist_radius_mm"] = 0.2
    outline = _results(doc).handoff("pupil", element="STOP")["element"]
    assert outline["instance_id"] == "STOP"
    (surface,) = outline["surfaces"]
    assert surface["shape"] == "rectangle" and surface["half_width_mm"] == pytest.approx(0.8) and surface["half_height_mm"] == pytest.approx(0.5)
    assert surface["center_uvn_mm"] == pytest.approx([-0.3, -0.4, 1.0], abs=1e-12)
    assert surface["normal_uvn"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    assert surface["u_s_uvn"] == pytest.approx([0.0, -1.0, 0.0], abs=1e-12)  # the slit's local X (design +Y) is -v here
    corners = sorted(tuple(round(c, 9) for c in corner) for corner in surface["corners_uvn_mm"])
    expected = sorted((round(-0.3 + sv * 0.5, 9), round(-0.4 - su * 0.8, 9), 1.0) for su in (-1, 1) for sv in (-1, 1))
    assert corners == expected
    assert surface["in_plane"] is False
    coplanar = _results(_stop_project({"opening_radius_mm": "8"}, plane_x=100.0)).handoff("pupil", element="STOP")["element"]
    assert coplanar["surfaces"][0]["in_plane"] is True and coplanar["surfaces"][0]["radius_mm"] == pytest.approx(8.0)


def test_unknown_element_and_monitor_are_refused():
    from os_optics.results import ResultsError

    results = _results(_stop_project({"opening_radius_mm": "8"}))
    with pytest.raises(ResultsError) as info:
        results.handoff("pupil", element="NOPE")
    assert info.value.code == "ELEMENT_NOT_FOUND" and "STOP" in info.value.message
    with pytest.raises(ResultsError) as info:
        results.handoff("nope")
    assert info.value.code == "MONITOR_NOT_FOUND"


# ------------------------------------------------------------------------------------------------ json_api, CLI, export


def test_json_api_cli_and_export_give_the_same_handoff(tmp_path):
    doc = _stop_project({"opening_radius_mm": "1.2"})
    reply = json_api.run_project(doc, output_dir=str(tmp_path / "run"))
    query = {"kind": "handoff", "monitor": "pupil", "element": "STOP"}
    via_api = json_api.run_results(reply["files"], query)
    assert via_api["ok"] is True
    direct = load_run(tmp_path / "run").handoff("pupil", element="STOP")
    assert json.loads(json.dumps(via_api["result"])) == json.loads(json.dumps(direct))
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli_main(["results", "handoff", str(tmp_path / "run"), "--monitor", "pupil", "--element", "STOP"])
    assert code == 0 and json.loads(out.getvalue()) == json.loads(json.dumps(direct))
    exported = json.loads(load_run(tmp_path / "run").export("handoff", "json", monitor="pupil", element="STOP"))
    assert exported["kind"] == "handoff" and exported["summary"]["element"]["instance_id"] == "STOP"
    assert exported["summary"]["conventions"]["normalization"].startswith("|E|^2 in W/mm^2")
    assert exported["rows"][0]["q_x_im_mm"] == pytest.approx(direct["beams"][0]["gaussian"]["q_x_mm"][1])
    csv_path = tmp_path / "handoff.csv"
    with redirect_stdout(io.StringIO()):
        code = cli_main(["export", "handoff", str(tmp_path / "run"), "--monitor", "after", "--out", str(csv_path)])
    after = load_run(tmp_path / "run").handoff("after")["beams"][0]["validity"]
    # (truncated_at, the cut share and its ripple figure; power_out_of_scope_at, power_taken_before_at, model_scope, value_basis:
    # none)
    assert code == 0 and csv_path.read_text(encoding="utf-8").splitlines()[1].endswith(
        f"TRUNCATED_GAUSSIAN,false,STOP/stop,{after['truncated_power_fraction']!r},{after['field_intensity_ripple']!r},,,,")
    refused = json_api.run_results(reply["files"], {"kind": "handoff", "monitor": "pupil", "element": "NOPE"})
    assert refused["ok"] is False and refused["error"]["code"] == "ELEMENT_NOT_FOUND"
    identity = load_run(tmp_path / "run").identity()
    assert identity["engine_input_hash"] == identity["input_hash"] == json_api.check_project(doc)["engine_input_hash"]


# ------------------------------------------------------------------------------------------------ phase behind a thin lens


def test_phase_and_optical_path_are_withheld_behind_an_ideal_thin_lens():
    """rv2 M5: two beams at y = 0 and y = 2 mm through an ideal f = 100 mm lens to its focus. Fermat: equal optical paths to the
    focus, but the Engine's chief-ray sums differ by 0.02 mm (118 rad) because the lens states no optical path. The handoff
    withholds phase and optical path length there (as the network export refuses PHASE_UNDEFINED_IDEAL_LENS); q, power and
    polarization stay. A plane before the lens keeps both."""

    from declared_helpers import thin_lens_model

    doc = project([source("S0", position=(0.0, 0.0, 0.0), waist_mm=0.3, wavelength_nm=1064.0),
                   source("S2", position=(0.0, 2.0, 0.0), waist_mm=0.3, wavelength_nm=1064.0),
                   declared_component("L", "lens", "thin", (100.0, 0.0, 0.0), ALONG_X)],
                  [plane("before", (50.0, 0.0, 0.0), radius_mm=10.0), plane("focus", (200.0, 0.0, 0.0), radius_mm=10.0)])
    results = _results(with_models(doc, [thin_lens_model("thin", "100", clear="10", edge="12.7")]))
    behind = results.handoff("focus")
    assert len(behind["beams"]) == 2
    for beam in behind["beams"]:
        assert beam["phase_rad"] is None and beam["optical_path_length_mm"] is None
        assert beam["validity"]["phase_defined"] is False and "thin lens" in beam["validity"]["statement"]
        assert beam["gaussian"] is not None and beam["power_W"] > 0.0
    assert "Behind an ideal or tabulated thin lens" in behind["conventions"]["phase_reference"]
    before = results.handoff("before")
    for beam in before["beams"]:
        assert beam["validity"]["phase_defined"] is True and beam["optical_path_length_mm"] == pytest.approx(50.0, abs=1e-9)
    rows, _meta = results.table("handoff", monitor="focus")
    assert all(row["phase_defined"] is False and row["phase_rad"] is None and row["optical_path_length_mm"] is None for row in rows)
