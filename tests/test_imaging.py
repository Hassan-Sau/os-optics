"""Imaging analysis of a run's image plane (``os_optics.results.imaging``): point spread function, MTF and encircled energy from a dense re-trace,
against references derived here, with tolerances set before the first run.

Stand-ins for the external A1 / B2 / C1 cases (their projects are not in this repository): A1-like, the double-Gauss fixture
(``tests/fixtures/prescriptions/double_gauss_6.csv``, f about 52 mm, about F/3 with its iris) on axis and at a 5 deg field; B2-like, the Cassegrain of
``tests/test_paraxial_mirrors_compounds.py`` (spherical mirrors, EFL 1200 mm, the focusing pass is the plane's third crossing); C1-like, an F/1.2
plano-hyperbolic singlet written here (perfect on axis for collimated light: conic -n^2), above the scalar limit. Most source beams underfill the
stop; a beam that fills it arrives on its pupil-sample basis and is analysed (``test_a_stop_filling_beam_arrives_and_is_analysed``); a field
whose chief ray falls beyond an aperture's opening goes on as its pupil rays that pass (``tests/test_promoted_reference.py``).

References: an F/8 plano-hyperbolic singlet (aberration-free, scalar) against the Airy pattern -- Strehl 1 (1e-3), MTF
``(2 / pi) (acos(v) - v sqrt(1 - v^2))`` (0.01), 80 % encircled-energy radius from ``1 - J0^2 - J1^2`` (1 %); a thin ideal lens refuses the diffraction
quantities (``RAY_OPL_UNDEFINED``) and keeps the geometric spot; the geometric MTF stays out of its model's scope below one wave RMS.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize, special

from os_optics.cli import main as cli_main
from os_optics.errors.status import QuantityState
from os_optics.local import json_api, run_optical_project
from os_optics.prescription import import_prescription
from os_optics.results import ImagingError, ResultsError, load_run

from declared_helpers import declared_component, thin_lens_model, with_models
from local_helpers import plane, project, quat_multiply, run, source, turn_about_z

ENGINE = Path(__file__).resolve().parents[1]
NBK7 = (1.03961212, 0.231792344, 1.01046945, 0.00600069867, 0.0200179144, 103.560653)


def _gallery():
    spec = importlib.util.spec_from_file_location("lens_prescription_gallery_for_imaging", ENGINE / "scripts" / "lens_prescription_gallery.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _nbk7(wavelength_nm: float) -> float:
    l2 = (wavelength_nm * 1e-3) ** 2
    b1, b2, b3, c1, c2, c3 = NBK7
    return math.sqrt(1.0 + b1 * l2 / (l2 - c1) + b2 * l2 / (l2 - c2) + b3 * l2 / (l2 - c3))


def _hyperbolic(f_number: float, focal: float, wavelength_nm: float) -> str:
    """A plano-hyperbolic N-BK7 singlet, flat side to the light: rear radius -(n - 1) f, conic -n^2 (a perfect focus on axis)."""

    n = _nbk7(wavelength_nm)
    a = focal / (2.0 * f_number)
    return "\n".join([
        "#! format = csv-surface-table/1", "#! name = plano-hyperbolic singlet", "#! units = mm", "[surfaces]",
        "surface,type,comment,radius,thickness,material,semi_diameter,mech_semi_diameter,conic,a4,a6,a8,cylinder_axis,stop",
        "OBJ,STANDARD,object at infinity,inf,inf,,,,,,,,,",
        f"1,STANDARD,flat front,inf,3,N-BK7,{a:.6f},{a + 0.5:.6f},,,,,,yes",
        f"2,EVENASPH,hyperbolic rear,{-(n - 1.0) * focal:.9f},{focal:.6f},,{a:.6f},{a + 0.5:.6f},{-n * n:.9f},,,,,",
        "IMA,STANDARD,image plane,,,,,,,,,,,", "", "[materials]", "name,kind,b1,b2,b3,c1,c2,c3,min_nm,max_nm,display_name",
        "N-BK7,SELLMEIER," + ",".join(str(v) for v in NBK7) + ",300,2500,N-BK7 (fixture Sellmeier)", ""])


def _lens_project(text: str, waist_mm: float, field_deg: tuple[float, ...] = ()) -> dict:
    gallery = _gallery()
    imported = {"ok": True, **import_prescription(text, model_id="lens", glass_source="inline").as_dict()}
    doc = gallery.project_compound(imported)
    # the closed-form designs here (a plano-hyperbolic singlet's perfect focus: rear radius -(n - 1) f, conic -n^2) take the
    # medium around the lens to have index 1: the Project operates in vacuum (a native Project defaults to air, n ~ 1.00027)
    doc["environment"] = {"pressure_pa": "0"}
    src = doc["components"][0]
    src["source_model"]["waist_radius_mm"] = waist_mm
    for index, angle in enumerate(field_deg):
        tilted = copy.deepcopy(src)
        tilted["instance_id"] = f"FIELD{index + 1}"
        x = src["nominal_pose"]["position_mm"][0]
        half = math.radians(angle) / 2.0
        tilted["nominal_pose"] = {"position_mm": [x, x * math.tan(math.radians(angle)), 0.0],
                                  "quaternion_wxyz": list(quat_multiply(turn_about_z(math.cos(half), math.sin(half)), tuple(src["nominal_pose"]["quaternion_wxyz"])))}
        doc["components"].append(tilted)
    return doc


def value(doc, quantity_id: str) -> float:
    q = doc.quantity(quantity_id)
    assert q.state is QuantityState.KNOWN, (quantity_id, q.state, q.reason, q.message)
    return float(q.value)


def _line(doc, prefix: str) -> str:
    keys = {q.quantity_id.rsplit(".", 1)[0] for q in doc.quantities if q.quantity_id.startswith(prefix)}
    assert len(keys) == 1, keys
    return keys.pop() + "."


# ------------------------------------------------------------------------------------------------- an aberration-free lens against Airy


@pytest.fixture(scope="module")
def f8():
    results = load_run(run(_lens_project(_hyperbolic(8.0, 40.0, 587.5618), waist_mm=1.0)))
    return results.imaging("image", {"psf_samples": 64})


def test_an_aberration_free_lens_gives_the_airy_pattern(f8):
    doc = f8
    p = _line(doc, "SRC#")
    na = value(doc, p + "image_space_na")
    assert na == pytest.approx(1.0 / 16.0, rel=1e-2)  # the marginal sine of the fitted ray grid, a / sqrt(f^2 + a^2) of the lens
    # (the lens's rim cut the launch disc at 3 beam radii: the pupil is the system's aperture)
    assert doc.arrays[p + "rays"]["pupil"] == "APERTURE" and doc.arrays[p + "rays"]["pupil_search_radii"] == 3.0
    assert value(doc, p + "strehl_ratio") == pytest.approx(1.0, abs=1e-3)
    assert value(doc, p + "opd_rms_waves") < 0.01
    lam_mm = 587.5618e-6
    frequencies = np.array(doc.arrays[p + "frequencies_cyc_per_mm"])
    cutoff = 2.0 * na / lam_mm
    v = np.clip(frequencies / cutoff, 0.0, 1.0)
    analytic = 2.0 / math.pi * (np.arccos(v) - v * np.sqrt(1.0 - v * v))
    for name in ("mtf_tangential", "mtf_sagittal", "diffraction_limited_mtf_tangential"):
        curve = np.array([np.nan if x is None else x for x in doc.arrays[p + name]])
        ok = np.isfinite(curve)
        assert ok.sum() > 20
        assert np.abs(curve[ok] - analytic[ok]).max() < 0.01, name
    k_na = 2.0 * math.pi * na / (lam_mm * 1e3)
    r80 = optimize.brentq(lambda x: 1.0 - special.j0(x) ** 2 - special.j1(x) ** 2 - 0.8, 0.5, 3.8) / k_na
    assert value(doc, p + "encircled_radius_80") == pytest.approx(r80, rel=0.01)
    assert value(doc, p + "diffraction_limited_encircled_radius_80") == pytest.approx(r80, rel=0.01)
    # the geometric MTF of a perfect lens is outside its regime (no wavefront error: diffraction sets the MTF). Its curve is
    # stated as what it is: the ray spot of a perfect lens is a point, so the geometric curve is 1 at every frequency -- far
    # above the diffraction MTF -- and it carries the regime check as an estimate's basis and decides no comparison
    for name in ("tangential", "sagittal"):
        geometric = doc.quantity(p + f"geometric_mtf_{name}")
        assert geometric.state is QuantityState.OUT_OF_MODEL_SCOPE and geometric.reason == "GEOMETRIC_REGIME_NOT_MET"
        curve = np.array(doc.arrays[geometric.values_ref])
        assert geometric.values_ref == p + f"geometric_mtf_{name}" and curve.shape == frequencies.shape and curve.min() > 0.999 and curve.max() <= 1.0 + 1e-12
        entry = geometric.value_basis
        (check,) = entry.validity
        assert entry.kind.value == "ESTIMATE" and not entry.omitted and not entry.assumed
        assert (check.parameter, check.minimum, check.within) == ("opd_rms_waves", 1.0, False) and check.observed == value(doc, p + "opd_rms_waves")
        assert geometric.depends_on == ("opd_rms_waves",) and geometric.convergence.converged is True
        assert not geometric.decides(0.5) and "mtf_" + name in geometric.message
    model = next(m for m in doc.models if m.model_id == p + "geometric_mtf")
    assert model.validity == (check,) and doc.analysis_version == "1.2"
    assert doc.settings["random_seed"] is None and doc.execution.complete
    for q in doc.quantities:
        if q.convergence is not None and q.convergence.converged is not None:
            assert q.convergence.converged, q.quantity_id


# ------------------------------------------------------------------------------------------------- stand-ins for A1, B2 and C1


@pytest.fixture(scope="module")
def double_gauss():
    text = (ENGINE / "tests" / "fixtures" / "prescriptions" / "double_gauss_6.csv").read_text(encoding="utf-8")
    results = load_run(run(_lens_project(text, waist_mm=3.0, field_deg=(5.0,))))
    return results, results.imaging("image")


def test_a1_like_double_gauss_has_a_supported_mtf_and_psf_at_every_field(double_gauss):
    _results, doc = double_gauss
    assert {q.scope.source_id for q in doc.quantities} == {"SRC", "FIELD1"}
    for source in ("SRC#", "FIELD1#"):
        p = _line(doc, source)
        na = value(doc, p + "image_space_na")
        assert 0.14 < na < 0.2  # about F/3 with the fixture's iris, below the scalar limit
        assert doc.arrays[p + "psf"]["kernel"] == "scalar"
        strehl = value(doc, p + "strehl_ratio")
        assert 0.0 < strehl <= 1.0
        for name in ("mtf_tangential", "mtf_sagittal"):
            q = doc.quantity(p + name)
            assert q.state is QuantityState.KNOWN and q.values_ref == p + name and q.convergence.converged
            curve = [x for x in doc.arrays[p + name] if x is not None]
            reference = [x for x in doc.arrays[p + "diffraction_limited_" + name] if x is not None]
            assert curve[0] == pytest.approx(1.0, abs=0.02)  # the light beyond the map is missing from zero frequency only
            assert all(c <= r + 0.02 for c, r in zip(curve, reference))  # aberrations lower the MTF below the same pupil's limit
        assert value(doc, p + "encircled_radius_80") > value(doc, p + "diffraction_limited_encircled_radius_80")
    model = next(m for m in doc.models if m.model_id.startswith("SRC#") and m.model_id.endswith("diffraction"))
    assert all(c.within for c in model.validity)
    # off axis the tangential and sagittal MTF differ by the aberrations
    off = _line(doc, "FIELD1#")
    assert np.abs(np.array(doc.arrays[off + "mtf_tangential"][1:10]) - np.array(doc.arrays[off + "mtf_sagittal"][1:10])).max() > 1e-3


def test_on_axis_the_tangential_and_sagittal_mtf_exchange_with_the_polarization(double_gauss):
    """On axis the lens is rotationally symmetric; the pupil's irradiance is not quite, for linearly polarized light: each ray's
    field carries its interfaces' s and p transmissions. Turning the polarization by 90 deg turns the pupil, so the tangential
    and sagittal MTF exchange."""

    _results, doc = double_gauss
    text = (ENGINE / "tests" / "fixtures" / "prescriptions" / "double_gauss_6.csv").read_text(encoding="utf-8")
    turned = _lens_project(text, waist_mm=3.0)
    turned["components"][0]["source_model"]["polarization"] = {"kind": "LINEAR", "angle_deg": 90}
    other = load_run(run(turned)).imaging("image")
    on, on_turned = _line(doc, "SRC#"), _line(other, "SRC#")
    assert doc.arrays[on + "mtf_tangential"] == pytest.approx(other.arrays[on_turned + "mtf_sagittal"], abs=1e-6)
    assert doc.arrays[on + "mtf_sagittal"] == pytest.approx(other.arrays[on_turned + "mtf_tangential"], abs=1e-6)


def test_b2_like_cassegrain_reads_the_focusing_pass():
    """At its full F/10 aperture the spherical-mirror Cassegrain has about 2.5 waves RMS of spherical aberration: the geometric MTF holds and
    the diffraction quantities are refused for the ray sampling; stopped down to F/100 it is diffraction-limited."""
    from test_paraxial_mirrors_compounds import _cassegrain

    results = load_run(run(_cassegrain(waist_mm=5.0)))
    full = results.imaging("image", {"crossing_index": 2, "launch_grid": 96})  # the third crossing: towards M1 after both mirrors
    p = _line(full, "SRC#")
    assert full.arrays[p + "rays"]["crossing_index"] == 2
    assert value(full, p + "image_space_na") == pytest.approx(0.05, rel=0.05)  # F/10: the 60 mm clear radius of M1 at EFL 1200 mm
    assert value(full, p + "opd_rms_waves") > 1.0
    geometric = full.quantity(p + "geometric_mtf_tangential")
    assert geometric.state is QuantityState.KNOWN and geometric.convergence.converged
    assert full.quantity(p + "mtf_tangential").reason == "PUPIL_UNDERSAMPLED"
    assert value(full, p + "geometric_spot_rms_radius") > 0.0
    stopped = results.imaging("image", {"crossing_index": 2, "launch_radius_beam_radii": 1.2})
    q = _line(stopped, "SRC#")
    assert value(stopped, q + "strehl_ratio") == pytest.approx(1.0, abs=1e-3)
    assert stopped.quantity(q + "mtf_tangential").state is QuantityState.KNOWN


def test_c1_like_f12_singlet_uses_the_vector_kernel():
    results = load_run(run(_lens_project(_hyperbolic(1.2, 10.0, 587.5618), waist_mm=1.0)))
    doc = results.imaging("image", {"launch_grid": 48, "psf_samples": 64})
    p = _line(doc, "SRC#")
    assert value(doc, p + "image_space_na") > 0.3
    assert doc.arrays[p + "psf"]["kernel"] == "vector"
    assert "vector" in next(m for m in doc.models if m.model_id == p + "diffraction").name
    assert value(doc, p + "strehl_ratio") == pytest.approx(1.0, abs=0.01)
    assert doc.quantity(p + "mtf_tangential").state is QuantityState.KNOWN


# ------------------------------------------------------------------------------------------------- refusals and surfaces


def _covered_singlet(f_number: float, focal: float, wavelength_nm: float, defocus_mm: float, plate_mm: float = 0.5, gap_after_mm: float = 0.01) -> str:
    """The plano-hyperbolic singlet with a plane N-BK7 cover plate ``plate_mm`` thick whose back face lies ``gap_after_mm`` before the
    image plane, the plane ``defocus_mm`` from the focus (negative: before it). The plate moves the focus by ``t (1 - 1 / n)``."""

    n = _nbk7(wavelength_nm)
    a = focal / (2.0 * f_number)
    gap = focal + plate_mm * (1.0 - 1.0 / n) + defocus_mm - plate_mm - gap_after_mm
    return "\n".join([
        "#! format = csv-surface-table/1", "#! name = covered plano-hyperbolic singlet", "#! units = mm", "[surfaces]",
        "surface,type,comment,radius,thickness,material,semi_diameter,mech_semi_diameter,conic,a4,a6,a8,cylinder_axis,stop",
        "OBJ,STANDARD,object at infinity,inf,inf,,,,,,,,,",
        f"1,STANDARD,flat front,inf,3,N-BK7,{a:.6f},{a + 0.5:.6f},,,,,,yes",
        f"2,EVENASPH,hyperbolic rear,{-(n - 1.0) * focal:.9f},{gap:.9f},,{a:.6f},{a + 0.5:.6f},{-n * n:.9f},,,,,",
        f"3,STANDARD,plate front,inf,{plate_mm},N-BK7,{a:.6f},{a + 0.5:.6f},,,,,,",
        f"4,STANDARD,plate back,inf,{gap_after_mm},,{a:.6f},{a + 0.5:.6f},,,,,,",
        "IMA,STANDARD,image plane,,,,,,,,,,,", "", "[materials]", "name,kind,b1,b2,b3,c1,c2,c3,min_nm,max_nm,display_name",
        "N-BK7,SELLMEIER," + ",".join(str(v) for v in NBK7) + ",300,2500,N-BK7 (fixture Sellmeier)", ""])


def test_the_wavefront_is_referred_to_infinity_whatever_the_last_gap():
    """A cover plate 10 um before the image plane makes the sphere through the last surface 10 um in radius, while 0.5 mm of defocus at
    F/8 spreads the rays over about 30 um: rays miss that sphere (the field used to lose its wavefront: RAY_MISSES_REFERENCE). The
    wavefront at infinity (``n`` times each ray's projection on its direction removed from its optical path) matches the defocus
    wavefront ``W_rms = dz NA^2 / (4 sqrt 3)`` (the plate's own spherical aberration is below 1e-3 waves at this NA) within 2 %, for the
    plate 10 um and 0.5 mm before the plane alike."""

    wavelength, focal, f_number, defocus = 587.5618, 40.0, 8.0, -0.5
    results = load_run(run(_lens_project(_covered_singlet(f_number, focal, wavelength, defocus), waist_mm=1.0)))
    doc = results.imaging("image", {"launch_grid": 48, "psf_samples": 64})
    p = _line(doc, "SRC#")
    rays = doc.arrays[p + "rays"]
    assert rays["reference_sphere"] == "INFINITY"
    assert rays["opd_rms_waves_on_last_surface_sphere"] is None  # (a ray's line passes beyond the 10 um sphere)
    na = value(doc, p + "image_space_na")
    expected = abs(defocus) * na * na / (4.0 * math.sqrt(3.0)) / (wavelength * 1e-6)
    assert value(doc, p + "opd_rms_waves") == pytest.approx(expected, rel=0.02)
    # with the plate 0.5 mm before the plane every ray meets the sphere through it, but that radius equals the focus offset: on
    # it the wavefront reads about twice the defocus (the finite reference was the value before), at infinity the defocus
    wide = load_run(run(_lens_project(_covered_singlet(f_number, focal, wavelength, defocus, gap_after_mm=0.5), waist_mm=1.0))).imaging(
        "image", {"launch_grid": 48, "psf_samples": 64})
    w = _line(wide, "SRC#")
    assert value(wide, w + "opd_rms_waves") == pytest.approx(expected, rel=0.02)
    assert wide.arrays[w + "rays"]["opd_rms_waves_on_last_surface_sphere"] > 1.8 * expected
    # in focus the sphere through the last surface holds the spot: the usual reference, and the singlet's Airy pattern
    focused = load_run(run(_lens_project(_covered_singlet(f_number, focal, wavelength, 0.0, gap_after_mm=0.5), waist_mm=1.0))).imaging(
        "image", {"launch_grid": 48, "psf_samples": 64})
    q = _line(focused, "SRC#")
    assert focused.arrays[q + "rays"]["opd_rms_waves_on_last_surface_sphere"] == pytest.approx(value(focused, q + "opd_rms_waves"), abs=1e-3)
    assert value(focused, q + "strehl_ratio") > 0.95


def test_a_thin_ideal_lens_keeps_the_geometric_spot_and_refuses_the_diffraction_quantities():
    doc = project([source(position=(-50.0, 0.0, 0.0), waist_mm=1.0)], [plane("image", (100.0, 0.0, 0.0), radius_mm=20.0)])
    doc["components"].append(declared_component("L", "lens", "thin", (0.0, 0.0, 0.0)))
    doc = with_models(doc, [thin_lens_model("thin", "100", clear="2")])
    results = load_run(run(doc))
    out = results.imaging("image", {"launch_grid": 32})
    p = _line(out, "SRC#")
    assert value(out, p + "geometric_spot_rms_radius") >= 0.0
    for name in ("strehl_ratio", "mtf_tangential", "opd_rms_waves"):
        q = out.quantity(p + name)
        assert q.state is QuantityState.OUT_OF_MODEL_SCOPE and q.reason == "RAY_OPL_UNDEFINED", name
    # without a wavefront the geometric MTF's domain (one wave RMS or more) is not known: not a statement that it is not met.
    # The curve of the traced spot is stated with that check (within null) as an estimate's basis
    geometric = out.quantity(p + "geometric_mtf_tangential")
    assert geometric.state is QuantityState.UNKNOWN and geometric.reason == "GEOMETRIC_REGIME_UNKNOWN"
    (check,) = geometric.value_basis.validity
    assert geometric.value_basis.kind.value == "ESTIMATE" and (check.parameter, check.observed, check.within) == ("opd_rms_waves", None, None)
    curve = np.array(out.arrays[geometric.values_ref])
    # (the same sum over the traced rays as in the regime: |sum_j p_j exp(-i 2 pi f (r_j - r_c))| is 1 at f = 0 and at most 1)
    assert curve[0] == pytest.approx(1.0, abs=1e-12) and curve.max() <= 1.0 + 1e-12 and curve.min() >= 0.0 and not geometric.decides(0.5)
    # the analysis ran to its end: what the field lacks is each quantity's own state, not the execution record's
    assert out.execution.complete and out.execution.outcome.value == "COMPLETED" and out.execution.reason == ""


def test_a_system_no_aperture_bounds_is_analysed_on_the_launch_disc_and_says_so():
    """A 0.1 mm beam through the f/8, 40 mm lens (clear radius 2.5 mm): the search for the system's aperture grows the launch disc
    to 15.2 beam radii (1.5 mm) and nothing cuts it. The analysis was one row, PUPIL_NOT_BOUNDED. It is now that of the disc of
    2 beam radii -- the document of a request that states launch_radius_beam_radii 2, number for number -- and every number says
    it stands on that stand-in pupil: UNKNOWN / PUPIL_NOT_BOUNDED beside the value, value_basis ESTIMATE with UNIFORM_PUPIL.
    Reference: the marginal sine of a 0.2 mm disc at 40 mm. A search that reaches the lens rim finds the aperture: KNOWN."""

    results = load_run(run(_lens_project(_hyperbolic(8.0, 40.0, 587.5618), waist_mm=0.1)))
    default = results.imaging("image", {"psf_samples": 64})
    stated = results.imaging("image", {"psf_samples": 64, "launch_radius_beam_radii": 2.0})
    p = _line(default, "SRC#")
    rays = default.arrays[p + "rays"]
    assert rays["pupil"] == "LAUNCH_DISC" and rays["launch_radius_beam_radii"] == 2.0 and rays["pupil_search_radii"] == pytest.approx(2.0 * 1.5 ** 5)
    assert stated.arrays[p + "rays"]["pupil"] == "LAUNCH_DISC" and stated.arrays[p + "rays"]["pupil_search_radii"] is None
    assert p + "field" not in {q.quantity_id for q in default.quantities} and default.execution.complete
    assert [q.quantity_id for q in default.quantities] == [q.quantity_id for q in stated.quantities]
    valued = 0
    for ours, theirs in zip(default.quantities, stated.quantities):
        assert (ours.value, ours.values_ref, ours.convergence) == (theirs.value, theirs.values_ref, theirs.convergence), ours.quantity_id
        if theirs.value is None and theirs.values_ref is None:
            assert (ours.state, ours.reason, ours.value_basis) == (theirs.state, theirs.reason, None), ours.quantity_id
            continue
        valued += 1
        if theirs.values_ref is not None:
            assert default.arrays[ours.values_ref] == stated.arrays[theirs.values_ref], ours.quantity_id
        (assumed,) = ours.value_basis.assumed
        assert ours.value_basis.kind.value == "ESTIMATE" and (assumed.input.value, assumed.value, assumed.unit) == ("UNIFORM_PUPIL", 2.0, "beam_radii"), ours.quantity_id
        assert not ours.decides(0.0) or ours.value is None
        if theirs.state is QuantityState.KNOWN:  # (a number of the caller's own pupil is the model's; of the stand-in it is not proven)
            assert theirs.value_basis is None and ours.state is QuantityState.UNKNOWN and ours.reason == "PUPIL_NOT_BOUNDED", ours.quantity_id
        else:  # (the geometric MTF below one wave: its regime's state and check stay, beside the stand-in)
            assert (ours.state, ours.reason) == (theirs.state, theirs.reason) and ours.value_basis.validity == theirs.value_basis.validity, ours.quantity_id
    assert valued >= 12
    na = default.quantity(p + "image_space_na")
    assert na.value == pytest.approx(0.2 / math.hypot(40.0, 0.2), rel=2e-2)  # the disc of 2 beam radii, focused at 40 mm
    assert default.quantity(p + "strehl_ratio").value == pytest.approx(1.0, abs=1e-3)
    assert "max_launch_radius_beam_radii" not in default.settings["request"]
    # a search that reaches the lens's rim (40 beam radii: 4 mm against the clear radius of 2.5 mm) finds the aperture
    wide = results.imaging("image", {"psf_samples": 64, "max_launch_radius_beam_radii": 40.0})
    assert wide.arrays[p + "rays"]["pupil"] == "APERTURE" and wide.settings["request"]["max_launch_radius_beam_radii"] == 40.0
    assert value(wide, p + "image_space_na") == pytest.approx(1.0 / 16.0, rel=1e-2) and wide.quantity(p + "image_space_na").value_basis is None
    with pytest.raises(ImagingError):
        results.imaging("image", {"max_launch_radius_beam_radii": 1.0})


def test_a_stop_filling_beam_arrives_and_is_analysed():
    """A Gaussian beam that fills the double Gauss's stop has no valid Gaussian state where the apertures cut it; the tracer follows
    its geometric rays on the pupil-sample basis, so the field arrives and its point spread function and MTF are supported."""

    text = (ENGINE / "tests" / "fixtures" / "prescriptions" / "double_gauss_6.csv").read_text(encoding="utf-8")
    results = load_run(run(_lens_project(text, waist_mm=8.0)))
    doc = results.imaging("image", {"fields": ["SRC"], "launch_grid": 32})
    line = _line(doc, "SRC#")
    assert doc.execution.complete
    assert line + "field" not in {q.quantity_id for q in doc.quantities}
    for name in ("strehl_ratio", "mtf_tangential", "mtf_sagittal", "encircled_radius_80"):
        assert doc.quantity(line + name).state is QuantityState.KNOWN, name
    assert 0.0 < doc.quantity(line + "strehl_ratio").value < 1.0


def test_invalid_requests_and_the_surfaces(f8, tmp_path, capsys):
    results = load_run(run(_lens_project(_hyperbolic(8.0, 40.0, 587.5618), waist_mm=1.0)))
    with pytest.raises(ImagingError) as exc:
        results.imaging("image", {"psf_samples": 63})
    assert exc.value.code == "IMAGING_INVALID"
    with pytest.raises(ResultsError):
        results.imaging("nowhere")
    run_dir = tmp_path / "run"
    run_optical_project(_lens_project(_hyperbolic(8.0, 40.0, 587.5618), waist_mm=1.0), output_dir=str(run_dir))
    request = tmp_path / "imaging.json"
    request.write_text(json.dumps({"launch_grid": 32, "psf_samples": 32}), encoding="utf-8")
    assert cli_main(["results", "imaging", str(run_dir), "--monitor", "image", "--request", str(request)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis"] == "IMAGING" and out["execution"]["outcome"] == "COMPLETED"
    request.write_text(json.dumps({"launch_grid": 3}), encoding="utf-8")
    assert cli_main(["results", "imaging", str(run_dir), "--monitor", "image", "--request", str(request)]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "IMAGING_INVALID"
    files = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    reply = json_api.run_results(files, {"kind": "imaging", "monitor": "image", "request": {"launch_grid": 32, "psf_samples": 32}})
    assert reply["ok"] is True and reply["result"]["analysis"] == "IMAGING"
    assert reply["result"]["quantities"] == out["quantities"]
    assert json_api.capabilities()["field_analyses"]["imaging"]["schema_kind"] == "imaging-request"
    assert json_api.json_schema("imaging-request")["schema"]["title"] == "ImagingRequestV1"
