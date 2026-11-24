"""Per-plane results computed from beams whose power is out of model scope are never stated as valid values.

A beam's power is out of model scope behind a variable attenuator whose density varies across it, a crystal whose
second-harmonic generation is out of model scope there (a pulse train), the side of a gradient-index rod, ...
(``ArrivalResult.power_out_of_scope_at``); the plane's own power is then ``OUT_OF_MODEL_SCOPE``. Before, the irradiance map
of such a plane read ``status OK`` and ``map_complete true`` with the chief ray's power (behind a pulsed doubler the
continuous-wave map, about 1.5e5 times too low; behind a rod's side exit 82 % too much), and so did everything computed from
the same beams: the detector response, the handoff (and the pupil and focal fields built from it), the coherent sums and the
two-beam quantities, the ghost powers, and the light a surface scatters out of such a beam (its plane read ``BOUNDED``).

The references are independent of the Engine: the light an attenuator of known density law passes (the Gaussian profile
integrated against ``10^-OD(x)``), the share of a beam reaching a gradient-index rod's side (``test_gradient_index``: RK4 rays
of the beam's Wigner function), a Lambertian scatterer's share on a coaxial disc (``a^2 / (a^2 + h^2)``) and the Gaussian
integral of a map's grid.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from local_helpers import PCX_F50, element, plane, project, run, source

from os_optics import EngineConfiguration, ValidationRequest, validate
from os_optics import builders as B
from os_optics.contracts import AbsorberInteraction, NeutralDensityInteraction, ProjectSnapshot, SurfaceScatter
from os_optics.native_library import add_to_project, build_native_model
from os_optics.results import load_run
from os_optics.results.irradiance import irradiance_map
from os_optics.results.model import ResultsError

OUT = "POWER_OUT_OF_MODEL_SCOPE"
ALONG_X = (0.5, 0.5, 0.5, 0.5)
W = 1.5  # (mm: the beam's radius at the attenuator; its Rayleigh range is 9 m)


def _attenuated_document(od_start: float, od_end: float) -> dict:
    """A 780 nm beam of 1.5 mm through a variable attenuator (``od_start`` at x = -6 mm to ``od_end`` at +6 mm, uncoated
    transmittance 1 beyond: 4 beam radii out) 20 mm behind the source, a plane P 120 mm behind it, a 50 mm plano-convex lens and
    an image plane IMG."""
    nd = build_native_model("filter.variable_nd", {"law": "LINEAR_GRADIENT", "od_start": repr(od_start), "od_end": repr(od_end),
                                                   "x_start_mm": "-6", "x_end_mm": "6", "width_mm": "40", "height_mm": "20",
                                                   "valid_band_nm": [400, 1100], "uncoated_transmittance": "1"})
    document = project([source(position=(-100.0, 0.0, 0.0), waist_mm=W, power_w=1.0), element("L", "lens", PCX_F50, (60.0, 0.0, 0.0))],
                       [plane("P", (40.0, 0.0, 0.0), radius_mm=20.0), plane("IMG", (112.0, 0.0, 0.0), radius_mm=20.0)])
    return add_to_project(document, nd, instance_id="ND", position_mm=(-80.0, 0.0, 0.0), quaternion_wxyz=ALONG_X)


def _attenuated(od_start: float, od_end: float):
    return load_run(run(_attenuated_document(od_start, od_end)))


def _passed_share(od_start: float, od_end: float) -> float:
    """The share of a Gaussian of radius ``W`` (1/e^2, centred) an attenuator of density ``od_start`` at x = -6 mm rising linearly to
    ``od_end`` at +6 mm (uncoated, transmittance 1, beyond) passes: its profile along x integrated against ``10^-OD(x)`` (Riemann,
    400001 points)."""
    x = np.linspace(-8.0 * W, 8.0 * W, 400001)
    od = np.where(np.abs(x) <= 6.0, od_start + (od_end - od_start) * (x + 6.0) / 12.0, 0.0)
    profile = np.sqrt(2.0 / math.pi) / W * np.exp(-2.0 * x * x / (W * W))
    return float(np.sum(10.0 ** -od * profile) * (x[1] - x[0]))


def test_every_result_of_a_plane_behind_a_varying_attenuator_is_out_of_model_scope():
    """OD 0 -> 8 over 12 mm across the beam: the chief ray's transmittance (OD 4, 1e-4) is not the share the light keeps (the
    reference: 1.94e-4, mostly through the thin side), so no map, detector response, handoff field, pupil field or export of P states it;
    the arrivals and monitors exports name the attenuator and the state. OD 2 -> 2.0001: every one is stated, the map's grid
    power the beam's (the Gaussian integral over the grid, 2.5 radii either side: 1 - 1e-6).

    Register row T-12, withheld: the chief ray's transmittance is not the mean transmittance over the beam's profile, which the
    attenuator step does not compute."""
    true_share = _passed_share(0.0, 8.0)
    results = _attenuated(0.0, 8.0)
    observation = results.result.observations[0]
    (arrival,) = observation.arrivals
    assert observation.quantity_state == "OUT_OF_MODEL_SCOPE" and arrival.power_out_of_scope_at == ("ND/attenuator",)
    assert true_share > 1.9 * arrival.beam.power_W  # (the chief ray's power, 1e-4, is not the light's)
    doc = results.irradiance_map("P")
    assert (doc["status"], doc["reason"], doc["power_out_of_scope_at"]) == ("UNAVAILABLE", OUT, ["ND/attenuator"])
    assert doc["quantity_state"] == "OUT_OF_MODEL_SCOPE" and "incoherent_W_per_mm2" not in doc
    detector = {"pixel_pitch_mm": [0.5, 0.5], "fill_factor": 1.0, "pixel_count": [4, 4]}
    response = results.detector_response("P", detector)
    assert (response["status"], response["reason"], response["upstream_reason"]) == ("UNAVAILABLE", "IRRADIANCE_UNAVAILABLE", OUT)
    (beam,) = results.handoff("P")["beams"]
    assert beam["power_W"] is None and beam["validity"]["beam_parameters"] == OUT and not beam["validity"]["tem00_field_applicable"]
    assert beam["validity"]["power_out_of_scope_at"] == ["ND/attenuator"]
    request = {"wavelength_nm": 780.0, "focal_length_mm": 100.0, "sampling": {"pixel_count": [64, 64], "pitch_mm": [0.2, 0.2]},
               "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}}
    field = results.pupil_field("P", request)
    assert (field["status"], field["reason"]) == ("UNAVAILABLE", OUT)
    with pytest.raises(ResultsError) as refused:
        results.export("irradiance", "csv", monitor="P")
    assert refused.value.code == OUT
    rows, _ = results.table("arrivals", monitor="P")
    assert rows[0]["power_out_of_scope_at"] == ["ND/attenuator"]
    monitors = {row["observation_id"]: row for row in results.table("monitors")[0]}
    assert monitors["P"]["quantity_state"] == "OUT_OF_MODEL_SCOPE"
    assert results.table("handoff", monitor="P")[0][0]["power_out_of_scope_at"] == ["ND/attenuator"]

    results = _attenuated(2.0, 2.0001)
    (arrival,) = results.result.observations[0].arrivals
    assert not arrival.power_out_of_scope_at and arrival.beam.power_W == pytest.approx(_passed_share(2.0, 2.0001), rel=1e-4)
    doc = results.irradiance_map("P")
    assert doc["status"] == "OK" and doc["untraced_fraction"] is None  # returning lens ghosts traverse N-BK7 with unknown bulk loss
    assert doc["grid_power_incoherent_W"] == pytest.approx(arrival.beam.power_W * math.erf(2.5 * math.sqrt(2.0)) ** 2, rel=1e-3)
    assert results.detector_response("P", detector)["status"] == "OK"
    (beam,) = results.handoff("P")["beams"]
    assert beam["power_W"] == arrival.beam.power_W and beam["validity"]["beam_parameters"] == "TEM00"
    assert "power_out_of_scope_at" not in beam["validity"] and results.table("handoff", monitor="P")[0][0]["power_out_of_scope_at"] == []
    assert results.pupil_field("P", request)["status"] in ("OK", "PARTIAL")
    (control,) = [o for o in results.result.observations if o.observation_id == "P"]
    assert control.quantity_state == "BOUNDED" and not control.power_complete
    assert control.total_power_W == arrival.beam.power_W and control.power_upper_bound_W is None
    # The primary forward crossing occurs before the lens. Unknown returning light can only add to it.
    basis = control.value_basis["total_power_W"]
    assert basis.kind.value == "LOWER_BOUND"
    (bulk,) = [part for part in basis.omitted if part.code == "BULK_LOSS_NOT_ESTABLISHED"]
    assert bulk.at == ("path:SRC#0",) and bulk.upper is None
    row = next(row for row in results.table("monitors")[0] if row["observation_id"] == "P")
    assert row["quantity_state"] == "BOUNDED" and row["value_basis"]["total_power_W"] == basis.model_dump(mode="json", exclude_none=True)


def test_a_ghost_analysis_states_no_power_of_light_whose_power_is_out_of_model_scope():
    """The same attenuator before the lens: the ghosts of the lens on IMG land where they land (focus offsets stated), but their
    powers, ratios and the bound of the ghosts not listed rest on the chief ray's power: OUT_OF_MODEL_SCOPE, each in its own
    row and the bound not proven; the enumeration ran to its end (COMPLETED). With the uniform attenuator the primary's power and the ratios of the ghosts on IMG are stated; only the ghosts
    reflected back onto the attenuator, whose wide footprints reach its uncoated border there, carry a power out of model scope
    (their own powers and the bound they enter are withheld, not the others').

    Register row T-13, withheld: the chief ray's transmittance is not the mean transmittance over the beam's profile, which the
    attenuator step does not compute."""
    for ods, scoped in (((0.0, 8.0), True), ((2.0, 2.0001), False)):
        doc = _attenuated(*ods).ghosts("IMG")
        by_id = {q.quantity_id: q for q in doc.quantities}
        relative = [q for qid, q in by_id.items() if qid.endswith(".relative_power")]
        focus = [q for qid, q in by_id.items() if qid.endswith(".focus_offset_mm")]
        bound = [q for qid, q in by_id.items() if qid.endswith("unlisted_power_upper_relative_to_line")]
        assert relative and focus and bound
        if scoped:
            assert all(q.state.value == "OUT_OF_MODEL_SCOPE" and q.reason == OUT and q.value is None for q in relative + bound)
            assert any(q.state.value == "KNOWN" for q in focus) and doc.execution.complete and not doc.completeness.proven
        else:
            assert by_id["SRC#0.primary_power_relative_to_line"].state.value == "KNOWN"
            assert all(q.state.value == "KNOWN" and q.value > 0.0 for q in relative)
            assert all(q.reason != OUT for qid, q in by_id.items() if qid.endswith((".relative_power", ".irradiance_ratio", ".power_relative_to_line")))
            # (the withheld ones: the two back-reflected ghosts that end beyond the attenuator, and the bound they enter)
            withheld = {qid[len("SRC#0."):qid.rindex(".")] for qid, q in by_id.items() if q.reason == OUT and "unlisted" not in qid}
            assert withheld <= {"SRC#0.0", "SRC#0.1"}, withheld


def test_a_pulsed_harmonic_has_no_irradiance_map():
    """1 W average at 1064 nm in 10 ns pulses at 10 Hz through a 2 mm PPLN: the harmonic is the pulse's, out of model scope
    (``test_second_harmonic_scope``); its map on OUT used to be the continuous-wave one (3.27e-6 W). The continuous-wave source's
    map is stated, its grid power the harmonic beam's (Gaussian integral over the grid).

    Register row T-14, withheld: a pulse's conversion depends on its peak power and temporal shape, which the continuous-wave
    model does not carry."""
    from test_second_harmonic_scope import _pulsed_doubler_with_a_rated_window

    def go(pulsed: bool):
        snapshot = _pulsed_doubler_with_a_rated_window(pulsed)
        # Isolate the pulse-scope comparison: these synthetic PPLN and window materials explicitly assume zero bulk loss.
        # Their helper's unstated coefficients otherwise withhold even the CW harmonic map independently of pulse scope.
        materials = tuple(type(m).model_validate({**m.model_dump(mode="json", exclude_none=True),
                                                   "absorption_coefficient_per_mm": 0.0}) for m in snapshot.materials)
        return validate(ValidationRequest(project=snapshot.model_copy(update={"materials": materials}),
                                          configuration=EngineConfiguration(run_layout=False, run_control=False)))

    result = go(True)
    observation = result.observations[0]
    assert observation.quantity_state == "OUT_OF_MODEL_SCOPE"
    doc = irradiance_map(result, "OUT", wavelength_nm=532.0)
    assert (doc["status"], doc["reason"], doc["power_out_of_scope_at"]) == ("UNAVAILABLE", OUT, ["X/front"])
    result = go(False)
    (harmonic,) = [a for a in result.observations[0].arrivals if abs(a.beam.wavelength_nm - 532.0) < 1e-6]
    doc = irradiance_map(result, "OUT", wavelength_nm=532.0)
    assert doc["status"] == "OK" and doc["map_complete"]
    assert doc["grid_power_incoherent_W"] == pytest.approx(harmonic.beam.power_W * math.erf(2.5 * math.sqrt(2.0)) ** 2, rel=1e-3)


def test_a_beam_leaving_a_gradient_index_rod_through_its_side_has_no_irradiance_map():
    """The 0.5 rad tilted beam of ``test_gradient_index``: 44 % of it reaches the rod's side (the reference), while the beam behind
    the rod carries all of it; the map of OUT read 9.28e-4 W, 82 % above the light that arrives. Now no map. At 0.45 rad the rod
    holds the beam whole (the reference finds no ray at the side): the map is stated.

    Register row T-15, withheld: what the rod's side does with the light that reaches it (refraction out, or reflection back in)
    is not followed."""
    from test_gradient_index import _side_share, _tilted

    from os_optics.local import run_optical_project

    def lossless_rod(tilt: float) -> dict:
        document = _tilted(tilt, 0.02)
        # This test's synthetic GRIN rod isolates side exit; explicitly assume no bulk absorption, not a supplier fact.
        document["model_materials"][0]["absorption_coefficient_per_mm"] = "0"
        return document

    assert 0.40 < _side_share(0.5, 0.02) < 0.50
    run_ = run_optical_project(lossless_rod(0.5), catalog=None)
    doc = irradiance_map(run_.result, "OUT")
    assert (doc["status"], doc["reason"]) == ("UNAVAILABLE", OUT) and "G1/S2" in doc["power_out_of_scope_at"]
    assert _side_share(0.45, 0.02) == 0.0
    run_ = run_optical_project(lossless_rod(0.45), catalog=None)
    assert irradiance_map(run_.result, "OUT")["status"] == "OK"


INV = 1.0 / math.sqrt(2.0)


def _scattering(graded: bool):
    """A 633 nm beam at 45 deg onto a Lambertian scattering absorber (scattered share 0.1) at the origin, through a variable
    attenuator (OD 0 -> 4 across it, or a fixed OD 2); an absorbing disc D (radius 5 mm) on the absorber's axis 10 mm above it,
    with a power_W target on D. The beam crosses D's plane outside its disc."""
    law = ({"kind": "linear_gradient", "od_start": 0.0, "od_end": 4.0, "x_start_mm": -2.0, "x_end_mm": 2.0} if graded
           else {"kind": "fixed", "optical_density": 2.0})
    src = B.laser("S", wavelength_nm=633.0, power_W=1.0, waist_radius_mm=1.5, position=(-50.0, 0.0, 50.0), direction=(INV, 0.0, -INV))
    nd = B.beam_dump("ND", position=(-30.0, 0.0, 30.0), axis=(-INV, 0.0, INV), radius_mm=12.0)
    nd = nd.model_copy(update={"surfaces": (nd.surfaces[0].model_copy(update={"interaction": NeutralDensityInteraction(law=law)}),)})
    dump = B.beam_dump("DUMP", position=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius_mm=12.0)
    scatter = SurfaceScatter(bsdf="LAMBERTIAN", fraction=0.1, side="reflect", sample_count=4096)
    dump = dump.model_copy(update={"surfaces": (dump.surfaces[0].model_copy(update={"interaction": AbsorberInteraction(scatter=scatter)}),)})
    disc = B.observation_plane("D", position=(0.0, 0.0, 10.0), normal=(0.0, 0.0, -1.0), radius_mm=5.0, absorbing=True)
    from os_optics.contracts import Target

    snapshot = ProjectSnapshot(name="scatter", components=(src, nd, dump), observations=(disc,),
                               targets=(Target(target_id="p", metric="power_W", observation_id="D", maximum=1.0),))
    return validate(ValidationRequest(project=snapshot, configuration=EngineConfiguration(run_layout=False, run_control=False)))


def test_light_scattered_out_of_a_beam_whose_power_is_out_of_model_scope_is_out_of_model_scope():
    """The scattered light carries the scattered share of the beam's chief-ray power: behind the varying attenuator that is not
    the light's, so its line on D names the attenuator, D's power is OUT_OF_MODEL_SCOPE (it read BOUNDED with a value), the target
    on it too, and no map of it is drawn. Behind the fixed OD 2 the plane's power is stated (KNOWN, or BOUNDED by light left out as
    negligible), the line's collection the reference: 0.1 x 0.01 x the Lambertian
    share on the disc, a^2 / (a^2 + h^2) = 0.2 (within the sampled fraction's tolerance, 0.004 of the scattered power).

    Register row T-16, withheld: the chief ray's transmittance is not the mean transmittance over the beam's profile, which the
    attenuator step does not compute."""
    result = _scattering(True)
    (disc,) = result.observations
    (line,) = disc.sampled_light
    assert line.power_out_of_scope_at == ("ND/dump",) and disc.quantity_state == "OUT_OF_MODEL_SCOPE"
    (target,) = result.targets
    assert target.status.value == "OUT_OF_MODEL_SCOPE" and target.value is None and "scattered" in target.message
    doc = irradiance_map(result, "D")
    assert (doc["status"], doc["reason"], doc["power_out_of_scope_at"]) == ("UNAVAILABLE", OUT, ["ND/dump"])
    result = _scattering(False)
    (disc,) = result.observations
    (line,) = disc.sampled_light
    assert not line.power_out_of_scope_at and disc.quantity_state in ("KNOWN", "BOUNDED") and "power_out_of_scope_at" not in line.model_dump()
    assert abs(line.collection_fraction - 25.0 / 125.0) < 0.004 and line.emitted_W == pytest.approx(0.1 * 0.01, rel=1e-6)
    assert result.targets[0].status.value == "PASS" and irradiance_map(result, "D")["status"] == "OK"


def _crossing(graded: bool):
    """Two mutually coherent 633 nm beams (one coherence group) crossing at 4 deg on a plane, B (0.5 mm) through a variable
    attenuator of 2 mm radius (OD 0 -> 4 across it) or a fixed OD 2, 4.3 radii of A beside A's axis."""
    law = ({"kind": "linear_gradient", "od_start": 0.0, "od_end": 4.0, "x_start_mm": -2.0, "x_end_mm": 2.0} if graded
           else {"kind": "fixed", "optical_density": 2.0})
    tilt = math.radians(4.0)
    a = B.laser("A", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=1.0, position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                coherence_group="G", coherence_length_mm=1000.0)
    b = B.laser("B", wavelength_nm=633.0, power_W=1e-3, waist_radius_mm=0.5, position=(-100.0, 100.0 * math.tan(tilt), 0.0),
                direction=(math.cos(tilt), -math.sin(tilt), 0.0), coherence_group="G", coherence_length_mm=1000.0)
    nd = B.beam_dump("ND", position=(-90.0, 90.0 * math.tan(tilt), 0.0), axis=(-math.cos(tilt), math.sin(tilt), 0.0), radius_mm=2.0)
    nd = nd.model_copy(update={"surfaces": (nd.surfaces[0].model_copy(update={"interaction": NeutralDensityInteraction(law=law)}),)})
    screen = B.observation_plane("S", position=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=20.0)
    snapshot = ProjectSnapshot(name="pair", components=(a, b, nd), observations=(screen,))
    return validate(ValidationRequest(project=snapshot, configuration=EngineConfiguration(run_layout=False, run_control=False)))


def test_the_coherent_sum_and_the_pair_quantities_of_a_beam_out_of_model_scope_are_withheld():
    """The coherent sum of the two beams and their intensity balance, fringe visibilities, irradiances and powers read the beams'
    powers: with B's power out of model scope none is stated (OUT_OF_MODEL_SCOPE), while the crossing angle and the fringe period
    (geometry: 2 pi / |dk_parallel|, lambda / sin(alpha) with A along the plane's normal) stay. With the fixed attenuator they are stated,
    the incoherent sum the two powers.

    Register row T-17, withheld: (the power quantities) the chief ray's transmittance is not the mean transmittance over the
    beam's profile, which the attenuator step does not compute."""
    tilt = math.radians(4.0)
    for graded in (True, False):
        result = _crossing(graded)
        (screen,) = result.observations
        scoped = [a for a in screen.arrivals if a.power_out_of_scope_at]
        (pair,) = result.interference
        (coherent,) = screen.superpositions
        assert pair.beam_crossing_angle_deg == pytest.approx(4.0, rel=1e-6)
        assert pair.fringe_period_nm == pytest.approx(633.0 / math.sin(tilt), rel=1e-6)  # (A along the plane's normal)
        if graded:
            assert len(scoped) == 1 and screen.quantity_state == "OUT_OF_MODEL_SCOPE"
            assert pair.status.value == "OUT_OF_MODEL_SCOPE" and "OUT_OF_MODEL_SCOPE" in pair.issue_codes
            assert pair.fringe_visibility is None and pair.intensity_balance is None and pair.incoherent_power_sum_W is None
            assert pair.peak_irradiance_W_per_mm2 is None and pair.combined_power_collinear_W is None
            assert coherent.status.value == "OUT_OF_MODEL_SCOPE" and coherent.total_power_W is None
        else:
            assert not scoped and pair.status.value != "OUT_OF_MODEL_SCOPE"
            powers = sorted(a.beam.power_W for a in screen.arrivals)
            assert pair.incoherent_power_sum_W == pytest.approx(sum(powers), rel=1e-12) and pair.intensity_balance is not None
            assert coherent.status.value != "OUT_OF_MODEL_SCOPE"  # (UNKNOWN here: the attenuator states no group delay)


def test_a_control_sensor_behind_a_varying_attenuator_has_no_reading():
    """A control sensor reading P's power behind the varying attenuator would read the chief ray's power (the reference says the
    light keeps about twice it): its reading is not known. Behind the uniform one it reads the beam's power.

    Register row T-18, withheld: the chief ray's transmittance is not the mean transmittance over the beam's profile, which the
    attenuator step does not compute."""
    from local_helpers import catalog_document

    from os_optics.config import NumericalConfig
    from os_optics.contracts import ControlSpec, SensorSpec
    from os_optics.control.readings import evaluate
    from os_optics.local.lowering import lower_project

    spec = ControlSpec(actuator_refs=("SRC/x",), sensors=(SensorSpec(sensor_id="s", observation_id="P", target_value=0.0, quantity="power_W"),))
    for ods, scoped in (((0.0, 8.0), True), ((2.0, 2.0001), False)):
        snapshot = lower_project(_attenuated_document(*ods), catalog_document(), check_solids=False).request.project
        reading = evaluate(snapshot, spec, {}, NumericalConfig())
        if scoped:
            assert reading.unknown and math.isnan(reading.values[0])
        else:
            assert not reading.unknown and reading.values[0] == pytest.approx(_passed_share(*ods), rel=1e-4)
