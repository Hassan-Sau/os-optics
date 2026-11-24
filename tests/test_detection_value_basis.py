"""A detector on a plane whose power is not simply the model's value reads the traced light and says what its number is.

The scene: a 1 mW laser through the fixture catalog's explicitly lossless N-BK7 window onto an absorbing detector plane. The trace
follows the beam through the window; the window's ghost reflections are light it does not follow as beams, which the
probes bound on the plane, so the plane's power is ``BOUNDED``: the traced value, and at most ``power_upper_bound_W``.

Every reference is computed here, independently of the Engine code under test:

* the traced power from the Fresnel transmission of the two faces, ``(1 - ((n - 1) / (n + 1))^2)^2``, with the index
  from the catalog's Sellmeier coefficients (``local_helpers.sellmeier_index``);
* the detection quantities from the closed forms ``R = QE q lambda / (h c)``, ``I = sum R P``, shot noise ``2 q I``, the
  dark floor ``2 q I_d + 4 k T / R_L``, the first-order noise bandwidth ``pi f_c / 2``, the square wave's fundamental
  ``(2 / pi) sin(pi D)`` and reference weight ``D + sin(2 pi D) / (2 pi)``, the lock-in's ``ENBW`` of two RC sections
  ``1 / (8 tau)``;
* the intervals by brute force: the quantity recomputed for every sampled allocation of the light that may add (its
  amount up to the plane's bound, its wavelength among the launched ones, its quantum efficiency from 0 to 1, its
  modulated share from 0 to 1) lies inside the interval the Engine grades the target on.
"""

from __future__ import annotations

import itertools
import json
import math

import pytest

from local_helpers import WINDOW, catalog_document, element, plane, project, run as run_fixture, sellmeier_index, source, with_lossless_bulk

from os_optics.contracts.detection import Detection
from os_optics.contracts.project import Target
from os_optics.errors.status import OmittedPart, QuantityState, ValueKind
from os_optics.results import load_run
from os_optics.validation.detection import PLANE_POWER_NOT_KNOWN, detect

Q = 1.602176634e-19
H = 6.62607015e-34
C = 299792458.0
K_B = 1.380649e-23
CHAR = {"kind": "DATASHEET", "reference": "photodiode datasheet rev B"}
QE_FLAT = {"rows": [{"wavelength_nm": 400.0, "qe": 0.8}, {"wavelength_nm": 900.0, "qe": 0.8}], "source": CHAR}
QE_SLOPE = {"rows": [{"wavelength_nm": 400.0, "qe": 0.5}, {"wavelength_nm": 900.0, "qe": 0.9}], "source": CHAR}
FC_HZ, DARK_A, LOAD_OHM, TEMP_K = 10000.0, 1e-12, 1e8, 295.0
BANDWIDTH_HZ = math.pi / 2.0 * FC_HZ
FLOOR2 = 2.0 * Q * DARK_A + 4.0 * K_B * TEMP_K / LOAD_OHM
DUTY, TAU_S = 0.5, 0.1
ENBW_HZ = 1.0 / (8.0 * TAU_S)
FUNDAMENTAL = 2.0 / math.pi * math.sin(math.pi * DUTY)
WEIGHT = DUTY + math.sin(2.0 * math.pi * DUTY) / (2.0 * math.pi)


def responsivity(qe: float, wavelength_nm: float) -> float:
    return qe * Q * wavelength_nm * 1e-9 / (H * C)


def slope_qe(wavelength_nm: float) -> float:
    return 0.5 + 0.4 * (wavelength_nm - 400.0) / 500.0


def electrical(**changes) -> dict:
    value = {"bandwidth_Hz": FC_HZ, "bandwidth_definition": "FIRST_ORDER_3DB", "dark_current_A": DARK_A,
             "load_resistance_ohm": LOAD_OHM, "temperature_K": TEMP_K, "source": CHAR}
    value.update(changes)
    return value


def lock_in(sources=("L",)) -> dict:
    return {"modulation": {"waveform": "SQUARE", "frequency_Hz": 1000.0, "duty_cycle": DUTY, "source_ids": list(sources)},
            "time_constant_s": TAU_S, "filter_order": 2}


def window_project(detection: dict, targets: list | None = None, *, window: bool = True, extra_sources: tuple = ()) -> dict:
    detector = {**plane("D", (100.0, 0.0, 0.0), radius_mm=20.0), "absorbing": True, "detection": detection}
    components = [source("L", (-100.0, 0.0, 0.0), wavelength_nm=780.0, waist_mm=0.5, power_w=1e-3), *extra_sources]
    if window:
        components.append(element("W", "window", WINDOW, (0.0, 0.0, 0.0)))
    return project(components, [detector], targets)


def run(document):
    """Prescribe lossless bulk transmission for the detector's quantitative references."""
    return run_fixture(document, catalog=with_lossless_bulk(catalog_document()))


def observation(run_, oid: str = "D"):
    return next(o for o in run_.result.observations if o.observation_id == oid)


def quantities_of(run_, oid: str = "D") -> dict:
    return {q.name: q for q in observation(run_, oid).detection.quantities}


def entry(quantity):
    return None if quantity.value_basis is None else quantity.value_basis["value"]


def detector_snr(current: float) -> float:
    return current / math.sqrt((2.0 * Q * current + FLOOR2) * BANDWIDTH_HZ)


def lock_in_snr(steady: float, modulated: float) -> float:
    signal = FUNDAMENTAL * modulated / math.sqrt(2.0)
    return signal / math.sqrt((2.0 * Q * (steady + WEIGHT * modulated) + FLOOR2) * ENBW_HZ)


def fresnel_window_transmission(wavelength_nm: float) -> float:
    n = sellmeier_index("N-BK7", wavelength_nm)
    reflectance = ((n - 1.0) / (n + 1.0)) ** 2
    return (1.0 - reflectance) ** 2


def test_a_detector_behind_a_window_reads_the_traced_light_and_says_what_the_number_is():
    detection = {"detector": {"quantum_efficiency": QE_FLAT, "electrical": electrical()}, "lock_in": lock_in()}
    run_ = run(window_project(detection))
    plane_ = observation(run_)
    assert plane_.quantity_state is QuantityState.BOUNDED and plane_.detection.power_state is QuantityState.BOUNDED
    power = plane_.total_power_W
    # the traced light is the beam through the two uncoated faces
    assert power == pytest.approx(1e-3 * fresnel_window_transmission(780.0), rel=1e-6)
    assert plane_.power_upper_bound_W > power
    quantities = quantities_of(run_)
    current = responsivity(0.8, 780.0) * power
    expected = {
        "detected_power_W": power,
        "photocurrent_A": current,
        "shot_noise_current_A": math.sqrt(2.0 * Q * current * BANDWIDTH_HZ),
        "total_noise_current_A": math.sqrt((2.0 * Q * current + FLOOR2) * BANDWIDTH_HZ),
        "detector_snr": detector_snr(current),
        "noise_equivalent_power_W_per_rtHz": math.sqrt(FLOOR2) / responsivity(0.8, 780.0),
        "modulated_photocurrent_A": current,
        "lock_in_snr": lock_in_snr(0.0, current),
    }
    for name, value in expected.items():
        assert quantities[name].status == "MODELED", name
        assert quantities[name].value == pytest.approx(value, rel=1e-12), name
    # what each number is: the power's entry states the most the light not followed adds; the rising quantities keep its kind
    light = entry(quantities["detected_power_W"])
    assert light.kind is ValueKind.LOWER_BOUND
    (part,) = light.omitted
    assert part.part is OmittedPart.NOT_FOLLOWED and part.upper == pytest.approx(plane_.power_upper_bound_W - power, rel=1e-9)
    for name in ("photocurrent_A", "shot_noise_current_A", "total_noise_current_A", "detector_snr", "modulated_photocurrent_A",
                 "lock_in_signal_A", "lock_in_noise_A"):
        rising = entry(quantities[name])
        assert rising.kind is ValueKind.LOWER_BOUND and [p.part for p in rising.omitted] == [OmittedPart.NOT_FOLLOWED], name
        assert rising.omitted[0].upper is None, name  # (the amount is a power: it stays with detected_power_W)
    assert entry(quantities["lock_in_snr"]).kind is ValueKind.ESTIMATE
    assert entry(quantities["noise_equivalent_power_W_per_rtHz"]) is None  # (one wavelength: a ratio the amount of light leaves alone)
    # what the detector states by itself carries no entry
    for name in ("detector_bandwidth_Hz", "noise_equivalent_bandwidth_Hz", "dark_noise_density_A_per_rtHz", "lock_in_enbw_Hz"):
        assert quantities[name].value_basis is None, name
    # the detection projection of the stored run carries each number with its entry
    (row,) = load_run(run_).detection("D")
    carried = {q["name"]: q for q in row["quantities"]}
    assert carried["detector_snr"]["value"] == quantities["detector_snr"].value
    assert carried["detector_snr"]["value_basis"] == {"value": entry(quantities["detector_snr"]).model_dump(mode="json")}
    assert "value_basis" not in carried["lock_in_enbw_Hz"]


def test_unstated_glass_loss_withholds_detection_but_keeps_the_absorbing_plane_bound():
    detection = {"detector": {"quantum_efficiency": QE_FLAT, "electrical": electrical()}, "lock_in": lock_in()}
    target = {"target_id": "signal", "metric": "detector_snr", "observation_id": "D", "minimum": 10.0}
    run_ = run_fixture(window_project(detection, [target]))
    detector_plane = observation(run_)
    assert detector_plane.quantity_state is detector_plane.detection.power_state is QuantityState.BOUNDED
    assert detector_plane.total_power_W is None and 0.0 < detector_plane.power_upper_bound_W <= 1e-3
    assert "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in run_.result.issues}
    quantities = quantities_of(run_)
    for name in ("detected_power_W", "photocurrent_A", "detector_snr", "lock_in_snr"):
        quantity = quantities[name]
        assert (quantity.status, quantity.value, quantity.reason) == ("UNAVAILABLE", None, PLANE_POWER_NOT_KNOWN), name
    assert quantities["detector_bandwidth_Hz"].value == FC_HZ
    assert quantities["lock_in_enbw_Hz"].value == ENBW_HZ
    (assessment,) = run_.result.targets
    assert (assessment.assessment.value, assessment.value, assessment.issue_codes) == (
        "INDETERMINATE", None, ("DETECTION_QUANTITY_UNAVAILABLE",))


def test_the_graded_intervals_hold_every_allocation_of_the_light_that_may_add():
    detection = {"detector": {"quantum_efficiency": QE_FLAT, "electrical": electrical()}, "lock_in": lock_in()}
    run_ = run(window_project(detection))
    plane_ = observation(run_)
    spec = next(o for o in run_.request.project.observations if o.observation_id == "D").detection
    detected = detect(plane_, spec, plane_.quantity_state)
    current = responsivity(0.8, 780.0) * plane_.total_power_W
    most = plane_.power_upper_bound_W - plane_.total_power_W
    dc_low, dc_high = detected.bounds["detector_snr"]
    li_low, li_high = detected.bounds["lock_in_snr"]
    assert dc_low == pytest.approx(detector_snr(current), rel=1e-12) and dc_high > dc_low
    assert li_low < lock_in_snr(0.0, current) < li_high
    for step, qe, share in itertools.product(range(9), (0.0, 0.4, 0.8, 1.0), (0.0, 0.25, 0.5, 1.0)):
        added = responsivity(qe, 780.0) * most * step / 8.0
        dc = detector_snr(current + added)
        li = lock_in_snr(added * (1.0 - share), current + added * share)
        assert dc_low * (1.0 - 1e-12) <= dc <= dc_high * (1.0 + 1e-12), (step, qe, share)
        assert li_low * (1.0 - 1e-12) <= li <= li_high * (1.0 + 1e-12), (step, qe, share)


def test_targets_on_a_bounded_plane_are_graded_on_the_interval_and_carry_the_entry():
    detection = {"detector": {"quantum_efficiency": QE_FLAT, "electrical": electrical()}, "lock_in": lock_in()}
    probe = run(window_project(detection))
    plane_ = observation(probe)
    spec = next(o for o in probe.request.project.observations if o.observation_id == "D").detection
    li_low, li_high = detect(plane_, spec, plane_.quantity_state).bounds["lock_in_snr"]
    li_value = quantities_of(probe)["lock_in_snr"].value
    targets = [
        {"target_id": "dc_min", "metric": "detector_snr", "observation_id": "D", "minimum": 10.0},
        {"target_id": "dc_max", "metric": "detector_snr", "observation_id": "D", "maximum": 10.0},
        {"target_id": "li_min", "metric": "lock_in_snr", "observation_id": "D", "minimum": 0.5 * li_low},
        {"target_id": "li_inside", "metric": "lock_in_snr", "observation_id": "D", "minimum": 0.5 * (li_low + li_value)},
        {"target_id": "li_above", "metric": "lock_in_snr", "observation_id": "D", "minimum": 2.0 * li_high},
    ]
    results = {t.target_id: t for t in run(window_project(detection, targets)).result.targets}
    assert (results["dc_min"].status.value, results["dc_min"].assessment.value) == ("PASS", "MET")
    assert (results["dc_max"].status.value, results["dc_max"].assessment.value) == ("FAIL", "UNMET")
    assert (results["li_min"].status.value, results["li_min"].assessment.value) == ("PASS", "MET")
    assert (results["li_inside"].status.value, results["li_inside"].assessment.value) == ("UNKNOWN", "INDETERMINATE")
    assert (results["li_above"].status.value, results["li_above"].assessment.value) == ("FAIL", "UNMET")
    for target in results.values():
        assert target.value is not None and "UNTRACED_LIGHT_BOUNDED" in target.issue_codes
        assert "DETECTION_QUANTITY_UNAVAILABLE" not in target.issue_codes
    assert results["dc_min"].value_basis["value"].kind is ValueKind.LOWER_BOUND
    assert results["li_inside"].value_basis["value"].kind is ValueKind.ESTIMATE
    assert results["li_inside"].value == li_value


def test_two_wavelengths_of_different_quantum_efficiency_and_one_modulated_source():
    """A modulated 500 nm source and a steady 850 nm source through the window onto one detector whose quantum efficiency
    rises with the wavelength: the values are the closed forms on each line's traced power, and the intervals hold the
    light that may add at either wavelength, steady or modulated."""

    detection = {"detector": {"quantum_efficiency": QE_SLOPE, "electrical": electrical()}, "lock_in": lock_in(("A",))}
    second = source("B", (-100.0, 0.0, 0.0), wavelength_nm=850.0, waist_mm=0.5, power_w=2e-4)
    document = window_project(detection, extra_sources=(second,))
    document["components"][0] = source("A", (-100.0, 0.0, 0.0), wavelength_nm=500.0, waist_mm=0.5, power_w=1e-3)
    run_ = run(document)
    plane_ = observation(run_)
    assert plane_.quantity_state is QuantityState.BOUNDED
    by_wavelength = {s.wavelength_nm: s.power_W for s in plane_.power_by_wavelength}
    assert by_wavelength[500.0] == pytest.approx(1e-3 * fresnel_window_transmission(500.0), rel=1e-6)
    assert by_wavelength[850.0] == pytest.approx(2e-4 * fresnel_window_transmission(850.0), rel=1e-6)
    modulated = responsivity(slope_qe(500.0), 500.0) * by_wavelength[500.0]
    steady = responsivity(slope_qe(850.0), 850.0) * by_wavelength[850.0]
    quantities = quantities_of(run_)
    assert quantities["photocurrent_A"].value == pytest.approx(modulated + steady, rel=1e-12)
    assert quantities["modulated_photocurrent_A"].value == pytest.approx(modulated, rel=1e-12)
    assert quantities["detector_snr"].value == pytest.approx(detector_snr(modulated + steady), rel=1e-12)
    assert quantities["lock_in_snr"].value == pytest.approx(lock_in_snr(steady, modulated), rel=1e-12)
    assert quantities["noise_equivalent_power_W_per_rtHz"].value == pytest.approx(
        math.sqrt(FLOOR2) * (by_wavelength[500.0] + by_wavelength[850.0]) / (modulated + steady), rel=1e-12)
    # two wavelengths: the ratio to the light's responsivity depends on which one the added light has
    assert entry(quantities["noise_equivalent_power_W_per_rtHz"]).kind is ValueKind.ESTIMATE
    assert entry(quantities["detector_snr"]).kind is ValueKind.LOWER_BOUND
    spec = next(o for o in run_.request.project.observations if o.observation_id == "D").detection
    detected = detect(plane_, spec, plane_.quantity_state)
    most = plane_.power_upper_bound_W - plane_.total_power_W
    dc_low, dc_high = detected.bounds["detector_snr"]
    li_low, li_high = detected.bounds["lock_in_snr"]
    for step, wavelength, share in itertools.product(range(5), (500.0, 850.0), (0.0, 0.5, 1.0)):
        added = responsivity(slope_qe(wavelength), wavelength) * most * step / 4.0
        dc = detector_snr(modulated + steady + added)
        li = lock_in_snr(steady + added * (1.0 - share), modulated + added * share)
        assert dc_low * (1.0 - 1e-12) <= dc <= dc_high * (1.0 + 1e-12), (step, wavelength, share)
        assert li_low * (1.0 - 1e-12) <= li <= li_high * (1.0 + 1e-12), (step, wavelength, share)


def test_light_that_may_reach_the_saturation_current_leaves_the_photocurrent_an_estimate():
    """A stated linear range between the traced peak and the peak with the light that may add (at a quantum efficiency of
    1): the linear range is not proven for the light on the plane, so the quantities that read the photocurrent are
    estimates with the open check, and a target on one is not decided."""

    plain = {"detector": {"quantum_efficiency": QE_FLAT, "electrical": electrical()}}
    plane_ = observation(run(window_project(plain)))
    current = responsivity(0.8, 780.0) * plane_.total_power_W
    peak_upper = current + responsivity(1.0, 780.0) * (plane_.power_upper_bound_W - plane_.total_power_W)
    limit = 0.5 * (current + peak_upper)
    targets = [{"target_id": "dc", "metric": "detector_snr", "observation_id": "D", "minimum": 10.0}]
    tight = run(window_project({"detector": {"quantum_efficiency": QE_FLAT, "electrical": electrical(saturation_current_A=limit)}}, targets))
    quantities = quantities_of(tight)
    assert quantities["detector_snr"].value == pytest.approx(detector_snr(current), rel=1e-12)
    for name in ("photocurrent_A", "shot_noise_current_A", "total_noise_current_A", "detector_snr", "noise_equivalent_power_W_per_rtHz"):
        basis = entry(quantities[name])
        assert basis.kind is ValueKind.ESTIMATE, name
        (check,) = basis.validity
        assert (check.parameter, check.unit, check.maximum, check.within) == ("peak_photocurrent_A", "A", limit, None), name
        assert check.observed == pytest.approx(peak_upper, rel=1e-9), name
    assert entry(quantities["detected_power_W"]).kind is ValueKind.LOWER_BOUND  # (the optical power does not read the photocurrent)
    (target,) = tight.result.targets
    assert (target.status.value, target.assessment.value) == ("UNKNOWN", "INDETERMINATE")
    assert target.value == quantities["detector_snr"].value and target.value_basis["value"].kind is ValueKind.ESTIMATE
    # a linear range above the peak with every watt that may add: proven, the kinds as without the statement
    wide = run(window_project({"detector": {"quantum_efficiency": QE_FLAT, "electrical": electrical(saturation_current_A=2.0 * peak_upper)}}, targets))
    assert entry(quantities_of(wide)["detector_snr"]).kind is ValueKind.LOWER_BOUND
    assert wide.result.targets[0].assessment.value == "MET"


def test_a_known_plane_keeps_its_detection_without_an_entry():
    detection = {"detector": {"quantum_efficiency": QE_FLAT, "electrical": electrical()}, "lock_in": lock_in()}
    targets = [{"target_id": "dc", "metric": "detector_snr", "observation_id": "D", "minimum": 10.0}]
    run_ = run(window_project(detection, targets, window=False))
    plane_ = observation(run_)
    assert plane_.quantity_state is QuantityState.KNOWN
    assert all(q.value_basis is None for q in plane_.detection.quantities)
    assert "value_basis" not in json.dumps(plane_.detection.model_dump(mode="json"))
    (target,) = run_.result.targets
    assert (target.status.value, target.issue_codes, target.value_basis, target.message) == ("PASS", (), None, "")
    assert target.value == pytest.approx(detector_snr(responsivity(0.8, 780.0) * 1e-3), rel=1e-9)


def test_a_plane_no_traced_beam_reaches_while_light_may_arrive_states_no_detection_value():
    """A detector behind the source that only the window's ghost reflection may reach: no traced light is on the plane, so
    no number is computed (the plane's power is ``BOUNDED`` from 0 with no value) and the target stays undecided."""

    detection = {"detector": {"quantum_efficiency": QE_FLAT, "electrical": electrical()}}
    ghost = {**plane("G", (-150.0, 0.0, 0.0), radius_mm=20.0), "absorbing": True, "detection": detection}
    document = window_project(detection, [{"target_id": "dc", "metric": "detector_snr", "observation_id": "G", "minimum": 10.0}])
    document["observations"].append(ghost)
    run_ = run(document)
    plane_ = observation(run_, "G")
    assert plane_.quantity_state is QuantityState.BOUNDED and plane_.total_power_W is None and not plane_.lit
    quantities = quantities_of(run_, "G")
    for name in ("detected_power_W", "photocurrent_A", "detector_snr"):
        assert (quantities[name].status, quantities[name].value, quantities[name].reason) == ("UNAVAILABLE", None, PLANE_POWER_NOT_KNOWN)
    (target,) = run_.result.targets
    assert (target.status.value, target.value, target.issue_codes) == ("UNKNOWN", None, ("DETECTION_QUANTITY_UNAVAILABLE",))


def test_an_annular_beam_counted_whole_makes_the_detection_an_upper_bound():
    """A beam whose detected power is at least ``detected_power_min_W`` (counted whole on an active area that may miss part
    of it): the detector's numbers are the reading's, upper bounds by the share that may be elsewhere."""

    detection = {"detector": {"quantum_efficiency": QE_FLAT, "electrical": electrical()}}
    run_ = run(window_project(detection, window=False))
    plane_ = observation(run_)
    (arrival,) = plane_.arrivals
    reading = arrival.detected_power_W
    annular = plane_.model_copy(update={"arrivals": (arrival.model_copy(update={"detected_power_min_W": 0.9 * reading}),)})
    spec = Detection.model_validate(next(o for o in run_.request.project.observations if o.observation_id == "D").detection.model_dump())
    detected = detect(annular, spec, QuantityState.KNOWN)
    quantities = {q.name: q for q in detected.result.quantities}
    assert quantities["detected_power_W"].value == reading
    light = entry(quantities["detected_power_W"])
    assert light.kind is ValueKind.UPPER_BOUND
    assert [(p.part, p.upper) for p in light.omitted] == [(OmittedPart.BEAM_PART_ELSEWHERE, pytest.approx(0.1 * reading, rel=1e-9))]
    assert quantities["detector_snr"].value == pytest.approx(detector_snr(responsivity(0.8, 780.0) * reading), rel=1e-12)
    assert entry(quantities["detector_snr"]).kind is ValueKind.UPPER_BOUND
    # (the SNR lies between that of the least power the entry allows -- the annular miss removed -- and the traced value)
    assert set(detected.bounds) == {"detector_snr"}
    low, high = detected.bounds["detector_snr"]
    assert low == pytest.approx(detector_snr(responsivity(0.8, 780.0) * 0.9 * reading), rel=1e-9)
    assert high == pytest.approx(detector_snr(responsivity(0.8, 780.0) * reading), rel=1e-12)


# ------------------------------------------------------------------ light left out as negligible (W28)


def _dark_detector_scene(targets: tuple, *, lit: bool = False, lit_power_W: float = 1e-9, electrical_changes: dict | None = None,
                         lock_in_: dict | None = None):
    """The dark plane of the left-out-light case (a 1 W 1064 nm beam through an N-BK7 window; the plane DARK beside it is
    absorbing, reached by nothing traced or followed, KNOWN_ZERO with a left-out bound of about 1.8e-10 W) with a
    detector on it. ``lit``: a beam of ``lit_power_W`` (1 nW) of a second laser F also lands on it (a KNOWN plane);
    ``electrical_changes``: stated electrical parameters changed; ``lock_in_``: a lock-in on the detector."""
    from helpers import run as run_snapshot
    from os_optics import builders as B
    from os_optics.contracts import ProjectSnapshot

    detection = Detection.model_validate({"detector": {
        "quantum_efficiency": {"rows": [{"wavelength_nm": 400.0, "qe": 0.8}, {"wavelength_nm": 1100.0, "qe": 0.8}], "source": CHAR},
        "electrical": electrical(**(electrical_changes or {}))}, **({"lock_in": lock_in_} if lock_in_ else {})})
    src = B.laser("S", wavelength_nm=1064.0, power_W=1.0, waist_radius_mm=0.5, position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    win = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(50.0, 0.0, 0.0))
    dump = B.beam_dump("D0", position=(150.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=12.7)
    plane = B.observation_plane("DARK", position=(50.0, 80.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=12.7)
    plane = plane.model_copy(update={"absorbing": True, "detection": detection})
    components = [src, win, dump]
    if lit:
        components.append(B.laser("F", wavelength_nm=1064.0, power_W=lit_power_W, waist_radius_mm=0.5, position=(50.0, 120.0, 0.0),
                                  direction=(0.0, -1.0, 0.0)))
    return run_snapshot(ProjectSnapshot(components=tuple(components), observations=(plane,), targets=targets))


def test_a_detector_on_a_plane_of_the_models_value_is_graded_with_the_light_left_out():
    """W28 for detection targets. On the dark plane the detector reads 0 W: SNR 0. The light left out as negligible may
    still bring at most its bound P_f (about 1.8e-10 W): at most R P_f of photocurrent, R = q lambda / (h c) (a quantum
    efficiency of 1 at the longest launched wavelength: the bound holds whatever line the light is), against the dark floor
    gives an SNR up to I / sqrt((2 q I + floor^2) B). A maximum of 1 inside [0, that] is not decided (the value 0
    and the truncation reported); a maximum far above it is met. The bandwidth does not depend on the light."""
    targets = (Target(target_id="snr_tight", observation_id="DARK", metric="detector_snr", maximum=1.0),
               Target(target_id="snr_loose", observation_id="DARK", metric="detector_snr", maximum=1e6),
               Target(target_id="band", observation_id="DARK", metric="detector_bandwidth_Hz", maximum=1e9))
    result = _dark_detector_scene(targets)
    (plane_,) = result.observations
    assert plane_.quantity_state.value == "KNOWN_ZERO"
    power = next(t for t in _dark_detector_scene((Target(target_id="p", observation_id="DARK", metric="power_W", maximum=1e-14),)).targets)
    (omitted,) = power.value_basis["value"].omitted
    bound_W = omitted.upper
    assert 1e-11 < bound_W <= 1e-9
    current = responsivity(1.0, 1064.0) * bound_W
    most = detector_snr(current)
    assert most > 1.0
    by_id = {t.target_id: t for t in result.targets}
    tight = by_id["snr_tight"]
    assert (tight.assessment.value, tight.value) == ("INDETERMINATE", 0.0) and "FAINT_LIGHT_BOUNDED" in tight.issue_codes
    (part,) = tight.value_basis["value"].omitted
    assert tight.value_basis["value"].kind.value == "MODEL" and part.part.value == "BELOW_NEGLIGIBLE_LEVEL"
    assert part.upper == pytest.approx(most, rel=1e-6)
    assert by_id["snr_loose"].assessment.value == "MET"
    assert by_id["band"].assessment.value == "MET" and by_id["band"].value_basis is None


def test_a_faint_left_out_bound_on_a_lit_plane_moves_the_snr_target_to_undecided():
    """A 1 nW beam lights the detector (a KNOWN plane): the SNR of its photocurrent, with the light left out as negligible
    that may add up to its bound, lies in [SNR(I), SNR(I + R P_f)]; a maximum between the two is not decided, one above
    the upper end is met."""
    probe = _dark_detector_scene((Target(target_id="s", observation_id="DARK", metric="detector_snr", maximum=1e9),), lit=True)
    (plane_,) = probe.observations
    assert plane_.quantity_state.value == "KNOWN"
    (target,) = probe.targets
    low, high = target.value, target.value + target.value_basis["value"].omitted[0].upper
    assert target.value == pytest.approx(detector_snr(responsivity(0.8, 1064.0) * plane_.total_power_W), rel=1e-9)
    assert low < high
    middle = Target(target_id="mid", observation_id="DARK", metric="detector_snr", maximum=0.5 * (low + high))
    above = Target(target_id="above", observation_id="DARK", metric="detector_snr", maximum=high * 1.01)
    by_id = {t.target_id: t for t in _dark_detector_scene((middle, above), lit=True).targets}
    assert by_id["mid"].assessment.value == "INDETERMINATE" and by_id["above"].assessment.value == "MET"


def _left_out_bound_W() -> float:
    """P_f: the bound of the light left out as negligible on the dark plane, as a power target there reads it."""
    power = next(t for t in _dark_detector_scene((Target(target_id="p", observation_id="DARK", metric="power_W", maximum=1e-14),)).targets)
    (omitted,) = power.value_basis["value"].omitted
    return omitted.upper


@pytest.mark.parametrize("saturation_A", [1e-11, 1e-6])
def test_light_left_out_that_may_leave_the_linear_range_does_not_decide_a_detection_target(saturation_A):
    """W28 with a stated linear range. The light left out as negligible may bring at most R P_f of photocurrent (about
    1.6e-10 A). Below a saturation current of 1e-11 A that light may drive the detector past its linear range, where the
    linear SNR formula does not hold: the interval [0, SNR(R P_f)] is not proven, so neither the tight nor the loose maximum
    is decided -- only the SNR's sign is -- and the target carries the open check (peak photocurrent R P_f against the
    stated maximum, within unknown). With 1e-6 A the light stays linear: the interval and the verdicts are those of the
    plane without a stated linear range. The plane's own detection stays the model's values (no entry)."""
    targets = (Target(target_id="snr_tight", observation_id="DARK", metric="detector_snr", maximum=1.0),
               Target(target_id="snr_loose", observation_id="DARK", metric="detector_snr", maximum=1e6),
               Target(target_id="snr_sign", observation_id="DARK", metric="detector_snr", minimum=0.0))
    result = _dark_detector_scene(targets, electrical_changes={"saturation_current_A": saturation_A})
    (plane_,) = result.observations
    assert plane_.quantity_state.value == "KNOWN_ZERO"
    snr = {q.name: q for q in plane_.detection.quantities}["detector_snr"]
    assert snr.value == 0.0 and snr.value_basis is None
    current = responsivity(1.0, 1064.0) * _left_out_bound_W()
    by_id = {t.target_id: t for t in result.targets}
    assert by_id["snr_tight"].assessment.value == "INDETERMINATE" and by_id["snr_sign"].assessment.value == "MET"
    (part,) = by_id["snr_tight"].value_basis["value"].omitted
    if saturation_A < current:
        assert by_id["snr_loose"].assessment.value == "INDETERMINATE"
        for target in by_id.values():
            (check,) = target.validity
            assert (check.parameter, check.unit, check.maximum, check.within) == ("peak_photocurrent_A", "A", saturation_A, None)
            assert check.observed == pytest.approx(current, rel=1e-6)
        assert part.upper is None
    else:
        assert by_id["snr_loose"].assessment.value == "MET" and all(not t.validity for t in by_id.values())
        assert part.upper == pytest.approx(detector_snr(current), rel=1e-6)


def test_light_left_out_that_may_be_chopped_above_the_detector_band_does_not_decide_a_detection_target():
    """W28 with a lock-in chopping the 1 W source (SQUARE) at 100 kHz, above the detector's 10 kHz band: none of that
    source's light is traced to the plane, but the light left out as negligible may be it, and a chopped signal above the
    band is not the detector model's. The SNR maximum is not decided, and the target names the open check."""
    chop = {"modulation": {"waveform": "SQUARE", "frequency_Hz": 1e5, "duty_cycle": 0.5,
                           "sources": [{"source_id": "S", "fraction": 1.0}]}, "time_constant_s": TAU_S, "filter_order": 2}
    (target,) = _dark_detector_scene((Target(target_id="snr", observation_id="DARK", metric="detector_snr", maximum=1e6),),
                                     lock_in_=chop).targets
    assert target.assessment.value == "INDETERMINATE" and target.value == 0.0
    (check,) = target.validity
    assert (check.parameter, check.maximum, check.observed, check.within) == ("modulation_frequency_Hz", FC_HZ, 1e5, False)


def _lock_in_detection() -> Detection:
    return Detection.model_validate({"detector": {
        "quantum_efficiency": {"rows": [{"wavelength_nm": 400.0, "qe": 0.8}, {"wavelength_nm": 1100.0, "qe": 0.8}], "source": CHAR},
        "electrical": electrical(dark_current_A=1e-15, load_resistance_ohm=1e12)},
        "lock_in": {"modulation": {"waveform": "SQUARE", "frequency_Hz": 1000.0, "duty_cycle": DUTY,
                                   "sources": [{"source_id": "F", "fraction": 1.0}]}, "time_constant_s": TAU_S, "filter_order": 2}})


def _lock_in_scene(targets: tuple):
    """The dark plane lit by a 100 nW beam of F, chopped (SQUARE, duty 0.5, 1 kHz) for a lock-in; a 1e-15 A dark current
    and a 1e12 ohm load keep the floor far below the shot noise."""
    spec = _lock_in_detection()
    return _dark_detector_scene(targets, lit=True, lit_power_W=1e-7, lock_in_=spec.lock_in.model_dump(mode="json"),
                                electrical_changes={"dark_current_A": 1e-15, "load_resistance_ohm": 1e12})


def test_a_lock_in_snr_beside_light_left_out_states_how_far_that_light_may_move_it_either_way():
    """The left-out light adds at most R P_f. Added as steady current it adds noise and no signal: the lock-in's SNR falls
    to SNR(least); added as modulated current it rises to SNR(most) -- both closed forms here. The fall is about twice the
    rise, and the target's entry states the larger: a minimum just below value - upper is met, one inside [least, value) is
    not decided."""
    probe = _lock_in_scene((Target(target_id="lk", observation_id="DARK", metric="lock_in_snr", minimum=1.0),))
    (plane_,) = probe.observations
    (target,) = probe.targets
    floor2 = 2.0 * Q * 1e-15 + 4.0 * K_B * TEMP_K / 1e12
    current = responsivity(0.8, 1064.0) * plane_.total_power_W
    added = responsivity(1.0, 1064.0) * _left_out_bound_W()
    signal = FUNDAMENTAL * current / math.sqrt(2.0)
    value = signal / math.sqrt((2.0 * Q * WEIGHT * current + floor2) * ENBW_HZ)
    least = signal / math.sqrt((2.0 * Q * (WEIGHT * current + added) + floor2) * ENBW_HZ)
    most = FUNDAMENTAL * (current + added) / math.sqrt(2.0) / math.sqrt((2.0 * Q * WEIGHT * (current + added) + floor2) * ENBW_HZ)
    assert target.value == pytest.approx(value, rel=1e-9)
    assert value - least > most - value > 0.0
    (part,) = target.value_basis["value"].omitted
    assert part.upper == pytest.approx(value - least, rel=1e-6)
    below = Target(target_id="below", observation_id="DARK", metric="lock_in_snr", minimum=value - 1.01 * part.upper)
    inside = Target(target_id="inside", observation_id="DARK", metric="lock_in_snr", minimum=0.5 * (least + value))
    by_id = {t.target_id: t for t in _lock_in_scene((below, inside)).targets}
    assert by_id["below"].assessment.value == "MET" and by_id["inside"].assessment.value == "INDETERMINATE"


def test_an_unbounded_left_out_light_leaves_a_lock_in_snr_only_its_sign():
    """Where nothing bounds the light left out as negligible, light added as steady current may lower the lock-in's SNR
    towards 0: the target is graded on [0, inf), so a minimum below the value is not decided. The photocurrent's SNR, which
    added light only raises, keeps its value as the lower end. The entry states no finite reach."""
    from os_optics.validation.metrics import _detection_metric

    (plane_,) = _lock_in_scene(()).observations
    for metric, lower in (("lock_in_snr", 0.0), ("detector_snr", None)):
        target = Target(target_id="t", observation_id="DARK", metric=metric, minimum=1.0)
        mv = _detection_metric(target, {"DARK": plane_}, {"DARK": _lock_in_detection()}, left_out_W=math.inf)
        assert mv.value is not None and mv.value > 1.0
        assert mv.untraced == ((mv.value if lower is None else lower), math.inf), metric
        assert mv.basis.omitted[0].upper is None and not mv.validity


# ------------------------------------------------------------- a taken share with light left out as negligible (W28)


def _taken_share_plane(lock_in_: dict | None = None, **electrical_changes):
    """The fold of ``test_taken_before_interval``: a 1 mW beam onto a 45 deg mirror (R = 1 - 1e-10), and an absorbing plane T
    with a flat-QE detector 0.5 mm before it -- the mirror may take a bounded share of the beam before T (its power an upper
    end, ``power_lower_bound_W`` below it), and the fate of T's light is complete."""
    from helpers import run as run_snapshot
    from os_optics import builders as B
    from os_optics.contracts import ProjectSnapshot

    tilt = math.radians(45.0)
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-math.cos(tilt), math.sin(tilt), 0.0), reflectance=1.0 - 1e-10,
                      clear_radius_mm=20.0)
    laser = B.laser("L", wavelength_nm=780.0, waist_radius_mm=0.5, power_W=1e-3, position=(-30.0, 0.0, 0.0),
                    direction=(1.0, 0.0, 0.0), waist_position_z_mm=30.0)
    detection = Detection.model_validate({"detector": {"quantum_efficiency": QE_FLAT, "electrical": electrical(**electrical_changes)},
                                          **({"lock_in": lock_in_} if lock_in_ is not None else {})})
    plane_t = B.observation_plane("T", position=(-0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=50.0,
                                  absorbing=True).model_copy(update={"detection": detection})
    (obs,) = run_snapshot(ProjectSnapshot(components=(laser, mirror), observations=(plane_t,), targets=())).observations
    return obs, detection


@pytest.mark.parametrize("bound, side, expected", [
    ("half", "minimum", "INDETERMINATE"),  # (UNMET before: the truncation decided it)
    ("half", "maximum", "INDETERMINATE"),  # (MET before)
    ("above", "minimum", "UNMET"),
    ("above", "maximum", "MET"),
    ("below", "minimum", "INDETERMINATE"),  # (just below the value, above the SNR of the least power: not MET)
])
def test_a_detector_snr_on_a_taken_share_keeps_the_light_left_out_as_negligible_in_its_interval(monkeypatch, bound, side,
                                                                                               expected):
    """The plane's power is an upper end (the share M may take before it), its fate complete, and light left out as negligible
    may add at most 1e-12 W (injected as ``_detection_left_out`` reads it: no faint light reaches T in this simple scene). The
    photocurrent's SNR rises with the light: it lies between the SNR of the least power the entry allows (P less the taken
    share, at the closed form) and the SNR with that light added at a quantum efficiency of 1 (closed form). A limit inside
    that interval is not decided by the truncation (decision W28), a maximum above it is met, a minimum above it is not; the
    entry is the upper bound merged with the stated truncation."""
    from os_optics.config import NumericalConfig
    from os_optics.validation import metrics as M

    obs, detection = _taken_share_plane()
    assert obs.quantity_state is QuantityState.BOUNDED and obs.power_complete and obs.power_lower_bound_W < obs.total_power_W
    value = next(q for q in obs.detection.quantities if q.name == "detector_snr").value
    current = responsivity(0.8, 780.0) * obs.total_power_W
    assert value == pytest.approx(detector_snr(current), rel=1e-9)
    faint = 1e-12
    reach = detector_snr(current + responsivity(1.0, 780.0) * faint) - value
    least = detector_snr(responsivity(0.8, 780.0) * obs.power_lower_bound_W)
    assert 0.0 < reach and least < value
    limit = {"half": value + 0.5 * reach, "above": value + 2.0 * reach, "below": 0.5 * (least + value)}[bound]
    monkeypatch.setattr(M, "_detection_left_out", lambda *args, **kwargs: faint)
    target = Target(target_id="s", metric="detector_snr", observation_id="T", unit="1", **{side: limit})
    graded = M._evaluate(target, {"T": obs}, [], None, None, NumericalConfig(), detections={"T": detection})
    assert graded.assessment.value == expected, graded.message
    assert graded.value == value
    entry = graded.value_basis["value"]
    assert entry.kind is ValueKind.UPPER_BOUND
    assert {o.part for o in entry.omitted} == {OmittedPart.BEAM_PART_ELSEWHERE, OmittedPart.BELOW_NEGLIGIBLE_LEVEL}
    truncation = next(o for o in entry.omitted if o.part is OmittedPart.BELOW_NEGLIGIBLE_LEVEL)
    assert truncation.upper == pytest.approx(reach, rel=1e-6)
    assert "light left out as negligible" in graded.message
    # (the cause of the upper end stays named beside the truncation, as on a power_W target of the same plane)
    assert {"POWER_TAKEN_BEFORE_PLANE", "FAINT_LIGHT_BOUNDED"} <= set(graded.issue_codes)
    assert len(graded.issue_codes) == len(set(graded.issue_codes))


@pytest.mark.parametrize("side, where, expected", [
    ("minimum", "below_least", "MET"),
    ("minimum", "between", "INDETERMINATE"),  # (INDETERMINATE for every minimum before: no interval was graded)
    ("minimum", "above", "UNMET"),
    ("maximum", "above", "MET"),
    ("maximum", "between", "INDETERMINATE"),
])
def test_a_detector_snr_on_a_taken_share_without_light_left_out_is_graded_on_its_proven_interval(side, where, expected):
    """The taken-share plane with nothing left out as negligible (the real ``_detection_left_out`` finds none here): the
    photocurrent's SNR lies between the SNR of the least power the entry allows (closed form at ``power_lower_bound_W``)
    and the traced value, and a target is graded on that interval as ``power_W`` on the same plane is -- a minimum below
    the least SNR is met, one between it and the value is not decided, one above the value is not met. The entry stays
    the plane's upper bound, its cause named."""
    from os_optics.config import NumericalConfig
    from os_optics.validation import metrics as M

    obs, detection = _taken_share_plane()
    assert obs.quantity_state is QuantityState.BOUNDED and obs.power_complete and obs.power_lower_bound_W < obs.total_power_W
    value = next(q for q in obs.detection.quantities if q.name == "detector_snr").value
    least = detector_snr(responsivity(0.8, 780.0) * obs.power_lower_bound_W)
    assert 0.0 < least < value
    limit = {"below_least": 0.5 * least, "between": 0.5 * (least + value), "above": value * (1.0 + 1e-6)}[where]
    target = Target(target_id="s", metric="detector_snr", observation_id="T", unit="1", **{side: limit})
    graded = M._evaluate(target, {"T": obs}, [], None, None, NumericalConfig(), detections={"T": detection})
    assert graded.assessment.value == expected, graded.message
    assert graded.value == value
    entry_ = graded.value_basis["value"]
    assert entry_.kind is ValueKind.UPPER_BOUND
    assert {o.part for o in entry_.omitted} == {OmittedPart.BEAM_PART_ELSEWHERE}
    assert "POWER_TAKEN_BEFORE_PLANE" in graded.issue_codes and "FAINT_LIGHT_BOUNDED" not in graded.issue_codes
    detected = detect(obs, detection, QuantityState.BOUNDED)
    low, high = detected.bounds["detector_snr"]
    assert low == pytest.approx(least, rel=1e-6) and high == value


@pytest.mark.parametrize("where", ["near", "far"])
def test_a_lock_in_snr_on_a_taken_share_carries_the_light_left_out_as_negligible(monkeypatch, where):
    """The taken-share plane with a SQUARE lock-in on L (duty 0.5, 1 kHz) and light left out as negligible of at most
    1e-12 W (injected): the lock-in's SNR is graded with its entry (an ESTIMATE: no proven interval on an upper-end plane),
    so a minimum near or far below the value is not decided; the entry and the codes name both the share taken before the
    plane and the stated truncation, whose reach is the larger of the closed-form fall (that light as steady current) and
    rise (as modulated current)."""
    from os_optics.config import NumericalConfig
    from os_optics.validation import metrics as M

    chop = {"modulation": {"waveform": "SQUARE", "frequency_Hz": 1000.0, "duty_cycle": DUTY,
                           "sources": [{"source_id": "L", "fraction": 1.0}]}, "time_constant_s": TAU_S, "filter_order": 2}
    obs, detection = _taken_share_plane(chop)
    value = next(q for q in obs.detection.quantities if q.name == "lock_in_snr").value
    current = responsivity(0.8, 780.0) * obs.total_power_W
    assert value == pytest.approx(lock_in_snr(0.0, current), rel=1e-9)
    faint = 1e-12
    added = responsivity(1.0, 780.0) * faint
    signal = FUNDAMENTAL * current / math.sqrt(2.0)
    least = signal / math.sqrt((2.0 * Q * (WEIGHT * current + added) + FLOOR2) * ENBW_HZ)
    most = lock_in_snr(0.0, current + added)
    reach = max(value - least, most - value)
    assert least < value < most and reach > 0.0
    monkeypatch.setattr(M, "_detection_left_out", lambda *args, **kwargs: faint)
    limit = {"near": value - 0.5 * reach, "far": 0.5 * value}[where]
    target = Target(target_id="lk", metric="lock_in_snr", observation_id="T", unit="1", minimum=limit)
    graded = M._evaluate(target, {"T": obs}, [], None, None, NumericalConfig(), detections={"T": detection})
    assert graded.assessment.value == "INDETERMINATE", graded.message
    assert graded.value == value
    assert {"POWER_TAKEN_BEFORE_PLANE", "FAINT_LIGHT_BOUNDED"} <= set(graded.issue_codes)
    assert len(graded.issue_codes) == len(set(graded.issue_codes))
    entry_ = graded.value_basis["value"]
    assert entry_.kind is ValueKind.ESTIMATE
    assert {o.part for o in entry_.omitted} == {OmittedPart.BEAM_PART_ELSEWHERE, OmittedPart.BELOW_NEGLIGIBLE_LEVEL}
    truncation = next(o for o in entry_.omitted if o.part is OmittedPart.BELOW_NEGLIGIBLE_LEVEL)
    assert truncation.upper == pytest.approx(reach, rel=1e-6)
    assert truncation.setting == "negligible_power_fraction"
    assert "light left out as negligible" in graded.message and "at most 1e-12 W" in graded.message


def test_a_detector_snr_on_a_taken_share_whose_left_out_light_may_leave_the_linear_range_has_no_upper_end(monkeypatch):
    """The same plane with a saturation current just above the traced photocurrent: the light left out as negligible may
    push the peak photocurrent past the linear range, where the detector model does not hold. The target is graded from the
    SNR of the least power with no upper end -- a maximum far above the value is not decided -- and carries that open
    validity condition."""
    from os_optics.config import NumericalConfig
    from os_optics.validation import metrics as M

    probe, _ = _taken_share_plane()
    current = responsivity(0.8, 780.0) * probe.total_power_W
    obs, detection = _taken_share_plane(saturation_current_A=current * (1.0 + 1e-12))
    value = next(q for q in obs.detection.quantities if q.name == "detector_snr").value
    monkeypatch.setattr(M, "_detection_left_out", lambda *args, **kwargs: 1e-12)
    target = Target(target_id="s", metric="detector_snr", observation_id="T", maximum=2.0 * value, unit="1")
    graded = M._evaluate(target, {"T": obs}, [], None, None, NumericalConfig(), detections={"T": detection})
    assert graded.assessment.value == "INDETERMINATE", graded.message
    assert [c.parameter for c in graded.validity] == ["peak_photocurrent_A"]


def test_the_physics_scope_and_readme_state_both_ends_of_the_detector_snr_interval_on_a_taken_share():
    """P9d and the README say what the tests above grade: on a plane whose power is an upper end the detector_snr interval
    starts at the SNR of the least power the plane's entry allows (below the traced value), and ends at the SNR with the
    left-out light at a quantum efficiency of 1, unbounded where a check it may leave open applies."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    scope = " ".join((root / "docs" / "engine" / "PHYSICS_SCOPE.md").read_text(encoding="utf-8").split())
    assert "(`detector_snr`: from the traced value to the value with that light's largest photocurrent;" not in scope
    assert ("on a plane whose power is an upper end (a share taken before it, an annular beam counted whole) and whose fate "
            "is complete, it is the SNR of the least power the plane's entry allows -- the taken shares and the annular "
            "misses removed, at the largest responsivity -- and 0 where that entry omits or assumes anything else") in scope
    assert ("there is none where nothing bounds that light or where a check it may leave open applies "
            "(`peak_photocurrent_A`, a `SQUARE` chop above the band), those checks carried in the target's `validity`") in scope
    assert "on a plane whose power is an upper end it is graded with its entry" in scope
    assert ("on a plane whose power is an upper end it is graded with its entry, which carries light left out as negligible "
            "as the stated truncation (`BELOW_NEGLIGIBLE_LEVEL`, how far that light may move the value either way) beside the "
            "plane's cause, with `FAINT_LIGHT_BOUNDED` among its codes") in scope
    assert ("there the interval up to the traced value holds and is graded with or without light left out as negligible, "
            "which only widens its upper end") in scope
    readme = " ".join((root / "docs" / "engine" / "README.md").read_text(encoding="utf-8").split())
    assert ("on a plane whose power is an upper end (a share taken before it, an annular beam counted whole) the "
            "`detector_snr` interval starts at the SNR of the least power the plane's entry allows, below the traced value, "
            "and a target is graded on that interval whether or not light was left out as negligible (that light only widens "
            "its upper end)") in readme
