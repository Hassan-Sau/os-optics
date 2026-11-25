"""The pulses a pulsed source sends while the optics stay in place (``PulseTrain.total_pulses``) and the pulse-count
condition of a power rating against the pulses one site of a rated surface receives.

Rules under test (``validation/rating_conditions.py``): each crossing of a source's light brings that source's total; a
site receives between the fewest one counted crossing brings and all of them; crossings that carry together at most 1 %
of the most light (in peak irradiance, the rating's own quantity, fluence and pulse energy) do not count. The rating's
count ``S`` is compared per tested site only where that is what it states: a single-pulse rating (``S = 1``) or a count
applied at one fluence level to each tested site (``test_protocol`` S_ON_1). Then: a source that states no total -> not
verifiable (DESIGN_INPUT_MISSING); exactly ``S`` pulses at every site -> verified; at least ``S`` at every site ->
threshold at most the rated value (a site damaged within ``S`` pulses is damaged within more); at most ``S`` -> at least
the rated value (a site that survived ``S`` survived its first fewer); several crossings of fewer than ``S`` each but more
together -> PULSES_PER_SITE_UNDETERMINED. A rating that states no count, or a count above one without its protocol, ->
not verifiable (RATING_INFORMATION_MISSING) whatever the sources' totals. The exposure never depends on the count.

Fixtures: an ideal mirror at 45 degrees (one beam: exact exposure), an ideal thin lens crossed twice (a retro-reflecting
mirror behind it), fixture lasers and fixture ratings written here (``value_definition`` PEAK; the beams' test beam size,
repetition rate and linear polarization), never product data.
"""

from __future__ import annotations

import copy
import math

import pytest

from os_optics import builders as B
from os_optics.contracts import PowerRating, ProjectSnapshot, PulseTrain, Target
from os_optics.local import json_api, lower_project
from os_optics.manifest import hash_exact

from catalog_store_helpers import REAL_CATALOG, catalog_version_of
from identity_mask import mask_request_identity
from os_optics.local.compile import environment_air

from helpers import run

FIXTURE = "test fixture rating, not product data"
PULSE = PulseTrain(repetition_rate_hz=10.0, pulse_duration_s=1e-8)
#: A fixture rating's count stated as applied at one fluence level to each tested site (S-on-1).
S_ON_1 = dict(test_protocol="S_ON_1")
#: The laser's 1/e^2 diameter at the mirror 100 mm away (w0 = 0.5 mm, Rayleigh range pi w0^2 / lambda).
TEST_BEAM_MM = 2.0 * 0.5 * math.sqrt(1.0 + (100.0 / (math.pi * 0.25 / (1064.0 * 1e-6))) ** 2)
#: The test conditions the fixture ratings state besides their value, count and pulse train, met by the beams here: a test
#: beam of the laser's size at the mirror and linear polarization (the lasers'), so that the count alone decides.
STATED = dict(beam_diameter_mm=TEST_BEAM_MM, beam_diameter_definition="1/E2", polarization_condition="linear", polarization="LINEAR")


def _laser(instance_id: str = "A", *, total: int | str | None = None, power_W: float = 1.0, position=(-100.0, 0.0, 0.0),
           direction=(1.0, 0.0, 0.0), pulsed: bool = True, **kwargs):
    laser = B.laser(instance_id, wavelength_nm=1064.0, power_W=power_W, waist_radius_mm=0.5, position=position, direction=direction,
                    **kwargs)
    pulse = PULSE.model_copy(update={"total_pulses": total}) if pulsed else None
    return laser.model_copy(update={"source": laser.source.model_copy(update={"pulse": pulse})})


def _pulsed_rating(value: float, count: int | None = 100, **extra) -> PowerRating:
    extra = {"value_definition": "PEAK", **STATED, **extra}
    return PowerRating(regime="PULSED", value=value, unit="J/cm2", wavelength_nm=1064.0, pulse_duration_s=1e-8,
                       repetition_rate_hz=10.0, pulse_count=count, status="DECLARED", basis=FIXTURE, **extra)


def _mirror_run(lasers, ratings: tuple[PowerRating, ...], maximum: float = 1.0):
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(update={"power_ratings": ratings})
    target = Target(target_id="u", device_id="M", metric="power_rating_utilization", maximum=maximum)
    result = run(ProjectSnapshot(components=(*lasers, mirror), targets=(target,)))
    face = next(s for s in result.exposures.surfaces if s.surface_id == "M/face")
    return face, result.targets[0]


def _count(comparison):
    return next(c for c in comparison.conditions if c.condition == "pulse_count")


def _state(condition) -> tuple:
    return condition.state, condition.effect or condition.reason


def _fluence_on_mirror() -> float:
    """The peak fluence of the 1 W, 10 Hz beam on the mirror from its own hit: 2 E / (pi w_x w_y), E = P / f."""
    face, _ = _mirror_run((_laser(),), (_pulsed_rating(1e30),))
    (hit,) = face.hits
    energy = hit.incident_power_W / PULSE.repetition_rate_hz
    own = 2.0 * energy / (math.pi * hit.beam_radius_x_mm * hit.beam_radius_y_mm) * 100.0  # J/mm2 -> J/cm2
    assert face.ratings[0].exposure_value == pytest.approx(own, rel=1e-12)
    return own


# ------------------------------------------------------------------------------------------------ one source, one crossing


@pytest.mark.parametrize(("total", "expected", "applicability"), [
    (None, ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING"), "NOT_VERIFIED"),
    ("UNLIMITED", ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), "THRESHOLD_AT_MOST_RATED"),
    (100, ("VERIFIED", None), "AS_RATED"),
    (101, ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), "THRESHOLD_AT_MOST_RATED"),
    (99, ("DIFFERS", "THRESHOLD_AT_LEAST_RATED"), "THRESHOLD_AT_LEAST_RATED"),
    (1, ("DIFFERS", "THRESHOLD_AT_LEAST_RATED"), "THRESHOLD_AT_LEAST_RATED"),
])
def test_one_crossing_against_a_100_on_1_rating(total, expected, applicability):
    """The rating states 100 pulses on each tested site at one fluence level (S-on-1): one crossing of a source sending
    exactly 100 is the test's condition; more give a threshold at most the rated value (damage within 100 pulses is damage
    within more), fewer one at least the rated value (a site that survived 100 survived its first fewer)."""
    fluence = _fluence_on_mirror()
    face, target = _mirror_run((_laser(total=total),), (_pulsed_rating(0.5 * fluence, **S_ON_1),))
    (comparison,) = face.ratings
    count = _count(comparison)
    assert _state(count) == expected and comparison.applicability == applicability
    # the exposure is the same whatever the count: computed independently of it
    assert comparison.exposure_value == pytest.approx(fluence, rel=1e-12)
    if total is None:
        assert "A states no pulse.total_pulses" in count.message
    else:
        assert count.design == total
    if applicability in ("AS_RATED", "THRESHOLD_AT_MOST_RATED"):
        # twice the rating: an exceedance stands, and the target fails
        assert comparison.status == "EXCEEDS_RATING" and target.assessment.value == "UNMET"
    elif applicability == "THRESHOLD_AT_LEAST_RATED":
        # (a threshold at least the rated value: twice the value is neither within nor shown to exceed it)
        assert comparison.status == "UNKNOWN" and target.assessment.value == "INDETERMINATE"
    else:
        assert comparison.status == "CONDITIONS_NOT_VERIFIED" and target.assessment.value == "INDETERMINATE"


@pytest.mark.parametrize("total", [None, "UNLIMITED", 100, 101, 99, 1])
def test_a_100_pulse_rating_that_states_no_protocol_is_not_verifiable_whatever_the_total(total):
    """The rating states 100 pulses but not how they were applied to each tested site (at one level, ramped, raster): no
    relation of its value to any number of pulses per site -- 100 included -- is established, so the count is not
    verifiable whatever the source sends, and neither the exceedance of twice the value nor any other decision follows."""
    fluence = _fluence_on_mirror()
    face, target = _mirror_run((_laser(total=total),), (_pulsed_rating(0.5 * fluence),))
    (comparison,) = face.ratings
    count = _count(comparison)
    assert _state(count) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING") and count.rated == 100
    assert "not how they were applied" in count.message
    assert comparison.exposure_value == pytest.approx(fluence, rel=1e-12)
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and target.assessment.value == "INDETERMINATE"


def test_more_pulses_than_the_test_never_decide_a_within():
    """Under a stated 100-on-1 test, unlimited pulses per site give a threshold at most the rated value: an exposure of
    half the value is not shown within it."""
    fluence = _fluence_on_mirror()
    face, target = _mirror_run((_laser(total="UNLIMITED"),), (_pulsed_rating(2.0 * fluence, **S_ON_1),))
    (comparison,) = face.ratings
    assert comparison.applicability == "THRESHOLD_AT_MOST_RATED" and comparison.status == "UNKNOWN"
    assert target.assessment.value == "INDETERMINATE"


@pytest.mark.parametrize(("total", "expected", "status"), [
    (1, ("VERIFIED", None), "WITHIN_RATING"),
    (2, ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), "UNKNOWN"),
    ("UNLIMITED", ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), "UNKNOWN"),
])
def test_a_single_pulse_rating_against_a_single_shot(total, expected, status):
    fluence = _fluence_on_mirror()
    face, target = _mirror_run((_laser(total=total),), (_pulsed_rating(2.0 * fluence, count=1),))
    (comparison,) = face.ratings
    assert _state(_count(comparison)) == expected and comparison.status == status
    assert target.assessment.value == ("MET" if status == "WITHIN_RATING" else "INDETERMINATE")


@pytest.mark.parametrize("total", ["UNLIMITED", 1, 2**53 - 1, None])
def test_a_rating_that_states_no_count_is_not_verifiable_whatever_the_total(total):
    """A pulsed rating that states no pulse count does not say how many pulses each tested site received: its value is
    compared with no number of pulses, so the count is not verifiable (RATING_INFORMATION_MISSING) and no result depends on
    the total -- an exposure half the value is not within it even for a single shot (an earlier draft rule
    that made UNLIMITED alone at most the rated value: an omitted or a very large finite total then still decided
    "within")."""
    fluence = _fluence_on_mirror()
    face, target = _mirror_run((_laser(total=total),), (_pulsed_rating(2.0 * fluence, count=None),))
    (comparison,) = face.ratings
    assert _state(_count(comparison)) == ("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")
    assert comparison.applicability == "NOT_VERIFIED" and comparison.exposure_value == pytest.approx(fluence, rel=1e-12)
    assert comparison.status == "CONDITIONS_NOT_VERIFIED" and target.assessment.value == "INDETERMINATE"


def _two_line_laser(total: int | str | None):
    """One 1 W, 10 Hz source with two equal lines (1063.5 and 1064.5 nm): both lines are parts of the same pulses."""
    laser = B.laser("A", wavelength_nm=None, spectrum=((1063.5, 1.0), (1064.5, 1.0)), power_W=1.0, waist_radius_mm=0.5,
                    position=(-100.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    pulse = PULSE.model_copy(update={"total_pulses": total})
    return laser.model_copy(update={"source": laser.source.model_copy(update={"pulse": pulse})})


@pytest.mark.parametrize(("total", "count", "expected", "design", "status"), [
    (1, 1, ("VERIFIED", None), 1, "WITHIN_RATING"),
    # (100-on-1: 60 pulses per site are fewer than the test; two crossings would make it 60 to 120, undetermined)
    (60, 100, ("DIFFERS", "THRESHOLD_AT_LEAST_RATED"), 60, "WITHIN_RATING"),
    ("UNLIMITED", 100, ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), "UNLIMITED", "UNKNOWN"),
])
def test_the_lines_of_one_source_along_one_path_are_one_crossing(total, count, expected, design, status):
    """A site receives the source's total once, not once per line (an earlier draft counted each line
    as a crossing: a two-line single shot read as two pulses per site). The 100-pulse ratings state S-on-1, so their
    count is compared per site."""
    fluence = _fluence_on_mirror()
    protocol = S_ON_1 if count > 1 else {}
    face, _ = _mirror_run((_two_line_laser(total),), (_pulsed_rating(2.0 * fluence, count=count, **protocol),))
    (comparison,) = face.ratings
    condition = _count(comparison)
    assert _state(condition) == expected and condition.design == design and comparison.status == status


def test_continuous_light_under_a_pulsed_count_rating_differs_in_its_count_too():
    face, _ = _mirror_run((_laser(pulsed=False),), (_pulsed_rating(1.0),))
    (comparison,) = face.ratings
    assert _state(_count(comparison)) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")
    assert comparison.status == "CONDITIONS_MISMATCH"


def test_a_metallic_single_pulse_rating_keeps_its_beam_size_direction_for_one_pulse_per_site_only():
    """A 10 um single-pulse test beam on a metallic reflector against the 1 mm beam: a single shot keeps the threshold at
    most the rated value; two pulses per site on metal have no established relationship across beam sizes; not stated:
    not verifiable."""
    fluence = _fluence_on_mirror()
    metal = dict(count=1, beam_diameter_mm=0.01, beam_diameter_definition="1/E2", surface_kind="METALLIC_REFLECTOR")
    expected = {1: ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), 2: ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP"),
                None: ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING")}
    for total, beam in expected.items():
        face, _ = _mirror_run((_laser(total=total),), (_pulsed_rating(0.5 * fluence, **metal),))
        (comparison,) = face.ratings
        condition = next(c for c in comparison.conditions if c.condition == "beam_diameter")
        assert _state(condition) == beam, total


def test_continuous_light_on_a_metallic_single_pulse_rating_has_no_pulses_per_site():
    """Continuous light brings no pulses: the metallic exception has no established relationship (an earlier draft found
    a draft that asked for the source's total_pulses)."""
    fluence = _fluence_on_mirror()
    metal = dict(count=1, beam_diameter_mm=0.01, beam_diameter_definition="1/E2", surface_kind="METALLIC_REFLECTOR")
    face, _ = _mirror_run((_laser(pulsed=False),), (_pulsed_rating(0.5 * fluence, **metal),))
    (comparison,) = face.ratings
    condition = next(c for c in comparison.conditions if c.condition == "beam_diameter")
    assert _state(condition) == ("DIFFERS", "NO_ESTABLISHED_RELATIONSHIP")
    assert "total_pulses" not in condition.message and comparison.status == "CONDITIONS_MISMATCH"


# ------------------------------------------------------------------------------------------------ several sources


def _two(total_a, total_b, power_b: float):
    a = _laser("A", total=total_a)
    b = _laser("B", total=total_b, power_W=power_b, position=(-120.0, 0.0, 0.0))
    return a, b


@pytest.mark.parametrize(("total_a", "total_b", "power_b", "expected", "named"), [
    ("UNLIMITED", "UNLIMITED", 1.0, ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), None),
    ("UNLIMITED", None, 1.0, ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING"), "B"),
    # a source carrying 0.05 % of the light (within the 1 % share) does not count; one carrying the light does
    ("UNLIMITED", None, 5e-4, ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), None),
    (None, "UNLIMITED", 5e-4, ("NOT_VERIFIABLE", "DESIGN_INPUT_MISSING"), "A"),
    # two crossings (one per source) of fewer pulses than the test each, more together at a site where both meet
    (60, 60, 1.0, ("NOT_VERIFIABLE", "PULSES_PER_SITE_UNDETERMINED"), None),
    # ... and no more than the test together: a threshold at least the rated value under the 100-on-1 test
    (50, 50, 1.0, ("DIFFERS", "THRESHOLD_AT_LEAST_RATED"), None),
    (100, 1000, 1.0, ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), None),
])
def test_several_sources(total_a, total_b, power_b, expected, named):
    """The pulses per site several sources bring, against a 100-on-1 rating (a count compared per site)."""
    face, _ = _mirror_run(_two(total_a, total_b, power_b), (_pulsed_rating(1e30, **S_ON_1),))
    (comparison,) = face.ratings
    count = _count(comparison)
    assert _state(count) == expected
    if named is not None:
        assert f"{named} states no pulse.total_pulses" in count.message
    if (total_a, total_b) == (60, 60):
        assert (count.design, count.design_low, count.design_high) == (None, 60.0, 120.0)


# ------------------------------------------------------------------------------------------------ one source, two crossings


def _double_pass(total, ratings: tuple[PowerRating, ...]):
    """The beam crosses an ideal thin lens (f = 1e7 mm: nearly no focusing), meets a mirror at normal incidence 50 mm behind
    it and crosses the lens again on its way back: two crossings of the lens's one surface, each a full-power beam."""
    lens = B.thin_lens("L", focal_length_mm=1e7, clear_radius_mm=10.0, position=(0.0, 0.0, 0.0)).model_copy(
        update={"power_ratings": ratings})
    back = B.mirror("R", position=(50.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0))
    result = run(ProjectSnapshot(components=(_laser(total=total), lens, back)))
    surface = next(s for s in result.exposures.surfaces if s.surface_id == "L/lens")
    return surface


@pytest.mark.parametrize(("total", "expected", "low", "high"), [
    (60, ("NOT_VERIFIABLE", "PULSES_PER_SITE_UNDETERMINED"), 60.0, 120.0),
    (50, ("DIFFERS", "THRESHOLD_AT_LEAST_RATED"), 50.0, 100.0),
    (100, ("DIFFERS", "THRESHOLD_AT_MOST_RATED"), 100.0, 200.0),
])
def test_a_double_pass_brings_the_pulses_of_both_crossings(total, expected, low, high):
    """Against a 100-on-1 rating (a count compared per site): a site receives one crossing's pulses to both crossings'."""
    surface = _double_pass(total, (_pulsed_rating(1e30, **S_ON_1),))
    assert len(surface.hits) == 2  # (the forward and the returning beam)
    (comparison,) = surface.ratings
    count = _count(comparison)
    assert _state(count) == expected
    assert (count.design_low, count.design_high) == (low, high)


# ------------------------------------------------------------------------------------------------ Project input and hashes


def _document(total=None) -> dict:
    pulse = {"repetition_rate_hz": 10.0, "pulse_duration_s": 5e-9, "duration_definition": "FWHM"}
    if total is not None:
        pulse["total_pulses"] = total
    return {
        "schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM",
        "components": [
            {"instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "Laser"},
             "nominal_pose": {"position_mm": [0.0, 0.0, -50.0]},
             "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 1064.0, "power_w": 0.5, "waist_radius_mm": 0.5,
                              "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0},
                              "pulse": pulse}},
        ],
        "observations": [{"observation_id": "S", "pose": {"position_mm": [0.0, 0.0, 100.0]}, "aperture": {"shape": "circle", "radius_mm": 10.0}}],
    }


def test_a_project_without_total_pulses_keeps_its_hashes():
    """The osc1 Project hash and the Engine input hash of a pulsed Project that states no total, as recorded by
    the Engine before the field existed."""
    checked = json_api.check_project(_document())
    assert checked["ok"], checked
    assert checked["project_hash"] == "6720fedd4fcee8279232db8f773455f6ba6a51699205c10945af0651dfeaacd4"
    # with the schema version the Engine request stamps masked; unmasked it moved at every schema bump and by nothing else
    # (96651cc1...4a3d at 1.9.0, d6a422a5...6293 at 1.7.0, 726b2c07...c283 at 1.6.0); masked, it moved when the numerical
    # configuration gained `max_return_analysis_work` and the crystal tolerances (config 1.6.0; b64629d2...e60e before), and
    # again when it gained the six second-harmonic tolerances `nonlinear_*_tol` (config 1.6.0; fac25b18...8d2c before, which
    # the request without those six keys still hashes to), and again when it gained the caller settings of config 1.7.0
    # (`negligible_power_fraction`, the scatter-origin and mesh-patch budgets, the `rating_*` windows and the `design_point_*`
    # tolerances; without them and with config_version 1.6.0 the request hashes to 3e8f8b63...b78b5c)
    # (and the request states the environment's air -- here air at 20 C and 101325 Pa, the Project states none --
    # as its first material; with it taken out the request hashes to the pin, nothing else changed)
    request = lower_project(_document()).request
    masked = mask_request_identity(request)
    assert masked["project"]["materials"][0] == environment_air().model_dump(mode="json")
    masked["project"]["materials"] = masked["project"]["materials"][1:]
    assert hash_exact(masked) == "55eff1041034c2578f951b335ec9e7d02222e8d0e39436c409ebae441451d56a"
    assert checked["engine_input_hash"] == hash_exact(request.model_dump(mode="json"))  # the binding
    (source,) = [c.source for c in request.project.components if c.source is not None]
    assert "total_pulses" not in source.pulse.model_dump(mode="json")


@pytest.mark.parametrize("total", [1, 100, "UNLIMITED"])
def test_a_stated_total_is_lowered_and_part_of_both_hashes(total):
    plain = json_api.check_project(_document())
    stated = json_api.check_project(_document(total))
    assert stated["ok"], stated
    assert stated["project_hash"] != plain["project_hash"] and stated["engine_input_hash"] != plain["engine_input_hash"]
    (source,) = [c.source for c in lower_project(_document(total)).request.project.components if c.source is not None]
    assert source.pulse.total_pulses == total


@pytest.mark.parametrize("total", [0, 1.5, "unlimited", True, 2**53])
def test_a_total_outside_the_contract_is_refused(total):
    checked = json_api.check_project(_document(total))
    assert not checked["ok"]
    assert "total_pulses" in str(checked)


def test_the_capabilities_name_the_pulse_count_input():
    assert "source_total_pulses" in json_api.capabilities()["project_features"]


def test_a_run_reply_carries_the_count_condition_through_the_local_api():
    """A pulsed source with UNLIMITED against a declared 100-on-1 rating, through ``json_api.run_project``: the rating's
    protocol is lowered, and the count condition and the per-site design value reach the run's exposure report unchanged."""
    document = copy.deepcopy(_document("UNLIMITED"))
    document["components"].append({
        "instance_id": "W", "optical_role": "window", "custom_part": {"custom_part_key": "win", "description": "Window"},
        "optical_model_id": "win", "nominal_pose": {"position_mm": [0.0, 0.0, 0.0]}})
    from declared_helpers import nbk7, sequence, surface, with_models
    rating = {"regime": "PULSED", "value": "1000", "unit": "J/cm2", "wavelength_nm": "1064", "pulse_duration_s": "0.000000005",
              "repetition_rate_hz": "10", "pulse_count": 100, "test_protocol": "S_ON_1", "value_definition": "PEAK",
              "status": "DECLARED", "basis": FIXTURE}
    model = {"model_id": "win", "model": sequence([surface(1, "0", clear="10"), surface(2, "3", clear="10")], ["N-BK7"], edge="12"),
             "power_ratings": [rating]}
    document = with_models(document, [model], [nbk7()], None)
    reply = json_api.run_project(document)
    assert reply["ok"], reply
    exposures = json_api.run_results(reply["files"], {"kind": "exposures"})
    counts = [c for s in exposures["result"]["surfaces"] for r in s["ratings"] for c in r["conditions"] if c["condition"] == "pulse_count"]
    assert counts and all((c["state"], c.get("effect"), c.get("design")) == ("DIFFERS", "THRESHOLD_AT_MOST_RATED", "UNLIMITED")
                          for c in counts)


def test_the_power_ratings_export_states_the_rated_conditions_and_the_light_against_them():
    """``power-ratings`` rows carry the rating's stated conditions, the light's value for each and the thresholds the value
    establishes (here a stated peak value: both the value; a 100-on-1 count, so the light's pulses per site)."""
    from os_optics.results.tables import power_rating_rows

    fluence = _fluence_on_mirror()
    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(
        update={"power_ratings": (_pulsed_rating(0.5 * fluence, **S_ON_1),)})
    result = run(ProjectSnapshot(components=(_laser(total="UNLIMITED"), mirror)))
    (row,) = power_rating_rows(result, "M")
    assert row["basis"] == FIXTURE
    assert "pulse_count=100 pulses" in row["rated_conditions"] and "repetition_rate=10 Hz" in row["rated_conditions"]
    assert "value_definition=PEAK" in row["rated_conditions"]
    assert "pulse_count=UNLIMITED" in row["design_values"] and "pulse_duration=1e-08 s" in row["design_values"]
    assert row["rated_threshold_low"] == row["rated_threshold_high"] == pytest.approx(0.5 * fluence)


def test_the_power_ratings_export_reads_one_pulse_and_its_column_help_lists_every_condition():
    """A single-pulse rating reads ``pulse_count=1 pulse`` (not "1 pulses"), and the ``conditions`` column's help
    lists every condition a comparison records, value_definition included."""
    from os_optics.results.tables import columns_of, power_rating_rows

    mirror = B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-1.0, 1.0, 0.0)).model_copy(
        update={"power_ratings": (_pulsed_rating(1e3, count=1),)})
    result = run(ProjectSnapshot(components=(_laser(total=1), mirror)))
    (row,) = power_rating_rows(result, "M")
    assert "pulse_count=1 pulse" in row["rated_conditions"] and "pulse_count=1 pulse" in row["design_values"]
    assert not any("1 pulses" in text for text in (*row["rated_conditions"], *row["design_values"]))
    (conditions,) = [c for c in columns_of("power-ratings") if c["name"] == "conditions"]
    recorded = {c.condition for comparison in result.exposures.surfaces[0].ratings for c in comparison.conditions}
    assert recorded and all(name in conditions["description"] for name in recorded)


@pytest.mark.parametrize(("sku", "bench", "with_count"), [
    # the 515 nm cube: 0.36 J/cm2, 290 fs, 100 Hz, 10.72 um test beam, 100 pulses; a 2 mm, 1 W beam
    ("OS-OPT-BS-CUBE-A10-R47T47-400-700-NBK7-CA8", dict(wavelength=515.0, power=1.0, w0=1.0, x=-55.0, waist=50.0, tau=2.9e-13, rate=100.0),
     True),
    # the 1064 nm polarizing cube: 4.28 J/cm2, 5 ns, 10 Hz, no count; a 20 mW beam of 0.1 mm waist 100 mm before it
    ("OS-OPT-PBS-CUBE-A25P4-1064-ER3000-TP95RS99P5-UVFS-CA20P32",
     dict(wavelength=1064.0, power=0.02, w0=0.1, x=-100.0, waist=0.0, tau=5e-9, rate=10.0), False),
])
def test_the_exposure_bounds_do_not_depend_on_the_stated_total(real_catalog_store, sku, bench, with_count):
    """Catalog products of the seed under pulsed beams: whatever the source states as its total -- nothing, one
    pulse, 100 pulses or UNLIMITED -- the run follows the light left out as negligible equally deep, so every surface's
    exposure bounds are the same. The catalog ratings state no protocol for their 100 pulses (the cube) or no count at all
    (the polarizing cube), and no value definition: the count and the definition are not verifiable whatever the total,
    so no comparison decides."""
    from os_optics.local import run_optical_project

    along_x = [0.7071067811865476, 0.0, 0.7071067811865476, 0.0]

    def document(total) -> dict:
        pulse = {"repetition_rate_hz": bench["rate"], "pulse_duration_s": bench["tau"], "duration_definition": "FWHM"}
        if total is not None:
            pulse["total_pulses"] = total
        laser = {"instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "laser", "description": "Laser"},
                 "nominal_pose": {"position_mm": [bench["x"], 0.0, 0.0], "quaternion_wxyz": along_x},
                 "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": bench["wavelength"], "power_w": bench["power"],
                                  "waist_radius_mm": bench["w0"], "waist_position_mm": bench["waist"], "m2": 1.0,
                                  "polarization": {"kind": "LINEAR", "angle_deg": 0.0}, "pulse": pulse}}
        cube = {"instance_id": "P", "optical_role": "splitter", "product": {"optical_sku": sku, "catalog_version": catalog_version_of(REAL_CATALOG)},
                "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": along_x}}
        screen = {"observation_id": "S", "pose": {"position_mm": [60.0, 0.0, 0.0], "quaternion_wxyz": along_x},
                  "aperture": {"shape": "circle", "radius_mm": 30.0}}
        return {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM",
                "components": [laser, cube], "observations": [screen],
                "targets": [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "P", "maximum": 1}]}

    bounds, counts, rated, statuses = {}, {}, {}, {}
    for total in (None, 1, 100, "UNLIMITED"):
        exposures = run_optical_project(document(total)).result.exposures
        bounds[total] = [(s.surface_id, c.exposure_value, c.exposure_lower_bound, c.exposure_upper_bound)
                         for s in exposures.surfaces for c in s.ratings]
        counts[total] = {_state(_count(c)) for s in exposures.surfaces for c in s.ratings}
        rated[total] = {_count(c).rated for s in exposures.surfaces for c in s.ratings}
        statuses[total] = {(c.status, next(k.state for k in c.conditions if k.condition == "value_definition"))
                           for s in exposures.surfaces for c in s.ratings}
    assert bounds[None] and all(b == bounds[None] for b in bounds.values())
    assert all(found == ({100} if with_count else {None}) for found in rated.values())
    assert all(found == {("NOT_VERIFIABLE", "RATING_INFORMATION_MISSING")} for found in counts.values())
    assert all(found == {("CONDITIONS_NOT_VERIFIED", "NOT_VERIFIABLE")} for found in statuses.values())
