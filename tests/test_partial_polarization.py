"""Partially polarized and unpolarized sources: two mutually incoherent orthogonal Jones components, their Stokes sum and
degree of polarization on each route of a plane, and two-beam quantities only within one component.

Independent references, computed here without the Engine: Mueller matrices built from each element's Jones matrix
(``M = A (J kron conj J) A^-1``) applied to the source's Stokes vector ``S0 (1, P s)``; Malus's law for partially polarized
light; and the linearity of the decomposition -- a partially polarized source's results are those of two fully polarized
runs (its polarized state and the orthogonal one) weighted by ``(1 + P) / 2`` and ``(1 - P) / 2``.
"""

from __future__ import annotations

import copy
import json
import math
import re

import numpy as np
import pytest

from os_optics.local import lower_project, run_optical_project
from os_optics.local.lowering import source_polarization
from os_optics._vendor.shared_contracts.optical_design import PartialPolarizationV1, UnpolarizedV1
from os_optics.network.errors import NetworkExportError
from os_optics.network.export import build_linear_optical_network
from os_optics.optics.polarization.partial import component_shares, orthogonal_jones
from os_optics.optics.polarization.routing import PolarizationRouter
from os_optics.optics.rays.path_ids import BRANCH_SUFFIX_PATTERN, root_path_id
from os_optics.results import load_run

from catalog_store_helpers import REAL_CATALOG
from local_helpers import POWER_W, plane, project, source
from michelson_helpers import michelson
from test_polarization_extensions import chain, polarizer, retarder

#: Stokes from the coherency vector (Ex Ex*, Ex Ey*, Ey Ex*, Ey Ey*): S2 = 2 Re(conj(Ex) Ey), S3 = 2 Im(conj(Ex) Ey).
A = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]], dtype=complex)


def mueller(jones: np.ndarray) -> np.ndarray:
    return np.real(A @ np.kron(jones, np.conj(jones)) @ np.linalg.inv(A))


def rotation(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def polarizer_jones(theta_deg: float, transmittance: float = 0.8, extinction: float = 1000.0) -> np.ndarray:
    return rotation(theta_deg) @ np.diag([math.sqrt(transmittance), math.sqrt(transmittance / extinction)]) @ rotation(-theta_deg)


def linear_stokes(angle_deg: float) -> np.ndarray:
    a = math.radians(2.0 * angle_deg)
    return np.array([1.0, math.cos(a), math.sin(a), 0.0])


def partial(dop: float, angle_deg: float = 0.0) -> dict:
    laser = source()
    laser["source_model"]["polarization"] = {"kind": "PARTIAL", "degree_of_polarization": dop,
                                             "polarized": {"kind": "LINEAR", "angle_deg": angle_deg}}
    return laser


def unpolarized() -> dict:
    laser = source()
    laser["source_model"]["polarization"] = {"kind": "UNPOLARIZED"}
    return laser


def bare(src: dict) -> dict:
    """``src`` along +X and the plane OUT, nothing between."""
    return project([src], [plane("OUT", (100.0, 0.0, 0.0))])


def out_mixture(run_):
    (obs,) = [o for o in run_.result.observations if o.observation_id == "OUT"]
    (mixture,) = obs.polarization_mixtures
    return obs, mixture


# ------------------------------------------------------------------------------------------------- the decomposition


def test_the_orthogonal_component_is_orthogonal_and_the_shares_sum_to_one():
    for jones in (np.array([1.0, 0.0], dtype=complex), np.array([0.6, 0.8j]), np.array([math.sqrt(0.5), -math.sqrt(0.5) * 1j])):
        perp = orthogonal_jones(jones)
        assert abs(np.vdot(perp, jones)) < 1e-15 and abs(np.vdot(perp, perp) - 1.0) < 1e-15
    for dop in (0.0, 0.3, 0.7, 0.999999):
        w0, w1 = component_shares(dop)
        assert w0 + w1 == 1.0 and w0 - w1 == pytest.approx(dop, abs=1e-15)


def test_the_project_kinds_lower_to_the_polarized_part_and_a_degree():
    jones, dop = source_polarization(PartialPolarizationV1.model_validate(
        {"kind": "PARTIAL", "degree_of_polarization": 0.25, "polarized": {"kind": "LINEAR", "angle_deg": 90.0}}))
    assert (jones.ex, jones.ey, dop) == ((0.0, 0.0), (1.0, 0.0), 0.25)
    jones, dop = source_polarization(UnpolarizedV1(kind="UNPOLARIZED"))
    assert (jones.ex, jones.ey, dop) == ((1.0, 0.0), (0.0, 0.0), 0.0)
    request = lower_project(chain(partial(0.4), ("P", "polarizer", polarizer(None), 0.0)), catalog=None).request
    (spec,) = [c.source for c in request.project.components if c.source is not None]
    assert spec.degree_of_polarization == 0.4
    fully = lower_project(chain(source(), ("P", "polarizer", polarizer(None), 0.0)), catalog=None).request
    assert "degree_of_polarization" not in fully.model_dump_json()


def test_a_partial_source_of_degree_1_is_the_fully_polarized_source_bit_for_bit():
    fully = chain(source(polarization_deg=30.0), ("P", "polarizer", polarizer(None), 10.0))
    same = chain(partial(1.0, 30.0), ("P", "polarizer", polarizer(None), 10.0))
    a, b = lower_project(fully, catalog=None).request, lower_project(same, catalog=None).request
    assert a.model_dump_json() == b.model_dump_json()
    ra, rb = run_optical_project(fully, catalog=None), run_optical_project(same, catalog=None)
    assert ra.result.model_dump_json() == rb.result.model_dump_json()
    assert "polarization_mixtures" not in ra.result.model_dump_json() and "polarization_component" not in ra.result.model_dump_json()


# ------------------------------------------------------------------------------------------- Stokes and DOP on a plane


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 90.0, 135.0])
def test_unpolarized_light_through_a_polarizer_matches_the_mueller_reference(theta_deg):
    run_ = run_optical_project(chain(unpolarized(), ("P", "polarizer", polarizer(None), theta_deg)), catalog=None)
    obs, mixture = out_mixture(run_)
    want = POWER_W * mueller(polarizer_jones(theta_deg)) @ np.array([1.0, 0.0, 0.0, 0.0])
    assert mixture.stokes_W == pytest.approx(tuple(want), rel=1e-12, abs=1e-18)
    assert mixture.degree_of_polarization == pytest.approx((1.0 - 1e-3) / (1.0 + 1e-3), rel=1e-12)
    assert mixture.polarization_state.azimuth_in_plane_deg == pytest.approx(theta_deg % 180.0, abs=1e-9)
    assert mixture.quantity_state.value == "KNOWN" and mixture.path_ids == ("SRC#0", "SRC#0:p1")
    assert obs.total_power_W == pytest.approx(want[0], rel=1e-12)
    assert [a.polarization_component for a in obs.arrivals] == [0, 1]


@pytest.mark.parametrize("dop, psi_deg, theta_deg", [(0.6, 20.0, 0.0), (0.6, 20.0, 65.0), (0.25, -40.0, 50.0), (0.0, 10.0, 10.0)])
def test_partially_polarized_light_follows_malus_law(dop, psi_deg, theta_deg):
    run_ = run_optical_project(chain(partial(dop, psi_deg), ("P", "polarizer", polarizer(None), theta_deg)), catalog=None)
    obs, mixture = out_mixture(run_)
    stokes_in = dop * linear_stokes(psi_deg)
    stokes_in[0] = 1.0
    want = POWER_W * mueller(polarizer_jones(theta_deg)) @ stokes_in
    # Malus for partially polarized light: the transmitted power (T along the axis, T / ER across it)
    c = math.cos(math.radians(2.0 * (theta_deg - psi_deg)))
    malus = POWER_W / 2.0 * (0.8 * (1.0 + dop * c) + 0.8e-3 * (1.0 - dop * c))
    assert want[0] == pytest.approx(malus, rel=1e-12)
    assert mixture.stokes_W == pytest.approx(tuple(want), rel=1e-12, abs=1e-18)
    assert obs.total_power_W == pytest.approx(malus, rel=1e-12)
    assert mixture.degree_of_polarization == pytest.approx(math.sqrt(want[1] ** 2 + want[2] ** 2 + want[3] ** 2) / want[0], rel=1e-12)


def test_a_half_wave_plate_turns_the_polarized_part_and_keeps_the_degree():
    dop, psi_deg, axis_deg = 0.5, 10.0, 22.5
    run_ = run_optical_project(chain(partial(dop, psi_deg), ("W", "waveplate", retarder(None, waves="0.5", transmittance="0.5"), axis_deg)),
                               catalog=None)
    _, mixture = out_mixture(run_)
    hwp = rotation(axis_deg) @ np.diag([1.0, -1.0]) @ rotation(-axis_deg) * math.sqrt(0.5)
    stokes_in = dop * linear_stokes(psi_deg)
    stokes_in[0] = 1.0
    want = POWER_W * mueller(hwp) @ stokes_in
    assert mixture.stokes_W == pytest.approx(tuple(want), rel=1e-12, abs=1e-18)
    assert mixture.degree_of_polarization == pytest.approx(dop, rel=1e-12)
    assert mixture.polarization_state.azimuth_in_plane_deg == pytest.approx(2.0 * axis_deg - psi_deg, abs=1e-9)


def test_a_quarter_wave_plate_keeps_the_degree_and_makes_the_polarized_part_circular():
    run_ = run_optical_project(chain(partial(0.3, 0.0), ("W", "waveplate", retarder(None, waves="0.25", transmittance="1"), 45.0)),
                               catalog=None)
    _, mixture = out_mixture(run_)
    assert mixture.degree_of_polarization == pytest.approx(0.3, rel=1e-12)
    s0, s1, s2, s3 = mixture.stokes_W
    assert abs(s3) / s0 == pytest.approx(0.3, rel=1e-12) and abs(s1) < 1e-15 and abs(s2) < 1e-15
    assert mixture.polarization_state.handedness in ("RIGHT", "LEFT")
    assert abs(mixture.polarization_state.ellipticity_deg) == pytest.approx(45.0, abs=1e-9)


def test_unpolarized_light_in_free_space_has_no_ellipse():
    run_ = run_optical_project(bare(unpolarized()), catalog=None)
    _, mixture = out_mixture(run_)
    assert mixture.degree_of_polarization == 0.0 and mixture.polarization_state is None
    assert mixture.stokes_W == (POWER_W, 0.0, 0.0, 0.0)


# ------------------------------------------------------------------------------------------------------- targets


def _targets(document: dict, *targets: dict) -> dict:
    out = copy.deepcopy(document)
    out["targets"] = [{"target_id": f"t{k}", "observation_id": "OUT", "minimum": -1e9, **t} for k, t in enumerate(targets)]
    return out


def test_polarization_metrics_read_the_light_on_the_route():
    document = _targets(chain(partial(0.6, 20.0), ("P", "polarizer", polarizer(None), 65.0)),
                        {"metric": "degree_of_polarization", "unit": "1"}, {"metric": "stokes_s1", "unit": "1"},
                        {"metric": "polarization_azimuth_deg", "unit": "deg", "parameters": [{"key": "path_id", "value": "SRC#0:p1"}]},
                        {"metric": "power_W", "unit": "W"}, {"metric": "beam_radius_mm", "unit": "mm"})
    run_ = run_optical_project(document, catalog=None)
    _, mixture = out_mixture(run_)
    values = {t.target_id: t for t in run_.result.targets}
    assert values["t0"].value == mixture.degree_of_polarization
    assert values["t1"].value == mixture.stokes_W[1] / mixture.stokes_W[0]
    # the polarized part's axis (Mueller reference): near the polarizer's axis, turned by the light its blocking axis leaks
    stokes_in = 0.6 * linear_stokes(20.0)
    stokes_in[0] = 1.0
    want = mueller(polarizer_jones(65.0)) @ stokes_in
    assert values["t2"].value == pytest.approx(math.degrees(math.atan2(want[2], want[1])) / 2.0 % 180.0, abs=1e-9)
    assert values["t3"].value == pytest.approx(mixture.stokes_W[0], rel=1e-12)
    # a geometric quantity both components share is the route's
    assert values["t4"].value is not None and values["t4"].status.value == "PASS"


def test_the_degree_of_polarization_of_a_fully_polarized_beam_is_1():
    run_ = run_optical_project(_targets(bare(source()), {"metric": "degree_of_polarization", "unit": "1"}), catalog=None)
    assert run_.result.targets[0].value == 1.0


def test_an_unpolarized_route_has_no_ellipse_metric():
    run_ = run_optical_project(_targets(bare(unpolarized()), {"metric": "polarization_ellipticity_deg", "unit": "deg"}), catalog=None)
    (target,) = run_.result.targets
    assert target.value is None and target.status.value == "NOT_APPLICABLE"
    assert target.issue_codes == ("PARTIAL_POLARIZATION_COMPONENTS",)


# ----------------------------------------------------------------------------------- interferometer: linearity, pairs


@pytest.fixture(scope="module")
def catalog() -> dict:
    return json.loads(REAL_CATALOG.read_text(encoding="utf-8"))


def _with_polarization(document: dict, polarization: dict) -> dict:
    out = copy.deepcopy(document)
    out["components"][0]["source_model"]["polarization"] = polarization
    return out


def _arrivals(run_) -> dict:
    (obs,) = [o for o in run_.result.observations if o.observation_id == "out"]
    return {a.beam.path_id: a for a in obs.arrivals}


def test_a_partial_interferometer_is_its_two_fully_polarized_runs_weighted(catalog):
    dop = 0.4
    w0, w1 = component_shares(dop)
    base = michelson(catalog, targets=[])
    s_run = run_optical_project(base, catalog=catalog)  # the polarized part: s (local +Y)
    p_run = run_optical_project(_with_polarization(base, {"kind": "LINEAR", "angle_deg": 0.0}), catalog=catalog)
    mixed = run_optical_project(_with_polarization(base, {"kind": "PARTIAL", "degree_of_polarization": dop,
                                                          "polarized": {"kind": "LINEAR", "angle_deg": 90.0}}), catalog=catalog)
    s, p, m = _arrivals(s_run), _arrivals(p_run), _arrivals(mixed)
    assert set(m) == set(s) | {pid.replace("SRC#0", "SRC#0:p1", 1) for pid in p}
    # Real catalog plate substrates do not state bulk loss. Do not impute their old lossless
    # output power: the two source components instead scale the *upper bounds* exactly.
    for pid, a in s.items():
        twin = m[pid]
        assert twin.beam.power_W is a.beam.power_W is None
        assert twin.power_upper_bound_W == pytest.approx(w0 * a.power_upper_bound_W, rel=1e-12)
        assert twin.beam.direction == pytest.approx(a.beam.direction, abs=1e-12)
        assert twin.polarization_component == 0 and twin.coherence_group == "source:SRC"
    for pid, a in p.items():
        twin = m[pid.replace("SRC#0", "SRC#0:p1", 1)]
        assert twin.beam.power_W is a.beam.power_W is None
        assert twin.power_upper_bound_W == pytest.approx(w1 * a.power_upper_bound_W, rel=1e-12)
        assert twin.beam.direction == pytest.approx(a.beam.direction, abs=1e-12)
        assert twin.polarization_component == 1 and twin.coherence_group == "source:SRC:p1"
    (obs,) = [o for o in mixed.result.observations if o.observation_id == "out"]
    assert obs.quantity_state.value == "UNKNOWN" and obs.total_power_W is None
    assert len(obs.polarization_mixtures) == 2
    for mixture in obs.polarization_mixtures:
        assert mixture.quantity_state.value == "UNKNOWN"
        assert mixture.stokes_W is None and mixture.degree_of_polarization is None
    # No unsupported interference or visibility is asserted from unknown amplitudes.
    assert s_run.result.interference == p_run.result.interference == mixed.result.interference == ()


def test_no_two_beam_quantity_is_formed_between_the_components(catalog):
    base = _with_polarization(michelson(catalog, targets=[]), {"kind": "UNPOLARIZED"})
    document = copy.deepcopy(base)
    document["targets"] = [
        {"target_id": "cross", "metric": "fringe_visibility", "observation_id": "out", "minimum": 0.0, "unit": "1",
         "parameters": [{"key": "path_id_a", "value": "SRC#0.0"}, {"key": "path_id_b", "value": "SRC#0:p1.1"}]},
        {"target_id": "within", "metric": "fringe_visibility", "observation_id": "out", "minimum": 0.0, "unit": "1",
         "parameters": [{"key": "path_id_a", "value": "SRC#0:p1.0"}, {"key": "path_id_b", "value": "SRC#0:p1.1"}]},
    ]
    run_ = run_optical_project(document, catalog=catalog)
    targets = {t.target_id: t for t in run_.result.targets}
    # The public substrate has no stated bulk loss: both pair quantities are UNKNOWN,
    # not an invented visibility for either a cross-component or within-component pair.
    assert run_.result.interference == ()
    for target in targets.values():
        assert target.value is None and target.status.value == "UNKNOWN"
        assert target.issue_codes == ("PHASE_UNDEFINED",)
    (obs,) = [o for o in run_.result.observations if o.observation_id == "out"]
    assert obs.quantity_state.value == "UNKNOWN"
    assert {a.beam.path_id for a in obs.arrivals} == {"SRC#0.0", "SRC#0.1", "SRC#0:p1.0", "SRC#0:p1.1"}
    assert all(a.beam.power_W is None and a.power_upper_bound_W is not None for a in obs.arrivals)


# ------------------------------------------------------------------------------------------- ids, routing, network


def test_component_path_ids_follow_the_path_id_grammar():
    assert root_path_id("SRC", 3, 1) == "SRC#3:p1" and root_path_id("SRC", 3, 0) == root_path_id("SRC", 3) == "SRC#3"
    for pid in ("SRC#0:p1", "SRC#0:p1.0.1", "SRC#0:p1~" + "0" * 32 + ".2", "SRC#0.1"):
        assert re.fullmatch(r"SRC#\d+" + BRANCH_SUFFIX_PATTERN, pid)
    run_ = run_optical_project(chain(unpolarized(), ("P", "polarizer", polarizer(None), 0.0)), catalog=None)
    ids = [p.path_id for p in run_.result.paths]
    assert ids == ["SRC#0", "SRC#0:p1"]
    from pydantic import TypeAdapter
    from os_optics._vendor.shared_contracts.optical_view.scene import BeamPathId

    for pid in ids:
        TypeAdapter(BeamPathId).validate_python(pid)
    # one ledger line, both components booked to it: it closes
    (line,) = run_.result.power_budget.lines
    assert line.input_power_W == POWER_W and line.source_id == "SRC"


def test_the_monitor_lists_the_mixtures():
    run_ = run_optical_project(chain(unpolarized(), ("P", "polarizer", polarizer(None), 0.0)), catalog=None)
    monitor = load_run(run_).monitor("OUT")
    assert monitor["polarization_mixtures"][0]["degree_of_polarization"] == pytest.approx(0.998001998, rel=1e-9)
    plain = load_run(run_optical_project(bare(source()), catalog=None)).monitor("OUT")
    assert "polarization_mixtures" not in plain


def test_assembly_routing_takes_partially_polarized_light_as_no_single_state():
    lowered = lower_project(chain(partial(0.5), ("P", "polarizer", polarizer(None), 0.0)), catalog=None)
    router = PolarizationRouter.for_lowered(lowered)
    assert router.source_state("SRC") is None
    fully = PolarizationRouter.for_lowered(lower_project(chain(source(), ("P", "polarizer", polarizer(None), 0.0)), catalog=None))
    assert fully.source_state("SRC") is not None


def test_the_network_export_refuses_partially_polarized_light():
    lowered = lower_project(chain(unpolarized(), ("P", "polarizer", polarizer(None), 0.0)), catalog=None)
    with pytest.raises(NetworkExportError) as caught:
        build_linear_optical_network(lowered.request.project, engine_input_hash="sha256:" + "0" * 64)
    (problem,) = caught.value.problems
    assert problem.code.value == "OUT_OF_MODEL_SCOPE" and "partially polarized" in problem.message
