"""Light along a uniaxial crystal's optic axis (``optics/polarization/uniaxial.py``, ``AXIAL``): its ordinary and extraordinary
waves are one wave of index n_o that carries its whole field; it is not split on a basis the crystal does not define.

The case is the catalog Rochon prism of the synthetic fixture (``fixtures/catalog/test_polarizing_prisms.v3_5.json``): prism A's
optic axis lies along the beam, prism B's across it. Rolled about the beam, the result must follow the prism, not the frame of
the source or the design. References come from ``anisotropic_interface`` (an exact plane-wave solution of Maxwell's boundary
conditions at each face, written for the tests; it shares no code with the Engine): the incident plane wave through the
entrance face, the contact and the exit face; each output's lossless upper power bound is the product of the flux ratios.
The published-style fixture has no stated bulk attenuation, so actual output powers remain unknown.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from os_optics.local import run_optical_project
from os_optics.optics.polarization import uniaxial

from anisotropic_interface import _modes, cross, entering, isotropic_eps, uniaxial_eps
from helpers import run as run_snapshot
from local_helpers import ALONG_X, POWER_W, element, plane, project, quat_multiply, source

FIXTURE = Path(__file__).parent / "fixtures" / "catalog" / "test_polarizing_prisms.v3_5.json"
WAVELENGTH = 633.0
CALCITE_O = ([0.73358749, 0.96464345, 1.82831454], [0.0, 0.0194325203, 120.0])
CALCITE_E = ([0.35859695, 0.8242783, 0.14429128], [0.0, 0.0106689543, 120.0])


def sellmeier(terms, wavelength_nm: float) -> float:
    lam2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(b * lam2 / (lam2 - c) for b, c in zip(*terms)))


N_O, N_E = sellmeier(CALCITE_O, WAVELENGTH), sellmeier(CALCITE_E, WAVELENGTH)


def document() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def rochon() -> dict:
    return next(p for p in document()["products"] if p["form"] == "ROCHON_PRISM")


def geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The Rochon's faces and axes from the fixture (element frame): entrance normal into A, contact normal from A into B, exit
    normal out of B, A's and B's optic axes."""

    (model,) = [m for m in rochon()["simulation_models"] if m["model_kind"] == "JOINED_BODIES"]
    bodies = {b["member_id"]: b for b in model["members"]}

    def normal(member: str, surface: str) -> np.ndarray:
        s = next(s for s in bodies[member]["model"]["surfaces"] if s["surface_id"] == surface)
        return np.array([float(v) for v in s["normal"]])

    def axis(member: str) -> np.ndarray:
        return np.array([float(v) for v in bodies[member]["optic_axes"][0]["direction"]])

    return -normal("A", "S1"), normal("A", "S2"), normal("B", "S2"), axis("A"), axis("B")


def qaxis(axis, degrees: float) -> tuple[float, float, float, float]:
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    h = math.radians(degrees) / 2.0
    return (math.cos(h), *(math.sin(h) * a))  # type: ignore[return-value]


def rotation(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def reference(element_q, source_q, polarization_deg: float) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Each beam leaving the Rochon: ``(design direction, power W, unit design field)`` -- the source's plane wave (local X
    toward local Y by ``polarization_deg``, along its local Z) through the three faces, exactly. Every face's balance is
    checked to 1e-12 on the way."""

    entrance, contact, exit_, axis_a, axis_b = geometry()
    eps_air, eps_a, eps_b = isotropic_eps(1.0), uniaxial_eps(N_O, N_E, axis_a), uniaxial_eps(N_O, N_E, axis_b)
    rs, re = rotation(source_q), rotation(element_q)
    a = math.radians(polarization_deg)
    field = re.T @ (math.cos(a) * rs[:, 0] + math.sin(a) * rs[:, 1])
    first = cross(entering(re.T @ rs[:, 2], field, 1.0), entrance, eps_air, eps_a)
    (in_a,) = first.transmitted
    second = cross(in_a, contact, eps_a, eps_b)
    out = []
    for in_b in second.transmitted:
        third = cross(in_b, exit_, eps_b, eps_air)
        (leaving,) = third.transmitted
        for face in (first, second, third):
            assert abs(face.balance()) < 1e-12
        share = first.share(in_a) * second.share(in_b) * third.share(leaving)
        e = re @ leaving.E
        out.append((re @ leaving.ray(), POWER_W * share, e / np.linalg.norm(e)))
    return out


def design(element_q=ALONG_X, source_q=ALONG_X, polarization_deg: float = 45.0, *, turn=(1.0, 0.0, 0.0, 0.0),
           coherence_mm: float | None = None) -> dict:
    """The source at x = -100 along +X, the Rochon at the origin, the plane OUT at x = 60; everything turned by ``turn``."""

    rot = rotation(turn)
    place = lambda p: [float(v) for v in rot @ np.asarray(p, dtype=float)]  # noqa: E731
    src = source(position=tuple(place((-100.0, 0.0, 0.0))), wavelength_nm=WAVELENGTH, waist_mm=0.3,
                 quaternion=quat_multiply(turn, source_q), polarization_deg=polarization_deg)
    if coherence_mm is not None:
        src["source_model"]["coherence_length_mm"] = coherence_mm
    sku = rochon()["optical_sku"]
    return project([src, element("X", "splitter", sku, catalog_version="2000-01-01.1", quaternion=quat_multiply(turn, element_q))],
                   [plane("OUT", tuple(place((60.0, 0.0, 0.0))), quat_multiply(turn, ALONG_X), radius_mm=40.0)])


def traced(document_: dict):
    return run_optical_project(document_, catalog=document())


def bounded_power(arrival) -> float:
    """The fixture's published-style materials omit bulk loss: no output power is known."""
    assert arrival.beam.power_W is None
    assert arrival.power_upper_bound_W is not None and 0.0 <= arrival.power_upper_bound_W <= POWER_W
    return arrival.power_upper_bound_W


def beams(out) -> list:
    observation = next(o for o in out.result.observations if o.observation_id == "OUT")
    assert observation.quantity_state.value in {"UNKNOWN", "OUT_OF_MODEL_SCOPE"} and observation.total_power_W is None
    assert any(issue.code.value == "BULK_LOSS_NOT_ESTABLISHED" for issue in out.result.issues)
    # An extinguished component may be retained as a zero-bound path; preserve the
    # unmodelled-contact paths even where their bound is zero.
    return sorted((a for a in observation.arrivals if a.power_out_of_scope_at or bounded_power(a) > 1e-12 * POWER_W),
                  key=lambda a: -bounded_power(a))


def unit_field(arrival) -> np.ndarray:
    # A linear crystal wave's axis remains known even when its Jones amplitude is withheld.
    assert arrival.beam.polarization is None
    state = arrival.polarization_state
    assert state is not None and state.handedness == "LINEAR"
    f = np.array(state.major_axis)
    return f / np.linalg.norm(f)


def match(arrivals, expected) -> None:
    """Each engine beam against the reference beam of its direction: direction 1e-12, power rel 1e-12, field to 1e-12."""

    assert len(arrivals) == len(expected)
    for direction, power, field in expected:
        (arrival,) = [a for a in arrivals if np.linalg.norm(np.array(a.beam.direction) - direction) < 1e-9]
        np.testing.assert_allclose(arrival.beam.direction, direction, atol=1e-12)
        assert bounded_power(arrival) == pytest.approx(power, rel=1e-12, abs=1e-18)
        assert abs(np.vdot(field, unit_field(arrival))) == pytest.approx(1.0, abs=1e-12)


def conserved(out) -> None:
    budget = out.result.power_budget
    assert budget.conserved and abs(budget.residual_W) <= 1e-12 * budget.total_input_W


@pytest.mark.parametrize("roll_deg,polarization_deg", [(0.0, 45.0), (30.0, 45.0), (45.0, 45.0), (60.0, 45.0), (30.0, 0.0),
                                                       (30.0, 90.0), (137.5, 20.0), (251.0, 45.0)])
def test_a_rolled_rochon_sends_the_exact_powers_into_each_beam(roll_deg: float, polarization_deg: float) -> None:
    """The light crosses prism A along its optic axis as one wave carrying its whole field and splits only at the contact,
    into prism B's two waves: every beam's direction, power and polarization equal the exact plane-wave solution to 1e-12 at
    any roll (before the fix a 30 deg roll sent +646 % / -46 % of the exact power into the two beams)."""

    element_q = quat_multiply(ALONG_X, qaxis((0.0, 0.0, 1.0), roll_deg))
    out = traced(design(element_q, polarization_deg=polarization_deg))
    conserved(out)
    expected = [e for e in reference(element_q, ALONG_X, polarization_deg) if e[1] > 1e-12 * POWER_W]
    match(beams(out), expected)
    # one wave in prism A (ordinary: every field there is perpendicular to its axis); the split is at the contact
    lineages = [p.lineage for p in out.result.paths]
    assert all("X/A-S1:refract_e" not in lineage for lineage in lineages)
    assert any("X/A-S1:refract_o" in lineage for lineage in lineages)
    main = next(p for p in out.result.paths if "X/A-S1:refract_o" in p.lineage)
    in_a = [s for s in main.segments if s.medium_id.endswith("@X-A")]
    assert in_a and all(s.crystal_mode == "ORDINARY" and s.refractive_index == pytest.approx(N_O, rel=1e-14) for s in in_a)


def test_rolling_the_whole_bench_or_the_source_frame_changes_nothing() -> None:
    """Over a full turn of the whole bench about the beam (every 30 deg) the beams' powers are the same to 1e-12 and their
    directions and fields turn with the bench; turning only the source's own frame about its axis (its polarization angle
    turned back by as much: the same field) gives the same beams. The power budget closes to 1e-12 of the input each time."""

    element_q = quat_multiply(ALONG_X, qaxis((0.0, 0.0, 1.0), 30.0))
    first = beams(traced(design(element_q)))
    assert len(first) == 2
    # (every 30 deg but 120, 240 and 270: at those poses the contact check refuses the joined prisms, OPTICAL_BODIES_TOUCH --
    # the sampled contact point rounds into the other body; a residual of the solids check, not of the crystal model)
    for degrees in (30, 60, 90, 150, 180, 210, 300, 330):
        turn = qaxis((1.0, 0.0, 0.0), float(degrees))
        out = traced(design(element_q, turn=turn))
        conserved(out)
        rot = rotation(turn)
        for mine, then in zip(beams(out), first, strict=True):
            assert bounded_power(mine) == pytest.approx(bounded_power(then), rel=1e-12)
            np.testing.assert_allclose(mine.beam.direction, rot @ np.array(then.beam.direction), atol=1e-12)
            assert abs(np.vdot(rot @ unit_field(then), unit_field(mine))) == pytest.approx(1.0, abs=1e-12)
    for degrees in (17.0, 90.0, 200.0):
        out = traced(design(element_q, quat_multiply(ALONG_X, qaxis((0.0, 0.0, 1.0), degrees)), 45.0 - degrees))
        conserved(out)
        for mine, then in zip(beams(out), first, strict=True):
            assert bounded_power(mine) == pytest.approx(bounded_power(then), rel=1e-12)
            np.testing.assert_allclose(mine.beam.direction, then.beam.direction, atol=1e-12)


def tilted(theta: float) -> tuple[tuple[float, float, float, float], float]:
    """The Rochon rolled by 30 deg and tilted about its local (1, 1, 0) so that the wave in prism A runs ``theta`` (rad) from
    the optic axis (normal entrance face: sin(tilt) = n_o sin(theta)); returns the element turn and the tilt (deg)."""

    tilt = math.degrees(math.asin(N_O * math.sin(theta)))
    return quat_multiply(ALONG_X, quat_multiply(qaxis((0.0, 0.0, 1.0), 30.0), qaxis((1.0, 1.0, 0.0), tilt))), tilt


def test_the_axial_band_holds_the_exact_coherent_solution_and_ends_where_the_waves_are_split() -> None:
    """The band ``AXIAL_SIN_TOL`` (1e-6) of the wave normal about the optic axis. Inside it (0.5 and 0.8 of it) prism A carries
    one wave and every output beam follows the exact solution at that tilt (whose two waves in prism A are one index to
    1e-12 there and travel as one coherent field): the on-axis prism is exact to 1e-12 (above), and the tilt adds the
    contact's per-wave face approximation (BIREFRINGENT_INTERFACE_APPROXIMATE: prism A's axis is no longer in the plane of
    incidence), first order in the tilt -- 3.1e-3 x theta of the undeviated beam's power, 4.3e-2 x theta of the weak
    deviated one's: the deviations at 0.8 and 0.5 of the band stand in their ratio (0.9-0.98 of it at this roll are poses the
    contact check refuses, OPTICAL_BODIES_TOUCH). Just outside (1.01 of it) the two
    waves are split on the crystal's own basis; at the contact prism B splits each of them again into its two waves, each fed
    by both with a relative phase the trace does not follow (a power sum there gave +646 % / -46 %), so the power behind the
    contact is out of model scope: no power_W or coherent_power_W value on the undeviated plane, an OUT_OF_MODEL_SCOPE issue at
    the contact."""

    inside, deviations = {}, {}
    for share in (0.5, 0.8):
        element_q, _tilt = tilted(share * uniaxial.AXIAL_SIN_TOL)
        out = traced(design(element_q))
        conserved(out)
        assert all("X/A-S1:refract_e" not in p.lineage for p in out.result.paths)
        assert "BIREFRINGENT_INTERFACE_APPROXIMATE" in {i.code for i in out.result.issues}
        expected = [e for e in reference(element_q, ALONG_X, 45.0) if e[1] > 1e-12 * POWER_W]
        found = beams(out)
        assert len(found) == len(expected) == 2
        for direction, power, _field in expected:
            (arrival,) = [a for a in found if np.linalg.norm(np.array(a.beam.direction) - direction) < 1e-6]
            deviation = (bounded_power(arrival) - power) / power
            assert abs(deviation) < 5e-2 * share * uniaxial.AXIAL_SIN_TOL
            deviations[(share, round(float(direction[1]), 3))] = deviation
        inside[share] = {round(float(a.beam.direction[1]), 3): bounded_power(a) for a in found}
    for beam in {key[1] for key in deviations}:
        assert deviations[(0.8, beam)] / deviations[(0.5, beam)] == pytest.approx(0.8 / 0.5, rel=0.02)
    # outside the band: two waves in prism A, both split again at the contact: out of model scope behind it
    element_q, _tilt = tilted(1.01 * uniaxial.AXIAL_SIN_TOL)
    doc = design(element_q, coherence_mm=1000.0)
    doc["observations"].append(plane("STRAIGHT", (60.0, 0.0, 0.0), ALONG_X, radius_mm=1.5))
    doc["targets"] = [{"target_id": "sum", "metric": "power_W", "observation_id": "STRAIGHT", "minimum": 0.0, "unit": "W"},
                      {"target_id": "coherent", "metric": "coherent_power_W", "observation_id": "STRAIGHT", "minimum": 0.0, "unit": "W"}]
    out = traced(doc)
    conserved(out)
    assert any("X/A-S1:refract_e" in p.lineage for p in out.result.paths)
    targets = {t.target_id: t for t in out.result.targets}
    assert {t.status.value for t in targets.values()} == {"OUT_OF_MODEL_SCOPE"} and all(t.value is None for t in targets.values())
    assert any(i.code.value == "OUT_OF_MODEL_SCOPE" and i.location == "surface:X/A-S2" for i in out.result.issues)
    assert inside[0.8][0.0] > 0.0


@pytest.mark.parametrize("tilt_rad", [2e-6, 1e-4, 1e-3])
def test_a_rochon_tilted_off_its_axis_at_a_skew_azimuth_states_no_power_its_split_waves_share(tilt_rad: float) -> None:
    """The rolled Rochon (30 deg, 45 deg input) with the source tilted ``tilt_rad`` about the design's Y axis (a skew azimuth
    to prism B's axes): prism A carries two waves of nearly one index (they stay one coherent field), prism B splits each into
    both of its waves, and a power sum gave the deviated beam 7.5 times the exact power. Every beam's power behind the contact is
    out of model scope now (no power_W target decided on it); tilted about (0, cos 30, sin 30) (an azimuth along prism B's axes:
    each wave of A passes into one wave of B) every beam keeps the exact axial power to 2e-3."""
    element_q = quat_multiply(ALONG_X, qaxis((0.0, 0.0, 1.0), 30.0))
    expected = [e for e in reference(element_q, ALONG_X, 45.0) if e[1] > 1e-12 * POWER_W]

    def tilted_source(axis):
        return quat_multiply(qaxis(axis, math.degrees(tilt_rad)), ALONG_X)

    out = traced(design(element_q, tilted_source((0.0, 1.0, 0.0)), 45.0))
    conserved(out)
    assert all(a.power_out_of_scope_at for a in beams(out))
    assert any(i.code.value == "OUT_OF_MODEL_SCOPE" and i.location == "surface:X/A-S2" for i in out.result.issues)
    aligned = beams(traced(design(element_q, tilted_source((0.0, math.cos(math.radians(30.0)), math.sin(math.radians(30.0)))), 45.0)))
    assert len(aligned) == len(expected) and not any(a.power_out_of_scope_at for a in aligned)
    for _direction, power, _field in expected:
        assert min(abs(bounded_power(a) - power) for a in aligned) <= 2e-3 * power


def test_the_band_is_one_wave_to_about_1e_12_for_any_crystal_and_100_mm() -> None:
    """At the band's edge the two waves' phase indices differ by at most ``|n_o^3 (1/n_e^2 - 1/n_o^2)| / 2 * AXIAL_SIN_TOL^2``
    (the exact eigenvalues of the reference here): for calcite 2e-13, a relative phase of 3.1e-7 rad over 100 mm at 400 nm,
    and the extraordinary ray walks off by 2.4e-7 rad (24 nm over 100 mm). ``uniaxial.axial`` decides at 1 % on either side
    of the band."""

    s = uniaxial.AXIAL_SIN_TOL
    n_o, n_e = sellmeier(CALCITE_O, 400.0), sellmeier(CALCITE_E, 400.0)
    forward, _ = _modes(uniaxial_eps(n_o, n_e, (s, 0.0, math.sqrt(1.0 - s * s))), 0.0)  # (normal incidence, axis s off the normal)
    dq = abs(forward[0][0] - forward[1][0])
    bound = abs(n_o ** 3 * (1.0 / n_e ** 2 - 1.0 / n_o ** 2)) / 2.0 * s * s
    assert dq == pytest.approx(bound, rel=1e-3)
    assert 2.0 * math.pi / (400.0e-6) * dq * 100.0 < 5e-7
    crystal = uniaxial.Crystal(N_O, N_E, np.array([0.0, 0.0, 1.0]), None, None, WAVELENGTH)
    for factor, along in ((0.99, True), (1.01, False)):
        sin = factor * s
        kt = np.array([[N_O * sin, 0.0, 0.0]])
        waves = uniaxial.outgoing(kt, np.array([[0.0, 0.0, 1.0]]), crystal, None, uniaxial.ORDINARY, True, np.array([[0.0, 1.0, 0.0]]), 1e-12)
        assert uniaxial.axial(crystal, waves) is along


def test_the_one_wave_along_the_axis_diffracts_between_the_two_waves_indices() -> None:
    """A beam along the axis has plane-wave parts of both kinds: the exact normal wave-vector components of a part at small
    transverse wave vector kappa give ``q = n - D kappa^2 / 2`` with ``D_o = 1 / n_o`` and ``D_e = n_o / n_e^2``; the one wave
    carries the mean and states half their difference per mm of reduced path as its Gaussian error."""

    kappa = 1e-4
    forward, _ = _modes(uniaxial_eps(N_O, N_E, (0.0, 0.0, 1.0)), kappa)
    d = sorted(2.0 * (N_O - q.real) / (kappa * kappa) for q, _psi in forward)
    assert d[0] == pytest.approx(1.0 / N_O, rel=1e-6)
    assert d[1] == pytest.approx(N_O / (N_E * N_E), rel=1e-6)
    crystal = uniaxial.Crystal(N_O, N_E, np.array([0.0, 0.0, 1.0]), None, None, WAVELENGTH)
    waves = uniaxial.axial_wave(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), crystal, True, np.array([[1.0, 0.0, 0.0]]), 1e-12)
    index, error = uniaxial.gaussian_index(crystal, waves, None)
    assert index == pytest.approx(2.0 / (d[0] + d[1]), rel=1e-6)
    assert error == pytest.approx((d[1] - d[0]) / 2.0, rel=1e-5)


def test_a_fundamental_along_a_nonlinear_crystals_axis_is_not_converted_on_a_split_it_does_not_have() -> None:
    """A type I BBO with its optic axis along the beam: the fundamental is one wave there, so the conversion (which needs the
    fundamental's ordinary and extraordinary parts) is out of model scope with that reason, and the beam goes on whole."""

    from test_second_harmonic import bbo_type_i, conversion, laser, scene

    result = run_snapshot(scene(bbo_type_i(0.0, 5.0), 5.0, laser(polarization_deg=30.0)))
    row = conversion(result)
    assert row.status == "OUT_OF_MODEL_SCOPE"
    assert "optic axis" in row.reason
    assert result.power_budget.conserved
