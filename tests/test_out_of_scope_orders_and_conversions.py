"""A result that reads the power of a beam whose power is out of model scope is never stated as a value.

A beam's power is out of model scope behind a surface part of it meets beside its chief ray (here a beam dump 1.1 mm beside a
0.5 mm beam), or a variable attenuator whose density varies across it (``ArrivalResult.power_out_of_scope_at``): its power is
its chief ray's, not the light's. Before, a crystal fed by such a beam converted the chief ray's power (``shg_power_W`` and
``shg_conversion_efficiency`` PASS on the full 1 W), a grating or an acousto-optic deflector stated each order's power
``KNOWN``, the partially polarized light on a plane its Stokes parameters ``KNOWN``, the power budget the plane's observed
power, the plane its upper bound, the light a surface scattered out of such a beam its emitted power as any other, and the
Designer's beam table and the 3-D view the chief ray's power.

Independent references (no Engine code): the share of a Gaussian beam a disc beside it takes (the 2-D Gaussian integral over the
disc, Riemann on a fine grid), the share a linear density gradient passes (the Gaussian profile against ``10^-OD(x)``: a
log-normal mean, closed form), the second harmonic of a collimated Gaussian beam (``P2 ~ P1^2``, the coupled-wave closed form of
``test_second_harmonic``), grating order powers ``P x efficiency`` of an absolute table and the Stokes vector of partially
polarized light after a mirror.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import (CircularAperture, ConstantCoating, DiffractionGratingInteraction, EngineeringComponentSnapshot,
                                      GratingOrderTable, PlaneSurface, ProjectSnapshot, RefractInteraction, Target)
from os_optics.local import json_api
from os_optics.native_library import add_to_project, build_native_model
from os_optics.results import basis_of, load_run
from os_optics.results.tables import grating_order_rows, path_rows, power_budget_rows

from helpers import arrivals, issue_codes, run
from local_helpers import run as local_run
from test_second_harmonic import LAMBDA, LN_E, gaussian_near_field, laser, matched_period, ppln, sellmeier, slab

W = 0.5  # (mm: the beams' radius at the dump; their Rayleigh range is 0.7 m at 1064 nm)
AR = ConstantCoating(reflectance_s=0.0, reflectance_p=0.0, transmittance_s=1.0, transmittance_p=1.0)


def _disc_share(centre_mm: float, radius_mm: float = 1.0, w_mm: float = W) -> float:
    """The share of a round Gaussian beam of radius ``w_mm`` (1/e^2) on its axis that a disc of ``radius_mm`` centred
    ``centre_mm`` beside the axis takes: the 2-D Gaussian integral over the disc (midpoint rule, 2001^2 points)."""
    n = 2001
    step = 2.0 * radius_mm / n
    ys = centre_mm - radius_mm + step * (np.arange(n) + 0.5)
    zs = -radius_mm + step * (np.arange(n) + 0.5)
    y, z = np.meshgrid(ys, zs)
    inside = (y - centre_mm) ** 2 + z ** 2 <= radius_mm ** 2
    return float(np.sum(np.exp(-2.0 * (y * y + z * z) / (w_mm * w_mm)) * inside) * step * step * 2.0 / (math.pi * w_mm * w_mm))


def _dump(beside_mm: float):
    """A beam dump of 1 mm radius 10 mm behind the source (x = -10 mm), its centre ``beside_mm`` beside the beam's axis."""
    return B.beam_dump("DUMP", position=(-10.0, beside_mm, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=1.0)


# ------------------------------------------------------------------------------------------ the second harmonic


def _doubler(beside_mm: float | None) -> ProjectSnapshot:
    """1 W at 1064 nm, 0.5 mm waist, through a 2 mm PPLN (AR faces, phase matched), a plane OUT behind; a dump beside the beam
    before the crystal (none: ``None``); targets on the conversion and on OUT's harmonic power."""
    crystal = slab("PPLN", 2.0)
    crystal = crystal.model_copy(update={"surfaces": tuple(s.model_copy(update={"interaction": RefractInteraction(coating=AR)})
                                                           for s in crystal.surfaces)})
    components = (laser(waist=W), crystal) + (() if beside_mm is None else (_dump(beside_mm),))
    targets = (Target(target_id="P2", metric="shg_power_W", device_id="X", minimum=0.0),
               Target(target_id="eta", metric="shg_conversion_efficiency", device_id="X", minimum=0.0),
               Target(target_id="p532", metric="power_W", observation_id="OUT", wavelength_nm=532.0, minimum=0.0))
    return ProjectSnapshot(materials=(ppln(2.0, matched_period()),), components=components,
                           observations=(B.observation_plane("OUT", position=(60.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=50.0),),
                           targets=targets)


def test_a_crystal_fed_by_a_beam_whose_power_is_out_of_model_scope_converts_nothing_stated():
    """The dump 1.1 mm beside the beam takes 30 % of it (the reference); the beam entering the crystal is that much weaker and no
    longer Gaussian, and the harmonic grows with the square of its power: the closed form gives at most 0.49 of the chief ray's
    conversion. The pass is OUT_OF_MODEL_SCOPE (the reason names the dump), no harmonic power or efficiency, both conversion
    targets and the harmonic's power on OUT OUT_OF_MODEL_SCOPE; the harmonic beam is still traced. A dump 1.6 mm beside (0.6 %
    of the beam, beyond every pupil sample) is found by the beam's tail: out of model scope too; one 4 mm beside leaves the
    conversion as without it: PASS with the closed form's value.

    Register row T-19, withheld: the beam that enters the crystal is weaker by the share the dump takes and no longer Gaussian,
    and the harmonic grows with the square of its power."""
    n1, n2 = sellmeier(LN_E, LAMBDA), sellmeier(LN_E, LAMBDA / 2)
    share = _disc_share(1.1)
    assert 0.29 < share < 0.31
    # (P2 ~ P1^2: what the closed form gives the light that passes, against the chief ray's full power)
    assert gaussian_near_field(1.0 - share, 25.0 * 2 / math.pi, 2.0, W, n1, n2, 0.0) < 0.49 * gaussian_near_field(1.0, 25.0 * 2 / math.pi, 2.0, W, n1, n2, 0.0)

    scoped = run(_doubler(1.1))
    (row,) = scoped.nonlinear_conversions
    assert row.status.value == "OUT_OF_MODEL_SCOPE" and "DUMP/dump" in row.reason and "out of model scope" in row.reason
    assert row.harmonic_power_W is None and row.conversion_efficiency is None and row.harmonic_path_id is not None
    targets = {t.target_id: t for t in scoped.targets}
    assert {k: t.status.value for k, t in targets.items()} == dict.fromkeys(("P2", "eta", "p532"), "OUT_OF_MODEL_SCOPE")
    assert all(t.value is None for t in targets.values())
    assert "NONLINEAR_CONVERSION_OUT_OF_SCOPE" in issue_codes(scoped)
    harmonic = [a for a in arrivals(scoped, "OUT") if abs(a.beam.wavelength_nm - 532.0) < 1e-9]
    # (the harmonic's power is out of model scope behind the crystal; the dump's bounded share stays listed with the beam)
    assert harmonic and all("X/front" in a.power_out_of_scope_at and "DUMP/dump" in a.power_taken_before_at for a in harmonic)
    assert scoped.power_budget.conserved

    # (1.6 mm beside, 0.6 % of the beam meets the dump while no pupil sample does: the beam's tail is weighed and the conversion
    # is out of model scope too; 4 mm beside, beyond the beam's envelope, the conversion stands as without the dump)
    assert 0.005 < _disc_share(1.6) < 0.007
    tail = run(_doubler(1.6))
    (row,) = tail.nonlinear_conversions
    assert row.status.value == "OUT_OF_MODEL_SCOPE" and "DUMP/dump" in row.reason
    clear, control = run(_doubler(None)), run(_doubler(4.0))
    (free,), (row,) = clear.nonlinear_conversions, control.nonlinear_conversions
    assert row.status.value == "PASS" and row.harmonic_power_W == free.harmonic_power_W
    reference = gaussian_near_field(1.0, 25.0 * 2 / math.pi, 2.0, W, n1, n2, 0.0)
    assert row.harmonic_power_W == pytest.approx(reference, rel=2e-3)  # (collimated: zR 2.6 m in the crystal against 2 mm)
    assert {t.target_id: t.status.value for t in control.targets} == dict.fromkeys(("P2", "eta", "p532"), "PASS")


# -------------------------------------------------------------------------------------------- grating orders


def _grating(beside_mm: float | None, *, mirror: bool = False) -> ProjectSnapshot:
    """1 W at 1064 nm (0.5 mm) through a transmission grating with an absolute table (order 0: 0.3, order 1: 0.6) 20 mm behind
    the source; a dump beside the beam before it; a plane OUT 80 mm behind."""
    table = GratingOrderTable(normalization="absolute", polarization_basis="te_tm", interpolation="linear", data_kind="assumed",
                              wavelength_nm=(1000.0, 1100.0), aoi_deg=(0.0, 10.0), orders=(0, 1),
                              efficiency_te=(0.3,) * 4 + (0.6,) * 4, efficiency_tm=(0.3,) * 4 + (0.6,) * 4)
    face = PlaneSurface(surface_id="g", origin_mm=(20.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), aperture=CircularAperture(radius_mm=12.0),
                        medium_positive="air", medium_negative="air",
                        interaction=DiffractionGratingInteraction(mode="transmission", period_nm=3333.0, grating_vector_local=(0.0, 1.0, 0.0),
                                                                  orders=(0, 1), response=table))
    grating = EngineeringComponentSnapshot(instance_id="G", pose=B.Pose3D(), surfaces=(face,))
    components = (laser(waist=W), grating) + (() if beside_mm is None else (_dump(beside_mm),))
    return ProjectSnapshot(materials=(), components=components,
                           observations=(B.observation_plane("OUT", position=(80.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=60.0),))


def test_grating_orders_of_a_beam_whose_power_is_out_of_model_scope_state_no_power():
    """The dump takes 30 % of the beam before the grating: the orders carry at most 0.21 / 0.42 W, not the table's 0.3 / 0.6 of the
    chief ray's 1 W. Every propagating order is OUT_OF_MODEL_SCOPE with no power and no bound, its efficiency (the table's) and
    direction stated, naming the dump; the grating-orders export likewise; the paths behind the grating carry the dump. Without
    the dump the orders are KNOWN, P x efficiency.

    Register row T-20, withheld: the beam carries the bound of the share the dump takes, but not how that share divides
    between the orders."""
    result = run(_grating(1.1))
    orders = {row.order: row for row in result.grating_orders}
    assert 1.0 - _disc_share(1.1) < 0.71
    for order in orders.values():
        if order.propagation == "PROPAGATING":
            assert order.power_state.value == "OUT_OF_MODEL_SCOPE" and order.power_W is None and order.power_upper_bound_W is None
            assert order.power_out_of_scope_at == ("DUMP/dump",) and order.direction is not None
    assert (orders[0].efficiency, orders[1].efficiency) == (pytest.approx(0.3), pytest.approx(0.6))
    rows = {row["order"]: row for row in grating_order_rows(result)}
    assert rows[1]["power_state"] == "OUT_OF_MODEL_SCOPE" and rows[1]["power_W"] is None and rows[1]["power_out_of_scope_at"] == ["DUMP/dump"]
    behind = [p for p in result.paths if p.branch_surface_id == "G/g"]
    assert behind and all("DUMP/dump" in p.power_out_of_scope_at for p in behind)
    assert all("DUMP/dump" in row["power_out_of_scope_at"] for row in path_rows(result, {}) if row["branch_surface_id"] == "G/g")

    clear = run(_grating(None))
    orders = {row.order: row for row in clear.grating_orders}
    assert orders[0].power_state.value == "KNOWN" and orders[0].power_W == pytest.approx(0.3, rel=1e-12)
    assert orders[1].power_W == pytest.approx(0.6, rel=1e-12) and not orders[1].power_out_of_scope_at
    assert "power_out_of_scope_at" not in orders[1].model_dump() and grating_order_rows(clear)[0]["power_out_of_scope_at"] == []
    assert all(not p.power_out_of_scope_at and "power_out_of_scope_at" not in p.model_dump() for p in clear.paths)


ALONG_X = (0.5, 0.5, 0.5, 0.5)


def _lognormal_share(od_centre: float, od_per_mm: float, w_mm: float) -> float:
    """The share a linear density gradient passes of a Gaussian of radius ``w_mm`` centred on it: the mean of
    ``10^-(od_centre + g x)`` over the profile ``x ~ N(0, (w/2)^2)``, ``10^-od_centre exp((g ln 10 w / 2)^2 / 2)``."""
    return 10.0 ** -od_centre * math.exp((od_per_mm * math.log(10.0) * w_mm / 2.0) ** 2 / 2.0)


def _deflector(od_start: float, od_end: float) -> dict:
    """``test_acousto_optic``'s 16-tone deflector (table: 0.05 per tone, 0.1 zero order) behind a variable attenuator of OD
    ``od_start`` at x = -2 mm to ``od_end`` at +2 mm across its 0.5 mm beam."""
    from test_acousto_optic import aod, bench, table

    from local_helpers import plane
    document = bench(aod(efficiency=table()), [plane("SCREEN", (100.0, 0.0, 0.0), radius_mm=40.0)], power_w=1e-3)
    nd = build_native_model("filter.variable_nd", {"law": "LINEAR_GRADIENT", "od_start": repr(od_start), "od_end": repr(od_end), "x_start_mm": "-2",
                                                   "x_end_mm": "2", "width_mm": "20", "height_mm": "20", "valid_band_nm": [400, 1100],
                                                   "uncoated_transmittance": "1"})
    return add_to_project(document, nd, instance_id="ND", position_mm=(-30.0, 0.0, 0.0), quaternion_wxyz=ALONG_X)


def test_an_acousto_optic_deflector_behind_a_varying_attenuator_states_no_order_power():
    """OD 0 -> 4 over 4 mm across the 0.5 mm beam: the chief ray passes 10^-2 of it, the light 1.18e-2 (the log-normal mean), so
    an order's power from the chief ray is 15 % low. Every order of the deflector is OUT_OF_MODEL_SCOPE with no power, its
    efficiency stated; the run summary the Designer reads marks each arrival, the 3-D view states no footprint power and no path
    power behind it. A uniform attenuator (OD 2 -> 2.0001): each order KNOWN, P x 10^-2 x efficiency.

    Register row T-21, withheld: the chief ray's transmittance is not the mean transmittance over the beam's profile, which the
    attenuator step does not compute."""
    assert _lognormal_share(2.0, 1.0, 0.5) == pytest.approx(0.01180, rel=1e-3)
    run_ = local_run(_deflector(0.0, 4.0))
    orders = load_run(run_).grating_orders("AOD")
    assert orders and all(o["power_state"] == "OUT_OF_MODEL_SCOPE" and o.get("power_W") is None and o["power_out_of_scope_at"] == ["ND/attenuator"]
                          for o in orders)
    assert {o["order"]: o["efficiency"] for o in orders}[0] == pytest.approx(0.1)
    summary = json_api.run_summary(run_)
    (screen,) = [o for o in summary["observations"] if o["observation_id"] == "SCREEN"]
    assert screen["arrivals"] and all(a["power_out_of_scope_at"] == ["ND/attenuator"] for a in screen["arrivals"])
    (view_plane,) = [p for p in run_.optical_view.observation_planes if p.observation_id == "SCREEN"]
    assert view_plane.footprints and all(f.power_W is None for f in view_plane.footprints)
    assert run_.optical_view.beam_paths and all(p.final_power_W is None for p in run_.optical_view.beam_paths)

    run_ = local_run(_deflector(2.0, 2.0001))
    orders = {o["order"]: o for o in load_run(run_).grating_orders("AOD")}
    assert orders[0]["power_state"] == "KNOWN" and "power_out_of_scope_at" not in orders[0]
    assert orders[0]["power_W"] == pytest.approx(1e-3 * 10.0 ** -2.00005 * 0.1, rel=1e-4)
    summary = json_api.run_summary(run_)
    (screen,) = [o for o in summary["observations"] if o["observation_id"] == "SCREEN"]
    assert all("power_out_of_scope_at" not in a for a in screen["arrivals"])
    (view_plane,) = [p for p in run_.optical_view.observation_planes if p.observation_id == "SCREEN"]
    assert all(f.power_W is not None for f in view_plane.footprints)


# --------------------------------------- polarization mixtures, the plane's bounds, the power budget, scattered light


def _partial(beside_mm: float | None) -> ProjectSnapshot:
    """1 W at 1064 nm (0.5 mm), partially polarized (degree 0.5), folded by a mirror 20 mm behind the source onto a plane OUT; a dump
    beside the beam before the mirror; an absorbing plane SIDE 6 mm beside the folded beam (its active area 1 mm: the beam passes it
    outside)."""
    source = laser(waist=W)
    source = type(source).model_validate({**source.model_dump(), "source": {**source.source.model_dump(), "degree_of_polarization": 0.5}})
    components = (source, B.mirror_for_reflection("M", position=(20.0, 0.0, 0.0), d_in=(1.0, 0.0, 0.0), d_out=(0.0, 1.0, 0.0))) + (
        () if beside_mm is None else (_dump(beside_mm),))
    observations = (B.observation_plane("OUT", position=(20.0, 60.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=50.0),
                    B.observation_plane("SIDE", position=(26.0, 40.0, 0.0), normal=(0.0, 1.0, 0.0), radius_mm=1.0, absorbing=True))
    return ProjectSnapshot(materials=(), components=components, observations=observations)


def test_partially_polarized_light_behind_a_share_taken_states_no_stokes_sum_and_the_plane_its_interval():
    """Behind the dump (30 % of the beam taken, the reference) the two polarization components' powers are their chief
    rays', each within the bound of the share the dump took of it. The plane reads ``[P - S, P]``, which holds the light
    that is there, ``0.7 P``; the wavelength's power carries the same bound as its entry; the route's Stokes parameters are
    not stated (the shares the two components lost are not known: OUT_OF_MODEL_SCOPE, naming the dump); the budget observes
    the plane's value; the beam crossing SIDE outside its active area names the dump. Without the dump: S = (P, P x 0.5, 0,
    0) up to the mirror's sign of S1 (degree 0.5), the budget observes P, the plane's upper bound is its power.

    Register row T-22, withheld: the shares the dump took of the two polarization components are not known, so the route's
    Stokes parameters are not stated."""
    result = run(_partial(1.1))
    share = _disc_share(1.1)
    out = next(o for o in result.observations if o.observation_id == "OUT")
    (mixture,) = out.polarization_mixtures
    assert mixture.quantity_state.value == "OUT_OF_MODEL_SCOPE" and mixture.stokes_W is None and mixture.degree_of_polarization is None
    assert mixture.power_out_of_scope_at == ("DUMP/dump",) and mixture.issue_codes == ("OUT_OF_MODEL_SCOPE",)
    assert out.quantity_state.value == "BOUNDED" and out.power_upper_bound_W == out.total_power_W
    assert all(a.power_out_of_scope_at == () and a.power_taken_before_at == ("DUMP/dump",) for a in out.arrivals)
    taken = sum(a.power_taken_before_upper_W for a in out.arrivals)
    assert share * out.total_power_W <= taken < 1.2 * share * out.total_power_W
    assert out.power_lower_bound_W == pytest.approx(out.total_power_W - taken, rel=1e-12)
    assert out.power_lower_bound_W <= (1.0 - share) * out.total_power_W <= out.total_power_W
    (spectral,) = out.power_by_wavelength
    assert spectral.power_out_of_scope_at == () and spectral.power_W == out.total_power_W
    entry = basis_of(spectral, "power_W")
    assert entry.kind.value == "UPPER_BOUND"
    assert [(o.part.value, o.at, o.code) for o in entry.omitted] == [("BEAM_PART_ELSEWHERE", ("DUMP/dump",), "POWER_TAKEN_BEFORE_PLANE")]
    assert entry.omitted[0].upper == pytest.approx(taken, rel=1e-12)
    (line,) = result.power_budget.lines
    assert line.observed_W["OUT"] == pytest.approx(out.total_power_W, rel=1e-12) and result.power_budget.conserved
    side = next(o for o in result.observations if o.observation_id == "SIDE")
    assert side.outside_active_area and all(c.power_out_of_scope_at == ("DUMP/dump",) for c in side.outside_active_area)

    clear = run(_partial(None))
    out = next(o for o in clear.observations if o.observation_id == "OUT")
    (mixture,) = out.polarization_mixtures
    s0, s1, s2, s3 = mixture.stokes_W
    assert mixture.quantity_state.value == "KNOWN" and s0 == pytest.approx(out.total_power_W, rel=1e-12)
    assert abs(s1) == pytest.approx(0.5 * s0, rel=1e-9) and abs(s2) < 1e-12 and abs(s3) < 1e-12
    assert mixture.degree_of_polarization == pytest.approx(0.5, rel=1e-9) and "power_out_of_scope_at" not in mixture.model_dump()
    assert out.power_upper_bound_W == pytest.approx(out.total_power_W, rel=1e-12)
    assert all("power_out_of_scope_at" not in s.model_dump() for s in out.power_by_wavelength)
    (line,) = clear.power_budget.lines
    assert line.observed_W["OUT"] == pytest.approx(out.total_power_W, rel=1e-12) and "observed_out_of_scope_at" not in line.model_dump()
    side = next(o for o in clear.observations if o.observation_id == "SIDE")
    assert side.outside_active_area and all("power_out_of_scope_at" not in c.model_dump() for c in side.outside_active_area)


def test_light_scattered_out_of_a_beam_whose_power_is_out_of_model_scope_names_it_as_a_source_and_in_the_ledger():
    """``test_out_of_scope_power_results``'s scatterer behind a varying attenuator: the scattered light's source row (its incident
    and emitted power, the chief ray's share) names the attenuator, and the budget observes no power of it on D. Behind the fixed
    OD 2 the row is 0.1 x 0.01 x P, the budget observes the plane's light.

    Register row T-23, withheld: the chief ray's transmittance is not the mean transmittance over the beam's profile, which the
    attenuator step does not compute."""
    from test_out_of_scope_power_results import _scattering

    result = _scattering(True)
    (row,) = result.sampled_sources
    assert row.power_out_of_scope_at == ("ND/dump",)
    (line,) = result.power_budget.lines
    assert "D" not in line.observed_W and line.observed_out_of_scope_at == ("D",)

    result = _scattering(False)
    (row,) = result.sampled_sources
    assert "power_out_of_scope_at" not in row.model_dump() and row.emitted_W == pytest.approx(0.1 * 0.01, rel=1e-6)
    (line,) = result.power_budget.lines
    (disc,) = result.observations
    assert line.observed_W["D"] == pytest.approx(disc.total_power_W, rel=1e-12)
