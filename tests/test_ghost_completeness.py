"""What the ghost analysis's bound on the light not listed covers (``os_optics.results.ghosts``, ``results.ghost_trace``).

References computed here, independently of the Engine:

* bulk scattering: a plate of thickness ``L`` with absorption ``alpha`` and scattering ``mu_s`` scatters, on the primary's single
  pass, ``P (1 - R) (1 - exp(-(alpha + mu_s) L)) mu_s / (alpha + mu_s)`` (``R`` the bare Fresnel reflectance at normal incidence);
* the beam-averaged reflectance of a tilted plate: a paraxial Gaussian beam's plane-wave spectrum, ``exp(-2 theta^2 / theta_0^2)``
  with ``theta_0 = lambda / (pi w_0)``, integrated on a grid; each plane wave meets the plate at its own angle of incidence and
  carries its own s and p shares of the source's linear polarization (the field projected across its direction);
* the geometric pupil-sample basis: behind a stop at 1 w (the Gaussian state no longer valid) and a stop at 0.5 w the power is
  ``P (1 - exp(-0.5))``, times ``T^2`` through a plate at normal incidence and ``T^2 R^2`` for its two-bounce ghost.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import ValidationRequest
from os_optics import builders as B
from os_optics.contracts import JonesVector, ProjectSnapshot
from os_optics.contracts.materials import ConstantIndexMaterial
from os_optics.contracts.optics import SurfaceScatter
from os_optics.errors.status import QuantityState
from os_optics.results.ghost_trace import SINK_REASONS
from os_optics.results.model import RunResults
from os_optics.optics.rays.tracer import UNKNOWN_FATE_SINKS
from os_optics.validation import validate

WL = 633.0
N_GLASS = 1.5
LINE_W = 1e-3


def _glass(**fields) -> ConstantIndexMaterial:
    # The reference models isolate scattering/reflectance: alpha=0 is an explicit test-only assumption.
    return ConstantIndexMaterial(material_id="g", valid_wavelength_nm=(400.0, 900.0), refractive_index=N_GLASS,
                                 absorption_coefficient_per_mm=fields.pop("absorption_coefficient_per_mm", 0.0), **fields)


def _results(components, materials, *, plane_x: float = 60.0) -> RunResults:
    image = B.observation_plane("image", position=(plane_x, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=50.0)
    request = ValidationRequest(project=ProjectSnapshot(name="ghosts", components=tuple(components), observations=(image,),
                                                        materials=tuple(materials)))
    return RunResults(result=validate(request), request=request.model_dump(mode="json"))


def _laser(**kwargs):
    defaults = {"wavelength_nm": WL, "power_W": LINE_W, "waist_radius_mm": 0.5, "position": (-20.0, 0.0, 0.0), "direction": (1.0, 0.0, 0.0)}
    return B.laser("L", **{**defaults, **kwargs})


def _line(doc) -> str:
    (key,) = [k for k in doc.arrays if k.endswith("ghosts")]
    return key[:-len("ghosts")]


def _bound(doc) -> tuple[float, str]:
    q = doc.quantity(_line(doc) + "unlisted_power_upper_relative_to_line")
    assert q.state is QuantityState.BOUNDED
    return float(q.interval.upper), q.interval.kind.value


def test_every_sink_of_unknown_fate_enters_the_bound_with_its_own_reason():
    # Unstated bulk loss is of unknown fate and uses the ghost tracer's OUT_OF_MODEL fallback, not SCATTERED.
    assert set(UNKNOWN_FATE_SINKS) - {"BULK_LOSS_UNKNOWN"} <= set(SINK_REASONS)
    assert "BULK_LOSS_UNKNOWN" in UNKNOWN_FATE_SINKS and SINK_REASONS.get("BULK_LOSS_UNKNOWN", "OUT_OF_MODEL") == "OUT_OF_MODEL"
    assert SINK_REASONS["BULK_SCATTER"] == SINK_REASONS["SURFACE_SCATTER"] == "SCATTERED"


# ------------------------------------------------------------------------------------------------- scattered light


def test_bulk_scattered_light_enters_the_bound_and_is_named_as_not_followed():
    alpha, mu_s, length = 0.01, 0.5, 5.0
    plate = B.window("W", thickness_mm=length, clear_radius_mm=9.0, material="g", position=(10.0, 0.0, 0.0))
    results = _results([_laser(), plate], [_glass(absorption_coefficient_per_mm=alpha, scattering_coefficient_per_mm=mu_s, anisotropy_g=0.8)])
    doc = results.ghosts("image")
    r = ((N_GLASS - 1.0) / (N_GLASS + 1.0)) ** 2
    scattered = LINE_W * (1.0 - r) * (1.0 - math.exp(-(alpha + mu_s) * length)) * mu_s / (alpha + mu_s)
    ledger = results.result.power_budget.lines[0].sinks["BULK_SCATTER"]
    assert ledger == pytest.approx(scattered, rel=1e-3)  # (the primary's pass, plus its weak internal reflections)
    # before: the scattered 87 % of the line never reached the bound, which read 2.9e-8 of the line, COMPLETED and proven
    bound, kind = _bound(doc)
    assert kind == "PROVEN" and bound * LINE_W >= scattered
    assert doc.completeness.unenumerated_power_upper_W >= scattered
    assert any(item.reason == "SCATTERED" for item in doc.completeness.not_followed)
    assert not any(item.reason == "OUT_OF_MODEL" for item in doc.completeness.not_followed)
    # the enumeration ran to its end: the scattered light is in not_followed and in the bound, not in the execution record
    assert doc.execution.outcome.value == "COMPLETED" and doc.execution.complete and doc.execution.reason == ""


def test_surface_scatter_is_named_as_scattered_light_not_as_light_outside_the_model():
    plate = B.window("W", thickness_mm=5.0, clear_radius_mm=9.0, material="g", position=(10.0, 0.0, 0.0))
    front = plate.surfaces[0]
    rough = front.model_copy(update={"interaction": front.interaction.model_copy(update={"scatter": SurfaceScatter(fraction=0.05, side="transmit", sample_count=64)})})
    plate = plate.model_copy(update={"surfaces": (rough, plate.surfaces[1])})
    doc = _results([_laser(), plate], [_glass()]).ghosts("image")
    reasons = {item.reason for item in doc.completeness.not_followed}
    assert "SCATTERED" in reasons and "OUT_OF_MODEL" not in reasons
    assert doc.execution.outcome.value == "COMPLETED" and doc.execution.complete
    scattered = sum(item.power_upper_W for item in doc.completeness.not_followed if item.reason == "SCATTERED")
    assert scattered >= 0.05 * LINE_W  # (the front face scatters 5 % of the light that meets it)


# ------------------------------------------------------------------------------------------------- the budget


def test_a_source_line_the_budget_does_not_launch_enters_the_bound_whole():
    other = B.laser("M", wavelength_nm=WL, power_W=0.5 * LINE_W, waist_radius_mm=0.5, position=(-20.0, 3.0, 0.0), direction=(1.0, 0.0, 0.0))
    plate = B.window("W", thickness_mm=5.0, clear_radius_mm=9.0, material="g", position=(10.0, 0.0, 0.0))
    doc = _results([_laser(), other, plate], [_glass()]).ghosts("image", {"max_paths": 1})
    assert doc.execution.outcome.value == "PARTIAL" and doc.execution.termination.value == "BUDGET_EXHAUSTED"
    # before: the line not launched had no piece at all, and its bound read 0 (PROVEN)
    q = doc.quantity("M#0.unlisted_power_upper_relative_to_line")
    assert q.interval.kind.value == "PROVEN" and q.interval.upper == pytest.approx(1.0, rel=1e-9)
    assert any(item.item_id == "M:PATH_BUDGET" for item in doc.completeness.not_followed)


# ------------------------------------------------------------------------------------------------- the reflectance over the beam


def _beam_average(theta0: float, aoi: float, polarization: str) -> tuple[float, float, float, float]:
    """(front reflectance, rear reflection T R) at the chief ray and averaged over the beam's plane-wave spectrum (module
    docstring) of a plate of index ``N_GLASS`` tilted by ``aoi`` about z, met along +x."""
    normal = np.array([-math.cos(aoi), -math.sin(aoi), 0.0])
    d, across_y, across_z = np.eye(3)
    grid = np.linspace(-6.0 * theta0, 6.0 * theta0, 801)
    tx, ty = np.meshgrid(grid, grid)
    weight = np.exp(-2.0 * (tx * tx + ty * ty) / (theta0 * theta0))
    k = d[None, None, :] + tx[..., None] * across_y[None, None, :] + ty[..., None] * across_z[None, None, :]
    k = k / np.linalg.norm(k, axis=-1)[..., None]
    ci = np.abs(k @ normal)
    st = np.sqrt(1.0 - ci * ci) / N_GLASS
    ct = np.sqrt(1.0 - st * st)
    rs1, rp1 = ((ci - N_GLASS * ct) / (ci + N_GLASS * ct)) ** 2, ((N_GLASS * ci - ct) / (N_GLASS * ci + ct)) ** 2
    rs2, rp2 = ((N_GLASS * ct - ci) / (N_GLASS * ct + ci)) ** 2, ((ct - N_GLASS * ci) / (ct + N_GLASS * ci)) ** 2
    field = across_y if polarization == "p" else across_z
    e = field[None, None, :] - (k @ field)[..., None] * k
    e = e / np.linalg.norm(e, axis=-1)[..., None]
    s = np.cross(k, normal)
    s = s / np.linalg.norm(s, axis=-1)[..., None]
    a = np.sum(e * s, axis=-1) ** 2
    front = a * rs1 + (1.0 - a) * rp1
    rear = a * (1.0 - rs1) * rs2 + (1.0 - a) * (1.0 - rp1) * rp2
    c = grid.size // 2
    total = float(weight.sum())
    return float(front[c, c]), float((weight * front).sum()) / total, float(rear[c, c]), float((weight * rear).sum()) / total


@pytest.mark.parametrize("angle_deg, polarization", ((0.0, "p"), (90.0, "s")))
def test_the_bound_covers_the_beam_averaged_reflectance_above_the_chief_rays(angle_deg, polarization):
    """A 45 deg plate met by a beam of far-field half angle 0.2 rad: Fresnel reflectance is convex in the angle, so the beam's mean
    reflectance exceeds its chief ray's (p: by 11 %, s: by 3 %). The chief ray's value is what the trace lists; the bound on the
    light not listed must cover the mean."""
    waist = 0.001
    theta0 = WL * 1e-6 / (math.pi * waist)
    c = math.sqrt(0.5)
    plate = B.window("W", thickness_mm=2.0, clear_radius_mm=10.0, material="g", position=(5.0, 0.0, 0.0), axis=(c, c, 0.0))
    results = _results([_laser(waist_radius_mm=waist, position=(0.0, 0.0, 0.0), polarization=JonesVector.linear(angle_deg)), plate], [_glass()],
                       plane_x=30.0)
    chief_front, mean_front, chief_rear, mean_rear = _beam_average(theta0, math.radians(45.0), polarization)
    assert mean_rear > chief_rear * 1.02  # (the effect the bound must cover)
    listed = results.ghosts("image", {"max_bounce_order": 1, "max_listed": 2})
    line = _line(listed)
    rows = {tuple(row["bounce_surfaces"]): row["path_id"] for row in listed.arrays[line + "ghosts"]}
    front_q = listed.quantity(f"{line}{rows[('W/front',)]}.reflected_power_relative_to_line")
    rear_q = listed.quantity(f"{line}{rows[('W/back',)]}.reflected_power_relative_to_line")
    assert front_q.value == pytest.approx(chief_front, rel=1e-6) and rear_q.value == pytest.approx(chief_rear, rel=1e-6)
    # the rear reflection not listed: its bound covers the beam's mean (the chief ray's value, the bound before, did not)
    doc = results.ghosts("image", {"max_bounce_order": 1, "max_listed": 1})
    (item,) = [i for i in doc.completeness.not_followed if i.reason == "NOT_LISTED"]
    assert item.item_id == f"{rows[('W/back',)]}:NOT_LISTED"
    assert chief_rear * LINE_W < mean_rear * LINE_W <= item.power_upper_W
    assert doc.completeness.proven and _bound(doc)[1] == "PROVEN"


def test_a_collimated_beam_at_normal_incidence_keeps_the_bound_at_the_chief_rays_value():
    plate = B.window("W", thickness_mm=5.0, clear_radius_mm=9.0, material="g", position=(10.0, 0.0, 0.0))
    doc = _results([_laser(waist_radius_mm=1.0), plate], [_glass()]).ghosts("image")
    r = ((N_GLASS - 1.0) / (N_GLASS + 1.0)) ** 2
    bound, kind = _bound(doc)
    # the two-bounce ghost's reflection at the rear face born beyond the bounce order: T R^3, the reflectances within 1e-5
    assert kind == "PROVEN" and (1.0 - r) * r ** 3 <= bound <= (1.0 - r) * r ** 3 * (1.0 + 1e-5)


# ------------------------------------------------------------------------------------------------- the geometric pupil-sample basis


def test_powers_on_the_geometric_basis_state_their_sampling_resolution():
    laser = _laser(waist_radius_mm=1.0)
    stops = (B.aperture_stop("S1", position=(-10.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=1.0),
             B.aperture_stop("S2", position=(-5.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=0.5))
    plate = B.window("W", thickness_mm=5.0, clear_radius_mm=9.0, material="g", position=(10.0, 0.0, 0.0))
    doc = _results([laser, *stops, plate], [_glass()]).ghosts("image")
    r = ((N_GLASS - 1.0) / (N_GLASS + 1.0)) ** 2
    t = 1.0 - r
    passed = 1.0 - math.exp(-0.5)
    line = _line(doc)
    primary = doc.quantity(line + "primary_power_relative_to_line")
    resolution = primary.convergence.resolution
    assert resolution is not None and resolution > 0.0 and "sampling resolution" in primary.message
    assert abs(primary.value - passed * t * t) <= resolution
    (ghost,) = [row for row in doc.arrays[line + "ghosts"] if row["bounce_surfaces"] == ["W/back", "W/front"]]
    power = doc.quantity(f"{line}{ghost['path_id']}.power_relative_to_line")
    assert abs(power.value - passed * t * t * r * r) <= power.convergence.resolution
    relative = doc.quantity(f"{line}{ghost['path_id']}.relative_power")
    assert relative.value == pytest.approx(r * r, rel=1e-9) and relative.convergence.resolution > 0.0
    # (the ghost and the primary share the same sampled share: their ratio is the Fresnel one; its resolution says how far apart
    # the two sampled powers may be)
    assert not doc.completeness.proven and _bound(doc)[1] == "NOMINAL"
