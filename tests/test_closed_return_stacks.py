"""Stacks of parallel planes and cavities whose chained returns the pair and open-path counts miss.

Light the trace leaves out as negligible between three or more parallel reflecting planes -- coupled linear cavities,
strongly coated plates, a mirror behind a window, three or more bare windows -- is reflected back and forth through
every sequence of them, closed ones included. Formerly such a stack fell back to the pair and open-path
counts when the sufficient all-orders series ``1 / (1 - A)`` (``A = sum m_s r_s < 1``) did not hold, and its surfaces
got a finite upper bound that rested on neither (the coupled cavity's end mirror 2.6e11 W/cm^2, three bare windows
2.2e12). Now every reflection sequence is counted where a series
converges (``validation/closed_returns.py``, ``_alternating_series``: between parallel planes the light's direction
along their normal alternates with each reflection), and the bound is withheld with a per-surface issue where none does.

The independent checks: the exact field of a plane wave in the stack (flux-normalized transfer matrices at normal
incidence with the Fresnel coefficients of N-BK7, for random layer phases) never exceeds the series, and the series
includes every reflection sequence (a direct enumeration of the alternating sequences). Every power rating is fixture
data declared in the test project, never product data.
"""

from __future__ import annotations

import cmath
import math
import random

import pytest

from os_optics import builders as B
from os_optics.contracts import PowerRating, ProjectSnapshot, Target

from helpers import run
from test_group_delay_and_returns import _bk7
from test_return_pairs import LAM, _coat, _constant, _gains, _surface

FIX = "verifier fixture rating, not product data"
COUPLER = 1.0 - 1e-12
HR = 1.0 - 1e-6


def _rating(value: float) -> PowerRating:
    return PowerRating(regime="CW", value=value, unit="W/cm2", wavelength_nm=LAM, status="DECLARED", basis=FIX,
                       value_definition="PEAK", beam_diameter_mm=50.0, beam_diameter_definition="1/E2",
                       polarization="LINEAR", polarization_condition="linear")


def _rated(project: ProjectSnapshot, device: str, value: float = 1e15) -> ProjectSnapshot:
    parts = tuple(c.model_copy(update={"power_ratings": (_rating(value),)}) if c.instance_id == device else c
                  for c in project.components)
    target = Target(target_id="u", device_id=device, metric="power_rating_utilization", maximum=1.0)
    return project.model_copy(update={"components": parts, "targets": (target,)})


def _source(x: float = -400.0):
    return B.laser("S", wavelength_nm=LAM, power_W=1.0, waist_radius_mm=0.5, position=(x, 0.0, 0.0), direction=(1.0, 0.0, 0.0))


def _flat(name: str, x: float, normal: float, r: float, both: bool = False, radius: float = 3.0):
    return _coat(B.mirror(name, position=(x, 0.0, 0.0), normal=(normal, 0.0, 0.0), clear_radius_mm=radius), _constant(r), both=both)


def coupled_cavity() -> ProjectSnapshot:
    """Two linear cavities sharing a mirror, fed only by left-out light: the coupler M1 (transmits 1e-12 of the 1 W
    beam), M2 (R 0.9 on both sides) and the HR end mirror M3, parallel on one axis."""
    return ProjectSnapshot(components=(_source(), _flat("M1", 0.0, 1.0, COUPLER, both=True), _flat("M2", 100.0, -1.0, 0.9, both=True),
                                       _flat("M3", 200.0, -1.0, HR)))


def windows(n: int, coating=None, gap: float = 30.0) -> ProjectSnapshot:
    parts = [_source(-100.0)]
    for i in range(n):
        w = B.window(f"W{i + 1}", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(i * gap, 0.0, 0.0))
        parts.append(_coat(w, coating) if coating is not None else w)
    return ProjectSnapshot(components=tuple(parts))


def window_before_mirror() -> ProjectSnapshot:
    """A bare window in front of a parallel mirror (R 0.99): a double pass."""
    w = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(0.0, 0.0, 0.0))
    return ProjectSnapshot(components=(_source(-100.0), w, _flat("M", 50.0, -1.0, 0.99, radius=12.7)))


def _withheld(result, surface_id: str, members: set[str], completed: bool = True) -> None:
    surface = _surface(result, surface_id)
    (comparison,) = surface.ratings
    assert comparison.exposure_upper_bound is None
    # (the followed beam's own exposure may be reported beside the missing bound: an estimate that names the return)
    if comparison.exposure_value is not None:
        entry = comparison.value_basis["exposure_value"]
        assert entry.kind.value == "ESTIMATE" and "POSSIBLE_RETURN" in {o.part.value for o in entry.omitted}
    assert surface.rating_status != "WITHIN_RATING"
    assert result.targets[0].assessment.value == "INDETERMINATE"
    assert "EXPOSURE_BOUND_WITHHELD" in surface.reason_codes
    (cause,) = [i for i in result.issues if i.numerical_context.get("cause") == "CLOSED_RETURN_NOT_BOUNDED"
                and i.numerical_context["quantity"] == "exposure_upper_bound"]
    assert (cause.code.value, cause.status.value) == ("EXPOSURE_BOUND_WITHHELD", "NOT_APPLICABLE")
    assert surface_id in cause.numerical_context["surface_ids"] or cause.numerical_context["surface_count"] > len(
        cause.numerical_context["surface_ids"])
    assert members <= {sid for cycle in cause.numerical_context["closed_return_surfaces"] for sid in cycle.split(", ")}
    # the calculation itself completed (the ghosts of the plate stacks exhaust the probes' default trace budget, an
    # auxiliary analysis recorded per phase: never partial, never failed); the traced light and its power ledger stay
    execution = result.outcome.execution
    assert execution.outcome.value == "COMPLETED"
    assert completed or set(execution.budgets.exhausted_by_phase) == {"untraced_light"}
    assert result.power_budget.total_input_W == pytest.approx(1.0)


def test_a_coupled_linear_cavity_withholds_its_bound_and_names_its_mirrors():
    """Formerly M3 reported an upper bound of 2.6e11 W/cm^2 from the pairs (M1, M2), (M2, M3), (M1, M3), which do
    not bound the light one cavity sends into the other. The stack's alternating series does not converge (the HR
    mirrors return everything to each other through M2), so the bound is withheld."""
    gains = _gains(coupled_cavity())
    assert gains.closed_cycles and gains.closed_series is None
    assert not math.isfinite(gains.crossing_field("M3/face")) and not math.isfinite(gains.crossings("M3/face"))
    result = run(_rated(coupled_cavity(), "M3"))
    _withheld(result, "M3/face", {"M1/face", "M2/face", "M3/face"})
    assert _surface(result, "M1/face").hits  # (the traced beam on the coupler stays)


@pytest.mark.parametrize("variant", ["coated plates", "three bare windows and a lens"])
def test_a_stack_no_series_bounds_withholds_its_bound(variant: str):
    """Two windows coated R 0.5 on every face (A = 2.8: the alternating series diverges too), and three bare windows with
    a thin lens between the first two (A = 1.2 > 1; the alternating series needs every surface that redirects light to be
    one of the planes, and a lens turns it). Formerly: finite bounds of 6.7e14 and 3.1e12 W/cm^2 from the
    pairs."""
    if variant == "coated plates":
        project, rated = windows(2, _constant(0.5)), "W2"
    else:
        project = windows(3, gap=100.0)
        lens = B.thin_lens("L", focal_length_mm=50.0, clear_radius_mm=12.7, position=(50.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0))
        project, rated = project.model_copy(update={"components": (*project.components, lens)}), "W2"
    gains = _gains(project)
    assert gains.closed_cycles and gains.closed_series is None
    result = run(_rated(project, rated))
    face = f"{rated}/back" if variant == "coated plates" else f"{rated}/front"
    _withheld(result, face, {f"{rated}/front", f"{rated}/back"}, completed=False)


def test_a_linear_cavity_with_a_window_inside_and_a_curved_end_mirror_withholds_its_bound():
    """The coupler M1, a bare window and a concave HR end mirror (radius 300 mm) on one axis: the window's faces and the
    flat coupler are parallel, the curved mirror closes them into one cavity. Formerly: 4.0e14 W/cm^2 on M2 from
    the pairs."""
    m1 = _flat("M1", 0.0, 1.0, COUPLER, both=True)
    w = B.window("W", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(50.0, 0.0, 0.0))
    m2 = _coat(B.spherical_mirror("M2", position=(100.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_of_curvature_mm=300.0,
                                  clear_radius_mm=3.0), _constant(HR))
    project = ProjectSnapshot(components=(_source(), m1, w, m2))
    gains = _gains(project)
    assert gains.closed_cycles and gains.closed_series is None
    _withheld(run(_rated(project, "M2")), "M2/face", {"M1/face", "W/front", "W/back", "M2/face"})


# ------------------------------------------------------------------------------------------------ the alternating series


def _stack_field(n: float, faces: int, phases: list[float]) -> float:
    """The largest |a+| + |a-| over the regions of a plane wave of flux amplitude 1 entering a stack of ``faces`` / 2
    windows of index ``n`` in air at normal incidence (``phases``: each layer's one-way phase, glass and air in turn),
    with flux-normalized transfer matrices: an interface from index n1 to n2 relates the amplitudes on its two sides by
    (1 / t) [[1, r], [r, 1]], r = (n1 - n2) / (n1 + n2), t = sqrt(1 - r^2); a layer by diag(exp(-i phi), exp(i phi))."""
    r_glass = (1.0 - n) / (1.0 + n)
    t = math.sqrt(1.0 - r_glass * r_glass)
    vector = (1.0 + 0j, 0j)  # (transmitted, nothing coming back from the right)
    regions = [vector]
    for k in reversed(range(faces)):
        r = r_glass if k % 2 == 0 else -r_glass  # (air -> glass at even faces, glass -> air at odd ones)
        a, b = vector
        vector = ((a + r * b) / t, (r * a + b) / t)
        regions.append(vector)
        if k > 0:
            phase = phases[k - 1]
            a, b = vector
            vector = (a * cmath.exp(-1j * phase), b * cmath.exp(1j * phase))
    incident = abs(regions[-1][0])
    return max((abs(a) + abs(b)) / incident for a, b in regions)


def _enumerated(r: list[float], depth: list[float], reflections: int) -> float:
    """1 plus the sum over every sequence of up to ``reflections`` reflections at planes of the stack (each next one
    ahead in the direction the previous one sent the light) of the products of the planes' amplitude reflectances,
    starting either way: dynamic programming over (last plane, direction)."""
    best = 0.0
    for start_up in (True, False):
        ways = {(t, not start_up): r[t] for t in range(len(r))}
        total = sum(ways.values())
        for _ in range(reflections - 1):
            nxt: dict[tuple[int, bool], float] = {}
            for (s, up), amp in ways.items():
                for t in range(len(r)):
                    if (depth[t] > depth[s]) if up else (depth[t] < depth[s]):
                        nxt[(t, not up)] = nxt.get((t, not up), 0.0) + amp * r[t]
            ways = nxt
            total += sum(ways.values())
        best = max(best, total)
    return 1.0 + best


@pytest.mark.parametrize("count", [3, 4])
def test_bare_window_stacks_get_a_series_that_holds_every_reflection_sequence(count: int):
    """Three and four bare N-BK7 windows (A = 1.21 and 1.62: the sufficient series fails). Their alternating series is
    finite, at least every enumerated sum of reflection sequences (up to 60 reflections), and at least the exact
    multiple-beam field of a plane wave in the stack for every one of 2000 random sets of layer phases."""
    gains = _gains(windows(count))
    assert gains.closed_cycles and gains.closed_series is not None and math.isfinite(gains.closed_series)
    assert math.isfinite(gains.crossing_field(f"W{count}/back"))
    n = _bk7(LAM * 1e-3)
    r = abs((n - 1.0) / (n + 1.0))
    faces = 2 * count
    depth = [x for i in range(count) for x in (30.0 * i - 1.5, 30.0 * i + 1.5)]
    assert gains.closed_series >= _enumerated([r] * faces, depth, 60)
    rng = random.Random(363)
    worst = max(_stack_field(n, faces, [rng.uniform(0.0, 2.0 * math.pi) for _ in range(faces - 1)]) for _ in range(2000))
    assert 1.0 < worst <= gains.closed_series


def test_a_window_before_a_mirror_keeps_a_supported_bound():
    """A bare window before a parallel R 0.99 mirror (A = 3.4 with the mirror's meetings): its returns converge in the
    alternating series (each round trip through a window face keeps at most 0.2 of the field), so the surfaces keep an
    upper bound and no closed-return issue is raised."""
    gains = _gains(window_before_mirror())
    assert gains.closed_cycles and gains.closed_series is not None and gains.closed_series < 10.0
    surface = _surface(run(_rated(window_before_mirror(), "W")), "W/front")
    assert surface.ratings[0].exposure_upper_bound is not None and "OUT_OF_MODEL_SCOPE" not in surface.reason_codes


@pytest.mark.parametrize("make, device, face", [(coupled_cavity, "M3", "M3/face"), (lambda: windows(3), "W3", "W3/back")])
def test_a_stack_s_bound_does_not_depend_on_the_optional_target(make, device: str, face: str):
    """Removing the optional power_rating_utilization target or changing only its threshold leaves the exposure,
    its bound (withheld or finite) and its applicability unchanged."""
    project = _rated(make(), device)
    with_target = run(project)
    without = run(project.model_copy(update={"targets": ()}))
    loosened = run(project.model_copy(update={"targets": tuple(t.model_copy(update={"maximum": 1e20}) for t in project.targets)}))
    assert with_target.exposures == without.exposures == loosened.exposures
    assert [i for i in with_target.issues if i.location == f"surface:{face}"] == \
        [i for i in without.issues if i.location == f"surface:{face}"]
