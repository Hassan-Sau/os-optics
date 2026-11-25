"""A coherent sum on a plane whose only uncounted light is left out as negligible (W1, W28).

That light's field may move a coherent sum ``P`` by at most ``(sqrt(P) + r)^2 - P`` upwards and ``P - (sqrt(P) - r)^2``
downwards, ``r`` the largest field amplitude it may add (``LineFate.faint_root``). It is an accounting omission of the
model's value only while that bound at the sum stays within the lines' negligible level (W1); above it the sum is an
estimate on the interval it leaves. Where the sum has no value (an open phase) or is not the model's value (a stand-in
coherence, an unstated phase), the grading reads no MODEL entry, so the interval itself counts that light (W28).

References: the closed-form bounds above, no Engine code. Where part of the lines is not followed, each omitted part of a
coherent sum states the same bound for its own field (``power.left_out_parts``, coherent).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from os_optics.config import NumericalConfig
from os_optics.contracts import Target
from os_optics.contracts.basis import Assumed, ValueBasis
from os_optics.errors.status import AssumedInput, ValueKind
from os_optics.validation import metrics as M
from os_optics.validation.power import LineFate

LEVEL = 1e-9


def _fate(root: float, faint_W: float) -> LineFate:
    return LineFate(input_W=1.0, unknown_W={}, below_floor_W=0.0, floor_significant=False, faint_root=root, faint_W=faint_W,
                    level_W=LEVEL)


def _graded(mv: M.MetricValue, maximum: float, monkeypatch) -> object:
    target = Target(target_id="t", metric="coherent_power_W", observation_id="P", maximum=maximum, unit="W")
    monkeypatch.setattr(M, "_measure", lambda *args, **kwargs: mv)
    return M._evaluate(target, {}, [], None, None, NumericalConfig())


def _upper(high: float, root: float) -> float:
    return (math.sqrt(high) + root) ** 2


def test_an_open_phase_sum_with_no_value_counts_the_left_out_field_in_its_interval(monkeypatch):
    """Open phase: no value, the traced sum in [1e-7, 1e-6] W; a field of at most 1e-12 sqrt(W) (bound at the sum 2e-15 W,
    within the level: accounting). A maximum 1e-15 W above the traced upper end lies inside the widened interval."""
    root = 1e-12
    out = M._untraced_interval(M.MetricValue(None, "W", untraced=(1e-7, 1e-6)), _fate(root, 5e-10), None, coherent=True)
    assert out.value is None and out.untraced == pytest.approx(((math.sqrt(1e-7) - root) ** 2, _upper(1e-6, root)), rel=1e-15)
    assert out.untraced[1] > 1e-6
    target = _graded(out, 1e-6 + 1e-15, monkeypatch)
    assert (target.status.value, target.assessment.value) == ("UNKNOWN", "INDETERMINATE")
    assert _graded(out, 2e-6, monkeypatch).assessment.value == "MET"


def test_a_stand_in_coherence_sum_counts_the_left_out_field_in_its_interval(monkeypatch):
    """A sum at full coherence (stand-in, ESTIMATE, graded on [0, value]): the left-out field widens the upper end."""
    root = 1e-12
    stand_in = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(Assumed(input=AssumedInput.FULL_COHERENCE, at=("source:S",), value=1.0),))
    mv = M.MetricValue(0.9, "W", untraced=(0.0, 0.9), basis=stand_in)
    out = M._untraced_interval(mv, _fate(root, 5e-10), None, coherent=True)
    assert out.untraced[1] == pytest.approx(_upper(0.9, root), rel=1e-15) and out.untraced[1] > 0.9
    assert out.basis.kind is ValueKind.ESTIMATE
    target = _graded(out, 0.9 + 1e-12, monkeypatch)
    assert target.assessment.value == "INDETERMINATE"


def test_a_model_value_keeps_its_interval_and_the_grading_reads_the_entry(monkeypatch):
    """A MODEL value with a banded traced sum: the interval is not widened (the grading adds the entry's bound, so counting
    it twice is avoided), the entry is MODEL with BELOW_NEGLIGIBLE_LEVEL."""
    root = 1e-12
    out = M._untraced_interval(M.MetricValue(0.5, "W", untraced=(0.4, 0.6)), _fate(root, 5e-10), None, coherent=True)
    assert out.untraced == (0.4, 0.6)
    (part,) = out.basis.omitted
    assert out.basis.kind is ValueKind.MODEL and part.part.value == "BELOW_NEGLIGIBLE_LEVEL"
    assert part.upper == pytest.approx(_upper(0.6, root) - 0.6, rel=1e-9)


def test_the_bound_at_the_sum_above_the_level_makes_it_an_estimate_even_when_the_power_is_within_it():
    """W1 reads the bound at the quantity: the light's power 5e-10 W is within the level 1e-9 W, but its field of 1e-5
    sqrt(W) may move a 1 W sum by 2e-5 W -- 20,000 times the level: no accounting omission, an estimate on the interval. A
    field of 1e-12 sqrt(W) moves it by 2e-12 W, within the level: the model's value with its stated truncation."""
    big = M._untraced_interval(M.MetricValue(1.0, "W"), _fate(1e-5, 5e-10), None, coherent=True)
    assert big.basis.kind is ValueKind.ESTIMATE and [o.part.value for o in big.basis.omitted] == ["POSSIBLE_RETURN"]
    assert big.untraced == pytest.approx(((1.0 - 1e-5) ** 2, (1.0 + 1e-5) ** 2), rel=1e-15)
    # the estimate's entry carries the bound the grading uses: the rise (1 + r)^2 - 1, which exceeds the fall
    (returned,) = big.basis.omitted
    assert returned.upper == pytest.approx((1.0 + 1e-5) ** 2 - 1.0, rel=1e-12)
    assert returned.upper == pytest.approx(max(big.untraced[1] - 1.0, 1.0 - big.untraced[0]), rel=1e-15)
    small = M._untraced_interval(M.MetricValue(1.0, "W"), _fate(1e-12, 5e-10), None, coherent=True)
    (part,) = small.basis.omitted
    assert small.basis.kind is ValueKind.MODEL and part.part.value == "BELOW_NEGLIGIBLE_LEVEL" and small.untraced is None
    assert part.upper == pytest.approx(_upper(1.0, 1e-12) - 1.0, rel=1e-3) and part.upper <= LEVEL


def test_a_dark_plane_coherent_sum_whose_left_out_field_bound_is_above_the_level_has_no_value(monkeypatch):
    """No traced beam reaches the plane (P = 0). The light left out as negligible brings at most 5e-10 W, within the level
    1e-9 W, but its field of 1e-4 sqrt(W) bounds the sum at r^2 = 1e-8 W, ten times the level (several faint pieces add their
    amplitudes: r^2 >= their power). W1 reads the bound at the quantity: no 0 W with an accounting omission, but no value and
    the interval [0, r^2], as on a lit plane (``_untraced_interval`` with P = 0). A field of 1e-5 sqrt(W) (r^2 = 1e-10 W,
    within the level) keeps the known 0 W with its stated truncation."""
    from os_optics.errors import Status

    fates = {"big": _fate(1e-4, 5e-10), "small": _fate(1e-5, 5e-11)}
    use = {"fate": fates["big"]}
    for name in ("_fate", "_dark_fate"):
        monkeypatch.setattr(M, name, lambda *args, **kwargs: use["fate"])
    monkeypatch.setattr(M, "_selection_problem", lambda *args, **kwargs: None)
    target = Target(target_id="t", metric="coherent_power_W", observation_id="P", maximum=5e-9, unit="W")
    big = fates["big"]
    assert big.complete and big.faint_accounted and big.faint_root ** 2 > LEVEL
    out = M._no_arrival(target, "W", None, None, NumericalConfig(), pair=False, what="beam", known_zero=True)
    assert out.value is None and out.basis is None and out.status is Status.PASS
    assert out.untraced == pytest.approx((0.0, 1e-8), rel=1e-12)
    lit = M._untraced_interval(M.MetricValue(1e-30, "W"), big, None, coherent=True)
    assert lit.basis.kind is ValueKind.ESTIMATE and lit.untraced[1] == pytest.approx(out.untraced[1], rel=1e-9)
    use["fate"] = fates["small"]
    kept = M._no_arrival(target, "W", None, None, NumericalConfig(), pair=False, what="beam", known_zero=True)
    (part,) = kept.basis.omitted
    assert kept.value == 0.0 and kept.basis.kind is ValueKind.MODEL and part.part.value == "BELOW_NEGLIGIBLE_LEVEL"
    assert part.upper == pytest.approx(1e-10, rel=1e-12)


def test_a_banded_estimate_carries_the_largest_move_of_its_value_inside_the_widened_interval():
    """A value 0.5 W inside its band [0.4, 0.6] W and a field of 1e-5 sqrt(W) (bound at the sum above the level): the
    entry's bound is the larger distance from the value to an end of the widened interval, here to the upper end; where
    nothing bounds that light's field the entry states no bound."""
    root = 1e-5
    out = M._untraced_interval(M.MetricValue(0.5, "W", untraced=(0.4, 0.6)), _fate(root, 5e-10), None, coherent=True)
    (part,) = out.basis.omitted
    assert out.basis.kind is ValueKind.ESTIMATE and part.part.value == "POSSIBLE_RETURN"
    assert out.untraced == pytest.approx(((math.sqrt(0.4) - root) ** 2, _upper(0.6, root)), rel=1e-15)
    assert part.upper == pytest.approx(_upper(0.6, root) - 0.5, rel=1e-12)
    unbounded = M._untraced_interval(M.MetricValue(1.0, "W"), _fate(math.inf, math.inf), None, coherent=True)
    assert [o.upper for o in unbounded.basis.omitted] == [None]


# --- a coherent sum where part of its lines was not followed: each omitted part's bound is the most its field moves the sum


def _part_fate(unknown: dict[str, float], root_sum: float, *, faint_root: float = 0.0, faint_W: float = 0.0,
               level: float = LEVEL, floor: bool = False) -> LineFate:
    return LineFate(input_W=1.0, unknown_W=unknown, below_floor_W=1e-3 if floor else 0.0, floor_significant=floor,
                    root_sum=root_sum, faint_root=faint_root, faint_W=faint_W, level_W=level)


def _moves(value: float, root: float) -> float:
    """The closed form: the most a field of amplitude ``root`` moves a coherent sum ``value`` (up or down)."""
    return max((math.sqrt(value) + root) ** 2 - value, value - max(0.0, math.sqrt(value) - root) ** 2)


def test_a_followed_part_of_a_coherent_sum_states_how_far_its_field_moves_the_sum_not_its_power():
    """P = 1 W, one followed contribution of 0.01 W (amplitude 0.1 sqrt(W)): it moves the sum by (1 + 0.1)^2 - 1 = 0.21 W,
    not by its 0.01 W. Two contributions of 0.01 W of one kind add their amplitudes, 0.2 sqrt(W): (1.2)^2 - 1 = 0.44 W, more
    than the square root of their summed power gives ((1 + sqrt(0.02))^2 - 1 = 0.303 W)."""
    from os_optics.optics.rays.tracer import SINK_UNTRACED_REFLECTION

    one = M._untraced_interval(M.MetricValue(1.0, "W"), _part_fate({SINK_UNTRACED_REFLECTION: 0.01}, 0.1), None, coherent=True)
    assert one.untraced == pytest.approx((0.81, 1.21), rel=1e-12)
    (part,) = one.basis.omitted
    assert one.basis.kind is ValueKind.ESTIMATE and part.part.value == "NOT_FOLLOWED"
    assert part.upper == pytest.approx(0.21, rel=1e-12)
    two = M._untraced_interval(M.MetricValue(1.0, "W"), _part_fate({SINK_UNTRACED_REFLECTION: 0.02}, 0.2), None, coherent=True)
    (part,) = two.basis.omitted
    assert part.upper == pytest.approx(0.44, rel=1e-12) and part.upper > (1.0 + math.sqrt(0.02)) ** 2 - 1.0
    assert part.upper == pytest.approx(max(two.untraced[1] - 1.0, 1.0 - two.untraced[0]), rel=1e-12)


def test_the_parts_of_a_coherent_sum_of_two_crossing_lasers_state_the_move_at_the_sum_and_the_faint_part_is_a_return():
    """The numbers of two 1 mW, 633 nm lasers of one coherence group crossing on an absorbing plane, one through a 3 mm
    N-BK7 window: the traced sum 3.6624e-3 W; followed untraced light of 3.235e-6 W whose amplitudes add up to
    sqrt(7.035e-6 W); left-out light of 4.37e-13 W (within the level 2e-12 W as a power) with a field of sqrt(4.37e-13 W).
    The followed light moves the sum by (sqrt(P) + sqrt(7.035e-6))^2 - P ~ 3.28e-4 W, not 3.235e-6 W; the left-out field
    moves it by ~2 sqrt(P) r ~ 8e-8 W, about 4e4 times the level: its possible return (W1), not an accounting part."""
    from os_optics.optics.rays.tracer import SINK_UNTRACED_REFLECTION

    P, root, faint = 3.6624e-3, math.sqrt(7.035e-6), math.sqrt(4.37e-13)
    fate = _part_fate({SINK_UNTRACED_REFLECTION: 3.235e-6}, root, faint_root=faint, faint_W=4.37e-13, level=2e-12)
    assert fate.faint_accounted  # (the power gate alone would call it negligible)
    absorbing = SimpleNamespace(observation_id="P", plane=SimpleNamespace(absorbing=True))
    out = M._untraced_interval(M.MetricValue(P, "W"), fate, absorbing, coherent=True)
    followed, left_out = out.basis.omitted
    assert out.basis.kind is ValueKind.ESTIMATE
    assert followed.part.value == "NOT_FOLLOWED" and followed.upper >= 3.28e-4
    assert followed.upper == pytest.approx(_moves(P, root), rel=1e-12)
    assert left_out.part.value == "POSSIBLE_RETURN" and left_out.upper >= 8e-8
    assert left_out.upper == pytest.approx(_moves(P, faint), rel=1e-9) and left_out.upper > 1e4 * fate.level_W
    # (each part's bound lies within the interval the sum is graded on: amplitudes add)
    assert max(followed.upper, left_out.upper) <= max(out.untraced[1] - P, P - out.untraced[0])


def test_light_of_unknown_fate_bounds_no_part_of_a_coherent_sum_on_an_absorbing_plane():
    """A budget stopped some light (unknown fate) and the power floor holds a significant remainder: either may arrive in any
    number of beams, so nothing bounds the coherent sum from above -- on an absorbing plane too (absorption bounds a power,
    not how many beams interfere). The followed part keeps the bound of its own field."""
    from os_optics.optics.rays.tracer import SINK_PATH_LIMIT, SINK_UNTRACED_REFLECTION

    fate = _part_fate({SINK_PATH_LIMIT: 0.01, SINK_UNTRACED_REFLECTION: 0.01}, 0.1, floor=True)
    assert fate.unbounded
    absorbing = SimpleNamespace(observation_id="P", plane=SimpleNamespace(absorbing=True))
    out = M._untraced_interval(M.MetricValue(1.0, "W"), fate, absorbing, coherent=True)
    assert out.untraced == (0.0, math.inf)
    uppers = {o.part.value: o.upper for o in out.basis.omitted}
    assert uppers == {"NOT_FOLLOWED": pytest.approx(0.21, rel=1e-12), "BUDGET": None, "BELOW_POWER_FLOOR": None}
    # (the power of the same light on the same plane keeps its absorbed-once bounds)
    power = M._untraced_interval(M.MetricValue(1.0, "W"), fate, absorbing, coherent=False)
    assert {o.part.value: o.upper for o in power.basis.omitted} == {"BUDGET": 0.01, "NOT_FOLLOWED": 0.01, "BELOW_POWER_FLOOR": 1e-3}


@pytest.mark.parametrize("port", ["port_x", "port_y"])
def test_the_window_ghosts_of_the_mach_zehnder_bound_a_coherent_port_power_by_their_field(port, monkeypatch):
    """``mzi_window`` with a coherent_power_W target on each port: the window's untraced reflections (4.15e-7 W, several
    separate contributions) may interfere with the traced light. Their entry states how far their field moves the sum,
    from the amplitudes of all their contributions together (sqrt(4.517e-7 W) > sqrt(4.15e-7 W)), not their power."""
    from fixture_scenes import mzi_window

    from os_optics import validate
    from os_optics.optics.rays.tracer import SINK_UNTRACED_REFLECTION

    seen = {}
    original = M._untraced_interval

    def spy(mv, fate, obs, *, coherent):
        if coherent:
            seen[obs.observation_id] = (mv.value, fate)
        return original(mv, fate, obs, coherent=coherent)

    monkeypatch.setattr(M, "_untraced_interval", spy)
    request = mzi_window()
    target = Target(target_id="coh", metric="coherent_power_W", observation_id=port, maximum=10.0, unit="W")
    request = request.model_copy(update={"project": request.project.model_copy(update={"targets": (target,)})})
    (result,) = validate(request).targets
    value, fate = seen[port]
    power = fate.unknown_W[SINK_UNTRACED_REFLECTION]
    assert result.value == value and power == pytest.approx(4.15e-7, rel=1e-3)
    assert fate.root_sum ** 2 >= power  # (separate contributions add their amplitudes)
    entry = result.value_basis["value"]
    followed = next(o for o in entry.omitted if o.part.value == "NOT_FOLLOWED")
    assert entry.kind is ValueKind.ESTIMATE and followed.code == "UNTRACED_LIGHT_BOUNDED"
    assert followed.upper == pytest.approx(_moves(value, fate.root_sum), rel=1e-12)
    assert followed.upper >= _moves(value, math.sqrt(power)) and followed.upper > 25.0 * power
