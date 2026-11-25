"""Image orientation as a computed design property: straight, inverted and folded relays with odd and even mirror
counts, and a design changed to restore the requested orientation. References: thin-lens imaging (M = -s'/s, a two-lens
4f relay M = -f2/f1) and reflection parity (each reflection mirrors the image)."""

from __future__ import annotations

import math

import pytest

from os_optics.builders import laser, mirror_for_reflection, observation_plane, thin_lens
from os_optics.contracts import ProjectSnapshot, Target

from helpers import one_arrival, run

X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
F1, F2 = 50.0, 100.0


def _source():
    return laser("L", wavelength_nm=633.0, waist_radius_mm=0.2)


def _relay_4f(*, folds: int):
    """Object (source plane) at x = 0, L1 (f1) at f1, L2 (f2) at 2 f1 + f2, image at 2 (f1 + f2) along the unfolded axis.
    ``folds`` 45 deg mirrors placed after L2 turn the beam in the horizontal plane (+X -> +Y -> -X ...)."""
    comps = [_source(), thin_lens("L1", focal_length_mm=F1, clear_radius_mm=10.0, position=(F1, 0.0, 0.0)),
             thin_lens("L2", focal_length_mm=F2, clear_radius_mm=10.0, position=(2 * F1 + F2, 0.0, 0.0))]
    position, direction, remaining = [2 * F1 + F2 + 40.0, 0.0, 0.0], X, F2 - 40.0
    turns = (Y, (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), X)
    for k in range(folds):
        new = turns[k]
        comps.append(mirror_for_reflection(f"M{k}", position=tuple(position), d_in=direction, d_out=new, clear_radius_mm=10.0))
        step = remaining / (folds - k + 1) if k < folds - 1 else remaining
        if k < folds - 1:
            position = [position[i] + new[i] * step for i in range(3)]
            remaining -= step
        direction = new
    if folds:
        position = [position[i] + direction[i] * remaining for i in range(3)]
    else:
        position = [2 * (F1 + F2), 0.0, 0.0]
    # bounded: with two folds the image plane lies in the plane of L2 (x = 2 f1 + f2), 20 mm beside it, and an unbounded
    # plane would also sample the beam where it crosses L2
    image = observation_plane("IMG", position=tuple(position), normal=direction, radius_mm=10.0)
    return ProjectSnapshot(components=tuple(comps), observations=(image,))


def test_two_lens_relay_inverts_with_magnification_f2_over_f1():
    a = one_arrival(run(_relay_4f(folds=0)), "IMG")
    fo = a.first_order
    assert fo.handedness_preserved and abs(abs(fo.image_rotation_deg) - 180.0) < 1e-4
    assert fo.transverse_magnification == pytest.approx(F2 / F1, rel=1e-6)
    assert abs(fo.matrix[0][2]) < 1e-4 and abs(fo.matrix[1][3]) < 1e-4  # B ~ 0: the plane is conjugate to the source plane


@pytest.mark.parametrize("folds", [1, 3])
def test_an_odd_number_of_folds_mirrors_the_image(folds):
    fo = one_arrival(run(_relay_4f(folds=folds)), "IMG").first_order
    assert fo.reflection_count == folds and not fo.handedness_preserved and fo.image_rotation_deg is None
    assert fo.mirror_axis_deg is not None and fo.transverse_magnification == pytest.approx(F2 / F1, rel=1e-6)


def test_an_even_number_of_folds_keeps_the_handedness_and_reports_the_rotation():
    fo = one_arrival(run(_relay_4f(folds=2)), "IMG").first_order
    assert fo.reflection_count == 2 and fo.handedness_preserved
    assert fo.image_rotation_deg is not None and fo.transverse_magnification == pytest.approx(F2 / F1, rel=1e-6)


def test_a_design_failing_an_upright_requirement_is_restored_by_a_second_relay():
    upright = (Target(target_id="parity", metric="image_parity", observation_id="IMG", target_value=1.0, tolerance_abs=0.5),
               Target(target_id="rotation", metric="image_rotation_deg", observation_id="IMG", target_value=0.0, tolerance_abs=0.5))
    first = run(_relay_4f(folds=0).model_copy(update={"targets": upright}))
    assert [t.status.value for t in first.targets] == ["PASS", "FAIL"]  # not mirrored, but inverted
    # restore: a second identical 4f relay re-inverts the image (total magnification (f2/f1)^2, upright)
    shift = 2 * (F1 + F2)
    relay = _relay_4f(folds=0)
    extra = (thin_lens("L3", focal_length_mm=F1, clear_radius_mm=10.0, position=(shift + F1, 0.0, 0.0)),
             thin_lens("L4", focal_length_mm=F2, clear_radius_mm=10.0, position=(shift + 2 * F1 + F2, 0.0, 0.0)))
    restored = relay.model_copy(update={"components": relay.components + extra, "targets": upright,
                                        "observations": (observation_plane("IMG", position=(2 * shift, 0.0, 0.0), normal=X),)})
    result = run(restored)
    assert [t.status.value for t in result.targets] == ["PASS", "PASS"]
    assert one_arrival(result, "IMG").first_order.transverse_magnification == pytest.approx((F2 / F1) ** 2, rel=1e-6)
    assert math.isfinite(result.targets[1].value)
