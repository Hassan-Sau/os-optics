"""PV-F1: the arithmetic behind a run's files has the same bits in every runtime (native CPython / NumPy and Pyodide).

A native NumPy fuses products and sums (fused multiply-add) in ``einsum``, ``matmul`` and complex multiplication on a CPU
with FMA; WebAssembly does not, and libm differs between the two. The geometry therefore uses ``math.vectors.dot`` /
``matvec`` / ``matmul`` (separately rounded products summed in index order), and wrapped phases are reported at the
resolution the optical path length determines (``math.phase``). The byte-for-byte comparison with Pyodide itself is in
``test_packaging_and_pyodide.py``; these tests pin the building blocks and keep the hazards out of the modules that need
them.

Not on the list, by design: the on-demand analyses that read a run's files and are not part of them, such as the field
analysis of ``results/field.py`` (its transforms, ``@`` and least-squares fit of the rays' aberration). Their documents
agree between runtimes to rounding (the Pyodide comparison reads them with a relative tolerance of 1e-9), not bit for bit.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from os_optics.math.phase import PHASE_RESOLUTION_RAD, reported_phase
from os_optics.math.vectors import dot, matmul, matvec

SRC = Path(__file__).resolve().parents[1] / "src" / "os_optics"

#: Modules whose results reach a run's files through the traced geometry, the Gaussian / polarization states, the
#: observations, exposures and the pair metrics: no einsum, no ``@``, no LAPACK and no NumPy inverse trigonometry there.
DETERMINISTIC_MODULES = (
    "math/vectors.py", "math/pose.py", "math/rotation.py", "scene/lower.py", "optics/surfaces/geometry.py", "optics/surfaces/sag.py",
    "optics/surfaces/interaction.py", "optics/rays/tracer.py", "optics/rays/decisions.py", "optics/rays/lower_share.py", "optics/rays/pupil.py",
    "optics/gaussian/ops.py", "optics/rays/gradient.py",
    "optics/gaussian/state.py", "optics/gaussian/truncation.py", "optics/interference/two_beam.py", "optics/attenuation.py",
    "validation/observations.py", "validation/superposition.py", "validation/exposures.py", "validation/rating_conditions.py",
    "math/exact_erf.py",
    # the left-out light's bounds and the coherence from the group delay (facing pairs, envelope spread, group index,
    # coatings' angle ranges, probes, plane power): their numbers reach a run's files too
    "validation/resonance.py", "validation/coating_use.py", "optics/rays/envelope.py", "optics/interference/coherence.py", "optics/materials/group.py",
    "optics/materials/library.py", "optics/materials/coatings.py", "optics/rays/hits.py", "validation/untraced.py",
    "validation/pipeline.py", "validation/metrics.py", "validation/power.py", "validation/window_bound.py",
    # partially polarized sources (their components and Stokes sums) and the time-domain readings
    "optics/polarization/partial.py", "validation/polarization_mixture.py", "validation/time_delay.py",
    # a retained DOF's stated limits against a target (travel margin, drift over a span): target values of a run
    "validation/mechanical.py",
    # aperture regions: membership, extents and the Gaussian power on an ellipse, a D shape, a polygon or through holes
    "optics/apertures.py", "optics/gaussian/aperture_regions.py",
    # the return-light analysis' closed-return screen and its work count, and the internal returns of plane-faced solids
    "validation/closed_returns.py", "validation/return_work.py", "validation/pair_selection.py", "validation/solid_folds.py",
    # the cavity analysis (its eigenmode, Gouy phase, finesse and linewidth) and the build-up of injected light
    "local/cavity.py", "local/cavity_build_up.py",
    # uniaxial crystals: their waves, power split, optical and group paths
    "optics/polarization/uniaxial.py",
    # the comparison of a Project's configurations (values copied from each configuration's run, identity hashes)
    "analysis/configurations.py", "local/configurations.py",
    # the ghost trace's ledger of the light it does not follow (its powers and their bounds)
    "results/ghost_trace.py",
    # scattered light: the scatter events, their rays and their results
    "optics/rays/scatter.py", "optics/rays/scatter_trace.py", "validation/scattered_light.py",
    # continuous spectra: the quadrature lines (an Engine input) and the coherence reported from the spectrum
    "optics/interference/spectrum.py", "optics/interference/source_coherence.py",
    # a thin complex-transmission sample: the tracer's power factor, the optical path it adds and the ray turn
    "optics/complex_transmission.py",
    # the imaging analysis' launch grid and the ghost-image analysis' path rows (values of their documents)
    "results/imaging_trace.py", "results/ghosts.py",
    # second-harmonic generation: the d tensor contracted with the traced fields, the overlap integral, the line average
    "optics/nonlinear/shg.py",
    # sampled sources: the emitter's samples (positions, directions, their placement in the design), the patterns' positions
    # and phase gradients, and the sampled trace of their rays
    "optics/rays/emitter_samples.py", "optics/fields/source_fields.py", "optics/rays/sampled.py",
)
#: Modules whose arithmetic is also free of library transcendental functions (``math`` / NumPy ``exp``, ``log``, ``cos``,
#: ``sin``, ``atan``, ``hypot``, ``pow``, ...) and of non-integer powers: they use ``math.exact_trig``.
LIBM_FREE_MODULES = (
    "validation/resonance.py", "validation/coating_use.py", "optics/rays/envelope.py", "optics/interference/coherence.py", "optics/materials/group.py",
    "optics/rays/hits.py", "validation/untraced.py", "validation/pipeline.py", "validation/metrics.py", "validation/power.py",
    "validation/exposures.py", "validation/rating_conditions.py", "validation/window_bound.py", "validation/mechanical.py",
    "optics/polarization/partial.py", "validation/polarization_mixture.py", "validation/time_delay.py", "optics/gaussian/aperture_regions.py",
    "validation/closed_returns.py", "validation/return_work.py", "validation/pair_selection.py", "validation/solid_folds.py", "local/cavity.py",
    "local/cavity_build_up.py", "optics/polarization/uniaxial.py", "optics/rays/gradient.py", "optics/rays/lower_share.py",
    "analysis/configurations.py", "local/configurations.py",
    "results/ghost_trace.py",
    "optics/rays/scatter.py", "optics/rays/scatter_trace.py", "validation/scattered_light.py",
    "optics/interference/spectrum.py", "optics/interference/source_coherence.py",
    "optics/complex_transmission.py", "results/imaging_trace.py", "results/ghosts.py", "optics/nonlinear/shg.py",
)
_LIBM = frozenset({"exp", "expm1", "log", "log1p", "log2", "log10", "pow", "power", "float_power", "cos", "sin", "tan", "atan",
                   "atan2", "acos", "asin", "arccos", "arcsin", "arctan", "arctan2", "hypot", "erf", "erfc", "cosh", "sinh",
                   "tanh", "cbrt", "exp2"})


def _sequential(a: list[float], b: list[float]) -> float:
    total = a[0] * b[0]
    for x, y in zip(a[1:], b[1:]):
        total = total + x * y
    return total


def test_dot_is_the_index_order_sum_of_separately_rounded_products():
    rng = np.random.default_rng(7)
    a, b = rng.standard_normal((500, 3)), rng.standard_normal((500, 3))
    want = [_sequential(x, y) for x, y in zip(a.tolist(), b.tolist())]
    assert dot(a, b).tolist() == want
    assert dot(a, b[0]).tolist() == [_sequential(x, b[0].tolist()) for x in a.tolist()]  # broadcast against one vector
    assert float(dot(a[0], b[0])) == want[0]
    planes = rng.standard_normal((4, 3))
    table = dot(a[:, None, :], planes[None, :, :])  # (rays, planes), as the plane batch of the tracer
    assert table.tolist() == [[_sequential(x, p) for p in planes.tolist()] for x in a.tolist()]


def test_matvec_and_matmul_are_rows_of_dots():
    rng = np.random.default_rng(8)
    m, v = rng.standard_normal((50, 3, 3)), rng.standard_normal((50, 3))
    assert matvec(m, v).tolist() == [[_sequential(row, x) for row in mm] for mm, x in zip(m.tolist(), v.tolist())]
    single = rng.standard_normal((3, 3))
    assert matvec(single, v).tolist() == [[_sequential(row, x) for row in single.tolist()] for x in v.tolist()]
    other = rng.standard_normal((3, 3))
    assert matmul(single, other).tolist() == [[_sequential(row, col) for col in other.T.tolist()] for row in single.tolist()]


def test_the_d_tensor_contraction_is_rows_of_dots_in_a_turned_crystal_frame():
    """``shg.d_eff_from_tensor`` turns the fields into the crystal frame with ``matvec`` (a NumPy ``frame @ v`` differs from it in
    the last bit for most turned frames natively, and differently in Pyodide) and sums the contraction in index order."""
    from os_optics.math.rotation import quat_normalize, quat_to_matrix
    from os_optics.optics.nonlinear.shg import d_eff_from_tensor

    rng = np.random.default_rng(11)
    tensor = tuple(tuple(float(x) for x in row) for row in rng.standard_normal((3, 6)))
    for _ in range(200):
        frame = quat_to_matrix(quat_normalize(rng.standard_normal(4)))
        e_h, e_a, e_b = rng.standard_normal((3, 3))
        h, a, b = ([_sequential(row, v.tolist()) for row in frame.tolist()] for v in (e_h, e_a, e_b))
        pairs = (a[0] * b[0], a[1] * b[1], a[2] * b[2], a[1] * b[2] + a[2] * b[1], a[0] * b[2] + a[2] * b[0], a[0] * b[1] + a[1] * b[0])
        want = 0.0
        for i in range(3):
            row = tensor[i]
            want += h[i] * (row[0] * pairs[0] + row[1] * pairs[1] + row[2] * pairs[2] + row[3] * pairs[3] + row[4] * pairs[4]
                            + row[5] * pairs[5])
        assert d_eff_from_tensor(tensor, frame, e_h, e_a, e_b) == want


@pytest.mark.parametrize(("phase", "reported"), [(0.6638685743562753, 0.663869), (0.6638685748219366, 0.663869),
                                                 (-1.1017907883054257, -1.101791), (-1.101790788072595, -1.101791),
                                                 (math.pi - 1e-12, 3.141593), (-math.pi, 3.141593),
                                                 (-math.pi + 1e-7, 3.141593), (-math.pi + 2e-7, -3.141592), (0.0, 0.0), (-1e-9, 0.0)])
def test_wrapped_phases_are_reported_at_their_resolution(phase, reported):
    """The two PV-F1 pairs (wedge prism, plain beam 3 deg off axis) that differed in the last bits before the fix report
    the same value; -0.0 is reported as 0.0; the reported interval is (-pi, pi]: -pi and a value within half a step of it
    read +3.141593 (a recorded change of convention: -pi read -3.141593 before)."""

    value = reported_phase(phase)
    assert value == reported and math.copysign(1.0, value) == math.copysign(1.0, reported or 1.0)
    assert PHASE_RESOLUTION_RAD == 1e-6
    assert round(value / PHASE_RESOLUTION_RAD) * PHASE_RESOLUTION_RAD == pytest.approx(value, abs=1e-15)


def test_a_runs_phases_are_on_the_reporting_grid():
    from os_optics.local import example_project_of_kind, run_optical_project

    run = run_optical_project(example_project_of_kind("two_beam_interference"))
    phases = [a.beam.phase_rad for o in run.result.observations for a in o.arrivals if a.beam.phase_rad is not None]
    phases += [i.phase_difference_rad for i in run.result.interference if i.phase_difference_rad is not None]
    assert phases
    for phase in phases:
        assert phase == reported_phase(phase), phase


@pytest.mark.parametrize("module", DETERMINISTIC_MODULES)
def test_the_traced_geometry_uses_no_fused_or_library_dependent_kernels(module):
    tree = ast.parse((SRC / module).read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            found.append(f"line {node.lineno}: '@'")
        if isinstance(node, ast.Attribute) and node.attr == "einsum":
            found.append(f"line {node.lineno}: einsum")
        if isinstance(node, ast.Attribute) and node.attr == "linalg":
            found.append(f"line {node.lineno}: numpy.linalg")
        if (isinstance(node, ast.Attribute) and node.attr in ("arccos", "arcsin", "arctan", "arctan2", "hypot")
                and isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy")):
            found.append(f"line {node.lineno}: numpy.{node.attr}")
    assert found == [], f"{module}: use math.vectors.dot / matvec / matmul or math.exact_linalg instead: {found}"


def _integer_power(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool)


@pytest.mark.parametrize("module", LIBM_FREE_MODULES)
def test_the_left_out_light_and_coherence_bounds_use_no_library_transcendentals(module):
    """libm (CPython's, Pyodide's) may round ``exp``, ``log``, ``cos``, ``atan2``, ``pow``, ... differently in the last place;
    these modules use the libm-free ``math.exact_trig`` and only integer powers."""
    tree = ast.parse((SRC / module).read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr in _LIBM and isinstance(node.value, ast.Name)
                and node.value.id in ("math", "np", "numpy", "cmath")):
            found.append(f"line {node.lineno}: {node.value.id}.{node.attr}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and not _integer_power(node.right):
            found.append(f"line {node.lineno}: '**' {ast.unparse(node)[:40]}")
    assert found == [], f"{module}: use math.exact_trig instead: {found}"
