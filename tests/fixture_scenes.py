"""Fixture scenes: the single source of the JSON requests in tests/fixtures/requests (written by generate_fixtures.py).

Every documented example is one of these executable fixtures (no hand-written examples that cannot run)."""

from __future__ import annotations

import math

import numpy as np

from os_optics import EngineConfiguration, NumericalConfig, ValidationRequest
from os_optics import builders as B
from os_optics.contracts import (
    ActuatorSpec, BaseplateSnapshot, CaptureScanSpec, ControlSpec, EngineeringComponentSnapshot, JonesVector, MonteCarloSpec, Pose3D,
    ProjectSnapshot, RecoverabilitySpec, SensorSpec, Target,
)

from _frozen_poses import Q_45_DIAG, Q_45_NEG, Q_NEG_Y, pin_pose

INV = 1.0 / math.sqrt(2.0)

# --- Frozen derived-geometry poses ---------------------------------------------------------------
# These poses are produced by trig / scipy.optimize.brentq reconstructions (not the four canonical
# quaternion constants above) and are therefore libm-sensitive; pin them to the committed byte-exact
# values so the request JSON is platform-independent.
# tilted_mirror_astigmatism: the reflected direction d_out carries cos/sin(45 deg) float residue.
_Q_TILTED_OBS = (0.7071067811865476, 0.7071067811865475, 7.850462293418873e-17, 7.850462293418873e-17)
_POS_TILTED_OBS = (99.99999999999999, 60.000000000000014, 20.0)
# interference_405_500: MA/MB mirrors and the substrate plane are solved by brentq + trig.
_Q_MA = (0.2093657348555765, 0.2093657348555765, -0.6754006137607401, -0.6754006137607401)
_Q_MB = (0.6256242848722584, 0.6256242848722585, 0.3295364231432927, 0.3295364231432927)
_POS_MB = (0.0, 100.0, 20.0)
_Q_SUB = (0.6532814824381883, 0.6532814824381882, -0.2705980500730985, -0.2705980500730985)
_POS_SUB = (162.8786030547298, 162.8786030547298, 20.0)


def _req(project: ProjectSnapshot, *, seed: int | None = None, layout: bool = True, control: bool = True) -> ValidationRequest:
    return ValidationRequest(project=project, configuration=EngineConfiguration(random_seed=seed, run_layout=layout, run_control=control))


def interference_405_500() -> ValidationRequest:
    from test_golden_interference import PERIOD, THETA, _interference_project

    targets = (
        Target(target_id="period", metric="fringe_period_nm", observation_id="sub", target_value=PERIOD, tolerance_abs=1e-6, unit="nm"),
        Target(target_id="half_angle", metric="half_angle_deg", observation_id="sub", target_value=math.degrees(THETA), tolerance_abs=1e-9),
        Target(target_id="opd", metric="optical_path_difference_mm", observation_id="sub", target_value=0.0, tolerance_abs=1e-9),
        Target(target_id="visibility", metric="fringe_visibility", observation_id="sub", minimum=0.99),
        Target(target_id="coherence", metric="coherence_margin", observation_id="sub", minimum=0.9),
        Target(target_id="exposure", metric="estimated_exposure_time_s", observation_id="sub", minimum=0.0, parameters={"dose_mj_per_cm2": 100.0}),
    )
    proj, *_ = _interference_project(targets=targets)
    # Pin the brentq/trig-derived geometry (MA/MB mirror poses, substrate plane) to the committed frozen values so the
    # request JSON is platform-independent. This is done here (not inside _interference_project) because that helper is
    # shared with parameterized golden tests whose geometry legitimately varies with opd_mm / waist_mm.
    components = tuple(
        pin_pose(c, _Q_MA) if c.instance_id == "MA"
        else pin_pose(c, _Q_MB, position=_POS_MB) if c.instance_id == "MB"
        else c
        for c in proj.components
    )
    observations = tuple(
        pin_pose(o, _Q_SUB, position=_POS_SUB) if o.observation_id == "sub" else o
        for o in proj.observations
    )
    proj = proj.model_copy(update={"components": components, "observations": observations})
    return _req(proj.model_copy(update={"name": "405 nm / 500 nm two-beam interference lithography"}), layout=False, control=False)


def keplerian_expander() -> ValidationRequest:
    lam, w0, f1, f2, x1 = 532.0, 0.5, 25.0, 100.0, 20.0
    x2 = x1 + f1 + f2
    proj = ProjectSnapshot(
        name="Keplerian 4x beam expander",
        components=(
            B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=(0, 0, 20), direction=(1, 0, 0), waist_position_z_mm=x1),
            B.thin_lens("L1", focal_length_mm=f1, clear_radius_mm=12.7, position=(x1, 0, 20), axis=(1, 0, 0)),
            B.thin_lens("L2", focal_length_mm=f2, clear_radius_mm=12.7, position=(x2, 0, 20), axis=(1, 0, 0)),
        ),
        observations=(B.observation_plane("out", position=(x2 + 150.0, 0, 20), normal=(1, 0, 0), radius_mm=20.0),),
        targets=(Target(target_id="waist", metric="waist_radius_x_mm", observation_id="out", target_value=4 * w0, tolerance_rel=1e-6),
                 Target(target_id="divergence", metric="divergence_half_angle_x_rad", observation_id="out",
                        target_value=lam * 1e-6 / (math.pi * w0) / 4.0, tolerance_rel=1e-6)),
    )
    return _req(proj, layout=False, control=False)


def mzi_window() -> ValidationRequest:
    from test_golden_interference import _mzi_project

    proj = _mzi_project(0.4123)
    return _req(proj.model_copy(update={"name": "Mach-Zehnder with BK7 window in one arm"}), layout=False, control=False)


def hwp_pbs() -> ValidationRequest:
    from test_golden_polarization import _hwp_pbs_project

    from helpers import stated_splitter

    proj = _hwp_pbs_project(22.5)
    # The splitter states the lossless splitter's phases, as the input builder's ideal splitter profile does.
    proj = proj.model_copy(update={"components": tuple(stated_splitter(c) if c.instance_id == "PBS" else c for c in proj.components)})
    targets = (Target(target_id="pt", metric="power_W", observation_id="T", target_value=5e-3, tolerance_rel=1e-9),
               Target(target_id="pr", metric="power_W", observation_id="R", target_value=5e-3, tolerance_rel=1e-9))
    return _req(proj.model_copy(update={"name": "HWP at 22.5 deg + PBS: even split", "targets": targets}), layout=False, control=False)


def tilted_mirror_astigmatism() -> ValidationRequest:
    a = math.radians(45.0)
    normal = (-math.cos(a), math.sin(a), 0.0)
    vertex = np.array([100.0, 0.0, 20.0])
    d_in = np.array([1.0, 0.0, 0.0])
    n = np.array(normal)
    d_out = d_in - 2 * (d_in @ n) * n
    mirror = pin_pose(
        B.spherical_mirror("M", position=tuple(vertex), normal=normal, radius_of_curvature_mm=200.0, clear_radius_mm=10.0),
        Q_45_NEG,
    )
    detector = pin_pose(
        B.observation_plane("d", position=tuple(vertex + 60.0 * d_out), normal=tuple(-d_out), radius_mm=20.0),
        _Q_TILTED_OBS,
        position=_POS_TILTED_OBS,
    )
    proj = ProjectSnapshot(
        name="Spherical mirror at 45 deg: tangential / sagittal astigmatism",
        components=(B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0)), mirror),
        observations=(detector,),
    )
    return _req(proj, layout=False, control=False)


def aperture_clipping() -> ValidationRequest:
    lam, w0, z = 633.0, 1.0, 40.0
    w = w0 * math.sqrt(1 + (z / (math.pi * w0**2 / (lam * 1e-6))) ** 2)
    proj = ProjectSnapshot(
        name="Centred circular aperture at 1.5 w (2 % clipping warning)",
        components=(B.laser("L", wavelength_nm=lam, waist_radius_mm=w0, position=(0, 0, 20), direction=(1, 0, 0), power_W=4e-3),
                    B.aperture_stop("Stop", position=(z, 0, 20), axis=(1, 0, 0), radius_mm=1.5 * w)),
        observations=(B.observation_plane("after", position=(z + 30, 0, 20), normal=(1, 0, 0), radius_mm=50.0),),
        targets=(Target(target_id="p", metric="power_W", observation_id="after", target_value=4e-3 * (1 - math.exp(-2 * 1.5**2)),
                        tolerance_rel=1e-9),),
    )
    return _req(proj, layout=False, control=False)


def multi_laser() -> ValidationRequest:
    from test_golden_interference import _two_laser_project

    return _req(_two_laser_project("normal").model_copy(update={"name": "Three independent lasers through one lens"}), layout=False, control=False)


def out_of_scope_targets() -> ValidationRequest:
    proj = ProjectSnapshot(
        name="Unsupported wave-optics / high-NA requests",
        components=(B.laser("L", wavelength_nm=405.0, waist_radius_mm=0.0003, position=(0, 0, 20), direction=(1, 0, 0)),),
        observations=(B.observation_plane("d", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=20.0),),
        targets=(Target(target_id="fresnel", metric="fresnel_propagation", observation_id="d", minimum=0.0),
                 Target(target_id="fdtd", metric="fdtd", observation_id="d", minimum=0.0),
                 Target(target_id="highna", metric="beam_radius_x_mm", observation_id="d", minimum=0.0),
                 Target(target_id="unknown", metric="not_a_metric", observation_id="d", minimum=0.0)),
    )
    return _req(proj, layout=False, control=False)


def layout_example() -> ValidationRequest:
    plate = BaseplateSnapshot(outline_mm=((0, 0), (300, 0), (300, 200), (0, 200)), edge_margin_mm=5.0)
    laser = B.laser("L", wavelength_nm=532.0, waist_radius_mm=1.0, position=(20, 100, 20), direction=(1, 0, 0),
                    envelope=B.rect_envelope(30, 30, 40, optical=True, required_clearance_mm=2.0))
    lens = B.thin_lens("Lens", focal_length_mm=100.0, clear_radius_mm=12.7, position=(80, 100, 20), axis=(1, 0, 0),
                       envelope=B.rect_envelope(10, 40, 40, optical=True, required_clearance_mm=2.0))
    mirror = pin_pose(B.mirror("M", position=(200, 100, 20), normal=(-INV, INV, 0), clear_radius_mm=12.7,
                               envelope=B.rect_envelope(12, 40, 40, optical=True, required_clearance_mm=2.0)), Q_45_NEG)
    post = EngineeringComponentSnapshot(instance_id="Post", role="post", pose=Pose3D(position_mm=(140.0, 104.0, 0.0)),
                                        envelope=B.rect_envelope(10, 10, 40))  # passive post, deliberately close to the beam
    proj = ProjectSnapshot(
        name="Layout with a keepout violation and a beam-blocking post", baseplate=plate,
        components=(laser, lens, mirror, post),
        observations=(pin_pose(B.observation_plane("d", position=(200, 200, 20), normal=(0, -1, 0), radius_mm=15.0), Q_NEG_Y),),
    )
    return _req(proj, layout=True, control=False)


def control_recoverability() -> ValidationRequest:
    yaw = ActuatorSpec(actuator_id="yaw", dof="rotate", axis_local=(0.0, 1.0, 0.0), travel_min=-3.0, travel_max=3.0)
    pitch = ActuatorSpec(actuator_id="pitch", dof="rotate", axis_local=(1.0, 0.0, 0.0), travel_min=-3.0, travel_max=3.0)
    mirror = pin_pose(B.mirror("M", position=(100, 0, 20), normal=(-INV, INV, 0), clear_radius_mm=25.0, actuators=(yaw, pitch)), Q_45_NEG)
    spec = ControlSpec(
        actuator_refs=("M/yaw", "M/pitch"),
        sensors=(SensorSpec(sensor_id="u", observation_id="det", quantity="centroid_u_mm", target_value=0.0),
                 SensorSpec(sensor_id="v", observation_id="det", quantity="centroid_v_mm", target_value=0.0)),
        capture_scan=CaptureScanSpec(actuator_refs=("M/yaw", "M/pitch"), points_per_axis=9),
        recoverability=RecoverabilitySpec(disturbances=({"M/yaw": 0.4, "M/pitch": -0.3}, {"M/yaw": 2.9}), convergence_tolerance=1e-4),
        monte_carlo=MonteCarloSpec(sample_count=6, seed=2026, amplitude={"M/yaw": 2.0, "M/pitch": 2.0}),
    )
    proj = ProjectSnapshot(
        name="Tip/tilt mirror alignment: sensitivity, recoverability, Monte Carlo",
        components=(B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0)), mirror),
        observations=(pin_pose(B.observation_plane("det", position=(100, 200, 20), normal=(0, -1, 0), radius_mm=5.0), Q_NEG_Y),),
        control_spec=spec,
    )
    return _req(proj, seed=2026, layout=False, control=True)


def benchmark_standard() -> ValidationRequest:
    """The standard reference workload: <=100 component instances, <=20 optical paths, <=500 deterministic pupil samples, <=30 targets."""
    splitters = 19
    laser = B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(-30, 0, 20), direction=(1, 0, 0), power_W=100e-3,
                    envelope=B.rect_envelope(30, 30, 40, optical=True))
    laser = laser.model_copy(update={"source": laser.source.model_copy(update={"pupil_ring_count": 20, "pupil_points_per_ring": 25})})  # 500 rays
    comps = [laser]
    obs = [B.observation_plane("main", position=(30 * splitters + 60, 0, 20), normal=(1, 0, 0), radius_mm=20.0)]
    targets = []
    reflectance = 0.05
    power = 100e-3
    for k in range(splitters):
        x = 30.0 * (k + 1)
        comps.append(pin_pose(B.npbs(f"BS{k}", position=(x, 0, 20), normal=(-INV, INV, 0), reflectance=reflectance, clear_radius_mm=6.0,
                                     envelope=B.rect_envelope(12, 12, 30, optical=True)), Q_45_NEG))
        obs.append(pin_pose(B.observation_plane(f"side{k}", position=(x, 50.0 + k, 20), normal=(0, -1, 0), radius_mm=10.0), Q_NEG_Y))  # distinct planes
        if k < 9:
            targets.append(Target(target_id=f"P{k}", metric="power_W", observation_id=f"side{k}",
                                  target_value=power * reflectance * (1 - reflectance) ** k, tolerance_rel=1e-9))
    targets.append(Target(target_id="Pmain", metric="power_W", observation_id="main", target_value=power * (1 - reflectance) ** splitters, tolerance_rel=1e-9))
    for k in range(10):
        targets.append(Target(target_id=f"R{k}", metric="beam_radius_mm", observation_id=f"side{k}", minimum=0.0, maximum=5.0))
        targets.append(Target(target_id=f"A{k}", metric="aoi_deg", observation_id=f"side{k}", target_value=0.0, tolerance_abs=1e-6))
    # 80 passive components with envelopes (bases / posts) on a grid away from the beams -> 100 component instances in total
    for i in range(80):
        gx, gy = 25.0 + 30.0 * (i % 20), -60.0 - 30.0 * (i // 20)
        comps.append(EngineeringComponentSnapshot(instance_id=f"Post{i}", role="post", pose=Pose3D(position_mm=(gx, gy, 0.0)),
                                                  envelope=B.rect_envelope(10, 10, 30, required_clearance_mm=2.0)))
    plate = BaseplateSnapshot(outline_mm=((-60, -200), (700, -200), (700, 200), (-60, 200)))
    proj = ProjectSnapshot(name="Standard local-validation benchmark", baseplate=plate, components=tuple(comps), observations=tuple(obs),
                           targets=tuple(targets))
    return _req(proj, layout=True, control=False)


def invalid_zero_normal() -> ValidationRequest:
    from test_robustness import _bad_plane

    proj = ProjectSnapshot(name="Structured failure: zero-length surface normal",
                           components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0)),
                                       _bad_plane((0.0, 0.0, 0.0))))
    return _req(proj, layout=False, control=False)


# --- Trace regression baselines (budget scheduling, partial beams, return light) -------------------------------------------
# Small scenes that drive the tracer's budget decisions (path / step / interaction limits, the untraced-light probe pass),
# partial beams at observation planes and a traced return-light chain, so a change to the tracer's scheduling or to its
# geometric / power decisions shows in a snapshot. ``test_trace_regression_fixtures.py`` asserts that each snapshot still
# exercises its condition.
_X = (1.0, 0.0, 0.0)
_MX = (-1.0, 0.0, 0.0)


def _limited(project: ProjectSnapshot, **numerical: int) -> ValidationRequest:
    return ValidationRequest(project=project, configuration=EngineConfiguration(
        run_layout=False, run_control=False, numerical=NumericalConfig(**numerical)))


def _facing_plates(name: str, count: int, *, reflectance: float = 0.5, lines: int = 1) -> ProjectSnapshot:
    """``count`` partial reflectors at normal incidence in series (alternate ones facing back): every pair is an etalon
    whose reflected branches the trace follows back and forth."""
    spectrum = tuple((780.0 + 0.5 * k, 1.0 / lines) for k in range(lines)) if lines > 1 else ()
    src = B.laser("SRC", wavelength_nm=None if lines > 1 else 780.0, spectrum=spectrum, power_W=1e-3, waist_radius_mm=0.5,
                  position=(-100.0, 0.0, 0.0), direction=_X)
    plates = tuple(B.npbs(f"P{k}", reflectance=reflectance, position=(20.0 * k, 0.0, 0.0), normal=_MX if k % 2 == 0 else _X)
                   for k in range(count))
    det = B.observation_plane("det", position=(150.0, 0.0, 0.0), normal=_X, radius_mm=10.0, absorbing=True)
    return ProjectSnapshot(name=name, components=(src, *plates), observations=(det,))


def trace_path_budget() -> ValidationRequest:
    """``max_paths`` 4 for ten spectral lines through a splitter: six lines are never launched and the four traced lines'
    reflected branches are refused (``PATH_LIMIT``, one ``MAX_PATHS`` issue)."""
    return _limited(_facing_plates("Path budget: ten spectral lines, four paths", 1, lines=10), max_paths=4)


def trace_step_budget() -> ValidationRequest:
    """``max_trace_steps`` 13 in a three-plate etalon with monitors between the plates: the path being traced stops between
    a monitor and the next plate (``TRACE_BUDGET``) and the weaker queued branches are not traced."""
    project = _facing_plates("Step budget: three-plate etalon cut mid-path", 3)
    monitors = (B.observation_plane("m01", position=(10.0, 0.0, 0.0), normal=_X, radius_mm=10.0),
                B.observation_plane("m12", position=(30.0, 0.0, 0.0), normal=_X, radius_mm=10.0))
    return _limited(project.model_copy(update={"observations": (*project.observations, *monitors)}), max_trace_steps=13)


def trace_interaction_limit() -> ValidationRequest:
    """``max_interactions_per_path`` 4 in a two-plate etalon: a branch followed back and forth stops at the limit
    (``MAX_INTERACTIONS``, ``INTERACTION_LIMIT``)."""
    return _limited(_facing_plates("Interaction limit: two-plate etalon", 2, reflectance=0.3), max_interactions_per_path=4)


def trace_partial_beams() -> ValidationRequest:
    """A beam wider than a small monitor's active area (a fraction of its power on it), then a 45 deg absorbing detector
    with a monitor just behind it that part of the beam reaches first while the chief ray never does (a partial crossing,
    the plane's power bounded)."""
    laser = B.laser("L", wavelength_nm=780.0, waist_radius_mm=0.5, power_W=1e-3, position=(-10.0, 0.0, 0.0), direction=_X)
    small = B.observation_plane("S", position=(-5.0, 0.0, 0.0), normal=_X, radius_mm=0.4)
    detector = pin_pose(B.observation_plane("D", position=(0.0, 0.0, 0.0), normal=(INV, INV, 0.0), radius_mm=50.0, absorbing=True),
                        Q_45_DIAG)
    behind = B.observation_plane("T", position=(0.1, 0.0, 0.0), normal=_X, radius_mm=50.0)
    proj = ProjectSnapshot(name="Partial beams: small active area and a plane behind a tilted detector", components=(laser,),
                           observations=(small, detector, behind))
    return _req(proj, layout=False, control=False)


def trace_return_chain() -> ValidationRequest:
    """A linear cavity (10 % output coupler facing the source, 1 % end reflector): every round trip's reflected branch is
    traced back toward the source down to the power floor. The small monitor inside is crossed twice by each path
    (``crossing_index`` 1) with part of the beam on its active area; the monitor outside sees the returns."""
    src = B.laser("L", wavelength_nm=1064.0, waist_radius_mm=0.5, power_W=1e-3, position=(-50.0, 0.0, 0.0), direction=_X)
    coupler = B.npbs("OC", reflectance=0.1, position=(0.0, 0.0, 0.0), normal=_MX)
    end = B.mirror("HR", position=(40.0, 0.0, 0.0), normal=_MX, reflectance=0.01)
    inside = B.observation_plane("M", position=(20.0, 0.0, 0.0), normal=_X, radius_mm=0.4)
    outside = B.observation_plane("B", position=(-20.0, 0.0, 0.0), normal=_MX, radius_mm=10.0)
    proj = ProjectSnapshot(name="Return light: traced round trips of a linear cavity", components=(src, coupler, end),
                           observations=(inside, outside))
    return _req(proj, layout=False, control=False)


def trace_probe_budget() -> ValidationRequest:
    """Two uncoated windows: the trace itself fits ``max_paths`` 4 / ``max_trace_steps`` 10, the probe pass that follows
    their untraced reflections (and the ghosts of those) does not (``MAX_PATHS`` in phase ``untraced_light``)."""
    src = B.laser("L", wavelength_nm=1064.0, waist_radius_mm=0.5, power_W=1e-3, position=(-50.0, 0.0, 0.0), direction=_X)
    windows = tuple(B.window(f"W{i}", thickness_mm=3.0, clear_radius_mm=12.7, material="N-BK7", position=(30.0 * i, 0.0, 0.0))
                    for i in range(2))
    out = B.observation_plane("O", position=(100.0, 0.0, 0.0), normal=_X, radius_mm=10.0)
    proj = ProjectSnapshot(name="Probe budget: ghost reflections of two windows", components=(src, *windows), observations=(out,))
    return _limited(proj, max_paths=4, max_trace_steps=10)

def trace_fresnel_branch_budget() -> ValidationRequest:
    """``mzi_window`` with the window's Fresnel reflections traced (``trace_fresnel_reflection``) and ``max_paths`` 3: the
    reflected branches the window's interfaces spawn (``fresnel_reflection``) are refused by the path budget once the
    splitters' branches hold it (``Tracer._spawn``: ``PATH_LIMIT``)."""
    from test_golden_interference import _mzi_project

    proj = _mzi_project(0.4123)
    window = next(c for c in proj.components if c.instance_id == "WB")
    traced = B.window("WB", thickness_mm=0.4123, clear_radius_mm=12.7, material="N-BK7", position=(50.0, 100.0, 20.0),
                      axis=(1.0, 0.0, 0.0), trace_fresnel_reflection=True)
    assert traced.model_copy(update={"surfaces": window.surfaces}) == window  # (only the traced reflections differ)
    components = tuple(traced if c.instance_id == "WB" else c for c in proj.components)
    return _limited(proj.model_copy(update={"name": "Mach-Zehnder with traced window reflections, three paths",
                                            "components": components}), max_paths=3)


def trace_freeform_surfaces() -> ValidationRequest:
    """A train of four lenses whose rear surfaces are a Zernike shape, an XY polynomial, a biconic and a toroid: their
    results, ``surface_profiles`` included, are the same bits in every runtime."""
    from os_optics.builders import _pose
    from os_optics.contracts import (BiconicSag, CircularAperture, PlaneSurface, RefractInteraction, SagSurface, ToroidalSag, XyPolynomialSag, XySagTerm,
                                          ZernikeSag, ZernikeSagTerm)

    def lens(instance_id: str, x: float, **rear: object) -> EngineeringComponentSnapshot:
        front = PlaneSurface(surface_id="front", origin_mm=(0.0, 0.0, -2.0), normal=(0.0, 0.0, -1.0), aperture=CircularAperture(radius_mm=9.0),
                             medium_positive="air", medium_negative="N-BK7", interaction=RefractInteraction())
        back = SagSurface(surface_id="back", origin_mm=(0.0, 0.0, 2.0), normal=(0.0, 0.0, 1.0), u_axis=(1.0, 0.0, 0.0), aperture=CircularAperture(radius_mm=9.0),
                          medium_positive="air", medium_negative="N-BK7", interaction=RefractInteraction(), **{"curvature_per_mm": 0.0, **rear})
        return EngineeringComponentSnapshot(instance_id=instance_id, role="lens", pose=_pose((x, 0.0, 0.0), _X), surfaces=(front, back))

    zernike = ZernikeSag(indexing="NOLL", normalization="RMS_ORTHONORMAL", normalization_radius_mm=9.5,
                         terms=(ZernikeSagTerm(index=4, value_mm=0.01), ZernikeSagTerm(index=6, value_mm=-0.02), ZernikeSagTerm(index=8, value_mm=0.004),
                                ZernikeSagTerm(index=11, value_mm=-0.002)))
    xy = XyPolynomialSag(normalization_radius_mm=9.0, terms=(XySagTerm(x_power=2, y_power=0, value_mm=0.1), XySagTerm(x_power=0, y_power=2, value_mm=-0.1),
                                                             XySagTerm(x_power=2, y_power=1, value_mm=0.01)))
    components = (
        B.laser("L", wavelength_nm=780.0, waist_radius_mm=1.0, power_W=1e-3, position=(-50.0, 0.5, -0.25), direction=_X),
        lens("Z", 0.0, curvature_per_mm=-1.0 / 200.0, sag_shape=zernike),
        lens("XY", 30.0, curvature_per_mm=-1.0 / 300.0, conic=-0.5, sag_shape=xy),
        lens("B", 60.0, sag_shape=BiconicSag(radius_x_mm=-400.0, radius_y_mm=-250.0, conic_x=-0.6, conic_y=0.2)),
        lens("T", 90.0, sag_shape=ToroidalSag(radius_y_mm=-300.0, conic=-0.2, y_terms=(1e-4,), radius_of_rotation_mm=-500.0)),
    )
    proj = ProjectSnapshot(name="Two-variable sag surfaces: Zernike, XY polynomial, biconic, toroid", components=components,
                           observations=(B.observation_plane("O", position=(150.0, 0.0, 0.0), normal=_X, radius_mm=30.0),))
    return _req(proj, layout=False, control=False)


def _lowered(components: list, observations: list, models: list, materials: list | None = None, coatings: list | None = None) -> ValidationRequest:
    """The Engine request of a Project with declared models: what a run sends to the physics."""
    from declared_helpers import with_models
    from local_helpers import project as project_document

    from os_optics.local.lowering import lower_project

    document = with_models(project_document(components, observations), models, materials, coatings, design_intent="OPTICAL_SYSTEM")
    return lower_project(document, None, check_solids=False).request


def trace_grating_orders() -> ValidationRequest:
    """A beam at 45 deg to the grooves through a transmission grating whose table states orders 0 and +1 with TE != TM and leaves order -1 out, then a second grating with no
    response: orders of known power and of unknown polarization, an order of unknown power, its bound, the grating report and the ledger, in every runtime."""
    from declared_helpers import declared_component
    from local_helpers import ALONG_X, plane, source

    def grating(model_id: str, orders: tuple[int, ...], **extra: object) -> dict:
        return {"model_id": model_id, "model": {"model_kind": "DIFFRACTION_GRATING", "model_version": "1", "mode": "TRANSMISSION", "line_density_per_mm": "300", "orders": list(orders),
                                                "clear_half_widths_mm": ["25", "25"], "edge_half_widths_mm": ["26", "26"], "status": "SPECIFIED", **extra}}

    rows = [{"wavelength_nm": "633", "aoi_deg": angle, "order": order, "efficiency_te": te, "efficiency_tm": tm}
            for angle in ("-5", "5") for order, te, tm in ((0, "0.6", "0.2"), (1, "0.25", "0.55"))]
    response = {"normalization": "ABSOLUTE", "polarization_basis": "TE_TM", "interpolation": "LINEAR", "rows": rows, "data_kind": "ASSUMED", "basis": "fixture", "max_conical_angle_deg": "1"}
    return _lowered([source("S", (-100.0, 0.0, 0.0), wavelength_nm=633.0, waist_mm=0.5, polarization_deg=45.0), declared_component("G", "other_optical", "tabled", (0.0, 0.0, 0.0), ALONG_X),
                     declared_component("H", "other_optical", "bare", (60.0, 0.0, 0.0), ALONG_X)],
                    [plane("BETWEEN", (30.0, 0.0, 0.0), radius_mm=40.0), plane("AFTER", (120.0, 0.0, 0.0), radius_mm=60.0)],
                    [grating("tabled", (-1, 0, 1), response=response), grating("bare", (0,))])


def trace_faraday_rotator() -> ValidationRequest:
    """A rotator of a stated rotation and transmittance, a mirror behind it, and a plane on each pass: the rotation of a fixed sense in the lab frame, forward and back, in every runtime."""
    from declared_helpers import declared_component, sequence, surface
    from local_helpers import AGAINST_X, ALONG_X, catalog_document, plane, source

    coating = next(c for c in catalog_document()["coatings"] if c["kind"] == "DIELECTRIC_HIGH_REFLECTOR")
    rotator = {"model_id": "fr", "model": {"model_kind": "FARADAY_ROTATOR", "model_version": "1", "rows": [{"wavelength_nm": "633", "rotation_deg": "30", "transmittance": "0.9"}],
                                           "interpolation": "POINTS_ONLY", "clear_radius_mm": "4", "edge_radius_mm": "5", "valid_aoi_deg": {"min_deg": "0", "max_deg": "5", "status": "SPECIFIED"},
                                           "operating_conditions": "fixture", "status": "SPECIFIED",
                                           "evidence": {"method": "fixture", "tools": [{"name": "fixture", "version": "1"}], "source_data_sha256": "ab" * 32, "fit_model": "fixture",
                                                        "validation_summary": "fixture", "date": "2026-10-30"}}}
    mirror = {"model_id": "m", "model": sequence([surface(1, "0", clear="10", traced="REFLECTED", coating_id=coating["coating_id"])], [], edge="11")}
    return _lowered([source("S", (-60.0, 0.0, 0.0), wavelength_nm=633.0, waist_mm=0.5), declared_component("F", "other_optical", "fr", (0.0, 0.0, 0.0), ALONG_X),
                     declared_component("M", "mirror", "m", (40.0, 0.0, 0.0), AGAINST_X)],
                    [plane("PASS", (20.0, 0.0, 0.0), radius_mm=8.0), plane("RETURN", (-30.0, 0.0, 0.0), AGAINST_X, radius_mm=8.0)], [rotator, mirror], None, [coating])


def trace_extended_emitter() -> ValidationRequest:
    """The sampled rays of a Lambertian disc through a glass lens onto an absorbing plane: the Sobol sequence, every crossing's position and power, the fractions and the ledger, in every runtime."""
    from declared_helpers import declared_component, nbk7, sequence, surface
    from local_helpers import ALONG_X, plane

    emitter = {"instance_id": "EM", "optical_role": "source", "custom_part": {"custom_part_key": "emitter", "description": "e"},
               "nominal_pose": {"position_mm": [-30.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)},
               "source_model": {"model_kind": "EXTENDED_EMITTER", "wavelength_nm": 633.0, "power_w": 1.0, "emitting_surface": {"shape": "DISC", "diameter_mm": 1.0},
                                "angular_law": "LAMBERTIAN", "sampling": {"sample_count": 256, "seed": 3}}}
    lens = {"model_id": "L", "model": sequence([surface(1, "0", radius="60", clear="12"), surface(2, "4", radius="-60", clear="12")], ["N-BK7"], edge="13")}
    return _lowered([emitter, declared_component("LENS", "lens", "L", (0.0, 0.0, 0.0), ALONG_X)], [{**plane("D", (60.0, 0.0, 0.0), radius_mm=50.0), "absorbing": True}], [lens], [nbk7()])


def trace_birefringent_crystals() -> ValidationRequest:
    """A calcite beam displacer (optic axis at (0.6, 0.8, 0) to the beam, its front face's reflections traced) then a calcite Wollaston
    pair (axes y and z, the diagonal's normal (0.8, 0.6, 0)): the ordinary / extraordinary split, the walk-off, the waves'
    indices along their rays, the reflected waves and the ledger, in every runtime."""
    from os_optics.contracts import CircularAperture, PlaneSurface, RefractInteraction, UniaxialMaterial
    from os_optics.contracts.materials import ConstantIndexLaw

    def calcite(material_id: str, axis) -> UniaxialMaterial:
        return UniaxialMaterial(material_id=material_id, valid_wavelength_nm=(400.0, 1000.0), ordinary=ConstantIndexLaw(refractive_index=1.6557),
                                extraordinary=ConstantIndexLaw(refractive_index=1.4849), optic_axis=axis)

    def face(sid, origin, normal, pos, neg, trace=False):
        return PlaneSurface(surface_id=sid, origin_mm=origin, normal=normal, aperture=CircularAperture(radius_mm=15.0), medium_positive=pos,
                            medium_negative=neg, interaction=RefractInteraction(trace_fresnel_reflection=trace))

    displacer = EngineeringComponentSnapshot(instance_id="D", pose=Pose3D(), surfaces=(
        face("front", (-5.0, 0.0, 0.0), (-1.0, 0.0, 0.0), "air", "calcite", True), face("back", (5.0, 0.0, 0.0), (1.0, 0.0, 0.0), "air", "calcite")))
    wollaston = EngineeringComponentSnapshot(instance_id="W", pose=Pose3D(), surfaces=(
        face("entrance", (25.0, 0.0, 0.0), (-1.0, 0.0, 0.0), "air", "calcite_a"), face("diagonal", (30.0, 0.0, 0.0), (0.8, 0.6, 0.0), "calcite_b", "calcite_a"),
        face("exit", (35.0, 0.0, 0.0), (1.0, 0.0, 0.0), "air", "calcite_b")))
    laser = B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.4, position=(-40.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), polarization=JonesVector.linear(30.0))
    project = ProjectSnapshot(name="calcite displacer and Wollaston", materials=(calcite("calcite", (0.6, 0.8, 0.0)), calcite("calcite_a", (0.0, 1.0, 0.0)),
                                                                                calcite("calcite_b", (0.0, 0.0, 1.0))),
                              components=(laser, displacer, wollaston),
                              observations=(B.observation_plane("OUT", position=(80.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=60.0),
                                            B.observation_plane("BACK", position=(-20.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), radius_mm=60.0)))
    return _req(project, layout=False, control=False)


FIXTURES = {
    "interference_405_500": interference_405_500,
    "keplerian_expander": keplerian_expander,
    "mzi_window": mzi_window,
    "hwp_pbs": hwp_pbs,
    "tilted_mirror_astigmatism": tilted_mirror_astigmatism,
    "aperture_clipping": aperture_clipping,
    "multi_laser": multi_laser,
    "out_of_scope_targets": out_of_scope_targets,
    "layout_example": layout_example,
    "control_recoverability": control_recoverability,
    "invalid_zero_normal": invalid_zero_normal,
    "benchmark_standard": benchmark_standard,
    "trace_path_budget": trace_path_budget,
    "trace_step_budget": trace_step_budget,
    "trace_interaction_limit": trace_interaction_limit,
    "trace_partial_beams": trace_partial_beams,
    "trace_return_chain": trace_return_chain,
    "trace_probe_budget": trace_probe_budget,
    "trace_fresnel_branch_budget": trace_fresnel_branch_budget,
    "trace_freeform_surfaces": trace_freeform_surfaces,
    "trace_grating_orders": trace_grating_orders,
    "trace_faraday_rotator": trace_faraday_rotator,
    "trace_extended_emitter": trace_extended_emitter,
    "trace_birefringent_crystals": trace_birefringent_crystals,
}
