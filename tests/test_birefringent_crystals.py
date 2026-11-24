"""Uniaxial birefringent crystals (``optics/polarization/uniaxial.py``): light entering a crystal splits into an ordinary and an
extraordinary beam, the extraordinary ray walks off its wave normal, each beam has its own index, polarization and share of the
power, and the faces recombine directions.

References are independent of the Engine's crystal code: the analytic walk-off angle
``tan(rho) = (n_o^2 - n_e^2) sin(theta) cos(theta) / (n_e^2 cos^2(theta) + n_o^2 sin^2(theta))``, Snell's law where each wave meets one
index (principal sections), the isotropic Fresnel coefficients at normal incidence, and a test-local solver of the Fresnel
wave-normal equation ``K x (K x E) + eps E = 0`` (numpy: a quartic in the normal wave-vector component, its null vectors and the
Poynting vector ``Re(E x H*)``) with the full 4 x 4 plane-wave boundary problem for the interface's exact power split.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import (
    CircularAperture,
    EngineeringComponentSnapshot,
    JonesVector,
    MirrorInteraction,
    PlaneSurface,
    ProjectSnapshot,
    RefractInteraction,
    SphereSurface,
    UniaxialMaterial,
)
from os_optics.contracts.materials import ConstantIndexLaw, SellmeierIndexLaw
from os_optics.errors import EngineInputError

from helpers import arrivals, issue_codes, run

#: Calcite at 633 nm.
NO, NE = 1.6557, 1.4849
#: Crystal quartz at 633 nm (its optical activity is outside the model).
QO, QE = 1.5427, 1.5518
WAVELENGTH = 633.0
POWER = 1e-3
IDENTITY = B.Pose3D()


def crystal(axis, material_id: str = "calcite", n_o: float = NO, n_e: float = NE, **kw) -> UniaxialMaterial:
    a = np.array(axis, dtype=float)
    a = a / np.linalg.norm(a)
    # An explicitly ideal lossless test crystal; real catalog crystals without loss data remain unknown.
    kw.setdefault("absorption_coefficient_per_mm", 0.0)
    return UniaxialMaterial(material_id=material_id, valid_wavelength_nm=(400.0, 1000.0), ordinary=ConstantIndexLaw(refractive_index=n_o),
                            extraordinary=ConstantIndexLaw(refractive_index=n_e), optic_axis=tuple(float(x) for x in a), **kw)


def plane(sid: str, origin, normal, pos: str, neg: str, *, radius: float = 20.0, trace_reflection: bool = False) -> PlaneSurface:
    n = np.array(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return PlaneSurface(surface_id=sid, origin_mm=tuple(float(x) for x in origin), normal=tuple(float(x) for x in n),
                        aperture=CircularAperture(radius_mm=radius), medium_positive=pos, medium_negative=neg,
                        interaction=RefractInteraction(trace_fresnel_reflection=trace_reflection))


def body(instance_id: str, *surfaces: PlaneSurface) -> EngineeringComponentSnapshot:
    return EngineeringComponentSnapshot(instance_id=instance_id, pose=IDENTITY, surfaces=surfaces)


def source(direction=(1.0, 0.0, 0.0), position=(-30.0, 0.0, 0.0), polarization_deg: float = 0.0, waist: float = 0.3, **kw):
    pol = kw.pop("polarization", None) or JonesVector.linear(polarization_deg)
    return B.laser("L", wavelength_nm=WAVELENGTH, waist_radius_mm=waist, position=position, direction=direction, polarization=pol,
                   power_W=POWER, **kw)


def displacer(axis, length: float = 10.0, **kw) -> tuple[UniaxialMaterial, EngineeringComponentSnapshot]:
    half = length / 2.0
    return crystal(axis, **kw), body("X", plane("front", (-half, 0, 0), (-1, 0, 0), "air", kw.get("material_id", "calcite")),
                                     plane("back", (half, 0, 0), (1, 0, 0), "air", kw.get("material_id", "calcite")))


def project(materials, components, *observations) -> ProjectSnapshot:
    return ProjectSnapshot(materials=tuple(materials), components=tuple(components), observations=tuple(observations))


def out_plane(x: float = 40.0, normal=(1.0, 0.0, 0.0), oid: str = "OUT"):
    return B.observation_plane(oid, position=(x, 0.0, 0.0), normal=normal, radius_mm=100.0)


def walk_off(theta: float, n_o: float = NO, n_e: float = NE) -> float:
    """The analytic walk-off angle (rad, its magnitude) of the extraordinary ray, wave normal at ``theta`` to the optic axis."""
    s, c = math.sin(theta), math.cos(theta)
    return abs(math.atan((n_o * n_o - n_e * n_e) * s * c / (n_e * n_e * c * c + n_o * n_o * s * s)))


def n_theta(theta: float, n_o: float = NO, n_e: float = NE) -> float:
    return 1.0 / math.sqrt(math.cos(theta) ** 2 / n_o**2 + math.sin(theta) ** 2 / n_e**2)


def fresnel_t_normal(n1: float, n2: float) -> float:
    return 4.0 * n1 * n2 / (n1 + n2) ** 2


def _field(arrival) -> np.ndarray:
    """The unit 3D field direction (complex) of an arrival's polarization."""
    j = arrival.beam.polarization
    e1, e2 = np.array(arrival.beam.basis.e1), np.array(arrival.beam.basis.e2)
    field = complex(*j.ex) * e1 + complex(*j.ey) * e2
    return field / np.linalg.norm(field)


def by_mode(result, oid: str = "OUT"):
    """Arrivals keyed by the crystal wave their path took (the lineage event of the first crystal face)."""
    found = {}
    for a in arrivals(result, oid):
        modes = [step.rsplit("_", 1)[-1] for step in a.lineage if step.endswith(("_o", "_e"))]
        found["".join(modes)] = a
    return found


# ------------------------------------------------------------------------- independent plane-wave solver (test-local)


def eps_tensor(axis, n_o: float, n_e: float) -> np.ndarray:
    c = np.array(axis, dtype=float)
    c = c / np.linalg.norm(c)
    return n_o**2 * np.eye(3) + (n_e**2 - n_o**2) * np.outer(c, c)


def plane_waves(kt: np.ndarray, normal: np.ndarray, eps: np.ndarray):
    if np.allclose(eps, eps[0, 0] * np.eye(3)):  # isotropic: the s and p waves of both directions (a double root each)
        n = math.sqrt(eps[0, 0])
        q0 = math.sqrt(n * n - kt @ kt)
        s_axis = np.cross(kt + q0 * normal, normal)
        s_axis = s_axis / np.linalg.norm(s_axis) if np.linalg.norm(s_axis) > 1e-12 else np.linalg.svd(np.outer(normal, normal))[0][:, 1]
        waves = []
        for q in (q0, -q0):
            k = kt + q * normal
            for e in (s_axis, np.cross(k, s_axis) / n):
                h = np.cross(k, e)
                s = np.cross(e, h)
                waves.append((q, k, e.astype(complex), h.astype(complex), s / np.linalg.norm(s)))
        return waves
    return _anisotropic_waves(kt, normal, eps)


def _anisotropic_waves(kt: np.ndarray, normal: np.ndarray, eps: np.ndarray):
    """Every plane wave of tangential wave vector ``kt`` in a medium of dielectric tensor ``eps``: the real roots ``q`` of
    ``det(K K^T - |K|^2 I + eps) = 0`` with ``K = kt + q normal`` (a quartic: fitted exactly from five samples), each with its
    unit field (null vector), H = K x E and the Poynting direction Re(E x H*). Returns (q, K, E, H, S) per wave."""
    def det(q: float) -> float:
        k = kt + q * normal
        return float(np.linalg.det(np.outer(k, k) - (k @ k) * np.eye(3) + eps))
    qs = np.linspace(-3.0, 3.0, 5)
    coeff = np.polyfit(qs, [det(q) for q in qs], 4)
    waves = []
    for q in np.roots(coeff):
        if abs(q.imag) > 1e-9:
            continue
        q = float(q.real)
        k = kt + q * normal
        m = np.outer(k, k) - (k @ k) * np.eye(3) + eps
        _u, _s, vh = np.linalg.svd(m)
        e = vh[-1]
        h = np.cross(k, e)
        s = np.cross(e, h)
        waves.append((q, k, e, h, s / np.linalg.norm(s)))
    return waves


def exact_split(k_in: np.ndarray, e_in: np.ndarray, normal: np.ndarray, eps1: np.ndarray, eps2: np.ndarray):
    """The exact power fractions of the plane waves leaving a plane interface (``normal`` into medium 2) for an incident wave
    ``(k_in, e_in)`` of medium 1: the four amplitudes from continuity of tangential E and H, each wave's power the normal
    component of its Poynting vector over the incident one's. Returns (transmitted [(K, S, fraction)], reflected [...])."""
    kt = k_in - (k_in @ normal) * normal
    t1, t2 = np.linalg.svd(np.eye(3) - np.outer(normal, normal))[0][:, :2].T
    out_t = [w for w in plane_waves(kt, normal, eps2) if w[4] @ normal > 0]
    out_r = [w for w in plane_waves(kt, normal, eps1) if w[4] @ normal < 0]
    assert len(out_t) == 2 and len(out_r) == 2
    h_in = np.cross(k_in, e_in)

    def rows(e, h):
        return np.array([e @ t1, e @ t2, h @ t1, h @ t2])
    a = np.column_stack([rows(w[2], w[3]) for w in out_r] + [-rows(w[2], w[3]) for w in out_t])
    amp = np.linalg.solve(a, -rows(e_in, h_in))

    def flux(e, h):
        return abs(np.real(np.cross(e, np.conj(h))) @ normal)
    p_in = flux(e_in, h_in)
    refl = [(w[1], w[4], flux(x * w[2], x * w[3]) / p_in) for w, x in zip(out_r, amp[:2])]
    tran = [(w[1], w[4], flux(x * w[2], x * w[3]) / p_in) for w, x in zip(out_t, amp[2:])]
    return tran, refl


# ------------------------------------------------------------------------------------------------- beam displacer


@pytest.mark.parametrize("theta_deg", [30.0, 45.0, 48.0, 60.0])
@pytest.mark.parametrize("psi_deg", [0.0, 30.0, 90.0])
def test_a_calcite_beam_displacer_splits_two_beams_with_the_analytic_walk_off_and_powers(theta_deg: float, psi_deg: float) -> None:
    """Optic axis at ``theta`` to the beam in the horizontal plane: the extraordinary beam leaves displaced by ``L tan(rho)``,
    away from the axis, parallel to the input; the ordinary beam goes straight through. At normal incidence the face split is
    exact: ``P_e = cos^2(psi) T(n(theta))^2``, ``P_o = sin^2(psi) T(n_o)^2`` (``psi`` the input's angle from the extraordinary
    polarization, the source's e1 = +y)."""
    theta = math.radians(theta_deg)
    length = 10.0
    material, block = displacer((math.cos(theta), math.sin(theta), 0.0), length)
    result = run(project([material], [source(polarization_deg=psi_deg), block], out_plane()))
    assert result.power_budget.conserved
    waves = by_mode(result)
    rho = walk_off(theta)
    psi = math.radians(psi_deg)
    p_e = math.cos(psi) ** 2 * fresnel_t_normal(1.0, n_theta(theta)) ** 2 * POWER
    p_o = math.sin(psi) ** 2 * fresnel_t_normal(1.0, NO) ** 2 * POWER
    if psi_deg != 90.0:
        e = waves["e"]
        assert e.beam.position_mm[1] == pytest.approx(-length * math.tan(rho), abs=1e-12)
        assert e.beam.position_mm[2] == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(e.beam.direction, (1.0, 0.0, 0.0), atol=1e-14)
        assert e.beam.power_W == pytest.approx(p_e, rel=1e-12)
        # the extraordinary beam leaves polarized along the projection of the optic axis (horizontal: +-y)
        assert abs(_field(e)[1]) == pytest.approx(1.0, abs=1e-12)
    if psi_deg != 0.0:
        o = waves["o"]
        np.testing.assert_allclose(o.beam.position_mm, (40.0, 0.0, 0.0), atol=1e-12)
        assert abs(_field(o)[2]) == pytest.approx(1.0, abs=1e-12)  # the ordinary beam: vertical (+-z)
        assert o.beam.power_W == pytest.approx(p_o, rel=1e-12)
    # the path inside the crystal: the extraordinary segment runs along the walked-off ray, its index the phase index along it
    for path in result.paths:
        for seg in path.segments:
            if seg.crystal_mode == "EXTRAORDINARY":
                assert seg.refractive_index == pytest.approx(n_theta(theta) * math.cos(rho), rel=1e-13)
                assert seg.length_mm == pytest.approx(length / math.cos(rho), rel=1e-13)
                assert seg.optical_path_length_mm == pytest.approx(n_theta(theta) * length, rel=1e-13)
            elif seg.crystal_mode == "ORDINARY":
                assert seg.refractive_index == NO
    assert "BIREFRINGENT_INTERFACE_APPROXIMATE" not in issue_codes(result)


def test_the_walk_off_angle_at_45_degrees_in_calcite_is_6_2_degrees() -> None:
    assert math.degrees(walk_off(math.radians(45.0))) == pytest.approx(6.2, abs=0.05)


def test_quartz_walks_off_toward_its_axis_by_a_small_angle() -> None:
    """Quartz is positive (n_e > n_o): the extraordinary ray walks toward the optic axis, ~0.33 deg at 45 deg."""
    theta = math.radians(45.0)
    length = 20.0
    material, block = displacer((math.cos(theta), math.sin(theta), 0.0), length, material_id="quartz", n_o=QO, n_e=QE)
    result = run(project([material], [source(polarization_deg=45.0), block], out_plane()))
    e = by_mode(result)["e"]
    rho = walk_off(theta, QO, QE)
    assert math.degrees(rho) == pytest.approx(0.336, abs=0.002)
    assert e.beam.position_mm[1] == pytest.approx(+length * math.tan(rho), abs=1e-12)  # toward the axis (+y side)


def test_oblique_entry_matches_the_wave_normal_equation() -> None:
    """A tilted beam into a plate whose optic axis lies in no symmetry plane: inside the crystal each beam's ray direction and
    phase index along the ray are those of the independent solver; both beams leave parallel to the input, displaced."""
    axis = np.array([0.3, 0.8, 0.52])
    axis = axis / np.linalg.norm(axis)
    material, block = displacer(axis, 8.0)
    tilt = math.radians(12.0)
    d = (math.cos(tilt), math.sin(tilt) * 0.6, math.sin(tilt) * 0.8)
    result = run(project([material], [source(direction=d, polarization_deg=40.0, position=(-30.0, -30.0 * d[1] / d[0], -30.0 * d[2] / d[0])),
                                      block], out_plane()))
    assert result.power_budget.conserved
    normal = np.array([1.0, 0.0, 0.0])
    k_in = np.array(d)
    kt = k_in - (k_in @ normal) * normal
    reference = [w for w in plane_waves(kt, normal, eps_tensor(axis, NO, NE)) if w[4] @ normal > 0]
    inside = {}
    for path in result.paths:
        for seg in path.segments:
            if seg.crystal_mode is not None:
                inside[seg.crystal_mode] = seg
    assert set(inside) == {"ORDINARY", "EXTRAORDINARY"}
    for mode, seg in inside.items():
        direction = (np.array(seg.end_mm) - np.array(seg.start_mm)) / seg.length_mm
        want = min(reference, key=lambda w: abs(np.linalg.norm(w[1]) - NO) if mode == "ORDINARY" else -abs(np.linalg.norm(w[1]) - NO))
        np.testing.assert_allclose(direction, want[4], atol=1e-12)
        assert seg.refractive_index == pytest.approx(float(want[1] @ want[4]), rel=1e-12)
    for a in arrivals(result, "OUT"):
        np.testing.assert_allclose(a.beam.direction, d, atol=1e-13)


def test_a_plate_with_its_axis_in_the_face_is_a_retarder() -> None:
    """Normal incidence, optic axis in the face at 45 deg to the input polarization: two collinear beams, n_o and n_e, no
    walk-off; their optical paths differ by (n_e - n_o) L -- the retardance 2 pi (n_e - n_o) L / lambda of a waveplate."""
    length = 0.5
    material, block = displacer((0.0, 1.0, 0.0), length)
    result = run(project([material], [source(polarization_deg=45.0), block], out_plane()))
    waves = by_mode(result)
    o, e = waves["o"], waves["e"]
    np.testing.assert_allclose(o.beam.position_mm, e.beam.position_mm, atol=1e-13)
    assert e.beam.optical_path_length_mm - o.beam.optical_path_length_mm == pytest.approx((NE - NO) * length, abs=1e-12)
    assert o.beam.power_W == pytest.approx(0.5 * fresnel_t_normal(1.0, NO) ** 2 * POWER, rel=1e-12)
    assert e.beam.power_W == pytest.approx(0.5 * fresnel_t_normal(1.0, NE) ** 2 * POWER, rel=1e-12)


# ------------------------------------------------------------------------------------------------- polarizing prisms


def _wedge(alpha: float, axis1, axis2, *, gap: float = 0.0, ids=("calcite_a", "calcite_b"), n_gap: str = "air", length: float = 10.0):
    """Two right-angle prisms along x: entrance face x = -L/2, the diagonal through the origin at ``alpha`` to the entrance face
    (its normal (cos alpha, sin alpha, 0)), exit face x = +L/2; a gap of ``gap`` mm of ``n_gap`` between them (0: cemented)."""
    half = length / 2.0
    n = (math.cos(alpha), math.sin(alpha), 0.0)
    shift = np.array(n) * gap / 2.0
    first = [plane("entrance", (-half, 0, 0), (-1, 0, 0), "air", ids[0])]
    if gap == 0.0:
        first.append(plane("diagonal", (0, 0, 0), n, ids[1], ids[0]))
        second = []
    else:
        first.append(plane("diagonal_1", tuple(-shift), n, n_gap, ids[0]))
        second = [plane("diagonal_2", tuple(shift), n, ids[1], n_gap)]
    second.append(plane("exit", (half, 0, 0), (1, 0, 0), "air", ids[1]))
    return ([crystal(axis1, ids[0]), crystal(axis2, ids[1])], body("P1", *first), body("P2", *second))


def _snell_exit(n1: float, n2: float, alpha: float) -> np.ndarray:
    """Direction leaving the exit face (normal +x) of a beam entering along +x that refracts at the diagonal (normal at
    ``alpha`` from x) from index n1 into n2 and leaves into air: Snell's law at both faces, in the x-y plane."""
    normal = np.array([math.cos(alpha), math.sin(alpha), 0.0])
    k = np.array([1.0, 0.0, 0.0]) * n1
    kt = k - (k @ normal) * normal
    k2 = kt + math.sqrt(n2 * n2 - kt @ kt) * normal
    kt = k2 - k2[0] * np.array([1.0, 0.0, 0.0])
    out = kt + math.sqrt(1.0 - kt @ kt) * np.array([1.0, 0.0, 0.0])
    return out


@pytest.mark.parametrize("alpha_deg", [15.0, 30.0])
def test_a_wollaston_prism_splits_symmetric_beams_at_the_snell_angles(alpha_deg: float) -> None:
    """Calcite Wollaston: optic axes along y (first prism, perpendicular to the beam) and z (second, perpendicular to the plane of
    incidence at the diagonal); every beam meets one index in each prism, so its directions follow Snell's law: polarized along
    y the beam goes n_e -> n_o, along z n_o -> n_e.
    The split angle is close to 2 (n_o - n_e) tan(alpha); the 45 deg input divides between the beams."""
    alpha = math.radians(alpha_deg)
    materials, p1, p2 = _wedge(alpha, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    result = run(project(materials, [source(polarization_deg=45.0), p1, p2], out_plane(60.0)))
    assert result.power_budget.conserved
    found = by_mode(result)
    assert set(found) == {"oe", "eo"}
    np.testing.assert_allclose(found["eo"].beam.direction, _snell_exit(NE, NO, alpha), atol=1e-13)
    np.testing.assert_allclose(found["oe"].beam.direction, _snell_exit(NO, NE, alpha), atol=1e-13)
    split = math.acos(float(np.dot(found["eo"].beam.direction, found["oe"].beam.direction)))
    assert split == pytest.approx(2.0 * (NO - NE) * math.tan(alpha), rel=0.08)
    total = found["eo"].beam.power_W + found["oe"].beam.power_W
    assert found["eo"].beam.power_W / total == pytest.approx(0.5, abs=0.01)
    assert "BIREFRINGENT_INTERFACE_APPROXIMATE" in issue_codes(result)


def test_a_rochon_prism_passes_the_ordinary_beam_undeviated() -> None:
    """Rochon: the first prism's axis along the beam (both waves n_o, no split), the second's along z: the beam polarized along
    y stays ordinary (n_o -> n_o) and leaves undeviated; the one along z becomes extraordinary (n_o -> n_e) and is deviated."""
    alpha = math.radians(25.0)
    materials, p1, p2 = _wedge(alpha, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    result = run(project(materials, [source(polarization_deg=45.0), p1, p2], out_plane(60.0)))
    assert result.power_budget.conserved
    beams = arrivals(result, "OUT")
    assert len(beams) == 2
    straight = [a for a in beams if abs(a.beam.direction[1]) < 1e-14]
    bent = [a for a in beams if abs(a.beam.direction[1]) >= 1e-14]
    assert len(straight) == 1 and len(bent) == 1
    np.testing.assert_allclose(straight[0].beam.position_mm, (60.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(bent[0].beam.direction, _snell_exit(NO, NE, alpha), atol=1e-13)


def test_an_air_spaced_glan_prism_reflects_the_ordinary_beam_and_passes_the_extraordinary() -> None:
    """Glan-type, air-spaced: optic axes along z, perpendicular to the plane of incidence at the gap, which is met at 39 deg --
    beyond the ordinary critical angle (37.1 deg) and inside the extraordinary one (42.3 deg). The z-polarized (s,
    extraordinary) light crosses the gap with the exact isotropic Fresnel transmittances; the y-polarized (p, ordinary) light
    is totally reflected. The split is exact here (the axes are perpendicular to the plane of incidence): no approximation
    warning."""
    alpha = math.radians(39.0)  # the diagonal's normal at alpha from the beam: incidence alpha on it
    materials, p1, p2 = _wedge(alpha, (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), gap=0.05)
    side = B.observation_plane("SIDE", position=(0.0, -40.0, 0.0), normal=(0.0, -1.0, 0.0), radius_mm=100.0)
    result = run(project(materials, [source(polarization_deg=45.0), p1, p2], out_plane(60.0), side))
    assert result.power_budget.conserved
    assert "BIREFRINGENT_INTERFACE_APPROXIMATE" not in issue_codes(result)
    through = arrivals(result, "OUT")
    assert len(through) == 1
    # z-polarized = s at every face; calcite e (n_e) -> air -> calcite e -> air: Fresnel T_s at each face
    cos_gap = math.cos(alpha)

    def t_s(n1, n2, cos_i):
        sin_t = n1 / n2 * math.sqrt(1.0 - cos_i * cos_i)
        cos_t = math.sqrt(1.0 - sin_t * sin_t)
        return 1.0 - ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    expected = 0.5 * POWER * fresnel_t_normal(1.0, NE) ** 2 * t_s(NE, 1.0, cos_gap) * t_s(1.0, NE, math.sqrt(1.0 - (NE * math.sin(alpha)) ** 2))
    assert through[0].beam.power_W == pytest.approx(expected, rel=1e-12)
    assert abs(_field(through[0])[2]) == pytest.approx(1.0, abs=1e-12)
    # the ordinary light is totally reflected at the gap: it carries the rest of the transmitted entrance power (less what
    # leaves through untraced faces)
    tir = [p for p in result.paths if any(step.endswith(":tir_o") for step in p.lineage)]
    assert tir


# ------------------------------------------------------------------------------------------------- exact split at oblique faces


@pytest.mark.parametrize("aoi_deg,tolerance,axis", [
    (2.0, 6e-4, (math.cos(math.radians(45.0)), math.sin(math.radians(45.0)), 0.0)),
    (10.0, 3e-3, (math.cos(math.radians(45.0)), math.sin(math.radians(45.0)), 0.0)),
    (30.0, 2.5e-2, (math.cos(math.radians(45.0)), math.sin(math.radians(45.0)), 0.0)),
    (30.0, 1e-12, (0.0, 0.0, 1.0)),  # the axis perpendicular to the plane of incidence: exact at any angle
])
def test_the_oblique_face_split_is_close_to_the_exact_plane_wave_solution(aoi_deg: float, tolerance: float, axis) -> None:
    """Calcite plate entered obliquely: the powers of the two beams just inside the crystal against the exact 4 x 4 boundary
    solution. With the optic axis at 45 deg in the plane of incidence (the approximate case) the difference grows about
    linearly with the angle of incidence, 2.6e-4 of the incident power per degree up to 10 deg, 0.023 at 30 deg (the most over
    axis directions and input polarizations); faces met beyond ``birefringent_interface_warn_aoi_deg`` (2 deg)
    warn. With the axis perpendicular to the plane of incidence the split is exact."""
    material = crystal(axis)
    aoi = math.radians(aoi_deg)
    d = (math.cos(aoi), math.sin(aoi), 0.0)
    block = body("X", plane("front", (0.0, 0.0, 0.0), (-1, 0, 0), "air", "calcite", radius=200.0),
                 plane("back", (40.0, 0.0, 0.0), (1, 0, 0), "air", "calcite", radius=200.0))
    # read the powers from the branch paths' first crystal segment
    for psi in (0.0, 35.0, 90.0):
        result = run(project([material], [source(direction=d, position=(-30.0 * d[0], -30.0 * d[1], 0.0), polarization_deg=psi), block]))
        engine = {}
        for path in result.paths:
            seg = next((s for s in path.segments if s.crystal_mode is not None), None)
            if seg is not None:
                engine[seg.crystal_mode] = seg.power_start_W / POWER
        # the source's field: e1 = +y rotated by the tilt (horizontal, transverse), e2 = +z
        e1 = np.array([-math.sin(aoi), math.cos(aoi), 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
        e_in = math.cos(math.radians(psi)) * e1 + math.sin(math.radians(psi)) * e2
        tran, _refl = exact_split(np.array(d), e_in.astype(complex), np.array([1.0, 0.0, 0.0]), np.eye(3), eps_tensor(axis, NO, NE))
        exact = {("ORDINARY" if abs(np.linalg.norm(k) - NO) < 1e-9 else "EXTRAORDINARY"): f for k, _s, f in tran}
        for mode in ("ORDINARY", "EXTRAORDINARY"):
            assert engine.get(mode, 0.0) == pytest.approx(exact[mode], abs=tolerance), (psi, mode, engine, exact)
        assert ("BIREFRINGENT_INTERFACE_APPROXIMATE" in issue_codes(result)) is (aoi_deg > 2.0 and axis[2] == 0.0)


def test_the_exact_solver_reproduces_normal_incidence_fresnel() -> None:
    """The test-local solver itself: at normal incidence with a tilted axis each wave's transmittance is the isotropic Fresnel
    value at its phase index (o: n_o, e: n(theta))."""
    theta = math.radians(45.0)
    axis = (math.cos(theta), math.sin(theta), 0.0)
    for e_in, want in ((np.array([0.0, 0.0, 1.0]), fresnel_t_normal(1.0, NO)), (np.array([0.0, 1.0, 0.0]), fresnel_t_normal(1.0, n_theta(theta)))):
        tran, refl = exact_split(np.array([1.0, 0.0, 0.0]), e_in.astype(complex), np.array([1.0, 0.0, 0.0]), np.eye(3), eps_tensor(axis, NO, NE))
        assert max(f for _k, _s, f in tran) == pytest.approx(want, rel=1e-9)
        assert sum(f for _k, _s, f in tran) + sum(f for _k, _s, f in refl) == pytest.approx(1.0, rel=1e-9)


# ------------------------------------------------------------------------------------------------- Gaussian, group, partial light


def test_the_extraordinary_beam_keeps_a_gaussian_state_within_the_stated_error() -> None:
    """A wide beam through a 10 mm displacer: both beams keep a valid Gaussian state (the anisotropic diffraction error is far below
    birefringent_gaussian_rel_tol of the Rayleigh range); a tightly focused beam's extraordinary state is out of model scope with
    GAUSSIAN_ANISOTROPIC_MEDIUM, its ordinary beam's stays valid."""
    theta = math.radians(45.0)
    material, block = displacer((math.cos(theta), math.sin(theta), 0.0), 10.0)
    wide = run(project([material], [source(polarization_deg=45.0, waist=0.5), block], out_plane()))
    for a in arrivals(wide, "OUT"):
        assert a.beam.gaussian.valid
    tight = run(project([material], [source(polarization_deg=45.0, waist=0.02), block], out_plane()))
    found = by_mode(tight)
    assert found["o"].beam.gaussian.valid
    assert not found["e"].beam.gaussian.valid
    assert "GAUSSIAN_ANISOTROPIC_MEDIUM" in issue_codes(tight)


def test_an_unpolarized_beam_splits_into_two_fully_polarized_beams() -> None:
    """Unpolarized light through a displacer: each beam is fully polarized (degree of polarization 1, its own route) and carries
    half of the light (times its faces' transmittance)."""
    theta = math.radians(45.0)
    material, block = displacer((math.cos(theta), math.sin(theta), 0.0), 10.0)
    laser = source(polarization_deg=0.0)
    laser = laser.model_copy(update={"source": laser.source.model_copy(update={"degree_of_polarization": 0.0})})
    result = run(project([material], [laser, block], out_plane()))
    assert result.power_budget.conserved
    obs = next(o for o in result.observations if o.observation_id == "OUT")
    mixtures = obs.polarization_mixtures
    assert len(mixtures) == 2
    for m in mixtures:
        assert m.degree_of_polarization == pytest.approx(1.0, abs=1e-12)
    powers = sorted(m.stokes_W[0] for m in mixtures)
    want = sorted(0.5 * POWER * fresnel_t_normal(1.0, n) ** 2 for n in (NO, n_theta(theta)))
    assert powers == pytest.approx(want, rel=1e-12)


def test_the_group_index_along_the_extraordinary_ray() -> None:
    """Sellmeier calcite-like indices: the extraordinary beam's group path in the crystal is n_g(theta) cos(rho) per mm of ray,
    n_g(theta) = n - lambda dn/dlambda at fixed theta (an independent central difference of n(theta) here)."""
    b_o, c_o = (0.73358749, 0.96464345, 1.82831454), (0.0129, 0.0197, 112.0)
    b_e, c_e = (0.35859695, 0.82427830, 0.14429128), (0.0103, 0.0105, 47.0)
    theta = math.radians(40.0)
    axis = (math.cos(theta), math.sin(theta), 0.0)
    material = UniaxialMaterial(material_id="calcite", valid_wavelength_nm=(400.0, 1000.0), ordinary=SellmeierIndexLaw(b=b_o, c_um2=c_o),
                                extraordinary=SellmeierIndexLaw(b=b_e, c_um2=c_e), optic_axis=axis)

    def sell(b, c, lam_nm):
        l2 = (lam_nm / 1000.0) ** 2
        return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))

    def n_th(lam):
        return n_theta(theta, sell(b_o, c_o, lam), sell(b_e, c_e, lam))
    h = 1e-3
    ng = n_th(WAVELENGTH) - WAVELENGTH * (n_th(WAVELENGTH + h) - n_th(WAVELENGTH - h)) / (2 * h)
    rho = walk_off(theta, sell(b_o, c_o, WAVELENGTH), sell(b_e, c_e, WAVELENGTH))
    from os_optics.optics.materials.library import MaterialLibrary
    from os_optics.optics.polarization import uniaxial

    library = MaterialLibrary([material], NumericalConfig())
    laws = library.uniaxial("calcite")
    crystal_now = uniaxial.Crystal(sell(b_o, c_o, WAVELENGTH), sell(b_e, c_e, WAVELENGTH), np.array(axis), library.law_group(laws.ordinary, WAVELENGTH),
                                   library.law_group(laws.extraordinary, WAVELENGTH), WAVELENGTH)
    waves = uniaxial.outgoing(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]), crystal_now, None, uniaxial.EXTRAORDINARY, True,
                              np.array([[0.0, 1.0, 0.0]]), 1e-12)
    group = uniaxial.ray_group(crystal_now, waves)
    assert group is not None
    assert group[0] == pytest.approx(ng * math.cos(rho), rel=1e-7)


# ------------------------------------------------------------------------------------------------- refusals


def _refused(proj):
    """The run is refused before tracing: one FAIL issue BIREFRINGENT_MEDIUM_UNSUPPORTED naming the place (the lowering raises
    EngineInputError with the same code)."""
    result = run(proj)
    refusals = [i for i in result.issues if i.code.value == "BIREFRINGENT_MEDIUM_UNSUPPORTED"]
    assert len(refusals) == 1 and refusals[0].status.value == "FAIL", result.issues
    from os_optics.optics.materials.library import MaterialLibrary
    from os_optics.scene.lower import lower_scene

    with pytest.raises(EngineInputError) as caught:
        lower_scene(proj, NumericalConfig(), MaterialLibrary(proj.materials, NumericalConfig()))
    assert caught.value.code.value == "BIREFRINGENT_MEDIUM_UNSUPPORTED"
    return refusals[0]


def test_a_curved_crystal_face_is_refused() -> None:
    material = crystal((0.0, 1.0, 0.0))
    lens = body("X", plane("front", (-2.0, 0, 0), (-1, 0, 0), "air", "calcite"),
                SphereSurface(surface_id="back", center_mm=(-20.0, 0.0, 0.0), radius_mm=22.0, cap_axis=(1.0, 0.0, 0.0), aperture_radius_mm=10.0,
                              medium_positive="air", medium_negative="calcite", interaction=RefractInteraction()))
    error = _refused(project([material], [source(), lens], out_plane()))
    assert "curved" in error.message and error.location == "surface:X/back"


def test_light_starting_or_observed_inside_a_crystal_is_refused() -> None:
    material, block = displacer((0.0, 1.0, 0.0))
    _refused(project([material], [source(medium="calcite", position=(0.0, 0.0, 0.0)), block], out_plane()))
    inside = B.observation_plane("IN", position=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=10.0, medium="calcite")
    _refused(project([material], [source(), block], inside))


def test_a_crystal_face_that_does_not_refract_is_refused() -> None:
    material = crystal((0.0, 1.0, 0.0))
    face = PlaneSurface(surface_id="mirror", origin_mm=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0), aperture=CircularAperture(radius_mm=5.0),
                        medium_positive="air", medium_negative="calcite", interaction=MirrorInteraction())
    _refused(project([material], [source(), body("X", face)], out_plane()))


def test_the_optic_axis_turns_with_its_component() -> None:
    """``optic_axis_component``: the axis is given in that component's frame and turns with its pose (here the component's local
    +z is the beam axis, local +x the scene's +y)."""
    theta = math.radians(45.0)
    length = 10.0
    in_frame = crystal((0.0, 0.0, 1.0), optic_axis_component="X")
    turned = B.window("X", thickness_mm=length, clear_radius_mm=10.0, material="calcite", position=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0))
    # local x of the window is the scene's +y (laser-like pose): an axis (sin, 0, cos) local = (cos, sin, 0) in the scene
    local = in_frame.model_copy(update={"optic_axis": (math.sin(theta), 0.0, math.cos(theta))})
    result = run(project([local], [source(polarization_deg=0.0), turned], out_plane()))
    e = by_mode(result)["e"]
    assert abs(e.beam.position_mm[1]) == pytest.approx(length * math.tan(walk_off(theta)), abs=1e-12)
