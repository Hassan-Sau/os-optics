"""The beam splitter's reflection is a physical reflection (r_p = -r_s in the (s, d_out x s) triad), finding F1 of the network export.

Independent references: the vector reflection law of a mirror, E_out = -(I - 2 n n^T) E_in (ideal metal), and textbook
2 x 2 beam-splitter algebra. A splitter reflection is the same vector map as a mirror times the scalar -i sqrt(R)
-- for every input polarization, every source frame and every angle of incidence -- so it never acts as a hidden
half-wave plate, and the result never depends on the Engine's arbitrary transverse basis at normal incidence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import NumericalConfig
from os_optics import builders as B
from os_optics.contracts import JonesVector, ProjectSnapshot
from os_optics.validation.pipeline import CHIEF_RAYS, run_physics

CFG = NumericalConfig()
INV = 1.0 / math.sqrt(2.0)


def _source_jones(field3: np.ndarray, d: np.ndarray, up: np.ndarray) -> JonesVector:
    """The Jones vector of the 3D field ``field3`` in a source frame with local +Y = ``up`` and local +Z = ``d``."""
    e2 = up / np.linalg.norm(up)
    e1 = np.cross(e2, d)
    return JonesVector(ex=(float(np.real(field3 @ e1)), float(np.imag(field3 @ e1))),
                       ey=(float(np.real(field3 @ e2)), float(np.imag(field3 @ e2))))


def _reflected_field(element, field3: np.ndarray, *, up=(0.0, 0.0, 1.0), back_normal=(1.0, 0.0, 0.0), back_position=(20.0, 0.0, 20.0)):
    """3D field (flux-normalized, without the propagation phase) of the beam reflected by ``element`` back to a plane."""
    d = np.array([1.0, 0.0, 0.0])
    project = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=tuple(d), power_W=1e-3,
                            polarization=_source_jones(field3, d, np.array(up)), up=up), element),
        observations=(B.observation_plane("back", position=back_position, normal=back_normal, radius_mm=10.0),))
    out = run_physics(project, CFG, plan=CHIEF_RAYS)
    reflected = [a for a in out.trace.arrivals if any(e.endswith(("split_reflect", ":reflect")) for e in a.lineage)]
    assert len(reflected) == 1
    a = reflected[0]
    return a.jones[0] * a.e1 + a.jones[1] * a.e2, a.direction


def _mirror_law(field3: np.ndarray, n: np.ndarray) -> np.ndarray:
    n = n / np.linalg.norm(n)
    return -(field3 - 2.0 * (n @ field3) * n)


INPUTS = {
    "horizontal": np.array([0.0, 1.0, 0.0], dtype=complex),
    "vertical": np.array([0.0, 0.0, 1.0], dtype=complex),
    "diagonal": np.array([0.0, INV, INV], dtype=complex),
    "circular": np.array([0.0, INV, 1j * INV], dtype=complex),
}


@pytest.mark.parametrize("roll_deg", [0.0, 30.0, 90.0])
@pytest.mark.parametrize("name", sorted(INPUTS))
def test_normal_incidence_reflection_does_not_depend_on_the_source_frame(roll_deg, name):
    """At normal incidence the s axis is arbitrary (the ray's e1): rotating the source frame about the beam must not change
    the reflected 3D field. It equals -i sqrt(R) times the mirror law, i.e. i sqrt(R) E_in (R = 0.36)."""
    field3 = INPUTS[name]
    a = math.radians(roll_deg)
    up = np.array([0.0, -math.sin(a), math.cos(a)])
    splitter = B.npbs("PR", position=(50, 0, 20), normal=(-1, 0, 0), reflectance=0.36, up=tuple(up))
    got, direction = _reflected_field(splitter, field3, up=tuple(up))
    assert direction == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)
    expected = -1j * 0.6 * _mirror_law(field3, np.array([-1.0, 0.0, 0.0]))
    assert got == pytest.approx(expected, abs=1e-12)
    mirror, _ = _reflected_field(B.mirror("PR", position=(50, 0, 20), normal=(-1, 0, 0), up=tuple(up)), field3, up=tuple(up))
    assert mirror == pytest.approx(_mirror_law(field3, np.array([-1.0, 0.0, 0.0])), abs=1e-12)


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_45_degree_splitter_reflection_is_a_mirror_up_to_a_global_phase(name):
    """A 50:50 non-polarizing splitter at 45 deg reflects exactly like a mirror at the same pose, scaled by -i sqrt(R)
    (no hidden half-wave plate between s and p): compared with the vector mirror law in the lab frame."""
    field3 = INPUTS[name]
    n = np.array([-INV, INV, 0.0])
    plane = dict(back_normal=(0.0, 1.0, 0.0), back_position=(50.0, 30.0, 20.0))
    got, direction = _reflected_field(B.npbs("BS", position=(50, 0, 20), normal=tuple(n), reflectance=0.5), field3, **plane)
    assert direction == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    law = _mirror_law(field3, n)
    assert got == pytest.approx(-1j * INV * law, abs=1e-12)
    mirror, _ = _reflected_field(B.mirror("BS", position=(50, 0, 20), normal=tuple(n)), field3, **plane)
    assert mirror == pytest.approx(law, abs=1e-12)


def _hom_project(polarization_deg: float) -> ProjectSnapshot:
    pol = JonesVector.linear(polarization_deg)
    return ProjectSnapshot(
        components=(
            B.laser("A", wavelength_nm=780.0, waist_radius_mm=0.5, position=(-50, 0, 20), direction=(1, 0, 0), power_W=1.0, polarization=pol),
            B.laser("Bsrc", wavelength_nm=780.0, waist_radius_mm=0.5, position=(0, -50, 20), direction=(0, 1, 0), power_W=1.0, polarization=pol),
            B.npbs("BS", position=(0, 0, 20), normal=(-INV, INV, 0), reflectance=0.5),
        ),
        observations=(B.observation_plane("D1", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=10.0),
                      B.observation_plane("D2", position=(0, 60, 20), normal=(0, 1, 0), radius_mm=10.0)))


@pytest.mark.parametrize("polarization_deg", [0.0, 45.0, 90.0, 30.0])
def test_both_sides_of_a_splitter_form_a_unitary_map_for_every_polarization(polarization_deg):
    """Two inputs meeting the two sides of one splitter: the 2 x 2 map of the 3D fields onto the two outputs is unitary
    (Stokes relations on both sides) whatever the polarization, so diagonal light interferes like s or p light."""
    out = run_physics(_hom_project(polarization_deg), CFG, plan=CHIEF_RAYS)
    fields = {(a.observation_id, a.source_id): (a.jones[0] * a.e1 + a.jones[1] * a.e2) * np.exp(1j * 2 * math.pi * a.opl_mm / 780e-6)
              for a in out.trace.arrivals}
    # the two inputs' 3D fields at the splitter carry the same lab polarization (H horizontal, V up): compare on the lab basis
    def lab(vec, d):
        v = np.array([0.0, 0.0, 1.0])
        h = np.cross(v, d)
        return np.array([h @ vec, v @ vec])
    d1, d2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    for k in range(2):  # each lab polarization component separately (they never mix at a horizontal 45 deg fold)
        m = np.array([[lab(fields[("D1", "A")], d1)[k], lab(fields[("D1", "Bsrc")], d1)[k]],
                      [lab(fields[("D2", "A")], d2)[k], lab(fields[("D2", "Bsrc")], d2)[k]]])
        pol = np.array([math.cos(math.radians(polarization_deg)), math.sin(math.radians(polarization_deg))])[k]
        if abs(pol) < 1e-9:
            continue
        u = m / pol
        assert u.conj().T @ u == pytest.approx(np.eye(2), abs=1e-9)
        assert abs(u[0, 0] * u[1, 1] + u[0, 1] * u[1, 0]) < 1e-9  # Hong-Ou-Mandel: the permanent of a 50:50 splitter vanishes
