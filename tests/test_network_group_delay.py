"""Per-path group delay of the linear optical network export (data contract).

Independent reference: the group index n_g = n - lambda dn/dlambda of N-BK7 from the Schott Sellmeier formula written here,
differentiated numerically (central difference), times the glass thickness; air is n_g = 1. The Engine computes n_g in closed
form from the same material model; both agree to the export's 1e-9 quantization.
"""

from __future__ import annotations

import math

import pytest

from os_optics import builders as B
from os_optics.contracts import JonesVector, ProjectSnapshot, ValidationRequest
from os_optics.contracts.materials import TabulatedIndexMaterial
from os_optics.config import NumericalConfig
from os_optics.local.network import export_request_network
from os_optics.network import contract_hash_of
from os_optics.optics.materials.group import group_index
from os_optics.optics.materials.library import BUILTIN_MATERIALS

from test_golden_interference import _mzi_project

INV = 1.0 / math.sqrt(2.0)
C_MM_PER_PS = 0.299792458
SCHOTT_B = (1.03961212, 0.231792344, 1.01046945)
SCHOTT_C = (0.00600069867, 0.0200179144, 103.560653)


def _n(lam_um: float) -> float:
    l2 = lam_um**2
    return math.sqrt(1 + sum(b * l2 / (l2 - c) for b, c in zip(SCHOTT_B, SCHOTT_C)))


def _ng(wavelength_nm: float) -> float:
    lam, h = wavelength_nm / 1000.0, 1e-5
    return _n(lam) - lam * (_n(lam + h) - _n(lam - h)) / (2 * h)


def _hom(window_mm: float) -> ProjectSnapshot:
    """Two 780 nm sources on a 50:50 splitter, an uncoated N-BK7 window in arm A."""

    pol = JonesVector.linear(90.0)
    comps = [
        B.laser("A", wavelength_nm=780.0, waist_radius_mm=0.5, position=(-50, 0, 20), direction=(1, 0, 0), power_W=1e-3, polarization=pol),
        B.laser("B", wavelength_nm=780.0, waist_radius_mm=0.5, position=(0, -50, 20), direction=(0, 1, 0), power_W=1e-3, polarization=pol),
        B.npbs("BS", position=(0, 0, 20), normal=(-INV, INV, 0), reflectance=0.5),
        B.window("W", thickness_mm=window_mm, clear_radius_mm=12.7, material="N-BK7", position=(-25, 0, 20), axis=(1, 0, 0)),
    ]
    return ProjectSnapshot(components=tuple(comps), observations=(
        B.observation_plane("D1", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=10.0),
        B.observation_plane("D2", position=(0, 60, 20), normal=(0, 1, 0), radius_mm=10.0)))


def test_group_delay_of_a_glass_window_matches_the_independent_group_index():
    t = 3.0
    net = export_request_network(ValidationRequest(project=_hom(t)))
    ng = _ng(780.0)
    for path in net.paths:
        glass = t if path.source_id == "A" else 0.0
        assert path.group_opl_mm == pytest.approx(path.length_mm + (ng - 1.0) * glass, abs=2e-9)
        assert path.group_delay_ps == pytest.approx(path.group_opl_mm / C_MM_PER_PS, abs=2e-9)
        assert path.group_opl_mm >= path.opl_mm  # normal dispersion: n_g > n
    for pair in net.hints.path_pairs:
        assert pair.group_delay_ps == pytest.approx((ng - 1.0) * t / C_MM_PER_PS, abs=5e-9)  # 5.278 ps for 3 mm at 780 nm
        assert pair.delay_ps == pytest.approx((_n(0.78) - 1.0) * t / C_MM_PER_PS, abs=5e-9)  # the phase delay, 5.115 ps
        assert pair.group_opd_mm == pytest.approx((ng - 1.0) * t, abs=2e-9)
    assert net.conventions.group_delay is not None and "n_g = n - lambda dn/dlambda" in net.conventions.group_delay
    assert "GROUP_DELAY_FIRST_ORDER" in {n.code for n in net.hints.notes}
    assert not any("group delay and dispersion" in item for item in net.conventions.not_modelled)
    assert contract_hash_of(net.model_dump(mode="json", by_alias=True)) == net.contract_hash


def test_branch_paths_include_their_ancestors_glass():
    """The MZI with a 1 mm window in arm B (532 nm): both output beams' paths through arm B carry the window's group delay,
    including the branches reflected at the first splitter."""

    net = export_request_network(ValidationRequest(project=_mzi_project(1.0)))
    ng = _ng(532.0)
    extras = sorted(round(p.group_opl_mm - p.length_mm, 9) for p in net.paths)
    assert extras == pytest.approx([0.0, 0.0, ng - 1.0, ng - 1.0], abs=2e-9)
    for pair in net.hints.path_pairs:
        assert abs(pair.group_delay_ps) == pytest.approx((ng - 1.0) / C_MM_PER_PS, abs=5e-9)


def test_group_index_of_each_material_kind():
    cfg = NumericalConfig()
    bk7 = next(m for m in BUILTIN_MATERIALS if m.material_id == "N-BK7")
    air = next(m for m in BUILTIN_MATERIALS if m.material_id == "air")
    assert group_index(bk7, 780.0, cfg) == pytest.approx(_ng(780.0), abs=1e-9)
    assert group_index(air, 780.0, cfg) == 1.0
    assert group_index(bk7, 5000.0, cfg) is None  # outside the data: never extrapolated
    # a linearly tabulated index n = 1.5 + 1e-4 (1000 - lambda) / nm: n_g = n + lambda 1e-4
    table = TabulatedIndexMaterial(material_id="lin", valid_wavelength_nm=(500.0, 1500.0), wavelength_nm=(500.0, 1500.0),
                                   refractive_index=(1.55, 1.45), interpolation="linear")
    assert group_index(table, 800.0, cfg) == pytest.approx(1.52 + 800.0 * 1e-4, abs=1e-9)
    assert group_index(table, 500.0, cfg) == pytest.approx(1.55 + 500.0 * 1e-4, abs=1e-9)  # one-sided at the band edge
