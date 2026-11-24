"""NETWORK_NOT_PASSIVE for stated phases that cannot belong to a passive element, at every arm length (review finding M4),
and the group-delay note on constant-index media (N4).

Independent reference: a two-port whose stated amplitudes r, t apply from both sides has the scattering matrix
[[r, t], [t, r]] with singular values |r + t| and |r - t|; it is passive only when R + T + 2 sqrt(R T) |cos(phi_r - phi_t)|
<= 1 (a lossless 50:50 splitter: phi_r - phi_t = +-pi/2). The Mach-Zehnder below states phi_r = phi_t = 0 on both splitters:
the old export refused it only at arm lengths where the whole T showed gain (singular value > 1) and exported it elsewhere.
"""

from __future__ import annotations

import math

import pytest

from os_optics.contracts import ValidationRequest
from os_optics.contracts.materials import ConstantCoating
from os_optics.contracts.optics import InterfacePhases
from os_optics.local.network import export_request_network
from os_optics.network import NetworkExportError
from os_optics.network import export as network_export

from test_golden_interference import _mzi_project


def _phases(r: float, t: float) -> InterfacePhases:
    return InterfacePhases(reflection_s_rad=(r,), reflection_p_rad=(r,), transmission_s_rad=(t,), transmission_p_rad=(t,))


def _stated(project, phases: InterfacePhases | None, *, lossless_window: bool = False):
    """The MZI with both splitters stating ``phases``; ``lossless_window``: the window WB gets an ideal AR coating (R 0, T 1),
    so every path reaches an output and the power ledger loses nothing."""

    comps = []
    for c in project.components:
        if c.instance_id in ("BS1", "BS2") and phases is not None:
            c = c.model_copy(update={"surfaces": tuple(s.model_copy(update={"interaction": s.interaction.model_copy(update={"phases": phases})})
                                                       for s in c.surfaces)})
        if c.instance_id == "WB" and lossless_window:
            ar = ConstantCoating(reflectance_s=0.0, reflectance_p=0.0, transmittance_s=1.0, transmittance_p=1.0)
            c = c.model_copy(update={"surfaces": tuple(s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": ar})})
                                                       for s in c.surfaces)})
        comps.append(c)
    return project.model_copy(update={"components": tuple(comps)})


def _codes(project) -> list[tuple[str, str | None]]:
    with pytest.raises(NetworkExportError) as info:
        export_request_network(ValidationRequest(project=project))
    return [(p.code.value, p.location) for p in info.value.problems]


@pytest.mark.parametrize("window_mm", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_non_passive_stated_splitter_phases_are_refused_at_every_arm_length(window_mm):
    codes = _codes(_stated(_mzi_project(window_mm), _phases(0.0, 0.0)))
    assert ("NETWORK_NOT_PASSIVE", "surface:BS1/coating") in codes and ("NETWORK_NOT_PASSIVE", "surface:BS2/coating") in codes


def test_passive_stated_phases_export_and_conserve_energy():
    """phi_r - phi_t = pi/2 is the lossless splitter: exported; with a lossless window every column keeps its power."""

    for window_mm in (0.0, 1.0):
        network = export_request_network(ValidationRequest(project=_stated(_mzi_project(window_mm), _phases(math.pi / 2, 0.0),
                                                                             lossless_window=True)))
        assert network.transfer_matrix.lossless is True
        assert all(t == pytest.approx(1.0, abs=1e-6) for t in network.transfer_matrix.column_transmission)


def test_the_element_check_uses_r_and_t_of_the_coating():
    """A lossy splitter (R = T = 0.4) is passive with |cos(phi_r - phi_t)| <= (1 - R - T) / (2 sqrt(R T)) = 0.25 and refused
    above it."""

    def lossy(phases: InterfacePhases):
        project = _stated(_mzi_project(0.0), phases)
        comps = []
        for c in project.components:
            if c.instance_id in ("BS1", "BS2"):
                coating = ConstantCoating(reflectance_s=0.4, reflectance_p=0.4, transmittance_s=0.4, transmittance_p=0.4)
                c = c.model_copy(update={"surfaces": tuple(s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": coating})})
                                                           for s in c.surfaces)})
            comps.append(c)
        return project.model_copy(update={"components": tuple(comps)})

    inside = math.acos(0.24)  # |cos| 0.24 < 0.25: passive
    outside = math.acos(0.30)
    export_request_network(ValidationRequest(project=lossy(_phases(inside, 0.0))))
    assert ("NETWORK_NOT_PASSIVE", "surface:BS1/coating") in _codes(lossy(_phases(outside, 0.0)))


@pytest.mark.parametrize("window_mm", [1.0, 1.5])
def test_a_lossless_ledger_with_a_lossy_transfer_matrix_is_refused(window_mm, monkeypatch):
    """The second guard, independent of the element check: when no path loses light, |T J|^2 must deliver all of it. With
    the element check switched off, the non-passive splitters are still refused at arm lengths where T shows no gain."""

    monkeypatch.setattr(network_export._Exporter, "check_element_passivity", lambda self, probes, wavelength_nm: None)
    with pytest.raises(NetworkExportError) as info:
        export_request_network(ValidationRequest(project=_stated(_mzi_project(window_mm), _phases(0.0, 0.0), lossless_window=True)))
    problems = info.value.problems
    assert [p.code.value for p in problems] == ["NETWORK_NOT_PASSIVE"] and problems[0].location == "source:L0"
    assert "lose no light" in problems[0].message


def test_group_delay_names_constant_index_media():
    network = export_request_network(ValidationRequest(project=_mzi_project(2.0)))
    notes = {n.code: n.message for n in network.hints.notes}
    assert "air" in notes["GROUP_INDEX_CONSTANT_MEDIA"] and "n_g = n" in notes["GROUP_INDEX_CONSTANT_MEDIA"]
    assert "N-BK7" not in notes["GROUP_INDEX_CONSTANT_MEDIA"]
