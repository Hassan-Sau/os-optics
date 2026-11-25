"""Native polarization elements (``POLARIZATION_ELEMENT`` models): ideal thin Jones-matrix elements at ``z = 0``.

* ``polarizer.linear``: power transmittance ``T`` for light polarized along the transmission axis (local +X) and
  ``T / extinction_ratio`` perpendicular to it: Jones matrix ``diag(sqrt(T), sqrt(T / ER))`` in the (axis, perpendicular)
  basis.
* ``waveplate.retarder``: retardance ``delta`` (waves) between the fast axis (local +X) and the slow axis: Jones matrix
  ``sqrt(T) diag(1, exp(i 2 pi delta))`` up to a common phase; ``ZERO_ORDER`` scales the retardance as
  ``delta(l) = delta_0 l_0 / l`` (birefringence dispersion not modelled), ``ACHROMATIC`` keeps ``delta_0`` across the band.
  A half-wave plate is ``delta = 0.5``, a quarter-wave plate ``0.25``.

Both are thin (plate thickness and refraction ignored). Rotate the component about its local Z (its ``nominal_pose``)
to set the axis angle (Hecht, *Optics*, 5th ed., sec. 8.12; Born & Wolf, sec. 10.9).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._vendor.shared_contracts.design_catalog import FactStatus, ModelFidelity, SimulationApproximation
from .common import (
    DEFAULT_CLEAR_FRACTION,
    THIN_ELEMENT_THICKNESS_MM,
    BuildContext,
    NativeBuild,
    aoi_range,
    band,
    cylinder,
    default_model_id,
    finish,
    optical_model,
    transmissive_frame,
)
from .errors import invalid
from .numbers import TWO, ZERO, text

RETARDANCE_DISPERSIONS = {"ZERO_ORDER": "INVERSE_WAVELENGTH", "ACHROMATIC": "CONSTANT"}


def _common(values: Mapping[str, Any]) -> dict[str, Any]:
    diameter = values["diameter_mm"]
    clear = values.get("clear_aperture_diameter_mm")
    clear = DEFAULT_CLEAR_FRACTION * diameter if clear is None else clear
    if clear > diameter:
        raise invalid("the clear aperture cannot exceed the outer diameter", "clear_aperture_diameter_mm")
    low, high = values["valid_band_nm"]
    a_min, a_max = aoi_range(values.get("valid_aoi_deg"), "valid_aoi_deg")
    return {
        "model_kind": "POLARIZATION_ELEMENT", "model_version": "1", "fidelity": ModelFidelity.IDEAL_THIN_ELEMENT.value, "plane_z_mm": "0",
        "axis": "+X", "clear_radius_mm": text(clear / TWO), "edge_radius_mm": text(diameter / TWO),
        "valid_band_nm": band(low, high, FactStatus.DESIGN_BASIS),
        "valid_aoi_deg": {"min_deg": text(a_min), "max_deg": text(a_max), "status": FactStatus.DESIGN_BASIS.value},
        "transmittance": {"value": text(values["transmittance"]), "status": FactStatus.DESIGN_BASIS.value},
        "status": FactStatus.DESIGN_BASIS.value,
    }


def build_linear_polarizer(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    physics = _common(values)
    extinction = values["extinction_ratio"]
    physics.update(element="LINEAR_POLARIZER", extinction_ratio={"value": text(extinction), "status": FactStatus.DESIGN_BASIS.value},
                   approximations=[SimulationApproximation.THIN_ELEMENT.value])
    thickness = values.get("thickness_mm") or THIN_ELEMENT_THICKNESS_MM
    model_id = values.get("model_id") or default_model_id("polarizer", "ER", extinction)
    model = optical_model(model_id, values.get("display_name") or f"Linear polarizer ER = {text(extinction)}", physics,
                          frame=transmissive_frame(rotational=False), geometry=cylinder(values["diameter_mm"], ZERO, thickness))
    design = {"jones_matrix": "diag(sqrt(T), sqrt(T / ER)) in the (transmission axis = local +X, perpendicular) basis",
              "transmittance_parallel": text(values["transmittance"]),
              "transmittance_perpendicular": text(values["transmittance"] / extinction)}
    return finish("polarizer.linear", model, role="polarizer", values=values, design=design)


def build_retarder(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    physics = _common(values)
    retardance, wavelength = values["retardance_waves"], values["design_wavelength_nm"]
    low, high = values["valid_band_nm"]
    if not low <= wavelength <= high:
        raise invalid("the design wavelength must lie inside valid_band_nm", "design_wavelength_nm")
    dispersion = values.get("retardance_dispersion") or "ZERO_ORDER"
    approximation = (SimulationApproximation.RETARDANCE_INVERSE_WAVELENGTH if dispersion == "ZERO_ORDER"
                     else SimulationApproximation.CONSTANT_RETARDANCE)
    physics.update(element="RETARDER", retardance_waves=text(retardance), design_wavelength_nm=text(wavelength),
                   retardance_dispersion=RETARDANCE_DISPERSIONS[dispersion],
                   approximations=sorted([SimulationApproximation.THIN_ELEMENT.value, approximation.value]))
    thickness = values.get("thickness_mm") or THIN_ELEMENT_THICKNESS_MM
    model_id = values.get("model_id") or default_model_id("retarder", retardance, "w", "at", wavelength)
    model = optical_model(model_id, values.get("display_name") or f"Retarder {text(retardance)} waves at {text(wavelength)} nm", physics,
                          frame=transmissive_frame(rotational=False), geometry=cylinder(values["diameter_mm"], ZERO, thickness))
    design = {"retardance_rad_at_design": f"2 pi x {text(retardance)}", "fast_axis": "local +X", "dispersion": dispersion}
    return finish("waveplate.retarder", model, role="waveplate", values=values, design=design)


__all__ = ["RETARDANCE_DISPERSIONS", "build_linear_polarizer", "build_retarder"]
