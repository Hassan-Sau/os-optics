"""Native thin elements described by a stated response: a variable neutral-density filter, a tabulated thin lens, a
Faraday rotator and a diffraction grating.

``filter.variable_nd`` -- a ``NEUTRAL_DENSITY`` model: a thin attenuator plane whose optical density follows a stated law
(power transmittance ``10^-OD`` at each crossing point). ``LINEAR_ROTARY``: a filter wheel, OD linear in the azimuth of
the crossing point about the element axis from ``od_start`` at ``arc_start_deg`` to ``od_end`` at ``arc_end_deg``
(azimuth from local +X toward local +Y, after the wheel is turned by ``nd_rotation_deg``); ``LINEAR_GRADIENT``: OD linear
along local X from ``x_start_mm`` to ``x_end_mm`` (after the filter is shifted by ``nd_offset_mm``). An optional
``coated_inner_radius_mm`` leaves a hub uncoated (a wheel coated on an annulus). Where the law leaves the plate uncoated the
``uncoated_transmittance`` applies (without it, light there is outside the model). ``reflectance`` (default 0) is
reflected; the rest is absorbed. ``position`` sets the wheel rotation (deg) or the shift (mm) in the component template
(``selected_parameters``); change it there to move the filter. The density is the same at every wavelength of the band.

``lens.tabulated_thin`` -- a ``TABULATED_THIN_LENS`` model: a thin focusing element whose focal length and delivered
power fraction are a table over wavelength (an external response: a metalens, a diffractive or a gradient-index element
characterized elsewhere), with the domain it was validated for (field angle, beam size, polarizations), the result metrics
it was validated for (each with its error bound) and the evidence of how the table was produced. A run outside that
domain, or a target asking for another metric, is outside the model's scope; the power not delivered into the focused
beam is booked as the element's unmodelled response. The rows come as a list of ``{"wavelength_nm", "focal_length_mm",
"transmittance"}`` objects or as columns ``{"wavelength_nm": [...], "focal_length_mm": [...], "transmittance": [...]}``.

``rotator.faraday`` -- a ``FARADAY_ROTATOR`` model: a thin nonreciprocal polarization rotator whose rotation and
transmittance are a stated table over wavelength, with the operating conditions and evidence the table rests on. The field
turns right-handed about local +Z whichever way the light travels, so a round trip adds the rotation (Saleh & Teich,
*Fundamentals of Photonics*, sec. 6.4). Light of a wavelength the table does not cover stops at the element, outside the
model. The rows come as a list of ``{"wavelength_nm", "rotation_deg", "transmittance"}`` objects or as columns.

``grating.plane`` -- a ``DIFFRACTION_GRATING`` model: a rectangular ruled plate, reflecting or transmitting. Its orders leave by
the vector grating equation (Loewen & Popov, *Diffraction Gratings and Applications*, ch. 2); its order powers come from a
``response`` stated with its provenance, and without one each order's power is not known while its direction is.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, localcontext
from math import gcd
from typing import Any

from pydantic import ValidationError

from .._vendor.shared_contracts.design_catalog import FactStatus, ModelFidelity, SimulationApproximation
from .._vendor.shared_contracts.optical_design.optical_model import FaradayResponseRowV1, GratingOrderResponseV1, ResponseEvidenceV1, ThinLensResponseRowV1, ValidatedMetricV1
from .common import (
    DEFAULT_CLEAR_FRACTION,
    THIN_ELEMENT_THICKNESS_MM,
    BuildContext,
    NativeBuild,
    aoi_range,
    band,
    box,
    cylinder,
    default_model_id,
    finish,
    optical_model,
    transmissive_frame,
)
from ..results.quantity import EXTERNALLY_CHARACTERIZED, MODELED, PARAMETER_NOT_STATED, UNAVAILABLE
from .errors import invalid
from .numbers import ONE, TWO, WORK_CONTEXT, ZERO, text, to_decimal

ND_LAWS = ("LINEAR_ROTARY", "LINEAR_GRADIENT")
FOCAL_INTERPOLATIONS = ("LINEAR_IN_OPTICAL_POWER", "POINTS_ONLY")
TRANSMITTANCE_INTERPOLATIONS = ("LINEAR", "POINTS_ONLY")
THIN_LENS_POLARIZATIONS = ("LINEAR_X", "LINEAR_Y", "ANY")
ROW_FIELDS = ("wavelength_nm", "focal_length_mm", "transmittance")
FARADAY_INTERPOLATIONS = ("LINEAR", "POINTS_ONLY")
FARADAY_ROW_FIELDS = ("wavelength_nm", "rotation_deg", "transmittance")
GRATING_MODES = ("REFLECTION", "TRANSMISSION")
GRATING_PLATE_THICKNESS_MM = Decimal(6)
AOD_ORDERS = ("+1", "-1")
MHZ_TO_HZ = Decimal(1000000)
M_TO_NM = Decimal(1000000000)
#: The coarsest grating a deflector's tones may compile to (10 mm): beyond it a beam has tens of thousands of propagating orders on every pass.
MAX_AOD_PERIOD_NM = Decimal(10000000)
#: The finest step of the RF tones of a deflector: their greatest common divisor is taken in whole units of this.
RF_RESOLUTION_HZ = Decimal("0.001")
DECADE = Decimal(10)


def _first_error(exc: ValidationError) -> str:
    first = exc.errors(include_url=False)[0]
    where = ".".join(str(p) for p in first["loc"])
    message = str(first["msg"]).removeprefix("Value error, ")
    return f"{where}: {message}" if where else message


# ------------------------------------------------------------------------------------------------- variable ND


def build_variable_nd(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    law_kind = values["law"]
    od_start, od_end = values["od_start"], values["od_end"]
    diameter, width, height = values.get("diameter_mm"), values.get("width_mm"), values.get("height_mm")
    if (diameter is None) == (width is None and height is None) or (width is None) != (height is None):
        raise invalid("give either diameter_mm (a round filter) or both width_mm and height_mm (a rectangular one)", "diameter_mm")
    fraction = values["clear_aperture_fraction"]
    if law_kind == "LINEAR_ROTARY":
        if diameter is None:
            raise invalid("a filter wheel (LINEAR_ROTARY) is round: give diameter_mm", "diameter_mm")
        start, end = values.get("arc_start_deg") or ZERO, values.get("arc_end_deg")
        if end is None:
            raise invalid("a LINEAR_ROTARY law states the end of its coated arc (arc_end_deg)", "arc_end_deg")
        law: dict[str, Any] = {"kind": "LINEAR_ROTARY", "od_start": text(od_start), "od_end": text(od_end), "arc_start_deg": text(start),
                               "arc_end_deg": text(end)}
        if values.get("linearity_percent") is not None:
            law["linearity_percent"] = text(values["linearity_percent"])
        key, unit = "nd_rotation_deg", "deg"
    else:
        x_start, x_end = values.get("x_start_mm"), values.get("x_end_mm")
        if x_start is None or x_end is None:
            raise invalid("a LINEAR_GRADIENT law states its coated interval (x_start_mm and x_end_mm)", "x_start_mm")
        if values.get("linearity_percent") is not None:
            raise invalid("linearity_percent belongs to a LINEAR_ROTARY wheel", "linearity_percent")
        law = {"kind": "LINEAR_GRADIENT", "od_start": text(od_start), "od_end": text(od_end), "x_start_mm": text(x_start), "x_end_mm": text(x_end)}
        key, unit = "nd_offset_mm", "mm"
    for name in ("arc_start_deg", "arc_end_deg") if law_kind == "LINEAR_GRADIENT" else ("x_start_mm", "x_end_mm"):
        if values.get(name) is not None:
            raise invalid(f"{name} does not belong to a {law_kind} law", name)
    low, high = values["valid_band_nm"]
    physics: dict[str, Any] = {
        "model_kind": "NEUTRAL_DENSITY", "model_version": "1", "fidelity": ModelFidelity.IDEAL_THIN_ELEMENT.value, "law": law,
        "plane_z_mm": "0", "valid_band_nm": band(low, high, FactStatus.DESIGN_BASIS), "position_parameter": key,
        "approximations": sorted(a.value for a in (SimulationApproximation.THIN_ELEMENT, SimulationApproximation.CONSTANT_OPTICAL_DENSITY,
                                                   SimulationApproximation.SUBSTRATE_NOT_MODELLED, SimulationApproximation.EDGE_AS_OPAQUE_RIM)),
        "status": FactStatus.DESIGN_BASIS.value,
    }
    if diameter is not None:
        physics.update(clear_radius_mm=text(fraction * diameter / TWO), edge_radius_mm=text(diameter / TWO))
    else:
        physics.update(clear_half_widths_mm=[text(fraction * width / TWO), text(fraction * height / TWO)],  # type: ignore[operator]
                       edge_half_widths_mm=[text(width / TWO), text(height / TWO)])  # type: ignore[operator]
    for name in ("uncoated_transmittance", "reflectance", "coated_inner_radius_mm"):
        if values.get(name) is not None:
            physics[name] = text(values[name])
    thickness = values.get("thickness_mm") or THIN_ELEMENT_THICKNESS_MM
    model_id = values.get("model_id") or default_model_id("variable-nd", law_kind.split("_")[1].lower(), "od", od_start, od_end)
    name = values.get("display_name") or (f"Variable ND {'wheel' if law_kind == 'LINEAR_ROTARY' else 'gradient'} OD {text(od_start)}-{text(od_end)}")
    geometry = cylinder(diameter, ZERO, thickness) if diameter is not None else box(width / TWO, height / TWO, ZERO, thickness)  # type: ignore[operator]
    model = optical_model(model_id, name, physics, frame=transmissive_frame(False), geometry=geometry, parameter="law")
    position = values.get("position")
    selected = None
    if position is not None:
        selected = [{"semantic_key": key, "value_kind": "DECIMAL", "value": text(position), "unit_code": unit}]
    design = {"law": law_kind, "position_parameter": key, "position": text(position) if position is not None else "0",
              "transmittance_range": [_transmittance(max(od_start, od_end)), _transmittance(min(od_start, od_end))],
              "equations": "T = 10^-OD at each crossing point; LINEAR_ROTARY: OD = od_start + (od_end - od_start) (phi - phi0) / (phi1 - phi0), "
                           "phi the azimuth minus nd_rotation_deg; LINEAR_GRADIENT: OD linear in x - nd_offset_mm"}
    return finish("filter.variable_nd", model, role="filter", values=values, design=design, selected_parameters=selected)


def _transmittance(od: Decimal) -> str:
    with localcontext(WORK_CONTEXT):
        return text(DECADE ** -od, Decimal("1e-12"))


# --------------------------------------------------------------------------------------------- tabulated thin lens


def _rows(raw: Any, fields: tuple[str, ...], row_model: Any) -> list[dict[str, str]]:
    if isinstance(raw, Mapping):
        if set(raw) != set(fields) or not all(isinstance(raw[k], (list, tuple)) for k in fields):
            raise invalid(f"columns are {{{', '.join(fields)}}}, each a list", "rows")
        lengths = {len(raw[k]) for k in fields}
        if len(lengths) != 1:
            raise invalid("the columns of the table have different lengths", "rows")
        raw = [{k: raw[k][i] for k in fields} for i in range(lengths.pop())]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise invalid(f"rows is a non-empty list of {{{', '.join(fields)}}} or a table of those columns", "rows")
    rows = []
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping) or set(row) != set(fields):
            raise invalid(f"row {index}: expected exactly {{{', '.join(fields)}}}", "rows")
        parsed = {k: to_decimal(row[k]) for k in fields}
        if any(v is None for v in parsed.values()):
            raise invalid(f"row {index}: every value is a decimal number", "rows")
        try:
            rows.append(row_model.model_validate({k: text(v) for k, v in parsed.items()}).model_dump(mode="json"))  # type: ignore[arg-type]
        except ValidationError as exc:
            raise invalid(f"row {index}: {_first_error(exc)}", "rows") from None
    return rows


def _metrics(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise invalid("supported_metrics is a non-empty list of {metric, abs_error?, rel_error?, basis}", "supported_metrics")
    out = []
    for index, entry in enumerate(raw):
        try:
            out.append(ValidatedMetricV1.model_validate(entry).model_dump(mode="json", exclude_none=True))
        except ValidationError as exc:
            raise invalid(f"entry {index}: {_first_error(exc)}", "supported_metrics") from None
    return out


def _evidence(raw: Any) -> dict[str, Any]:
    try:
        return ResponseEvidenceV1.model_validate(raw).model_dump(mode="json", exclude_none=True)
    except ValidationError as exc:
        raise invalid(_first_error(exc), "evidence") from None


def _polarizations(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    items = raw.split(",") if isinstance(raw, str) else raw
    if not isinstance(items, (list, tuple)) or not items or any(not isinstance(p, str) or p.strip().upper() not in THIN_LENS_POLARIZATIONS
                                                                 for p in items):
        raise invalid(f"validated_polarizations lists values among {list(THIN_LENS_POLARIZATIONS)}", "validated_polarizations")
    return [p.strip().upper() for p in items]


def build_tabulated_thin(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    rows = _rows(values["rows"], ROW_FIELDS, ThinLensResponseRowV1)
    clear = values["clear_aperture_diameter_mm"]
    diameter = values.get("diameter_mm") or clear
    if clear > diameter:
        raise invalid("the clear aperture cannot exceed the outer diameter", "clear_aperture_diameter_mm")
    physics: dict[str, Any] = {
        "model_kind": "TABULATED_THIN_LENS", "model_version": "1", "rows": rows,
        "focal_interpolation": values["focal_interpolation"], "transmittance_interpolation": values["transmittance_interpolation"],
        "clear_radius_mm": text(clear / TWO), "edge_radius_mm": text(diameter / TWO), "plane_z_mm": "0",
        "status": FactStatus.DESIGN_BASIS.value,
    }
    if values.get("max_field_angle_deg") is not None:
        physics["max_field_angle_deg"] = text(values["max_field_angle_deg"])
    if values.get("supported_metrics") is not None:
        physics["supported_metrics"] = _metrics(values["supported_metrics"])
    if values.get("evidence") is not None:
        physics["evidence"] = _evidence(values["evidence"])
    if values.get("max_beam_radius_mm") is not None:
        physics["max_beam_radius_mm"] = text(values["max_beam_radius_mm"])
    polarizations = _polarizations(values.get("validated_polarizations"))
    if polarizations is not None:
        physics["validated_polarizations"] = polarizations
    thickness = values.get("thickness_mm") or THIN_ELEMENT_THICKNESS_MM
    first, last = rows[0], rows[-1]
    model_id = values.get("model_id") or default_model_id("tabulated-lens", "f", Decimal(first["focal_length_mm"]), "w",
                                                          Decimal(first["wavelength_nm"]), Decimal(last["wavelength_nm"]))
    name = values.get("display_name") or f"Tabulated thin lens {first['wavelength_nm']}-{last['wavelength_nm']} nm"
    model = optical_model(model_id, name, physics, geometry=cylinder(diameter, ZERO, thickness), parameter="rows")
    with localcontext(WORK_CONTEXT):
        powers = [text(ONE / Decimal(r["focal_length_mm"]), Decimal("1e-12")) for r in rows]
    design = {"rows": len(rows), "valid_band_nm": [first["wavelength_nm"], last["wavelength_nm"]], "optical_power_per_mm": powers,
              "supported_metrics":[m["metric"] for m in physics["supported_metrics"]] if "supported_metrics" in physics else None,
              "drawn_thickness_mm": text(thickness),
              "equations": "per traced line: u' = u - y / f(lambda), power x T(lambda); LINEAR_IN_OPTICAL_POWER interpolates 1/f "
                           "linearly in wavelength between rows"}
    return finish("lens.tabulated_thin", model, role="lens", values=values, design=design)


# ------------------------------------------------------------------------------------------------ Faraday rotator


def build_faraday_rotator(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    rows = _rows(values["rows"], FARADAY_ROW_FIELDS, FaradayResponseRowV1)
    diameter = values["diameter_mm"]
    clear = values.get("clear_aperture_diameter_mm")
    clear = DEFAULT_CLEAR_FRACTION * diameter if clear is None else clear
    if clear > diameter:
        raise invalid("the clear aperture cannot exceed the outer diameter", "clear_aperture_diameter_mm")
    physics: dict[str, Any] = {
        "model_kind": "FARADAY_ROTATOR", "model_version": "1", "rows": rows, "interpolation": values["interpolation"],
        "clear_radius_mm": text(clear / TWO), "edge_radius_mm": text(diameter / TWO), "plane_z_mm": "0",
        "status": FactStatus.DESIGN_BASIS.value,
    }
    if values.get("valid_aoi_deg") is not None:
        a_min, a_max = aoi_range(values["valid_aoi_deg"], "valid_aoi_deg")
        physics["valid_aoi_deg"] = {"min_deg": text(a_min), "max_deg": text(a_max), "status": FactStatus.DESIGN_BASIS.value}
    if values.get("operating_conditions") is not None:
        physics["operating_conditions"] = values["operating_conditions"]
    if values.get("evidence") is not None:
        physics["evidence"] = _evidence(values["evidence"])
    length = values["length_mm"]
    first, last = rows[0], rows[-1]
    model_id = values.get("model_id") or default_model_id("faraday-rotator", Decimal(first["rotation_deg"]), "deg", "w",
                                                          Decimal(first["wavelength_nm"]), Decimal(last["wavelength_nm"]))
    name = values.get("display_name") or f"Faraday rotator {first['wavelength_nm']}-{last['wavelength_nm']} nm"
    model = optical_model(model_id, name, physics, geometry=cylinder(diameter, -length / TWO, length / TWO), parameter="rows")
    design = {"rows": len(rows), "valid_band_nm": [first["wavelength_nm"], last["wavelength_nm"]], "field_axis": "local +Z",
              "rotation_sense": "right-handed about local +Z, whichever way the light travels", "drawn_length_mm": text(length),
              "jones_matrix": "sqrt(T) [[cos b, -sin b], [sin b, cos b]] in the ray's (e1, e2) with e1 x e2 = d, b = sign(d . z) theta(lambda)"}
    return finish("rotator.faraday", model, role="other_optical", values=values, design=design)


# ------------------------------------------------------------------------------------------------ diffraction grating


def _orders(values: Mapping[str, Any]) -> dict[str, list[int]]:
    orders, every = values.get("orders"), values.get("all_propagating_orders")
    if (orders is None) == (every is None):
        raise invalid("state exactly one of orders and all_propagating_orders", "orders")
    if orders is not None:
        if not isinstance(orders, (list, tuple)) or not orders or any(isinstance(m, bool) or not isinstance(m, int) for m in orders):
            raise invalid("orders is a non-empty list of integers, strictly increasing", "orders")
        return {"orders": [int(m) for m in orders]}
    if not isinstance(every, (list, tuple)) or len(every) != 2 or any(isinstance(m, bool) or not isinstance(m, int) for m in every):
        raise invalid("all_propagating_orders is [first, last], two integers", "all_propagating_orders")
    return {"all_propagating_orders": [int(m) for m in every]}


def build_grating(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    return _grating_build("grating.plane", values, {})


def _grating_build(class_id: str, values: Mapping[str, Any], design_extra: dict[str, Any]) -> NativeBuild:
    density, period = values.get("line_density_per_mm"), values.get("period_nm")
    if (density is None) == (period is None):
        raise invalid("state exactly one of line_density_per_mm and period_nm", "line_density_per_mm")
    width, height = values["width_mm"], values["height_mm"]
    clear_width = values.get("clear_width_mm")
    clear_width = DEFAULT_CLEAR_FRACTION * width if clear_width is None else clear_width
    clear_height = values.get("clear_height_mm")
    clear_height = DEFAULT_CLEAR_FRACTION * height if clear_height is None else clear_height
    if clear_width > width or clear_height > height:
        raise invalid("the clear aperture cannot exceed the outer dimensions", "clear_width_mm")
    response = values.get("response")
    if response is not None:
        try:
            response = GratingOrderResponseV1.model_validate(response).model_dump(mode="json", exclude_none=True)
        except ValidationError as exc:
            raise invalid(_first_error(exc), "response") from None
    mode = values["mode"]
    physics: dict[str, Any] = {
        "model_kind": "DIFFRACTION_GRATING", "model_version": "1", "mode": mode, **_orders(values),
        "grating_vector_angle_deg": text(values["grating_vector_angle_deg"]),
        "clear_half_widths_mm": [text(clear_width / TWO), text(clear_height / TWO)], "edge_half_widths_mm": [text(width / TWO), text(height / TWO)],
        "plane_z_mm": "0", "status": FactStatus.DESIGN_BASIS.value,
    }
    physics.update({"line_density_per_mm": text(density)} if density is not None else {"period_nm": text(period)})
    if response is not None:
        physics["response"] = response
    thickness = values.get("thickness_mm") or GRATING_PLATE_THICKNESS_MM
    label = f"{text(density)} lines/mm" if density is not None else f"{text(period)} nm period"
    model_id = values.get("model_id") or default_model_id("grating", mode.lower(), density if density is not None else period)
    name = values.get("display_name") or f"{'Reflection' if mode == 'REFLECTION' else 'Transmission'} grating {label}"
    model = optical_model(model_id, name, physics, geometry=box(width / TWO, height / TWO, -thickness, ZERO), parameter="orders")
    design = {"period_nm": format(model.model.period, "f"), "grooves": "along local +Z x K", "grating_vector": "local +X" if not physics["grating_vector_angle_deg"].strip("0.") else
              f"{physics['grating_vector_angle_deg']} deg from local +X toward local +Y",
              "orders_equation": "n_exit t_exit = n_in t_in + m (lambda / period) K",
              "order_powers": "from the stated response" if response is not None else "not known: no response stated"}
    return finish(class_id, model, role="other_optical", values=values, design={**design, **design_extra})


def _rf_tones_hz(value: Any) -> list[int]:
    """The stated RF tones as whole numbers of ``RF_RESOLUTION_HZ``, increasing."""

    if not isinstance(value, (list, tuple)) or not value:
        raise invalid("rf_tones_mhz is a list of frequencies in MHz", "rf_tones_mhz")
    tones = []
    for item in value:
        tone = to_decimal(item)
        if tone is None or tone <= ZERO:
            raise invalid(f"rf_tones_mhz holds positive frequencies in MHz, not {item!r}", "rf_tones_mhz")
        units = tone * MHZ_TO_HZ / RF_RESOLUTION_HZ
        if units != units.to_integral_value():
            raise invalid(f"{item!r} MHz is finer than the {text(RF_RESOLUTION_HZ)} Hz this class resolves", "rf_tones_mhz")
        tones.append(int(units))
    if len(set(tones)) != len(tones):
        raise invalid("rf_tones_mhz repeats a frequency", "rf_tones_mhz")
    return sorted(tones)


def _aod_response(table: Any, tones: list[int], sign: int, step: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """The grating response of an efficiency table: its order per tone (``rf_mhz`` ``0``: the zero order), the same for both polarizations,
    the same across ``valid_aoi_deg``. Returns the response and its quantity."""

    if not isinstance(table, Mapping):
        raise invalid("efficiency is an object {rows, valid_aoi_deg, data_kind, basis, evidence?}", "efficiency")
    missing = [key for key in ("rows", "valid_aoi_deg", "data_kind", "basis") if key not in table]
    unknown = sorted(set(table) - {"rows", "valid_aoi_deg", "data_kind", "basis", "evidence", "max_conical_angle_deg"})
    if missing or unknown:
        raise invalid(f"efficiency needs {['rows', 'valid_aoi_deg', 'data_kind', 'basis']}, may add evidence and max_conical_angle_deg; missing {missing}, unknown {unknown}", "efficiency")
    aoi = table["valid_aoi_deg"]
    low, high = (to_decimal(aoi[0]), to_decimal(aoi[1])) if isinstance(aoi, (list, tuple)) and len(aoi) == len("ab") else (None, None)
    if low is None or high is None or not low < high:
        raise invalid("efficiency.valid_aoi_deg is [min_deg, max_deg], increasing", "efficiency")
    rows = table["rows"]
    if not isinstance(rows, (list, tuple)) or not rows:
        raise invalid("efficiency.rows is a list of {wavelength_nm, rf_mhz, efficiency}", "efficiency")
    known = {tone: sign * tone // step for tone in tones} | {0: 0}
    cells: dict[tuple[Decimal, int], Decimal] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"wavelength_nm", "rf_mhz", "efficiency"}:
            raise invalid("efficiency.rows is a list of {wavelength_nm, rf_mhz, efficiency}", "efficiency")
        wavelength, rf, value = to_decimal(row["wavelength_nm"]), to_decimal(row["rf_mhz"]), to_decimal(row["efficiency"])
        if wavelength is None or rf is None or value is None:
            raise invalid("every value of efficiency.rows is a number", "efficiency")
        units = rf * MHZ_TO_HZ / RF_RESOLUTION_HZ
        if units != units.to_integral_value() or int(units) not in known:
            raise invalid(f"efficiency.rows names {row['rf_mhz']!r} MHz, which is neither a tone of rf_tones_mhz nor 0 (the zero order)", "efficiency")
        cells[(wavelength, known[int(units)])] = value
    wavelengths = sorted({w for w, _ in cells})
    orders = sorted({m for _, m in cells})
    if len(cells) != len(wavelengths) * len(orders) or not {known[tone] for tone in tones} <= set(orders):
        raise invalid("efficiency.rows states every tone (and the zero order, if it states it) at every wavelength it names", "efficiency")
    response = {"normalization": "ABSOLUTE", "polarization_basis": "TE_TM", "interpolation": "LINEAR", "data_kind": table["data_kind"], "basis": table["basis"],
                "max_conical_angle_deg": text(to_decimal(table.get("max_conical_angle_deg", 0)) or ZERO),
                "rows": [{"wavelength_nm": text(w), "aoi_deg": text(a), "order": m, "efficiency_te": text(cells[(w, m)]), "efficiency_tm": text(cells[(w, m)])}
                         for w in wavelengths for a in (low, high) for m in orders]}
    if "evidence" in table:
        response["evidence"] = table["evidence"]
    return response, {"status": EXTERNALLY_CHARACTERIZED, "unit": None, "source": {"data_kind": table["data_kind"], "basis": table["basis"]},
                      "zero_order_stated": 0 in orders}


def build_acousto_optic(values: Mapping[str, Any], context: BuildContext) -> NativeBuild:
    """An acousto-optic deflector as one grating: period ``v / df``, ``df`` the greatest common divisor of the tones, each tone an integer order of it (module docstring)."""

    velocity = values["acoustic_velocity_m_per_s"]
    tones = _rf_tones_hz(values["rf_tones_mhz"])
    sign = 1 if values["diffraction_order"] == "+1" else -1
    step = 0
    for tone in tones:
        step = gcd(step, tone)
    step_hz = step * RF_RESOLUTION_HZ
    orders = sorted({0, *(sign * tone // step for tone in tones)})
    with localcontext(WORK_CONTEXT):
        period_nm = velocity / step_hz * M_TO_NM
    if period_nm > MAX_AOD_PERIOD_NM:
        raise invalid(f"these tones share a step of only {text(step_hz)} Hz, which makes a grating of period {text(period_nm)} nm and more orders than a run can list; state the tones on a coarser "
                      f"common step (at most a period of {text(MAX_AOD_PERIOD_NM)} nm, so at least {text(velocity / MAX_AOD_PERIOD_NM * M_TO_NM)} Hz)", "rf_tones_mhz")
    efficiency = values.get("efficiency")
    response, power = None, {"status": UNAVAILABLE, "unit": None, "reason": PARAMETER_NOT_STATED, "message": "no efficiency table is stated: each order's power is not known", "missing": ["efficiency"]}
    if efficiency is not None:
        response, power = _aod_response(efficiency, tones, sign, step)
    design_wavelength = values.get("design_wavelength_nm")
    table = []
    for tone in tones:
        hertz = tone * RF_RESOLUTION_HZ
        row = {"rf_mhz": text(hertz / MHZ_TO_HZ), "grating_order": sign * tone // step, "optical_frequency_shift_hz": text(sign * hertz)}
        if design_wavelength is not None:
            with localcontext(WORK_CONTEXT):
                row["direction_sine_at_design_wavelength"] = text(sign * design_wavelength * hertz / (velocity * M_TO_NM))
        table.append(row)
    missing_dynamics = "not modeled: the deflector here is one static grating"
    design = {
        "rf_tone_step_hz": text(step_hz), "tones": table, "zero_order": {"grating_order": 0, "optical_frequency_shift_hz": "0"}, "acoustic_velocity_m_per_s": text(velocity),
        "deflection_equation": "sin(theta) = sin(theta_in) + m lambda f_RF / v, m the diffraction order (+1 or -1) of each tone; each tone is grating order m f_RF / df",
        "optical_frequency_shift": "m f_RF per tone: the optical frequency of that order is the input's plus this. A run does not trace frequency; read each path's grating order against tones",
        "quantities": {
            "direction": {"status": MODELED, "unit": None, "basis": "the grating equation at the period v / df"},
            "order_power": power,
            "optical_frequency_shift_hz": {"status": MODELED, "unit": "Hz", "basis": "m f_RF"},
            "tone_phases": {"status": UNAVAILABLE, "unit": "rad", "reason": "NOT_MODELED", "message": missing_dynamics, "missing": []},
            "timing": {"status": UNAVAILABLE, "unit": "s", "reason": "NOT_MODELED", "message": missing_dynamics, "missing": []},
            "rf_crosstalk": {"status": UNAVAILABLE, "unit": None, "reason": "NOT_MODELED", "message": missing_dynamics, "missing": []},
            "intermodulation": {"status": UNAVAILABLE, "unit": None, "reason": "NOT_MODELED", "message": missing_dynamics, "missing": []},
        },
    }
    grating = {
        "mode": "TRANSMISSION", "period_nm": period_nm, "orders": orders, "grating_vector_angle_deg": values["acoustic_axis_angle_deg"], "width_mm": values["width_mm"],
        "height_mm": values["height_mm"], "clear_width_mm": values.get("clear_width_mm"), "clear_height_mm": values.get("clear_height_mm"), "thickness_mm": values.get("thickness_mm"),
        "response": response, "instance_id": values.get("instance_id"),
        "model_id": values.get("model_id") or default_model_id("aod", velocity, f"{len(tones)}tones", tones[0] * RF_RESOLUTION_HZ / MHZ_TO_HZ),
        "display_name": values.get("display_name") or f"Acousto-optic deflector, {len(tones)} tone{'' if len(tones) == 1 else 's'}",
    }
    return _grating_build("deflector.acousto_optic", grating, design)


__all__ = ["FARADAY_INTERPOLATIONS", "FOCAL_INTERPOLATIONS", "GRATING_MODES", "ND_LAWS", "THIN_LENS_POLARIZATIONS", "TRANSMITTANCE_INTERPOLATIONS",
           "AOD_ORDERS", "build_acousto_optic", "build_faraday_rotator", "build_grating", "build_tabulated_thin", "build_variable_nd"]
