"""Targets on a crystal's second-harmonic generation (``NonlinearConversionResult``).

``shg_conversion_efficiency`` (and its general name ``nonlinear_conversion_efficiency``): the harmonic power the element
``device_id`` generates from the selected fundamental line (``source_id``, ``wavelength_nm``; default: every line that
enters it) divided by that line's fundamental power entering it, summed over every traced pass of that light into the
crystal (one pass: its ``conversion_efficiency``). ``shg_power_W``: the harmonic power generated, summed the same way.

No value -- ``UNKNOWN`` -- when no traced beam of the selection enters the element (``TARGET_SELECTION_EMPTY``), or when
part of the selected lines' power has an unknown fate (the trace budget, the interaction limit, a path ended out of model
scope; the codes of ``validation/power.py``): that light may enter the crystal too. A selected pass with unstated bulk
absorption affecting the entering pump or the harmonic in the crystal returns ``UNKNOWN`` with ``BULK_LOSS_NOT_ESTABLISHED``.
``OUT_OF_MODEL_SCOPE`` with ``NONLINEAR_CONVERSION_OUT_OF_SCOPE`` when a selected pass is outside the second-harmonic model's
conditions (its reason is in the message). The harmonic power counts what the crystal generates at its exit face, after the
crystal's absorption.
"""

from __future__ import annotations

from typing import NamedTuple

from ..config import NumericalConfig
from ..contracts.project import Target
from ..contracts.results import NonlinearConversionResult, PowerBudget
from ..errors import ErrorCode, Status
from ..math.vectors import correctly_rounded_sum
from .power import line_fate

#: metric -> unit.
NONLINEAR_METRICS: dict[str, str] = {
    "shg_conversion_efficiency": "1",
    "nonlinear_conversion_efficiency": "1",
    "shg_power_W": "W",
}


class ConversionValue(NamedTuple):
    """A nonlinear metric's value, unit, measurement status, codes and message (``metrics.MetricValue`` fields)."""

    value: float | None
    unit: str
    status: Status
    codes: tuple[str, ...] = ()
    message: str = ""


def _selected(t: Target, conversions: tuple[NonlinearConversionResult, ...], cfg: NumericalConfig) -> list[NonlinearConversionResult]:
    tol = cfg.interference_wavelength_tol_nm
    return [c for c in conversions if c.element_id == t.device_id and (t.source_id is None or c.source_id == t.source_id)
            and (t.wavelength_nm is None or abs(c.wavelength_nm - t.wavelength_nm) <= tol)]


def nonlinear_value(t: Target, conversions: tuple[NonlinearConversionResult, ...], power_budget: PowerBudget | None,
                    cfg: NumericalConfig) -> ConversionValue:
    """The value of ``t`` (a ``NONLINEAR_METRICS`` target) from the run's conversions (module docstring)."""

    unit = NONLINEAR_METRICS[t.metric]
    if t.device_id is None:
        return ConversionValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                               f"{t.metric} names the crystal by device_instance_id (device_id in an Engine request)")
    passes = _selected(t, conversions, cfg)
    if not passes:
        crystals = sorted({c.element_id for c in conversions})
        have = f"; the run's converting crystals: {', '.join(crystals)}" if crystals else "; no beam of this run enters such a crystal"
        return ConversionValue(None, unit, Status.UNKNOWN, (ErrorCode.TARGET_SELECTION_EMPTY.value,),
                               f"no traced beam of the selected light enters a crystal {t.device_id!r} that generates a second "
                               f"harmonic{have}")
    scoped = [c for c in passes if c.status is Status.OUT_OF_MODEL_SCOPE]
    if scoped:
        return ConversionValue(None, unit, Status.OUT_OF_MODEL_SCOPE, (ErrorCode.NONLINEAR_CONVERSION_OUT_OF_SCOPE.value,),
                               f"the second harmonic of {scoped[0].path_id} in {t.device_id} is out of model scope: {scoped[0].reason}")
    unknown = next((c for c in passes if c.status is Status.UNKNOWN), None)
    if unknown is not None:
        return ConversionValue(None, unit, Status.UNKNOWN, (ErrorCode.BULK_LOSS_NOT_ESTABLISHED.value,),
                               f"the second harmonic of {unknown.path_id} in {t.device_id} is unknown: {unknown.reason}")
    lines = {(c.source_id, c.spectral_index) for c in passes}
    if power_budget is not None:
        fate = line_fate([line for line in power_budget.lines if (line.source_id, line.spectral_index) in lines], cfg)
        if not fate.complete:
            return ConversionValue(None, unit, Status.UNKNOWN, fate.cause_codes(),
                                   f"part of the selected light has an unknown fate and may enter {t.device_id} too: {fate.describe()}")
    harmonic = correctly_rounded_sum(c.harmonic_power_W or 0.0 for c in passes)
    if t.metric == "shg_power_W":
        return ConversionValue(harmonic, unit, Status.PASS, (), f"{t.device_id} generates {harmonic:.6g} W of second harmonic "
                               f"over {len(passes)} pass(es)")
    fundamental = correctly_rounded_sum(c.fundamental_power_W for c in passes if c.fundamental_power_W is not None)
    value = harmonic / fundamental if fundamental > 0.0 else 0.0
    return ConversionValue(value, unit, Status.PASS, (), f"{t.device_id} converts {value:.6g} of the fundamental entering it "
                           f"({harmonic:.6g} W of {fundamental:.6g} W over {len(passes)} pass(es))")


__all__ = ["NONLINEAR_METRICS", "ConversionValue", "nonlinear_value"]
