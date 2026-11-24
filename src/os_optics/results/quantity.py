"""The status of one quantity in a stage-by-stage optical chain (generator, pupil, focal field, detector).

Each quantity of ``pupil_field`` and ``detector_response`` says how it is known:

* ``MODELED``: computed here by the stated model, from the values it names;
* ``EXTERNALLY_CHARACTERIZED``: a value stated from outside the model (a datasheet, a measurement, your own setting), used as
  stated, with its ``source`` when one is given;
* ``BOUNDED``: computed from the light known to arrive while light the trace does not state may also arrive: ``values`` counts the known light, ``bound`` says how the
  true value stands to it (``LOWER``: at least ``values``; ``INTERVAL``: between ``lower`` and ``upper``, that light may also interfere; ``NOMINAL``: only the size of that light
  is known, ``untraced_power_W``), and a ``null`` in a yes / no map is not known;
* ``UNAVAILABLE``: not known, with a ``reason`` and, when a stated parameter is what is missing, the ``missing`` parameters.
  Quantities that do not need it keep their values.

Any of them may carry ``value_basis`` (one ``os_optics.contracts.basis.ValueBasis``, in its JSON form): what the number is
where it is not simply the model's value (a lower or upper bound, an estimate) or a caller setting other than the Engine
default produced it. An ``UNAVAILABLE`` quantity carries ``value`` / ``values`` exactly when it carries ``value_basis``.
"""

from __future__ import annotations

from typing import Any

from ..contracts.basis import ValueBasis

MODELED = "MODELED"
EXTERNALLY_CHARACTERIZED = "EXTERNALLY_CHARACTERIZED"
BOUNDED = "BOUNDED"
LOWER = "LOWER"
INTERVAL = "INTERVAL"
NOMINAL = "NOMINAL"
UNAVAILABLE = "UNAVAILABLE"
#: A stated parameter that a quantity needs is absent.
PARAMETER_NOT_STATED = "PARAMETER_NOT_STATED"
#: The share of the light of a map that may be missing from it (``untraced_power_W`` / the known power) above which a value that light can move either way is not stated. At 0.1 the
#: field of that light is 0.32 of the known field, so by interference a pixel holding all the known light reads 0.47 to 1.73 times its value: about a factor of two either way, the
#: most a reading is still useful for. A ghost of an uncoated lens is about 0.002 of the light, an order a table leaves out usually more than 0.1.
MAX_UNTRACED_FRACTION = 0.1
#: Light that is not a beam of unknown power and is more than ``MAX_UNTRACED_FRACTION`` of the map.
UNTRACED_POWER_TOO_LARGE = "UNTRACED_POWER_TOO_LARGE"
#: The plane bounds were not computed (an evaluation that did not run ``plane_bounds``): the light the trace does not
#: follow, and so what may move a value, is not known -- evaluate with ``plane_bounds`` for the value.
NOT_COMPUTED = "NOT_COMPUTED"
#: A beam of known direction whose power is not known reaches the plane (a grating or a deflector with no stated efficiency).
BEAM_POWER_UNKNOWN = "BEAM_POWER_UNKNOWN"
#: A beam whose power is out of model scope reaches the plane (``ArrivalResult.power_out_of_scope_at``: behind a crystal whose
#: second-harmonic generation is out of model scope there, a variable attenuator or transmission map that varies across the beam,
#: a gradient-index rod the beam leaves through its side, an absorbing plane or element part of the beam meets beside its chief
#: ray): the power it carries is the chief ray's, not the light's, so nothing computed from it is stated (the plane's own
#: quantity state is ``OUT_OF_MODEL_SCOPE``).
POWER_OUT_OF_MODEL_SCOPE = "POWER_OUT_OF_MODEL_SCOPE"


#: A beam part of which surfaces take before the plane reaches it (``ArrivalResult.power_taken_before_at``: a bounded share,
#: ``power_taken_before_upper_W``): its power is the whole beam's, the light crossing lies only between that less the bound and
#: it (the plane's quantity state ``BOUNDED``), and where on the plane the light was taken is not known, so no map, field or
#: power of it is stated.
POWER_TAKEN_BEFORE_PLANE = "POWER_TAKEN_BEFORE_PLANE"


def taken_before_surfaces(arrivals: Any) -> list[str]:
    """The surfaces that take a bounded share of any of ``arrivals`` before the plane (``POWER_TAKEN_BEFORE_PLANE``), in order."""

    return sorted({surface for a in arrivals for surface in a.power_taken_before_at})


def taken_before_message(surfaces: list[str], what: str) -> str:
    """Why ``what`` is not stated for a beam part of which ``surfaces`` take before the plane."""

    return (f"{', '.join(surfaces)} {'take' if len(surfaces) > 1 else 'takes'} part of a beam here before the plane "
            f"(power_taken_before_at; the plane's power is BOUNDED): the beam's power is the whole beam's and where the light was "
            f"taken is not known, so {what} is not stated")


def power_scope_surfaces(arrivals: Any) -> list[str]:
    """The surfaces behind which the power of any of ``arrivals`` is out of model scope (``POWER_OUT_OF_MODEL_SCOPE``), in order."""

    return sorted({surface for a in arrivals for surface in a.power_out_of_scope_at})


def power_scope_message(surfaces: list[str], what: str) -> str:
    """Why ``what`` is not stated for light whose power is out of model scope behind ``surfaces``."""

    return (f"the power of a beam here is out of model scope behind {', '.join(surfaces)} (power_out_of_scope_at; the plane's power is "
            f"OUT_OF_MODEL_SCOPE): it is the chief ray's power, not the light's, so {what} is not stated")


def _with_basis(doc: dict[str, Any], value_basis: ValueBasis | None) -> dict[str, Any]:
    """``doc`` with its ``value_basis`` entry (its JSON form), when it has one."""

    if value_basis is not None:
        doc["value_basis"] = value_basis.model_dump(mode="json")
    return doc


def modeled(unit: str | None, basis: str, *, value_basis: ValueBasis | None = None, **values: Any) -> dict[str, Any]:
    """A computed quantity: ``value`` (a scalar) or ``values`` (a map)."""

    return _with_basis({"status": MODELED, "unit": unit, "basis": basis, **values}, value_basis)


def lower_bound(unit: str | None, basis: str, *, value_basis: ValueBasis | None = None, **values: Any) -> dict[str, Any]:
    """A quantity of which only the light known to arrive is counted: the true value is at least this."""

    return _with_basis({"status": BOUNDED, "bound": LOWER, "unit": unit, "basis": basis, **values}, value_basis)


def within(unit: str | None, basis: str, values: Any, lower: Any, upper: Any, *, value_basis: ValueBasis | None = None) -> dict[str, Any]:
    """A quantity of the known light that other light may move either way: between ``lower`` and ``upper``."""

    return _with_basis({"status": BOUNDED, "bound": INTERVAL, "unit": unit, "basis": basis, "values": values, "lower": lower,
                        "upper": upper}, value_basis)


def nominal(unit: str | None, basis: str, *, value_basis: ValueBasis | None = None, **values: Any) -> dict[str, Any]:
    """A quantity of the known light that other light may move either way, by an amount that depends on where it falls (``untraced_power_W``)."""

    return _with_basis({"status": BOUNDED, "bound": NOMINAL, "unit": unit, "basis": basis, **values}, value_basis)


def stated(unit: str | None, value: Any, source: Any = None) -> dict[str, Any]:
    """A value stated from outside the model, with where it comes from."""

    return {"status": EXTERNALLY_CHARACTERIZED, "unit": unit, "value": value, "source": source}


def unavailable(unit: str | None, reason: str, message: str, missing: list[str] | None = None, *,
                value_basis: ValueBasis | None = None, **values: Any) -> dict[str, Any]:
    """A quantity that is not known, and why. With ``value_basis`` it also carries the number computed for it (``value`` /
    ``values``) and what that number is; without one it carries none."""

    if values and value_basis is None:
        raise ValueError("an UNAVAILABLE quantity carries a value only with the value_basis that says what it is")
    if value_basis is not None and not values:
        raise ValueError("a value_basis qualifies a value: give the value it names")
    return _with_basis({"status": UNAVAILABLE, "unit": unit, "reason": reason, "message": message, "missing": missing or [],
                        **values}, value_basis)
