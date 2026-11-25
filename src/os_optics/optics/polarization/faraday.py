"""The stated response of a Faraday rotator (``FaradayRotatorInteraction``): rotation angle and power transmittance of one
traced line, or nothing where the response is not stated.

``linear``: both values linear in the vacuum wavelength between the two rows around it, over the rows' span. ``points_only``
(and a one-row table): the rows' wavelengths exactly. Nothing is extrapolated and nothing is held from a neighbouring row:
a wavelength outside the domain has no response. Interpolation uses ``+ - * /`` only, so every runtime computes the same
bits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaradayResponse:
    """The table of a Faraday rotator in the lowered scene (see ``contracts.optics.FaradayRotatorInteraction``)."""

    wavelength_nm: tuple[float, ...]
    rotation_rad: tuple[float, ...]
    transmittance: tuple[float, ...]
    interpolation: str  # "linear" | "points_only"

    def at(self, wavelength_nm: float) -> tuple[float, float] | None:
        """``(rotation_rad, transmittance)`` of one traced line, ``None`` outside the domain."""

        rows = self.wavelength_nm
        for i, row in enumerate(rows):
            if wavelength_nm == row:
                return self.rotation_rad[i], self.transmittance[i]
        if self.interpolation != "linear" or wavelength_nm < rows[0] or wavelength_nm > rows[-1] or len(rows) == 1:
            return None
        i = next(k for k in range(len(rows) - 1) if wavelength_nm < rows[k + 1])
        s = (wavelength_nm - rows[i]) / (rows[i + 1] - rows[i])
        return (self.rotation_rad[i] + (self.rotation_rad[i + 1] - self.rotation_rad[i]) * s,
                self.transmittance[i] + (self.transmittance[i + 1] - self.transmittance[i]) * s)

    def domain_text(self) -> str:
        """The wavelengths the response is stated for, in words."""

        if self.interpolation == "linear" and len(self.wavelength_nm) > 1:
            return f"{self.wavelength_nm[0]!r}-{self.wavelength_nm[-1]!r} nm"
        return ", ".join(f"{w!r}" for w in self.wavelength_nm) + " nm exactly"


@dataclass(frozen=True)
class RetardanceTable:
    """A retarder's stated retardance per wavelength (``RetarderInteraction.retardance_table_nm``): the same rows rule as a
    Faraday rotator's response."""

    wavelength_nm: tuple[float, ...]
    retardance_rad: tuple[float, ...]
    interpolation: str  # "linear" | "points_only"

    def _response(self) -> FaradayResponse:
        return FaradayResponse(self.wavelength_nm, self.retardance_rad, tuple(1.0 for _ in self.wavelength_nm), self.interpolation)

    def at(self, wavelength_nm: float) -> float | None:
        """The retardance (rad) of one traced line, ``None`` outside the domain."""
        found = self._response().at(wavelength_nm)
        return None if found is None else found[0]

    def domain_text(self) -> str:
        return self._response().domain_text()


__all__ = ["FaradayResponse", "RetardanceTable"]
