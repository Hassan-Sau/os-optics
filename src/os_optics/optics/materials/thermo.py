"""A uniaxial crystal's indices at its temperature (``CrystalTemperature``): ``n(lambda, T) = n(lambda, T_ref) + dn/dT(lambda)
(T - T_ref)``, ``dn/dT`` linear in the vacuum wavelength between the stated rows (one row: the same at every wavelength).

The offset adds to the group index ``n_g = n - lambda dn/dlambda`` as ``delta - lambda d delta / d lambda`` (``delta`` the
offset; its wavelength slope is the rows' slope times ``T - T_ref``) and nothing to the group-delay dispersion (piecewise linear
in wavelength)."""

from __future__ import annotations

from ...contracts.materials import CrystalTemperature


def coefficients(temperature: CrystalTemperature, wavelength_nm: float) -> tuple[float, float, float, float] | None:
    """``(dn_o/dT, dn_e/dT, d/dlambda dn_o/dT, d/dlambda dn_e/dT)`` at ``wavelength_nm`` (per K and per K per nm); ``None``
    outside the rows' span (two rows or more)."""

    rows = temperature.thermo_optic
    if len(rows) == 1:
        return rows[0].dno_dT_per_K, rows[0].dne_dT_per_K, 0.0, 0.0
    if wavelength_nm < rows[0].wavelength_nm or wavelength_nm > rows[-1].wavelength_nm:
        return None
    k = next(i for i in range(len(rows) - 1) if wavelength_nm <= rows[i + 1].wavelength_nm)
    lo, hi = rows[k], rows[k + 1]
    span = hi.wavelength_nm - lo.wavelength_nm
    t = (wavelength_nm - lo.wavelength_nm) / span
    slope_o = (hi.dno_dT_per_K - lo.dno_dT_per_K) / span
    slope_e = (hi.dne_dT_per_K - lo.dne_dT_per_K) / span
    return lo.dno_dT_per_K + t * (hi.dno_dT_per_K - lo.dno_dT_per_K), lo.dne_dT_per_K + t * (hi.dne_dT_per_K - lo.dne_dT_per_K), slope_o, slope_e


def offsets(temperature: CrystalTemperature, wavelength_nm: float, temperature_c: float | None = None
            ) -> tuple[float, float, float, float] | None:
    """``(delta_o, delta_e, d delta_o / d lambda, d delta_e / d lambda)`` of the two indices at ``temperature_c`` (default:
    the crystal's), per nm for the slopes; ``None`` outside the rows' span."""

    got = coefficients(temperature, wavelength_nm)
    if got is None:
        return None
    dt = (temperature.temperature_c if temperature_c is None else temperature_c) - temperature.reference_temperature_c
    return got[0] * dt, got[1] * dt, got[2] * dt, got[3] * dt


__all__ = ["coefficients", "offsets"]
