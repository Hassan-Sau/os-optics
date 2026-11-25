"""Structured refusals of the native component library.

A native class never guesses a value it was not given and never builds a model it cannot make physically valid: every
problem with a request is reported, each with a stable code, the parameter it concerns and a plain message.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class NativeErrorCode(StrEnum):
    """Why a native component request is refused (stable values).

    ``NATIVE_CLASS_UNKNOWN``: no class with this id (``native_components()`` lists them).
    ``NATIVE_PARAMETER_UNKNOWN``: the class takes no parameter of this name.
    ``NATIVE_PARAMETER_MISSING``: a required parameter is absent.
    ``NATIVE_PARAMETER_INVALID``: a value of the wrong type, outside its allowed range, or inconsistent with another one.
    ``NATIVE_DESIGN_INFEASIBLE``: the parameters are each valid but no physically valid element satisfies them (for
    example a focal length too short for the diameter: a surface would not be defined over the outline, or an edge
    would be thinner than allowed).
    ``NATIVE_CATALOG_DATA_UNAVAILABLE``: a material or coating named by its public catalog id cannot be copied (no
    catalog release on this device, or the release has no such material / coating).
    """

    CLASS_UNKNOWN = "NATIVE_CLASS_UNKNOWN"
    PARAMETER_UNKNOWN = "NATIVE_PARAMETER_UNKNOWN"
    PARAMETER_MISSING = "NATIVE_PARAMETER_MISSING"
    PARAMETER_INVALID = "NATIVE_PARAMETER_INVALID"
    DESIGN_INFEASIBLE = "NATIVE_DESIGN_INFEASIBLE"
    CATALOG_DATA_UNAVAILABLE = "NATIVE_CATALOG_DATA_UNAVAILABLE"


@dataclass(frozen=True)
class NativeProblem:
    code: NativeErrorCode
    message: str
    parameter: str | None = None

    def as_dict(self) -> dict[str, str]:
        data = {"code": self.code.value, "message": self.message}
        if self.parameter is not None:
            data["parameter"] = self.parameter
        return data


class NativeComponentError(ValueError):
    """A native component request that cannot be built; ``problems`` lists every reason found."""

    def __init__(self, problems: list[NativeProblem] | tuple[NativeProblem, ...]) -> None:
        self.problems: tuple[NativeProblem, ...] = tuple(problems)
        super().__init__("; ".join(
            f"{p.code.value}{f' ({p.parameter})' if p.parameter else ''}: {p.message}" for p in self.problems))

    @property
    def codes(self) -> tuple[NativeErrorCode, ...]:
        return tuple(p.code for p in self.problems)


def refuse(code: NativeErrorCode, message: str, parameter: str | None = None) -> NativeComponentError:
    return NativeComponentError([NativeProblem(code, message, parameter)])


def infeasible(message: str, parameter: str | None = None) -> NativeComponentError:
    return refuse(NativeErrorCode.DESIGN_INFEASIBLE, message, parameter)


def invalid(message: str, parameter: str | None = None) -> NativeComponentError:
    return refuse(NativeErrorCode.PARAMETER_INVALID, message, parameter)


__all__ = ["NativeComponentError", "NativeErrorCode", "NativeProblem", "infeasible", "invalid", "refuse"]
