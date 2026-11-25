"""The retained products of an evaluation: ``EvaluationState``.

An evaluation runs its capabilities over a state. A caller that holds the state and passes it to the next evaluation
(``os_optics.evaluate(request, state=state)``) reuses every product that is still valid: the products belong to one
physical identity -- the Engine build, the design's physical inputs (baseplate, components, observation planes,
materials), the numerical configuration and the schema version -- and to the parameters of the capability that made each
one. An evaluation of another identity drops them first; a changed target threshold is not part of the identity, so it
reuses every physical product. The state is an explicit object: the Engine keeps none of its own, and two evaluations
share products only through a state the caller passes to both, one evaluation at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capabilities import CAPABILITIES

#: Why a state dropped products (``EvaluationState.invalidations``): another Engine build, other physical inputs, another
#: numerical configuration, another schema version, a capability run with other parameters, another observer of the trace.
BUILD_CHANGED = "engine_build"
INPUTS_CHANGED = "inputs"
NUMERICAL_CHANGED = "numerical_configuration"
SCHEMA_CHANGED = "schema_version"
PARAMETERS_CHANGED = "parameters"
RECORDER_CHANGED = "recorder"


@dataclass(frozen=True)
class PhysicalIdentity:
    """What a state's products are valid for: the Engine build that computed them, the hash of the design's physical
    inputs (baseplate, components, observation planes and materials: ``ValidationManifest.design_projection_hash``; the
    targets are no part of it), the hash of the numerical configuration and the request's schema version."""

    engine_build_id: str
    inputs_hash: str
    numerical_hash: str
    schema_version: str


def _dependents() -> dict[str, tuple[str, ...]]:
    direct: dict[str, set[str]] = {row.id: set() for row in CAPABILITIES}
    for row in CAPABILITIES:
        for need in (*row.requires, *row.traced_requires, *row.rated_requires):
            direct[need].add(row.id)
    out: dict[str, tuple[str, ...]] = {}
    for row in CAPABILITIES:
        found: set[str] = set()
        pending = [row.id]
        while pending:
            for dependent in direct[pending.pop()]:
                if dependent not in found:
                    found.add(dependent)
                    pending.append(dependent)
        out[row.id] = tuple(c.id for c in CAPABILITIES if c.id in found)
    return out


#: Per capability, every capability that requires its product for some request (``requires``, ``traced_requires``,
#: ``rated_requires``), directly or through another (execution order): dropping a product drops theirs. A product a
#: capability reads only when it is in the evaluation is keyed by that reader's parameters as well (the surface exposures:
#: which readers of the return-light analysis came before them, ``validation/pipeline.py``).
DEPENDENTS: dict[str, tuple[str, ...]] = _dependents()


class EvaluationState:
    """The products of the capabilities an evaluation executed, kept for the next evaluation of the same physical identity.

    ``executions``: how often each capability ran over this state's lifetime (a capability whose product was reused did
    not run). ``invalidations``: how often products were dropped, by reason. ``last_executed`` / ``last_reused``: the
    capabilities the latest evaluation ran and those whose retained product it read. ``build_id``: the Engine build id
    the state's products belong to, computed once by the first evaluation (a caller may state it instead). ``retained``:
    the capabilities with a product. A capability stores its product only when it finished: an evaluation stopped part
    way leaves the products of the capabilities completed before, and the next evaluation continues from them.
    """

    def __init__(self, build_id: str | None = None) -> None:
        self.build_id = build_id
        self.identity: PhysicalIdentity | None = None
        self.executions: dict[str, int] = {}
        self.invalidations: dict[str, int] = {}
        self.last_executed: tuple[str, ...] = ()
        self.last_reused: tuple[str, ...] = ()
        self.recorder: Any = None
        self._products: dict[str, Any] = {}
        self._parameters: dict[str, Any] = {}

    @property
    def retained(self) -> tuple[str, ...]:
        return tuple(row.id for row in CAPABILITIES if row.id in self._products)

    def bind(self, identity: PhysicalIdentity) -> None:
        """Make ``identity`` the state's: the products of another identity are dropped first, counted by what differs."""
        held = self.identity
        if held is not None and held != identity:
            for reason, same in ((BUILD_CHANGED, held.engine_build_id == identity.engine_build_id),
                                 (INPUTS_CHANGED, held.inputs_hash == identity.inputs_hash),
                                 (NUMERICAL_CHANGED, held.numerical_hash == identity.numerical_hash),
                                 (SCHEMA_CHANGED, held.schema_version == identity.schema_version)):
                if not same:
                    self._count(reason)
            self.clear()
        self.identity = identity

    def clear(self) -> None:
        """Drop every product (the counters stay)."""
        self._products.clear()
        self._parameters.clear()
        self.recorder = None

    def holds(self, capability: str, parameters: Any = ()) -> bool:
        """Whether the capability's product is retained and was made with these parameters."""
        return capability in self._products and self._parameters[capability] == parameters

    def has(self, capability: str) -> bool:
        return capability in self._products

    def product(self, capability: str) -> Any:
        """The capability's retained product (``KeyError`` when it holds none)."""
        return self._products[capability]

    def store(self, capability: str, parameters: Any, product: Any) -> None:
        """Keep the product of a capability that finished, and count its execution."""
        self._products[capability] = product
        self._parameters[capability] = parameters
        self.executions[capability] = self.executions.get(capability, 0) + 1

    def count(self, capability: str) -> None:
        """Count the execution of a capability whose product is not retained (the requirements: every evaluation assesses
        its targets anew)."""
        self.executions[capability] = self.executions.get(capability, 0) + 1

    def discard(self, capability: str, reason: str = PARAMETERS_CHANGED) -> None:
        """Drop the capability's product and the product of everything that read it (counted once when anything was held)."""
        dropped = False
        for name in (capability, *DEPENDENTS[capability]):
            if name in self._products:
                del self._products[name]
                del self._parameters[name]
                dropped = True
        if dropped:
            self._count(reason)

    def _count(self, reason: str) -> None:
        self.invalidations[reason] = self.invalidations.get(reason, 0) + 1


__all__ = ["BUILD_CHANGED", "DEPENDENTS", "INPUTS_CHANGED", "NUMERICAL_CHANGED", "PARAMETERS_CHANGED", "RECORDER_CHANGED",
           "SCHEMA_CHANGED", "EvaluationState", "PhysicalIdentity"]
