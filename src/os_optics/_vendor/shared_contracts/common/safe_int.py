"""Safe JSON integer primitive for fields that are semantically integers on the wire.

``osc1`` itself already rejects a Python ``int`` outside the exactly
IEEE-754-representable range (``MAX_SAFE_INTEGER`` / ``MIN_SAFE_INTEGER``,
``common/osc1.py``). ``SafeJsonInt`` pushes that same bound to contract
*validation* time, so a schema consumer sees the limit in the generated JSON
Schema's ``minimum``/``maximum`` rather than discovering it only when hashing
fails. Used for every semantically-integer contract field (definition
versions, counts, indexes). Not used for ``revision``, which is a
``DecimalString`` on the wire, never a JSON integer.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .osc1 import MAX_SAFE_INTEGER, MIN_SAFE_INTEGER


SafeJsonInt = Annotated[
    int,
    Field(
        strict=True,
        ge=MIN_SAFE_INTEGER,
        le=MAX_SAFE_INTEGER,
        description="Integer safely representable as an IEEE-754 double / JSON number.",
    ),
]
