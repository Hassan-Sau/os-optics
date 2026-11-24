"""Hash reference contracts, and the osc1 exact canonical-data hash profile.

``osc1 = SHA-256(UTF-8(JCS(value)))`` (RFC 8785 JSON Canonicalization Scheme,
no byte prefix) is implemented in :mod:`shared_contracts.common.osc1`; this
module re-exports its public surface so callers only need to import
``shared_contracts.common.hashes``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import Field, StringConstraints

from ..base import StrictContractModel
from .frozen_string_map import FrozenStringMap
from .identifiers import OpenSciId
from .osc1 import (
    MAX_SAFE_INTEGER,
    MIN_SAFE_INTEGER,
    Osc1Error,
    Osc1IntegerOutOfRangeError,
    Osc1InvalidObjectKeyError,
    Osc1InvalidUnicodeError,
    Osc1NonFiniteFloatError,
    Osc1UnsupportedTypeError,
    canonicalize_osc1_bytes,
    canonicalize_osc1_text,
    osc1_digest,
)


__all__ = [
    "FrozenStringMap",
    "HashProfile",
    "HashRef",
    "MAX_SAFE_INTEGER",
    "MIN_SAFE_INTEGER",
    "Osc1Error",
    "Osc1IntegerOutOfRangeError",
    "Osc1InvalidObjectKeyError",
    "Osc1InvalidUnicodeError",
    "Osc1NonFiniteFloatError",
    "Osc1UnsupportedTypeError",
    "Sha256Digest",
    "canonicalize_osc1_bytes",
    "canonicalize_osc1_text",
    "osc1_digest",
    "osc1_ref",
]


class HashProfile(StrEnum):
    """Closed hash-profile taxonomy; unknown profiles are rejected."""

    OSC1 = "osc1"
    ENGINE_EXACT_PY1 = "eng-exact-py1"
    ENGINE_QUANTIZED_PY1 = "eng-quant-py1"
    SHA256_BYTES = "sha256-bytes"


Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
    Field(description="Lowercase hexadecimal SHA-256 digest."),
]


# `params` keeps its original wire shape (`object<string,string> | null`, e.g. the `eng-quant-py1`
# parameters) exactly: this is a normative contract and must not become a breaking wire/schema
# change. Deep immutability (`frozen=True` alone does not make a `dict`-valued field hashable) is solved internally by
# `FrozenStringMap`, a tuple-backed `Mapping[str, str]` with a custom Pydantic core schema and JSON
# Schema that serializes and validates as a plain JSON object, byte-identical to what a bare
# `dict[str, str] | None` field produced. Callers still write ``params={"sig_digits": "9", ...}``;
# reading `hash_ref.params["sig_digits"]` or iterating it works exactly like a `dict`. The
# class docstring is kept short on purpose: Pydantic copies `__doc__` verbatim into the JSON Schema
# `description` of every schema that embeds `HashRef`.
class HashRef(StrictContractModel):
    """A profile-qualified reference to a hashed subject."""

    profile: HashProfile
    digest: Sha256Digest
    subject: OpenSciId
    params: FrozenStringMap | None = None


def osc1_ref(value: Any, subject: OpenSciId) -> HashRef:
    """Trivial convenience: hash ``value`` with osc1 and wrap it as a ``HashRef``.

    Does not obscure the canonical bytes -- it is exactly
    ``HashRef(profile=HashProfile.OSC1, digest=osc1_digest(value), subject=subject)``.
    """

    return HashRef(profile=HashProfile.OSC1, digest=osc1_digest(value), subject=subject)


def require_hash_profile(value: HashRef, profile: HashProfile, *, field_name: str, allow_params: bool = False) -> HashRef:
    """Reusable per-field invariant: a ``HashRef``-typed field with fixed semantics must reject the
    wrong hash profile rather than accepting any of the four. Generic ``HashRef`` itself is
    unchanged; call this from a field validator on the owning model.

    ``allow_params`` defaults to ``False``: no ``params`` entry is defined for
    ``osc1``, ``sha256-bytes`` or ``eng-exact-py1`` (only ``eng-quant-py1`` legitimately carries
    ``sig_digits``/``abs_floor``). Without this, an arbitrary ``params`` entry could give the same
    fixed digest two different wire identities.
    """

    if value.profile is not profile:
        raise ValueError(f"{field_name} must use the {profile.value!r} hash profile, got {value.profile.value!r}")
    if not allow_params and value.params is not None:
        raise ValueError(f"{field_name} ({profile.value!r}) has no profile parameters; params must be omitted")
    return value


ENGINE_QUANTIZED_PARAM_KEYS = frozenset({"sig_digits", "abs_floor"})


def require_engine_quantized_params(value: HashRef, *, field_name: str) -> HashRef:
    """Opt-in, per-field invariant for the specific fields that cite ``eng-quant-py1``, whose
    ``sig_digits`` and ``abs_floor`` must always be recorded. Requires ``profile == eng-quant-py1`` (via
    :func:`require_hash_profile`) AND ``params`` present with EXACTLY the keys ``{"sig_digits",
    "abs_floor"}`` -- neither missing.

    This does NOT change :class:`HashRef`, :func:`require_hash_profile`, or any other ``eng-quant-py1``
    citation: it is called explicitly, per field, only where a contract chooses it. ``HashRef``
    itself stays a generic citation contract in which ``params`` remains optional even for
    ``eng-quant-py1``.
    """

    value = require_hash_profile(value, HashProfile.ENGINE_QUANTIZED_PY1, field_name=field_name, allow_params=True)
    if value.params is None:
        raise ValueError(f"{field_name} (eng-quant-py1) must carry params {{sig_digits, abs_floor}}, got none")
    keys = set(value.params.keys())
    if keys != ENGINE_QUANTIZED_PARAM_KEYS:
        raise ValueError(
            f"{field_name} (eng-quant-py1) params keys must be exactly {sorted(ENGINE_QUANTIZED_PARAM_KEYS)}, got {sorted(keys)}"
        )
    return value
