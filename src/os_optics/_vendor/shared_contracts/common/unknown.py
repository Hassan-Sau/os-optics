"""Explicit unknown-value and presence policy primitives."""

from enum import StrEnum


class UnknownValuePolicy(StrEnum):
    """The unknown-value policy of a domain enum."""

    REJECT = "reject"
    UNKNOWN_PRESERVE = "unknown_preserve"


class ValueState(StrEnum):
    """Explicit states that must not be collapsed into field omission."""

    ABSENT = "absent"
    NULL = "null"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
