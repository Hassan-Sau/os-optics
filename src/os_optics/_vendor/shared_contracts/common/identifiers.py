"""Common identifier primitives and their wire constraints."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints


# The OpenSci identifier. Values are checked, never rewritten; Unicode
# normalization (NFC) is not checked. An identifier carries no control character (C0, DEL or C1): such a value cannot
# be stored or shown faithfully, so it is refused wherever an identifier is read.
OpenSciId = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[^\s/\x00-\x1f\x7f-\x9f][^/\x00-\x1f\x7f-\x9f]*$",
    ),
    Field(description="Stable OpenSci identifier (contains no '/' and no control character)."),
]
ComponentKey = OpenSciId
InstanceKey = OpenSciId
FrameKey = OpenSciId

FrameRef = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=3,
        max_length=257,
        pattern=r"^[^\s/\x00-\x1f\x7f-\x9f][^/\x00-\x1f\x7f-\x9f]*/[^\s/\x00-\x1f\x7f-\x9f][^/\x00-\x1f\x7f-\x9f]*$",
    ),
    Field(description="Instance or fixture key and frame key separated by one '/'."),
]

# The frame-graph root sentinel (the frame graph is an acyclic forest rooted at world). Every other node is a FrameRef ('<instance_or_fixture_key>/
# <frame_key>'); FrameNodeRef is the union used wherever a frame-graph edge may point at the root.
WORLD_FRAME_ID = "world"

FrameNodeRef = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=257,
        pattern=r"^(?:world|[^\s/\x00-\x1f\x7f-\x9f][^/\x00-\x1f\x7f-\x9f]*/[^\s/\x00-\x1f\x7f-\x9f][^/\x00-\x1f\x7f-\x9f]*)$",
    ),
    Field(description="A frame graph node: the 'world' root, or '<instance_or_fixture_key>/<frame_key>'."),
]

SchemaId = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=9,
        max_length=128,
        pattern=r"^opensci\.[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$",
    ),
    Field(description="OpenSci contract schema identifier."),
]

BuildVersion = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=128, pattern=r"^\S+$"),
    Field(description="Opaque non-empty build or package version string."),
]

UuidString = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
    Field(description="Lowercase hyphenated UUID; the exact wire identity, never a UUID object."),
]

# A DesignPart revision number is a positive integer. The wire form stays a STRING, but its syntax
# is a strict positive-integer string: no sign, no fraction, no exponent, no leading zeros, no zero.
RevisionNumberString = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[1-9][0-9]*$"),
    Field(description="Positive integer revision number as a canonical string (no sign/fraction/exponent/leading zeros/zero)."),
]
