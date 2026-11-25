"""Beam path ids: ``<source_id>#<spectral index>`` for a source line, then one ``.<k>`` per branch (``k`` counts the
branches a path creates, in creation order), so an id depends only on the geometry and the order of a path's own
interactions, never on queue order or Python hashing. The line of a partially polarized source is two mutually incoherent
polarization components (``optics/polarization/partial.py``): the first keeps the line's id, the second, orthogonal one is
``<source_id>#<spectral index>:p1`` (``COMPONENT_MARK``), and its branches append ``.<k>`` to that.

The branch part (everything after ``<source_id>#<spectral index>``) is kept at most ``BRANCH_SUFFIX_MAX_LENGTH``
characters: a branch whose suffix would be longer gets ``~<digest>`` instead, the first 32 hexadecimal digits (128 bits) of
the SHA-256 of the suffix it would have had, and its own branches append ``.<k>`` to that again. The suffix of every
parent is unique among the paths of its source line, so the compact suffix is unique too (up to a 128-bit digest
collision), and an uncompacted suffix holds only ``.`` and digits, so the two kinds never coincide. The digest does not see the
source id or the line index, so a line traced under another index (the network export's probes launch the selected line as
line 0) gets the same suffix.

A path id is therefore at most ``len(source_id) + 1 + digits(spectral index) + len(COMPONENT_MARK) +
BRANCH_SUFFIX_MAX_LENGTH`` characters: ``PATH_ID_MAX_LENGTH`` (256) for a source id of at most 128 characters and a spectral
index below 10**28 (a source with that many lines does not fit in memory), whatever the trace limits. Deep branch trees appear with partially reflecting surfaces
facing each other: a two-splitter cavity at R = 0.9 is followed through about 240 round trips before its power falls below
``power_epsilon_W``.
"""

from __future__ import annotations

import hashlib

#: The branch part of a path id is kept at most this long (``.0`` per level: 48 levels before the first compaction).
BRANCH_SUFFIX_MAX_LENGTH = 96
#: Hexadecimal digits of the SHA-256 digest of a compacted branch suffix (128 bits).
DIGEST_HEX_DIGITS = 32
#: Marks a compacted branch suffix.
COMPACT_MARK = "~"
#: The longest path id the Engine makes for a source id of at most 128 characters (the scene contract's limit).
PATH_ID_MAX_LENGTH = 256
#: Marks the root of the second (orthogonal) polarization component of a partially polarized source's line.
COMPONENT_MARK = ":p1"
#: The part of every path id the Engine makes after ``<source_id>#<spectral index>`` (regular expression): the optional
#: polarization component mark, an optional compacted suffix, then ``.<k>``s.
BRANCH_SUFFIX_PATTERN = rf"(?:{COMPONENT_MARK})?(?:{COMPACT_MARK}[0-9a-f]{{{DIGEST_HEX_DIGITS}}})?(?:\.\d+)*"


def root_path_id(source_id: str, spectral_index: int, polarization_component: int | None = None) -> str:
    """The id of a source line's root path (of its second polarization component when ``polarization_component`` is 1)."""
    return f"{source_id}#{spectral_index}" + (COMPONENT_MARK if polarization_component == 1 else "")


def child_path_id(parent_path_id: str, root: str, k: int) -> str:
    """The id of the ``k``-th branch (0, 1, ...) that path ``parent_path_id`` creates; ``root`` is its source line's root id."""
    suffix = f"{parent_path_id[len(root):]}.{k}"
    if len(suffix) > BRANCH_SUFFIX_MAX_LENGTH:
        suffix = COMPACT_MARK + hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:DIGEST_HEX_DIGITS]
    return root + suffix
