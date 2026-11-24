"""The facts an issue carries as structured fields, by code.

An issue's ``message`` is the English sentence for a person; a local kit or headless Agent reads the facts of
an issue from its ``numerical_context``, never from the sentence. ``ISSUE_FACTS`` lists, for each code a client words or
routes on, the context keys every issue of that code carries: the key, its unit (``None`` for a fraction, a count, a word
or a flag) and what it holds. A key's name ends with its unit (``_deg``, ``_rad``, ``_mm``, ``_W``, ``_nm``) unless it is
dimensionless.

The rule: a producer that states a fact in a message also gives it as a context key; a code whose facts a client needs
is listed here with those keys, in the same change that makes its producers carry them. ``tests/test_issue_facts.py``
runs a scene for every listed code and compares each key with the value computed independently in the test. A test of an
issue asserts the code, the location and these keys, not the sentence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .codes import ErrorCode


@dataclass(frozen=True)
class IssueFact:
    """One context key of an issue: its name, its unit (``None`` when dimensionless) and what it holds."""

    key: str
    unit: str | None
    meaning: str


_TRUNCATION = (IssueFact("loss_fraction", None, "share of the beam's power the aperture takes"),)

ISSUE_FACTS: Mapping[ErrorCode, tuple[IssueFact, ...]] = {
    ErrorCode.GAUSSIAN_TRUNCATION: _TRUNCATION,
    ErrorCode.GAUSSIAN_SEVERE_TRUNCATION: _TRUNCATION,
    ErrorCode.RETARDER_OBLIQUE_INCIDENCE: (
        IssueFact("aoi_deg", "deg", "angle of incidence on the polarizing element"),
        IssueFact("warn_deg", "deg", "angle above which the thin-element polarization model is reported (retarder_warn_aoi_deg)"),
    ),
    ErrorCode.ND_DENSITY_VARIES_ACROSS_BEAM: (
        IssueFact("od_min", None, "least optical density across the beam's footprint"),
        IssueFact("od_max", None, "greatest optical density across the beam's footprint"),
        IssueFact("od_variation", None, "od_max - od_min"),
        IssueFact("od_variation_tolerance", None, "variation above which the chief ray's transmittance does not stand for the beam"),
    ),
}


def issue_fact_keys(code: ErrorCode) -> tuple[str, ...]:
    """The context keys every issue of ``code`` carries (empty for a code not listed)."""

    return tuple(fact.key for fact in ISSUE_FACTS.get(code, ()))
