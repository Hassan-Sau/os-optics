"""The text of the result schema stays what a schema is for.

A description of ``ValidationResult``'s JSON schema states what a field or record is, its unit, its operands and when it
is null. A model, a derivation, a list of cases or validation evidence belongs to the record's documentation (the class
docstring, which ``help()`` shows), which the description names. These budgets keep the text from growing back: raising
one is a decision to make the schema carry more, taken with the text in view.
"""

from __future__ import annotations

import inspect

from os_optics.contracts import ValidationResult
from os_optics.contracts import results

#: The longest description of the schema, in characters (the longest today: a record of 2814).
MAX_DESCRIPTION_CHARS = 3000
#: The words of all descriptions together (11983 when this budget was set at 12600; 13078 when raised, with the text in
#: view: the followed light of a plane and its bounds, the reasons an element's utilization is open, the issue text
#: contract, the execution outcome of a run whose auxiliary budget ran out -- 1101 words of new fields and records; 13723
#: when raised to 13800, with the text in view: the value_basis of a sampled line at a plane and of a polarization
#: mixture, where a model-scope finding reaches them -- 215 words, D5b).
MAX_TOTAL_WORDS = 13800
#: The exposure records, whose model is long: their schema description is the meaning, at most this many characters.
MAX_EXPOSURE_RECORD_CHARS = 700
EXPOSURE_RECORDS = ("ExposureReport", "ElementExposure", "SurfaceExposure", "ExposureHit", "RatingComparison", "RatingCondition")


def _descriptions(node, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        if isinstance(node.get("description"), str):
            found.append((path, node["description"]))
        for key, value in node.items():
            found.extend(_descriptions(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_descriptions(value, f"{path}[{index}]"))
    return found


def test_no_description_of_the_result_schema_exceeds_the_budget():
    rows = _descriptions(ValidationResult.model_json_schema())
    assert len(rows) > 200
    too_long = sorted(((len(text), path) for path, text in rows if len(text) > MAX_DESCRIPTION_CHARS), reverse=True)
    assert too_long == [], "state the meaning in the schema and move the model to the record's documentation"
    words = sum(len(text.split()) for _path, text in rows)
    assert words <= MAX_TOTAL_WORDS, f"{words} words of schema description (budget {MAX_TOTAL_WORDS})"


def test_an_exposure_record_s_schema_states_its_meaning_and_names_its_documentation():
    definitions = ValidationResult.model_json_schema()["$defs"]
    for name in EXPOSURE_RECORDS:
        description = definitions[name]["description"]
        assert len(description) <= MAX_EXPOSURE_RECORD_CHARS, name
        assert f"help(os_optics.contracts.{name})" in description, name  # (where the model is stated, for a wheel's reader)
        # the model itself stays with the class: what the schema no longer carries is still stated once
        assert len(inspect.cleandoc(getattr(results, name).__doc__)) > len(description), name
        # (the description only: no field of the record changed)
        assert set(definitions[name]["properties"]) == set(getattr(results, name).model_fields), name


def test_a_schema_description_does_not_change_the_record():
    """``schema_description`` replaces the description of the JSON schema only: the record still refuses unknown fields and
    non-finite numbers and is immutable, as every result record is."""

    for name in EXPOSURE_RECORDS:
        config = getattr(results, name).model_config
        assert (config["frozen"], config["extra"], config["allow_inf_nan"]) == (True, "forbid", False), name
