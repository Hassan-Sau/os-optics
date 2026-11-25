"""The text contract of ``ValidationIssue`` (``errors/issues.py``): ``message`` states the finding, ``suggested_action`` is a
step inside the tool that obtains the missing value or decision, absent where none exists, and the action is never part of
the message. Checked structurally over the fixture scenes and over scenes that raise the notices whose message once ended
with their action (withheld bounds: ``test_closed_return_search.py``; a detector's measured facts:
``test_diagnostic_outcomes.py``)."""

from __future__ import annotations

import pytest
from fixture_scenes import FIXTURES

from os_optics import validate
from os_optics.errors import ValidationIssue


def _violations(issues) -> list[tuple[str, str | None]]:
    return [(i.code.value, i.location) for i in issues if i.suggested_action and i.suggested_action in i.message]


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_no_fixture_issue_repeats_its_action_in_its_message(fixture):
    result = validate(FIXTURES[fixture]())
    assert not _violations(result.issues)


def test_the_contract_is_stated_on_the_fields():
    fields = ValidationIssue.model_fields
    assert "never the action" in fields["message"].description
    assert "inside the tool" in fields["suggested_action"].description

