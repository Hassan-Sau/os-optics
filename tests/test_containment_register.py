"""The containment register (``docs/engine/audit/CONTAINMENT_REGISTER.md``) lists every place where a beam's power is marked as
not the light's and every test that holds such a finding; this module keeps the register and the code in step.

* Section 1 has one row for each method of ``optics/rays/tracer.py`` that adds a surface to ``_Packet.power_scope``.
* Section 2 has one row for each test function of the three containment modules.
* A row with a withheld case states the physical reason, and the test's docstring carries the same row id
  (``Register row T-xx, withheld: ...``); a test whose row has no withheld case carries none.

A change that fixes a containment edits its row in the same commit, so a site or a test that appears, disappears or is renamed
without its row fails here."""

from __future__ import annotations

import ast
import re
from pathlib import Path


ENGINE = Path(__file__).resolve().parents[1]
REGISTER = ENGINE / "docs" / "engine" / "audit" / "CONTAINMENT_REGISTER.md"

TRACER = ENGINE / "src" / "os_optics" / "optics" / "rays" / "tracer.py"
MODULES = ("test_closeout_containments.py", "test_out_of_scope_power_results.py", "test_out_of_scope_orders_and_conversions.py")
STATES = {"computed", "qualified", "withheld", "control"}


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _rows(prefix: str) -> dict[str, list[str]]:
    """The register's table rows whose first cell is ``<prefix>-<number>``, by that id."""
    rows: dict[str, list[str]] = {}
    for line in REGISTER.read_text(encoding="utf-8").splitlines():
        if re.match(rf"\| {prefix}-\d\d \|", line):
            cells = _cells(line)
            assert cells[0] not in rows, f"{cells[0]} is listed twice"
            rows[cells[0]] = cells
    return rows


def _scope_sites() -> set[str]:
    """The methods of the tracer that assign ``<packet>.power_scope`` (a restore of a held state by tuple unpacking is not one)."""
    tree = ast.parse(TRACER.read_text(encoding="utf-8"))
    found: set[str] = set()
    for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        for method in (node for node in cls.body if isinstance(node, ast.FunctionDef)):
            for node in ast.walk(method):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "power_scope" for t in node.targets):
                    found.add(f"{cls.name}.{method.name}")
    return found


def _tests(module: str) -> dict[str, str]:
    """The test functions of ``module`` with their docstrings."""
    tree = ast.parse((ENGINE / "tests" / module).read_text(encoding="utf-8"))
    return {node.name: ast.get_docstring(node) or "" for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def test_every_site_that_marks_a_beams_power_out_of_scope_has_a_row():
    rows = _rows("S")
    listed = {cells[1].strip("`") for cells in rows.values()}
    assert len(listed) == len(rows), "a method is listed in two rows"
    assert listed == _scope_sites()


def test_every_test_of_the_containment_modules_has_a_row():
    listed = [cells[1].strip("`") for cells in _rows("T").values()]
    assert len(set(listed)) == len(listed), "a test is listed in two rows"
    present = {name for module in MODULES for name in _tests(module)}
    assert set(listed) == present


def test_every_row_states_its_owner_and_a_defined_state():
    for prefix in ("S", "T"):
        for row, cells in _rows(prefix).items():
            owner, state = cells[-2], cells[-1]
            assert owner, f"{row} names no owner"
            named = set(re.findall(r"`([a-z]+)`", state))
            assert named and named <= STATES, f"{row}: {state!r}"


def test_a_withheld_row_states_its_reason_in_the_register_and_in_the_tests_docstring():
    docstrings = {name: doc for module in MODULES for name, doc in _tests(module).items()}
    rows = _rows("T")
    for row, cells in rows.items():
        test, state = cells[1].strip("`"), cells[-1]
        marker = f"Register row {row}, withheld: "
        if "`withheld`" in state:
            assert re.search(r"`withheld`(: \S|, as T-\d\d)", state), f"{row} is withheld without a stated reason"
            assert marker in " ".join(docstrings[test].split()), f"{test} does not carry {marker!r}"
        else:
            assert "Register row" not in docstrings[test], f"{test} names a register row that has no withheld case"
    for test, doc in docstrings.items():
        for named in re.findall(r"Register row (T-\d\d)", doc):
            assert rows[named][1].strip("`") == test, f"{test} names {named}, the row of another test"
