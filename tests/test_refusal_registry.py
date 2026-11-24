"""The refusal census (``os_optics.errors.registry``) names every code vocabulary and exception class of the Engine.

A vocabulary or an exception class the tables do not name, a code string two vocabularies share without a declared alias,
a pending code its vocabulary does not hold, and an ``EngineInputError`` raised with a code that is neither an
input-contract code nor pending each fail here: a new refusal is classified when it is written.
"""

from __future__ import annotations

import enum
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import os_optics
from os_optics.errors import registry
from os_optics.errors.registry import ALIASES, EXCEPTIONS, INPUT_CONTRACT_CODES, PENDING, VOCABULARIES, RefusalClass

SOURCE = Path(os_optics.__file__).parent


def _resolve(name: str):
    module, attribute = name.split(":")
    return getattr(importlib.import_module(module), attribute)


def _defined() -> tuple[dict[str, type], dict[str, type]]:
    """Every code vocabulary (a ``StrEnum`` whose name ends in ``Code``) and every exception class defined in the Engine's own
    modules (the vendored contracts are another package's)."""

    vocabularies, exceptions = {}, {}
    for info in pkgutil.walk_packages([str(SOURCE)], "os_optics."):
        if "._vendor" in info.name or info.name.endswith("__main__"):
            continue
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if not inspect.isclass(obj) or obj.__module__ != module.__name__:
                continue
            if issubclass(obj, enum.StrEnum) and name.endswith("Code"):
                vocabularies[f"{module.__name__}:{name}"] = obj
            if issubclass(obj, BaseException):
                exceptions[f"{module.__name__}:{name}"] = obj
    return vocabularies, exceptions


def test_the_tables_name_every_code_vocabulary_and_every_exception_class():
    vocabularies, exceptions = _defined()
    assert sorted(v.name for v in VOCABULARIES) == sorted(vocabularies), "a code vocabulary is not in errors/registry.py VOCABULARIES"
    assert sorted(e.name for e in EXCEPTIONS) == sorted(exceptions), "an exception class is not in errors/registry.py EXCEPTIONS"
    assert len({v.name for v in VOCABULARIES}) == len(VOCABULARIES) and len({e.name for e in EXCEPTIONS}) == len(EXCEPTIONS)
    for entry in EXCEPTIONS:
        assert issubclass(_resolve(entry.name), BaseException)
        assert entry.vocabulary is None or entry.vocabulary in vocabularies, entry
    assert all(v.classes and v.family for v in VOCABULARIES) and all(e.family for e in EXCEPTIONS)


def test_a_code_string_in_two_vocabularies_is_a_declared_alias():
    vocabularies, _ = _defined()
    holders: dict[str, list[str]] = {}
    for name, vocabulary in vocabularies.items():
        for member in vocabulary:
            holders.setdefault(member.value, []).append(name)
    shared = {code: tuple(sorted(names)) for code, names in holders.items() if len(names) > 1}
    assert shared == {code: tuple(sorted(names)) for code, names in ALIASES.items()}


def test_pending_per_item_refusals_name_codes_their_vocabulary_holds():
    by_name = {e.name: e for e in EXCEPTIONS}
    seen: set[tuple[str, str]] = set()
    for pending in PENDING:
        raised = by_name[pending.exception]
        assert pending.per_item_form
        if not pending.codes:
            assert raised.vocabulary is None, f"{pending.exception} carries a vocabulary: name the pending codes"
            continue
        members = {member.value for member in _resolve(raised.vocabulary)}
        assert set(pending.codes) <= members, (pending.exception, set(pending.codes) - members)
        assert not seen & {(pending.exception, code) for code in pending.codes}, "a pending code is listed twice"
        seen |= {(pending.exception, code) for code in pending.codes}
    # (a refusal whose exception is itself of class PER_ITEM is whole-operation today by definition: all of it is pending)
    assert {e.name for e in EXCEPTIONS if e.refusal is RefusalClass.PER_ITEM} <= {p.exception for p in PENDING}


RAISE = re.compile(r"\b(EngineInputError|ModelScopeRefusal)\(\s*ErrorCode\.([A-Z_0-9]+)")


def test_every_engine_input_error_in_the_source_is_classified():
    """Each ``EngineInputError(ErrorCode.X`` / ``ModelScopeRefusal(ErrorCode.X`` written in the source: ``X`` is an
    input-contract code (the request is malformed) or a pending per-item refusal of that exception."""

    pending = {(p.exception.split(":")[1], code) for p in PENDING for code in p.codes}
    found = set()
    for path in SOURCE.rglob("*.py"):
        if "_vendor" in path.parts:
            continue
        found |= set(RAISE.findall(path.read_text(encoding="utf-8")))
    assert found, "the pattern no longer finds the raise sites"
    unclassified = sorted((kind, code) for kind, code in found
                          if not (kind == "EngineInputError" and code in INPUT_CONTRACT_CODES) and (kind, code) not in pending)
    assert unclassified == []
    assert set(INPUT_CONTRACT_CODES) <= {member.value for member in _resolve(registry.ERROR_CODE)}


def test_the_paraxial_and_cavity_modules_refuse_only_with_codes_of_their_vocabularies():
    """Every constant a refusal of ``local/paraxial.py`` / ``local/cavity.py`` is raised with is a value of
    ``ParaxialErrorCode`` or ``CavityErrorCode``, and a problem holds it as a plain string (as before the enums)."""

    from os_optics.local import cavity, paraxial

    codes = {member.value for member in paraxial.ParaxialErrorCode} | {member.value for member in cavity.CavityErrorCode}
    call = re.compile(r"\b_(?:cavity_)?refuse\(\s*([A-Z_][A-Z_0-9]*)\s*,")
    for module in (paraxial, cavity):
        names = set(call.findall(Path(module.__file__).read_text(encoding="utf-8")))
        assert names, module.__name__
        for name in names:
            value = getattr(module, name)
            assert type(value) is str and value in codes, (module.__name__, name, value)
    problem = paraxial._refuse(paraxial.REQUEST_INVALID, "m").problems[0]
    assert problem == {"code": "PARAXIAL_REQUEST_INVALID", "message": "m", "location": None} and type(problem["code"]) is str
    assert paraxial.ParaxialError([]).as_dict()["code"] == "PARAXIAL_ANALYSIS_INVALID"
    assert cavity.CavityError([]).as_dict()["code"] == "CAVITY_ANALYSIS_INVALID"
