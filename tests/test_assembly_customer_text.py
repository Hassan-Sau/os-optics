"""The printable text of the assembly, drawing and protected-delivery surfaces states what they are and what they did.

These modules ship in the public wheel: their docstrings (``help()``), the CLI help of ``os-optics assembly`` and
the generated drawing are read by customers and their Agents. Each must describe the product positively; none of the
wording below, which describes instead what a result is not, belongs there.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from drawing_helpers import detail_bytes, preview

from os_optics import cli
from os_optics.assembly_drawing import generate_assembly_drawing
from os_optics.assembly_drawing.generate import DRAWING_FILES

RETIRED = (
    "not a manufacturing drawing", "proxy", "proxies", "engineering review", "engineering preview", "not verified",
    "never verifies", "practical deterrence", "not drm", "supplier", "reverse engineering", "reverse-engineering",
    "gross check", "gross-collision", "gross collision", "approximate", "preview assumption", "strict verification",
    "not claimed", "internal test", "first-version", "first version", "not supplied", "not modelled", "bom authority",
    "private resolution", "resolver", "delivery layer", "vendored", "vendor it", "bundle it", "persisted commercial",
    "projector", "identity_state", "demo", "pilot", "self-declared", "not trusted", "issue number", "first issue",
    "when the submission contract exists", "customer commercial bom", "csprng", "production callers",
)

MODULES = (
    "os_optics.assembly",
    "os_optics.assembly.checks",
    "os_optics.assembly.interfaces",
    "os_optics.assembly.overlap",
    "os_optics.assembly.result",
    "os_optics.assembly.tolerances",
    "os_optics.assembly_drawing",
    "os_optics.assembly_drawing.generate",
    "os_optics.assembly_drawing.sheet",
    "os_optics.assembly_drawing.steps",
    "os_optics.protected_delivery",
    "os_optics._vendor.shared_contracts.assembly_preview",
    "os_optics._vendor.shared_contracts.assembly_preview.scene",
    "os_optics._vendor.shared_contracts.assembly_preview.manifest",
    "os_optics._vendor.shared_contracts.assembly_preview.validation",
    "os_optics._vendor.shared_contracts.assembly_detail",
    "os_optics._vendor.shared_contracts.assembly_detail.detail",
    "os_optics._vendor.shared_contracts.assembly_detail.drawing",
    "os_optics._vendor.shared_contracts.protected_assembly_delivery",
    "os_optics._vendor.shared_contracts.protected_assembly_delivery.envelope",
    "os_optics._vendor.shared_contracts.protected_assembly_delivery.detail_envelope",
)


def _hits(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in RETIRED if phrase in lowered]


@pytest.mark.parametrize("name", MODULES)
def test_module_and_class_docstrings_state_what_the_surface_does(name: str) -> None:
    module = importlib.import_module(name)
    texts = [module.__doc__ or ""]
    for _, member in inspect.getmembers(module):
        if (inspect.isclass(member) or inspect.isfunction(member)) and getattr(member, "__module__", "") == name:
            texts.append(member.__doc__ or "")
    assert [(name, _hits(t)) for t in texts if _hits(t)] == []


def test_assembly_cli_help_describes_the_check_and_the_drawing() -> None:
    assert _hits(cli.ASSEMBLY_DESCRIPTION) == [] and _hits(cli.DRAWING_DESCRIPTION) == []
    assert "layout and collision checks" in cli.ASSEMBLY_DESCRIPTION
    assert cli.DRAWING_DESCRIPTION.startswith("Detailed Assembly Drawing of a resolved assembly")


def test_the_generated_drawing_text_states_what_it_is() -> None:
    drawing = generate_assembly_drawing(preview(), detail_bytes())
    assert set(drawing.files) == set(DRAWING_FILES)
    for name, data in drawing.files.items():
        text = data.decode("latin-1" if name.endswith(".pdf") else "utf-8")
        assert _hits(text) == [], name
    assert b"DETAILED ASSEMBLY DRAWING" in drawing.files["assembly_drawing.svg"]
