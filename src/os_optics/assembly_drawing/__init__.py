"""Local Detailed Assembly Drawing of one resolved assembly, generated on this device.

Input: a verified Assembly Preview Package (identities, BOM line references, poses, support tree, mates,
settings, fixations, breadboard, beam reference, DOF delivery, assembly notes and part geometry) and its verified
Assembly Detail Package (the dimension references and their basis). Output: four files::

    assembly_drawing.svg     one landscape sheet (A3; A2..A0 when several support chains need it): FRONT, SIDE, PLAN
                             and PICTORIAL views, B-line balloons, dimensions marked with their basis, parts legend (the
                             Customer Purchase BOM lines), steps grouped by support chain, notes
    assembly_drawing.pdf     the same sheet as a one-page PDF
    drawing_result.json      AssemblyDrawingResultV1: what the sheet shows, as data
    drawing_manifest.json    AssemblyDrawingManifestV1: bindings, generator build, every file's size and SHA-256

Usage::

    from os_optics.assembly_drawing import generate_assembly_drawing, load_assembly_detail_package, verify_assembly_drawing

    detail = load_assembly_detail_package("assembly-detail.json", "assembly-preview.zip")   # VerifiedAssemblyDetailV1
    drawing = generate_assembly_drawing("assembly-preview.zip", "assembly-detail.json", diagnostic=False, locale="zh-CN")
    drawing.files            # {"assembly_drawing.svg": b"...", "assembly_drawing.pdf": ..., "drawing_result.json": ..., ...}
    drawing.result           # AssemblyDrawingResultV1 (drawing_status, delivery_ready, views, dimensions, balloons, ...)
    drawing.write("drawing") # exactly the four files (an earlier drawing's four are replaced; nothing else is touched)
    verify_assembly_drawing("drawing")   # () when the output set verifies (strict: exactly the four files)

Sources are what ``os_optics.assembly`` accepts for the preview (archive path, package directory, archive bytes,
``{path: bytes}`` map, verified package) and a path, bytes or ``VerifiedAssemblyDetailV1`` for the detail -- the
bytes of an opened delivery.

It always runs the local assembly validation (``os_optics.assembly``) on the same package first: PASS / WARNING give
a ``DETAILED_ASSEMBLY_DRAWING`` (``delivery_ready``). A check that did not pass is refused unless ``diagnostic=True``
(FAIL: ``DRAWING_REFUSED_VALIDATION_FAILED``; UNKNOWN / OUT_OF_MODEL_SCOPE: ``DRAWING_REFUSED_VALIDATION_INCONCLUSIVE``);
with it, FAIL gives a ``DIAGNOSTIC_VALIDATION_FAILED`` sheet and UNKNOWN / OUT_OF_MODEL_SCOPE a
``DIAGNOSTIC_VALIDATION_INCOMPLETE`` sheet (not delivery ready, bannered and watermarked, listing the issues behind the
status). What the drawing itself needs is refused in every mode (``PART_GEOMETRY_UNREADABLE``,
``BOARD_GEOMETRY_MISMATCH`` ...).
Every refusal is an ``AssemblyDrawingError`` with a stable ``DrawingErrorCode``; malformed input (deeply nested,
non-finite, bad UTF-8) is refused with a code, never an escaping exception. A drawing is made for a package bound to a
commercial resolution and its Customer Purchase BOM (``PREVIEW_NOT_COMMERCIALLY_BOUND``).
Every output token comes from the verified inputs, the generator's fixed templates or is a number (a property of the
generator, checked at build time over every drawing its tests generate).

Every sheet text comes from the message catalog (``i18n``) in the requested ``locale`` (``en`` default, ``zh-CN``); the PDF
writes Chinese with the predefined CID font ``STSong-Light`` (no font file embedded). The drawing manifest names its files
by the artifact store's roles (``DRAWING_SVG``, ``DRAWING_PDF``, ``DRAWING_RESULT``).

Shell: ``os-optics assembly draw PREVIEW --detail DETAIL --out DIR [--locale zh-CN]`` and ``os-optics assembly
verify-drawing DIR``.
"""

from __future__ import annotations

from .._vendor.shared_contracts.assembly_detail import (
    AssemblyDrawingManifestV1,
    AssemblyDrawingResultV1,
    DrawingStatusV1,
    VerifiedAssemblyDetailV1,
)
from .errors import AssemblyDrawingError, DrawingErrorCode
from .generate import (
    DRAWING_FILES,
    GENERATOR_VERSION,
    AssemblyDrawing,
    drawing_output_files,
    generate_assembly_drawing,
    load_assembly_detail_package,
    load_preview,
    verify_assembly_drawing,
)

__all__ = [
    "DRAWING_FILES",
    "GENERATOR_VERSION",
    "AssemblyDrawing",
    "AssemblyDrawingError",
    "AssemblyDrawingManifestV1",
    "AssemblyDrawingResultV1",
    "DrawingErrorCode",
    "DrawingStatusV1",
    "VerifiedAssemblyDetailV1",
    "drawing_output_files",
    "generate_assembly_drawing",
    "load_assembly_detail_package",
    "load_preview",
    "verify_assembly_drawing",
]
