"""Optical Layout Drawing of an optics-only purchase, rendered on this device.

An optics-only purchase buys the optical elements the validated design selected and no mounting hardware. Its
deliverable is the Optical Layout Package (``GET /api/v1/orders/{order_id}/optical-layout`` once ordered, or
``GET /api/v1/resolutions/{resolution_id}/optical-layout``): the relative optical layout of the design. This package
renders it into three files::

    optical_layout.svg              one landscape sheet: PLAN and ELEVATION views, element table, spacing table, notes,
                                    title block (Order Number and customer once ordered)
    optical_layout.pdf              the same sheet as a one-page PDF
    optical_layout_manifest.json    OpticalLayoutDrawingManifestV1: the package it renders, language, build, file digests

Usage::

    from os_optics.optical_layout import generate_optical_layout

    drawing = generate_optical_layout("optical-layout.json", locale="zh-CN")   # or the JSON dict / bytes, or the model
    drawing.write("layout")                                                     # replaces an earlier layout's three files

Shell: ``os-optics layout draw optical-layout.json --out layout [--locale zh-CN]`` (offline), or
``os-optics order layout ORDER_ID --out layout`` / ``os-optics resolution layout RESOLUTION_ID --out layout``
(fetched with your Account token). Browser: ``os_optics.local.json_api.generate_optical_layout(package, locale)``.
"""

from __future__ import annotations

from .._vendor.shared_contracts.optical_layout import OpticalLayoutPackageV1
from .._vendor.shared_contracts.optical_layout.drawing import LAYOUT_DRAWING_FILES, OpticalLayoutDrawingManifestV1
from .generate import (
    GENERATOR_VERSION,
    LOCALES,
    OpticalLayoutDrawing,
    OpticalLayoutError,
    OpticalLayoutErrorCode,
    generate_optical_layout,
    load_layout_package,
    verify_optical_layout_drawing,
)

__all__ = [
    "GENERATOR_VERSION",
    "LAYOUT_DRAWING_FILES",
    "LOCALES",
    "OpticalLayoutDrawing",
    "OpticalLayoutDrawingManifestV1",
    "OpticalLayoutError",
    "OpticalLayoutErrorCode",
    "OpticalLayoutPackageV1",
    "generate_optical_layout",
    "load_layout_package",
    "verify_optical_layout_drawing",
]
