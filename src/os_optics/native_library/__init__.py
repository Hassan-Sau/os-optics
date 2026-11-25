"""The native component library: parametric optical components, independent of any catalog product, as Project optical models.

A native class turns a few engineering parameters into a complete, physically valid ``ProjectOpticalModelV1`` -- with the
``ProjectModelMaterialV1`` / ``PublicCoatingV1`` it uses -- that a Project declares and a ``custom_part`` component uses
through ``optical_model_id``. The model is simulated by the same Engine path as a catalog product's model; nothing is
priced or sourced (a component using it is customer-supplied).

Classes (``native_components()`` describes each one: parameters with units, constraints and defaults, physics,
approximations, references):

* stops (``APERTURE_STOP``): ``aperture.circular`` (iris, stop, pinhole), ``aperture.slit``, ``aperture.annular``,
  ``aperture.central_obstruction``;
* lenses: ``lens.ideal_thin`` (``IDEAL_THIN_LENS``), ``lens.singlet`` and ``lens.achromat`` (``SURFACE_SEQUENCE``,
  designed for a focal length), ``lens.axicon`` (``SURFACE_SEQUENCE`` with a cone), ``lens.tabulated_thin``
  (``TABULATED_THIN_LENS``: a response table from another tool);
* plates and prisms: ``plate.window``, ``plate.layered`` (cuvette, plate stack), ``prism.wedge`` (``SURFACE_SET``);
* reflectors and splitters: ``mirror.flat``, ``mirror.curved``, ``splitter.plate``, ``mirror.hollow_roof`` and
  ``retroreflector.hollow_corner`` (``COMPOUND`` units of flat mirrors);
* filters: ``filter.variable_nd`` (``NEUTRAL_DENSITY``: a wheel or gradient with its position);
* polarization: ``polarizer.linear``, ``waveplate.retarder`` (``POLARIZATION_ELEMENT``), ``rotator.faraday``
  (``FARADAY_ROTATOR``: a stated nonreciprocal rotation);
* uniaxial crystals: ``splitter.beam_displacer`` (``SURFACE_SEQUENCE``: a crystal plate splitting a beam into two parallel
  beams by walk-off), ``polarizer.glan`` (``COMPOUND`` of two air-spaced crystal prisms: Glan-Taylor or Glan-Foucault);
* gratings: ``grating.plane`` (``DIFFRACTION_GRATING``: order directions from its period and grooves, order powers from a stated
  response);
* modulators: ``deflector.acousto_optic`` (``DIFFRACTION_GRATING``: RF tones as the orders of one grating, each tone's optical frequency shift in its design report);
* compound units (``COMPOUND``): ``compound.keplerian_telescope``, ``compound.galilean_telescope``, ``compound.relay_4f``.

Field models (``native_field_models()``) are devices that act on a field or an irradiance map, with no optical model of
their own: ``slm.pupil_mask`` (``pupil_field``) and ``detector.camera_response`` (``detector_response``).

Use::

    from os_optics.native_library import build_native_model, add_to_project

    build = build_native_model("lens.singlet", {"focal_length_mm": "100", "design_wavelength_nm": "633",
                                                "material": "N-BK7", "diameter_mm": "25.4", "form": "BEST_FORM"})
    build.optical_model, build.model_materials, build.component_template, build.design
    project = add_to_project(project_document, build, instance_id="L1", position_mm=(0, 0, 100))

Every class description states its ``availability`` -- ``LOCAL_SIMULATION`` (it simulates on this device) and
``CUSTOMER_SUPPLIED`` (a component using it is your own optic, a ``custom_part``; a native class is never an orderable
product) -- and ``catalog_equivalents``: the public catalog category, forms and summary values to search for a comparable
catalog product (``os-optics catalog list --category C --form F --wavelength-nm W``); empty when the catalog has none.

Every value is exact decimal arithmetic (identical in CPython and Pyodide). A request that cannot be built raises
``NativeComponentError`` listing every problem (``code``, ``parameter``, ``message``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._vendor.shared_contracts.design_catalog import PublicOpticalCatalogV2
from .common import BuildContext, NativeBuild
from .errors import NativeComponentError, NativeErrorCode, NativeProblem
from .field_models import native_field_models
from .materials import MaterialResolver
from .project import add_to_project
from .registry import NATIVE_CLASSES, NativeClass, native_class


def native_class_ids() -> list[str]:
    """The ids of every native component class, sorted."""

    return sorted(NATIVE_CLASSES)


def native_components() -> list[dict[str, Any]]:
    """The description of every native component class (``NativeClass.describe``), sorted by id."""

    return [NATIVE_CLASSES[class_id].describe() for class_id in native_class_ids()]


def describe_native_class(class_id: str) -> dict[str, Any]:
    """One class's description; ``class_id``: its id, or the family before the dot where that family has one class (``grating``)."""

    family = [known for known in native_class_ids() if known.startswith(f"{class_id}.")]
    return native_class(family[0] if len(family) == 1 else class_id).describe()


def build_native_model(class_id: str, parameters: Mapping[str, Any] | None = None, *,
                       catalog: PublicOpticalCatalogV2 | None = None) -> NativeBuild:
    """Build one native component. ``catalog``: the public catalog to copy named materials / coatings from (default: the
    active release, loaded only when one is named)."""

    return native_class(class_id).build_with(parameters, BuildContext(MaterialResolver(catalog)))


__all__ = [
    "NATIVE_CLASSES",
    "NativeBuild",
    "NativeClass",
    "NativeComponentError",
    "NativeErrorCode",
    "NativeProblem",
    "add_to_project",
    "build_native_model",
    "describe_native_class",
    "native_class_ids",
    "native_components",
    "native_field_models",
]
