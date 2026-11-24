"""Adding a built native component to a Project document.

``add_to_project(project, build)`` returns the Project with the build's models, materials and coatings added to its
``optical_models`` / ``model_materials`` / ``model_coatings`` (an entry already present with the same id and the same
content -- for a material: the same physics, ``catalog_copy.physics_difference`` -- is kept once; the same id with a
different content is refused) and the build's component instance appended to
``components`` (its ``instance_id`` and pose optionally replaced). The result is checked as a ``ProjectDocumentV2``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from .._vendor.shared_contracts.optical_design import ProjectDocumentV2
from ..catalog_copy import material_conflict
from ..project_document import parse_project_document, plain_location
from .common import NativeBuild
from .errors import NativeComponentError, NativeErrorCode, NativeProblem

LIBRARY_FIELDS = (("optical_models", "model_id"), ("model_materials", "material_id"), ("model_coatings", "coating_id"))


def _dump(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True, exclude_none=True)


def add_to_project(project: Mapping[str, Any] | ProjectDocumentV2, build: NativeBuild, *, instance_id: str | None = None,
                   position_mm: Sequence[float] | None = None, quaternion_wxyz: Sequence[float] | None = None) -> dict[str, Any]:
    document = project.payload() if isinstance(project, ProjectDocumentV2) else copy.deepcopy(dict(project))
    additions = {"optical_models": [_dump(m) for m in build.models], "model_materials": [_dump(m) for m in build.model_materials],
                 "model_coatings": [_dump(c) for c in build.model_coatings]}
    for field, key in LIBRARY_FIELDS:
        existing = list(document.get(field) or [])
        by_id = {item.get(key): item for item in existing if isinstance(item, Mapping)}
        for item in additions[field]:
            present = by_id.get(item[key])
            if present is None:
                existing.append(item)
                by_id[item[key]] = item
            elif field == "model_materials":
                conflict = material_conflict(present, item)  # the same glass copied by another route or release is shared
                if conflict is not None:
                    raise NativeComponentError([NativeProblem(NativeErrorCode.PARAMETER_INVALID, conflict, "material")])
            elif present != item:
                parameter = "model_id" if field == "optical_models" else "material"
                raise NativeComponentError([NativeProblem(
                    NativeErrorCode.PARAMETER_INVALID,
                    f"the Project already has a different {key} {item[key]!r} in {field}; give this one another id", parameter)])
        if existing:
            document[field] = existing
    component = _dump(build.component_template)
    if instance_id is not None:
        component["instance_id"] = instance_id
    if position_mm is not None:
        component["nominal_pose"]["position_mm"] = [float(v) for v in position_mm]
    if quaternion_wxyz is not None:
        component["nominal_pose"]["quaternion_wxyz"] = [float(v) for v in quaternion_wxyz]
    components = list(document.get("components") or [])
    if any(isinstance(c, Mapping) and c.get("instance_id") == component["instance_id"] for c in components):
        raise NativeComponentError([NativeProblem(NativeErrorCode.PARAMETER_INVALID,
                                                  f"the Project already has a component {component['instance_id']!r}", "instance_id")])
    components.append(component)
    document["components"] = components
    try:
        return parse_project_document(document).payload()
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        where = ".".join(str(p) for p in plain_location(tuple(first["loc"]))) or "the document"
        raise NativeComponentError([NativeProblem(NativeErrorCode.PARAMETER_INVALID,
                                                  f"the Project with this component is not valid ({where}: {first['msg']})", "project")]) from None


__all__ = ["add_to_project"]
