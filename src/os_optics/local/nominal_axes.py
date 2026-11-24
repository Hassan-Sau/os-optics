"""The nominal beam axis of each source of a lowered Project: what ``json_api.project_layout`` draws
(``os_optics.optical_view.layout``) and a headless editor can use to place parts before a run.

The nominal beam axis of a source is a straight-line construction on the lowered surfaces (``scene.lower``), the same
surfaces a run traces, followed only where its direction follows from geometry alone:

* it starts at the source's origin along its emission axis (local +Z) and goes to the nearest surface it meets
  (``optics.surfaces.geometry.intersect``: the exact surface, inside its support);
* at a mirror (a reflecting surface, met on its reflecting side) it turns by the law of reflection about the surface's
  normal at that point; at an ideal thin lens it turns as that lens turns a ray (exact for any ray, ``thin_lens_dir``),
  a lens with a wavelength table by its focal length at the axis's wavelength (met at its centre it goes on straight);
* through any other surface that transmits (a refracting surface, a beam splitter's transmitted side, a polarizer, a
  retarder, a neutral density, a Faraday rotator, an aperture stop's opening, a non-absorbing observation plane) it goes on
  straight where both sides are one medium, and between two media it refracts by Snell's law at the axis's wavelength
  (``optics.surfaces.interaction.refract``, the function the trace uses, with the indices of the Project's materials; on
  total internal reflection it follows the reflected direction, as in a prism);
* its wavelength is the source's first line (``wavelength_nm`` of the axis record): the line of the chief ray a run
  traces first;
* it ends where it is not followed (``NOT_FOLLOWED``): at a diffraction grating, a complex transmission, a nonlinear crystal,
  a uniaxial crystal or a gradient-index medium, at a surface whose medium has no index at that wavelength (outside the
  material's data) or is not the medium the axis is in, and at a tabulated thin lens met off its centre outside its table;
* it ends on an absorbing observation plane or an absorber, on a mount ring or the back of a mirror (``ABSORBED``), when
  it meets one surface for the third time (``RETURNS``: light going round between mirrors), or when it leaves the design
  (``FREE``: drawn on for ``free_ray_display_length_mm``).

A beam splitter's reflected branch is not followed. Where a run's traced beams exist they describe the light; the
nominal axis only helps place things along the design's optical axis before a run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts.request import ValidationRequest
from ..errors import EngineInputError, ModelScopeRefusal
from ..math.vectors import dot, normalize
from ..optics import apertures
from ..optics.materials.library import MaterialLibrary
from ..optics.surfaces.geometry import intersect
from ..optics.surfaces.interaction import reflect_dir, refract, thin_lens_dir
from ..optics.surfaces.tabulated_lens import response_at
from ..scene.lower import lower_scene

#: A surface may be met this many times by one axis (a double pass); one more ends the axis (RETURNS).
MAX_MEETINGS_PER_SURFACE = 2
#: How close to a thin lens's centre (mm) a ray passes it undeviated whatever the lens's focal length.
THIN_LENS_CENTRE_TOL_MM = 1e-9

END_FREE = "FREE"
END_ABSORBED = "ABSORBED"
END_NOT_FOLLOWED = "NOT_FOLLOWED"
END_RETURNS = "RETURNS"

#: Surfaces the axis crosses: straight where both sides are one medium, refracted between two media.
_TRANSMITTING = frozenset({"refract", "beam_splitter", "polarizer", "retarder", "neutral_density", "faraday_rotator", "aperture_stop"})


def _round(value: float, decimals: int) -> float:
    return round(float(value), decimals) + 0.0  # (as the scene rounds: -0.0 folded)


def _rv(v: Any, decimals: int) -> tuple[float, float, float]:
    return (_round(v[0], decimals), _round(v[1], decimals), _round(v[2], decimals))


def _axis(start: np.ndarray, direction: np.ndarray, source_id: str, surfaces: list[Any], cfg: Any, free_length_mm: float,
          d: int, special_media: set[str], library: MaterialLibrary, wavelength_nm: float, medium: str) -> dict[str, Any]:
    """The nominal beam axis of one source (module docstring): at ``wavelength_nm``, starting in ``medium``."""

    def index(name: str) -> float | None:
        value, issue = library.evaluator(name).index(wavelength_nm)
        return None if issue is not None or value is None else float(value)

    o = np.asarray(start, dtype=np.float64).reshape(1, 3)
    v, _ = normalize(np.asarray(direction, dtype=np.float64).reshape(1, 3), cfg.zero_vector_tol)
    points = [_rv(tuple(float(x) for x in o[0]), d)]
    surface_ids: list[str] = []
    met: dict[str, int] = {}
    end, end_surface = END_FREE, None
    for _ in range(MAX_MEETINGS_PER_SURFACE * len(surfaces) + 1):
        best = None
        for surface in surfaces:
            hit = intersect(surface, o, v, cfg)
            t = float(hit.t[0])
            if np.isfinite(t) and (best is None or t < best[0]):
                best = (t, surface, hit)
        if best is None:
            break
        _, surface, hit = best
        point = hit.point[0]
        points.append(_rv(tuple(float(x) for x in point), d))
        surface_ids.append(surface.surface_id)
        met[surface.surface_id] = met.get(surface.surface_id, 0) + 1
        if met[surface.surface_id] > MAX_MEETINGS_PER_SURFACE:
            end, end_surface = END_RETURNS, surface.surface_id
            break
        inter = surface.interaction
        next_v: np.ndarray | None = v
        if inter.kind == "observe":
            active = inter.active_aperture
            on_active = active is None or bool(apertures.inside(active, hit.u, hit.v, cfg.aperture_edge_tol_mm)[0])
            if inter.absorbing and on_active:
                next_v, end = None, END_ABSORBED
        elif not bool(hit.inside[0]):
            next_v, end = None, END_ABSORBED  # (the mount ring around the clear aperture, or an aperture stop's plate)
        elif surface.medium_pos in special_media or surface.medium_neg in special_media:
            next_v, end = None, END_NOT_FOLLOWED
        elif inter.kind == "mirror":
            from_positive = float(dot(v, hit.normal)[0]) < 0.0
            if inter.reflective_side == "both" or (inter.reflective_side == "positive") == from_positive:
                next_v = reflect_dir(v, hit.normal, cfg.zero_vector_tol)[0]
            else:
                next_v, end = None, END_ABSORBED
        elif inter.kind == "absorber":
            next_v, end = None, END_ABSORBED
        elif inter.kind == "thin_lens" and inter.table is None and inter.focal_length != 0.0:
            out, ok = thin_lens_dir(hit.point, surface.origin, surface.normal, v, inter.focal_length, cfg.zero_vector_tol)
            next_v = out if bool(ok[0]) else None
            if next_v is None:
                end = END_NOT_FOLLOWED
        elif inter.kind == "thin_lens":
            if abs(float(hit.u[0])) > THIN_LENS_CENTRE_TOL_MM or abs(float(hit.v[0])) > THIN_LENS_CENTRE_TOL_MM:
                # (a tabulated thin lens met off its centre: its focal length at the axis's wavelength, inside its table)
                next_v = None
                if inter.table is not None:
                    focal, _transmittance, in_table = response_at(inter.table, wavelength_nm)
                    if in_table and focal != 0.0:
                        out, ok = thin_lens_dir(hit.point, surface.origin, surface.normal, v, focal, cfg.zero_vector_tol)
                        next_v = out if bool(ok[0]) else None
                if next_v is None:
                    end = END_NOT_FOLLOWED
        elif inter.kind in _TRANSMITTING:
            if surface.medium_pos != surface.medium_neg:
                # between two media: Snell's law at the axis's wavelength, as the trace refracts its chief ray
                from_positive = float(dot(v, hit.normal)[0]) < 0.0
                before, after = (surface.medium_pos, surface.medium_neg) if from_positive else (surface.medium_neg, surface.medium_pos)
                n1, n2 = (index(before), index(after)) if medium == before else (None, None)
                if n1 is None or n2 is None:
                    next_v, end = None, END_NOT_FOLLOWED
                else:
                    bent = refract(v, hit.normal, n1, n2, cfg.zero_vector_tol)
                    next_v = bent.d_out
                    if not bool(bent.tir[0]):
                        medium = after
        else:
            next_v, end = None, END_NOT_FOLLOWED
        if next_v is None:
            end_surface = surface.surface_id
            break
        o, v = point.reshape(1, 3), np.asarray(next_v, dtype=np.float64).reshape(1, 3)
    extension = None
    if end == END_FREE:
        last = tuple(float(x) for x in o[0])
        extension = _rv(tuple(a + free_length_mm * float(b) for a, b in zip(last, v[0])), d)
    return {"source_id": source_id, "wavelength_nm": wavelength_nm, "points_mm": [list(p) for p in points], "surface_ids": surface_ids, "end": end,
            "end_surface_id": end_surface, "display_extension_to_mm": None if extension is None else list(extension)}


def nominal_axes(request: ValidationRequest, free_length_mm: float, d: int) -> tuple[list[dict[str, Any]], str | None]:
    """The nominal beam axis of every source of the request, in component order (module docstring), its points rounded to
    ``d`` decimals and a free end drawn on for ``free_length_mm``; and, when the lowered surfaces cannot be built, no axis
    and the reason's code."""

    cfg = request.configuration.numerical
    library = MaterialLibrary(request.project.materials, cfg)
    try:
        scene, _ = lower_scene(request.project, cfg, library)
    except (EngineInputError, ModelScopeRefusal) as exc:
        code = getattr(exc, "code", None)
        return [], str(getattr(code, "value", code) or type(exc).__name__)
    special = set(scene.optic_axes) | {gradient.medium for gradient in scene.gradients.values()} | set(scene.nonlinear)
    #: Per source: where its axis starts, along what, at which wavelength (its first line: the chief ray a run traces first) and in
    #: which medium.
    starts: dict[str, tuple[np.ndarray, np.ndarray, float, str]] = {}
    for emission in (*scene.sources, *scene.emitters):
        starts.setdefault(emission.owner_id, (emission.position, emission.direction, float(emission.spectrum[0][0]), emission.medium))
    order = [c.instance_id for c in request.project.components if c.instance_id in starts]
    axes = []
    for owner in order:
        position, direction, wavelength_nm, medium = starts[owner]
        axes.append(_axis(position, direction, owner, scene.surfaces, cfg, free_length_mm, d, special, library, wavelength_nm, medium))
    return axes, None
