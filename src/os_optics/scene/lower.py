"""Lowering: public contracts -> internal, validated, global-frame scene (once per validation call)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import NumericalConfig
from ..contracts import optics as co
from ..contracts.components import EngineeringComponentSnapshot, ExtendedEmitter, SourceSpec
from ..contracts.project import Observation, ProjectSnapshot
from .._vendor.shared_contracts.common.sag_shapes import (
    ADDITIVE_KINDS,
    SAG_COMBINATION_NOT_SUPPORTED,
    SAG_OUTSIDE_DOMAIN,
    ZERNIKE,
    sag_form,
    sag_value,
    xy_terms_problem,
    zernike_terms_problem,
)
from .._vendor.shared_contracts.common.polygons import polygon_problem
from ..errors import EngineInputError, ErrorCode, ModelScopeRefusal
from ..errors.issues import make_issue
from ..errors.status import Status
from ..math.exact_trig import cos_sin
from ..math.pose import Pose
from ..math.vectors import correctly_rounded_sum, dot, matvec, normalize, transverse_basis
from ..units import NM_TO_MM
from ..optics import apertures
from ..optics.attenuation import AttenuatorG
from ..optics.materials.coatings import CoatingEvaluator
from ..optics.polarization.partial import partially_polarized
from ..optics.polarization.phases import PhaseEvaluator
from ..optics.materials.library import MaterialLibrary
from ..optics.polarization.faraday import FaradayResponse, RetardanceTable
from ..optics.surfaces.grating import GratingG, GratingResponse
from ..optics.surfaces.tabulated_lens import ThinLensTable
from ..optics.complex_transmission import TransmissionMapG, grid_extent, lower_map
from ..optics.surfaces.facet_topology import solid_problem
from ..optics.fields.source_fields import (
    MAX_PHASE_STEP_RAD,
    PatternG,
    SourceGrid,
    SpeckleParams,
    field_validity,
    illumination_weights,
    realization_phase,
)
from ..optics.surfaces.facets import FITTED_FACE_NOTICE_FRACTION, coplanar_tolerance, mesh_extent, outline_problem, planar_patches
from .model import ApertureG, EmitterG, FacetG, FreeformG, GradientFaceG, GradientG, InteractionG, NonlinearG, ScatterG, Scene, SourceG, SurfaceG

Array = NDArray[np.float64]



def coherence_length_from_linewidth_mm(wavelength_nm: float, linewidth_nm: float) -> float:
    """``L_c = (2 sqrt(ln 2) / pi) lambda^2 / d_lambda`` (|gamma| = 1/e at OPD = L_c for a Gaussian spectrum), in mm."""
    return (2.0 * math.sqrt(math.log(2.0)) / math.pi) * (wavelength_nm * wavelength_nm) / linewidth_nm * NM_TO_MM


def _unit(v: tuple[float, ...], cfg: NumericalConfig, what: str, location: str) -> Array:
    """The unit vector along ``v``: a direction is taken by its direction, whatever its length; only a zero-length one is
    refused."""
    a = np.array(v, dtype=np.float64)
    n = math.sqrt(float(dot(a, a)))
    if n <= cfg.zero_vector_tol:
        raise EngineInputError(ErrorCode.ZERO_LENGTH_VECTOR, f"{what} is a zero-length vector", location=location)
    return a / n


def _global_unit(pose: Pose, v: tuple[float, ...], cfg: NumericalConfig, what: str, location: str) -> Array:
    return matvec(pose.rotation, _unit(v, cfg, what, location))


def _lower_pose(pose3d: object, cfg: NumericalConfig, location: str) -> Pose:
    """The pose with its quaternion normalized (a rotation is taken by its direction, whatever the quaternion's length);
    only one that cannot be normalized is refused."""
    try:
        return pose3d.to_pose()  # type: ignore[attr-defined]
    except ValueError as exc:
        raise EngineInputError(ErrorCode.ZERO_LENGTH_VECTOR, f"the pose's quaternion has no direction ({exc})", location=location) from exc


def _source_jones(jones: object, cfg: NumericalConfig, location: str) -> Array:
    """A source's Jones vector as the trace uses it: unit norm. Within ``energy_conservation_tol`` of 1 the components are
    used as given; any other non-zero vector is divided by its norm (the source's power is stated separately)."""
    n = jones.norm  # type: ignore[attr-defined]
    if not math.isfinite(n) or n <= cfg.zero_vector_tol:
        raise EngineInputError(ErrorCode.INVALID_INPUT, "source polarization Jones vector is zero: state the polarization as a "
                               "non-zero vector (its length does not matter)", location=location)
    vector = np.array([complex(*jones.ex), complex(*jones.ey)])  # type: ignore[attr-defined]
    return vector if abs(n - 1.0) <= cfg.energy_conservation_tol else vector / n


#: Degrees per quarter turn and per half turn.
QUARTER_TURN_DEG = 90.0
HALF_TURN_DEG = 180.0


def _turn(angle_deg: float) -> tuple[float, float]:
    """``(cos, sin)`` of an in-plane angle (exact at multiples of 90 degrees; libm-free otherwise)."""
    quarter = angle_deg / QUARTER_TURN_DEG
    if quarter == int(quarter):
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[int(quarter) % 4]
    c, s = cos_sin(np.array([angle_deg * (math.pi / HALF_TURN_DEG)]))
    return float(c[0]), float(s[0])


def lower_aperture(ap: co.Aperture | None) -> ApertureG:
    """A contract aperture as the region ``optics/apertures.py`` answers about (``none`` for ``None``)."""
    return _lower_aperture(ap)


def _lower_aperture(ap: co.Aperture | None) -> ApertureG:
    if ap is None:
        return ApertureG()
    if isinstance(ap, co.CircularAperture):
        return ApertureG("circle", ap.radius_mm, 0.0, ap.center_offset_mm[0], ap.center_offset_mm[1], ap.obstruction_radius_mm)
    if isinstance(ap, co.EllipticalAperture):
        return ApertureG("ellipse", ap.semi_axes_mm[0], ap.semi_axes_mm[1], ap.center_offset_mm[0], ap.center_offset_mm[1],
                         axis=_turn(ap.rotation_deg))
    if isinstance(ap, co.DShapedAperture):
        return ApertureG("d_shape", ap.radius_mm, ap.flat_distance_mm, ap.center_offset_mm[0], ap.center_offset_mm[1],
                         axis=_turn(ap.flat_angle_deg))
    if isinstance(ap, co.PolygonAperture):
        return ApertureG("polygon", 0.0, 0.0, ap.center_offset_mm[0], ap.center_offset_mm[1],
                         vertices=tuple((float(x), float(y)) for x, y in ap.vertices_mm))
    return ApertureG("rectangle", ap.half_width_mm, ap.half_height_mm, ap.center_offset_mm[0], ap.center_offset_mm[1], 0.0)


def _with_holes(ap: ApertureG, outline: ApertureG | None, margin: float, holes: tuple[co.Aperture, ...] | None, cfg: NumericalConfig,
                loc: str) -> tuple[ApertureG, ApertureG | None]:
    """The clear aperture and outline with the surface's holes (each inside the clear aperture, the holes apart), and a
    polygon's mount ring checked: the polygon grown by ``margin`` must stay a simple polygon."""
    if ap.shape == "polygon" and margin > 0.0 and outline is None:
        problem = polygon_problem(apertures.offset_polygon(ap.vertices, margin))
        if problem is not None:
            raise EngineInputError(ErrorCode.DEGENERATE_SURFACE, f"the polygon grown by its mount ring ({margin:.6g} mm) is not a "
                                   f"simple polygon ({problem}): state its mount_outline", location=loc)
    if not holes:
        return ap, outline
    lowered = tuple(_lower_aperture(hole) for hole in holes)
    for i, hole in enumerate(lowered):
        room = apertures.room(ap, hole.offset_u, hole.offset_v)
        if room + cfg.aperture_edge_tol_mm < apertures.reach(hole):
            raise EngineInputError(ErrorCode.DEGENERATE_SURFACE, f"hole {i} does not lie inside the clear aperture (it reaches "
                                   f"{apertures.reach(hole):.6g} mm from its centre, {max(room, 0.0):.6g} mm of the aperture lie around it)",
                                   location=loc)
        for j in range(i):
            other = lowered[j]
            du, dv = hole.offset_u - other.offset_u, hole.offset_v - other.offset_v
            if math.sqrt(du * du + dv * dv) < apertures.reach(hole) + apertures.reach(other):
                raise EngineInputError(ErrorCode.DEGENERATE_SURFACE, f"holes {j} and {i} are not apart (their enclosing circles "
                                       "about their centres overlap)", location=loc)
    return replace(ap, holes=lowered), (None if outline is None else replace(outline, holes=lowered))


def _phases(inter: co.RefractInteraction | co.MirrorInteraction | co.BeamSplitterInteraction, cfg: NumericalConfig) -> PhaseEvaluator | None:
    return PhaseEvaluator(inter.phases, cfg) if inter.phases is not None else None


def _scatter(inter: co.RefractInteraction | co.MirrorInteraction | co.AbsorberInteraction) -> ScatterG | None:
    spec = inter.scatter
    if spec is None:
        return None
    return ScatterG(fraction=spec.fraction, side=spec.side, law=spec.bsdf, exponent=1.0 if spec.exponent is None else spec.exponent,
                    sample_count=spec.sample_count, seed=spec.seed)


def _lower_interaction(inter: co.Interaction, pose: Pose, cfg: NumericalConfig, location: str) -> InteractionG:
    if isinstance(inter, co.RefractInteraction):
        return InteractionG("refract", coating=CoatingEvaluator(inter.coating, cfg) if inter.coating else None,
                            trace_reflection=inter.trace_fresnel_reflection, phases=_phases(inter, cfg), scatter=_scatter(inter))
    if isinstance(inter, co.MirrorInteraction):
        return InteractionG("mirror", coating=CoatingEvaluator(inter.coating, cfg) if inter.coating else None,
                            reflective_side=inter.reflective_side, phase_s=inter.phase_s_rad, phase_p=inter.phase_p_rad,
                            phases=_phases(inter, cfg), scatter=_scatter(inter))
    if isinstance(inter, co.ThinLensInteraction):
        return InteractionG("thin_lens", focal_length=inter.focal_length_mm, transmittance=inter.transmittance)
    if isinstance(inter, co.TabulatedThinLensInteraction):
        # the per-line focal length / transmittance come from the table; the plane's in-plane axes are filled in by
        # _lower_surface (polarization check)
        table = ThinLensTable(
            wavelength_nm=inter.wavelength_nm, focal_length_mm=inter.focal_length_mm, transmittance=inter.transmittance,
            focal_interpolation=inter.focal_interpolation, transmittance_interpolation=inter.transmittance_interpolation,
            point_tolerance=inter.point_tolerance, max_field_angle_deg=inter.max_field_angle_deg,
            max_beam_radius_mm=inter.max_beam_radius_mm, validated_polarizations=inter.validated_polarizations,
            supported_metrics=None if inter.supported_metrics is None else frozenset(inter.supported_metrics))
        return InteractionG("thin_lens", focal_length=inter.focal_length_mm[0], transmittance=inter.transmittance[0], table=table)
    if isinstance(inter, co.BeamSplitterInteraction):
        return InteractionG("beam_splitter", coating=CoatingEvaluator(inter.coating, cfg),
                            reflection_phase=inter.reflection_phase_rad, transmission_phase=inter.transmission_phase_rad,
                            phases=_phases(inter, cfg))
    if isinstance(inter, co.PolarizerInteraction):
        return InteractionG("polarizer", axis=_global_unit(pose, inter.axis_local, cfg, "polarizer axis", location),
                            t_par=inter.transmittance_parallel, t_perp=inter.transmittance_perpendicular,
                            bands=tuple((b.min_nm, b.max_nm, b.transmittance_parallel, b.transmittance_perpendicular)
                                        for b in inter.transmittance_bands or ()))
    if isinstance(inter, co.RetarderInteraction):
        return InteractionG("retarder", axis=_global_unit(pose, inter.fast_axis_local, cfg, "retarder fast axis", location),
                            retardance=inter.retardance_rad, design_wavelength_nm=inter.design_wavelength_nm,
                            achromatic=inter.achromatic, transmittance=inter.transmittance,
                            bands=tuple((b.min_nm, b.max_nm, b.transmittance, 0.0) for b in inter.transmittance_bands or ()),
                            retardance_table=None if inter.retardance_table_nm is None else RetardanceTable(
                                inter.retardance_table_nm, inter.retardance_table_rad, inter.retardance_interpolation))  # type: ignore[arg-type]
    if isinstance(inter, co.FaradayRotatorInteraction):
        return InteractionG("faraday_rotator", faraday=FaradayResponse(inter.wavelength_nm, inter.rotation_rad, inter.transmittance, inter.interpolation))
    if isinstance(inter, co.DiffractionGratingInteraction):
        table = inter.response
        response = None if table is None else GratingResponse(
            table.normalization, table.polarization_basis, table.interpolation, table.data_kind, table.wavelength_nm, table.aoi_deg, table.orders,
            table.efficiency_te, table.efficiency_tm, table.absorption_te, table.absorption_tm, table.max_conical_angle_deg)
        # (the plane's normal gives the groove axis: _lower_surface completes it)
        return InteractionG("diffraction_grating", grating=GratingG(
            inter.mode == "reflection", inter.period_nm, _global_unit(pose, inter.grating_vector_local, cfg, "grating vector", location),
            np.zeros(3), inter.orders, inter.all_propagating, response))
    if isinstance(inter, co.AbsorberInteraction):
        return InteractionG("absorber", scatter=_scatter(inter))
    if isinstance(inter, co.ApertureStopInteraction):
        return InteractionG("aperture_stop", transmittance=inter.transmittance)
    if isinstance(inter, co.NeutralDensityInteraction):
        return InteractionG("neutral_density")  # the plane frame is set by _lower_surface (_attenuator)
    if isinstance(inter, co.ComplexTransmissionInteraction):
        return InteractionG("complex_transmission", transmission_map=lower_map(inter))
    raise EngineInputError(ErrorCode.INVALID_INPUT, f"unsupported interaction {type(inter).__name__}", location=location)  # pragma: no cover


def _refuse_mesh(surf: co.MeshSurface, why: str, loc: str, action: str, **context: object) -> None:
    approximation = surf.approximation.kind + ("" if surf.approximation.max_chord_deviation_mm is None
                                               else f", chord deviation {surf.approximation.max_chord_deviation_mm:g} mm")
    raise ModelScopeRefusal(
        ErrorCode.FACET_SURFACE_NOT_TRACED,
        f"surface {surf.surface_id} is a facet surface ({len(surf.triangles)} triangles, {approximation}) the trace does not take: {why}",
        location=loc, suggested_action=action, **context)


#: The most fitted faces a FACET_FACE_FITTED notice names (it counts them all).
NAMED_FITTED_FACES = 12


def _fitted_face_notice(owner: str, surf: co.MeshSurface, vertices, patches, loc: str):
    """The notice (``FACET_FACE_FITTED``, NOT_APPLICABLE) for the faces of a facet surface fitted to facets whose corners lie
    farther from the face than ``FITTED_FACE_NOTICE_FRACTION`` of the surface's extent, or ``None``."""
    extent = mesh_extent(vertices)
    limit = FITTED_FACE_NOTICE_FRACTION * extent
    fitted = [p for p in patches if len(p.triangles) > 1 and p.deviation_mm > limit]
    if not fitted:
        return None
    worst = max(p.deviation_mm for p in fitted)
    names = [f"{owner}/{surf.surface_id}.t{p.first}" for p in fitted]
    listed = ", ".join(names[:NAMED_FITTED_FACES]) + (", ..." if len(names) > NAMED_FITTED_FACES else "")
    which = (f"face {listed} is traced as one flat face fitted to facets that lie within the coordinates' rounding of one plane; "
             "its corners lie" if len(names) == 1 else
             f"{len(names)} faces ({listed}) are each traced as one flat face fitted to facets that lie within the coordinates' "
             "rounding of one plane; their corners lie")
    return make_issue(
        ErrorCode.FACET_FACE_FITTED, Status.NOT_APPLICABLE,
        f"surface {surf.surface_id}: {which} up to {worst:.3g} mm from that plane, {worst / extent:.2g} of the part's size of "
        f"{extent:.4g} mm (this notice is given above {FITTED_FACE_NOTICE_FRACTION:g} of the size): a bend of the part smaller than "
        "that is traced flat",
        location=loc, suggested_action="state the vertices with more decimals where the facets are meant to bend",
        faces=names[:NAMED_FITTED_FACES], face_count=len(names), max_face_deviation_mm=worst, part_size_mm=extent,
        notice_fraction=FITTED_FACE_NOTICE_FRACTION, coplanar_tolerance_mm=coplanar_tolerance(vertices))


def _lower_mesh(owner: str, surf: co.MeshSurface, pose: Pose, cfg: NumericalConfig, library: MaterialLibrary,
                notices: list | None = None) -> list[SurfaceG]:
    """A facet surface as its planar patches (``optics/surfaces/facets.py``): one plane per patch, ``<surface id>.t<k>`` (``k``
    the patch's first triangle), its clear aperture the patch's outline, its interaction the patch's facet group's. The
    patches are found from the vertices as written and then moved by the surface's ``placement`` (if any) and ``pose``."""
    loc = f"component:{owner}/surface:{surf.surface_id}"
    library.evaluator(surf.medium_positive, location=loc)
    library.evaluator(surf.medium_negative, location=loc)
    interactions = (surf.interaction, *(group.interaction for group in surf.facet_groups or ()))
    if not surf.closed and surf.medium_positive != surf.medium_negative and any(isinstance(i, co.RefractInteraction) for i in interactions):
        _refuse_mesh(surf, f"its facets are an open sheet (closed false) that refracts between {surf.medium_positive} and "
                     f"{surf.medium_negative}: a solid of {surf.medium_negative} is bounded by closed facets, and light that enters "
                     "through an open sheet has no face to leave by", loc,
                     "close the facet surface around the solid (closed true, every edge on two facets), or give the open sheet "
                     "the same medium on both sides (a reflecting or absorbing sheet)")
    crystals = sorted(m for m in {surf.medium_positive, surf.medium_negative} if library.uniaxial(m) is not None)
    if crystals:
        _refuse_mesh(surf, f"the uniaxial crystal {crystals[0]} lies on a side of its facets: crystal faces are traced on declared "
                     "plane surfaces", loc, "state the crystal's faces as plane surfaces", material=crystals[0])
    vertices = tuple(tuple(float(x) for x in v) for v in surf.vertices_mm)
    patches = planar_patches(vertices, surf.triangles, surf.facet_group_of())
    # (each patch is a plane every trace step tests: the run's compute budget bounds them)
    if len(patches) > cfg.max_mesh_patches:
        _refuse_mesh(surf, f"its facets form {len(patches)} planar patches, more than the {cfg.max_mesh_patches} the trace "
                     "takes on one surface (compute_budget.max_mesh_patches)", loc,
                     "merge coplanar facets, tessellate the surface more coarsely within its stated chord deviation, or raise "
                     f"compute_budget.max_mesh_patches to at least {len(patches)} (set_compute_budget; every trace step then "
                     "tests that many planes)", patch_count=len(patches), limit=cfg.max_mesh_patches,
                     setting="max_mesh_patches")
    if surf.closed:
        problem = solid_problem(vertices, surf.triangles, [p.triangles for p in patches])
        if problem is not None:
            _refuse_mesh(surf, f"its facets do not bound one solid: {problem}", loc,
                         "state each solid's surface once (shells that do not touch or cross; the surface of a cavity facing into "
                         "it), or state touching or overlapping parts as separate components")
    notice = _fitted_face_notice(owner, surf, vertices, patches, loc)
    if notice is not None and notices is not None:
        notices.append(notice)
    lowered = [_lower_interaction(inter, pose, cfg, loc) for inter in interactions]
    if surf.placement is not None:  # (the faces of the vertices as written, placed in the owning frame)
        pose = pose.compose(_lower_pose(surf.placement, cfg, f"{loc}/placement"))
    chord = surf.approximation.max_chord_deviation_mm
    out: list[SurfaceG] = []
    for patch in patches:
        problem = outline_problem(patch.polygon)
        if problem is not None:  # pragma: no cover - a valid (non-degenerate) facet is a simple triangle
            raise EngineInputError(ErrorCode.DEGENERATE_SURFACE, f"facet {patch.first} has no simple outline ({problem})", location=loc)
        normal = matvec(pose.rotation, np.array(patch.normal, dtype=np.float64))
        u_axis = matvec(pose.rotation, np.array(patch.u_axis, dtype=np.float64))
        v_axis = np.cross(normal, u_axis)
        ap = ApertureG("polygon", 0.0, 0.0, 0.0, 0.0, vertices=patch.polygon)
        facet = FacetG(f"{owner}/{surf.surface_id}", surf.approximation.kind, chord, surf.closed, patch.triangles)
        out.append(SurfaceG(f"{owner}/{surf.surface_id}.t{patch.first}", owner, "plane", pose.apply_points(np.array(patch.origin)),
                            normal, u_axis, v_axis, np.zeros(3), 0.0, np.zeros(3), ap, 0.0, surf.medium_positive,
                            surf.medium_negative, lowered[patch.group], facet=facet))
    return out


def facet_patches(owner: str, surf: co.MeshSurface) -> tuple[tuple[str, co.RefractInteraction | co.MirrorInteraction | co.AbsorberInteraction], ...]:
    """The scene id (``<owner>/<surface id>.t<k>``) and contract interaction of each planar patch of a facet surface."""
    vertices = tuple(tuple(float(x) for x in v) for v in surf.vertices_mm)
    interactions = (surf.interaction, *(group.interaction for group in surf.facet_groups or ()))
    return tuple((f"{owner}/{surf.surface_id}.t{p.first}", interactions[p.group])
                 for p in planar_patches(vertices, surf.triangles, surf.facet_group_of()))


def facet_patch_ids(owner: str, surf: co.MeshSurface) -> tuple[str, ...]:
    """The scene ids of a facet surface's planar patches (``<owner>/<surface id>.t<k>``)."""
    return tuple(sid for sid, _inter in facet_patches(owner, surf))


def _lower_surface(owner: str, surf: co.OpticalSurface, pose: Pose, cfg: NumericalConfig, library: MaterialLibrary,
                   notices: list | None = None) -> SurfaceG:
    loc = f"component:{owner}/surface:{surf.surface_id}"
    library.evaluator(surf.medium_positive, location=loc)
    library.evaluator(surf.medium_negative, location=loc)
    inter = _lower_interaction(surf.interaction, pose, cfg, loc)
    zeros = np.zeros(3)
    if isinstance(surf, co.PlaneSurface):
        normal = _global_unit(pose, surf.normal, cfg, "surface normal", loc)
        if surf.aperture_u_axis is not None:
            u_raw = matvec(pose.rotation, np.array(surf.aperture_u_axis, dtype=np.float64))
            u, ok = normalize((u_raw - float(dot(u_raw, normal)) * normal)[None, :], cfg.zero_vector_tol)
            if not bool(ok[0]):
                raise EngineInputError(ErrorCode.DEGENERATE_SURFACE, "aperture_u_axis is parallel to the normal", location=loc)
            u_len = math.sqrt(float(dot(u_raw, u_raw)))
            tilt = math.asin(min(1.0, abs(float(dot(u_raw, normal))) / u_len)) if u_len > 0.0 else 0.0
            if tilt > cfg.unit_vector_tol and notices is not None:
                notices.append(_u_axis_notice(surf, tilt, cfg, loc, "aperture_u_axis", "The aperture's u direction"))
            u_axis = u[0]
            v_axis = np.cross(normal, u_axis)
        else:
            e1, e2 = transverse_basis(normal[None, :], cfg.zero_vector_tol)
            u_axis, v_axis = e1[0], e2[0]
        ap = _lower_aperture(surf.aperture)
        if ap.shape == "circle" and ap.p1 <= 0:  # pragma: no cover - contract prevents
            raise EngineInputError(ErrorCode.DEGENERATE_SURFACE, "zero-size aperture", location=loc)
        ap, outline = _with_holes(ap, _lower_outline(surf.mount_outline), surf.mount_margin_mm, surf.holes, cfg, loc)
        if inter.grating is not None:
            k_axis, ok = normalize((inter.grating.k_axis - float(dot(inter.grating.k_axis, normal)) * normal)[None, :], cfg.zero_vector_tol)
            if not bool(ok[0]):
                raise EngineInputError(ErrorCode.DEGENERATE_SURFACE, "the grating vector is parallel to the normal", location=loc)
            inter = replace(inter, grating=replace(inter.grating, k_axis=k_axis[0], groove_axis=np.cross(normal, k_axis[0])))
        if inter.table is not None:  # a tabulated thin lens judges polarization along the plane's in-plane axes
            inter = replace(inter, table=replace(inter.table, u_axis=u_axis.copy(), v_axis=v_axis.copy()))
        if inter.transmission_map is not None:
            _check_transmission_map(surf, ap, inter.transmission_map, cfg, loc)
        if isinstance(surf.interaction, co.NeutralDensityInteraction):
            inter = InteractionG("neutral_density", attenuator=_attenuator(surf.interaction, pose, pose.apply_points(np.array(surf.origin_mm)),
                                                                          normal, cfg, loc))
        return SurfaceG(f"{owner}/{surf.surface_id}", owner, "plane", pose.apply_points(np.array(surf.origin_mm)), normal,
                        u_axis, v_axis, zeros, 0.0, zeros, ap, surf.mount_margin_mm, surf.medium_positive,
                        surf.medium_negative, inter, outline=outline)
    if isinstance(surf, co.SagSurface):
        return _lower_sag(owner, surf, pose, cfg, inter, loc, notices)
    cap = _global_unit(pose, surf.cap_axis, cfg, "cap_axis", loc)
    e1, e2 = transverse_basis(cap[None, :], cfg.zero_vector_tol)
    return SurfaceG(f"{owner}/{surf.surface_id}", owner, "sphere", zeros, cap, e1[0], e2[0],
                    pose.apply_points(np.array(surf.center_mm)), surf.radius_mm, cap,
                    ApertureG("circle", surf.aperture_radius_mm), surf.mount_margin_mm,
                    surf.medium_positive, surf.medium_negative, inter, aperture_radius=surf.aperture_radius_mm)


def _attenuator(inter: co.NeutralDensityInteraction, pose: Pose, origin: Array, normal: Array, cfg: NumericalConfig,
                location: str) -> AttenuatorG:
    """The attenuator's density law on its plane (global frame): ``x`` = the law's axis projected into the plane."""
    x_raw = _global_unit(pose, inter.x_axis_local, cfg, "attenuator x axis", location)
    x, ok = normalize((x_raw - float(dot(x_raw, normal)) * normal)[None, :], cfg.zero_vector_tol)
    if not bool(ok[0]):
        raise EngineInputError(ErrorCode.DEGENERATE_SURFACE, "the attenuator's x axis is parallel to its normal", location=location)
    x_axis = x[0]
    law = inter.law
    params: dict = {}
    if isinstance(law, co.FixedDensity):
        params = {"optical_density": law.optical_density}
    elif isinstance(law, co.LinearRotaryDensity):
        params = {"od_start": law.od_start, "od_end": law.od_end, "start": law.arc_start_deg, "end": law.arc_end_deg}
    elif isinstance(law, co.LinearGradientDensity):
        params = {"od_start": law.od_start, "od_end": law.od_end, "start": law.x_start_mm, "end": law.x_end_mm}
    else:
        params = {"steps": tuple((step.start_deg, step.end_deg, step.optical_density) for step in law.steps)}
    return AttenuatorG(kind=law.kind, origin=origin, normal=normal, x_axis=x_axis, y_axis=np.cross(normal, x_axis),
                       position=inter.position, reflectance=inter.reflectance, uncoated_transmittance=inter.uncoated_transmittance,
                       inner_radius=inter.coated_inner_radius_mm, **params)


def _check_transmission_map(surf: co.PlaneSurface, ap: ApertureG, m: TransmissionMapG, cfg: NumericalConfig, loc: str) -> None:
    """A complex-transmission plane is thin (the same medium on both sides) and bounded by a clear aperture its sampled map
    covers and its Zernike terms' normalization radius holds."""
    if surf.medium_positive != surf.medium_negative:
        raise EngineInputError(ErrorCode.INVALID_INPUT, "a complex-transmission element is thin: it has the same medium on both sides",
                               location=loc)
    if not apertures.bounded(ap):
        raise EngineInputError(ErrorCode.INVALID_INPUT, "a complex-transmission element states a clear aperture", location=loc)
    u_min, u_max, v_min, v_max = apertures.bounds(ap)
    if m.grid is not None:
        g_u0, g_u1, g_v0, g_v1 = grid_extent(m.grid)
        tol = cfg.aperture_edge_tol_mm
        if u_min < g_u0 - tol or u_max > g_u1 + tol or v_min < g_v0 - tol or v_max > g_v1 + tol:
            raise EngineInputError(ErrorCode.INVALID_INPUT, f"the sampled map (u {g_u0:.6g} .. {g_u1:.6g}, v {g_v0:.6g} .. {g_v1:.6g} mm) does not "
                                   f"cover the clear aperture (u {u_min:.6g} .. {u_max:.6g}, v {v_min:.6g} .. {v_max:.6g} mm)", location=loc)
    if m.zernike_radius_mm is not None:
        reach = max(math.sqrt(u * u + v * v) for u in (u_min, u_max) for v in (v_min, v_max))
        if reach > m.zernike_radius_mm + cfg.aperture_edge_tol_mm:
            raise EngineInputError(ErrorCode.INVALID_INPUT, f"the clear aperture reaches {reach:.6g} mm from the origin, beyond the Zernike "
                                   f"phase's normalization radius {m.zernike_radius_mm:.6g} mm", location=loc)


def _lower_outline(outline) -> ApertureG | None:
    """A surface's mount outline (``None``: the clear aperture grown by its mount ring)."""
    return None if outline is None else _lower_aperture(outline)


def _aperture_reach(ap: ApertureG, cylindrical: bool) -> float:
    """Largest ``rho`` (``|a|`` for a cylinder) of a point of the clear aperture, measured from the vertex (any shape but a
    circle and a rectangle: a bound that holds it)."""
    if ap.shape not in ("circle", "rectangle"):
        return apertures.far_u(ap) if cylindrical else apertures.rho_range(ap)[1]
    half_v = ap.p1 if ap.shape == "circle" else ap.p2
    if cylindrical:
        return abs(ap.offset_u) + ap.p1
    if ap.shape == "circle":
        return math.hypot(ap.offset_u, ap.offset_v) + ap.p1
    return math.hypot(abs(ap.offset_u) + ap.p1, abs(ap.offset_v) + half_v)


#: Rational points per quarter turn of a circular clear aperture's rim where a two-variable profile must be defined
#: (``t = tan(angle / 2)`` in equal steps from 0 to 1).
APERTURE_SAMPLE_STEPS = 16


def _aperture_samples(ap: ApertureG) -> list[tuple[float, float]]:
    """The centre and boundary points of a clear aperture where a two-variable profile must be defined (a circle: its rim
    at rational points, ``APERTURE_SAMPLE_STEPS`` per quarter turn)."""
    points = [(ap.offset_u, ap.offset_v)]
    if ap.shape not in ("circle", "rectangle"):  # (its boundary ring: corners exactly, round parts at RING_POINTS per turn)
        return points + apertures.ring(ap)
    half_v = ap.p1 if ap.shape == "circle" else ap.p2
    if ap.shape != "circle":
        return points + [(ap.offset_u + su * ap.p1, ap.offset_v + sv * half_v) for su in (-1.0, 0.0, 1.0) for sv in (-1.0, 0.0, 1.0)]
    for k in range(APERTURE_SAMPLE_STEPS + 1):
        t = k / APERTURE_SAMPLE_STEPS
        c, s = (1.0 - t * t) / (1.0 + t * t), 2.0 * t / (1.0 + t * t)
        points += [(ap.offset_u + su * c * ap.p1, ap.offset_v + sv * s * ap.p1) for su in (-1.0, 1.0) for sv in (-1.0, 1.0)]
    return points


def _refuse_shape(problem: str, loc: str) -> None:
    """Raise a refused shape's reason (``CODE: message``) as its error code."""
    code, _, message = problem.partition(": ")
    raise EngineInputError(ErrorCode(code), message, location=loc)


def _lower_freeform(surf: co.SagSurface, ap: ApertureG, cfg: NumericalConfig, loc: str) -> FreeformG:
    """A surface's two-variable sag shape: every combination, term and aperture it cannot honour is refused with its reason."""
    shape = surf.sag_shape
    assert shape is not None
    if surf.cylindrical or surf.linear_sag_coefficient is not None:
        _refuse_shape(f"{SAG_COMBINATION_NOT_SUPPORTED}: a {shape.kind} sag shape does not combine with a cylinder or an axicon term", loc)
    if shape.kind not in ADDITIVE_KINDS and (surf.curvature_per_mm != 0.0 or surf.conic != 0.0 or surf.aspheric_coefficients):
        _refuse_shape(f"{SAG_COMBINATION_NOT_SUPPORTED}: a {shape.kind} sag shape replaces the vertex curvature, conic and aspheric terms", loc)
    if shape.kind == ZERNIKE:
        problem = zernike_terms_problem(shape.indexing, shape.normalization, [term.index for term in shape.terms])
    elif shape.kind in ADDITIVE_KINDS:
        problem = xy_terms_problem([(term.x_power, term.y_power) for term in shape.terms])
    else:
        problem = None
    if problem is not None:
        _refuse_shape(problem, loc)
    # (a Zernike sum is a polynomial defined at every radius: a clear aperture beyond its normalization radius evaluates the
    # stated terms there; the reach is stated on the surface's evidence, ``SurfaceProfileEvidence.aperture_reach_mm``)
    reach = None
    if shape.kind == ZERNIKE:
        reach = _aperture_reach(ap, False)
        reach = reach if reach > shape.normalization_radius_mm + cfg.aperture_edge_tol_mm else None
    form = sag_form(shape)
    for a, b in _aperture_samples(ap):
        if not sag_value(form, a, b)[1]:
            _refuse_shape(f"{SAG_OUTSIDE_DOMAIN}: the {shape.kind} profile is not defined at ({a:.6g}, {b:.6g}) mm in the clear aperture", loc)
    return FreeformG(form, shape.kind, getattr(shape, "indexing", None), getattr(shape, "normalization", None),
                     getattr(shape, "normalization_radius_mm", None), float(sag_value(form, 0.0, 0.0)[0]) + 0.0, aperture_reach=reach)


def _u_axis_notice(surf: co.SagSurface | co.PlaneSurface, tilt_rad: float, cfg: NumericalConfig, loc: str,
                   field: str = "u_axis", what: str = "The profile's u direction"):
    """The notice (``U_AXIS_PROJECTED``, NOT_APPLICABLE) for a surface whose stated in-plane axis (a sag surface's
    ``u_axis``, a plane's ``aperture_u_axis``) lies ``tilt_rad`` out of the surface plane: the direction used is the axis'
    projection onto the plane, the tilt removed."""
    degrees = math.degrees(tilt_rad)
    return make_issue(
        ErrorCode.U_AXIS_PROJECTED, Status.NOT_APPLICABLE,
        f"surface {surf.surface_id}: {field} lies {degrees:.6g} deg out of the surface plane. {what} is its "
        "projection onto that plane.",
        location=loc, suggested_action=f"state {field} perpendicular to the surface normal",
        tilt_deg=degrees, tolerance_rad=cfg.unit_vector_tol)


def _lower_sag(owner: str, surf: co.SagSurface, pose: Pose, cfg: NumericalConfig, inter: InteractionG, loc: str,
               notices: list | None = None) -> SurfaceG:
    """A conic / even-asphere surface: global vertex, axis and (u, v) axes; the clear aperture must lie inside the profile's
    domain (``1 - (1 + k) c^2 rho^2 >= 0``). A ``u_axis`` off the surface plane is taken by its projection onto the plane
    (``U_AXIS_PROJECTED`` notice into ``notices`` above ``unit_vector_tol``); one along the normal is refused."""
    normal = _global_unit(pose, surf.normal, cfg, "surface normal", loc)
    # (u_axis gives the in-plane direction of u: its component along the normal is projected out)
    u_raw = _global_unit(pose, surf.u_axis, cfg, "u_axis", loc)
    u, ok = normalize((u_raw - float(dot(u_raw, normal)) * normal)[None, :], cfg.zero_vector_tol)
    if not bool(ok[0]):
        raise EngineInputError(ErrorCode.DEGENERATE_SURFACE, "u_axis is parallel to the normal", location=loc)
    tilt = math.asin(min(1.0, abs(float(dot(u_raw, normal)))))
    if tilt > cfg.unit_vector_tol and notices is not None:
        notices.append(_u_axis_notice(surf, tilt, cfg, loc))
    u_axis = u[0]
    v_axis = np.cross(normal, u_axis)
    ap, outline = _with_holes(_lower_aperture(surf.aperture), _lower_outline(surf.mount_outline), surf.mount_margin_mm, surf.holes,
                              cfg, loc)
    k1c2 = (1.0 + surf.conic) * (surf.curvature_per_mm * surf.curvature_per_mm)
    if k1c2 > 0.0:
        reach, limit = _aperture_reach(ap, surf.cylindrical), 1.0 / math.sqrt(k1c2)
        if reach > limit + cfg.aperture_edge_tol_mm:
            raise EngineInputError(
                ErrorCode.DEGENERATE_SURFACE,
                f"the clear aperture reaches {reach:.6g} mm from the vertex, beyond the profile's domain ({limit:.6g} mm)",
                location=loc,
            )
    zeros = np.zeros(3)
    return SurfaceG(f"{owner}/{surf.surface_id}", owner, "sag", pose.apply_points(np.array(surf.origin_mm)), normal, u_axis, v_axis,
                    zeros, 0.0, normal, ap, surf.mount_margin_mm, surf.medium_positive, surf.medium_negative, inter,
                    curvature=surf.curvature_per_mm, conic=surf.conic, coefficients=tuple(surf.aspheric_coefficients),
                    cylindrical=surf.cylindrical, linear=surf.linear_sag_coefficient or 0.0, outline=outline,
                    freeform=None if surf.sag_shape is None else _lower_freeform(surf, ap, cfg, loc))


def _lower_source(owner: str, pose: Pose, spec: SourceSpec, cfg: NumericalConfig, library: MaterialLibrary) -> SourceG:
    loc = f"component:{owner}/source:{spec.source_id}"
    sp = pose.compose(_lower_pose(spec.local_pose, cfg, loc))
    library.evaluator(spec.medium, location=loc)
    if spec.spectrum:
        total = correctly_rounded_sum(line.weight for line in spec.spectrum)
        spectrum = tuple((line.wavelength_nm, line.weight / total) for line in spec.spectrum)
    else:
        spectrum = ((float(spec.wavelength_nm), 1.0),)  # type: ignore[arg-type]
    jones = _source_jones(spec.polarization, cfg, loc)
    # A declared coherence length holds for every line; a linewidth gives each line its own length from its wavelength.
    coh: tuple[float, ...] | None
    if spec.coherence_length_mm is not None:
        coh = tuple(spec.coherence_length_mm for _ in spectrum)
    elif spec.linewidth_nm is not None:
        coh = tuple(coherence_length_from_linewidth_mm(wavelength, spec.linewidth_nm) for wavelength, _ in spectrum)
    else:
        coh = None
    return SourceG(
        source_id=spec.source_id, owner_id=owner, position=sp.translation, direction=sp.rotation[:, 2].copy(),
        e1=sp.rotation[:, 0].copy(), e2=sp.rotation[:, 1].copy(), spectrum=spectrum, power_W=spec.power_W,
        w0x=spec.waist_radius_x_mm, w0y=spec.waist_radius_y_mm, z_waist=spec.waist_position_z_mm,
        m2x=spec.m2_x, m2y=spec.m2_y, jones=jones, medium=spec.medium, coherence_lengths_mm=coh,
        coherence_group=spec.coherence_group if spec.coherence_group is not None else f"source:{spec.source_id}",
        ring_count=spec.pupil_ring_count or cfg.pupil_ring_count,
        points_per_ring=spec.pupil_points_per_ring or cfg.pupil_points_per_ring,
        degree_of_polarization=partially_polarized(spec.degree_of_polarization),
        pulse=None if spec.pulse is None else (spec.pulse.repetition_rate_hz, spec.pulse.pulse_duration_s),
    )


def _source_grid(grid) -> SourceGrid:
    return SourceGrid(nx=grid.columns, ny=grid.rows, px=grid.pitch_x_mm, py=grid.pitch_y_mm)


def _lower_pattern(spec: ExtendedEmitter, cfg: NumericalConfig, library: MaterialLibrary, loc: str) -> PatternG | None:
    """The emitter's sampled pattern (``optics/fields/source_fields.py``); a coherent field's validity for its rays: every
    wrapped phase step between neighbouring lit grid points (``source_fields.lit``) at most ``MAX_PHASE_STEP_RAD``
    (``SOURCE_FIELD_UNDERSAMPLED``) and the phase gradient below the medium's wavenumber at every grid point a ray's direction is
    interpolated from (``SOURCE_FIELD_EVANESCENT``), in every realization. A field outside either is returned with that
    finding (``PatternG.out_of_scope``, an ``OUT_OF_MODEL_SCOPE`` issue at the source): the sampled trace launches none of
    its rays and books its lines as not traced, and the rest of the scene is evaluated."""
    if spec.intensity_map is not None:
        m = spec.intensity_map
        grid = _source_grid(m.grid)
        return PatternG(kind="map", grid=grid, interpolation=m.interpolation,
                        weights=np.array(m.values, dtype=np.float64).reshape(grid.ny, grid.nx))
    if spec.field is None and spec.speckle is None:
        return None
    coherent = spec.field if spec.field is not None else spec.speckle
    assert coherent is not None
    jones = _source_jones(coherent.jones, cfg, loc)
    grid = _source_grid(coherent.grid)
    if spec.field is not None:
        f = spec.field
        amplitude = np.array(f.amplitude, dtype=np.float64).reshape(grid.ny, grid.nx)
        pattern = PatternG(kind="field", grid=grid, interpolation="PIECEWISE_CONSTANT", weights=amplitude * amplitude,
                           phase=np.array(f.phase_rad, dtype=np.float64).reshape(grid.ny, grid.nx), jones=jones)
    else:
        sp = spec.speckle
        assert sp is not None
        params = SpeckleParams(illumination=sp.illumination, size_mm=sp.illumination_size_mm, phase_rms_rad=sp.phase_rms_rad,
                               correlation_length_mm=sp.correlation_length_mm, seed=sp.seed)
        weights = illumination_weights(params, grid)
        if not float(weights.max()) > 0.0:
            raise EngineInputError(ErrorCode.INVALID_INPUT, "the speckle illumination does not reach any grid point", location=loc)
        pattern = PatternG(kind="speckle", grid=grid, interpolation="PIECEWISE_CONSTANT", weights=weights, jones=jones,
                           speckle=params, realizations=sp.realizations)
    wavelength = spec.spectrum[0].wavelength_nm
    n, issue = library.evaluator(spec.medium, location=loc).index(wavelength, location=loc)
    if n is None:
        return pattern  # (no medium data: the line is booked SOURCE_NOT_TRACED by the sampled trace, with its issue)
    k_medium = 2.0 * math.pi * n / (wavelength * NM_TO_MM)
    worst_step, worst_sine, where = 0.0, 0.0, 0
    for m in range(pattern.realizations):
        step, sine = field_validity(realization_phase(pattern, m), pattern.weights, grid, k_medium)
        if step > worst_step:
            worst_step, where = step, m
        worst_sine = max(worst_sine, sine)
    context = {"pitch_mm": [grid.px, grid.py], "realizations": pattern.realizations}
    if worst_step > MAX_PHASE_STEP_RAD:
        return replace(pattern, out_of_scope=make_issue(
            ErrorCode.SOURCE_FIELD_UNDERSAMPLED, Status.OUT_OF_MODEL_SCOPE,
            f"source {spec.source_id}: the field's phase changes by {worst_step:.4g} rad between neighbouring lit grid points"
            + (f" (realization {where})" if pattern.realizations > 1 else "")
            + f"; its rays need at most {MAX_PHASE_STEP_RAD:.4g} rad (four grid points per 2 pi of phase), so this source is not traced",
            location=loc, suggested_action="sample the field on a finer grid" + (
                " (a speckle screen: a smaller pitch, a longer correlation length or a smaller phase_rms_rad)" if spec.speckle else ""),
            largest_phase_step_rad=worst_step, **context))
    if worst_sine >= 1.0:
        return replace(pattern, out_of_scope=make_issue(
            ErrorCode.SOURCE_FIELD_EVANESCENT, Status.OUT_OF_MODEL_SCOPE,
            f"source {spec.source_id}: the field's phase gradient reaches {worst_sine:.4g} times the medium's wavenumber at a grid "
            "point the rays take their directions from (a point of an emitting cell or its neighbour): that light is evanescent "
            "and does not propagate, so this source is not traced", location=loc,
            suggested_action="state a field whose transverse wavevector stays below the medium's wavenumber wherever its amplitude "
                             "is not zero (set the amplitude of unused grid points to 0 and keep their neighbours' phase smooth)",
            largest_transverse_sine=worst_sine, **context))
    return pattern


def _lower_emitter(owner: str, pose: Pose, spec: ExtendedEmitter, cfg: NumericalConfig, library: MaterialLibrary) -> EmitterG:
    loc = f"component:{owner}/source:{spec.source_id}"
    sp = pose.compose(_lower_pose(spec.local_pose, cfg, loc))
    library.evaluator(spec.medium, location=loc)
    total = correctly_rounded_sum(line.weight for line in spec.spectrum)
    law, parameter = spec.angular_law, spec.angular_parameter
    cone = parameter if law == "UNIFORM_CONE" else 0.0
    cone_sine = 1.0 if cone >= QUARTER_TURN_DEG else float(cos_sin(np.array([math.radians(cone)]))[1][0])  # type: ignore[operator]
    return EmitterG(
        source_id=spec.source_id, owner_id=owner, position=sp.translation, direction=sp.rotation[:, 2].copy(),
        e1=sp.rotation[:, 0].copy(), e2=sp.rotation[:, 1].copy(),
        spectrum=tuple((line.wavelength_nm, line.weight / total) for line in spec.spectrum), power_W=spec.power_W, medium=spec.medium,
        shape="disc" if spec.shape == "DISC" else "rectangle", half_x=spec.size_x_mm / 2.0, half_y=spec.size_y_mm / 2.0, law=law,
        sample_count=spec.sample_count, seed=spec.seed, coherence_group=f"source:{spec.source_id}",
        exponent=parameter if law == "COSINE_POWER" else 1.0, cone_half_angle_deg=cone, cone_sine=cone_sine,  # type: ignore[arg-type]
        pattern=_lower_pattern(spec, cfg, library, loc),
    )


def observation_global_pose(obs: Observation, component_poses: dict[str, Pose], cfg: NumericalConfig) -> Pose:
    local = _lower_pose(obs.pose, cfg, f"observation:{obs.observation_id}")
    if obs.device_id is None:
        return local
    if obs.device_id not in component_poses:
        raise EngineInputError(ErrorCode.UNKNOWN_REFERENCE, f"observation attached to unknown component {obs.device_id!r}",
                               location=f"observation:{obs.observation_id}")
    return component_poses[obs.device_id].compose(local)


def lower_scene(
    project: ProjectSnapshot,
    cfg: NumericalConfig,
    library: MaterialLibrary,
    *,
    component_poses: dict[str, Pose] | None = None,
) -> tuple[Scene, dict[str, Pose]]:
    """Build the lowered scene. ``component_poses`` may be supplied to override poses (actuator offsets)."""
    scene = Scene()
    poses: dict[str, Pose] = {}
    for comp in project.components:
        poses[comp.instance_id] = (component_poses or {}).get(comp.instance_id) or _lower_pose(
            comp.pose, cfg, f"component:{comp.instance_id}"
        )
    for comp in project.components:
        pose = poses[comp.instance_id]
        for surf in comp.surfaces:
            if isinstance(surf, co.MeshSurface):
                scene.surfaces.extend(_lower_mesh(comp.instance_id, surf, pose, cfg, library, scene.notices))
            else:
                scene.surfaces.append(_lower_surface(comp.instance_id, surf, pose, cfg, library, scene.notices))
        if comp.source is not None:
            scene.sources.append(_lower_source(comp.instance_id, pose, comp.source, cfg, library))
        if comp.emitter is not None:
            scene.emitters.append(_lower_emitter(comp.instance_id, pose, comp.emitter, cfg, library))
    for obs in project.observations:
        pose = observation_global_pose(obs, poses, cfg)
        library.evaluator(obs.medium, location=f"observation:{obs.observation_id}")
        active = _lower_aperture(obs.aperture)
        rot = pose.rotation
        scene.surfaces.append(
            SurfaceG(
                surface_id=f"observation/{obs.observation_id}", owner_id=obs.device_id or "observation", geometry="plane",
                origin=pose.translation, normal=rot[:, 2].copy(), u_axis=rot[:, 0].copy(), v_axis=rot[:, 1].copy(),
                center=np.zeros(3), radius=0.0, cap_axis=np.zeros(3), aperture=ApertureG(), mount_margin=0.0,
                medium_pos=obs.medium, medium_neg=obs.medium,
                interaction=InteractionG("observe", observation_id=obs.observation_id, absorbing=obs.absorbing,
                                         active_aperture=active if active.shape != "none" else None,
                                         acceptance_half_angle_deg=obs.acceptance_half_angle_deg, observation_medium=obs.medium),
            )
        )
    scene.surface_index = {s.surface_id: i for i, s in enumerate(scene.surfaces)}
    if len(scene.surface_index) != len(scene.surfaces):
        raise EngineInputError(ErrorCode.DUPLICATE_ID, "duplicate surface ids in the scene")
    scene.optic_axes = _crystals(scene, project, poses, cfg, library)
    scene.nonlinear = _nonlinear_crystals(scene, poses, cfg, library)
    scene.gradients = _gradients(scene, project, poses, cfg, library)
    return scene, poses


#: How far from 1 ``|n_1 . n_2|`` of the two faces of a nonlinear crystal may be (parallel faces of a slab).
NONLINEAR_FACE_PARALLEL_TOL = 1e-12


def _nonlinear_crystals(scene: Scene, poses: dict[str, Pose], cfg: NumericalConfig, library: MaterialLibrary) -> dict[str, NonlinearG]:
    """Every crystal of the scene that generates a second harmonic, with its slab -- exactly two faces, planes, parallel,
    apart -- and its crystal frame when it states a ``d`` tensor (``crystal_x_axis`` turned like its optic axis). A crystal
    that is not such a slab is kept with the reason (``NonlinearG.out_of_scope``): it is traced as the linear crystal it
    also is, and each pass into it reports its conversion, and the powers behind it, out of model scope."""

    out: dict[str, NonlinearG] = {}
    for material_id, axis in sorted(scene.optic_axes.items()):
        laws = library.uniaxial(material_id)
        assert laws is not None
        spec = laws.material.nonlinear
        if spec is None:
            continue
        faces = [i for i, surf in enumerate(scene.surfaces) if surf.interaction.kind != "observe"
                 and material_id in (surf.medium_pos, surf.medium_neg)]
        where = f"material:{material_id}"
        if len(faces) != 2:
            out[material_id] = NonlinearG(spec, None, (-1, -1), 0.0, out_of_scope=(
                f"the nonlinear crystal {material_id} has {len(faces)} faces: second-harmonic generation is traced in a slab "
                "between two parallel plane faces"))
            continue
        a, b = scene.surfaces[faces[0]], scene.surfaces[faces[1]]
        cos_ab = float(np.dot(a.normal, b.normal))
        thickness = abs(float(np.dot(b.origin - a.origin, a.normal)))
        if abs(abs(cos_ab) - 1.0) > NONLINEAR_FACE_PARALLEL_TOL or thickness <= cfg.ray_min_advance_mm:
            out[material_id] = NonlinearG(spec, None, (faces[0], faces[1]), thickness, out_of_scope=(
                f"the faces {a.surface_id} and {b.surface_id} of the nonlinear crystal {material_id} are not two parallel planes "
                "apart: second-harmonic generation is traced in a slab"))
            continue
        frame = None
        if spec.crystal_x_axis is not None:
            x = np.array(spec.crystal_x_axis, dtype=np.float64)
            component = laws.material.optic_axis_component
            if component is not None:
                x = matvec(poses[component].rotation, x)
            x = x - float(np.dot(x, axis)) * axis
            unit, ok = normalize(x[None, :], cfg.zero_vector_tol)
            if not bool(ok[0]):  # pragma: no cover - the contract holds a perpendicular unit vector
                raise EngineInputError(ErrorCode.ZERO_LENGTH_VECTOR, "crystal_x_axis along the optic axis", location=where)
            frame = np.vstack([unit[0], np.cross(axis, unit[0]), axis])
        out[material_id] = NonlinearG(spec, frame, (faces[0], faces[1]), thickness)
    return out


#: An end face of a gradient-index rod is perpendicular to its axis when its unit normal's component along the axis is at
#: least ``1 - GRADIENT_FACE_TOL`` in magnitude.
GRADIENT_FACE_TOL = 1e-12
#: The rod's transverse axes start from the scene X axis unless the rod's axis lies within this cosine of it (then Y).
GRADIENT_SEED_AXIS_COS = 0.9
#: A curved end face is centred on the rod's axis when its vertex lies within this distance (mm) of it.
GRADIENT_VERTEX_TOL_MM = 1e-9


def _gradients(scene: Scene, project: ProjectSnapshot, poses: dict[str, Pose], cfg: NumericalConfig,
               library: MaterialLibrary) -> dict[str, GradientG]:
    """Every gradient-index medium (radial or axial) the scene uses (``GradientG``), after checking it is used as the model
    traces it (``optics/rays/gradient.py``): exactly two faces bound it, each a refracting plane (across the axis or slanted)
    or a sphere / conic surface of revolution about the axis (``_gradient_face``), the medium between them, the faces not
    reaching past each other along the axis within the rod's radius; no source, emitter or observation plane lies in it."""

    def refuse(message: str, location: str, **context: object) -> None:
        raise EngineInputError(ErrorCode.GRADIENT_INDEX_UNSUPPORTED, message, location=location, **context)

    faces: dict[str, list[int]] = {}
    for index, surf in enumerate(scene.surfaces):
        if surf.interaction.kind == "observe":
            continue
        for medium in sorted({surf.medium_pos, surf.medium_neg}):
            if library.gradient(medium) is not None:
                faces.setdefault(medium, []).append(index)
    for src in (*scene.sources, *scene.emitters):
        if library.gradient(src.medium) is not None:
            refuse(f"source {src.source_id} starts inside the gradient-index medium {src.medium}", f"source:{src.source_id}",
                   material=src.medium, suggested_action="place the source outside the rod")
    for obs in project.observations:
        if library.gradient(obs.medium) is not None:
            refuse(f"observation {obs.observation_id} lies inside the gradient-index medium {obs.medium}: light is observed outside "
                   "it", f"observation:{obs.observation_id}", material=obs.medium,
                   suggested_action="place the observation plane outside the rod")
    out: dict[str, GradientG] = {}
    for material_id in sorted(faces):
        laws = library.gradient(material_id)
        assert laws is not None
        material = laws.material
        origin = np.array(material.axis_point_mm, dtype=np.float64)
        axis = np.array(material.axis_direction, dtype=np.float64)
        frame = material.axis_component
        if frame is not None:
            if frame not in poses:
                raise EngineInputError(ErrorCode.UNKNOWN_REFERENCE, f"the axis of {material_id} is given in the frame of the "
                                       f"unknown component {frame!r}", location=f"material:{material_id}")
            origin = poses[frame].apply_points(origin[None, :])[0]
            axis = poses[frame].apply_dirs(axis[None, :])[0]
        unit, _ok = normalize(axis[None, :], cfg.zero_vector_tol)
        axis = unit[0]
        ex = np.cross(axis, np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < GRADIENT_SEED_AXIS_COS else np.array([0.0, 1.0, 0.0]))
        ex, _ok = normalize(ex[None, :], cfg.zero_vector_tol)
        ex = ex[0]
        ey = np.cross(axis, ex)
        indices = faces[material_id]
        local: list[GradientFaceG] = []
        for index in indices:
            surf = scene.surfaces[index]
            where = f"surface:{surf.surface_id}"
            if surf.interaction.kind != "refract" or surf.geometry not in ("plane", "sag", "sphere"):
                refuse(f"surface {surf.surface_id} bounds the gradient-index medium {material_id} and is not a refracting plane or "
                       "conic face: the rod's end faces are traced as refracting planes (across the axis or slanted) or conic "
                       "surfaces of revolution about its axis", where, material=material_id, geometry=surf.geometry,
                       interaction=surf.interaction.kind)
            local.append(_gradient_face(surf, origin, axis, ex, ey, material.radius_mm, material_id, refuse))
        if len(indices) != 2:
            refuse(f"the gradient-index medium {material_id} is bounded by {len(indices)} surfaces: a rod is traced between exactly "
                   "two end faces", f"material:{material_id}", material=material_id,
                   faces=[scene.surfaces[i].surface_id for i in indices])
        z = [float(dot(_vertex(scene.surfaces[i]) - origin, axis)) for i in indices]
        lo, hi = (0, 1) if z[0] < z[1] else (1, 0)
        for k, toward in ((lo, 1.0), (hi, -1.0)):
            surf = scene.surfaces[indices[k]]
            facing = float(dot(surf.normal, axis)) * toward  # (the face's normal points into the rod: +1)
            inside = surf.medium_pos if facing > 0.0 else surf.medium_neg
            if inside != material_id or z[lo] == z[hi]:
                refuse(f"the gradient-index medium {material_id} does not lie between its faces {scene.surfaces[indices[lo]].surface_id} "
                       f"and {scene.surfaces[indices[hi]].surface_id}", f"material:{material_id}", material=material_id)
        if local[lo].z_max >= local[hi].z_min:
            refuse(f"the end faces {scene.surfaces[indices[lo]].surface_id} and {scene.surfaces[indices[hi]].surface_id} of the "
                   f"gradient-index medium {material_id} overlap along its axis within the rod's radius ({material.radius_mm:.6g} mm): "
                   "a rod is traced between faces that do not reach past each other", f"material:{material_id}", material=material_id,
                   suggested_action="lengthen the rod or reduce the slant or curvature of its end faces")
        across = all(f.kind == "plane" and f.normal[2] >= 1.0 - GRADIENT_FACE_TOL for f in local)
        out[material_id] = GradientG(material_id, origin, axis, ex, ey, material.radius_mm, indices[lo], indices[hi], z[lo], z[hi],
                                     None if across else (local[lo], local[hi]))
    return out


def _vertex(surf: SurfaceG) -> Array:
    """A face's point on its own axis: a plane's point, a sag surface's vertex, a sphere's cap vertex."""

    return surf.center + surf.radius * surf.cap_axis if surf.geometry == "sphere" else surf.origin


def _gradient_face(surf: SurfaceG, origin: Array, axis: Array, ex: Array, ey: Array, radius: float, material_id: str,
                   refuse: Any) -> GradientFaceG:
    """An end face of a gradient-index rod in the rod's frame (``GradientFaceG``): a plane at any slant, or a sphere / conic
    surface of revolution about the rod's axis (vertex on the axis, no aspheric, cylindrical, axicon or two-variable terms)
    whose profile is defined out to the rod's radius."""

    where = f"surface:{surf.surface_id}"

    def local(v: Array) -> tuple[float, float, float]:
        return float(dot(v, ex)), float(dot(v, ey)), float(dot(v, axis))

    vertex = local(_vertex(surf) - origin)
    if surf.geometry == "plane":
        m = local(surf.normal)
        m = (-m[0], -m[1], -m[2]) if m[2] < 0.0 else m
        if m[2] >= 1.0 - GRADIENT_FACE_TOL:  # (a plane across the axis, as the rod's other checks take it)
            return GradientFaceG("plane", vertex, (0.0, 0.0, 1.0), vertex[2], vertex[2], 0.0)
        if m[2] <= GRADIENT_FACE_TOL:
            refuse(f"surface {surf.surface_id} bounds the gradient-index medium {material_id} and contains its axis: an end face "
                   "crosses the axis", where, material=material_id)
        slope = math.sqrt(m[0] * m[0] + m[1] * m[1]) / m[2]
        z_axis = vertex[2] - (m[0] * (0.0 - vertex[0]) + m[1] * (0.0 - vertex[1])) / m[2]
        return GradientFaceG("plane", vertex, m, z_axis - radius * slope, z_axis + radius * slope, slope)
    along = float(dot(surf.normal, axis))
    off_axis = math.sqrt(vertex[0] * vertex[0] + vertex[1] * vertex[1])
    plain = surf.geometry == "sphere" or (not surf.cylindrical and surf.freeform is None and surf.facet is None
                                          and surf.linear == 0.0 and not surf.coefficients)
    if not plain or abs(along) < 1.0 - GRADIENT_FACE_TOL or off_axis > GRADIENT_VERTEX_TOL_MM:
        refuse(f"surface {surf.surface_id} bounds the gradient-index medium {material_id} and is not a sphere or conic surface of "
               "revolution about its axis: a curved end face is traced centred on the rod's axis, without aspheric, cylindrical, "
               "axicon or two-variable terms", where, material=material_id,
               suggested_action="centre the curved end face on the rod's axis, or use a plane end face")
    if surf.geometry == "sphere":
        centre = local(surf.center - origin)
        curvature, conic = 1.0 / (centre[2] - vertex[2]), 0.0
    else:
        curvature, conic = surf.curvature * (1.0 if along > 0.0 else -1.0), surf.conic
    s = radius * radius
    inside = 1.0 - (1.0 + conic) * curvature * curvature * s
    if inside <= 0.0:
        refuse(f"surface {surf.surface_id} bounds the gradient-index medium {material_id} and its profile ends inside the rod's "
               f"radius ({radius:.6g} mm)", where, material=material_id)
    root = math.sqrt(inside)
    sag = curvature * s / (1.0 + root)
    slope = abs(curvature) * radius / root
    return GradientFaceG("conic", (0.0, 0.0, vertex[2]), (0.0, 0.0, 1.0), vertex[2] + min(0.0, sag), vertex[2] + max(0.0, sag),
                         slope, curvature, conic)


def _crystals(scene: Scene, project: ProjectSnapshot, poses: dict[str, Pose], cfg: NumericalConfig,
              library: MaterialLibrary) -> dict[str, Array]:
    """The optic axis (scene frame) of every uniaxial crystal the scene uses, after checking it is used as the model traces it:
    every surface with a crystal on a side is a plane that refracts, and no source, emitter or observation plane is in one."""

    def refuse(message: str, location: str, **context: object) -> None:
        raise EngineInputError(ErrorCode.BIREFRINGENT_MEDIUM_UNSUPPORTED, message, location=location, **context)

    used: set[str] = set()
    for surf in scene.surfaces:
        if surf.interaction.kind == "observe":
            continue
        crystals = sorted(m for m in {surf.medium_pos, surf.medium_neg} if library.uniaxial(m) is not None)
        if not crystals:
            continue
        where = f"surface:{surf.surface_id}"
        if surf.geometry != "plane":
            refuse(f"surface {surf.surface_id} bounds the uniaxial crystal {crystals[0]} and is curved: crystal faces are traced as "
                   "planes only", where, material=crystals[0], geometry=surf.geometry,
                   suggested_action="give the crystal plane faces (a curved crystal surface is outside the model)")
        if surf.interaction.kind != "refract":
            refuse(f"surface {surf.surface_id} bounds the uniaxial crystal {crystals[0]} with a {surf.interaction.kind} interaction: a "
                   "crystal face is traced as a refracting interface only", where, material=crystals[0],
                   interaction=surf.interaction.kind,
                   suggested_action="model the element on the crystal's face as a separate element in the surrounding medium")
        used.update(crystals)
    for src in scene.sources:
        if library.uniaxial(src.medium) is not None:
            refuse(f"source {src.source_id} starts inside the uniaxial crystal {src.medium}", f"source:{src.source_id}",
                   material=src.medium, suggested_action="place the source outside the crystal")
    for emitter in scene.emitters:
        if library.uniaxial(emitter.medium) is not None:
            refuse(f"source {emitter.source_id} starts inside the uniaxial crystal {emitter.medium}", f"source:{emitter.source_id}",
                   material=emitter.medium, suggested_action="place the source outside the crystal")
    for obs in project.observations:
        if library.uniaxial(obs.medium) is not None:
            refuse(f"observation {obs.observation_id} lies inside the uniaxial crystal {obs.medium}: light is observed outside "
                   "crystals", f"observation:{obs.observation_id}", material=obs.medium,
                   suggested_action="place the observation plane outside the crystal")
    axes: dict[str, Array] = {}
    for material_id in sorted(used):
        laws = library.uniaxial(material_id)
        assert laws is not None
        frame = laws.material.optic_axis_component
        axis = np.array(laws.material.optic_axis, dtype=np.float64)
        if frame is not None:
            if frame not in poses:
                raise EngineInputError(ErrorCode.UNKNOWN_REFERENCE, f"the optic axis of {material_id} is given in the frame of the "
                                       f"unknown component {frame!r}", location=f"material:{material_id}")
            axis = matvec(poses[frame].rotation, axis)
        unit, ok = normalize(axis[None, :], cfg.zero_vector_tol)
        if not bool(ok[0]):  # pragma: no cover - the contract holds a unit vector
            raise EngineInputError(ErrorCode.ZERO_LENGTH_VECTOR, "zero optic axis", location=f"material:{material_id}")
        axes[material_id] = unit[0]
    return axes


def find_component(project: ProjectSnapshot, instance_id: str) -> EngineeringComponentSnapshot:
    for comp in project.components:
        if comp.instance_id == instance_id:
            return comp
    raise EngineInputError(ErrorCode.UNKNOWN_REFERENCE, f"unknown component {instance_id!r}")
