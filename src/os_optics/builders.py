"""Convenience factories that build *contract objects* (no physics here).

Element frame convention: local ``+z`` is the nominal optical axis (light travels along ``+z`` through transmissive
elements; for mirrors/splitters local ``+z`` is the front-face normal pointing toward the incident side), local ``+y``
is the ``up`` direction orthogonalized against the axis, local ``x = y × z``. Angles of polarizer / waveplate axes are
measured in the local ``x-y`` plane from local ``x`` toward local ``y``.
"""

from __future__ import annotations

import math

import numpy as np

from .contracts import (
    AbsorberInteraction,
    ApertureStopInteraction,
    BeamSplitterInteraction,
    CircularAperture,
    ConstantCoating,
    EngineeringComponentSnapshot,
    EngineeringEnvelopeSnapshot,
    JonesVector,
    MirrorInteraction,
    Observation,
    OpticalPort,
    PlaneSurface,
    Pose3D,
    PolarizerInteraction,
    RectangularAperture,
    RefractInteraction,
    RetarderInteraction,
    SourceSpec,
    SphereSurface,
    ThinLensInteraction,
)
from .config import DEFAULT_NUMERICAL_CONFIG
from .contracts.base import Vector3
from .contracts.components import ActuatorSpec

_Z = (0.0, 0.0, 1.0)

# Default opaque mount ring (mm) around the clear aperture of transmissive elements: rays hitting it are clipped.
DEFAULT_MOUNT_MARGIN_MM = 2.0
INCH_MM = 25.4
ONE_INCH_CLEAR_RADIUS_MM = 12.7  # clear radius of a 1-inch optic (half of 25.4 mm)
DEFAULT_LASER_POWER_W = 1e-3
_MOSTLY_PARALLEL = 0.9  # |axis.x| above which the x axis is a poor helper axis
# An iris / aperture plate extends well beyond its opening.
DEFAULT_STOP_PLATE_MARGIN_MM = 50.0  # an iris / aperture plate extends well beyond its opening


def _pose(position: Vector3, axis: Vector3, up: Vector3 = _Z) -> Pose3D:
    """Pose with local z = axis. If ``axis`` is parallel to ``up`` a horizontal reference is used instead."""
    a = np.array(axis, dtype=float)
    a = a / np.linalg.norm(a)
    u = np.array(up, dtype=float)
    if abs(float(a @ u / np.linalg.norm(u))) > 1.0 - DEFAULT_NUMERICAL_CONFIG.unit_vector_tol:
        u = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < _MOSTLY_PARALLEL else np.array([0.0, 1.0, 0.0])
    return Pose3D.from_axis_direction(position, tuple(float(x) for x in a), tuple(float(x) for x in u))  # type: ignore[arg-type]


def rect_envelope(width_mm: float, depth_mm: float, height_mm: float = INCH_MM, *, base_z_mm: float = 0.0, optical: bool = False,
                  **kwargs: object) -> EngineeringEnvelopeSnapshot:
    """Rectangular footprint ``width (envelope x) × depth (envelope y)`` centred on the element origin.

    For elements built with the other factories (local ``+z`` = beam axis) pass ``optical=True``: the envelope frame is then the
    plan frame with ``x`` along the beam axis and ``y`` transverse horizontal (see ``Pose3D.plan_frame_for_optical_axis``)."""
    hw, hd = width_mm / 2.0, depth_mm / 2.0
    return EngineeringEnvelopeSnapshot(
        frame_pose=Pose3D.plan_frame_for_optical_axis() if optical else Pose3D(),
        footprint_mm=((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)), z_range_mm=(base_z_mm, base_z_mm + height_mm), **kwargs  # type: ignore[arg-type]
    )


def laser(
    instance_id: str,
    *,
    wavelength_nm: float | None = None,
    spectrum: tuple[tuple[float, float], ...] = (),
    power_W: float = DEFAULT_LASER_POWER_W,
    waist_radius_mm: float = 0.5,
    waist_radius_y_mm: float | None = None,
    position: Vector3 = (0.0, 0.0, 0.0),
    direction: Vector3 = (1.0, 0.0, 0.0),
    up: Vector3 = _Z,
    m2: float = 1.0,
    m2_y: float | None = None,
    waist_position_z_mm: float = 0.0,
    polarization: JonesVector | None = None,
    coherence_length_mm: float | None = None,
    linewidth_nm: float | None = None,
    coherence_group: str | None = None,
    medium: str = "air",
    envelope: EngineeringEnvelopeSnapshot | None = None,
    source_id: str | None = None,
) -> EngineeringComponentSnapshot:
    from .contracts import SpectralLine

    src = SourceSpec(
        source_id=source_id or instance_id,
        wavelength_nm=wavelength_nm,
        spectrum=tuple(SpectralLine(wavelength_nm=w, weight=wt) for w, wt in spectrum),
        power_W=power_W,
        waist_radius_x_mm=waist_radius_mm,
        waist_radius_y_mm=waist_radius_y_mm if waist_radius_y_mm is not None else waist_radius_mm,
        waist_position_z_mm=waist_position_z_mm,
        m2_x=m2,
        m2_y=m2_y if m2_y is not None else m2,
        polarization=polarization or JonesVector.linear(0.0),
        coherence_length_mm=coherence_length_mm,
        linewidth_nm=linewidth_nm,
        coherence_group=coherence_group,
        medium=medium,
    )
    return EngineeringComponentSnapshot(
        instance_id=instance_id, role="source", pose=_pose(position, direction, up), source=src, envelope=envelope,
        ports=(OpticalPort(port_id="out", position_mm=(0.0, 0.0, 0.0), axis=_Z, kind="output"),),
    )


def thin_lens(
    instance_id: str, *, focal_length_mm: float, clear_radius_mm: float, position: Vector3, axis: Vector3 = (1.0, 0.0, 0.0),
    up: Vector3 = _Z, transmittance: float = 1.0, envelope: EngineeringEnvelopeSnapshot | None = None,
    actuators: tuple[ActuatorSpec, ...] = (), medium: str = "air",
) -> EngineeringComponentSnapshot:
    surf = PlaneSurface(
        surface_id="lens", origin_mm=(0.0, 0.0, 0.0), normal=_Z, aperture=CircularAperture(radius_mm=clear_radius_mm),
        mount_margin_mm=DEFAULT_MOUNT_MARGIN_MM, medium_positive=medium, medium_negative=medium,
        interaction=ThinLensInteraction(focal_length_mm=focal_length_mm, transmittance=transmittance),
    )
    return EngineeringComponentSnapshot(instance_id=instance_id, role="lens", pose=_pose(position, axis, up), surfaces=(surf,),
                                        envelope=envelope, actuators=actuators)


def mirror(
    instance_id: str, *, position: Vector3, normal: Vector3, clear_radius_mm: float | None = ONE_INCH_CLEAR_RADIUS_MM,
    half_width_mm: float | None = None, half_height_mm: float | None = None, up: Vector3 = _Z,
    reflectance: float | None = None, envelope: EngineeringEnvelopeSnapshot | None = None,
    actuators: tuple[ActuatorSpec, ...] = (), mount_margin_mm: float = 0.0, medium: str = "air",
) -> EngineeringComponentSnapshot:
    """Flat mirror; ``normal`` points toward the incident (reflective) side."""
    aperture = (
        RectangularAperture(half_width_mm=half_width_mm, half_height_mm=half_height_mm)
        if half_width_mm is not None and half_height_mm is not None
        else (CircularAperture(radius_mm=clear_radius_mm) if clear_radius_mm else None)
    )
    coating = None if reflectance is None else ConstantCoating(reflectance_s=reflectance, reflectance_p=reflectance,
                                                                transmittance_s=0.0, transmittance_p=0.0)
    surf = PlaneSurface(surface_id="face", origin_mm=(0.0, 0.0, 0.0), normal=_Z, aperture=aperture,
                        mount_margin_mm=mount_margin_mm,
                        medium_positive=medium, medium_negative=medium,
                        interaction=MirrorInteraction(reflective_side="positive", coating=coating))
    return EngineeringComponentSnapshot(instance_id=instance_id, role="mirror", pose=_pose(position, normal, up), surfaces=(surf,),
                                        envelope=envelope, actuators=actuators)


def mirror_for_reflection(instance_id: str, *, position: Vector3, d_in: Vector3, d_out: Vector3, **kwargs: object) -> EngineeringComponentSnapshot:
    """Flat mirror oriented so that ``d_in`` reflects into ``d_out`` (normal ∝ d_out - d_in, toward the incident side)."""
    a, b = np.array(d_in, dtype=float), np.array(d_out, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    n = b - a
    if np.linalg.norm(n) < DEFAULT_NUMERICAL_CONFIG.zero_vector_tol:
        raise ValueError("d_in and d_out are identical: no mirror can produce that")
    n = n / np.linalg.norm(n)
    return mirror(instance_id, position=position, normal=tuple(float(x) for x in n), **kwargs)  # type: ignore[arg-type]


def spherical_mirror(
    instance_id: str, *, position: Vector3, normal: Vector3, radius_of_curvature_mm: float, clear_radius_mm: float,
    up: Vector3 = _Z, envelope: EngineeringEnvelopeSnapshot | None = None, medium: str = "air",
) -> EngineeringComponentSnapshot:
    """Spherical mirror; ``radius_of_curvature_mm > 0`` is concave (centre of curvature on the incident side)."""
    r = abs(radius_of_curvature_mm)
    concave = radius_of_curvature_mm > 0
    center = (0.0, 0.0, r if concave else -r)
    cap = (0.0, 0.0, -1.0 if concave else 1.0)
    surf = SphereSurface(
        surface_id="face", center_mm=center, radius_mm=r, cap_axis=cap, aperture_radius_mm=clear_radius_mm,
        medium_positive=medium, medium_negative=medium,
        interaction=MirrorInteraction(reflective_side="negative" if concave else "positive"),
    )
    return EngineeringComponentSnapshot(instance_id=instance_id, role="mirror", pose=_pose(position, normal, up), surfaces=(surf,),
                                        envelope=envelope)


def singlet_lens(
    instance_id: str, *, r1_mm: float, r2_mm: float, thickness_mm: float, clear_radius_mm: float, material: str,
    position: Vector3, axis: Vector3 = (1.0, 0.0, 0.0), up: Vector3 = _Z, surround: str = "air",
    envelope: EngineeringEnvelopeSnapshot | None = None,
) -> EngineeringComponentSnapshot:
    """Thick spherical singlet (exact refraction). ``r1``/``r2`` follow the usual sign convention: positive radius = centre
    on the ``+z`` (exit) side; ``math.inf`` = flat. The element origin is the lens mid-plane; light travels along ``+z``."""
    half = thickness_mm / 2.0

    def surface(name: str, vertex_z: float, radius: float, source_side: str, exit_side: str) -> PlaneSurface | SphereSurface:
        if math.isinf(radius):
            return PlaneSurface(surface_id=name, origin_mm=(0.0, 0.0, vertex_z), normal=(0.0, 0.0, -1.0),
                                aperture=CircularAperture(radius_mm=clear_radius_mm), mount_margin_mm=DEFAULT_MOUNT_MARGIN_MM,
                                medium_positive=source_side, medium_negative=exit_side, interaction=RefractInteraction())
        r = abs(radius)
        center_z = vertex_z + radius
        cap_z = -1.0 if radius > 0 else 1.0  # from the centre toward the vertex
        centre_on_exit_side = radius > 0
        # sphere interior = the side that contains the centre
        inside, outside = (exit_side, source_side) if centre_on_exit_side else (source_side, exit_side)
        return SphereSurface(surface_id=name, center_mm=(0.0, 0.0, center_z), radius_mm=r, cap_axis=(0.0, 0.0, cap_z),
                             aperture_radius_mm=min(clear_radius_mm, r), mount_margin_mm=DEFAULT_MOUNT_MARGIN_MM,
                             medium_positive=outside, medium_negative=inside,
                             interaction=RefractInteraction())

    surfaces = (
        surface("front", -half, r1_mm, surround, material),
        surface("back", half, r2_mm, material, surround),
    )
    return EngineeringComponentSnapshot(instance_id=instance_id, role="lens", pose=_pose(position, axis, up), surfaces=surfaces,
                                        envelope=envelope)


def window(
    instance_id: str, *, thickness_mm: float, clear_radius_mm: float, material: str, position: Vector3,
    axis: Vector3 = (1.0, 0.0, 0.0), up: Vector3 = _Z, surround: str = "air", trace_fresnel_reflection: bool = False,
    envelope: EngineeringEnvelopeSnapshot | None = None,
) -> EngineeringComponentSnapshot:
    """Plane-parallel plate (two refracting planes); light travels along ``axis`` at the plate's mid-plane."""
    half = thickness_mm / 2.0
    ap = CircularAperture(radius_mm=clear_radius_mm)
    front = PlaneSurface(surface_id="front", origin_mm=(0.0, 0.0, -half), normal=(0.0, 0.0, -1.0), aperture=ap,
                         mount_margin_mm=DEFAULT_MOUNT_MARGIN_MM, medium_positive=surround, medium_negative=material,
                         interaction=RefractInteraction(trace_fresnel_reflection=trace_fresnel_reflection))
    back = PlaneSurface(surface_id="back", origin_mm=(0.0, 0.0, half), normal=(0.0, 0.0, 1.0), aperture=ap,
                        mount_margin_mm=DEFAULT_MOUNT_MARGIN_MM, medium_positive=surround, medium_negative=material,
                        interaction=RefractInteraction(trace_fresnel_reflection=trace_fresnel_reflection))
    return EngineeringComponentSnapshot(instance_id=instance_id, role="window", pose=_pose(position, axis, up),
                                        surfaces=(front, back), envelope=envelope)


def beam_splitter(
    instance_id: str, *, position: Vector3, normal: Vector3, coating: ConstantCoating, clear_radius_mm: float = ONE_INCH_CLEAR_RADIUS_MM,
    reflection_phase_rad: float = math.pi / 2, up: Vector3 = _Z, envelope: EngineeringEnvelopeSnapshot | None = None,
    medium: str = "air",
) -> EngineeringComponentSnapshot:
    surf = PlaneSurface(surface_id="coating", origin_mm=(0.0, 0.0, 0.0), normal=_Z,
                        aperture=CircularAperture(radius_mm=clear_radius_mm),
                        medium_positive=medium, medium_negative=medium,
                        interaction=BeamSplitterInteraction(coating=coating, reflection_phase_rad=reflection_phase_rad))
    return EngineeringComponentSnapshot(instance_id=instance_id, role="splitter", pose=_pose(position, normal, up), surfaces=(surf,),
                                        envelope=envelope)


def npbs(instance_id: str, *, reflectance: float = 0.5, **kwargs: object) -> EngineeringComponentSnapshot:
    """Non-polarizing (lossless) splitter with power reflectance ``reflectance``."""
    coating = ConstantCoating(reflectance_s=reflectance, reflectance_p=reflectance,
                              transmittance_s=1.0 - reflectance, transmittance_p=1.0 - reflectance)
    return beam_splitter(instance_id, coating=coating, **kwargs)  # type: ignore[arg-type]


def pbs(instance_id: str, *, extinction_reflect_p: float = 0.0, extinction_transmit_s: float = 0.0, **kwargs: object) -> EngineeringComponentSnapshot:
    """Polarizing splitter: p transmitted, s reflected; leakage fractions ``R_p`` and ``T_s`` (default ideal 0)."""
    coating = ConstantCoating(reflectance_s=1.0 - extinction_transmit_s, reflectance_p=extinction_reflect_p,
                              transmittance_s=extinction_transmit_s, transmittance_p=1.0 - extinction_reflect_p)
    return beam_splitter(instance_id, coating=coating, **kwargs)  # type: ignore[arg-type]


def polarizer(instance_id: str, *, position: Vector3, axis: Vector3, angle_deg: float, clear_radius_mm: float = ONE_INCH_CLEAR_RADIUS_MM,
              up: Vector3 = _Z, t_par: float = 1.0, t_perp: float = 0.0, envelope: EngineeringEnvelopeSnapshot | None = None,
              medium: str = "air") -> EngineeringComponentSnapshot:
    a = math.radians(angle_deg)
    surf = PlaneSurface(surface_id="plate", origin_mm=(0.0, 0.0, 0.0), normal=_Z, aperture=CircularAperture(radius_mm=clear_radius_mm),
                        mount_margin_mm=DEFAULT_MOUNT_MARGIN_MM, medium_positive=medium, medium_negative=medium,
                        interaction=PolarizerInteraction(axis_local=(math.cos(a), math.sin(a), 0.0),
                                                         transmittance_parallel=t_par, transmittance_perpendicular=t_perp))
    return EngineeringComponentSnapshot(instance_id=instance_id, role="polarizer", pose=_pose(position, axis, up), surfaces=(surf,),
                                        envelope=envelope)


def waveplate(instance_id: str, *, position: Vector3, axis: Vector3, fast_axis_angle_deg: float, retardance_rad: float,
              design_wavelength_nm: float, clear_radius_mm: float = ONE_INCH_CLEAR_RADIUS_MM, up: Vector3 = _Z, achromatic: bool = False,
              envelope: EngineeringEnvelopeSnapshot | None = None, medium: str = "air") -> EngineeringComponentSnapshot:
    a = math.radians(fast_axis_angle_deg)
    surf = PlaneSurface(surface_id="plate", origin_mm=(0.0, 0.0, 0.0), normal=_Z, aperture=CircularAperture(radius_mm=clear_radius_mm),
                        mount_margin_mm=DEFAULT_MOUNT_MARGIN_MM, medium_positive=medium, medium_negative=medium,
                        interaction=RetarderInteraction(fast_axis_local=(math.cos(a), math.sin(a), 0.0), retardance_rad=retardance_rad,
                                                        design_wavelength_nm=design_wavelength_nm, achromatic=achromatic))
    return EngineeringComponentSnapshot(instance_id=instance_id, role="waveplate", pose=_pose(position, axis, up), surfaces=(surf,),
                                        envelope=envelope)


def half_waveplate(instance_id: str, **kwargs: object) -> EngineeringComponentSnapshot:
    return waveplate(instance_id, retardance_rad=math.pi, **kwargs)  # type: ignore[arg-type]


def quarter_waveplate(instance_id: str, **kwargs: object) -> EngineeringComponentSnapshot:
    return waveplate(instance_id, retardance_rad=math.pi / 2.0, **kwargs)  # type: ignore[arg-type]


def aperture_stop(instance_id: str, *, position: Vector3, axis: Vector3, radius_mm: float, up: Vector3 = _Z,
                  envelope: EngineeringEnvelopeSnapshot | None = None, medium: str = "air") -> EngineeringComponentSnapshot:
    surf = PlaneSurface(surface_id="stop", origin_mm=(0.0, 0.0, 0.0), normal=_Z, aperture=CircularAperture(radius_mm=radius_mm),
                        mount_margin_mm=DEFAULT_STOP_PLATE_MARGIN_MM, medium_positive=medium, medium_negative=medium, interaction=ApertureStopInteraction())
    return EngineeringComponentSnapshot(instance_id=instance_id, role="aperture", pose=_pose(position, axis, up), surfaces=(surf,),
                                        envelope=envelope)


def beam_dump(instance_id: str, *, position: Vector3, axis: Vector3, radius_mm: float = ONE_INCH_CLEAR_RADIUS_MM, up: Vector3 = _Z,
              medium: str = "air") -> EngineeringComponentSnapshot:
    surf = PlaneSurface(surface_id="dump", origin_mm=(0.0, 0.0, 0.0), normal=_Z, aperture=CircularAperture(radius_mm=radius_mm),
                        medium_positive=medium, medium_negative=medium, interaction=AbsorberInteraction())
    return EngineeringComponentSnapshot(instance_id=instance_id, role="dump", pose=_pose(position, axis, up), surfaces=(surf,))


def observation_plane(observation_id: str, *, position: Vector3, normal: Vector3, radius_mm: float | None = None,
                      half_width_mm: float | None = None, half_height_mm: float | None = None, up: Vector3 = _Z,
                      acceptance_half_angle_deg: float | None = None, absorbing: bool = False, medium: str = "air",
                      device_id: str | None = None) -> Observation:
    aperture = None
    if radius_mm is not None:
        aperture = CircularAperture(radius_mm=radius_mm)
    elif half_width_mm is not None and half_height_mm is not None:
        aperture = RectangularAperture(half_width_mm=half_width_mm, half_height_mm=half_height_mm)
    return Observation(observation_id=observation_id, device_id=device_id, pose=_pose(position, normal, up), aperture=aperture,
                       acceptance_half_angle_deg=acceptance_half_angle_deg, absorbing=absorbing, medium=medium)


def unit(v: Vector3) -> Vector3:
    a = np.array(v, dtype=float)
    a = a / np.linalg.norm(a)
    return (float(a[0]), float(a[1]), float(a[2]))
