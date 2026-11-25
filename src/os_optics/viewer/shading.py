"""How a surface is lit: one material model for the PNG snapshots and the browser page.

The constants below are the only copy of the model's numbers. ``shading_document`` hands them to the page (the view
document's ``shading``), whose shader (``assets/viewer.js``, ``STUDIO_FRAGMENT``) evaluates the same equations for each
fragment; ``raster.py`` calls the NumPy functions here. A mesh's own numbers (metallic, roughness, transmission, index,
thickness) come from the Engine GLB's material (``optical_view.glb``). ``docs/VIEWER_MATERIALS.md`` writes the model out.

All directions are in the camera frame: x to the right of the screen, y up the screen, z towards the viewer. The studio
turns with the camera, so a lens is lit the same way in every view and a highlight slides over a curved surface as it turns.

* Environment: a floor and a sky meeting at a soft horizon, plus broad soft lights (``LIGHTS``) without edges, widened by a surface's roughness.
* Reflection: Fresnel-Schlick from the surface's index (``F0 = ((n - 1) / (n + 1)) ** 2``) or, for a metal, its colour,
  times the environment in the mirror direction, plus the GGX highlight of one key light.
* Glass: what lies behind shows through ``T = (1 - F) * exp(-sigma * path)``, ``path`` being the thickness along the ray
  refracted by Snell's law and ``sigma`` set so that the reference thickness absorbs the material's alpha; the absorbed
  part shows the glass's own colour. Behind the glass the image is not displaced: refraction is the tracer's, not the picture's.
* Output: colours are tone mapped (``1 - exp(-exposure * x)``), encoded to sRGB and blended premultiplied, in display values:
  ``out = src + dst * (1 - a)``. Glass covers what lies behind by what it shows as well as by what it absorbs, so a stack of
  surfaces saturates softly and tends to the glass's colour instead of adding up to white.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

OPTICAL_VIEW = "OPTICAL_SIMULATION_VIEW"
# Studio: broad soft lights, each (azimuth, elevation, half width in degrees, linear intensity) -- azimuth 0 faces the viewer's side of
# the scene, +90 the right of the screen, 180 behind the scene. A light falls off as a smooth lobe with no edge, ``exp((cos a - 1) / s)``
# with ``s = 1 - cos(half width)``, so its reflection rolls continuously over a dome instead of drawing a shape on it.
LIGHTS = (
    (-50.0, 24.0, 16.0, 2.4),  # the key: a soft highlight on a dome facing the viewer's left
    (-105.0, 16.0, 15.0, 4.5),  # far left and right: the same on a dome seen three-quarter on
    (100.0, 14.0, 15.0, 4.0),
    (180.0, 26.0, 22.0, 0.55),  # behind the scene: what a surface seen at a grazing angle reflects
    (180.0, -26.0, 22.0, 0.25),
)
ROUGHNESS_SPREAD = 1.2  # a surface's roughness squared widens a light's lobe by this much; its peak falls in proportion
# A dark floor and a bright sky with a soft horizon: whatever a dome faces, its upper part reflects sky and its lower part floor.
ENV_FLOOR = (0.02, 0.025, 0.03)
ENV_CEILING = (0.8, 0.88, 1.0)
HORIZON_SOFTNESS = 0.4  # the horizon's half width, in the sine of the elevation
BEHIND_DIM = 0.12  # the sky behind the scene, of the sky behind the viewer: a surface seen edge-on is a dark mirror, not a white one
GROUND_DIM = 0.7  # a rough surface's reflection falls by this much times its roughness: a ground edge stays quiet
KEY_LIGHT = (-0.45, 0.62, 0.64)  # towards the key light, camera frame (normalized where used)
KEY_INTENSITY = 1.5
MIN_ROUGHNESS = 0.14  # the GGX highlight of a mirror-smooth surface stays a soft spot
DEFAULT_IOR = 1.5  # glTF's default for a glass whose material is not known
REFERENCE_THICKNESS_MM = 4.0  # a glass this thick absorbs its material's alpha along a straight ray
DEFAULT_THICKNESS_MM = 1.5  # a surface without a known thickness (a coated plane, a thin plate)
COATED_FACE_TINT = 0.55  # of the material's alpha: a coated or modelled face shows its identity colour whatever its thickness
GLASS_ABSORPTION = 0.25  # of the material's alpha: the identity colour stays present without turning the glass opaque
BODY_AMBIENT = 0.35  # the light on the glass's own colour and on the diffuse part of a satin surface
BODY_DIFFUSE = 0.45
EXPOSURE = 1.35
MIN_COSINE = 1e-3
SRGB_KNEE = 0.0031308
SRGB_SLOPE = 12.92
SRGB_SCALE = 1.055
SRGB_OFFSET = 0.055
SRGB_EXPONENT = 1.0 / 2.4
FRESNEL_POWER = 5
#: The drawing's backdrop, top to bottom (sRGB): dark for the Optical Simulation View, where glass shows by what it reflects;
#: light for the assembly, whose parts show by their class colours.
OPTICAL_BACKDROP = ((0.19, 0.22, 0.265), (0.06, 0.07, 0.09))
ASSEMBLY_BACKDROP = ((0.965, 0.972, 0.98), (0.965, 0.972, 0.98))
#: The outline of a see-through optic against that backdrop.
OPTICAL_GLASS_EDGE = (0.74, 0.86, 0.98)
ASSEMBLY_GLASS_EDGE = (0.16, 0.36, 0.56)
OPTICAL_GLASS_EDGE_ALPHA = 0.5
FAINT_EDGE_SHARE = 0.3  # a cemented interface's lines, of that alpha: glass on both sides shows hardly at all
ASSEMBLY_GLASS_EDGE_ALPHA = 1.0
#: A beam colour with less contrast than this against the dark backdrop (a blue or a violet) gets a light casing under its line.
MIN_BEAM_CONTRAST = 3.0
BEAM_CASING_RGB = (0.86, 0.89, 0.93)
BEAM_CASING_ALPHA = 0.55
BEAM_CASING_EXTRA_PX = 2.0
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)
LUMINANCE_FLARE = 0.05  # the flare term of a contrast ratio (WCAG 2)


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a / float(np.linalg.norm(a))


def light_frames() -> list[dict[str, Any]]:
    """Each light as its centre direction, the lobe's softness ``s`` and intensity."""

    out = []
    for azimuth, elevation, half_width, intensity in LIGHTS:
        az, el = math.radians(azimuth), math.radians(elevation)
        out.append({"centre": np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)]),
                    "softness": 1.0 - math.cos(math.radians(half_width)), "intensity": intensity})
    return out


def environment(directions: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    """Linear radiance arriving from ``directions`` (n, 3 unit vectors, camera frame), blurred by ``roughness`` (n,)."""

    t = np.clip((directions[:, 1] + HORIZON_SOFTNESS) / (2.0 * HORIZON_SOFTNESS), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    colour = np.asarray(ENV_FLOOR)[None, :] * (1.0 - t)[:, None] + np.asarray(ENV_CEILING)[None, :] * t[:, None]
    colour = colour * (BEHIND_DIM + (1.0 - BEHIND_DIM) * (0.5 + 0.5 * directions[:, 2]))[:, None]
    total = np.zeros(len(directions))
    for light in light_frames():
        soft = light["softness"] + ROUGHNESS_SPREAD * roughness * roughness
        total += light["intensity"] * light["softness"] / soft * np.exp((directions @ light["centre"] - 1.0) / soft)
    return colour + total[:, None]


def fresnel_f0(ior: np.ndarray) -> np.ndarray:
    return ((ior - 1.0) / (ior + 1.0)) ** 2


def schlick(f0: np.ndarray, cosine: np.ndarray) -> np.ndarray:
    return f0 + (1.0 - f0) * (1.0 - cosine) ** FRESNEL_POWER


def _key_highlight(normals: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    """The key light's GGX specular term (distribution x correlated Smith visibility x n.l), without Fresnel; the viewer is +z."""

    light = _unit(KEY_LIGHT)
    half = _unit(light + np.array([0.0, 0.0, 1.0]))
    a = np.maximum(roughness, MIN_ROUGHNESS) ** 2
    a2 = a * a
    nh = np.clip(normals @ half, 0.0, 1.0)
    nl = np.clip(normals @ light, 0.0, 1.0)
    nv = np.clip(normals[:, 2], MIN_COSINE, 1.0)
    d = a2 / (math.pi * (nh * nh * (a2 - 1.0) + 1.0) ** 2)
    v = 0.5 / (nl * np.sqrt(nv * nv * (1.0 - a2) + a2) + nv * np.sqrt(nl * nl * (1.0 - a2) + a2) + MIN_COSINE)
    return KEY_INTENSITY * d * v * nl


def body_light(normals: np.ndarray) -> np.ndarray:
    return BODY_AMBIENT + BODY_DIFFUSE * np.abs(normals @ _unit(KEY_LIGHT))


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=float)
    return np.where(c <= SRGB_KNEE * SRGB_SLOPE, c / SRGB_SLOPE, ((c + SRGB_OFFSET) / SRGB_SCALE) ** (1.0 / SRGB_EXPONENT))


def encode(linear: np.ndarray) -> np.ndarray:
    """Tone map, then encode to sRGB."""

    c = 1.0 - np.exp(-EXPOSURE * np.maximum(linear, 0.0))
    return np.where(c <= SRGB_KNEE, c * SRGB_SLOPE, SRGB_SCALE * np.maximum(c, SRGB_KNEE) ** SRGB_EXPONENT - SRGB_OFFSET)


def shade(normals: np.ndarray, base_rgb, *, alpha: float, metallic: float, roughness: float, transmission: float,
          ior: float | None, thickness_mm: float | None, tint: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Premultiplied display colour and coverage ``(src (n, 3), a (n,))`` of a surface whose camera-frame normals are ``normals``
    (n, 3, either side): ``out = src + dst * (1 - a)``. ``tint`` > 0: what a coated face absorbs, in place of the glass's path."""

    linear = srgb_to_linear(np.asarray(base_rgb, dtype=float))
    n = normals * np.where(normals[:, 2] < 0.0, -1.0, 1.0)[:, None]  # the side facing the viewer
    cosine = np.clip(n[:, 2], MIN_COSINE, 1.0)
    index = ior if ior is not None else DEFAULT_IOR
    f0 = fresnel_f0(np.full(3, index)) * (1.0 - metallic) + linear * metallic
    fresnel = schlick(f0[None, :], cosine[:, None])
    mirror = 2.0 * cosine[:, None] * n - np.array([0.0, 0.0, 1.0])
    rough = np.full(len(n), roughness)
    specular = fresnel * (environment(mirror, rough) + _key_highlight(n, rough)[:, None]) * (1.0 - GROUND_DIM * roughness)
    light = body_light(n)[:, None]
    if transmission > 0.0:
        thickness = thickness_mm if thickness_mm is not None else DEFAULT_THICKNESS_MM
        sigma = -math.log(1.0 - alpha * GLASS_ABSORPTION) / REFERENCE_THICKNESS_MM
        path = thickness / np.sqrt(np.maximum(1.0 - (1.0 - cosine * cosine) / (index * index), MIN_COSINE))
        absorbed = np.full(len(n), tint) if tint > 0.0 else 1.0 - np.exp(-sigma * path)
        through = (1.0 - fresnel.mean(axis=1)) * (1.0 - absorbed)
        src = encode(specular) + absorbed[:, None] * encode(linear[None, :] * light)
        return src, 1.0 - through * (1.0 - np.minimum(src.max(axis=1), 1.0))  # what it shows also covers: a stack saturates softly
    diffuse = (1.0 - metallic) * linear[None, :] * light
    return encode(specular + diffuse) * alpha, np.full(len(n), alpha)


def smooth_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals of a mesh that states none (a part geometry of the assembly)."""

    normals = np.zeros_like(positions)
    a, b, c = positions[triangles[:, 0]], positions[triangles[:, 1]], positions[triangles[:, 2]]
    faces = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(normals, triangles[:, k], faces)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.where(length > 0.0, normals / np.maximum(length, MIN_COSINE), np.array([0.0, 0.0, 1.0]))


def mesh_normals(mesh: Any) -> np.ndarray:
    return mesh.normals if mesh.normals is not None else smooth_normals(mesh.positions, mesh.triangles)


def backdrop_colours(view_kind: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return OPTICAL_BACKDROP if view_kind == OPTICAL_VIEW else ASSEMBLY_BACKDROP


def backdrop(view_kind: str, height: int, width: int) -> np.ndarray:
    """The (height, width, 3) background image in display colours: a vertical gradient."""

    top, bottom = (np.asarray(c, dtype=float) for c in backdrop_colours(view_kind))
    t = ((np.arange(height) + 0.5) / height)[:, None, None]
    return np.broadcast_to(top * (1.0 - t) + bottom * t, (height, width, 3)).copy()


def luminance(rgb) -> float:
    return float(srgb_to_linear(np.asarray(rgb, dtype=float)) @ np.asarray(LUMINANCE_WEIGHTS))


def contrast(a, b) -> float:
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + LUMINANCE_FLARE) / (min(la, lb) + LUMINANCE_FLARE)


def beam_needs_casing(rgb, view_kind: str) -> bool:
    """Whether a beam colour needs a casing to stand out everywhere on the view's backdrop."""

    return view_kind == OPTICAL_VIEW and contrast(rgb, OPTICAL_BACKDROP[0]) < MIN_BEAM_CONTRAST


def glass_edge(view_kind: str) -> tuple[tuple[float, float, float], float]:
    if view_kind == OPTICAL_VIEW:
        return OPTICAL_GLASS_EDGE, OPTICAL_GLASS_EDGE_ALPHA
    return ASSEMBLY_GLASS_EDGE, ASSEMBLY_GLASS_EDGE_ALPHA


def shading_document(view_kind: str) -> dict[str, Any]:
    """The model's numbers for the page's shader (the view document's ``shading``)."""

    edge, edge_alpha = glass_edge(view_kind)
    top, bottom = backdrop_colours(view_kind)
    return {
        "lights": [{"centre": [float(c) for c in b["centre"]], "softness": b["softness"], "intensity": b["intensity"]} for b in light_frames()],
        "roughness_spread": ROUGHNESS_SPREAD,
        "horizon_softness": HORIZON_SOFTNESS, "behind_dim": BEHIND_DIM, "ground_dim": GROUND_DIM, "env_floor": list(ENV_FLOOR), "env_ceiling": list(ENV_CEILING),
        "key_light": [float(c) for c in _unit(KEY_LIGHT)], "key_intensity": KEY_INTENSITY, "min_roughness": MIN_ROUGHNESS,
        "default_ior": DEFAULT_IOR, "default_thickness_mm": DEFAULT_THICKNESS_MM, "reference_thickness_mm": REFERENCE_THICKNESS_MM,
        "glass_absorption": GLASS_ABSORPTION, "body_ambient": BODY_AMBIENT, "body_diffuse": BODY_DIFFUSE, "exposure": EXPOSURE,
        "beam_casing": {"rgb": list(BEAM_CASING_RGB), "alpha": BEAM_CASING_ALPHA, "extra_px": BEAM_CASING_EXTRA_PX},
        "backdrop": {"top": list(top), "bottom": list(bottom), "dark": view_kind == OPTICAL_VIEW}, "glass_edge": list(edge), "glass_edge_alpha": edge_alpha, "faint_edge_share": FAINT_EDGE_SHARE,
    }
