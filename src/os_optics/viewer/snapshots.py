"""Headless snapshots: PNG images of the standard views, or of cameras stated by angles (``Shot``), plus
``scene_summary.json`` (what each image shows, its size, scale and the camera pose it was drawn from) and ``legend.md``."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import PRESET_ORDER, PartsFrame, ViewerScene, camera_from_angles, scene_summary
from .png import png_bytes
from .raster import render_view, view_title

DEFAULT_VIEWS = ("iso", "top", "front", "side")
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 960
MIN_SIZE_PX = 240
MAX_SIZE_PX = 4096
SUMMARY_FILE = "scene_summary.json"
LEGEND_FILE = "legend.md"
CHANNEL_MAX = 255
#: A shot's ``frame`` that takes the framing of the whole set.
SET_FRAME = "set"
#: A shot name is a file stem: letters, digits, ``_``, ``-`` and ``.``, not starting with ``.``.
SHOT_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}")


@dataclass(frozen=True)
class Shot:
    """One image of a snapshot set, written as ``<name>.png``: a camera preset ``view``, or a camera stated by angles
    (``model.camera_from_angles``: ``angle_to_axis_deg``, ``elevation_deg``, ``roll_deg``). ``distance``: the eye distance
    recorded for a perspective camera (mm, or a string ``k`` + ``x``: k times the size of what is framed). ``frame``
    (``model.frame_points``): what this image fits, else the set's framing."""

    name: str
    view: str | None = None
    angle_to_axis_deg: float | None = None
    elevation_deg: float = 0.0
    roll_deg: float = 0.0
    distance: float | str | None = None
    frame: Any = SET_FRAME

    def pose(self, scene: ViewerScene) -> str | dict[str, Any]:
        """What ``raster.render_view`` draws: the preset's name, or the explicit pose."""

        if self.view is not None:
            return self.view
        direction, up = camera_from_angles(scene, self.angle_to_axis_deg, self.elevation_deg, self.roll_deg)  # type: ignore[arg-type]
        roll = f", roll {self.roll_deg:g} deg" if self.roll_deg else ""
        return {"name": self.name, "view_direction": direction, "up": up, "distance": self.distance,
                "title": f"CAMERA {self.name}: {self.angle_to_axis_deg:g} deg to the axis, elevation {self.elevation_deg:g} deg{roll}"}


def _check_shots(shots: Sequence[Shot]) -> None:
    names = [s.name for s in shots]
    bad = [n for n in names if not SHOT_NAME.fullmatch(n)]
    if bad:
        raise ValueError(f"image name(s) {bad} are not file names: use letters, digits, '_', '-' and '.'")
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f"image name(s) {twice} are given more than once")
    unknown = [s.view for s in shots if s.view is not None and s.view not in PRESET_ORDER]
    if unknown:
        raise ValueError(f"unknown view(s) {unknown}; choose from {list(PRESET_ORDER)}")
    for shot in shots:
        if shot.view is None and shot.angle_to_axis_deg is None:
            raise ValueError(f"image {shot.name} needs a view or an angle to the axis")


def snapshot_images(scene: ViewerScene, views=DEFAULT_VIEWS, *, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                    labels: bool = True, frame=None, shots: Sequence[Shot] = (), background=None,
                    size_mode: str = "fit") -> dict[str, tuple[bytes, dict[str, Any]]]:
    """``{name: (png bytes, what was drawn)}`` in memory, one per preset of ``views`` (named after it) and per ``shots``;
    nothing is written. ``frame``: ``model.frame_points``; ``background`` / ``size_mode``: ``raster.render_view``."""

    unknown = [v for v in views if v not in PRESET_ORDER]
    if unknown:
        raise ValueError(f"unknown view(s) {unknown}; choose from {list(PRESET_ORDER)}")
    if not (MIN_SIZE_PX <= width <= MAX_SIZE_PX and MIN_SIZE_PX <= height <= MAX_SIZE_PX):
        raise ValueError(f"image size must be {MIN_SIZE_PX}..{MAX_SIZE_PX} pixels per side")
    every = [Shot(v, view=v) for v in views] + list(shots)
    _check_shots(every)
    out = {}
    for shot in every:
        shot_frame = frame if isinstance(shot.frame, str) and shot.frame == SET_FRAME else shot.frame
        view = shot.pose(scene)
        image, drawn = render_view(scene, view, width, height, labels=labels, frame=shot_frame, background=background, size_mode=size_mode)
        drawn = {"view": shot.view or "camera", "title": view_title(scene, view), "frame": frame_name(shot_frame), **drawn}
        out[shot.name] = (png_bytes(image), drawn)
    return out


def frame_name(frame) -> str | list[float]:
    """How a framing is recorded: ``default``, ``optics``, ``scene``, a box ``[x0, y0, z0, x1, y1, z1]`` or ``element:<ids>``."""

    if frame is None:
        return "default"
    if isinstance(frame, PartsFrame):
        return frame.describe()
    if isinstance(frame, str):
        return frame
    return [float(v) for v in frame]


def _legend_markdown(scene: ViewerScene, summary: dict[str, Any], files: dict[str, dict[str, Any]]) -> str:
    lines = [f"# {scene.title}: snapshots", "",
             f"Frame `{scene.frame}`, units mm, +Z up, right-handed. Orthographic images of the exact geometry (nothing moved or "
             "spaced). Part ids are drawn next to each part; the legend box in each image names the colour classes.", ""]
    lines += ["| Image | View | Scale bar | Pixels per mm |", "|---|---|---|---|"]
    for name, info in files.items():
        lines.append(f"| `{name}` | {info['title']} | {info['scale_bar_mm']:g} mm | {info['pixels_per_mm']:.3g} |")
    lines += ["", "## Colour classes", ""]
    for cls in summary["classes"]:
        rgb = "#" + "".join(f"{round(c * CHANNEL_MAX):02x}" for c in cls["rgb"])
        lines.append(f"- `{cls['key']}` {cls['label']} ({rgb}): {cls['count']}")
    lines += ["", "## Parts", ""]
    for key, parts in summary["parts_by_class"].items():
        for p in parts:
            size = " x ".join(f"{v:g}" for v in p["bounds"]["size_mm"])
            lines.append(f"- `{p['id']}` [{key}] {p['label']}{' - ' + p['sku'] if p.get('sku') else ''}: bounding box {size} mm, "
                         f"centre {p['center_mm']} mm")
    for m in summary["markers"]:
        lines.append(f"- `{m['id']}` [{m['class']}] {m['label']}: at {m['position_mm']} mm")
    if summary["beams"]:
        lines += ["", "## Beams", ""]
        for b in summary["beams"]:
            # an Engine beam path states its display role (where it ends: its footprints and display extension); a
            # design axis of the assembly view has none, so its drawn runs name it
            role = b.get("display_role")
            roles = [role] if role else sorted({r["role"] for r in b["runs"] if r["role"] in ("PRIMARY", "SECONDARY", "DESIGN_AXIS")})
            wl = f"{b['wavelength_nm']:g} nm" if b.get("wavelength_nm") else "-"
            lines.append(f"- `{b['id']}` {wl}: {', '.join(roles)}")
    if scene.notes:
        lines += ["", "## Notes", ""] + [f"- {n['text']}" for n in scene.notes]
    return "\n".join(lines) + "\n"


def write_snapshots(scene: ViewerScene, out_dir: str | Path, views=DEFAULT_VIEWS, *, width: int = DEFAULT_WIDTH,
                    height: int = DEFAULT_HEIGHT, labels: bool = True, frame=None, shots: Sequence[Shot] = (), background=None,
                    size_mode: str = "fit") -> dict[str, Path]:
    """Write ``<name>.png`` for each view and shot, ``scene_summary.json`` (``snapshots``: per image its view, framing,
    size, scale and camera pose) and ``legend.md`` into ``out_dir``; returns the paths."""

    images = snapshot_images(scene, views, width=width, height=height, labels=labels, frame=frame, shots=shots, background=background,
                             size_mode=size_mode)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary = scene_summary(scene)
    written: dict[str, Path] = {}
    files: dict[str, dict[str, Any]] = {}
    for stem, (data, drawn) in images.items():
        name = f"{stem}.png"
        (directory / name).write_bytes(data)
        written[stem] = directory / name
        files[name] = {"projection": "orthographic", **drawn}
    summary["snapshots"] = files
    (directory / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / LEGEND_FILE).write_text(_legend_markdown(scene, summary, files), encoding="utf-8")
    written["summary"] = directory / SUMMARY_FILE
    written["legend"] = directory / LEGEND_FILE
    return written
