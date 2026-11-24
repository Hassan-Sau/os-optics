"""Local-simulation coverage probe: does the real local runtime simulate every public optical product in scope?

    python scripts/optical_simulation_probe.py --catalog <public_optical_catalog.v2.json> --out <simulation_probe.v1.json>

For every product of the given public catalog release the probe builds a one-element
design and runs ``os_optics.local.run_optical_project`` -- the same lowering, single Engine run, validity check and
Optical Simulation View a user gets:

* one Gaussian source (1 mW, waist radius ``min(0.5 mm, clear radius / 5)`` placed at the element, linear
  polarization) per band of the element's first coating that lies inside the model band and inside a band of every
  other coating of the element -- at the band's centre (a single characterized wavelength exactly) -- 100 mm before the
  element on a beam along design +X; so a filter is run in its pass and its blocking bands, a dichroic in both; a
  coating tabulated as a spectrum (a typical curve) is run at its node of highest transmittance and, when another, its
  node of highest reflectance inside the model band;
* a transmissive product on the beam axis (its local +Z along +X); a face-normal product (mirror, splitter, dichroic)
  turned so the chief ray meets its coated face at 45 deg, or at the middle of its coating's angle range if 45 deg is
  outside it; a polarization element at normal incidence; a product placed by its ports (cube, prism, off-axis
  parabola: ``frame.ports``) turned and shifted so that its ``IN`` port lies at the design origin with its direction
  along +X (local +Y kept as close to design +Z as the ``IN`` direction allows), every output port's beam leaving
  from that port's point along its direction -- the same placement for every ported form, read from the frame alone;
* a non-absorbing observation plane 20 mm after the element on every beam the element sends on (reflected and / or
  transmitted), with a circular active area of 20 mm radius centred on that beam: a beam that crosses the plane outside
  its active area is not counted there (a wedged plate deviates its transmitted beam by a fraction of a degree toward
  the reflected beam's side, so the unbounded plane of the reflected beam would catch the transmitted beam metres away
  at grazing incidence). A beam sent back toward the source side gets its plane moved out along that beam until the incoming
  beam can no longer cross it: a normal-incidence mirror (coating specified at 0 deg only) reflects the beam straight
  back onto the source, so its plane lies behind the source. A plane on the incoming beam would be crossed twice by
  one beam path (the view draws both crossings, ``BeamFootprintV1.crossing_index``); the probe keeps one crossing per
  plane so that each plane measures exactly the beam the element sends on.

``IN_SCOPE``: no model-scope issue, every source's chief ray met the element and reached an observation plane or was
blocked there by the element's own published data (a filter's blocking band; otherwise ``NOT_EXERCISED``). The probe
checks the *geometric* route independently of bulk-transmission knowledge: a release material with unstated loss still
has a traced ray but no known delivered power. In that case the run may be UNKNOWN, and the record explicitly states
``power_evidence: UNKNOWN_BULK_LOSS`` and ``codes: [BULK_LOSS_NOT_ESTABLISHED]``. This is not a 100% transmission claim.
Other unknown statuses remain UNKNOWN. An axicon is ``IN_SCOPE`` with run status ``OUT_OF_MODEL_SCOPE``
and ``documented_scope`` ``["GAUSSIAN_UNDEFINED_AFTER_AXICON"]`` when that is its run's only issue (raised at its own
surfaces) and the same ray and power conditions hold: the probe's sources are Gaussian beams, and a Gaussian beam behind
a cone is outside the Gaussian model by definition (rays, powers and polarization are traced). Anything else is recorded with
the codes that explain it (local-run error codes, Engine issue codes, model-scope codes). Results are checkpointed to
``--out`` every 100 products and a rerun with the same catalog digest and Engine build resumes where it stopped.

The probe file is release QA data of one catalog release (it records the catalog's SHA-256 and the Engine build that
probed it); it is not bound to any Engine artifact, so an Engine release does not require re-probing an unchanged catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from os_optics.local import LocalRunError, load_public_catalog, run_optical_project
from os_optics.manifest import engine_build_id

SOURCE_DISTANCE_MM = 100.0
OBSERVATION_DISTANCE_MM = 20.0
OBSERVATION_RADIUS_MM = 20.0  # the active area of every probe plane, centred on the beam it measures
PREFERRED_AOI_DEG = 45.0
IN_SCOPE_STATUSES = ("PASS", "WARNING")
RETURN_TOLERANCE = 1e-9  # a 45 deg fold sends the beam along +-Y (its x component is rounding noise, not a return)
AXICON_GAUSSIAN_SCOPE = "GAUSSIAN_UNDEFINED_AFTER_AXICON"  # the documented model scope of a Gaussian beam behind a cone


def _quat_from_frame(x: tuple[float, float, float], y: tuple[float, float, float], z: tuple[float, float, float]) -> list[float]:
    """Unit quaternion (w, x, y, z) of the rotation whose columns are the local axes x, y, z (right-handed, orthonormal)."""

    m00, m10, m20 = x
    m01, m11, m21 = y
    m02, m12, m22 = z
    trace = m00 + m11 + m22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    norm = math.sqrt(sum(c * c for c in q))
    return [c / norm for c in q]


def _frame_along_any(direction: tuple[float, float, float]) -> list[float]:
    """Pose quaternion whose local +Z is ``direction``: ``_frame_along`` for a horizontal direction; otherwise local +Y is
    design +Z made perpendicular to it (design +Y for a vertical direction)."""

    if abs(direction[2]) <= RETURN_TOLERANCE:
        return _frame_along(direction)
    up = (0.0, 0.0, 1.0) if abs(direction[2]) < 0.9 else (0.0, 1.0, 0.0)
    dot = sum(u * d for u, d in zip(up, direction))
    y = tuple(u - dot * d for u, d in zip(up, direction))
    norm = math.sqrt(sum(c * c for c in y))
    y = tuple(c / norm for c in y)
    dx, dy, dz = direction
    x = (y[1] * dz - y[2] * dy, y[2] * dx - y[0] * dz, y[0] * dy - y[1] * dx)
    return _quat_from_frame(x, y, direction)  # type: ignore[arg-type]


def _frame_along(direction: tuple[float, float, float]) -> list[float]:
    """Pose quaternion whose local +Z is ``direction`` (horizontal) and local +Y is design +Z (up)."""

    dx, dy, dz = direction
    y = (0.0, 0.0, 1.0)
    x = (y[1] * dz - y[2] * dy, y[2] * dx - y[0] * dz, y[0] * dy - y[1] * dx)
    return _quat_from_frame(x, y, direction)


def _ported_pose(product) -> tuple[list[float], list[float], list[tuple[tuple[float, float, float], tuple[float, float, float]]]]:
    """(quaternion, position, [(point, direction) of every output port in design coordinates]) of a product placed by its
    ports: the ``IN`` port at the origin, its direction along design +X, the local axis closest to perpendicular to it
    (local +Y, else +X) projected to design +Z."""

    def vec(values) -> tuple[float, float, float]:
        return tuple(float(Decimal(v)) for v in values)  # type: ignore[return-value]

    def unit(v):
        n = math.sqrt(sum(c * c for c in v))
        return tuple(c / n for c in v)

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    entry = product.frame.port("IN")
    a = unit(vec(entry.direction))
    up = (0.0, 1.0, 0.0) if abs(a[1]) < 0.9 else (1.0, 0.0, 0.0)
    dot = sum(x * y for x, y in zip(up, a))
    b = unit(tuple(u - dot * x for u, x in zip(up, a)))
    c = cross(a, b)
    # the rotation maps local a -> design +X, b -> design +Z, c = a x b -> design +X x +Z = -Y
    def rotate(v):
        la, lb, lc = (sum(x * y for x, y in zip(v, w)) for w in (a, b, c))
        return (la, -lc, lb)

    axes = [rotate(e) for e in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
    quaternion = _quat_from_frame(*axes)
    shift = rotate(vec(entry.point_mm))
    position = [-shift[0], -shift[1], -shift[2]]
    outputs = []
    for port in product.frame.ports:
        if port.role != "OUTPUT":
            continue
        point = rotate(vec(port.point_mm))
        outputs.append(((point[0] + position[0], point[1] + position[1], point[2] + position[2]), rotate(unit(vec(port.direction)))))
    return quaternion, position, outputs


def _z_max(geometry) -> float:
    """The largest local z of a design geometry (cylinder, box or convex polyhedron)."""

    if getattr(geometry, "z_max_mm", None) is not None:
        return float(Decimal(geometry.z_max_mm))
    if getattr(geometry, "half_extents_mm", None) is not None:
        return float(Decimal(geometry.center_mm[2]) + Decimal(geometry.half_extents_mm[2]))
    return max(float(Decimal(v[2])) for v in geometry.vertices_mm)


def _clear_radius(product) -> float:
    model = product.simulation_models[0]
    radius = getattr(model, "clear_radius_mm", None)
    if radius is not None:
        return float(radius)
    return float(min(product.specifications.clear_aperture_extents_mm())) / 2


def _wavelengths(product, catalog) -> list[float]:
    """The probe wavelengths: the centre of every first-coating band inside the model band and inside a band of every
    other coating of the element; the model's design wavelength or band centre for an element without coatings."""

    model = product.simulation_models[0]
    low, high = Decimal(model.valid_band_nm.min_nm), Decimal(model.valid_band_nm.max_nm)
    coatings = [catalog.coating(s.coating_id) for s in getattr(model, "surfaces", ()) or () if s.coating_id]
    found: list[float] = []
    spectrum = coatings[0].spectrum if coatings else None
    if spectrum is not None:
        nodes = [(Decimal(w), Decimal(r), Decimal(t)) for w, r, t in zip(spectrum.wavelength_nm, spectrum.reflectance, spectrum.transmittance)
                 if low <= Decimal(w) <= high]
        for pick in (max(nodes, key=lambda n: (n[2], -n[0])), max(nodes, key=lambda n: (n[1], -n[0]))) if nodes else ():
            if all(c.band_at(pick[0]) is not None for c in coatings[1:]) and float(pick[0]) not in found:
                found.append(float(pick[0]))
    for band in (coatings[0].bands or ()) if coatings else ():
        lo, hi = max(low, Decimal(band.min_nm)), min(high, Decimal(band.max_nm))
        centre = (lo + hi) / 2
        if lo <= hi and all(c.band_at(centre) is not None for c in coatings[1:]) and float(centre) not in found:
            found.append(float(centre))
    if found:
        return found
    if getattr(model, "design_wavelength_nm", None):
        return [float(Decimal(model.design_wavelength_nm))]
    return [float((low + high) / 2)]


def _probe_aoi(product, catalog) -> float:
    model = product.simulation_models[0]
    ranges = [catalog.coating(s.coating_id).valid_aoi_deg for s in model.surfaces if s.coating_id and s.traced != "TRANSMITTED"]
    if not ranges:
        return PREFERRED_AOI_DEG
    low, high = Decimal(ranges[-1].min_deg), Decimal(ranges[-1].max_deg)
    if low <= Decimal(PREFERRED_AOI_DEG) <= high:
        return PREFERRED_AOI_DEG
    return float((low + high) / 2)


def probe_project(product, catalog) -> dict[str, Any]:
    clear = _clear_radius(product)
    waist = min(0.5, clear / 5)
    beam_x = [0.5, 0.5, 0.5, 0.5]  # local +Z -> design +X, local +Y -> design +Z
    components: list[dict[str, Any]] = [{
        "instance_id": f"S{index}", "optical_role": "source",
        "custom_part": {"custom_part_key": "probe-source", "description": "Probe Gaussian source."},
        "nominal_pose": {"position_mm": [-SOURCE_DISTANCE_MM, 0.0, 0.0], "quaternion_wxyz": beam_x},
        "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": wavelength, "power_w": 0.001, "waist_radius_mm": waist,
                         "waist_position_mm": SOURCE_DISTANCE_MM, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}},
    } for index, wavelength in enumerate(_wavelengths(product, catalog), 1)]
    role = sorted(product.design_roles)[0]
    outgoing: list[tuple[float, float, float]] = []
    starts: list[tuple[float, float, float]] = []  # where each outgoing beam leaves the element (ported products)
    position = [0.0, 0.0, 0.0]
    if product.frame.has_ports:
        pose, position, ports = _ported_pose(product)
        starts = [point for point, _ in ports]
        outgoing = [direction for _, direction in ports]
    elif product.frame.is_reflector:
        theta = math.radians(_probe_aoi(product, catalog))
        normal = (-math.cos(theta), math.sin(theta), 0.0)
        pose = _frame_along(normal)
        outgoing.append((-math.cos(2 * theta), math.sin(2 * theta), 0.0))
        if product.frame.nominal_propagation == "SPLIT_ABOUT_+Z":
            outgoing.append((1.0, 0.0, 0.0))
    else:
        pose = beam_x
        outgoing.append((1.0, 0.0, 0.0))
    components.append({"instance_id": "E1", "optical_role": role, "product": {"optical_sku": product.optical_sku, "catalog_version": catalog.catalog_version},
                       "nominal_pose": {"position_mm": position, "quaternion_wxyz": pose}})
    ported = product.frame.has_ports
    depth = _z_max(product.design_geometry) if not (product.frame.is_reflector or ported) else 0.0
    observations = []
    for index, direction in enumerate(outgoing, 1):
        start = starts[index - 1] if ported else (0.0, 0.0, 0.0)
        distance = OBSERVATION_DISTANCE_MM + (depth if direction == (1.0, 0.0, 0.0) else 0.0)
        if direction[0] < -RETURN_TOLERANCE:
            # The incoming beam runs from x = -SOURCE_DISTANCE_MM to the element; its projection on a returning direction
            # is at most SOURCE_DISTANCE_MM * -direction[0], so a plane beyond that is crossed by the returning beam only.
            distance += SOURCE_DISTANCE_MM * -direction[0]
        plane = [start[k] + direction[k] * distance for k in range(3)]
        observations.append({"observation_id": f"P{index}", "pose": {"position_mm": plane, "quaternion_wxyz": _frame_along_any(direction)},
                             "aperture": {"shape": "circle", "radius_mm": OBSERVATION_RADIUS_MM}, "absorbing": False})
    return {"schema": "opensci.project", "project_schema_version": "2.0", "components": components, "observations": observations,
            "targets": []}


def _axicon_scope_only(product, run, status: str, scope: list[str], unexercised: list[str]) -> bool:
    """True for an axicon whose run is out of model scope only because its probe beams are Gaussian: every issue that is
    not PASS / WARNING / NOT_APPLICABLE (a notice) is ``GAUSSIAN_UNDEFINED_AFTER_AXICON`` raised as ``OUT_OF_MODEL_SCOPE`` at one of the element's own
    surfaces, the local runtime adds no model-scope issue, and every source's rays met the element and reached an
    observation plane geometrically. Behind a cone the light forms a Bessel-like zone, which the Gaussian model does not
    describe (PHYSICS_SCOPE: rays are traced, power is measured only where material loss supports it)."""

    if getattr(product.form, "value", product.form) != "AXICON" or status != "OUT_OF_MODEL_SCOPE" or scope or unexercised:
        return False
    failing = [i for i in run.result.issues if i.status.value not in ("PASS", "WARNING", "NOT_APPLICABLE")]
    return bool(failing) and all(
        i.code.value == AXICON_GAUSSIAN_SCOPE and i.status.value == "OUT_OF_MODEL_SCOPE" and (i.location or "").startswith("surface:E1/")
        for i in failing)


def probe_one(product, catalog) -> dict[str, Any]:
    try:
        run = run_optical_project(probe_project(product, catalog), catalog=catalog)
    except LocalRunError as exc:
        return {"status": "LOCAL_RUN_ERROR", "codes": sorted({str(p.code.value if hasattr(p.code, "value") else p.code) for p in exc.problems})}
    except Exception as exc:  # noqa: BLE001 - a crash is a finding, recorded, never hidden
        return {"status": "CRASH", "codes": [type(exc).__name__], "detail": str(exc)[:300]}
    scope = sorted({issue.code for issue in run.model_scope_issues})
    status = run.status.value
    sources = {c.instance_id for c in run.project.components if c.source_model is not None}
    met = {p.source_id for p in run.ray_paths.paths if any(s is not None and s.startswith("E1/") for s in p.chief_surface_ids)}
    reached = {a.beam.source_id for o in run.result.observations for a in o.arrivals
               if a.beam.power_W is None or a.beam.power_W > 0}
    # A traced arrival with unknown power still exercises the geometric route, not a transmission claim. A line the
    # published coating blocks completely (a filter's blocking band: transmittance 0) is exercised too.
    blocked = {p.source_id for p in run.result.paths if p.termination == "ZERO_POWER" and (p.termination_surface_id or "").startswith("E1/")}
    unexercised = sorted(sources - (met & (reached | blocked)))
    loss_unknown = "BULK_LOSS_NOT_ESTABLISHED" in {i.code.value for i in run.result.issues}
    only_loss_unknown = (status == "UNKNOWN" and loss_unknown and
                         all(i.status.value in ("PASS", "WARNING", "NOT_APPLICABLE") for i in run.result.issues))
    power_evidence = ({"power_evidence": "UNKNOWN_BULK_LOSS", "codes": ["BULK_LOSS_NOT_ESTABLISHED"]} if loss_unknown else {})
    if (status in IN_SCOPE_STATUSES or only_loss_unknown) and not scope and unexercised:
        return {"status": "NOT_EXERCISED", "run_status": status, "codes": unexercised, **({"power_evidence": "UNKNOWN_BULK_LOSS"} if loss_unknown else {})}
    if (status in IN_SCOPE_STATUSES or only_loss_unknown) and not scope:
        return {"status": "IN_SCOPE", "run_status": status, "wavelengths": len(sources), **power_evidence}
    engine_codes = sorted({i.code.value for i in run.result.issues if i.status.value not in ("PASS", "WARNING", "NOT_APPLICABLE")})
    if _axicon_scope_only(product, run, status, scope, unexercised):
        return {"status": "IN_SCOPE", "run_status": status, "wavelengths": len(sources), "documented_scope": engine_codes,
                **power_evidence}
    return {"status": "OUT_OF_SCOPE" if status == "OUT_OF_MODEL_SCOPE" or scope else status, "run_status": status, "codes": scope + engine_codes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True, help="the public optical catalog release file to probe")
    args = parser.parse_args(argv)
    raw = args.catalog.read_bytes()
    catalog = load_public_catalog(raw)
    digest = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    build = engine_build_id()
    done: dict[str, Any] = {}
    if args.out.exists():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        if previous.get("catalog_sha256") == digest and previous.get("engine_build_id") == build:
            done = previous.get("optics", {})
            print(f"resuming: {len(done)} of {len(catalog.products)} already probed", flush=True)
    start = time.time()

    def save() -> None:
        document = {"schema": "opensci.internal.optical_simulation_probe", "schema_version": "1", "catalog_version": catalog.catalog_version,
                    "catalog_sha256": digest, "engine_build_id": build, "optics": dict(sorted(done.items()))}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes((json.dumps(document, sort_keys=True, indent=1) + "\n").encode("utf-8"))

    todo = [p for p in catalog.products if p.optical_sku not in done]
    for index, product in enumerate(todo, 1):
        done[product.optical_sku] = probe_one(product, catalog)
        if index % 100 == 0:
            save()
            print(f"[{time.time() - start:7.1f}s] probed {len(done)}/{len(catalog.products)}", flush=True)
    save()
    counts: dict[str, int] = {}
    for entry in done.values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    print(f"wrote {args.out}: {dict(sorted(counts.items()))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
