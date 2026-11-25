"""The rays of the light the run's surfaces scatter: each ``ScatterOrigin`` traced as a secondary extended emitter.

An origin (``optics/rays/scatter.py``) becomes a point emitter at the beam's hit point with the surface's angular law about the
surface normal on the scatter side (``LAMBERTIAN``, or ``COSINE_POWER`` with its exponent), unpolarized, of the scattered power,
in the medium it leaves into: the same rays, the same sampling sequence (``emitter_samples.py``), the same physics and the same
compact record as an extended emitter's (``sampled.py``). Its rays belong to the beam's own source line: their fates (detected,
absorbed, escaped, ...) are booked to that line's ledger in place of the origin's ``SURFACE_SCATTER`` entry, so the line's
ledger closes over the specular and the scattered light together; each origin keeps its own crossings per plane, its own
surface hits and what the model-scope check reads of its own rays (``SampledScope``: the surfaces they met and at which
angles, where they stopped).

At most ``max_scatter_origins`` origins (the run's compute budget, ``NumericalConfig``) are traced, the strongest first
(ties in trace order); their rays share a ray-step budget of ``max_interactions_per_path`` steps per declared ray, as an
emitter's. The rays are not kept as whole paths (the
crossings hold each ray's power, position and direction on every plane).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

from ...config import NumericalConfig
from ...math.vectors import transverse_basis
from ...scene.model import EmitterG, Scene
from ..materials.library import MaterialLibrary
from .emitter_samples import global_samples
from .hits import SurfaceHit
from .path_ids import COMPACT_MARK, DIGEST_HEX_DIGITS, root_path_id
from .sampled import SAMPLES_NOT_LAUNCHED, SampledLine, SampledRows, SampledScope, SampleTracer, _NO_PATH_LIMIT
from .scatter import ScatterOrigin
from .scheduler import TraceScheduler, order_power
from .tracer import SINK_PATH_LIMIT

#: A scatter origin whose emitting medium has no data at the line's wavelength is not traced (its booking stays).
MEDIUM_NOT_KNOWN = "medium"


@dataclass
class ScatterLine:
    """One traced origin: its emitter line (the beam's source line, the scattered power), its crossings per plane, the
    power its rays brought to each physical surface (one ``SurfaceHit`` per surface, holding the sum) and what the
    model-scope check reads of its rays (``scope``: this origin's alone)."""

    origin: ScatterOrigin
    line: SampledLine
    rows: dict[str, SampledRows] = field(default_factory=dict)
    hits: list[SurfaceHit] = field(default_factory=list)
    scope: SampledScope = field(default_factory=SampledScope)


@dataclass
class ScatterTrace:
    """The scattered light of a run: the traced origins (``lines``) and those left booked (``not_followed``; of them
    ``beyond_budget``: the origins beyond the ``max_origins`` strongest, the budget ``max_scatter_origins`` in force)."""

    lines: list[ScatterLine] = field(default_factory=list)
    not_followed: list[ScatterOrigin] = field(default_factory=list)
    beyond_budget: list[ScatterOrigin] = field(default_factory=list)
    max_origins: int | None = None
    flows: list = field(default_factory=list)  # the rays' ledger entries (``tracer.PowerFlow``)
    ray_steps: int = 0
    budget_reached: bool = False


def scatter_emitter(origin: ScatterOrigin, cfg: NumericalConfig) -> EmitterG:
    """The point emitter of an origin, in the design frame."""

    e1, e2 = transverse_basis(origin.axis, cfg.zero_vector_tol)
    return EmitterG(
        source_id=origin.source_id, owner_id=origin.source_owner_id, position=origin.point, direction=origin.axis,
        e1=e1[0], e2=e2[0], spectrum=((origin.wavelength_nm, 1.0),), power_W=origin.power_W, medium=origin.medium, shape="disc",
        half_x=0.0, half_y=0.0, law="COSINE_POWER" if origin.law == "COSINE_POWER" else "LAMBERTIAN", sample_count=origin.sample_count,
        seed=origin.seed + origin.ordinal, exponent=origin.exponent)


def scatter_ray_path_id(origin: ScatterOrigin, k: int) -> str:
    """The path id of an origin's ``k``-th ray: its line's root and a compacted suffix (``path_ids.py``) naming the beam's path,
    the surface and the ray, which no branch of the beam has."""

    root = root_path_id(origin.source_id, origin.spectral_index)
    suffix = f"{origin.path_id[len(root):]}@{origin.surface_id}/{k}"
    return root + COMPACT_MARK + hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:DIGEST_HEX_DIGITS]


class ScatterTracer(SampleTracer):
    """Traces the rays of scatter origins (module docstring). One instance serves one run."""

    def __init__(self, scene: Scene, library: MaterialLibrary, cfg: NumericalConfig, origins: list[ScatterOrigin], *, quad_rule=None) -> None:
        super().__init__(scene, library, cfg, recorder=None, quad_rule=quad_rule)
        # (strongest first on the run's ordering grid, ``scheduler.order_power``; equal there: in origin order)
        strongest = sorted(origins, key=lambda o: (-order_power(o.power_W, self.order_quantum_W), o.ordinal))
        followed = {o.ordinal for o in strongest[:cfg.max_scatter_origins]}
        self.followed = [o for o in origins if o.ordinal in followed]
        beyond = [o for o in origins if o.ordinal not in followed]
        self.scattered = ScatterTrace(not_followed=list(beyond), beyond_budget=beyond, max_origins=cfg.max_scatter_origins)
        rays = 0
        for o in self.followed:
            rays += o.sample_count
        self.sampled.step_budget = rays * cfg.max_interactions_per_path
        self.schedule = TraceScheduler(_NO_PATH_LIMIT, self.sampled.step_budget, cfg.max_interactions_per_path, self.order_quantum_W)
        self._origin: ScatterOrigin | None = None

    def run_origins(self) -> ScatterTrace:
        for origin in self.followed:
            emitter = scatter_emitter(origin, self.cfg)
            j, wavelength_nm = origin.spectral_index, origin.wavelength_nm
            line = SampledLine(origin.source_id, origin.source_owner_id, j, wavelength_nm, origin.power_W,
                               origin.power_W / origin.sample_count, origin.sample_count, emitter.seed)
            state = self._sample_state(emitter, wavelength_nm)
            if state is None:  # (the medium has no data here: the scattered light stays booked, not followed)
                line.medium_unknown = True
                self.scattered.not_followed.append(origin)
                continue
            self._origin, self._line = origin, line
            gstate, n = state
            samples = global_samples(emitter)
            bases = transverse_basis(samples.directions, self.cfg.zero_vector_tol)
            for k in range(origin.sample_count):
                if self.schedule.steps >= self.sampled.step_budget:
                    self.sampled.budget_reached = True
                    self._book(SINK_PATH_LIMIT, line.weight_W * (origin.sample_count - k), origin.source_id, j, wavelength_nm,
                               None, origin.source_owner_id, SAMPLES_NOT_LAUNCHED)
                    break
                line.launched = k + 1
                self._sample_index = k
                self._keep = False
                pk = self._packet(emitter, j, k, wavelength_nm, line.weight_W, gstate, n, samples, bases)
                pk.path_id = scatter_ray_path_id(origin, k)
                self._enqueue(pk)
                self._drain()
            self.scattered.lines.append(self._harvest(origin, line))
        self._origin = None
        self.scattered.ray_steps = self.sampled.ray_steps = self.schedule.steps
        self._close()
        self.scattered.budget_reached = self.sampled.budget_reached
        return self.scattered

    def _harvest(self, origin: ScatterOrigin, line: SampledLine) -> ScatterLine:
        """This origin's crossings, surface hits and model-scope record; the record starts empty for the next origin."""

        rows = {oid: r for (oid, _source, _j), r in self.sampled.rows.items()}
        hits = [h for h in self.out.surface_hits if h.path_id in self.sampled.hit_paths]
        sums = {surface: power for (surface, _source, _j), power in self.sampled.hit_W.items()}
        # (light scattered out of a beam whose power is out of model scope carries that beam's chief-ray power: so are its hits)
        combined = [replace(h, power_W=sums[h.surface_id], power_scope=tuple(dict.fromkeys((*h.power_scope, *origin.power_scope))))
                    for h in hits]
        self.out.surface_hits = [h for h in self.out.surface_hits if h.path_id not in self.sampled.hit_paths]
        scope = self.sampled.scope
        self.sampled.rows, self.sampled.hit_W, self.sampled.hit_paths, self.sampled.scope = {}, {}, set(), SampledScope()
        return ScatterLine(origin, line, rows, combined, scope)


def trace_scatter(scene: Scene, library: MaterialLibrary, cfg: NumericalConfig, origins: list[ScatterOrigin], quad_rule=None
                  ) -> tuple[ScatterTrace, ScatterTracer]:
    """Trace the origins' rays; the tracer holds what they booked (``out``: ledger entries, issues)."""

    tracer = ScatterTracer(scene, library, cfg, origins, quad_rule=quad_rule)
    return tracer.run_origins(), tracer


__all__ = ["ScatterLine", "ScatterTrace", "ScatterTracer", "scatter_emitter", "scatter_ray_path_id", "trace_scatter"]
