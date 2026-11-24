"""Project-level contracts: Baseplate, Observation, Target, ControlSpec and
ProjectSnapshot."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from .._version import SCHEMA_VERSION
from .base import Id, OmitAbsentFields, StrictModel, Vector2
from .components import EngineeringComponentSnapshot
from .detection import Detection
from .geometry import Pose3D
from .materials import MaterialOpticalModel
from .optics import Aperture


class BaseplateSnapshot(StrictModel):
    outline_mm: tuple[Vector2, ...] = Field(min_length=3, description="Plate outline polygon in global XY.")
    top_z_mm: float = 0.0
    edge_margin_mm: float = Field(0.0, ge=0, description="Required distance from component footprints to the plate edge.")


class Observation(OmitAbsentFields, StrictModel):
    """A virtual observation plane (monitor / detector / substrate exposure plane).

    The plane frame is ``pose`` (optionally attached to a component: ``global = component.pose ∘ pose``);
    its normal is the frame ``+z`` axis. ``absorbing`` planes terminate the beams they receive. ``detection``: a stated
    detector on the plane and, optionally, a lock-in measurement (``contracts/detection.py``); omitted when absent.
    """

    absent_omitted: ClassVar[tuple[str, ...]] = ("detection",)

    observation_id: Id
    device_id: str | None = Field(None, description="Component instance the plane is attached to (moves with its actuators); `pose` is then in that component's local frame.")
    pose: Pose3D = Pose3D()
    aperture: Aperture | None = Field(None, description="Active area in the plane (u = frame x, v = frame y).")
    acceptance_half_angle_deg: float | None = Field(None, gt=0, le=90, description="Maximum AOI accepted by the detector.")
    absorbing: bool = False
    medium: Id = "air"
    detection: Detection | None = None


class TargetNotAssessable(StrictModel):
    """Why a declared requirement cannot be read as written: the ``code`` of the input problem and its ``message``."""

    code: str = Field(min_length=1, description="The code of the input problem (the target result's issue code).")
    message: str = Field(min_length=1, description="What is wrong with the requirement as written and what to state instead.")


class Target(OmitAbsentFields, StrictModel):
    """A verifiable requirement.

    ``wavelength_nm_b`` selects the second operand's spectral line of a geometric pair metric (``validation/alignment.py``;
    default: ``wavelength_nm``); other metrics refuse it.

    Comparison: ``target_value`` with ``tolerance_abs``/``tolerance_rel`` (the larger tolerance applies), and/or
    ``minimum``/``maximum`` bounds. At least one comparison must be given.

    ``not_assessable``: the requirement as declared cannot be read (a metric that does not exist, a selector of the wrong
    type, a source that is not one). Such a target states its id, its metric and the reason, and nothing else: it selects
    no beam, no plane, no element and no analysis, and it is never compared with a value. Its result is ``UNKNOWN``
    (assessment ``INDETERMINATE``) with ``not_assessable.code`` as its issue code, so a declared requirement is counted
    in every run. Omitted when absent.

    A metric that reads an observation plane needs ``observation_id``, and a metric with a required parameter needs it in
    ``parameters`` (``validation/target_parameters.py``): a target without them is refused here, before any run (the local
    lowering carries such a declared target as ``not_assessable``).
    """

    absent_omitted: ClassVar[tuple[str, ...]] = ("wavelength_nm_b", "not_assessable")
    #: The fields a target that is not assessable states (``not_assessable`` itself and what names the requirement).
    NOT_ASSESSABLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"target_id", "metric", "not_assessable"})

    target_id: Id
    metric: str = Field(min_length=1)
    device_id: str | None = None
    observation_id: str | None = None
    source_id: str | None = None
    source_id_b: str | None = Field(None, description="Second beam for two-beam metrics.")
    wavelength_nm: float | None = Field(None, gt=0, description="Select one spectral component; default: the first.")
    wavelength_nm_b: float | None = Field(
        None, gt=0, description="The second operand's spectral line of a geometric pair metric (default: wavelength_nm).")
    target_value: float | None = None
    tolerance_abs: float | None = Field(None, ge=0)
    tolerance_rel: float | None = Field(None, ge=0)
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    parameters: dict[str, float | str] = Field(default_factory=dict)
    source_revision: str | None = None
    not_assessable: TargetNotAssessable | None = Field(
        None, description="Why the requirement cannot be read as written; the target then states only target_id and metric.")

    @field_validator("parameters", mode="before")
    @classmethod
    def _parameters_of_the_metric(cls, value: Any, info: ValidationInfo) -> Any:
        """Only the keys the metric takes, each of its type (``validation/target_parameters.py``): anything else is refused,
        never ignored or rounded."""

        from ..validation.target_parameters import check_target_parameters  # the metric registry imports this module

        metric = info.data.get("metric")
        if isinstance(metric, str):
            check_target_parameters(metric, value)
        return value

    @model_validator(mode="after")
    def _comparison(self) -> Target:
        if self.not_assessable is not None:
            fields = type(self).model_fields
            stated = sorted(name for name in fields if name not in self.NOT_ASSESSABLE_FIELDS
                            and getattr(self, name) != fields[name].get_default(call_default_factory=True))
            if stated:
                raise ValueError("a Target that is not assessable states only target_id, metric and not_assessable, not "
                                 + ", ".join(stated))
            return self
        if self.target_value is None and self.minimum is None and self.maximum is None:
            raise ValueError("a Target needs target_value and/or minimum/maximum")
        if self.target_value is not None and self.tolerance_abs is None and self.tolerance_rel is None:
            raise ValueError("target_value requires tolerance_abs and/or tolerance_rel")
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("maximum must be >= minimum")
        from ..validation.target_parameters import REQUIRED_PARAMETERS, requires_observation  # the metric registry imports this module

        if self.observation_id is None and requires_observation(self.metric):
            raise ValueError(f"metric {self.metric!r} reads an observation plane: set observation_id to the plane it is read on")
        missing = [key for key in REQUIRED_PARAMETERS.get(self.metric, ()) if key not in self.parameters]
        if missing:
            raise ValueError(f"metric {self.metric!r} needs parameter {missing[0]!r}: add it to the target's parameters")
        if self.wavelength_nm_b is not None:
            from ..validation.alignment import ALIGNMENT_METRICS  # the metric registry imports this module

            if self.metric not in ALIGNMENT_METRICS:
                raise ValueError(f"wavelength_nm_b selects the second beam of a geometric pair metric ({', '.join(sorted(ALIGNMENT_METRICS))}), "
                                 f"not of {self.metric!r}")
        return self


# ------------------------------------------------------------------------------------------- control


class SensorSpec(StrictModel):
    sensor_id: Id
    observation_id: Id
    quantity: Literal["centroid_u_mm", "centroid_v_mm", "power_W"]
    source_ids: tuple[str, ...] = Field((), description="Beams contributing; empty = all beams reaching the plane.")
    target_value: float
    weight: float = Field(1.0, gt=0)


class MonteCarloSpec(StrictModel):
    sample_count: int = Field(ge=1)
    seed: int | None = Field(None, ge=0, description="Seed of the samples; None: configuration.random_seed, else 0. The seed "
                             "used is recorded (ControlResult.monte_carlo_seed, manifest.random_seed).")
    amplitude: dict[str, float] = Field(
        default_factory=dict, description="Uniform +-amplitude per actuator ref 'instance/actuator' (in the actuator's unit)."
    )
    distribution: Literal["uniform", "normal"] = "uniform"


class RecoverabilitySpec(StrictModel):
    disturbances: tuple[dict[str, float], ...] = Field((), description="Explicit disturbance vectors keyed by actuator ref.")
    convergence_tolerance: float = Field(gt=0, description="Weighted residual norm regarded as converged.")


class CaptureScanSpec(StrictModel):
    actuator_refs: tuple[str, ...] = Field(min_length=1, max_length=2)
    points_per_axis: int = Field(21, ge=2)


class ControlSpec(StrictModel):
    """Which DOFs are controlled, what is sensed, and which analyses run."""

    actuator_refs: tuple[str, ...] = Field(min_length=1, description="'<instance_id>/<actuator_id>' of controlled DOFs.")
    sensors: tuple[SensorSpec, ...] = Field(min_length=1)
    capture_scan: CaptureScanSpec | None = None
    recoverability: RecoverabilitySpec | None = None
    monte_carlo: MonteCarloSpec | None = None
    compute_sensitivity: bool = True


class ProjectSnapshot(StrictModel):
    """Engine-side project input: components, observations, targets, materials and the optional control spec."""

    project_schema_version: str = SCHEMA_VERSION
    project_id: str | None = None
    revision_id: str | None = None
    name: str = ""
    baseplate: BaseplateSnapshot | None = None
    components: tuple[EngineeringComponentSnapshot, ...] = ()
    observations: tuple[Observation, ...] = ()
    targets: tuple[Target, ...] = ()
    materials: tuple[MaterialOpticalModel, ...] = Field((), description="Project material library (adds to / overrides built-ins).")
    control_spec: ControlSpec | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> ProjectSnapshot:
        for label, ids in (
            ("component", [c.instance_id for c in self.components]),
            ("observation", [o.observation_id for o in self.observations]),
            ("target", [t.target_id for t in self.targets]),
            ("material", [m.material_id for m in self.materials]),
        ):
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate {label} ids")
        src_ids = [c.source.source_id for c in self.components if c.source is not None]
        src_ids += [c.emitter.source_id for c in self.components if c.emitter is not None]
        if len(set(src_ids)) != len(src_ids):
            raise ValueError("duplicate source ids")
        return self
