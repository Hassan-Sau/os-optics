"""MetricDescriptor: the machine-readable meaning of one public target metric (``validation/descriptors.py``)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import StrictModel


class MetricDescriptor(StrictModel):
    """What a metric means, precisely enough to use it without reading source code.

    * ``kind``: ``observation`` (one beam arriving at an observation plane), ``pair`` (two beams on one plane), ``element``
      (one optical element, ``device_id``; with ``parameters.axis`` one of its retained DOFs), ``layout`` / ``control``
      (mechanical stages; not available in a pure-optical design);
    * ``definition``: the physical quantity and how it is computed; ``sign_convention`` / ``operands``: signs, operand order
      and what a zero means; ``reference_frame``: the frame its coordinates or angles are in;
    * ``selectors``: target fields used to pick the beam(s): ``observation_id`` is required for observation / pair metrics;
      with several beams on the plane select one with ``source_id`` / ``wavelength_nm`` / ``parameters.path_id``
      (pair: ``source_id`` + ``source_id_b`` or ``parameters.path_id_a`` + ``path_id_b``). ``parameters.path_id*`` are
      strings, ``parameters.crossing_index*`` integers; a target's ``parameters`` take only these keys of its metric
      (and ``dose_mj_per_cm2`` for ``estimated_exposure_time_s``), anything else is refused as input;
    * ``aggregation``: how several matching beams combine (``single_beam``: they must be narrowed to one;
      ``declared_limit``: no beam, the stated limits of one retained DOF);
    * ``model_basis``: which Engine model produces the number; ``validity``: when it is withheld (``null`` with a status)
      instead of reported; ``version``: the metric definition version (bumped when the meaning changes);
    * ``result_fields``: where its value appears among a run's results, so a script reading results finds the field a target
      metric names (``divergence_half_angle_x_rad`` is ``arrivals.far_field_half_angle_x_rad``);
    * ``requires``: the evaluation capabilities a target on the metric reads (``os_optics.evaluation``): an evaluation that
      asks for the requirements runs them for each selected target; a target on a stage the request's configuration or
      content turns off is assessed without it and reads as in ``validate``: a layout, control or rating-utilization target
      says the stage did not run, a target on an observation plane of a request that does not trace reads the plane as
      unknown, and a two-beam target is not met. A threshold never changes them.
    """

    name: str
    unit: str
    kind: Literal["observation", "pair", "element", "layout", "control"]
    definition: str
    sign_convention: str = ""
    operands: str = ""
    reference_frame: str = ""
    selectors: tuple[str, ...] = ()
    aggregation: Literal["single_beam", "sum_over_matching_beams", "beam_pair", "max_over_surfaces", "stage", "declared_limit"] = "single_beam"
    model_basis: str
    validity: str = ""
    version: str = Field("1", description="Metric definition version.")
    result_fields: tuple[str, ...] = Field((), description="Where the metric's value appears among a run's results: '<export kind>.<column>' "
                                           "(a column of 'os-optics export KIND' / RunResults.table) and 'monitor:<path>' (a field of "
                                           "RunResults.monitor(id)); empty when the metric is computed from several fields.")
    requires: tuple[str, ...] = Field((), description="The evaluation capabilities a target on this metric reads (capability ids of "
                                      "os_optics.evaluation); empty: the metric reads the design's declared values alone.")
