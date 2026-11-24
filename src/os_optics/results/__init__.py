"""Scientific results of a local run: discover, inspect and export everything the Engine computed, from Python.

A run directory (``os-optics project run project.json --out-dir run`` or ``run_optical_project(project,
output_dir="run")``) holds the complete record of one Engine execution. This module reads it -- nothing is simulated
again -- and gives every result as plain JSON-safe values (``dict`` / ``list`` / numbers / strings / ``None``; never NaN)::

    from os_optics.results import load_run

    results = load_run("run")              # a run directory, its {file name: bytes}, or a LocalOpticalRun
    results.describe()                     # identity, status, result types present, frame and sign conventions, files
    results.targets()                      # every target outcome (value, unit, bounds, status, issue codes), counts per
                                           # status, the model-scope issues and the run identity
    results.monitors()                     # every observation plane: frame (origin, u, v, n), active area, absorbing, power
    results.monitor("focal_plane")         # every arrival's full beam state (position, direction, power, OPL, phase, Jones
                                           # and Stokes, Gaussian q / waist / radius, first-order transfer, lineage), crossings
                                           # outside the active area, power by wavelength, coherent sums, targets
    results.paths()                        # the beam-path tree: parent / children, branch, termination, power, OPL
    results.power_budget()                 # where the power went: input, every sink, power seen at monitors, residual
    results.rays(monitor="focal_plane")    # chief + pupil rays at the plane: u, v, xyz, directions, OPL, launch offsets
    results.wavefront("focal_plane", "SRC#0", reference="sphere", center="ray_focus")   # OPD per pupil ray, RMS, PV
    results.irradiance_map("focal_plane")  # irradiance grid (incoherent and coherent sums of the Gaussian beams)
    results.handoff("stop_plane", element="STOP")  # beams in the plane's (u, v, n) frame + an element's apertures there, with
                                           # the TEM00 field-construction convention (for external diffraction / full-wave tools)
    results.exposures()                    # per element surface: incident power, both beam radii, AOI, peak irradiance /
                                           # fluence, P / d, and the comparison with the elements' power ratings
    results.metrics()                      # what every target metric means: unit, definition, sign, operands, validity
    results.export("arrivals", "csv")      # bytes: csv / jsonl / json (npz for the irradiance grid)
    results.detector_response("focal_plane", detector)  # photons, electrons, ADU, saturation, noise and SNR per pixel of a stated detector
    results.detection("D")                 # a plane's stated detector: photocurrent, noise, SNR and its lock-in's signal, noise and SNR
    results.sample_emission()              # how each fluorescence / Raman sample's line powers follow from its excitation
    results.pupil_field("pupil_plane", request)  # the scalar low-NA focal field of the plane's beams through a pupil mask

The same from a shell: ``os-optics results describe run``, ``results targets run``, ``results freshness run --project
project.json`` (is the run current for that Project), ``results monitors run``, ``results monitor run ID``,
``results paths run``, ``results power-budget run``, ``results rays run --monitor ID``, ``results wavefront run --monitor
ID --path P``, ``results irradiance run --monitor ID``, ``results handoff run --monitor ID [--element ID]``, ``results
exposures run [--element ID]``, ``results detector``, ``results detection run [--monitor ID]``, ``results sample-emission run``,
``results pupil-field``, ``results metrics`` and ``os-optics export KIND run --format
csv --out FILE``.

Export kinds (``EXPORT_KINDS``; ``columns(kind)`` documents every column with its unit): ``monitors``, ``arrivals``,
``rays``, ``paths``, ``segments``, ``power-budget``, ``superpositions``, ``wavefront``, ``irradiance``, ``sampled-light``,
``sampled-crossings``, ``metrics``,
``handoff`` (one row per beam; its JSON form adds the plane, the element outline and the field-construction convention),
``exposures`` (one row per beam meeting an element surface) and ``power-ratings`` (one row per surface and rating). Column
names carry their unit (``beam_radius_x_mm``, ``power_W``, ``opd_waves``); see ``os_optics.results.export`` for the
exact CSV / JSONL / JSON / NPZ layout. Rays and paths are tables of a few thousand rows at most, so they are exported as
text; only the dense irradiance grid is also offered as NPZ.

Conventions (``describe()["conventions"]`` states them in full): positions and directions are in the right-handed design
frame (mm); each observation plane has unit axes ``u``, ``v`` and normal ``n = u x v`` and ``u_mm`` / ``v_mm`` are plane
coordinates; Gaussian ``*_x`` / ``*_y`` quantities refer to the beam's own axes; ``crossing_index`` counts a path's
crossings of one plane from 0. Model boundaries stay explicit: a quantity the model does not define is ``None`` with a
status or reason, a wavefront request the data cannot support is refused with a ``reason`` code
(``RAY_OPL_UNDEFINED`` after a thin lens, a Faraday rotator or a diffraction grating, ``TOO_FEW_RAYS``, ...), and the wavefront and irradiance results are
geometric-optics / paraxial Gaussian quantities without diffraction.

Two calculations go beyond what a run computed, stage by stage, each quantity stating ``MODELED``, ``EXTERNALLY_CHARACTERIZED`` or
``UNAVAILABLE`` (``os_optics.results.quantity``): ``pupil_field`` (``os_optics.results.pupil_field``) takes a pupil-plane field and mask to
a scalar low-NA focal irradiance; ``focal_field`` (``os_optics.results.focal_field``) takes an ideal aplanatic objective and its pupil
illumination to the vector focal field at any numerical aperture, as an analysis document (``AnalysisResultV1``); ``RunResults.field``
(``os_optics.results.field``) propagates the coherent field of a run's sampled-field or speckle source to a plane (angular spectrum in
free space, the Collins integral of the source axis's first-order transfer through elements) to its irradiance, power and speckle
contrast over realizations, as an analysis document; ``RunResults.imaging``
(``os_optics.results.imaging``) traces a run's fields again with a dense pupil grid to the point spread function, MTF and encircled energy
of its image plane, as an analysis document; ``RunResults.ghosts`` (``os_optics.results.ghosts``) traces them again with every lens or
window face's reflection followed up to a bounce order to the ghost images of an image plane, their focus and relative power, with a
proven bound on the ghosts not listed; ``detector_response`` (``os_optics.results.detector``) takes any irradiance map to detector counts. Both
are also plain functions of their documents, with no run needed.

A bare Engine run (``os_optics.validate_with_ray_paths(request)``) is read with ``results_from_engine(result,
ray_paths)``. An evaluation held in memory (``os_optics.local.EvaluationSession``) gives its results with
``Evaluation.results()``, without files; a query of a part the evaluation did not compute raises ``ResultsError`` with code
``NOT_COMPUTED`` (``QUERY_REQUIRES``: every capability each query reads), whose message names each capability to request.

What a number is: ``basis_of(record, field)`` returns its ``value_basis`` entry (``os_optics.contracts.basis``) -- the one the
record stores, else the one its flat fields imply -- or ``None`` when the number is the model's value. The rows of every query
and table carry the same entries under ``value_basis``, keyed by the row's own name of the number.
"""

from .basis import CARRIAGE, DERIVATIONS, bases_of, basis_of
from .detector import DETECTOR_SCHEMA, DetectorError, detector_response
from .handoff import FIELD_CONVENTION, HANDOFF_SCHEMA, HANDOFF_SCHEMA_VERSION, handoff
from .focal_field import FOCAL_FIELD_ANALYSIS, FocalFieldError, focal_field
from .export import EXPORT_SCHEMA, EXPORT_SCHEMA_VERSION, FORMATS, NPZ_KINDS
from .irradiance import IRRADIANCE_MODEL, irradiance_map
from .model import (
    CONVENTIONS,
    EXPORT_KINDS,
    NOT_COMPUTED,
    QUERY_REQUIRES,
    RESULTS_SCHEMA,
    RESULTS_SCHEMA_VERSION,
    TABLE_FILTERS,
    ResultsError,
    RunResults,
    load_run,
    results_from_engine,
    results_from_files,
)
from .imaging import IMAGING_ANALYSIS, ImagingError, imaging
from .field import FIELD_ANALYSIS, FieldError, field
from .ghosts import GHOST_ANALYSIS, GhostError, ghosts
from .pupil_field import PUPIL_FIELD_SCHEMA, SCALAR_SINE_LIMIT, PupilFieldError, pupil_field
from .rays import MIN_WAVEFRONT_RAYS, WAVEFRONT_MODEL, wavefront
from .tables import columns_of as columns

__all__ = [
    "CARRIAGE",
    "CONVENTIONS",
    "DERIVATIONS",
    "NOT_COMPUTED",
    "QUERY_REQUIRES",
    "DETECTOR_SCHEMA",
    "DetectorError",
    "EXPORT_KINDS",
    "EXPORT_SCHEMA",
    "EXPORT_SCHEMA_VERSION",
    "FIELD_ANALYSIS",
    "FIELD_CONVENTION",
    "FieldError",
    "FOCAL_FIELD_ANALYSIS",
    "FORMATS",
    "FocalFieldError",
    "GHOST_ANALYSIS",
    "GhostError",
    "HANDOFF_SCHEMA",
    "HANDOFF_SCHEMA_VERSION",
    "IMAGING_ANALYSIS",
    "IRRADIANCE_MODEL",
    "ImagingError",
    "MIN_WAVEFRONT_RAYS",
    "NPZ_KINDS",
    "PUPIL_FIELD_SCHEMA",
    "PupilFieldError",
    "RESULTS_SCHEMA",
    "RESULTS_SCHEMA_VERSION",
    "SCALAR_SINE_LIMIT",
    "TABLE_FILTERS",
    "WAVEFRONT_MODEL",
    "ResultsError",
    "RunResults",
    "bases_of",
    "basis_of",
    "columns",
    "detector_response",
    "field",
    "focal_field",
    "ghosts",
    "handoff",
    "imaging",
    "irradiance_map",
    "load_run",
    "pupil_field",
    "results_from_engine",
    "results_from_files",
    "wavefront",
]
