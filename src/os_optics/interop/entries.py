"""The entries of the interop capability catalog, as Python data (so they are part of the Engine build identity).

Each entry follows ``model.InteropEntryV1``. Tool licenses are quoted as the upstream distribution reported them on the
stated date (PyPI metadata or the source repository). Entry points are exact names that exist in this Engine version; a
test checks every one of them.
"""

from __future__ import annotations

from .model import EntryPointV1 as E
from .model import InteropEntryV1, ToolV1, VerifiedRecordV1

CHECKED = "2026-10-15"
PYPI = "PyPI metadata"
REPO = "source repository"
#: The commit in the history of the public kit repository (JunLin615/os-optics-kit) at which the cases of the verified
#: records below were executed (Engine 0.8.0); they then lived under ``examples/interop/`` and are now the kit's
#: ``cases/interop-*``.
COMPANION_COMMIT = "629e563757f7f26f9e6a5e3066b9524144473978"


def _tool(name: str, role: str, url: str, computes: str, license: str | None = None, source: str | None = PYPI, **kw) -> ToolV1:
    if license is None:
        return ToolV1(name=name, role=role, url=url, computes=computes, **kw)  # type: ignore[arg-type]
    return ToolV1(name=name, role=role, url=url, computes=computes, license=license, license_source=source,  # type: ignore[arg-type]
                  license_checked=CHECKED, **kw)


# ---------------------------------------------------------------------------------------------- shared entry points

RUN_PY = E(kind="PYTHON", name="os_optics.local.run_optical_project", purpose="simulate an edited Project in memory (one Engine run)")
RUN_CLI = E(kind="CLI", name="os-optics project run", purpose="simulate a Project file and write the run directory")
RUN_JSON = E(kind="JSON_API", name="run_project", purpose="the same run from a local JSON host (including Pyodide or another language)")
LOAD_RUN = E(kind="PYTHON", name="os_optics.results.load_run", purpose="read every result of a run: monitors, beams, paths, rays")
METRICS_CLI = E(kind="CLI", name="os-optics results metrics", purpose="what every target metric means, with its result field")
EXPORT_CLI = E(kind="CLI", name="os-optics export", purpose="one kind of result as CSV / JSON Lines / JSON")
HANDOFF_PY = E(kind="PYTHON", name="os_optics.results.RunResults.handoff",
               purpose="beam state and element outline at an observation plane, with the field-construction convention")
HANDOFF_CLI = E(kind="CLI", name="os-optics results handoff", purpose="the transverse-plane handoff from a shell")
NETWORK_JSON = E(kind="JSON_API", name="linear_optical_network", purpose="the Project's optics as opensci-linear-optical-network/v1")
NETWORK_CLI = E(kind="CLI", name="os-optics project network", purpose="export the linear optical network to a JSON file")
NETWORK_SCHEMA_CLI = E(kind="CLI", name="os-optics schema linear-optical-network", purpose="the JSON Schema of the network document")
PARAXIAL_CLI = E(kind="CLI", name="os-optics project paraxial",
                 purpose="thick-lens and mirror ABCD, focal distances, principal planes, afocal and collimation spacing")
CATALOG_SHOW = E(kind="CLI", name="os-optics catalog show", purpose="a product's public document: geometry, frame, materials, coatings")
MODEL_FROM_CATALOG = E(kind="CLI", name="os-optics project model-from-catalog",
                       purpose="a catalog product's simulation model as a declared, editable optical model")
NATIVE_DESCRIBE = E(kind="CLI", name="os-optics native describe", purpose="a native component class: parameters, physics, approximations")

RESULTS_CONVENTIONS = "results.describe()['conventions'] (os-optics results describe RUN): design frame, plane frame, Gaussian frame, Jones basis, phase"
HANDOFF_CONVENTIONS = "the handoff document's conventions: plane (u, v, n) frame, Gaussian axes, TEM00 field construction from q, normalization to power"
NETWORK_CONVENTIONS = ("network['conventions'] and os-optics schema linear-optical-network: exp(-i w t), flux-normalized amplitudes, "
                       "b_out = T a_in, lab H / V basis")
SEPARATE_RESULTS = ("External results are kept in their own files next to the run they used, naming the run's project_hash, "
                    "engine_input_hash and result_hash; they never replace the run's hashes or its Local Validation.")

ENTRIES: tuple[InteropEntryV1, ...] = (
    InteropEntryV1(
        entry_id="optimization-tolerance",
        title="Optimization, tolerancing and sensitivity with SciPy / Optuna / SALib",
        direction="OPTIMIZATION_TOLERANCE",
        status="VERIFIED_EXAMPLE",
        verified=(
            VerifiedRecordV1(tool="scipy", version="1.18.1", python="CPython 3.12.14", date="2026-10-15", companion_commit=COMPANION_COMMIT, engine_version="0.8.0",
                             example="cases/interop-optimization/optimize_expander.py",
                             reference="independent ABCD Gaussian-beam optimum (numpy)",
                             outcome="WITHIN_TOLERANCE: SciPy optimum within 6e-14 mm of the reference; Engine vs reference within 3e-14 relative over 709 runs"),
            VerifiedRecordV1(tool="optuna", version="5.0.0", python="CPython 3.12.14", date="2026-10-15", companion_commit=COMPANION_COMMIT, engine_version="0.8.0",
                             example="cases/interop-optimization/optimize_expander.py",
                             reference="independent ABCD Gaussian-beam optimum (numpy)",
                             outcome="WITHIN_TOLERANCE: best trial within 0.14 % of the reference divergence"),
            VerifiedRecordV1(tool="SALib", version="1.6.0", python="CPython 3.12.14", date="2026-10-15", companion_commit=COMPANION_COMMIT, engine_version="0.8.0",
                             example="cases/interop-optimization/optimize_expander.py",
                             reference="Sobol indices of the independent ABCD model",
                             outcome="WITHIN_TOLERANCE: total-order indices within 1e-14 of the reference model"),
        ),
        scenario="Search component positions or parameters for a target (focus position, collimation, beam size, coupling power), "
                 "compare candidate components, or propagate assembly errors to a distribution of results, each evaluation being a "
                 "full OpenSci run.",
        tools=(
            _tool("SciPy", "RECOMMENDED", "https://docs.scipy.org/doc/scipy/reference/optimize.html",
                  "deterministic local and global optimizers (minimize_scalar, minimize, least_squares, differential_evolution)",
                  "BSD-3-Clause", notes=("PyPI classifier: BSD License",)),
            _tool("Optuna", "ALTERNATIVE", "https://github.com/optuna/optuna",
                  "sampler-based search (TPE, CMA-ES) over mixed continuous / categorical choices, e.g. which catalog lens", "MIT"),
            _tool("SALib", "COMPANION", "https://github.com/SALib/SALib",
                  "global sensitivity indices (Sobol, Morris) of a metric to assembly errors", "MIT"),
            _tool("BoTorch", "ALTERNATIVE", "https://botorch.org", "Bayesian optimization when each evaluation is expensive", "MIT"),
        ),
        opensci_provides=(
            "An editable Project document (ProjectDocumentV2): positions, orientations and declared model parameters addressed by "
            "instance_id / model_id.",
            "A deterministic run per evaluation: status, every target's value and status, beams at every monitor, power budget, "
            "with project_hash, engine_input_hash, result_hash and engine_build_id identifying it.",
            "Metric descriptors (unit, definition, sign, operands, validity) and the result field that carries each metric.",
            "Paraxial helper for a closed-form starting point: afocal and collimation spacing from the thick-lens matrices.",
        ),
        tool_computes=("The search or sampling strategy, the objective built from OpenSci results, the error distribution and its "
                       "statistics.",),
        entry_points=(RUN_PY, LOAD_RUN, RUN_CLI, RUN_JSON, METRICS_CLI, EXPORT_CLI, PARAXIAL_CLI,
                      E(kind="JSON_API", name="metric_descriptors", purpose="metric meanings and their result fields from a JSON host")),
        data=("ProjectDocumentV2 (os-optics schema project)", "run summary and simulation_result.json",
              "export kinds arrivals / monitors / paths / power-budget", "metric descriptors"),
        user_inputs=("The variables and their bounds, the objective and its weights, the error model (distributions, correlations) "
                     "of a tolerance study.",),
        assumptions=("Every evaluation is an independent run of the current Engine: the objective is as smooth as the physics it "
                     "reads (a beam clipped by an aperture or leaving the model scope changes status, not only value).",),
        limits=(
            "retained_dofs state the design intent for assembly (which adjustments the built system keeps); they are not optimizer "
            "variables. The optimizer edits nominal_pose or model parameters itself.",
            "A run whose status is UNKNOWN or OUT_OF_MODEL_SCOPE has no valid value for the affected metrics: treat it as infeasible, "
            "never as zero.",
            "Declared decimal fields are plain decimal strings without trailing zeros (write 3, not 3.0).",
        ),
        conventions=(RESULTS_CONVENTIONS, "metric descriptors: sign conventions and operand order of every metric"),
        runtime=("Runs locally with the Engine in the same Python environment (pip install os-optics scipy optuna salib).",
                 "Evaluations are independent: parallel workers each run their own Project copy."),
    ),
    InteropEntryV1(
        entry_id="scalar-diffraction-imaging",
        title="Scalar diffraction, PSF and spatial filtering with LightPipes / prysm / POPPY",
        direction="IMAGING_SCALAR_DIFFRACTION",
        status="VERIFIED_EXAMPLE",
        verified=(
            VerifiedRecordV1(tool="LightPipes", version="2.1.5", python="CPython 3.12.14", date="2026-10-15", companion_commit=COMPANION_COMMIT, engine_version="0.8.0",
                             example="cases/interop-diffraction/external_focus.py",
                             reference="Hankel integral of the truncated TEM00 field (scipy quad)",
                             outcome="WITHIN_TOLERANCE: focal peak within 9e-4 relative of the reference for stops of 3, 1.2 and 0.5 mm; sampling and window spread at most 1.8e-3"),
            VerifiedRecordV1(tool="prysm", version="0.21.1", python="CPython 3.12.14", date="2026-10-15", companion_commit=COMPANION_COMMIT, engine_version="0.8.0",
                             example="cases/interop-diffraction/external_focus.py",
                             reference="Hankel integral of the truncated TEM00 field (scipy quad)",
                             outcome="WITHIN_TOLERANCE: normalized focal profile within 4e-7 of the reference (prysm and POPPY "
                                     "compute the same discrete Fourier transform)"),
            VerifiedRecordV1(tool="poppy", version="1.2.0", python="CPython 3.12.14", date="2026-10-15", companion_commit=COMPANION_COMMIT, engine_version="0.8.0",
                             example="cases/interop-diffraction/external_focus.py",
                             reference="Hankel integral of the truncated TEM00 field (scipy quad)",
                             outcome="WITHIN_TOLERANCE: normalized focal profile within 4e-7 of the reference (prysm and POPPY "
                                     "compute the same discrete Fourier transform)"),
        ),
        scenario="Focal-plane irradiance of a truncated TEM00 beam, PSF and MTF of a stop and lens, pinhole spatial filtering: "
                 "diffraction the Engine's Gaussian and geometric models do not contain.",
        tools=(
            _tool("LightPipes", "RECOMMENDED", "https://opticspy.github.io/lightpipes/",
                  "Fresnel / Forvard propagation of a sampled scalar field through apertures and lenses", "BSD-3-Clause",
                  notes=("PyPI license field BSD-3-Clause; its classifier says MIT License.",)),
            _tool("prysm", "ALTERNATIVE", "https://prysm.readthedocs.io/en/stable/",
                  "pupil-to-focus propagation (matrix DFT), PSF and MTF from a pupil amplitude and phase", "MIT",
                  notes=("prysm 0.21 imports pkg_resources: install setuptools below version 81 in its environment.",)),
            _tool("POPPY", "ALTERNATIVE", "https://poppy-optics.readthedocs.io",
                  "Fraunhofer and Fresnel propagation with amplitude / OPD optical elements", "BSD-3-Clause", source=REPO),
            _tool("Diffractio", "ALTERNATIVE", "https://diffractio.readthedocs.io",
                  "scalar and vector diffraction, Rayleigh-Sommerfeld propagation", "GPLv3",
                  notes=("Diffractio 1.0.0 requires numpy below 2 and cannot share an environment with the Engine: exchange files.",)),
        ),
        opensci_provides=(
            "The beam state at an observation plane placed where the external propagation starts: complex q per axis, waist, "
            "curvature, M^2, Jones vector and basis, power, wavelength and medium, in the plane's (u, v) frame.",
            "The outline of a named element (stop, lens aperture) transformed into that plane's (u, v) frame.",
            "The stated TEM00 field-construction convention from q, so the sampled field the client builds has a defined sign of "
            "the curvature term and normalization.",
            "The power that passes each aperture (exact for a Gaussian beam) and the geometric wavefront (OPD) of real "
            "prescriptions from the pupil rays.",
        ),
        tool_computes=("The sampled complex field and its diffraction: focal-plane irradiance, PSF, MTF, encircled energy, "
                       "filtering by a pinhole.",),
        entry_points=(HANDOFF_PY, HANDOFF_CLI,
                      E(kind="CLI", name="os-optics export handoff", purpose="the handoff as a CSV / JSON Lines / JSON file"),
                      E(kind="PYTHON", name="os_optics.results.RunResults.wavefront",
                        purpose="geometric OPD of the pupil rays of a real prescription"),
                      RUN_PY, LOAD_RUN),
        data=("handoff export (opensci.transverse_handoff)", "wavefront export", "Project aperture models (APERTURE_STOP)"),
        user_inputs=("Sampling grid, window size and propagation method of the external code; a convergence check over both.",
                     "How an ideal thin lens is represented (an ideal quadratic phase is an assumption: the Engine models no phase "
                     "through it)."),
        assumptions=(
            "A TEM00 beam (M^2 = 1) is fully described by its q parameters, power and Jones vector: the client can construct its "
            "complex field under the stated convention. M^2 > 1 alone does not determine the modal content or the coherence of a "
            "beam, so no field is constructed for it.",
            "The Engine's irradiance and spot results are paraxial Gaussian quantities: they contain no diffraction at hard "
            "apertures.",
        ),
        limits=(
            "Intensity or power alone does not determine a complex field; OpenSci exports beam parameters, not a sampled field.",
            "Sparse pupil rays are samples of the geometric wavefront, not a sampled pupil field; fit Zernike terms of a stated "
            "order, never interpolate them as a field.",
            "After GAUSSIAN_TRUNCATION the transmitted Gaussian power is exact, while spot size and peak irradiance downstream are "
            "not diffraction-accurate: the handoff states it in its validity block. GAUSSIAN_SEVERE_TRUNCATION withholds the "
            "Gaussian state.",
            "Optical path through an ideal thin lens is not modelled (RAY_OPL_UNDEFINED for the wavefront).",
        ),
        conventions=(HANDOFF_CONVENTIONS, RESULTS_CONVENTIONS),
        runtime=("LightPipes, prysm and POPPY install next to the Engine (Python 3.12+).",
                 "Diffractio runs in its own environment (numpy < 2) and reads the handoff JSON file.",
                 SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="vector-doe-inverse-design",
        title="DOE / SLM, vector focusing and differentiable inverse design",
        direction="DOE_VECTOR_INVERSE_DESIGN",
        status="GUIDANCE_ONLY",
        scenario="Beam shaping with a diffractive element or SLM, focal fields of high-NA vector beams, phase retrieval and "
                 "end-to-end differentiable design around an OpenSci layout.",
        tools=(
            _tool("Diffractio", "RECOMMENDED", "https://diffractio.readthedocs.io", "scalar and vector (Ex, Ey, Ez) propagation, "
                  "DOE and SLM masks", "GPLv3", notes=("Requires numpy below 2: use a separate environment.",)),
            _tool("TorchOptics", "ALTERNATIVE", "https://github.com/matthewfilipovich/torchoptics",
                  "differentiable wave optics in PyTorch", "MIT"),
            _tool("Chromatix", "ALTERNATIVE", "https://github.com/chromatix-team/chromatix",
                  "differentiable wave optics in JAX", "MIT", source=REPO),
        ),
        opensci_provides=(
            "Public element geometry and pose, source parameters, and the beam at a plane with its stated frame, phase and "
            "polarization-basis conventions (handoff export).",
            "In the scalar low-NA domain, the focal field of a thin pupil mask (os-optics results pupil-field).",
        ),
        tool_computes=("Vector and high-NA fields, propagation beyond a focal plane, gradients for inverse design.",),
        entry_points=(HANDOFF_PY, HANDOFF_CLI, CATALOG_SHOW, LOAD_RUN),
        data=("handoff export", "public product documents"),
        user_inputs=("The DOE / SLM pixel pitch, phase levels and fill factor; the target field; the optimization settings.",),
        assumptions=("The beam entering the DOE is constructed from the exported q under the stated TEM00 convention.",),
        limits=("A run exports its beams' parameters, not their complex field: a field that cannot be constructed from them "
                "(multimode beams, measured beams) comes from measurement or another solver; a sampled field stated as a SAMPLED_FIELD "
                "source is propagated to a plane by os-optics results field (--field-out writes it as a sampled field).",
                "A DOE designed externally is not part of the Engine trace; its result stays in the external report."),
        conventions=(HANDOFF_CONVENTIONS,),
        runtime=("TorchOptics and Chromatix run next to the Engine on CPU or GPU; Diffractio needs its own environment.",
                 SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="cavity-interferometer",
        title="Optical cavities and complex interferometers with Finesse 3",
        direction="CAVITIES_INTERFEROMETERS",
        status="GUIDANCE_ONLY",
        scenario="Resonance, round-trip mode matching, error signals and quantum noise of cavities and interferometers whose "
                 "free-space layout is designed in OpenSci.",
        tools=(
            _tool("Finesse 3", "RECOMMENDED", "https://finesse.ifosim.org/docs/latest/",
                  "frequency-domain cavity and interferometer models: Hermite-Gauss modes, resonance, signals, quantum noise",
                  "GPL-3.0-or-later", notes=("PyPI classifier GPLv3+; read the license file of the version you install.",)),
        ),
        opensci_provides=(
            "Path topology, element positions and optical path lengths, wavelength, the beam q per axis at observation planes and "
            "the power reflectance / transmittance of mirrors and splitters.",
            "Thick-lens and mirror paraxial matrices (os-optics project paraxial).",
        ),
        tool_computes=("The cavity field: resonance, finesse, mode matching over many round trips, length and alignment signals.",),
        entry_points=(LOAD_RUN, PARAXIAL_CLI, CATALOG_SHOW, NETWORK_SCHEMA_CLI),
        data=("arrivals export (q, waist, curvature at planes)", "paths and segments exports (lengths, media)"),
        user_inputs=("Mirror amplitude reflectivities and their phases, losses, cavity tuning, modulation and readout scheme.",),
        assumptions=("Catalog coatings publish powers only; amplitude phases of mirrors are user inputs.",),
        limits=("The linear optical network export refuses resonators (CAVITY_NOT_SUPPORTED): a cavity is an infinite series of "
                "round trips that Finesse sums in closed form.",),
        conventions=(RESULTS_CONVENTIONS,),
        runtime=("Finesse 3 installs from PyPI or conda-forge in its own environment (Python above 3.10).", SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="quantum-photonics",
        title="Multi-photon interference with Perceval (Piquasso cross-check)",
        direction="QUANTUM_PHOTONICS",
        status="VERIFIED_EXAMPLE",
        verified=(
            VerifiedRecordV1(tool="perceval-quandela", version="1.3.1", python="CPython 3.12.14", date="2026-10-15", companion_commit=COMPANION_COMMIT, engine_version="0.8.0",
                             example="cases/interop-quantum/run_quantum.py",
                             reference="closed-form two-photon statistics of lossy 2x2 networks with detector efficiency and dark counts",
                             outcome="WITHIN_TOLERANCE: coincidence probabilities within 5e-8 of the reference over 21 cases (HOM, delay scan, lossy window, MZI)"),
            VerifiedRecordV1(tool="piquasso", version="8.0.1", python="CPython 3.12.14", date="2026-10-15", companion_commit=COMPANION_COMMIT, engine_version="0.8.0",
                             example="cases/interop-quantum/run_quantum.py",
                             reference="closed-form two-photon statistics of lossy 2x2 networks",
                             outcome="WITHIN_TOLERANCE: within 2e-9 of the reference; exported group delay within 2e-9 ps of an independent glass calculation"),
        ),
        scenario="Hong-Ou-Mandel interference, multi-photon interferometers and their loss, using the linear optical network of a "
                 "free-space layout designed in OpenSci.",
        tools=(
            _tool("Perceval", "RECOMMENDED", "https://perceval.quandela.net/docs/v1.3/getting_started.html",
                  "photonic circuit simulation: Unitary(Matrix(U)), partial distinguishability, source and detector noise models",
                  "MIT", notes=("Package perceval-quandela. Its required dependency exqalibur 1.4.2 reports the license "
                                "CC BY-NC-ND 4.0 in its PyPI metadata (checked 2026-10-15).",)),
            _tool("Piquasso", "CROSS_CHECK", "https://piquasso.readthedocs.io/en/latest/",
                  "Fock and Gaussian (CV) simulation: Interferometer(U), loss channels", "Apache-2.0"),
            _tool("QuTiP", "COMPANION", "https://qutip.org", "source dynamics (emitters, cavities) that produce the photon wavepackets",
                  "BSD-3-Clause"),
            _tool("NetSquid", "COMPANION", "https://netsquid.org", "quantum network and protocol simulation",
                  distribution="FREE_WITH_REGISTRATION",
                  notes=("Free to download and use after registering and accepting its terms of service (netsquid.org, "
                         "checked 2026-10-15).",)),
        ),
        opensci_provides=(
            "opensci-linear-optical-network/v1: input modes (each source, lab H / V), output modes (each beam on an output plane, "
            "lab H / V), the complex transfer matrix T with b_out = T a_in, column transmission and singular values (loss), every "
            "path's optical path length, group delay and Jones amplitudes, and the classical-limit check.",
            "T[row = output mode, column = input mode] is the orientation Perceval 1.3.1 Unitary(Matrix(U)) and Piquasso 8.0.1 "
            "Interferometer(U) use (U[out, in]).",
        ),
        tool_computes=("Multi-photon output probabilities, coincidence rates, HOM visibility, with the user's photon source and "
                       "detector models.",),
        entry_points=(NETWORK_JSON, NETWORK_CLI, NETWORK_SCHEMA_CLI,
                      E(kind="PYTHON", name="os_optics.local.network.export_linear_optical_network",
                        purpose="the network of a Project from Python")),
        data=("opensci-linear-optical-network/v1 (contract_hash over its canonical body)",),
        user_inputs=(
            "The quantum state of every source: photon number, spectral-temporal wavepacket, indistinguishability, g2, "
            "brightness.",
            "Detector models: efficiency, dark counts, photon-number resolution, timing.",
            "How a lossy or non-square T is completed: a unitary dilation with loss modes, or the simulator's loss channel.",
        ),
        assumptions=(
            "Quantum source and detector properties are user inputs: they are never inferred from the classical beam (power, "
            "coherence length, M^2).",
            "Path delays for wavepacket overlap use the exported group delay (dispersion of the glass on each path), not the "
            "phase delay.",
        ),
        limits=(
            "Only sources are input ports: an unused (vacuum) input has no column; the client completes T to a unitary when the "
            "simulator needs one (any completion of the vacuum columns is valid).",
            "Refused layouts: cavities (CAVITY_NOT_SUPPORTED), partially overlapping beams (PARTIAL_MODE_OVERLAP), truncated modes "
            "(MODE_TRUNCATED), ideal and tabulated thin lenses (PHASE_UNDEFINED_IDEAL_LENS), neutral-density attenuators "
            "(PHASE_UNDEFINED_ATTENUATOR), Faraday rotators (PHASE_UNDEFINED_FARADAY_ROTATOR), diffraction gratings (PHASE_UNDEFINED_GRATING), gain (NETWORK_NOT_PASSIVE).",
            "A coating that states its phases (a declared model's coating) uses them; every other interface follows the stated "
            "reflection models (catalog coatings publish powers only).",
        ),
        conventions=(NETWORK_CONVENTIONS,),
        runtime=("Perceval and Piquasso install next to the Engine (Python 3.11 to 3.14).",
                 "Piquasso with numba: give each run an empty NUMBA_CACHE_DIR.", SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="quantum-photonics-strawberry-fields",
        title="Strawberry Fields (archived): reproducing earlier work only",
        direction="QUANTUM_PHOTONICS",
        status="DEPRECATED",
        replaced_by="quantum-photonics",
        scenario="Re-running an earlier continuous-variable or photonic-circuit study written for Strawberry Fields.",
        tools=(
            _tool("Strawberry Fields", "HISTORICAL", "https://github.com/XanaduAI/strawberryfields",
                  "continuous-variable and photonic circuit simulation", "Apache-2.0", source=REPO, maintenance="ARCHIVED",
                  notes=("The upstream repository is archived (read-only).",)),
        ),
        opensci_provides=("The same linear optical network as the quantum-photonics entry.",),
        tool_computes=("The circuit simulation of the earlier study.",),
        entry_points=(NETWORK_JSON, NETWORK_CLI),
        data=("opensci-linear-optical-network/v1",),
        user_inputs=("As for quantum-photonics: every quantum source and detector property.",),
        limits=("Not recommended for new work: use Perceval or Piquasso (entry quantum-photonics).",),
        conventions=(NETWORK_CONVENTIONS,),
        runtime=("Install the archived release in its own environment pinned to the dependency versions it was released with.",),
    ),
    InteropEntryV1(
        entry_id="micro-nano-pic",
        title="Micro- and nano-photonic devices, gratings and photonic integrated circuits",
        direction="MICRO_NANO_PIC",
        status="GUIDANCE_ONLY",
        scenario="Full-wave simulation of a local device in a free-space layout (grating coupler, fibre or chip facet, "
                 "sub-wavelength structure) and circuit-level PIC models.",
        tools=(
            _tool("Meep", "RECOMMENDED", "https://meep.readthedocs.io/en/stable/", "FDTD of local devices, mode decomposition, "
                  "S-parameters", "GPL-2.0", source=REPO, notes=("Install from conda-forge (pymeep); the PyPI package named meep is "
                                                                 "a different project.",)),
            _tool("Tidy3D", "ALTERNATIVE", "https://www.flexcompute.com/tidy3d/", "cloud FDTD; the Python client prepares and reads "
                  "simulations", "LGPL-2.1", source=REPO, notes=("The Python client is open source; the solver runs as a paid cloud "
                                                                 "service of the vendor.",)),
            _tool("grcwa", "ALTERNATIVE", "https://github.com/weiliangjinca/grcwa", "RCWA of periodic structures", "GPLv3"),
            _tool("torcwa", "ALTERNATIVE", "https://github.com/kch3782/torcwa", "RCWA on PyTorch (GPU)", "LGPL"),
            _tool("gdsfactory", "COMPANION", "https://gdsfactory.github.io/gdsfactory/", "PIC layout (GDS)", "MIT", source=REPO),
            _tool("SAX", "COMPANION", "https://github.com/gdsfactory/sax", "S-parameter circuit simulation", "Apache-2.0"),
            _tool("PhotonForge", "ALTERNATIVE", "https://www.flexcompute.com/photonforge/", "PIC design and simulation platform",
                  maintainer="COMMERCIAL_VENDOR", distribution="COMMERCIAL"),
            _tool("Ansys Lumerical", "ALTERNATIVE", "https://www.ansys.com/products/optics", "FDTD, mode and circuit solvers",
                  maintainer="COMMERCIAL_VENDOR", distribution="COMMERCIAL"),
        ),
        opensci_provides=(
            "The incident beam at the device's reference plane: q, waist, curvature, Jones vector and basis, power and direction "
            "in the plane's (u, v, n) frame (handoff export).",
        ),
        tool_computes=("The device's local complex response: transmitted / reflected fields, diffraction orders, port "
                       "S-parameters, coupling efficiency.",),
        entry_points=(HANDOFF_PY, HANDOFF_CLI, LOAD_RUN),
        data=("handoff export",),
        user_inputs=("The device structure, materials at the device scale, simulation domain, mesh and boundaries.",),
        assumptions=("Full-wave solvers are for the local device only; the free-space layout stays in OpenSci.",),
        limits=("A mode-port S-matrix and a sampled free-space field are different interfaces: an S-parameter of a waveguide port "
                "does not describe a free-space beam without the port's mode definition and its overlap with the beam.",
                "Device results stay in the external report; they do not enter the Engine trace."),
        conventions=(HANDOFF_CONVENTIONS,),
        runtime=("Meep runs on Linux or macOS (on Windows through WSL); Tidy3D and the commercial tools need vendor accounts.",
                 SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="metasurface-virtual-element",
        title="Metasurface lens response as a validated virtual focusing element",
        direction="METASURFACE_VIRTUAL_ELEMENT",
        status="VERIFIED_EXAMPLE",
        verified=(
            VerifiedRecordV1(tool="grcwa", version="0.1.2", python="CPython 3.12.14", date="2026-10-15", companion_commit=COMPANION_COMMIT, engine_version="0.8.0",
                             example="cases/interop-metasurface/step4_engine_check.py",
                             reference="grcwa unit-cell library with local-periodic angular-spectrum focusing (convergence spread included)",
                             outcome="WITHIN_TOLERANCE: held-out focus, spot and power inside the model's stated errors; outside its wavelength and field-angle domain the Engine reports OUT_OF_MODEL_SCOPE"),
        ),
        scenario="Turn an externally simulated metalens response into a Project optical model (TABULATED_THIN_LENS) with its "
                 "wavelength table, validated input domain and per-metric error, and design the rest of the system around it.",
        tools=(
            _tool("grcwa", "RECOMMENDED", "https://github.com/weiliangjinca/grcwa", "meta-atom library: complex zeroth-order "
                  "transmission versus pillar size and wavelength (RCWA)", "GPLv3"),
            _tool("meent", "ALTERNATIVE", "https://github.com/kc-ml2/meent", "RCWA with NumPy, JAX or PyTorch backends", "MIT",
                  source=REPO),
            _tool("DFlat", "ALTERNATIVE", "https://github.com/DeanHazineh/DFlat", "differentiable metasurface design with neural "
                  "meta-atom models", "MIT", source=REPO,
                  notes=("The PyPI package dflat-opt 4.3 is classified GPLv3 while the repository license file is MIT.",
                         "Its gdspy dependency needs a C++ compiler on Windows: install with --no-deps or use Linux.")),
            _tool("metabox", "ALTERNATIVE", "https://github.com/Luochenghuang/metabox", "metasurface design and system modelling",
                  "MIT", notes=("Runs with Python 3.8 and TensorFlow 2.11: use a separate environment and exchange files.",)),
            _tool("torcwa", "ALTERNATIVE", "https://github.com/kch3782/torcwa", "RCWA on PyTorch (GPU)", "LGPL"),
        ),
        opensci_provides=(
            "The TABULATED_THIN_LENS Project model: per-wavelength focal length and transmittance (the fraction delivered into the "
            "focused beam), interpolation rules, clear aperture, maximum field angle and beam size, validated polarizations, "
            "supported metrics with their validated error, and the evidence (tools, versions, data and configuration hashes, fit "
            "model, validation summary).",
            "Run-time scope checks: a wavelength outside the table, a field angle or beam size beyond the validated domain, an "
            "unvalidated polarization or a metric the model was not validated for gives OUT_OF_MODEL_SCOPE instead of a number.",
            "The power the model does not deliver into the focused beam is a named sink in the power budget.",
        ),
        tool_computes=("The meta-atom library, the metalens field (local periodic approximation and angular-spectrum propagation "
                       "or full-wave), the focal length and efficiency per wavelength, and the fit with its hold-out errors.",),
        entry_points=(E(kind="JSON_API", name="build_native_model", purpose="native class lens.tabulated_thin: the table (rows or "
                          "columns), domain, metrics and evidence as a TABULATED_THIN_LENS model and a component, optionally added to "
                          "a Project (options.into)"),
                      E(kind="CLI", name="os-optics native build lens.tabulated_thin",
                        purpose="the same from a shell (--param rows=<JSON> ... --into project.json)"),
                      RUN_PY, RUN_CLI, LOAD_RUN, E(kind="CLI", name="os-optics schema project",
                                                   purpose="the Project schema, including the optical model kinds")),
        data=("ProjectDocumentV2 optical_models (model_kind TABULATED_THIN_LENS)", "arrivals export (waist position and radius)"),
        user_inputs=("The meta-atom structure and materials, the simulation settings and their convergence, the validation "
                     "wavelengths, field angles and beam sizes.",),
        assumptions=(
            "One external output (a spot, a focal length, an efficiency for one input) does not define the element for other "
            "inputs: the model holds only inside the wavelengths, field angles, beam sizes and polarizations it was validated for.",
            "Amplitude and power are kept apart: transmittance is a power fraction; a complex amplitude is t = sqrt(T) "
            "exp(i phi).",
        ),
        limits=("The model carries focal power and transmittance, not a phase: interference and the linear optical network through "
                "it are refused as for an ideal thin lens (PHASE_UNDEFINED_IDEAL_LENS).",
                "PSF, side lobes, high-NA focusing and wide-field image quality are outside a focusing equivalent: compute them "
                "with the external solver."),
        conventions=(RESULTS_CONVENTIONS, "TABULATED_THIN_LENS field docstrings (os-optics schema project)"),
        runtime=("grcwa and meent run next to the Engine; DFlat and metabox need their own environments.", SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="coatings-materials-polarization",
        title="Thin-film coatings, material data and partial polarization",
        direction="COATINGS_MATERIALS_POLARIZATION",
        status="GUIDANCE_ONLY",
        scenario="Design or check a coating stack, compare material dispersion data, analyse depolarizing elements (Mueller "
                 "calculus) that OpenSci's Jones calculus does not model.",
        tools=(
            _tool("tmm", "RECOMMENDED", "https://github.com/sbyrnes321/tmm", "transfer-matrix reflectance, transmittance and "
                  "phase of thin-film stacks per s / p and angle", "MIT"),
            _tool("PyMoosh", "ALTERNATIVE", "https://github.com/AnMoreau/PyMoosh", "multilayer optics, including absorbing and "
                  "guided-mode cases", "MIT"),
            _tool("refractiveindex.info", "COMPANION", "https://refractiveindex.info", "published refractive-index data sets with "
                  "their sources"),
            _tool("py_pol", "COMPANION", "https://py-pol.readthedocs.io", "Jones, Stokes and Mueller calculus, depolarization",
                  "MIT", notes=("PyPI package py-pol.",)),
        ),
        opensci_provides=(
            "Public coating models (R / T per band and angle of incidence, s / p when published), material dispersion models and "
            "their valid bands, the Jones vector and its basis at every monitor.",
            "Project-declared coatings and materials, so an externally designed stack can be declared as a coating model of the "
            "Project.",
        ),
        tool_computes=("Coating spectra and phases from a layer stack, fitted dispersion, Mueller-matrix and depolarization "
                       "analysis.",),
        entry_points=(CATALOG_SHOW, MODEL_FROM_CATALOG, LOAD_RUN,
                      E(kind="JSON_API", name="catalog_coatings", purpose="the published coating models"),
                      E(kind="JSON_API", name="catalog_materials", purpose="the published material models")),
        data=("public catalog coatings and materials", "arrivals export (Jones, Stokes)"),
        user_inputs=("Layer thicknesses and materials of a stack; measured data and its uncertainty.",),
        assumptions=("The Engine's polarization is Jones calculus: a partially polarized or unpolarized source is traced as two "
                     "incoherent orthogonal components whose Stokes parameters add; depolarization by an element is an "
                     "external analysis.",),
        limits=("An external coating or material result is evaluated on its own; it never replaces a catalog fact. To simulate "
                "it, declare it in the Project as your own coating or material.",
                "Power reflectance alone does not give the reflection phase."),
        conventions=(RESULTS_CONVENTIONS,),
        runtime=("tmm, PyMoosh and py_pol install next to the Engine.", SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="cad-brep-geometry",
        title="Exact CAD geometry from public optical data (USD BRep / SMLib, smlibpy)",
        direction="CAD_BREP_GEOMETRY",
        status="GUIDANCE_ONLY",
        scenario="Build exact B-rep solids of the public optical elements of a Project in a CAD kernel, check dimensions, units, "
                 "poses, normals and bounding boxes, and exchange them as OpenUSD or STEP.",
        tools=(
            _tool("NVIDIA USD BRep / SMLib", "RECOMMENDED", "https://github.com/NVIDIA-Omniverse/usd-brep",
                  "NURBS B-rep solid modelling with C++ and Python APIs and OpenUSD import / export (first public release 1.0.0 on "
                  "2026-10-08)", "Apache-2.0", source=REPO,
                  notes=("Official NVIDIA library. Its KNOWN_ISSUES lists cases where healing reports success incorrectly and "
                         "tessellation normals or areas / volumes lose accuracy: check the resulting geometry numerically.",)),
            _tool("smlibpy", "ALTERNATIVE", "https://github.com/awv-informatik/smlibpy",
                  "Python modelling library and MCP server on the SMLib kernel, with STEP export", "MIT AND Apache-2.0 AND OFL-1.1",
                  maintainer="THIRD_PARTY_WRAPPER",
                  notes=("Third-party package by awv informatik, not an NVIDIA package. Its STEP support is that package's own and is "
                         "checked per version.", "Repository license: MIT (GitHub); PyPI reports MIT AND Apache-2.0 AND OFL-1.1.")),
        ),
        opensci_provides=(
            "Public surface prescriptions (radii, conics, aspheric terms, cylinder axes), clear and outer apertures, thicknesses, "
            "element frames and the component poses (millimetres, design frame, unit quaternions), with each component's "
            "instance_id and the product or model it uses.",
            "The exact meridional section of every element (json_api optical_section) to compare a constructed solid against.",
        ),
        tool_computes=("Exact B-rep solids and assemblies, geometric checks, OpenUSD / STEP files.",),
        entry_points=(CATALOG_SHOW, MODEL_FROM_CATALOG,
                      E(kind="JSON_API", name="optical_section", purpose="the exact meridional section of an element"),
                      E(kind="CLI", name="os-optics project solids", purpose="the optical-solid check of a Project's bodies")),
        data=("public product documents", "ProjectDocumentV2 (poses, declared models)"),
        user_inputs=("Modelling tolerance, the target CAD format and the mechanical parts the user adds.",),
        assumptions=("A B-rep holds geometry only: materials, dispersion, coatings and wavelength / polarization behaviour stay in the "
                     "public optical data next to it.",),
        limits=(
            "A display mesh or an unchecked B-rep is not an optical model: no simulation model is derived from CAD geometry.",
            "Only public, customer-visible geometry is exchanged; mechanical parts data and protected assembly packages stay outside "
            "this workflow.",
            "A geometric check of the CAD kernel is not an optical or physical acceptance.",
        ),
        conventions=("Design frame in millimetres, +Z up; element frames per the public catalog's frame rules (OpticalFrameV1).",),
        runtime=("Install the CAD libraries in their own environment; the exchange is through files.", SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="adaptive-optics",
        title="Adaptive optics and wavefront sensing with HCIPy",
        direction="SPECIALIST",
        status="GUIDANCE_ONLY",
        scenario="Wavefront sensing and correction, atmospheric turbulence and coronagraphy for an optical train laid out in "
                 "OpenSci.",
        tools=(_tool("HCIPy", "RECOMMENDED", "https://hcipy.org", "wavefront sensors, deformable mirrors, turbulence, "
                     "coronagraphs", "MIT"),),
        opensci_provides=("Geometric wavefront (OPD) of real prescriptions at a plane, pupil geometry and the beam state "
                          "(handoff export).",),
        tool_computes=("Sensor signals, control loops, corrected PSF.",),
        entry_points=(HANDOFF_PY, E(kind="PYTHON", name="os_optics.results.RunResults.wavefront",
                                    purpose="geometric OPD per pupil ray")),
        data=("wavefront export", "handoff export"),
        user_inputs=("Turbulence model, sensor and deformable-mirror geometry, control parameters.",),
        limits=("The exported wavefront is a sparse set of pupil-ray OPD values without diffraction; fit modes of a stated order.",),
        conventions=(RESULTS_CONVENTIONS,),
        runtime=("Installs next to the Engine.", SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="multiphysics-stray-light",
        title="Thermo-optical, stray-light and commercial lens-design verification",
        direction="SPECIALIST",
        status="GUIDANCE_ONLY",
        scenario="Thermal lensing and stress, scattered and ghost light, and cross-checks of a design in commercial optical "
                 "software.",
        tools=(
            _tool("COMSOL Multiphysics", "ALTERNATIVE", "https://www.comsol.com", "coupled thermal, structural and wave-optics models",
                  maintainer="COMMERCIAL_VENDOR", distribution="COMMERCIAL"),
            _tool("Ansys Zemax OpticStudio", "ALTERNATIVE", "https://www.ansys.com/products/optics", "sequential and non-sequential "
                  "lens design and analysis", maintainer="COMMERCIAL_VENDOR", distribution="COMMERCIAL"),
            _tool("FRED", "ALTERNATIVE", "https://photonengr.com", "non-sequential stray-light and illumination analysis",
                  maintainer="COMMERCIAL_VENDOR", distribution="COMMERCIAL"),
            _tool("TracePro", "ALTERNATIVE", "https://lambdares.com", "non-sequential ray tracing and stray light",
                  maintainer="COMMERCIAL_VENDOR", distribution="COMMERCIAL"),
        ),
        opensci_provides=("Public prescriptions, apertures, materials, coatings and poses; the paths and power budget of the run, "
                          "including untraced ghost reflections listed as UNTRACED_REFLECTION.",),
        tool_computes=("Thermal and structural fields, scatter and ghost analysis, an independent lens-design evaluation.",),
        entry_points=(CATALOG_SHOW, MODEL_FROM_CATALOG, LOAD_RUN),
        data=("public product documents", "power-budget and paths exports"),
        user_inputs=("Absorbed power, thermal and mechanical properties, mounting, surface scatter models; the licenses of the "
                     "commercial tools.",),
        limits=("Scattering, thermal effects and damage are outside the Engine's model; their results stay in the external report.",),
        conventions=(RESULTS_CONVENTIONS,),
        runtime=("Commercial software runs under the user's own license, locally or in the vendor's cloud.", SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="nonlinear-ultrafast-communications",
        title="Pulse propagation and optical communication links",
        direction="SPECIALIST",
        status="GUIDANCE_ONLY",
        scenario="Nonlinear pulse propagation in fibres and optical communication link budgets fed by a free-space front end "
                 "designed in OpenSci.",
        tools=(
            _tool("gnlse-python", "RECOMMENDED", "https://github.com/WUST-FOG/gnlse-python", "generalized nonlinear Schroedinger "
                  "equation in fibres", "MIT", notes=("PyPI package gnlse.",)),
            _tool("PyNLO (original)", "HISTORICAL", "https://github.com/pyNLO/PyNLO", "nonlinear optics of pulses in fibres and "
                  "crystals", "GPL-3.0", source=REPO, maintenance="LEGACY_ENVIRONMENT",
                  notes=("Its README states that it requires Python 2 and is tested on Python 2.7: run it in a dedicated legacy "
                         "environment.",)),
            _tool("OptiCommPy", "ALTERNATIVE", "https://github.com/edsonportosilva/OptiCommPy", "optical communication system "
                  "simulation", "GPL-3.0", source=REPO),
        ),
        opensci_provides=("The beam delivered to a fibre facet or receiver plane (handoff export), power after the free-space "
                          "optics, per-path group delay in the linear optical network export.",),
        tool_computes=("Spectral and temporal pulse evolution, link performance.",),
        entry_points=(HANDOFF_PY, NETWORK_CLI, LOAD_RUN),
        data=("handoff export", "opensci-linear-optical-network/v1"),
        user_inputs=("Pulse shape and energy, fibre parameters (dispersion, nonlinearity, mode field), modulation format.",),
        limits=("Coupling into a fibre needs the fibre's mode definition; the Engine reports the beam, not a coupling efficiency.",),
        conventions=(HANDOFF_CONVENTIONS,),
        runtime=("gnlse-python and OptiCommPy install next to the Engine; the original PyNLO needs a Python 2.7 environment.",
                 SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="scattering-media",
        title="Scattering media and particles (Monte Carlo, Mie)",
        direction="SPECIALIST",
        status="GUIDANCE_ONLY",
        scenario="Light transport in tissue or turbid samples and scattering by particles, illuminated by a beam designed in "
                 "OpenSci.",
        tools=(
            _tool("MCX", "RECOMMENDED", "https://mcx.space", "GPU Monte Carlo photon transport in voxelized media", "GPLv3+",
                  notes=("PyPI package pmcx (the Python interface).",)),
            _tool("miepython", "COMPANION", "https://github.com/scottprahl/miepython", "Mie scattering of spheres", "MIT"),
        ),
        opensci_provides=("The incident beam at the sample plane: position, direction, waist, curvature, power (handoff export).",),
        tool_computes=("Fluence and scattered light in the medium.",),
        entry_points=(HANDOFF_PY, HANDOFF_CLI),
        data=("handoff export",),
        user_inputs=("Optical properties of the medium (absorption, scattering, anisotropy, index) and the geometry of the sample.",),
        limits=("Scattering is outside the Engine's model; the result stays in the external report.",),
        conventions=(HANDOFF_CONVENTIONS,),
        runtime=("MCX needs a GPU for its fast path.", SEPARATE_RESULTS),
    ),
    InteropEntryV1(
        entry_id="laboratory-instruments",
        title="Laboratory measurements compared with a design (PyVISA / PyLabLib)",
        direction="SPECIALIST",
        status="GUIDANCE_ONLY",
        scenario="Control power meters, cameras and stages to measure a built setup and compare the measurements with the run "
                 "that designed it.",
        tools=(
            _tool("PyVISA", "RECOMMENDED", "https://pyvisa.readthedocs.io", "instrument control over VISA (GPIB, USB, Ethernet)",
                  "MIT"),
            _tool("PyLabLib", "ALTERNATIVE", "https://pylablib.readthedocs.io", "drivers for cameras, stages and lab devices",
                  "GPLv3"),
        ),
        opensci_provides=("The predicted values with their identity (targets, beams at monitors, power budget) to compare against.",),
        tool_computes=("Instrument control and data acquisition.",),
        entry_points=(LOAD_RUN, EXPORT_CLI),
        data=("arrivals, monitors and power-budget exports",),
        user_inputs=("Instruments, calibration and measurement uncertainty.",),
        limits=("A measurement is evidence about the built setup; it does not change the run or its Local Validation.",),
        conventions=(RESULTS_CONVENTIONS,),
        runtime=("Runs on the lab computer next to the instruments.", SEPARATE_RESULTS),
    ),
)

__all__ = ["CHECKED", "ENTRIES"]
