"""The view fails closed when its inputs do not describe one optical system (moved from the earlier
``opensci-optical-view`` package; scene schema 2.0). A catalog element is bound to the Engine's own compilation of the
public model the run lowered (``os_optics.local.compile.compile_element``): any other surface, interaction,
aperture, medium or material is refused, whatever the product category."""

from __future__ import annotations

import pytest
from optical_view_reference import folded_project, lowered_request, polarization_project
from optical_view_reference import catalog, engine_request, engine_result, engine_run, project, ray_paths, scene  # noqa: F401  (pytest fixtures)

from os_optics import trace_ray_paths, validate, validate_with_ray_paths
from os_optics._vendor.shared_contracts.common.geometry import PoseV1
from os_optics._vendor.shared_contracts.optical_design import ProjectDocumentV2
from os_optics.contracts import AbsorberInteraction, CircularAperture, PlaneSurface
from os_optics.optical_view import OpticalViewBindingError, build_optical_simulation_scene


def test_result_of_another_request_is_refused(project, catalog, engine_request):
    other = engine_request.model_copy(update={"configuration": engine_request.configuration.model_copy(update={"random_seed": 7})})
    with pytest.raises(OpticalViewBindingError, match="different Engine inputs"):
        build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request, engine_result=validate(other))


def test_engine_pose_that_differs_from_the_project_is_refused(project, catalog, engine_request, engine_result):
    shifted = tuple(c.model_copy(update={"nominal_pose": PoseV1(position_mm=(0.5, 0.0, 0.0), quaternion_wxyz=c.nominal_pose.quaternion_wxyz)})
                    if c.instance_id == "L1" else c for c in project.components)
    moved = ProjectDocumentV2.model_validate(project.model_copy(update={"components": shifted}).payload())
    with pytest.raises(OpticalViewBindingError, match="component 'L1'.*pose"):
        build_optical_simulation_scene(project=moved, catalog=catalog, engine_request=engine_request, engine_result=engine_result)


def test_project_pinned_to_another_catalog_release_is_refused(project, catalog, engine_request, engine_result):
    pinned = tuple(c.model_copy(update={"product": c.product.model_copy(update={"catalog_version": "2025-01-01.1"})}) if c.product else c
                   for c in project.components)
    other = ProjectDocumentV2.model_validate(project.model_copy(update={"components": pinned}).payload())
    with pytest.raises(OpticalViewBindingError, match="designed against catalog"):
        build_optical_simulation_scene(project=other, catalog=catalog, engine_request=engine_request, engine_result=engine_result)


def test_untraceable_request_is_refused(project, catalog, engine_request):
    off = engine_request.model_copy(update={"configuration": engine_request.configuration.model_copy(update={"run_physics": False})})
    with pytest.raises(OpticalViewBindingError, match="could not trace"):
        build_optical_simulation_scene(project=project, catalog=catalog, engine_request=off)


def test_ray_paths_of_another_run_are_refused(project, catalog, engine_request, engine_result):
    other = engine_request.model_copy(update={"configuration": engine_request.configuration.model_copy(update={"random_seed": 7})})
    with pytest.raises(OpticalViewBindingError, match="different Engine inputs"):
        build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request, engine_result=engine_result,
                                       ray_paths=trace_ray_paths(other))


def test_ray_paths_without_their_result_are_refused(project, catalog, engine_request, ray_paths):
    with pytest.raises(OpticalViewBindingError, match="without the Engine result"):
        build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request, ray_paths=ray_paths)


def test_every_way_of_supplying_one_run_gives_the_same_scene(project, catalog, engine_request, engine_result, scene):
    """Result + ray paths of one execution (the local runtime), result only (rays re-recorded and hash-bound), or
    nothing (one execution here) all draw the identical scene."""
    only_result = build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request, engine_result=engine_result)
    nothing = build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request)
    assert only_result == scene == nothing
    result, rays = validate_with_ray_paths(engine_request)
    assert result.model_dump_json() == validate(engine_request).model_dump_json() == engine_result.model_dump_json()


def test_a_project_that_differs_from_the_engine_request_is_refused(catalog, project, engine_request, engine_result):
    fewer = ProjectDocumentV2.model_validate(project.model_copy(update={"observations": project.observations[:1]}).payload())
    with pytest.raises(OpticalViewBindingError, match="different observation planes"):
        build_optical_simulation_scene(project=fewer, catalog=catalog, engine_request=engine_request, engine_result=engine_result)


def test_an_unknown_model_kind_selection_is_refused(project, catalog, engine_request, engine_result):
    with pytest.raises(OpticalViewBindingError, match="match model_kind='POLARIZATION_ELEMENT'"):
        build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request, engine_result=engine_result,
                                       model_kind="POLARIZATION_ELEMENT")


# ----------------------------------------------------------------------------- adversarial-review regressions


def _with_component(request, instance_id, **update):
    components = tuple(c.model_copy(update=update) if c.instance_id == instance_id else c for c in request.project.components)
    return request.model_copy(update={"project": request.project.model_copy(update={"components": components})})


def _surfaces(request, instance_id):
    (component,) = [c for c in request.project.components if c.instance_id == instance_id]
    return component.surfaces


def _refused(project, catalog, request, match):
    with pytest.raises(OpticalViewBindingError, match=match):
        build_optical_simulation_scene(project=project, catalog=catalog, engine_request=request, engine_result=validate(request))


def test_a_hidden_blocker_surface_inside_a_catalog_element_is_refused(project, catalog, engine_request):
    ring = PlaneSurface(surface_id="ring", origin_mm=(0.0, 0.0, 1.0), normal=(0.0, 0.0, 1.0),
                        aperture=CircularAperture(radius_mm=12.0, obstruction_radius_mm=1.0), interaction=AbsorberInteraction())
    request = _with_component(engine_request, "L1", surfaces=(*_surfaces(engine_request, "L1"), ring))
    _refused(project, catalog, request, r"surfaces \[.*'ring'\] instead of the compiled SURFACE_SEQUENCE surfaces")


def test_a_missing_or_reordered_surface_is_refused(project, catalog, engine_request):
    surfaces = _surfaces(engine_request, "L1")
    _refused(project, catalog, _with_component(engine_request, "L1", surfaces=surfaces[:-1]), "instead of the compiled")
    _refused(project, catalog, _with_component(engine_request, "L1", surfaces=tuple(reversed(surfaces))), "instead of the compiled")


def test_an_element_that_is_not_the_compiled_public_model_is_refused(project, catalog, engine_request):
    surfaces = list(_surfaces(engine_request, "L1"))
    first = surfaces[0]

    def replaced(**update):
        return _with_component(engine_request, "L1", surfaces=(first.model_copy(update=update), *surfaces[1:]))

    if first.geometry == "sphere":
        _refused(project, catalog, replaced(radius_mm=first.radius_mm * 2), r"surface S1 differs \(radius_mm")
        _refused(project, catalog, replaced(aperture_radius_mm=first.aperture_radius_mm / 2), r"surface S1 differs \(aperture_radius_mm")
    else:
        _refused(project, catalog, replaced(origin_mm=(0.0, 0.0, 0.5)), r"surface S1 differs \(origin_mm")
        _refused(project, catalog, replaced(aperture=CircularAperture(radius_mm=1.2)), r"surface S1 differs \(aperture")
    _refused(project, catalog, replaced(mount_margin_mm=first.mount_margin_mm + 2.0), r"surface S1 differs \(mount_margin_mm")
    _refused(project, catalog, replaced(interaction=AbsorberInteraction()), r"surface S1 differs \(interaction")
    _refused(project, catalog, replaced(medium_positive="air", medium_negative="air"), r"surface S1 differs \(medium_")


def test_a_catalog_material_that_is_not_the_published_one_is_refused(project, catalog, engine_request):
    materials = engine_request.project.materials
    if not materials:
        pytest.skip("the reference lens refracts through no catalog material")
    # every material of the request: the environment's air (stated first) and the catalog material the lens is made of
    assert materials[0].material_id == "air" and len(materials) > 1
    for k, first in enumerate(materials[:2]):
        others = materials[:k] + materials[k + 1:]
        altered = first.model_copy(update={"valid_wavelength_nm": (first.valid_wavelength_nm[0], first.valid_wavelength_nm[1] + 1.0)})
        changed = materials[:k] + (altered,) + materials[k + 1:]
        request = engine_request.model_copy(update={"project": engine_request.project.model_copy(update={"materials": changed})})
        _refused(project, catalog, request, f"material {first.material_id!r} differs")
        # a request that lost the material: the Engine cannot trace it or the view finds it undefined; either way no scene
        request = engine_request.model_copy(update={"project": engine_request.project.model_copy(update={"materials": others})})
        with pytest.raises(OpticalViewBindingError, match=f"could not trace|material {first.material_id!r} is not defined"):
            build_optical_simulation_scene(project=project, catalog=catalog, engine_request=request, engine_result=validate(request))


def test_path_ids_cannot_carry_lowering_chosen_names(project, catalog, engine_request):
    (src,) = [c for c in engine_request.project.components if c.instance_id == "SRC"]
    renamed = src.source.model_copy(update={"source_id": "ACME-XQ412-B"})
    _refused(project, catalog, _with_component(engine_request, "SRC", source=renamed), "source_id 'SRC'")


def test_only_project_sources_emit(project, catalog, engine_request):
    (src,) = [c for c in engine_request.project.components if c.instance_id == "SRC"]
    stray = src.source.model_copy(update={"source_id": "L1"})
    _refused(project, catalog, _with_component(engine_request, "L1", source=stray), "only a Project source may emit")


def test_an_altered_engine_result_is_refused(project, catalog, engine_request, engine_result):
    obs = engine_result.observations[0]
    arrival = obs.arrivals[0].model_copy(update={"uv_mm": (9.0, 9.0)})
    forged = engine_result.model_copy(update={"observations": (obs.model_copy(update={"arrivals": (arrival,)}), *engine_result.observations[1:])})
    with pytest.raises(OpticalViewBindingError, match="result_hash"):
        build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request, engine_result=forged)


def test_an_absorbing_engine_plane_the_project_does_not_declare_is_refused(project, catalog, engine_request):
    observations = tuple(o.model_copy(update={"absorbing": True}) if o.observation_id == "focal" else o for o in engine_request.project.observations)
    request = engine_request.model_copy(update={"project": engine_request.project.model_copy(update={"observations": observations})})
    _refused(project, catalog, request, "absorbing flag")


def test_a_path_that_ends_at_a_surface_stops_there(project, catalog, engine_request):
    """An absorbing target plane (declared in the Project) ends the beam: the chief ray stops at it, like its pupil rays."""
    observations = tuple(o.model_copy(update={"absorbing": True}) if o.observation_id == "focal" else o for o in project.observations)
    absorbing_project = ProjectDocumentV2.model_validate(project.model_copy(update={"observations": observations}).payload())
    engine_obs = tuple(o.model_copy(update={"absorbing": True}) if o.observation_id == "focal" else o for o in engine_request.project.observations)
    request = engine_request.model_copy(update={"project": engine_request.project.model_copy(update={"observations": engine_obs})})
    scene = build_optical_simulation_scene(project=absorbing_project, catalog=catalog, engine_request=request)
    (path,) = scene.beam_paths
    assert path.termination == "DETECTED" and path.chief_ray.end.value == "STOPPED_AT_SURFACE"
    assert path.chief_ray.display_extension_to_mm is None
    assert all(r.end.value == "STOPPED_AT_SURFACE" and r.display_extension_to_mm is None for r in path.pupil_rays)


@pytest.fixture(scope="module")
def folded(catalog):
    project = folded_project(catalog)
    return project, lowered_request(project, catalog)


@pytest.fixture(scope="module")
def polarization(catalog):
    project = polarization_project(catalog)
    return project, lowered_request(project, catalog)


@pytest.mark.parametrize(("design", "instance_id"), [("folded", "M1"), ("folded", "W1"), ("folded", "BS"), ("folded", "F1"),
                                                     ("polarization", "WP"), ("polarization", "POL")])
def test_every_category_is_bound_to_its_compiled_public_model(design, instance_id, catalog, request):
    """Mirror, window, splitter, filter, waveplate and polarizer elements are bound the same generic way as a lens."""
    project, engine_request = request.getfixturevalue(design)
    build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request)  # the lowered design draws
    surfaces = _surfaces(engine_request, instance_id)
    last = surfaces[-1]
    lossy = last.model_copy(update={"interaction": AbsorberInteraction()})
    _refused(project, catalog, _with_component(engine_request, instance_id, surfaces=(*surfaces[:-1], lossy)),
             rf"component '{instance_id}' .* surface {last.surface_id} differs \(interaction")
    blocker = PlaneSurface(surface_id="S9", origin_mm=(0.0, 0.0, -0.5), normal=(0.0, 0.0, 1.0), aperture=CircularAperture(radius_mm=5.0),
                           interaction=AbsorberInteraction())
    _refused(project, catalog, _with_component(engine_request, instance_id, surfaces=(*surfaces, blocker)), "instead of the compiled")


# ---- the scene of an evaluation's own result -------------------------------------------------------------------------------------
def _forged(engine_result):
    obs = engine_result.observations[0]
    arrival = obs.arrivals[0].model_copy(update={"uv_mm": (9.0, 9.0)})
    return engine_result.model_copy(update={"observations": (obs.model_copy(update={"arrivals": (arrival,)}), *engine_result.observations[1:])})


def test_same_evaluation_skips_only_the_serialization_and_hash_of_the_result(project, catalog, engine_request, engine_result, ray_paths, scene):
    """``run_optical_project`` builds the view of the result object it just made (``same_evaluation``): the result is not
    dumped and hashed again. The scene is the same scene, and every other binding still refuses."""

    same = build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request, engine_result=engine_result,
                                          ray_paths=ray_paths, same_evaluation=True)
    assert same.content_hash() == scene.content_hash() and same.payload() == scene.payload()
    # what the flag skips: the re-hash of the result (a caller that brings a result from elsewhere keeps it: the test above)
    build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request, engine_result=_forged(engine_result),
                                   ray_paths=ray_paths, same_evaluation=True)
    with pytest.raises(OpticalViewBindingError, match="result_hash"):
        build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request,
                                       engine_result=_forged(engine_result), ray_paths=ray_paths)
    # what it keeps: the input and trace hashes as stated, and the element bindings
    other = engine_request.model_copy(update={"configuration": engine_request.configuration.model_copy(update={"random_seed": 7})})
    with pytest.raises(OpticalViewBindingError, match="different Engine inputs"):
        build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request, engine_result=engine_result,
                                       ray_paths=trace_ray_paths(other), same_evaluation=True)
    manifest = engine_result.manifest.model_copy(update={"trace_hash": "0" * 64})
    with pytest.raises(OpticalViewBindingError, match="trace hash"):
        build_optical_simulation_scene(project=project, catalog=catalog, engine_request=engine_request,
                                       engine_result=engine_result.model_copy(update={"manifest": manifest}), ray_paths=ray_paths,
                                       same_evaluation=True)
    shifted = tuple(c.model_copy(update={"nominal_pose": PoseV1(position_mm=(0.5, 0.0, 0.0), quaternion_wxyz=c.nominal_pose.quaternion_wxyz)})
                    if c.instance_id == "L1" else c for c in project.components)
    moved = ProjectDocumentV2.model_validate(project.model_copy(update={"components": shifted}).payload())
    with pytest.raises(OpticalViewBindingError, match="component 'L1'.*pose"):
        build_optical_simulation_scene(project=moved, catalog=catalog, engine_request=engine_request, engine_result=engine_result,
                                       ray_paths=ray_paths, same_evaluation=True)


def test_the_view_of_a_run_is_the_view_built_with_every_check(tmp_path):
    """The run's own scene (built with ``same_evaluation``) has the content hash of the scene built from the same run's
    pieces with the result re-hashed, for each reference design: the skipped step decides nothing in a consistent run."""

    from os_optics.local import EXAMPLE_KINDS, example_project_of_kind, run_optical_project
    from os_optics.local.run import project_catalog

    for kind in EXAMPLE_KINDS:
        run = run_optical_project(example_project_of_kind(kind))
        rebuilt = build_optical_simulation_scene(project=run.project, catalog=project_catalog(run.project, None),
                                                 engine_request=run.request, engine_result=run.result, ray_paths=run.ray_paths)
        assert rebuilt.content_hash() == run.optical_view.content_hash(), kind
