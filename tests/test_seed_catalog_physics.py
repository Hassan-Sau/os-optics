"""Generated products of the repository's seed public catalog release behave as the physics of their published prescriptions predicts.

The committed test catalog (``local_helpers``) pins the physics of each primitive; this module checks the *generated*
release instead, selecting products by query only (form, material, band, specified focal length): a cemented
achromatic doublet (prescription from the supplier's lens file) corrects the chromatic focal shift that a plano-convex
N-BK7 singlet of the same specified focal length shows, and each lens focuses where its published paraxial block says.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from os_optics.local import load_public_catalog, run_optical_project

from catalog_store_helpers import REAL_CATALOG

from local_helpers import arrivals, element, lens_matrix, plane, project, q_through, rayleigh_range, source

F_LINE_NM, D_LINE_NM, C_LINE_NM = 486.1327, 587.5618, 656.2725
SCREEN_BEYOND_FOCUS_MM = 20.0
CATALOG = load_public_catalog(REAL_CATALOG)  # the real release: this module checks generated products
DOCUMENT = json.loads(REAL_CATALOG.read_bytes())
SOURCE_WAIST_MM = 1.0  # local_helpers.source default


def _covers(product, *wavelengths: float) -> bool:
    band = product.simulation_models[0].valid_band_nm
    return all(Decimal(band.min_nm) <= Decimal(str(w)) <= Decimal(band.max_nm) for w in wavelengths)


def _focal_length(product) -> Decimal | None:
    spec = product.specifications.focal_length_mm
    return Decimal(spec.value) if spec is not None else None


def _candidates(form: str, media: list[str] | None = None) -> dict[Decimal, object]:
    """The first product (by optical SKU) of each specified focal length for a form, prescription fidelity, valid F..C."""

    found: dict[Decimal, object] = {}
    for product in sorted(CATALOG.products, key=lambda p: p.optical_sku):
        model = product.simulation_models[0]
        if product.form.value != form or model.fidelity.value != "SURFACE_PRESCRIPTION" or not _covers(product, F_LINE_NM, C_LINE_NM):
            continue
        if media is not None and list(model.media) != media:
            continue
        f = _focal_length(product)
        if f is not None and f > 0:
            found.setdefault(f, product)
    return found


def _pair():
    doublets, singlets = _candidates("ACHROMATIC_DOUBLET"), _candidates("PLANO_CONVEX_LENS", ["N-BK7"])
    common = sorted(f for f in set(doublets) & set(singlets) if f >= 50)
    # a failure, not a skip: a generator regression that dropped the doublets must not pass silently
    assert common, "the seed release has no doublet and N-BK7 plano-convex lens of one focal length valid over F..C"
    return doublets[common[0]], singlets[common[0]], float(common[0])


def _waist_positions(product, wavelengths: tuple[float, ...]) -> dict[float, float]:
    """Axial waist position (mm after the first vertex) of a collimated 1 mm-waist beam per wavelength, one Engine run."""

    model = product.simulation_models[0]
    last_vertex = float(max(Decimal(s.vertex_z_mm) for s in model.surfaces))
    screen_x = last_vertex + float(Decimal(model.paraxial.back_focal_length_mm)) + SCREEN_BEYOND_FOCUS_MM
    sources = [source(f"S{i}", (-50.0, 0.0, 0.0), wavelength_nm=w, waist_position_mm=50.0) for i, w in enumerate(wavelengths)]
    document = project([*sources, element("L1", "lens", product.optical_sku, catalog_version=CATALOG.catalog_version)],
                       [plane("screen", (screen_x, 0.0, 0.0), radius_mm=float(model.edge_radius_mm))])
    run = run_optical_project(document, catalog=CATALOG)
    assert run.model_scope_issues == ()
    bulk_unknown = "BULK_LOSS_NOT_ESTABLISHED" in {issue.code.value for issue in run.result.issues}
    unstated_loss = any(CATALOG.material(medium).bulk_loss is None for medium in model.media)
    assert bulk_unknown is unstated_loss  # absent catalog loss must never silently become known zero
    assert run.status.value == ("UNKNOWN" if unstated_loss else "PASS")
    if bulk_unknown:
        assert all(arrival.beam.power_W is None and arrival.power_upper_bound_W is not None
                   for observation in run.result.observations for arrival in observation.arrivals)
    positions = {}
    for i, w in enumerate(wavelengths):
        (arrival,) = arrivals(run.result, "screen", f"S{i}")
        positions[w] = screen_x - arrival.beam.gaussian.waist_distance_x_mm
    return positions


def test_an_achromatic_doublet_corrects_the_chromatic_focal_shift_of_a_singlet():
    doublet, singlet, f = _pair()
    shifts = {}
    for product in (doublet, singlet):
        positions = _waist_positions(product, (F_LINE_NM, C_LINE_NM))
        shifts[product.form.value] = positions[C_LINE_NM] - positions[F_LINE_NM]
    singlet_shift, doublet_shift = shifts["PLANO_CONVEX_LENS"], shifts["ACHROMATIC_DOUBLET"]
    assert singlet_shift == pytest.approx(f / 64.17, rel=0.05)  # f / V_d of N-BK7: red focuses farther
    assert abs(doublet_shift) < singlet_shift / 5  # the cemented pair's glasses cancel most of it


@pytest.mark.parametrize("which", [0, 1], ids=["doublet", "singlet"])
def test_each_lens_focuses_where_its_published_prescription_predicts(which):
    """Independent paraxial q propagation through the published surfaces and Sellmeier indices (``local_helpers``)
    predicts the Engine's waist; the prescription's plane-wave back focal length is the published paraxial block's."""

    product = _pair()[which]
    model = product.simulation_models[0]
    if any(CATALOG.material(m).dispersion.kind != "SELLMEIER" for m in model.media):
        pytest.skip("the independent check evaluates Sellmeier materials only")
    wavelength = float(model.paraxial.wavelength_nm)
    last_vertex = float(max(Decimal(s.vertex_z_mm) for s in model.surfaces))
    matrix = lens_matrix(product.optical_sku, wavelength, document=DOCUMENT)
    waist = _waist_positions(product, (wavelength,))[wavelength]
    # the source waist sits on the first vertex; a 1 mm waist is not a plane wave, so it lands short of the BFL
    predicted = last_vertex - q_through(matrix, complex(0.0, rayleigh_range(SOURCE_WAIST_MM, wavelength))).real
    assert waist == pytest.approx(predicted, abs=1e-6)
    assert -matrix[0, 0] / matrix[1, 0] == pytest.approx(float(model.paraxial.back_focal_length_mm), abs=1e-4)


# ------------------------------------------------------------------------------------------------ interference filters

FILTER_N_EFF = 1.7  # the conservative effective index of the published coating derivation


def _typical_bandpass(centre_nm: str, fwhm_nm: str):
    """The first ORDERABLE typical-curve bandpass filter of a centre wavelength and FWHM (selected by query)."""

    for product in sorted(CATALOG.products, key=lambda p: p.optical_sku):
        spec = product.specifications
        if (product.form.value == "BANDPASS_FILTER" and product.commercial_readiness.value == "ORDERABLE"
                and "TYPICAL_SPECTRAL_CURVE" in product.simulation_models[0].approximations
                and spec.center_wavelength_nm is not None and spec.center_wavelength_nm.value == centre_nm
                and spec.bandwidth_fwhm_nm is not None and spec.bandwidth_fwhm_nm.value == fwhm_nm):
            return product
    raise AssertionError(f"the seed release has no ORDERABLE typical bandpass filter {centre_nm} nm / FWHM {fwhm_nm} nm")


def _filter_angle_range(product) -> tuple[float, float]:
    angles = CATALOG.coating(product.simulation_models[0].surfaces[0].coating_id).valid_aoi_deg
    return float(angles.min_deg), float(angles.max_deg)


def _tilted_filter_run(product, tilt_deg: float):
    """A collimated beam at the filter's centre wavelength through the filter turned by ``tilt_deg`` about the vertical."""

    import math

    from local_helpers import ALONG_X, quat_multiply

    half = math.radians(tilt_deg) / 2.0
    pose = quat_multiply((math.cos(half), 0.0, 0.0, math.sin(half)), ALONG_X)
    centre = float(product.specifications.center_wavelength_nm.value)
    document = project([source("S", (-50.0, 0.0, 0.0), wavelength_nm=centre),
                        element("F", "filter", product.optical_sku, quaternion=pose, catalog_version=CATALOG.catalog_version)],
                       [plane("screen", (50.0, 0.0, 0.0), radius_mm=20.0)])
    return run_optical_project(document, catalog=CATALOG)


@pytest.mark.parametrize(("centre_nm", "fwhm_nm"), [("532", "3"), ("850", "25")], ids=["narrow-532-3nm", "broad-850-25nm"])
def test_an_interference_filter_is_in_model_scope_only_where_its_angle_shift_is_small(centre_nm, fwhm_nm):
    """Tilting a filter shifts its spectrum by lambda (1 - sqrt(1 - sin^2 theta / n_eff^2)); the published normal-
    incidence curve is used only while that shift stays within a quarter of the FWHM (n_eff 1.7). Inside its published
    range the run is PASS without scope issues; a narrow filter tilted to 10 deg is out of model scope."""

    import math

    product = _typical_bandpass(centre_nm, fwhm_nm)
    low, high = _filter_angle_range(product)
    centre, fwhm = float(centre_nm), float(fwhm_nm)
    s = math.sin(math.radians(high)) / FILTER_N_EFF
    assert low == 0.0 and centre * (1.0 - math.sqrt(1.0 - s * s)) <= fwhm / 4.0 + 1e-9  # independent of the generator
    inside = _tilted_filter_run(product, high - 0.5)
    assert inside.status.value == "PASS" and inside.model_scope_issues == ()
    tilted = _tilted_filter_run(product, 10.0)
    if high < 10.0:
        assert [i.code for i in tilted.model_scope_issues] == ["COATING_AOI_OUTSIDE_RANGE"]
    else:  # a broad filter keeps the near-normal range
        assert tilted.model_scope_issues == ()
