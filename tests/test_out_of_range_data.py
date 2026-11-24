"""Light that meets optical data outside the range the data is stated for.

A coating read outside its wavelengths or angles: a traced beam goes on with the coating's values at the nearest stated
point; where it goes is unchanged, its powers are estimates that name the surface (``value_basis``), and a power-derived
target behind it reports its value without deciding. A sampled ray, which carries no such mark, still ends there. Local
validity names the element and what the beam reaches behind it. A prescription whose stated band reaches beyond its
materials is imported over the part they cover.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from os_optics import ValidationRequest, validate
from os_optics import builders as B
from os_optics.contracts import ConstantCoating, ProjectSnapshot, TabulatedCoating, Target
from os_optics.local import load_public_catalog, lower_project
from os_optics.local.validity import element_poses, element_scope_issues
from os_optics.optical_view import geometry as g
from os_optics.prescription import PrescriptionImportError, import_prescription
from os_optics.results import basis_of

from catalog_store_helpers import REAL_CATALOG
from helpers import run
from local_helpers import ALONG_X, FLAT_MIRROR, FOLD_X_TO_Y, catalog_document, element, plane, project, source
from test_sampled_sources import _component as sampled_component
from test_sampled_sources import _intensity as sampled_intensity

SPLIT = 0.37  # the splitter's transmittance at 500 nm, the edge row of its table


def _table(wavelengths: tuple[float, ...]) -> TabulatedCoating:
    """R / T the same at every angle; the 500 nm row is the last stated row or, in the longer table, repeated at 600 nm."""

    rows = {400.0: 0.25, 500.0: SPLIT, 600.0: SPLIT}
    transmit = tuple((rows[w], rows[w]) for w in wavelengths)
    reflect = tuple((1.0 - rows[w], 1.0 - rows[w]) for w in wavelengths)
    return TabulatedCoating(wavelength_nm=wavelengths, aoi_deg=(0.0, 60.0), reflectance_s=reflect, reflectance_p=reflect,
                            transmittance_s=transmit, transmittance_p=transmit)


TARGETS = (
    Target(target_id="power", metric="power_W", observation_id="d", minimum=1e-4),
    Target(target_id="where", metric="position_u_mm", observation_id="d", target_value=0.0, tolerance_abs=1e-6),
    Target(target_id="size", metric="beam_radius_x_mm", observation_id="d", maximum=1.0),
)


def _split(wavelengths: tuple[float, ...]) -> ProjectSnapshot:
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=501.0, waist_radius_mm=0.4, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.beam_splitter("BS", position=(30, 0, 20), normal=(-1, 1, 0), coating=_table(wavelengths))),
        observations=(B.observation_plane("d", position=(80, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
        targets=TARGETS,
    )


def test_one_nanometre_beyond_a_coating_table_reads_its_edge_row_and_only_power_targets_stop_deciding():
    """501 nm light on a table stated for 400-500 nm against the same table with its 500 nm row repeated at 600 nm (so
    501 nm is inside it and reads the same values): every number of the two runs is equal bit for bit; the power is
    1 mW x the edge row's transmittance; the power target reports that value as an estimate and is undecided, the
    position and beam-size targets are graded as in the run inside the table."""

    held, inside = run(_split((400.0, 500.0))), run(_split((400.0, 500.0, 600.0)))
    assert held.observations[0].total_power_W == inside.observations[0].total_power_W
    assert (held.observations[0].quantity_state.value, inside.observations[0].quantity_state.value) == ("OUT_OF_MODEL_SCOPE", "KNOWN")
    (arrival,), (expected,) = held.observations[0].arrivals, inside.observations[0].arrivals
    assert arrival.model_dump(exclude={"power_out_of_scope_at", "value_basis"}) == expected.model_dump(exclude={"power_out_of_scope_at", "value_basis"})
    assert arrival.beam.power_W == pytest.approx(1e-3 * SPLIT, rel=1e-12)
    by_id, reference = {t.target_id: t for t in held.targets}, {t.target_id: t for t in inside.targets}
    power = by_id["power"]
    assert (power.status.value, power.assessment.value, power.issue_codes) == ("OUT_OF_MODEL_SCOPE", "INDETERMINATE", ("COATING_OUT_OF_RANGE",))
    assert power.value == reference["power"].value == arrival.beam.power_W and power.error == reference["power"].error
    assert reference["power"].status.value == "PASS" and reference["power"].value_basis is None
    entry = power.value_basis["value"]
    assert entry == arrival.value_basis["beam.power_W"] == basis_of(held.observations[0], "total_power_W")
    assert entry.kind.value == "ESTIMATE" and entry.assumed[0].value == 500.0 and entry.validity[0].observed == 501.0
    assert (entry.validity[0].minimum, entry.validity[0].maximum, entry.validity[0].within) == (400.0, 500.0, False)
    assert "BS/coating" in power.message and "does not decide" in power.message
    for name in ("where", "size"):
        assert by_id[name].model_dump() == reference[name].model_dump() and by_id[name].status.value == "PASS"
    assert held.outcome.requirements.indeterminate == ("power",) and set(held.outcome.requirements.met) == {"where", "size"}
    assert held.status.value == "OUT_OF_MODEL_SCOPE" and inside.status.value == "PASS"  # the power target's own status


def _level(axis) -> list[float]:
    """The pose of a plane facing ``axis`` with the design Z axis in it (plain algebra on the stated axes)."""
    z, y = g.unit(axis), (0.0, 0.0, 1.0)
    return list(g.matrix_to_quat((g.cross(y, z), y, z)))


def _narrow_mirror_run(first: dict):
    """A catalog mirror whose Engine coating is replaced by one stated for 400-500 nm (R = 0.9), met by 780 nm / 633 nm light
    of ``first``; a plane before it and one behind the fold."""

    document = project([first, element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), FOLD_X_TO_Y)],
                       [plane("screen", (0.0, 100.0, 0.0), _level((0.0, 1.0, 0.0))), plane("before", (-50.0, 0.0, 0.0))])
    lowered = lower_project(document, catalog_document())
    (mirror,) = [c for c in lowered.request.project.components if c.instance_id == "M1"]
    narrow = ConstantCoating(reflectance_s=0.9, reflectance_p=0.9, transmittance_s=0.0, transmittance_p=0.0, valid_wavelength_nm=(400.0, 500.0))
    faces = tuple(s.model_copy(update={"interaction": s.interaction.model_copy(update={"coating": narrow})})
                  if getattr(s.interaction, "coating", None) is not None else s for s in mirror.surfaces)
    assert faces != mirror.surfaces
    components = tuple(c.model_copy(update={"surfaces": faces}) if c.instance_id == "M1" else c for c in lowered.request.project.components)
    request = lowered.request.model_copy(update={"project": lowered.request.project.model_copy(update={"components": components})})
    return lowered, validate(ValidationRequest.model_validate_json(request.model_dump_json()))


def test_a_sampled_ray_still_ends_at_a_coating_outside_its_data():
    """An extended emitter's rays carry no mark of a coating read outside its data: they end there (their power is booked as
    ended outside the model), so the plane behind the mirror reports no power of them."""

    emitter = {**sampled_component(sampled_intensity([[1.0, 1.0], [1.0, 1.0]], 0.5, rays=64)), "instance_id": "SRC"}
    emitter["nominal_pose"] = {"position_mm": [-100.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}
    _lowered, result = _narrow_mirror_run(emitter)
    (issue,) = [i for i in result.issues if i.code.value == "COATING_OUT_OF_RANGE"]
    assert issue.status.value == "OUT_OF_MODEL_SCOPE" and issue.numerical_context["sampled_ray_events"] == 64
    by_id = {o.observation_id: o for o in result.observations}
    assert any(line.counted_crossing_count for line in by_id["before"].sampled_light)
    assert not by_id["screen"].arrivals and not any(line.counted_crossing_count for line in by_id["screen"].sampled_light)
    assert by_id["screen"].total_power_W is None and by_id["screen"].quantity_state.value != "KNOWN"
    (line,) = result.power_budget.lines
    assert line.sinks["TERMINATED_OUT_OF_MODEL"] == pytest.approx(1e-3, rel=1e-9)


def test_local_validity_names_the_element_and_what_the_beam_reaches_behind_it():
    """The Engine carries a 780 nm beam on past the mirror coating stated for 400-500 nm; the model-scope finding
    COATING_DATA_OUTSIDE_RANGE names the mirror, its surface and the plane behind it -- not the plane before it."""

    lowered, result = _narrow_mirror_run(source(position=(-100.0, 0.0, 0.0)))
    assert sorted((o.observation_id, len(o.arrivals)) for o in result.observations) == [("before", 1), ("screen", 1)]
    found = [i for i in element_scope_issues(result, lowered.elements, poses=element_poses(lowered)) if i.code == "COATING_DATA_OUTSIDE_RANGE"]
    (issue,) = found
    assert issue.instance_id == "M1" and issue.affected_observation_ids == ("screen",)
    assert "780.0 nm light meets M1" in issue.message and "estimates" in issue.message and "stopped" not in issue.message


SEED = load_public_catalog(REAL_CATALOG.read_bytes())
SINGLET = "surface,radius,thickness,material,semi_diameter\n1,50,5,N-BK7,10\n2,-50,10,,10\n"


def test_a_stated_band_wider_than_the_glass_is_narrowed_to_what_the_glass_covers():
    """N-BK7 is published for 300-2500 nm. A band of 250-600 nm is imported as 300-600 nm with a BAND_NARROWED warning that
    names the glass and its row; a band inside the glass's is kept as stated; a band with nothing inside it is refused."""

    narrowed = import_prescription(SINGLET, catalog=SEED, wavelength_band_nm=(Decimal(250), Decimal(600)))
    band = narrowed.report["wavelength_band_nm"]
    assert (band["min_nm"], band["max_nm"]) == ("300", "600") and "narrowed" in band["basis"]
    (warning,) = [w for w in narrowed.report["warnings"] if w["code"] == "BAND_NARROWED"]
    assert "250-600 nm" in warning["message"] and "300-600 nm" in warning["message"] and "N-BK7" in warning["message"]
    assert warning["location"]["row"] == 2 and warning["location"]["column"] == "material"
    assert all((m.valid_band_nm.min_nm, m.valid_band_nm.max_nm, m.valid_band_nm.status.value) == ("300", "600", "DERIVED")
               for m in narrowed.optical_models if getattr(m, "valid_band_nm", None) is not None)
    inside = import_prescription(SINGLET, catalog=SEED, wavelength_band_nm=(Decimal(400), Decimal(700)))
    assert inside.report["wavelength_band_nm"]["min_nm"] == "400" and not [w for w in inside.report["warnings"] if w["code"] == "BAND_NARROWED"]
    with pytest.raises(PrescriptionImportError) as refused:
        import_prescription(SINGLET, catalog=SEED, wavelength_band_nm=(Decimal(100), Decimal(200)))
    (problem,) = [p.as_dict() for p in refused.value.problems]
    assert problem["code"] == "MATERIAL_OUTSIDE_BAND" and problem["location"]["row"] == 2
