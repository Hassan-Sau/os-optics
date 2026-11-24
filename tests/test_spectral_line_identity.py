"""A spectral line's identity in an analysis document is the run's own (its source and spectral index, as in its path ids), never a
display format of its wavelength: two lines 0.0001 nm apart, and the quadrature lines of a narrow continuous spectrum, each get
their own quantities, with the exact wavelength in their scope (imaging and ghost-path analyses)."""

from __future__ import annotations

import pytest

from os_optics.local import run_optical_project
from os_optics.results import load_run

from test_ghosts import _paraxial_ghost_focus, _prescription
from test_imaging import _hyperbolic, _lens_project, _nbk7

CLOSE = (587.5617, 587.5618)  # equal to six significant digits, the display format of a wavelength


def _lines(model: dict, lines: tuple[float, ...]) -> None:
    power = model.pop("power_w")
    del model["wavelength_nm"]
    model["spectral_lines"] = [{"wavelength_nm": w, "power_w": power / len(lines)} for w in lines]


def _narrow_band(model: dict, line_count: int) -> None:
    del model["wavelength_nm"]
    model["spectral_density"] = {"shape": {"kind": "GAUSSIAN", "center_nm": "587.5618", "fwhm_nm": "0.0004"},
                                 "band_nm": ["587.5613", "587.5623"], "line_count": line_count}


def _results(text: str, tmp_path, shape) -> object:
    project = _lens_project(text, waist_mm=0.5)
    shape(project["components"][0]["source_model"])
    run_optical_project(project, output_dir=str(tmp_path / "run"))
    return load_run(tmp_path / "run")


def _line_ids(doc) -> dict[str, float]:
    """The line prefix of every per-line quantity (``<source>#<index>``) with the wavelength in its scope."""

    lines: dict[str, set[float]] = {}
    for q in doc.quantities:
        head = q.quantity_id.split(".", 1)[0]
        if "#" in head and q.scope.wavelength_nm is not None:
            lines.setdefault(head, set()).add(q.scope.wavelength_nm)
    assert all(len(w) == 1 for w in lines.values()), lines  # one wavelength per line id
    return {head: w.pop() for head, w in lines.items()}


def _paths_lines(results) -> dict[str, float]:
    return {f"{p.source_id}#{p.spectral_index}": p.wavelength_nm for p in results.result.paths}


def test_imaging_keeps_two_lines_a_tenth_of_a_picometre_apart(tmp_path):
    results = _results(_hyperbolic(8.0, 40.0, 587.5618), tmp_path, lambda m: _lines(m, CLOSE))
    doc = results.imaging("image", {"launch_grid": 32, "psf_samples": 32})
    ids = _line_ids(doc)
    assert ids == {"SRC#0": CLOSE[0], "SRC#1": CLOSE[1]} == _paths_lines(results)
    for line in ids:  # both lines are analysed in full: an aberration-free focus for each
        assert doc.quantity(f"{line}.strehl_ratio").value == pytest.approx(1.0, abs=1e-2)


def test_ghosts_keep_two_lines_a_tenth_of_a_picometre_apart(tmp_path):
    bfl, _ghost = _paraxial_ghost_focus(_nbk7(587.5618), 40.0, -40.0, 5.0)
    results = _results(_prescription(40.0, -40.0, 5.0, bfl), tmp_path, lambda m: _lines(m, CLOSE))
    doc = results.ghosts("image")
    ids = _line_ids(doc)
    assert ids == {"SRC#0": CLOSE[0], "SRC#1": CLOSE[1]} == _paths_lines(results)
    for line in ids:
        assert f"{line}.ghosts" in doc.arrays and f"{line}.completeness" in doc.arrays
        assert doc.quantity(f"{line}.primary_power_relative_to_line").value > 0.0


@pytest.mark.parametrize("analysis", ["imaging", "ghosts"])
def test_the_quadrature_lines_of_a_narrow_spectrum_each_have_their_quantities(tmp_path, analysis):
    """A 0.4 pm wide Gaussian spectrum over 1 pm in 8 quadrature lines (125 fm apart): every line is its own field."""

    if analysis == "imaging":
        results = _results(_hyperbolic(8.0, 40.0, 587.5618), tmp_path, lambda m: _narrow_band(m, 8))
        doc = results.imaging("image", {"launch_grid": 32, "psf_samples": 32})
    else:
        bfl, _ghost = _paraxial_ghost_focus(_nbk7(587.5618), 40.0, -40.0, 5.0)
        results = _results(_prescription(40.0, -40.0, 5.0, bfl), tmp_path, lambda m: _narrow_band(m, 8))
        doc = results.ghosts("image")
    lines = _paths_lines(results)
    assert len(set(lines.values())) == 8  # eight distinct wavelengths, the same to six significant digits in pairs
    assert len({f"{w:g}" for w in lines.values()}) < 8
    assert _line_ids(doc) == lines
