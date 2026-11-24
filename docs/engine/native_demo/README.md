# Native component demonstration: polarization-isolated confocal reflectance bench (632.8 nm)

This is a substantially larger, non-template design composed by an Agent from the native vocabulary. It is not a
reference design: no example kind or template produces it.

`build_confocal_bench.py` builds it the way a customer's Agent would, through the public JSON API only:

* discover the native classes (`native_components`);
* build the parts the catalog does not sell (`build_native_model`);
* query the catalog for orderable laboratory optics (`catalog_products`, `catalog_product`, `catalog_materials`,
  `catalog_coatings`);
* place the components, add observations and targets;
* check the Project (`check_project`), and with `--run`, run it.

`confocal_bench.project.json` is the Project it wrote. It pins catalog release 2026-10-28.1 and has
`design_intent: LAB_ASSEMBLY`.

## What the bench does

A HeNe laser beam is cleaned and expanded by a spatial filter, then focused onto a reflective sample. The light reflected
by the sample is separated from the input by polarization (a polarizing splitter with a quarter-wave plate in double
pass) and focused through a confocal pinhole onto a photodiode. A second photodiode monitors the laser power on the
splitter's forward leak.

```
LASER -> POL -> HWP -> [ L1 f25 | pinhole 40 um | L2 achromat f125 ]  -> IRIS 6 mm -> PBS -> QWP 45 deg -> FOLD -+
 (-440)  (-410) (-390)          EXPANDER 5x (-350 ... -194)               (-170)      (-100)    (-60)      (0,0)   |
                                                                                       |                           v
                                                                     MONITOR (-100,-60)  (p leak, -Y)    OBJECTIVE f50 (0,-40)
                                                                                       |                           |
                     DETECTOR (-100,219.1) <- PINHOLE 50 um <- TUBE LENS f150 <- (+Y, s)   SAMPLE R = 0.5 (0,-95.0)
                                             (-100,214.1)     (-100,60)
```

Coordinates are in mm. The beam plane is z = 0, with +Z up.

| Component | Role | Source | Readiness |
|---|---|---|---|
| LASER | source | customer laser, Gaussian 0.4 mm waist, 1 mW, linear (horizontal = p at the splitter) | customer light source |
| POL | polarizer | native `polarizer.linear` (T 0.9, ER 10^4) | customer-supplied (all catalog polarizers are SIMULATION_ONLY) |
| HWP | waveplate | catalog achromatic HWP 600-800 nm | ORDERABLE |
| EXPANDER | other_optical | native `compound.keplerian_telescope`: `lens.singlet` f 25 mm N-BK7 plano-convex, 40 um pinhole at the internal focus, `lens.achromat` f 125 mm N-BK7/N-SF5 | customer-supplied unit |
| IRIS | aperture | native `aperture.circular`, 6 mm | customer-supplied |
| PBS | splitter | native `splitter.plate`, R_s 0.995 / T_p 0.995 at 40-50 deg | customer-supplied (catalog PBS cubes at 633 nm are SIMULATION_ONLY) |
| QWP | waveplate | catalog achromatic QWP 600-800 nm, fast axis at 45 deg | ORDERABLE |
| FOLD | mirror | catalog HR 632.8 nm mirror, AOI 45 deg | ORDERABLE |
| OBJECTIVE | lens | catalog achromat f 50 mm | ORDERABLE |
| SAMPLE | mirror | native `mirror.flat`, declared R = 0.5 | the customer's sample |
| TUBE_LENS | lens | catalog achromat f 150 mm | ORDERABLE |
| PINHOLE | aperture | native `aperture.circular`, 50 um | customer-supplied |
| DETECTOR, MONITOR | detector | customer photodiodes (absorbing observation planes) | customer equipment |

The catalog parts are chosen by query, not by hard-coded SKUs. The rules are:

* ORDERABLE only;
* 25.4 mm outer diameter;
* the waveplates with the highest transmittance;
* a mirror whose coating holds at 45 deg;
* achromats of f = 50 mm and 150 mm whose glasses are published as Sellmeier formulas (so the script can trace them at
  632.8 nm).

## Design reasoning

* **Polarization isolation.** The laser is p-polarized at the plate splitter, so 99.5 % is transmitted. A quarter-wave
  plate with its fast axis at 45 deg, passed twice (out to the sample and back), acts as a half-wave plate at 45 deg. The
  returning beam is therefore s-polarized, and the splitter reflects 99.5 % of it into the detection arm (+Y). A 50:50
  splitter would lose 75 % of the light instead.
* **Power monitor.** The splitter reflects 0.5 % of the forward p light toward -Y, onto the monitor photodiode. The
  half-wave plate is the power-control element: turning it sends more of the light to the monitor.
* **Spatial filter.** The expander is one compound unit, placed as a whole.
  * Lens 1 is a plano-convex singlet, f 25 mm, with its curved face toward the laser.
  * Lens 2 is an f 125 mm achromat, both designed by the library.
  * The lenses are spaced afocally at 632.8 nm: the rear focus of L1 coincides with the front focus of L2. The design
    report gives the system matrix C = 0 and a beam magnification of -5.0015.
  * The 40 um pinhole sits at the internal focus. The focused spot there has a radius of about 12.6 um
    (`lambda f / (pi w)`), so the pinhole passes the beam's core.
* **Confocal geometry.**
  * The sample sits at the objective's back focus at 632.8 nm. The script finds it with a y-nu trace of the catalog
    prescription: BFL 43.4928 mm.
  * The returning beam is collimated again and focused by the tube lens (BFL 146.1921 mm) onto the 50 um pinhole.
  * The detected spot radius is `lambda f / (pi w)` = 0.0150 mm for w = 2.01 mm.
* **Lab assembly.** Every part is a separate mounted component on one beam plane, except the spatial filter, which is a
  single unit. The catalog optics are orderable. The native parts and the laser, sample and photodiodes are
  customer-supplied. Under the role-based supply rule, the customer-supplied inline optics make this a self-assembly
  purchase.

## Targets and independent expectations

| Target | Expectation (independent estimate) | Criterion |
|---|---|---|
| `expanded_beam_radius` (after the expander) | 0.4026 mm at L1 (0.4 mm waist, z_R = 794 mm, 90 mm away) x 5.0015 = 2.013 mm | 2.0 +- 0.1 mm |
| `confocal_spot_radius` (0.5 mm before the pinhole) | w0 = 0.0150 mm, z_R = 1.12 mm -> 0.0165 mm | <= 0.03 mm |
| `detected_power` | about 0.35 mW (budget below) | >= 0.1 mW |
| `monitor_power` | 0.005 x about 0.83 mW = about 4 uW | >= 1 uW |

The detected-power estimate multiplies these factors:

1. polarizer 0.9;
2. HWP 0.95;
3. expander about 0.978 (AR R = 0.005 per declared face, plus the bare cemented interface);
4. iris 0.988 (Gaussian clip at r = 3 mm);
5. PBS 0.995;
6. then, for the round trip:
   * QWP 0.95, twice;
   * objective about 0.98, twice;
   * mirror 0.9999, twice;
   * sample 0.5;
   * PBS reflection 0.995;
7. tube lens about 0.98.

The result is about 0.35 mW.

### Result (os-optics 0.14.0)

The committed Project was run with os-optics 0.14.0 against public catalog release 2026-10-28.1, with the shell
command in [How to run it](#how-to-run-it). The run completes (execution outcome `COMPLETED`), but the overall status is
UNKNOWN and the command exits with code 2:

| Target | Value | Status | Assessment |
|---|---|---|---|
| `expanded_beam_radius` | 2.00470 mm | WARNING | MET |
| `confocal_spot_radius` | 0.02547 mm | WARNING | MET |
| `detected_power` | not decided | UNKNOWN | INDETERMINATE (`BULK_LOSS_NOT_ESTABLISHED`) |
| `monitor_power` | not decided | UNKNOWN | INDETERMINATE (`BULK_LOSS_NOT_ESTABLISHED`) |

The power targets are not decided because no glass on the path has a stated bulk loss at 632.8 nm. This covers the
expander's N-BK7 and N-SF5 and the catalog achromats' N-BAF10, N-SF10, N-BK7 and SF5. The engine therefore bounds each
detector plane's power only as between 0 W and 0.000850725 W. The geometric paths stay valid, but the bound straddles
both criteria. A Project can state bulk loss for the materials of its own native parts. It cannot state it for the glasses of
a catalog lens, so the power targets stay undecided until the catalog publishes bulk-loss data for those glasses.

The WARNINGs on the beam-radius targets are truthful: each stop clips part of the Gaussian beam (`GAUSSIAN_TRUNCATION`,
at the expander pinhole, the iris and the confocal pinhole).

The returning beam's polarization leak crosses the `EXPANDED` plane backwards. The target therefore selects the forward
crossing (`path_id LASER#0`, `crossing_index 0`).

The committed Project was written with this script against the same catalog release. Running the script now writes a
different Project: its catalog queries choose other wave plates. The figures above are those of the committed Project.

## How to run it

From the repository root (engine environment; the catalog file is the public release the test suite fetches with `python tests/catalog_store_helpers.py`):

```bash
uv run --no-sync python docs/engine/native_demo/build_confocal_bench.py \
    --catalog .cache/public-catalog/public_optical_catalog.2026-10-28.1.json \
    --out <OUT_DIR>/confocal_bench.project.json --run <RUN_DIR>
```

The same Project runs from a shell with:

```bash
os-optics project run docs/engine/native_demo/confocal_bench.project.json --out-dir <RUN_DIR>
```

Then `os-optics results describe <RUN_DIR>` and `os-optics view <RUN_DIR>` show the results.

The run uses the engine's Project-declared optical models: `capabilities()["project_features"]` lists `optical_models`,
`compound_units` and `aperture_stops`.
