"""The ``csv-surface-table/1`` adapter: a documented CSV surface table -> the prescription IR (``ir.py``).

The format:

* comma-separated UTF-8 text; a cell may be quoted (``"..."``, ``""`` for a quote); blank lines are ignored;
* ``# ...`` lines are comments; ``#! key = value`` lines are directives: ``format`` (``csv-surface-table/1``), ``units``
  (``mm`` default, ``cm``, ``m``, ``in``), ``name``, ``wavelength_min_nm`` and ``wavelength_max_nm``, and the ranges the
  file declares for its own elements, each optional: ``edge_thickness_min_mm`` / ``edge_thickness_max_mm`` (the glass
  thickness at the edge, in mm whatever the file's unit) and ``mech_to_clear_ratio_min`` / ``mech_to_clear_ratio_max`` (a
  surface's mechanical semi-diameter over its clear one); an element outside a declared range is refused;
* the wavelength convention and reference environment, each optional (OpticStudio semantics):

  - ``wavelengths_in`` (``VACUUM`` or ``AIR``): how the file writes its wavelengths (``wavelength_min_nm`` /
    ``wavelength_max_nm`` and the ``[coatings]`` ``wavelength_nm``). Absent: ``VACUUM`` -- this format's own
    convention. ``AIR``: wavelengths measured in air at the file's ``temperature_c`` and pressure (the
    system temperature and pressure); each is converted once to the vacuum wavelength with the shared air model
    (``material_physics.air_index_exact``: Edlen 1953 scaled as OpticStudio does), coating wavelengths to 1e-6 nm, the band
    rounded inward to 0.001 nm; at zero pressure air is vacuum and nothing changes. The air formula is evaluated from
    200 nm to 100 um only. A ``[materials]`` law states its own convention (its ``index_reference`` column), not this one;
  - ``temperature_c`` (above -273.15) and ``pressure_pa`` or ``pressure_atm`` (one of them; 0 or more; 1 atm =
    101325 Pa): the reference environment -- the air outside the elements and in every air gap, and the temperature of
    the materials. Absent: the file states no environment (the Project's own applies: air at 20 C and 101325 Pa unless it
    states another); one of the two stated: the other is the default (20 C, 101325 Pa). A file that states any of
    ``wavelengths_in = AIR``, ``temperature_c``, ``pressure_*`` carries that environment into the import
    (``merge_into_project`` sets it on the Project and refuses a Project whose environment differs);

* ``glass_catalogs`` (also ``glass_catalogues``, ``gcat``): the prescription's catalogue order -- maker codes or
  catalogue names separated by spaces, commas or semicolons, first preferred, e.g. ``#! glass_catalogs = CDGM SCHOTT``
  (the ``GCAT`` line of a Zemax file). A glass name several published records answer to (their designations, aliases
  or ids) is then the record of the first catalogue in the list that has exactly one of them; without it such a name is
  refused as ``GLASS_AMBIGUOUS`` naming every candidate;
* ``ignore_columns`` names extra columns of any table that are skipped;
* a column no table knows is refused (``COLUMN_UNKNOWN``) unless it is named in ``ignore_columns`` (directive or option)
  or blank in every row; a skipped column is listed in the import report (``ignored_columns``);
* ``[surfaces]`` (the default section), ``[materials]`` and ``[coatings]`` start a table; the first row of each table is
  its header;
* ``units`` states the file's length unit; a caller's ``units`` option is used when the file states none and must agree
  with it otherwise (``UNITS_CONFLICT``);
* surface table: one row per surface in the order light meets them; columns ``surface`` (label: ``OBJ``, ``IMA``,
  ``STO`` or any other printable-ASCII label), ``radius`` or ``curvature``, ``thickness``, ``material``, ``semi_diameter`` (required),
  and optionally ``mech_semi_diameter`` or ``edge_semi_diameter`` (one column, two names: the radius the surface is edged
  to; the element's outline is the largest one),
  ``conic``, ``a4`` .. ``a18``, ``linear_sag`` (the axicon term, dimensionless;
  also ``a1`` / ``axicon``), ``cylinder_axis``, ``stop``, ``coating``, ``type``, ``glass_manufacturer`` (also
  ``manufacturer``, ``glass_maker``, ``glass_catalog``: the maker or catalogue of the row's glass),
  ``comment``, ``norm_radius``, ``sag_terms``, ``radius_x``, ``conic_x``, ``radius_of_rotation`` (two-variable sag shapes, below)
  and ``decenter_x`` / ``decenter_y`` / ``tilt_x`` / ``tilt_y`` / ``tilt_z`` / ``a2`` (accepted only when blank or zero);
* two-variable sag shapes, chosen by ``type``:

  - ``ZERNIKE`` (also ``SZERNSAG``, ``ZERNIKE_STANDARD``: Noll indices, each term of unit RMS) and ``FZERNSAG`` (also
    ``ZERNIKE_FRINGE``: Fringe indices, each term of unit peak) add ``sum value Z_i(rho, theta)`` to the row's ``radius``,
    ``conic`` and ``a4`` ..; ``norm_radius`` is the normalization radius, ``sag_terms`` the terms: ``Z4=0.01; Z6=-0.002``;
  - ``XY_POLY`` (also ``XY_POLYNOMIAL``) adds ``sum value (x / R)^i (y / R)^j``; ``norm_radius`` is ``R``, ``sag_terms`` the terms:
    ``X2Y0=0.01; X0Y2=0.01``. Zernike and XY values are lengths, in the file's unit;
  - ``BICONIC``: ``radius`` and ``conic`` state the section along Y, ``radius_x`` and ``conic_x`` the one along X;
  - ``TOROIDAL`` (also ``TOROID``): ``radius``, ``conic`` and the ``sag_terms`` ``Y2=1e-4; Y4=-2e-7`` (``alpha_i`` of ``y^(2i)``, in the
    file's unit to the power ``1 - 2i``) state the curve in the YZ plane, ``radius_of_rotation`` the distance from the vertex
    to the axis it turns about (blank or ``inf``: no rotation).
* the ``material`` cell names the medium after the surface: blank or ``AIR`` for the Project's air,
  ``VACUUM`` for an evacuated gap (index 1 regardless of environment), a material ``[materials]`` defines, or a published glass (``glass.py`` resolves it with the shared resolver): its stable id (``SCHOTT_F5``), its
  exact designation (``S-FPL53``, ``D-K9-25``) or a documented alias; ``MAKER:DESIGNATION`` (``SCHOTT:F5``, the same as
  ``F5`` with ``glass_manufacturer`` ``SCHOTT``) names that maker's (or catalogue's) record; ``ID:<material id>``
  (``ID:F2``) names a stable id only -- no designation lookup and no catalogue order;
* materials table: ``name``, ``kind`` (``SELLMEIER`` or ``CONSTANT_INDEX``), ``b1`` .. ``b3`` and ``c1`` .. ``c3``
  (um^2) or ``index``, ``min_nm``, ``max_nm``, and optionally ``absorption_per_mm``, ``display_name``, ``basis`` and
  ``index_reference``: blank or ``VACUUM`` (the law takes vacuum wavelengths and gives the absolute index; ``min_nm`` /
  ``max_nm`` are vacuum wavelengths) or ``AIR`` (the glass catalogues' convention: the law takes air wavelengths and
  gives the index relative to air, of the standard reference air 20 C / 101325 Pa; ``min_nm`` / ``max_nm`` are air
  wavelengths). A file with ``wavelengths_in = AIR`` whose material leaves it blank is read as ``VACUUM`` with a
  warning;
* coatings table (``[coatings]``): one row per tabulated point, the rows of one coating sharing its ``name``:
  ``wavelength_nm``, optionally ``aoi_deg`` (on every row of an angle-resolved coating, on none of a spectrum), the power
  fractions ``reflectance`` / ``transmittance`` and / or ``reflectance_s``, ``reflectance_p``, ``transmittance_s``,
  ``transmittance_p``, optionally the phases ``reflection_phase_s`` / ``reflection_phase_p`` and
  ``transmission_phase_s`` / ``transmission_phase_p`` (radians); and, on the coating's first row, ``kind`` (a coating
  kind, e.g. ``ANTI_REFLECTION``), ``min_aoi_deg`` / ``max_aoi_deg``, ``aoi_reference``, ``status``, ``display_name``,
  ``basis`` (later rows leave them blank or repeat them).

Numbers are plain decimals of magnitude 0 or 1e-300 .. 1e300. The ``type`` column is a cross-check: a type the row's
values contradict (``PLANE`` with a radius, ``STANDARD`` with aspheric terms, ...) is refused. The last surface's
thickness only places the image (any number, ``inf`` for an image at infinity, or blank).

Every problem found is reported (``PrescriptionImportError``) with its row, column and surface; nothing is guessed.
"""

from __future__ import annotations

import hashlib
import re
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation

from .._vendor.shared_contracts.common.sag_shapes import FRINGE, NOLL, PEAK_UNIT, RMS_ORTHONORMAL, SAG_COMBINATION_NOT_SUPPORTED
from .errors import ImportWarning, PrescriptionErrorCode as E, ProblemList, SourceLocation, refuse
from .._vendor.shared_contracts.design_catalog.material_physics import (
    AIR_MODEL_BAND_NM,
    BAND_QUANTUM_NM,
    STANDARD_PRESSURE_PA,
    STANDARD_TEMPERATURE_C,
)
from .ir import (
    LAW_AIR,
    LAW_VACUUM,
    MM_PER_UNIT,
    VACUUM_WAVELENGTH_QUANTUM_NM,
    WAVELENGTHS_AIR,
    WAVELENGTHS_VACUUM,
    EdgeLimits,
    ReferenceEnvironment,
    UNIT_SPELLINGS,
    air_to_vacuum_nm,
    CoatingDefinition,
    CoatingNode,
    CoatingRef,
    ConstantIndexDefinition,
    IrSagShape,
    IrSurface,
    MaterialDefinition,
    MaterialRef,
    ObjectSpace,
    Prescription,
    SellmeierDefinition,
    power_to_mm,
    reciprocal,
    to_mm,
)

FORMAT_ID = "csv-surface-table/1"
SURFACES = "surfaces"
MATERIALS = "materials"
COATINGS = "coatings"
SECTIONS = (SURFACES, MATERIALS, COATINGS)
EDGE_LIMIT_DIRECTIVES = ("edge_thickness_min_mm", "edge_thickness_max_mm", "mech_to_clear_ratio_min", "mech_to_clear_ratio_max")
#: Directive naming extra columns (of any table) the import does not read: ``#! ignore_columns = vendor_pn, price``.
IGNORE_DIRECTIVE = "ignore_columns"
ENVIRONMENT_DIRECTIVES = ("wavelengths_in", "temperature_c", "pressure_pa", "pressure_atm")
DIRECTIVES = ("format", "units", "name", "wavelength_min_nm", "wavelength_max_nm", *EDGE_LIMIT_DIRECTIVES, *ENVIRONMENT_DIRECTIVES,
              "glass_catalogs", IGNORE_DIRECTIVE)
DIRECTIVE_ALIASES = {"glass_catalogues": "glass_catalogs", "gcat": "glass_catalogs"}
WAVELENGTH_CONVENTIONS = (WAVELENGTHS_VACUUM, WAVELENGTHS_AIR)
#: The catalogue-order entries a ``glass_catalogs`` directive may hold (catalogue tags: upper case after reading).
CATALOGUE_TAG_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9_.+-]*$")
GLASS_ID_PREFIX = "ID"
CATALOGUE_TAG_MAX = 64
#: ``report.ignored_columns[].reason``: the caller named the column (option or directive) / every cell of it is blank.
IGNORED_DECLARED, IGNORED_BLANK = "DECLARED", "BLANK"
#: Even-asphere columns (A4 .. A18: the native model's eight polynomial terms).
ASPHERIC_COLUMNS = ("a4", "a6", "a8", "a10", "a12", "a14", "a16", "a18")
#: Columns accepted only when blank or zero (coaxial, no a2 term).
ZERO_ONLY_COLUMNS = ("decenter_x", "decenter_y", "tilt_x", "tilt_y", "tilt_z", "a2")
#: Columns of the two-variable sag shapes.
SHAPE_COLUMNS = ("norm_radius", "sag_terms", "radius_x", "conic_x", "radius_of_rotation")
SURFACE_COLUMNS = ("surface", "type", "radius", "curvature", "thickness", "material", "semi_diameter", "mech_semi_diameter",
                   "conic", *ASPHERIC_COLUMNS, "linear_sag", "cylinder_axis", "stop", "coating", "comment", *SHAPE_COLUMNS, *ZERO_ONLY_COLUMNS,
                   "glass_manufacturer")
SURFACE_ALIASES = {
    "surf": "surface", "label": "surface", "surface_type": "type", "r": "radius", "roc": "radius", "c": "curvature",
    "curv": "curvature", "t": "thickness", "thick": "thickness", "distance": "thickness", "glass": "material",
    "medium": "material", "sd": "semi_diameter", "semi_dia": "semi_diameter", "semidiameter": "semi_diameter",
    "clear_semi_diameter": "semi_diameter", "semi_aperture": "semi_diameter", "sdia": "semi_diameter", "mech_sd": "mech_semi_diameter",
    "mechanical_semi_diameter": "mech_semi_diameter", "mech_semi_dia": "mech_semi_diameter", "mech": "mech_semi_diameter",
    "edge_semi_diameter": "mech_semi_diameter", "edge_sd": "mech_semi_diameter", "k": "conic",
    "conic_constant": "conic", "note": "comment", "notes": "comment", "a1": "linear_sag", "axicon": "linear_sag",
    "linear_sag_coefficient": "linear_sag", "normalization_radius": "norm_radius", "rx": "radius_x", "kx": "conic_x",
    "radius_rotation": "radius_of_rotation", "manufacturer": "glass_manufacturer", "glass_maker": "glass_manufacturer",
    "glass_catalog": "glass_manufacturer",
}
SURFACE_REQUIRED = ("surface", "thickness", "material", "semi_diameter")
MATERIAL_COLUMNS = ("name", "kind", "b1", "b2", "b3", "c1", "c2", "c3", "index", "min_nm", "max_nm", "absorption_per_mm",
                    "display_name", "basis", "index_reference")
MATERIAL_ALIASES = {"material": "name", "glass": "name", "n": "index", "refractive_index": "index"}
MATERIAL_REQUIRED = ("name", "min_nm", "max_nm")
SELLMEIER_COLUMNS = ("b1", "b2", "b3", "c1", "c2", "c3")
#: Per-node columns of [coatings]: power fractions and stated phases (radians) at one wavelength (and angle).
COATING_VALUE_COLUMNS = ("reflectance", "transmittance", "reflectance_s", "reflectance_p", "transmittance_s", "transmittance_p")
COATING_PHASE_COLUMNS = ("reflection_phase_s", "reflection_phase_p", "transmission_phase_s", "transmission_phase_p")
#: Per-coating columns of [coatings]: stated on a coating's first row; later rows leave them blank or repeat them.
COATING_ATTRIBUTE_COLUMNS = ("kind", "min_aoi_deg", "max_aoi_deg", "aoi_reference", "status", "display_name", "basis")
COATING_COLUMNS = ("name", "wavelength_nm", "aoi_deg", *COATING_VALUE_COLUMNS, *COATING_PHASE_COLUMNS, *COATING_ATTRIBUTE_COLUMNS)
COATING_ALIASES = {"coating": "name", "wavelength": "wavelength_nm", "aoi": "aoi_deg", "r": "reflectance", "t": "transmittance",
                   "r_s": "reflectance_s", "r_p": "reflectance_p", "t_s": "transmittance_s", "t_p": "transmittance_p"}
COATING_REQUIRED = ("name", "wavelength_nm")
COATING_ID_MAX = 96

OBJECT_LABELS = frozenset({"OBJ", "OBJECT"})
IMAGE_LABELS = frozenset({"IMA", "IMAGE"})
STOP_LABELS = frozenset({"STO", "STOP"})
AIR_NAMES = frozenset({"", "AIR"})
VACUUM_NAME = "VACUUM"
MIRROR_NAMES = frozenset({"MIRROR", "MIRR", "REFLECT", "REFLECTIVE"})
TRUE_WORDS = frozenset({"yes", "y", "true", "1", "x", "stop", "sto"})
FALSE_WORDS = frozenset({"", "no", "n", "false", "0"})
UNCOATED_WORDS = frozenset({"", "NONE", "UNCOATED", "BARE"})
INFINITY_WORDS = frozenset({"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})
#: ``type`` values for the profiles the native model holds (the values decide the profile; the type is a cross-check).
ZERNIKE_NOLL_TYPES = frozenset({"ZERNIKE", "SZERNSAG", "ZERNIKE_STANDARD"})
ZERNIKE_FRINGE_TYPES = frozenset({"FZERNSAG", "ZERNIKE_FRINGE"})
XY_TYPES = frozenset({"XY_POLY", "XY_POLYNOMIAL"})
BICONIC_TYPES = frozenset({"BICONIC"})
TOROID_TYPES = frozenset({"TOROIDAL", "TOROID"})
SHAPE_TYPES = ZERNIKE_NOLL_TYPES | ZERNIKE_FRINGE_TYPES | XY_TYPES | BICONIC_TYPES | TOROID_TYPES
SUPPORTED_TYPES = frozenset({"", "STANDARD", "SPHERE", "SPHERICAL", "PLANE", "FLAT", "CONIC", "EVENASPH", "EVEN_ASPHERE",
                             "ASPHERE", "ASPHERIC", "CYLINDER", "CYLINDRICAL", "AXICON", "CONE"}) | SHAPE_TYPES
AXICON_TYPES = frozenset({"AXICON", "CONE"})
CYLINDER_TYPES = frozenset({"CYLINDER", "CYLINDRICAL"})
PLANE_TYPES = frozenset({"PLANE", "FLAT"})
SPHERE_TYPES = frozenset({"SPHERE", "SPHERICAL"})
#: Types whose profile has no aspheric polynomial terms (the even-asphere types and a blank type allow them).
NO_ASPHERE_TYPES = PLANE_TYPES | SPHERE_TYPES | {"STANDARD", "CONIC"} | CYLINDER_TYPES
#: ``type`` prefixes of surface kinds a coaxial sequential import cannot represent, and why.
UNSUPPORTED_TYPE_PREFIXES = (
    (("COORDBRK", "COORDINATE_BREAK", "DECENTER", "TILT"), "a coordinate break (decentre / tilt)"),
    (("NSC", "NON_SEQUENTIAL", "NONSEQ"), "a non-sequential surface"),
    (("GRIN", "GRADIENT", "GRADIUM"), "a gradient-index medium"),
    (("DIFFRACT", "GRATING", "BINARY", "HOLOGRAM", "DOE", "KINOFORM"), "a diffractive surface"),
    (("PARAXIAL",), "an ideal paraxial surface (declare an IDEAL_THIN_LENS model instead)"),
    (("ODD", "EXTENDED", "QTYPE", "Q_TYPE", "CHEBYSHEV", "FREEFORM"),
     "a profile other than plane, sphere, conic, even asphere, axicon, cylinder, Zernike, XY polynomial, biconic or toroid (an odd "
     "asphere whose only odd term is r^1 is an axicon: give that term in linear_sag)"),
)
MATERIAL_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9_.-]*$")
MATERIAL_ID_MAX = 64
PUBLIC_TEXT_PATTERN = re.compile(r"^[ -~]*[!-~][ -~]*$")
NAME_MAX = 200
TEXT_MAX = 1000
#: The magnitudes a number may have (the Engine computes in binary64): 0, or 1e-300 .. 1e300.
NUMBER_MIN = Decimal("1e-300")
NUMBER_MAX = Decimal("1e300")
NUMBER_PATTERN = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
SECTION_PATTERN = re.compile(r"^\[\s*([A-Za-z_]+)\s*\]$")
DIRECTIVE_PATTERN = re.compile(r"^#!\s*([A-Za-z_]+)\s*=\s*(.*?)\s*$")
ZERNIKE_TERM_PATTERN = re.compile(r"^Z([0-9]+)$", re.IGNORECASE)
XY_TERM_PATTERN = re.compile(r"^X([0-9]+)Y([0-9]+)$", re.IGNORECASE)
TOROID_TERM_PATTERN = re.compile(r"^Y([0-9]+)$", re.IGNORECASE)


# ------------------------------------------------------------------------------------------------ tokenizing


def _cells(line: str) -> list[str] | None:
    """The comma-separated cells of one line (quotes: ``"..."`` with ``""`` inside); ``None`` for an unterminated quote."""

    cells: list[str] = []
    cell: list[str] = []
    quoted = False
    i = 0
    while i < len(line):
        ch = line[i]
        if quoted:
            if ch == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    cell.append('"')
                    i += 1
                else:
                    quoted = False
            else:
                cell.append(ch)
        elif ch == '"':
            quoted = True
        elif ch == ",":
            cells.append("".join(cell))
            cell = []
        else:
            cell.append(ch)
        i += 1
    if quoted:
        return None
    cells.append("".join(cell))
    return [c.strip() for c in cells]


def _not_ascii(text: str) -> str | None:
    """Why ``text`` is not printable ASCII (the first offending character, named), or ``None``."""

    for position, ch in enumerate(text, start=1):
        if not " " <= ch <= "~":
            if ch == "\ufeff":
                return (f"character {position} is a byte-order mark (U+FEFF); a BOM is accepted only at the very start of the file "
                        "(re-save it as UTF-8, or remove the stray mark)")
            return f"character {position} is {ch!r} (U+{ord(ch):04X})"
    return None


def _header_name(cell: str) -> str:
    return re.sub(r"[\s-]+", "_", cell.strip().lower())


# --------------------------------------------------------------------------------------------------- values


def _decimal(raw: str) -> Decimal | None:
    if not NUMBER_PATTERN.match(raw):
        return None
    try:
        return Decimal(raw)
    except InvalidOperation:  # pragma: no cover - the pattern admits only finite decimals
        return None


class _Row:
    """One table row: its cells by column name, its location, and the shared problem list."""

    def __init__(self, cells: dict[str, str], location: SourceLocation, problems: ProblemList) -> None:
        self.cells = cells
        self.location = location
        self.problems = problems

    def raw(self, column: str) -> str:
        return self.cells.get(column, "")

    def at(self, column: str) -> SourceLocation:
        return self.location.at(column)

    def number(self, column: str, *, required: bool = False, what: str = "a number") -> Decimal | None:
        raw = self.raw(column)
        if raw == "":
            if required:
                self.problems.add(E.FIELD_MISSING, f"{column} is required here; give {what}", self.at(column))
            return None
        value = _decimal(raw)
        if value is None:
            self.problems.add(E.VALUE_INVALID, f"{column} {raw!r} is not a number; give {what}", self.at(column))
        elif value != 0 and not NUMBER_MIN <= abs(value) <= NUMBER_MAX:
            self.problems.add(E.VALUE_INVALID, f"{column} {raw!r} is beyond the magnitudes the Engine computes with (1e-300 to "
                              f"1e300); give {what}", self.at(column))
            return None
        return value

    def positive(self, column: str, *, required: bool = False, what: str = "a positive number") -> Decimal | None:
        value = self.number(column, required=required, what=what)
        if value is not None and not value > 0:
            self.problems.add(E.VALUE_INVALID, f"{column} must be positive, got {self.raw(column)!r}", self.at(column))
            return None
        return value


# ------------------------------------------------------------------------------------------------ the adapter


class _Reader:
    def __init__(self, text: str, default_unit: str, requested_unit: str | None = None, ignore_columns: tuple[str, ...] = ()) -> None:
        self.text = text
        self.requested_unit = requested_unit
        self.ignore_option = tuple(ignore_columns)
        self.ignore: frozenset[str] = frozenset()
        self.ignored: list[tuple[str, str, str]] = []  # (section, column as written, reason)
        self.image_at_infinity = False
        self.problems = ProblemList()
        self.warnings: list[ImportWarning] = []
        self.directives: dict[str, tuple[str, SourceLocation]] = {}
        self.tables: dict[str, tuple[list[str], SourceLocation, list[tuple[list[str], int]]]] = {}
        self.default_unit = default_unit

    # --- lines -> directives and tables

    def split(self) -> None:
        section = SURFACES
        seen_sections: set[str] = set()
        for number, line in enumerate(self.text.split("\n"), start=1):
            line = line.rstrip("\r")
            if number == 1:
                line = line.lstrip("﻿")
            stripped = line.strip()
            if stripped.startswith("#"):
                self._directive_or_comment(stripped, number)
                continue
            cells = _cells(line)
            if cells is None:
                self.problems.add(E.TEXT_INVALID, "a quoted cell is not closed (a quote opens a cell, \"\" writes a quote inside it)",
                                  SourceLocation(row=number))
                continue
            if all(c == "" for c in cells):
                continue
            first, rest = cells[0], cells[1:]
            if first.startswith("#") and all(c == "" for c in rest):
                self._directive_or_comment(first, number)
                continue
            match = SECTION_PATTERN.match(first)
            if match and all(c == "" for c in rest):
                name = match.group(1).lower()
                if name not in SECTIONS:
                    self.problems.add(E.SECTION_INVALID, f"unknown section [{match.group(1)}]; the sections are [surfaces], [materials] "
                                      "and [coatings]",
                                      SourceLocation(row=number))
                    section = ""
                elif name in seen_sections or name in self.tables:
                    self.problems.add(E.SECTION_INVALID, f"section [{name}] appears twice; give each table once", SourceLocation(row=number))
                    section = ""
                else:
                    section = name
                    seen_sections.add(name)
                continue
            if section == "":
                continue
            if section not in self.tables:
                self.tables[section] = (cells, SourceLocation(row=number, section=section), [])
            else:
                self.tables[section][2].append((cells, number))

    def _directive_or_comment(self, text: str, number: int) -> None:
        if not text.startswith("#!"):
            return
        text = text.rstrip(", \t")
        match = DIRECTIVE_PATTERN.match(text)
        where = SourceLocation(row=number)
        if match is None:
            self.problems.add(E.DIRECTIVE_INVALID, "a directive is written '#! key = value'", where)
            return
        key, value = match.group(1).lower(), match.group(2)
        key = DIRECTIVE_ALIASES.get(key, key)
        if key not in DIRECTIVES:
            self.problems.add(E.DIRECTIVE_INVALID, f"unknown directive {key!r}; the directives are {', '.join(DIRECTIVES)}", where)
        elif key in self.directives:
            self.problems.add(E.DIRECTIVE_INVALID, f"directive {key!r} is given twice (first on row {self.directives[key][1].row})", where)
        else:
            self.directives[key] = (value, where)

    # --- directives

    def unit(self) -> str:
        if "units" not in self.directives:
            return self.default_unit
        value, where = self.directives["units"]
        unit = UNIT_SPELLINGS.get(value.strip().lower())
        if unit is None:
            self.problems.add(E.UNITS_INVALID, f"unknown length unit {value!r}; use one of {', '.join(MM_PER_UNIT)}", where)
            return self.default_unit
        if self.requested_unit is not None and unit != self.requested_unit:
            self.problems.add(E.UNITS_CONFLICT, f"the file states units = {unit}, but the import was asked to use the unit "
                              f"{self.requested_unit}; drop the units option (the file's own unit is used) or correct the directive", where)
        return unit

    def check_format(self) -> None:
        if "format" in self.directives:
            value, where = self.directives["format"]
            if value.strip().lower() != FORMAT_ID:
                self.problems.add(E.FORMAT_UNSUPPORTED, f"this file states format {value!r}; this reader reads {FORMAT_ID}", where)

    def name(self) -> str | None:
        if "name" not in self.directives:
            return None
        value, where = self.directives["name"]
        if not PUBLIC_TEXT_PATTERN.match(value) or len(value) > NAME_MAX:
            detail = _not_ascii(value)
            self.problems.add(E.DIRECTIVE_INVALID, f"name must be 1 to {NAME_MAX} printable ASCII characters{f': {detail}' if detail else ''}",
                              where)
            return None
        return value

    def band(self) -> tuple[Decimal, Decimal] | None:
        given = [key for key in ("wavelength_min_nm", "wavelength_max_nm") if key in self.directives]
        if not given:
            return None
        if len(given) == 1:
            self.problems.add(E.DIRECTIVE_INVALID, "give both wavelength_min_nm and wavelength_max_nm (or neither: the band then "
                              "follows from the materials)", self.directives[given[0]][1])
            return None
        values = []
        for key in ("wavelength_min_nm", "wavelength_max_nm"):
            raw, where = self.directives[key]
            value = _decimal(raw)
            if value is None or not value > 0:
                self.problems.add(E.DIRECTIVE_INVALID, f"{key} must be a positive number of nanometres, got {raw!r}", where)
                return None
            values.append(value)
        if not values[0] <= values[1]:
            self.problems.add(E.DIRECTIVE_INVALID, "wavelength_min_nm cannot exceed wavelength_max_nm", self.directives["wavelength_max_nm"][1])
            return None
        return values[0], values[1]

    def wavelengths_in(self) -> str:
        if "wavelengths_in" not in self.directives:
            return WAVELENGTHS_VACUUM
        value, where = self.directives["wavelengths_in"]
        convention = value.strip().upper()
        if convention not in WAVELENGTH_CONVENTIONS:
            self.problems.add(E.DIRECTIVE_INVALID, f"wavelengths_in {value!r} is not VACUUM or AIR (how the file writes its "
                              "wavelengths: AIR = measured in air at the file's temperature_c and pressure)", where)
            return WAVELENGTHS_VACUUM
        return convention

    def environment(self, wavelengths_in: str) -> ReferenceEnvironment | None:
        """The environment the file states (module docstring), or None when it states none."""

        stated: list[str] = []
        temperature, pressure = STANDARD_TEMPERATURE_C, STANDARD_PRESSURE_PA
        if "temperature_c" in self.directives:
            raw, where = self.directives["temperature_c"]
            value = _decimal(raw.strip())
            if value is None or not value > Decimal("-273.15"):
                self.problems.add(E.DIRECTIVE_INVALID, f"temperature_c must be a number of degrees Celsius above -273.15, got {raw!r}", where)
            else:
                temperature = value
                stated.append("temperature_c")
        given = [key for key in ("pressure_pa", "pressure_atm") if key in self.directives]
        if len(given) == 2:
            self.problems.add(E.DIRECTIVE_INVALID, "give pressure_pa or pressure_atm, not both", self.directives["pressure_atm"][1])
        elif given:
            raw, where = self.directives[given[0]]
            value = _decimal(raw.strip())
            if value is None or value < 0:
                self.problems.add(E.DIRECTIVE_INVALID, f"{given[0]} must be a number, 0 (vacuum) or more, got {raw!r}", where)
            else:
                pressure = value * STANDARD_PRESSURE_PA if given[0] == "pressure_atm" else value
                stated.append(given[0])
        if not stated and wavelengths_in != WAVELENGTHS_AIR:
            return None
        return ReferenceEnvironment(temperature_c=temperature, pressure_pa=pressure, stated=tuple(stated))

    def glass_catalogs(self) -> tuple[str, ...]:
        if "glass_catalogs" not in self.directives:
            return ()
        raw, where = self.directives["glass_catalogs"]
        entries = [entry.upper() for entry in re.split(r"[\s,;]+", raw.strip()) if entry]
        bad = [entry for entry in entries if not CATALOGUE_TAG_PATTERN.match(entry) or len(entry) > CATALOGUE_TAG_MAX]
        if not entries or bad:
            self.problems.add(E.DIRECTIVE_INVALID, "glass_catalogs lists maker codes or catalogue names (letters, digits, '.', '_', '+', "
                              f"'-'), separated by spaces or commas, first preferred, e.g. 'CDGM SCHOTT'{f'; not {bad[0]!r}' if bad else ''}",
                              where)
            return ()
        return tuple(dict.fromkeys(entries))

    def to_vacuum(self, value: Decimal, environment: ReferenceEnvironment, where: SourceLocation, what: str) -> Decimal | None:
        """An air wavelength of the file as the vacuum wavelength (unrounded), or None (problem added) outside the air
        formula's range."""

        low, high = AIR_MODEL_BAND_NM
        if not low <= value <= high:
            self.problems.add(E.VALUE_INVALID, f"{what} {value} nm is an air wavelength (wavelengths_in = AIR) outside the air formula's "
                              f"range {low}-{high} nm; give it as a vacuum wavelength (wavelengths_in = VACUUM)", where)
            return None
        return air_to_vacuum_nm(value, environment.temperature_c, environment.pressure_pa)

    def edge_limits(self) -> EdgeLimits | None:
        values: dict[str, Decimal] = {}
        for key in EDGE_LIMIT_DIRECTIVES:
            if key not in self.directives:
                continue
            raw, where = self.directives[key]
            value = _decimal(raw)
            if value is None or not value > 0:
                self.problems.add(E.DIRECTIVE_INVALID, f"{key} must be a positive number, got {raw!r}", where)
                continue
            values[key] = value
        for low, high in zip(EDGE_LIMIT_DIRECTIVES[::2], EDGE_LIMIT_DIRECTIVES[1::2]):
            if low in values and high in values and values[low] > values[high]:
                self.problems.add(E.DIRECTIVE_INVALID, f"{low} cannot exceed {high}", self.directives[high][1])
        if not values:
            return None
        return EdgeLimits(thickness_min_mm=values.get("edge_thickness_min_mm"), thickness_max_mm=values.get("edge_thickness_max_mm"),
                          ratio_min=values.get("mech_to_clear_ratio_min"), ratio_max=values.get("mech_to_clear_ratio_max"))

    def ignored_names(self) -> None:
        """The extra columns the caller names (option ``ignore_columns`` and directive ``ignore_columns``), normalized as
        header cells are."""
        names = list(self.ignore_option)
        if IGNORE_DIRECTIVE in self.directives:
            value, where = self.directives[IGNORE_DIRECTIVE]
            listed = [part.strip() for part in value.split(",")]
            if not all(listed):
                self.problems.add(E.DIRECTIVE_INVALID, "ignore_columns is a comma-separated list of column names", where)
            names += [part for part in listed if part]
        self.ignore = frozenset(_header_name(name) for name in names)

    # --- tables

    def rows(self, section: str, known: tuple[str, ...], aliases: dict[str, str], required: tuple[str, ...]) -> list[_Row]:
        if section not in self.tables:
            return []
        header_cells, header_location, data = self.tables[section]
        columns: list[str] = []
        blank: set[int] = set()
        for index, cell in enumerate(header_cells):
            name = _header_name(cell)
            name = aliases.get(name, name)
            where = SourceLocation(row=header_location.row, column=cell or f"#{index + 1}", section=section)
            if name == "":
                blank.add(index)
                columns.append("")
                continue
            if name not in known:
                # a column the import does not read: skipped when the caller names it or it is blank in every row; one
                # holding values is never dropped unnamed (a misspelt optical column would change the lens)
                if name in self.ignore:
                    self.ignored.append((section, cell, IGNORED_DECLARED))
                elif all(index >= len(cells) or cells[index] == "" for cells, _number in data):
                    self.ignored.append((section, cell, IGNORED_BLANK))
                else:
                    self.problems.add(E.COLUMN_UNKNOWN, f"unknown column {cell!r} in [{section}]; the columns are {', '.join(known)}; "
                                      f"a column the import should not read is named with the ignore_columns option or directive", where)
                columns.append("")
                continue
            if name in {aliases.get(n, n) for n in self.ignore}:
                self.problems.add(E.DIRECTIVE_INVALID, f"column {cell!r} of [{section}] is read by the import and cannot be ignored; "
                                  "remove it from ignore_columns", where)
            if name in columns:
                self.problems.add(E.COLUMN_DUPLICATE, f"column {name!r} appears twice in [{section}]", where)
                columns.append("")
                continue
            columns.append(name)
        header_where = SourceLocation(row=header_location.row, section=section)
        for name in required:
            if name not in columns:
                self.problems.add(E.COLUMN_MISSING, f"[{section}] needs a {name!r} column", header_where)
        rows: list[_Row] = []
        for cells, number in data:
            values: dict[str, str] = {}
            for index, cell in enumerate(cells):
                if index >= len(columns) or columns[index] == "":
                    if cell != "" and (index >= len(columns) or index in blank):
                        self.problems.add(E.ROW_INVALID, f"cell {index + 1} of this row holds {cell!r} but the header names no column "
                                          "there", SourceLocation(row=number, section=section))
                        break
                    continue
                values[columns[index]] = cell
            rows.append(_Row(values, SourceLocation(row=number, section=section), self.problems))
        return rows


def _surface_type(row: _Row) -> str:
    kind = row.raw("type").upper().replace(" ", "_").replace("-", "_")
    if kind in SUPPORTED_TYPES:
        return kind
    for prefixes, what in UNSUPPORTED_TYPE_PREFIXES:
        if kind.startswith(prefixes):
            row.problems.add(E.UNSUPPORTED_FEATURE, f"surface type {row.raw('type')!r} is {what}, which a coaxial sequential import "
                             "does not represent", row.at("type"))
            return kind
    row.problems.add(E.VALUE_INVALID, f"unknown surface type {row.raw('type')!r}; leave it blank or use STANDARD, PLANE, CONIC, "
                     "EVENASPH, AXICON, CYLINDER, ZERNIKE, FZERNSAG, XY_POLY, BICONIC or TOROIDAL", row.at("type"))
    return kind


def _type_contradiction(kind: str, radius: Decimal | None, conic: Decimal | None, aspheric: tuple[Decimal, ...],
                        axis: str | None, linear: Decimal | None = None) -> list[str]:
    """How the row's values contradict its stated ``type`` (the type is a cross-check: the values decide the profile)."""

    reasons = []
    if kind in PLANE_TYPES:
        reasons += [what for present, what in ((radius is not None, f"a plane has no radius (radius {radius} mm)"),
                                               (conic is not None, "a plane has no conic constant"),
                                               (axis is not None, "a plane has no cylinder_axis"),
                                               (linear is not None, "a plane has no axicon term (linear_sag)")) if present]
    elif kind in SPHERE_TYPES | {"CONIC"} and radius is None:
        reasons.append(f"a {kind.lower()} surface has a radius (the radius is blank or inf: a plane)")
    elif kind in CYLINDER_TYPES and radius is None and linear is None:
        reasons.append(f"a {kind.lower()} surface has a radius or an axicon term (linear_sag)")
    if kind in AXICON_TYPES and linear is None:
        reasons.append("an axicon surface has an axicon term (linear_sag)")
    if kind in SPHERE_TYPES | {"STANDARD", "CONIC", "EVENASPH", "EVEN_ASPHERE", "ASPHERE", "ASPHERIC"} and linear is not None:
        reasons.append("an axicon term (linear_sag) belongs to type AXICON")
    if kind in SPHERE_TYPES and conic is not None:
        reasons.append(f"a sphere has no conic constant (conic {conic})")
    if kind in NO_ASPHERE_TYPES and aspheric:
        reasons.append("aspheric terms (a4 ..) belong to type EVENASPH")
    if kind not in CYLINDER_TYPES | PLANE_TYPES | {""} and axis is not None:
        reasons.append("a cylinder_axis belongs to type CYLINDER")
    return reasons


def _flag(row: _Row, column: str) -> bool:
    raw = row.raw(column).lower()
    if raw in TRUE_WORDS:
        return True
    if raw not in FALSE_WORDS:
        row.problems.add(E.VALUE_INVALID, f"{column} {row.raw(column)!r} is not yes or no", row.at(column))
    return False


class _SurfaceReader:
    """Reads one surface row into an ``IrSurface`` (or the object / image record)."""

    def __init__(self, reader: _Reader, unit: str) -> None:
        self.reader = reader
        self.unit = unit

    def length(self, row: _Row, column: str, value: Decimal | None) -> Decimal | None:
        return None if value is None else to_mm(value, self.unit)

    def rounded(self, row: _Row, column: str, raw: str, value: Decimal) -> None:
        self.reader.warnings.append(ImportWarning(
            "VALUE_ROUNDED", f"{column} {raw!r} converted to mm is not a finite decimal; it is rounded to {value} (28 significant digits)",
            row.at(column)))

    def radius(self, row: _Row, has_curvature: bool) -> Decimal | None:
        if has_curvature:
            raw = row.raw("curvature")
            if raw.lower() in INFINITY_WORDS:
                row.problems.add(E.VALUE_INVALID, "an infinite curvature is not a surface", row.at("curvature"))
                return None
            curvature = row.number("curvature", what="a curvature (1 / radius), blank or 0 for a plane")
            if curvature is None or curvature == 0:
                return None
            radius, exact = reciprocal(curvature)
            radius = to_mm(radius, self.unit)
            if not exact:
                self.rounded(row, "curvature", raw, radius)
            return radius
        raw = row.raw("radius")
        if raw.lower() in INFINITY_WORDS:
            return None
        radius = row.number("radius", what="a radius, or blank / inf for a plane")
        if radius is not None and radius == 0:
            row.problems.add(E.VALUE_INVALID, "radius 0 is not a surface; leave the cell blank or write inf for a plane", row.at("radius"))
            return None
        return self.length(row, "radius", radius)

    def aspheric(self, row: _Row) -> tuple[Decimal, ...]:
        terms: list[Decimal] = []
        for power, column in enumerate(ASPHERIC_COLUMNS, start=2):
            value = row.number(column, what="an aspheric coefficient")
            if value is None or value == 0:
                terms.append(Decimal(0))
                continue
            converted, exact = power_to_mm(value, self.unit, 1 - 2 * power)
            if not exact:
                self.rounded(row, column, row.raw(column), converted)
            terms.append(converted)
        while terms and terms[-1] == 0:
            terms.pop()
        return tuple(terms)

    def material(self, row: _Row) -> MaterialRef | None:
        raw = row.raw("material")
        name = raw.upper()
        maker = row.raw("glass_manufacturer").strip() or None
        if name in AIR_NAMES:
            if maker is not None:
                row.problems.add(E.VALUE_INVALID, f"glass_manufacturer {maker!r} names the maker of the row's glass, but the medium after "
                                 "this surface is air; leave it blank", row.at("glass_manufacturer"))
            return None
        if name == VACUUM_NAME:
            if maker is not None:
                row.problems.add(E.VALUE_INVALID, f"glass_manufacturer {maker!r} does not apply to a vacuum gap; leave it blank",
                                 row.at("glass_manufacturer"))
                return None
            return MaterialRef(VACUUM_NAME, row.at("material"))
        if name in MIRROR_NAMES:
            row.problems.add(E.UNSUPPORTED_FEATURE, "a reflecting surface in the sequence is not imported; import the "
                             "refracting groups on each side and place a mirror component between them", row.at("material"))
            return None
        if maker is not None and _not_ascii(maker) is not None:
            row.problems.add(E.VALUE_INVALID, f"glass_manufacturer must be printable ASCII: {_not_ascii(maker)}", row.at("glass_manufacturer"))
            return None
        prefix, colon, rest = raw.partition(":")
        if not colon:
            return MaterialRef(raw, row.at("material"), manufacturer=maker)
        prefix, rest = prefix.strip(), rest.strip()
        if not prefix or not rest:
            row.problems.add(E.VALUE_INVALID, f"material {raw!r}: write MAKER:DESIGNATION (e.g. SCHOTT:F5) or ID:<material id> (e.g. "
                             "ID:F2), or the glass alone", row.at("material"))
            return None
        if prefix.upper() == GLASS_ID_PREFIX:
            if maker is not None:
                row.problems.add(E.VALUE_INVALID, f"material {raw!r} names a stable material id; a glass_manufacturer does not apply to it "
                                 "(leave it blank)", row.at("glass_manufacturer"))
                return None
            return MaterialRef(rest, row.at("material"), explicit_id=True)
        if maker is not None and maker.upper() != prefix.upper():
            row.problems.add(E.VALUE_INVALID, f"material {raw!r} names maker {prefix!r} and glass_manufacturer says {maker!r}; state the "
                             "maker once", row.at("glass_manufacturer"))
            return None
        return MaterialRef(rest, row.at("material"), manufacturer=prefix)

    def zero_only(self, row: _Row) -> None:
        for column in ZERO_ONLY_COLUMNS:
            value = row.number(column)
            if value is not None and value != 0:
                what = "an a2 (r^2) polynomial term" if column == "a2" else "a decentre or tilt"
                row.problems.add(E.UNSUPPORTED_FEATURE, f"{column} = {row.raw(column)} is {what}; this import is coaxial (put the "
                                 "curvature in radius, and place a decentred or tilted group as its own component)", row.at(column))

    def _terms(self, row: _Row) -> list[tuple[str, Decimal, str]]:
        """The ``name=value`` entries of ``sag_terms``, separated by ``;``."""

        out = []
        for entry in row.raw("sag_terms").split(";"):
            if not entry.strip():
                continue
            name, separator, raw = entry.partition("=")
            value = _decimal(raw.strip())
            if not separator or value is None:
                row.problems.add(E.VALUE_INVALID, f"sag_terms entry {entry.strip()!r} is not name=number, e.g. Z4=0.01", row.at("sag_terms"))
                continue
            if value != 0:
                out.append((name.strip(), value, raw.strip()))
        return out

    def _length_or_infinite(self, row: _Row, column: str, what: str) -> Decimal | None:
        if row.raw(column).lower() in INFINITY_WORDS:
            return None
        value = row.number(column, what=what)
        if value is not None and value == 0:
            row.problems.add(E.VALUE_INVALID, f"{column} 0 is not a radius; leave the cell blank or write inf", row.at(column))
            return None
        return self.length(row, column, value)

    def _named_terms(self, row: _Row, pattern: re.Pattern, example: str) -> list[tuple[tuple[int, ...], Decimal, str]]:
        out = []
        for name, value, raw in self._terms(row):
            match = pattern.match(name)
            if match is None:
                row.problems.add(E.VALUE_INVALID, f"sag_terms entry {name!r} is not a term of this surface type; write {example}", row.at("sag_terms"))
                continue
            out.append((tuple(int(group) for group in match.groups()), value, raw))
        if len({numbers for numbers, _, _ in out}) != len(out):
            row.problems.add(E.VALUE_INVALID, "sag_terms names one term twice", row.at("sag_terms"))
        return out

    def shape(self, row: _Row, kind: str, radius: Decimal | None, conic: Decimal | None) -> IrSagShape | None:
        """The two-variable sag shape ``type`` chooses (the ``SHAPE_COLUMNS`` belong to it and to nothing else)."""

        if kind not in SHAPE_TYPES:
            for column in SHAPE_COLUMNS:
                if row.raw(column) != "":
                    row.problems.add(E.VALUE_INVALID, f"{column} belongs to a surface type ZERNIKE, FZERNSAG, XY_POLY, BICONIC or TOROIDAL",
                                     row.at(column))
            return None
        allowed = ("radius_x", "conic_x") if kind in BICONIC_TYPES else ("sag_terms", "radius_of_rotation") if kind in TOROID_TYPES else ("norm_radius", "sag_terms")
        for column in SHAPE_COLUMNS:
            if column not in allowed and row.raw(column) != "":
                row.problems.add(E.VALUE_INVALID, f"{column} does not belong to surface type {row.raw('type')}", row.at(column))
        if kind in BICONIC_TYPES:
            radius_x = self._length_or_infinite(row, "radius_x", "the radius of the section along X, blank or inf for none")
            conic_x = row.number("conic_x", what="the conic constant of the section along X")
            return IrSagShape("BICONIC", radius_x_mm=radius_x, radius_y_mm=radius, conic_x=None if conic_x == 0 else conic_x, conic_y=conic)
        if kind in TOROID_TYPES:
            rotation = self._length_or_infinite(row, "radius_of_rotation", "the radius of rotation, blank or inf for none")
            alphas: dict[int, Decimal] = {}
            for (power,), value, raw in self._named_terms(row, TOROID_TERM_PATTERN, "Y2=1e-4; Y4=-2e-7"):
                if power % 2 or power < 2:
                    row.problems.add(E.VALUE_INVALID, f"sag_terms Y{power}: a toroid's curve has even powers of y from y^2", row.at("sag_terms"))
                    continue
                converted, exact = power_to_mm(value, self.unit, 1 - power)
                if not exact:
                    self.rounded(row, "sag_terms", raw, converted)
                alphas[power // 2] = converted
            y_terms = tuple(alphas.get(i, Decimal(0)) for i in range(1, max(alphas, default=0) + 1))
            return IrSagShape("TOROIDAL", radius_y_mm=radius, conic=conic, y_terms=y_terms, radius_of_rotation_mm=rotation)
        scale = row.positive("norm_radius", required=True, what="the normalization radius")
        if kind in XY_TYPES:
            terms = tuple((numbers[0], numbers[1], to_mm(value, self.unit))
                          for numbers, value, _ in self._named_terms(row, XY_TERM_PATTERN, "X2Y0=0.01; X0Y2=0.01"))
            if not terms and row.raw("sag_terms") == "":
                row.problems.add(E.FIELD_MISSING, "an XY polynomial surface needs its sag_terms", row.at("sag_terms"))
            return IrSagShape("XY_POLYNOMIAL", normalization_radius_mm=self.length(row, "norm_radius", scale), terms=terms)
        terms = tuple((numbers[0], to_mm(value, self.unit)) for numbers, value, _ in self._named_terms(row, ZERNIKE_TERM_PATTERN, "Z4=0.01; Z6=-0.002"))
        if not terms and row.raw("sag_terms") == "":
            row.problems.add(E.FIELD_MISSING, "a Zernike surface needs its sag_terms", row.at("sag_terms"))
        noll = kind in ZERNIKE_NOLL_TYPES
        return IrSagShape("ZERNIKE", indexing=NOLL if noll else FRINGE, normalization=RMS_ORTHONORMAL if noll else PEAK_UNIT,
                          normalization_radius_mm=self.length(row, "norm_radius", scale), terms=terms)

    def surface(self, row: _Row, label: str, has_curvature: bool, last: bool = False) -> IrSurface:
        problems_before = len(row.problems.items)
        kind = _surface_type(row)
        self.zero_only(row)
        radius = self.radius(row, has_curvature)
        conic = row.number("conic", what="a conic constant")
        if conic is not None and conic == 0:
            conic = None
        aspheric = self.aspheric(row)
        linear = row.number("linear_sag", what="an axicon term (the cone slope dz / dr, dimensionless)")
        if linear is not None and linear == 0:
            linear = None
        axis_raw = row.raw("cylinder_axis").upper()
        axis = None
        if axis_raw in ("X", "Y"):
            axis = axis_raw
        elif axis_raw != "":
            row.problems.add(E.VALUE_INVALID, f"cylinder_axis {row.raw('cylinder_axis')!r} must be X or Y (the axis the cylinder is "
                             "straight along)", row.at("cylinder_axis"))
        if kind in CYLINDER_TYPES and axis is None and axis_raw == "":
            row.problems.add(E.FIELD_MISSING, "a cylindrical surface needs cylinder_axis (X or Y)", row.at("cylinder_axis"))
        shape = self.shape(row, kind, radius, conic)
        if shape is not None:
            if axis is not None or linear is not None:
                row.problems.add(E.UNSUPPORTED_FEATURE, f"{SAG_COMBINATION_NOT_SUPPORTED}: surface type {row.raw('type')} does not combine with "
                                 "cylinder_axis or linear_sag", row.at("type"))
            if shape.kind in ("BICONIC", "TOROIDAL"):
                if aspheric:
                    row.problems.add(E.UNSUPPORTED_FEATURE, f"{SAG_COMBINATION_NOT_SUPPORTED}: surface type {row.raw('type')} replaces the even-asphere "
                                     "terms a4 ..; a toroid's curve takes its polynomial in sag_terms", row.at("type"))
                radius, conic = None, None  # (the shape's own section)
        if len(row.problems.items) == problems_before and shape is None:
            contradiction = _type_contradiction(kind, radius, conic, aspheric, axis, linear)
            if contradiction:
                row.problems.add(E.VALUE_INVALID, f"type {row.raw('type')} contradicts this row: {'; '.join(contradiction)}. Correct the "
                                 "type or the values (or leave type blank: the values decide the profile)", row.at("type"))
        thickness_raw = row.raw("thickness")
        thickness = None
        if last:
            # the last surface's thickness only places the image (ir.py): any finite value, blank, or inf (afocal)
            if thickness_raw.lower() in INFINITY_WORDS:
                self.reader.image_at_infinity = True
            else:
                thickness = row.number("thickness", what="the distance from the last surface to the image (a number, inf for an "
                                       "image at infinity, or blank when the prescription states none)")
        elif thickness_raw.lower() in INFINITY_WORDS:
            row.problems.add(E.THICKNESS_INVALID, "only the object surface, and the last surface (an image at infinity), may be at an "
                             "infinite distance", row.at("thickness"))
        else:
            thickness = row.number("thickness", what="the distance to the next surface")
            if thickness is not None and not thickness > 0:
                row.problems.add(E.THICKNESS_INVALID, f"thickness {thickness_raw} must be positive (surfaces follow each other along +Z)",
                                 row.at("thickness"))
                thickness = None
        clear = row.positive("semi_diameter", required=True, what="the clear semi-diameter (radius of the clear aperture)")
        mechanical = row.positive("mech_semi_diameter")
        coating_raw = row.raw("coating").upper()
        coating = None if coating_raw in UNCOATED_WORDS else CoatingRef(coating_raw, row.at("coating"))
        comment = row.raw("comment") or None
        return IrSurface(
            label=label, location=row.location, clear_semi_diameter_mm=self.length(row, "semi_diameter", clear) or Decimal(0),
            radius_mm=radius, conic=conic, aspheric_mm=aspheric, linear_sag=linear, cylinder_axis=axis, sag=shape,
            thickness_mm=self.length(row, "thickness", thickness), material=self.material(row),
            mechanical_semi_diameter_mm=self.length(row, "mech_semi_diameter", mechanical),
            is_stop=label.upper() in STOP_LABELS or _flag(row, "stop"), coating=coating, comment=comment,
        )

    def object_space(self, row: _Row) -> ObjectSpace:
        raw = row.raw("thickness")
        if raw.lower() in INFINITY_WORDS:
            space = ObjectSpace(location=row.location, infinite=True)
        else:
            distance = row.positive("thickness", what="the object distance (or inf)")
            space = ObjectSpace(location=row.location, distance_mm=self.length(row, "thickness", distance))
        if row.raw("material").upper() not in AIR_NAMES:
            row.problems.add(E.UNSUPPORTED_FEATURE, "the object space must be air (an immersed object is not imported)", row.at("material"))
        return space


def _materials(reader: _Reader, wavelengths_in: str = WAVELENGTHS_VACUUM) -> tuple[MaterialDefinition, ...]:
    definitions: list[MaterialDefinition] = []
    seen: dict[str, int] = {}
    for row in reader.rows(MATERIALS, MATERIAL_COLUMNS, MATERIAL_ALIASES, MATERIAL_REQUIRED):
        raw_name = row.raw("name")
        name = raw_name.upper()
        if raw_name == "":
            row.problems.add(E.FIELD_MISSING, "each material needs a name", row.at("name"))
            continue
        row.location = SourceLocation(row=row.location.row, section=MATERIALS, surface=None)
        if name == "AIR" or not MATERIAL_ID_PATTERN.match(name) or len(name) > MATERIAL_ID_MAX:
            row.problems.add(E.MATERIAL_INVALID, f"material name {raw_name!r} must be letters, digits, '.', '_' or '-' (starting with a "
                             f"letter or digit, at most {MATERIAL_ID_MAX}) and not AIR", row.at("name"))
            continue
        if name in seen:
            row.problems.add(E.MATERIAL_INVALID, f"material {name} is defined twice (first on row {seen[name]})", row.at("name"))
            continue
        seen[name] = row.location.row or 0
        kind = row.raw("kind").upper().replace(" ", "_").replace("-", "_")
        has_index = row.raw("index") != ""
        has_sellmeier = any(row.raw(c) != "" for c in SELLMEIER_COLUMNS)
        if kind == "":
            kind = "CONSTANT_INDEX" if has_index and not has_sellmeier else "SELLMEIER" if has_sellmeier and not has_index else ""
        elif kind == "CONSTANT":
            kind = "CONSTANT_INDEX"
        dispersion: SellmeierDefinition | ConstantIndexDefinition | None = None
        if kind == "SELLMEIER":
            if has_index:
                row.problems.add(E.MATERIAL_INVALID, "a SELLMEIER material has b1..b3 and c1..c3, not index", row.at("index"))
            values = [row.number(c, required=True, what="a Sellmeier coefficient") for c in SELLMEIER_COLUMNS]
            if all(v is not None for v in values):
                dispersion = SellmeierDefinition(b=tuple(values[:3]), c_um2=tuple(values[3:]))  # type: ignore[arg-type]
        elif kind == "CONSTANT_INDEX":
            if has_sellmeier:
                row.problems.add(E.MATERIAL_INVALID, "a CONSTANT_INDEX material has index, not Sellmeier coefficients", row.at("kind"))
            index = row.positive("index", required=True, what="the refractive index")
            if index is not None:
                dispersion = ConstantIndexDefinition(index)
        else:
            row.problems.add(E.MATERIAL_INVALID, f"kind {row.raw('kind')!r} is not SELLMEIER or CONSTANT_INDEX" if row.raw("kind")
                             else "give kind SELLMEIER (b1..b3, c1..c3) or CONSTANT_INDEX (index)", row.at("kind"))
        low = row.positive("min_nm", required=True, what="the shortest valid wavelength in nm")
        high = row.positive("max_nm", required=True, what="the longest valid wavelength in nm")
        if low is not None and high is not None and not low < high:
            row.problems.add(E.MATERIAL_INVALID, "min_nm must be smaller than max_nm", row.at("max_nm"))
        absorption = row.number("absorption_per_mm", what="the bulk absorption coefficient per mm")
        if absorption is not None and absorption < 0:
            row.problems.add(E.MATERIAL_INVALID, "absorption_per_mm cannot be negative", row.at("absorption_per_mm"))
        texts: dict[str, str | None] = {}
        for column, limit in (("display_name", NAME_MAX), ("basis", TEXT_MAX)):
            text = row.raw(column) or None
            if text is not None and (not PUBLIC_TEXT_PATTERN.match(text) or len(text) > limit):
                detail = _not_ascii(text)
                row.problems.add(E.VALUE_INVALID, f"{column} must be 1 to {limit} printable ASCII characters{f': {detail}' if detail else ''}",
                                 row.at(column))
                text = None
            texts[column] = text
        law = row.raw("index_reference").strip().upper()
        if law == "":
            law = LAW_VACUUM
            if wavelengths_in == WAVELENGTHS_AIR:
                reader.warnings.append(ImportWarning(
                    "MATERIAL_CONVENTION", f"material {name}: the file writes air wavelengths (wavelengths_in = AIR), but a [materials] "
                    "law states its own convention: index_reference is blank, so its law is read on vacuum wavelengths with an "
                    "absolute index (and min_nm / max_nm as vacuum wavelengths); write index_reference AIR for a glass-catalogue law "
                    "(air wavelengths, index relative to air at 20 C / 101325 Pa)", row.at("index_reference")))
        elif law not in (LAW_VACUUM, LAW_AIR):
            row.problems.add(E.MATERIAL_INVALID, f"index_reference {row.raw('index_reference')!r} is not VACUUM or AIR",
                             row.at("index_reference"))
        if dispersion is not None and low is not None and high is not None:
            definitions.append(MaterialDefinition(
                name=name, dispersion=dispersion, min_nm=low, max_nm=high, location=row.at("name"), absorption_per_mm=absorption,
                display_name=texts["display_name"], basis=texts["basis"], index_reference=law))
    return tuple(definitions)


def _coatings(reader: _Reader, air: ReferenceEnvironment | None = None) -> tuple[CoatingDefinition, ...]:
    """The ``[coatings]`` table: rows grouped by coating name, in first-seen order (each definition is checked against the
    coating contract when it is resolved, ``glass.py``). ``air``: the file writes air wavelengths of this environment
    (each node's wavelength is converted to the vacuum wavelength, to ``VACUUM_WAVELENGTH_QUANTUM_NM``)."""

    groups: dict[str, list[_Row]] = {}
    for row in reader.rows(COATINGS, COATING_COLUMNS, COATING_ALIASES, COATING_REQUIRED):
        raw_name = row.raw("name")
        row.location = SourceLocation(row=row.location.row, section=COATINGS)
        if raw_name == "":
            row.problems.add(E.FIELD_MISSING, "each coating row needs the coating's name", row.at("name"))
            continue
        name = raw_name.upper()
        if name in UNCOATED_WORDS or not MATERIAL_ID_PATTERN.match(name) or len(name) > COATING_ID_MAX:
            row.problems.add(E.COATING_INVALID, f"coating name {raw_name!r} must be letters, digits, '.', '_' or '-' (starting with a "
                             f"letter or digit, at most {COATING_ID_MAX}) and not {', '.join(sorted(w for w in UNCOATED_WORDS if w))}",
                             row.at("name"))
            continue
        groups.setdefault(name, []).append(row)
    definitions: list[CoatingDefinition] = []
    for name, rows in groups.items():
        attributes: dict[str, str] = {}
        for row in rows:
            for column in COATING_ATTRIBUTE_COLUMNS:
                value = row.raw(column)
                if value == "":
                    continue
                if column in attributes and attributes[column] != value:
                    row.problems.add(E.COATING_INVALID, f"coating {name}: {column} is {attributes[column]!r} on an earlier row and "
                                     f"{value!r} here; state it once (on the coating's first row)", row.at(column))
                attributes.setdefault(column, value)
        first = rows[0]
        if "kind" not in attributes:
            first.problems.add(E.FIELD_MISSING, f"coating {name} needs its kind (e.g. ANTI_REFLECTION, BEAMSPLITTER) on its first row",
                               first.at("kind"))
        texts: dict[str, str | None] = {}
        for column, limit in (("display_name", NAME_MAX), ("basis", TEXT_MAX)):
            text = attributes.get(column)
            if text is not None and (not PUBLIC_TEXT_PATTERN.match(text) or len(text) > limit):
                detail = _not_ascii(text)
                first.problems.add(E.VALUE_INVALID, f"{column} must be 1 to {limit} printable ASCII characters{f': {detail}' if detail else ''}",
                                   first.at(column))
                text = None
            texts[column] = text
        aoi_limits: list[Decimal | None] = []
        for column in ("min_aoi_deg", "max_aoi_deg"):
            raw = attributes.get(column)
            value = _decimal(raw) if raw is not None else None
            if raw is not None and value is None:
                first.problems.add(E.VALUE_INVALID, f"{column} {raw!r} is not a number of degrees", first.at(column))
            aoi_limits.append(value)
        nodes = []
        for row in rows:
            wavelength = row.positive("wavelength_nm", required=True, what="the wavelength in nm" if air is not None else "the vacuum wavelength in nm")
            if wavelength is not None and air is not None:
                vacuum = reader.to_vacuum(wavelength, air, row.at("wavelength_nm"), "wavelength_nm")
                wavelength = None if vacuum is None else vacuum.quantize(VACUUM_WAVELENGTH_QUANTUM_NM)
            aoi = row.number("aoi_deg", what="the angle of incidence in degrees")
            values = {column: row.number(column, what="a power fraction") for column in COATING_VALUE_COLUMNS}
            phases = {column: row.number(column, what="a phase in radians") for column in COATING_PHASE_COLUMNS}
            if wavelength is None:
                continue
            nodes.append(CoatingNode(
                wavelength_nm=wavelength, location=row.at("wavelength_nm"), aoi_deg=aoi, **values,  # type: ignore[arg-type]
                reflection_s_rad=phases["reflection_phase_s"], reflection_p_rad=phases["reflection_phase_p"],
                transmission_s_rad=phases["transmission_phase_s"], transmission_p_rad=phases["transmission_phase_p"]))
        definitions.append(CoatingDefinition(
            name=name, kind=attributes.get("kind", "").upper(), nodes=tuple(nodes), location=first.at("name"),
            min_aoi_deg=aoi_limits[0], max_aoi_deg=aoi_limits[1], aoi_reference=attributes.get("aoi_reference", "").upper() or None,
            status=attributes.get("status", "").upper() or None, display_name=texts["display_name"], basis=texts["basis"]))
    return tuple(definitions)


def read_csv_surface_table(text: str, *, default_unit: str = "mm", requested_unit: str | None = None,
                           ignore_columns: tuple[str, ...] = ()) -> Prescription:
    """Read a ``csv-surface-table/1`` document into the prescription IR (``PrescriptionImportError`` lists every problem).

    ``default_unit``: the length unit when the file states none. ``requested_unit``: a unit the caller asked for
    explicitly; a file that states another unit is refused (``UNITS_CONFLICT``) rather than silently read in either.
    ``ignore_columns``: extra columns (of any table) not read, beside those the file's ``ignore_columns`` directive names;
    an unknown column blank in every row is skipped too, and both are listed in ``Prescription.ignored_columns``."""

    if not text.strip():
        raise refuse(E.TEXT_INVALID, "the prescription is empty")
    reader = _Reader(text, default_unit, requested_unit, ignore_columns)
    reader.split()
    reader.check_format()
    reader.ignored_names()
    unit = reader.unit()
    name = reader.name()
    band = reader.band()
    edge_limits = reader.edge_limits()
    wavelengths_in = reader.wavelengths_in()
    environment = reader.environment(wavelengths_in)
    glass_catalogs = reader.glass_catalogs()
    air = environment if wavelengths_in == WAVELENGTHS_AIR and environment is not None and environment.pressure_pa != 0 else None
    if band is not None and air is not None:
        where = reader.directives["wavelength_min_nm"][1]
        low = reader.to_vacuum(band[0], air, where, "wavelength_min_nm")
        high = reader.to_vacuum(band[1], air, reader.directives["wavelength_max_nm"][1], "wavelength_max_nm")
        if low is not None and high is not None:
            converted = (low.quantize(BAND_QUANTUM_NM, rounding=ROUND_CEILING), high.quantize(BAND_QUANTUM_NM, rounding=ROUND_FLOOR))
            if converted[0] > converted[1]:  # (a band narrower than the rounding: keep the exact conversion)
                converted = (low.quantize(VACUUM_WAVELENGTH_QUANTUM_NM), high.quantize(VACUUM_WAVELENGTH_QUANTUM_NM))
            reader.warnings.append(ImportWarning(
                "WAVELENGTHS_CONVERTED", f"the band {band[0]}-{band[1]} nm is written in air (wavelengths_in = AIR, air at "
                f"{environment.temperature_c} C and {environment.pressure_pa} Pa) and is read as the vacuum band "  # type: ignore[union-attr]
                f"{converted[0]}-{converted[1]} nm (shared air model, rounded inward to {BAND_QUANTUM_NM} nm)", where))
            band = converted
        else:
            band = None
    if SURFACES not in reader.tables:
        reader.problems.add(E.SURFACE_ORDER_INVALID, "no surface table: give a header row and one row per surface")
        reader.problems.raise_if_any()
    rows = reader.rows(SURFACES, SURFACE_COLUMNS, SURFACE_ALIASES, SURFACE_REQUIRED)
    header = reader.tables[SURFACES][0]
    header_names = {SURFACE_ALIASES.get(_header_name(c), _header_name(c)) for c in header}
    has_curvature = "curvature" in header_names
    header_where = reader.tables[SURFACES][1]
    if has_curvature and "radius" in header_names:
        reader.problems.add(E.COLUMN_DUPLICATE, "give radius or curvature, not both", header_where)
    elif not has_curvature and "radius" not in header_names:
        reader.problems.add(E.COLUMN_MISSING, "[surfaces] needs a 'radius' (or 'curvature') column", header_where)
    surfaces_reader = _SurfaceReader(reader, unit)
    surfaces: list[IrSurface] = []
    surface_rows: list[_Row] = []
    object_space: ObjectSpace | None = None
    image: SourceLocation | None = None
    labels: dict[str, int] = {}
    real_rows = [r for r in rows if r.raw("surface") != "" and r.raw("surface").upper() not in OBJECT_LABELS | IMAGE_LABELS]
    last_row = real_rows[-1] if real_rows else None
    for position, row in enumerate(rows):
        label = row.raw("surface")
        if label == "":
            row.problems.add(E.FIELD_MISSING, "each surface row needs its label in the surface column", row.at("surface"))
            continue
        problem = _not_ascii(label)
        if problem is not None or len(label) > NAME_MAX:
            row.problems.add(E.VALUE_INVALID, f"surface label {label!r} must be 1 to {NAME_MAX} printable ASCII characters (letters, "
                             f"digits, punctuation, spaces){f': {problem}' if problem else ''}",
                             SourceLocation(row=row.location.row, column="surface", section=SURFACES))
            continue
        row.location = SourceLocation(row=row.location.row, surface=label, section=SURFACES)
        key = label.upper()
        if key in labels:
            row.problems.add(E.SURFACE_ORDER_INVALID, f"surface label {label!r} is used twice (first on row {labels[key]})", row.at("surface"))
            continue
        labels[key] = row.location.row or 0
        if image is not None:
            row.problems.add(E.SURFACE_ORDER_INVALID, "the image surface must be the last row", row.at("surface"))
            continue
        if key in OBJECT_LABELS:
            if position != 0:
                row.problems.add(E.SURFACE_ORDER_INVALID, "the object surface must be the first row", row.at("surface"))
                continue
            object_space = surfaces_reader.object_space(row)
            continue
        if key in IMAGE_LABELS:
            image = row.location
            continue
        surfaces.append(surfaces_reader.surface(row, label, has_curvature, last=row is last_row))
        surface_rows.append(row)
    if not surfaces and not reader.problems:
        reader.problems.add(E.SURFACE_ORDER_INVALID, "the surface table has no real surface (only object / image rows)", header_where)
    for row in surface_rows[:-1]:
        if row.raw("thickness") == "":
            reader.problems.add(E.FIELD_MISSING, "thickness is required here; give the distance to the next surface", row.at("thickness"))
    materials = _materials(reader, wavelengths_in)
    coatings = _coatings(reader, air)
    if air is not None and any(c.nodes for c in coatings):
        reader.warnings.append(ImportWarning(
            "WAVELENGTHS_CONVERTED", f"the [coatings] wavelengths are written in air (wavelengths_in = AIR, air at {air.temperature_c} C "
            f"and {air.pressure_pa} Pa) and are read as vacuum wavelengths (shared air model, to {VACUUM_WAVELENGTH_QUANTUM_NM} nm)",
            SourceLocation(section=COATINGS)))
    reader.problems.raise_if_any()
    return Prescription(
        source_format=FORMAT_ID, source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), input_unit=unit,
        surfaces=tuple(surfaces), name=name, object=object_space, image=image, materials=materials, coatings=coatings, band_nm=band,
        warnings=tuple(reader.warnings), image_at_infinity=reader.image_at_infinity, edge_limits=edge_limits,
        wavelengths_in=wavelengths_in, environment=environment, glass_catalogs=glass_catalogs,
        ignored_columns=tuple(reader.ignored),
    )


__all__ = ["FORMAT_ID", "read_csv_surface_table"]
