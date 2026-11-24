# Composing optical systems from reusable layouts

Large optical systems are easier to build as modules: design and validate each stage on its own, then import the stages
into one system, arrange them, and validate the whole system. OpenSci does this with **workspaces**.

A workspace (`opensci.optical_workspace` 1.0, `json_api.json_schema("optical-workspace")`) is a Project together with:

| part | what it holds |
|---|---|
| `project` | the ordinary Project: components, observation planes, targets, declared models. It runs, validates and exports exactly as on its own; its hash never depends on the rest of the workspace. |
| `module` | what makes the Project reusable: `module_id`, `title`, `revision`, its test fixtures, its optical interfaces and the scope of its requirements. |
| `instances` | every imported module: its source (`module_id`, `module_digest`, revision), where it was placed, how each source id maps to a Project id, what happened to each fixture, the content digest of each imported member, the module's interface ports (as the module states them) and the module's own validations. |
| `groups` | editing groups: named, possibly overlapping sets of components and observation planes, each with an optional reference frame. |
| `operations` | the journal of the last 256 operations (what makes retries safe). |
| `validations` | system runs of this workspace's Project, with their Project hash and Engine input hash. |

The same operations are available as Python functions (`os_optics.composition`), as JSON calls
(`os_optics.local.json_api.workspace_*`, also used by a browser (Pyodide) host) and as commands
(`os-optics workspace ...`).

## 1. Make a module

Author the stage as an ordinary Project, then describe it as a module:

```json
{
  "module_id": "fold-90", "title": "90 degree fold", "revision": "r1",
  "fixtures": [
    {"object_kind": "COMPONENT",   "object_id": "SRC",   "role": "TEST_SOURCE",      "default_action": "REMOVE"},
    {"object_kind": "OBSERVATION", "object_id": "OUT",   "role": "TEST_OBSERVATION", "default_action": "REMOVE"},
    {"object_kind": "TARGET",      "object_id": "T_OUT", "role": "TEST_TARGET",      "default_action": "REMOVE"}
  ],
  "interfaces": [
    {"port_id": "IN",  "role": "INPUT",  "point_mm": [-20, 0, 0], "direction": [1, 0, 0], "beam_reference": "SRC"},
    {"port_id": "OUT", "role": "OUTPUT", "point_mm": [0, 20, 0],  "direction": [0, 1, 0]}
  ],
  "requirements": [{"target_id": "T_MON_CENTER", "scope": "CARRY_TO_PARENT"}]
}
```

* **Fixtures** are objects that exist only to test the module on its own: a test source, a terminal test screen, a test
  target, a test element. An import retains or removes each one (`default_action`), so the system never gets a second
  light source or a beam stop at an internal boundary by accident.
* **Interfaces** state where light enters and leaves, in the module's design frame (`beam_reference`: the source whose
  beam defines the boundary condition). An import reports them in the system's frame.
* **Requirements**: `CARRY_TO_PARENT` targets are copied into the system (renamed with the instance); `MODULE_ONLY`
  targets stay with the module. Targets not listed are carried.

The descriptor's fields (`ModuleDescriptorV1`; every id it names is an object of the module's Project):

| field | entries | values |
|---|---|---|
| `module_id`, `title` | required | an id and a name; `revision` (a label such as `r1`) and `description` are optional |
| `fixtures[]` | `object_kind`, `object_id`, `role`, `default_action` | `role` `TEST_SOURCE` (a source component), `TEST_ELEMENT` (any other component), `TEST_OBSERVATION` (an observation plane), `TEST_TARGET` (a target); `object_kind` matches it (`COMPONENT`, `OBSERVATION`, `TARGET`); `default_action` `REMOVE` or `RETAIN` |
| `interfaces[]` | `port_id`, `role`, `point_mm`, `direction` | `role` `INPUT` or `OUTPUT`; `direction` a unit vector in the module's design frame; optional `wavelength_band_nm` `[min, max]`, `beam_reference` (a source component id), `aperture_radius_mm`, `description` |
| `requirements[]` | `target_id`, `scope` | `scope` `CARRY_TO_PARENT` or `MODULE_ONLY` |

In the example, the module's own laser `SRC`, its screen `OUT` and the power target `T_OUT` on that screen are test
fixtures an import removes; the centroid target `T_MON_CENTER` on the monitor plane `MON` (not a fixture) travels into the
system. `IN` says the light the fold expects arrives 20 mm before the mirror along +X, shaped like `SRC`'s beam; `OUT`
says it leaves 20 mm after the mirror along +Y. Imported with `--rebind SRC=LASER`, the system keeps the monitor plane
and its carried target (`<key>_MON`, `<key>_T_MON_CENTER`), reads the system's `LASER` wherever a target used `SRC`, and
lists `IN` / `OUT` in the system's frame.

```
os-optics workspace init fold.json --module fold.module-descriptor.json -o fold.ws.json
os-optics workspace run fold.ws.json            # the module's own validation
os-optics workspace export fold.ws.json --descriptor fold.module-descriptor.json -o fold.module.json
```

## 2. Import modules into a system

```
os-optics workspace init system.json -o system.ws.json
os-optics workspace import system.ws.json fold.module.json --key fold --rebind SRC=LASER --op-id imp-fold
os-optics workspace import system.ws.json lens.module.json --key focus --rebind SRC=LASER \
    --position 0,120,0 --quaternion 0.7071067811865476,0,0,0.7071067811865476 --op-id imp-focus
```

Each import is a new, independent **instance**:

* components, observation planes and targets get the id `<instance_key>_<id>` (`fold_M1`), and so do the optical
  models, materials and coatings the instance uses (`<INSTANCE_KEY>_<ID>` for materials and coatings): each instance
  owns its copy, also when the Project already has an identical definition, so editing one instance's model never
  changes another instance or the system's own elements. An id that would be too long is refused (`ID_TOO_LONG`) with
  the id;
* every pose of the module's design frame is placed with `placement` (`--position`, `--quaternion`); an observation plane
  attached to a component stays attached;
* fixtures are retained, removed or **rebound**: `--rebind SRC=LASER` removes the module's test source and makes the
  targets that used it read the system's laser (reported as `TARGET_REBOUND`: they now read that source's wavelength,
  power and coherence group). A target that uses a removed fixture without a rebind is refused
  (`FIXTURE_REBIND_REQUIRED`) unless you leave it out (`"unbound_targets": "remove"`). Full form: `--fixtures
  '{"policy": "module_defaults" | "retain_all" | "remove_all", "overrides": {"OUT": "RETAIN"}, "rebind": {"SRC": "LASER"},
  "unbound_targets": "refuse" | "remove"}'`;
* sources and absorbing planes a module does not mark as fixtures are imported and reported
  (`UNMARKED_SOURCE_IMPORTED`, `ABSORBING_PLANE_IMPORTED`); an absorbing plane attached to the module's own detector or
  beam dump (`device_instance_id` of a `detector` / `dump` component) is that instrument's sensing face and is imported
  without a warning;
* when catalog components would pin different catalog releases, they are moved to the active release if every product is
  published there unchanged; otherwise the import is refused (`CATALOG_BINDING_CONFLICT`) listing each component and SKU;
* the module's own validations travel with the instance; they never count as validations of the system;
* one editing group per import (id: the instance key) holds its members, with the placement as its frame.

The result lists the id map, the fixture outcomes, the carried requirements, the interfaces in the system's frame and
every warning. Importing the same module twice needs two keys; the source module file is never changed, and editing one
instance changes neither the source nor the other instances.

A composed workspace exports as a module again; importing it keeps its inner instances (`<key>_<inner key>`, each naming
the instance that brought it in as `parent_instance`) and groups (`<key>_<group id>`) with their provenance.

## 3. Select, group, move

Selectors combine `ids`, `groups`, `instances` and `kinds`; the result is the union, each object once:

```
os-optics workspace select system.ws.json --instance fold --ids LASER
os-optics workspace group create system.ws.json train --ids fold_M1,focus_L1 --frame component:fold_M1 --op-id group-train
os-optics workspace transform system.ws.json --group train --axis 0,0,1 --angle 15 \
    --frame group:train --pivot component:fold_M1 --op-id turn-train
```

* Groups never change the Project: create, dissolve and redefine them freely, across imports and with elements you
  authored directly. A later transform acts on the group's current members.
* `transform` moves the selection as one rigid body: `x -> R (x - pivot) + pivot + t`. The translation and the rotation
  axis are given in the axes of `frame` (`design`, `group:ID`, `component:ID`, `instance:KEY`); the pivot is
  `frame_origin` (default), `selection_centroid`, `component:ID` or a point `X,Y,Z` in the frame. Rotations are
  `--axis/--angle` (degrees, right-hand rule) or `--rotation-quaternion W,X,Y,Z`.
* Each selected object moves once, whatever the number of selectors that name it. Objects outside the selection never
  move. An observation plane attached to a moved component moves with it, selected or not (listed in `followers`, with
  `selected`); selected without its component, it moves on its own and stays attached. An instance's placement and a group's frame move with the selection when all their
  members moved, and a whole-instance move does not count as an edit of the instance. The instances nested in an
  instance that moves as a whole move with it, also one whose members were all deleted: replacing it later with
  `--discard-edits` puts its members where the enclosing instance's module has them, and the enclosing instance holds
  them as imported again. Its import group's frame moves with it when the group holds
  nothing else; a group that also holds objects of your own keeps the group rule (its frame moves when all its members
  move).
* The result lists every moved object's pose before and after, in the design frame.
* Rotation arithmetic uses `+ - * /` and `sqrt` only: quarter turns are exact, other angles use fixed polynomials, so
  Python and a browser (Pyodide) host write the same Project bit for bit.

## 4. Safe retries and atomic changes

Every changing operation either applies completely or is refused with structured problems and changes nothing.

* `--op-id` / `op_id`: name each intended operation. Repeating it with the same request returns the recorded result and
  changes nothing (`replayed: true`); the same id with another request is refused (`OP_ID_REUSED`). After a lost
  response, retry with the same op id, or ask `os-optics workspace history WS --op-id ID`. An operation without an
  op id is always applied as a new operation (the journal records it under a generated `auto-...` id, returned as
  `result.op_id`), so only an op id you chose makes a retry safe.
* `--expect-base` / `expected_base`: the `workspace_digest` you read; a new operation on a changed workspace is refused
  (`BASE_CHANGED`, with the current digest). The digest covers the organization and the physics -- the Project, the
  module descriptor, the instances (without the module validations they carry) and the groups -- never the journal or a
  validation record, so the same composition has the same digest whichever runtime validated its modules.
* An import's `instance_key` separates a retry from a second instance: the same module at the same placement under an
  existing key answers with the existing instance; a second instance needs a new key.
* An operation is applied whatever the local check says of its result, and the reply's `result.check` reports it: `ok`
  (a run accepts the result), `new_problems` (what the operation introduced, for example two optical bodies overlapping)
  and `carried_problems` (what the workspace already had). `--require-check` / `require_check: true` refuses an operation
  that introduces a problem (`CHECK_PROJECT_REFUSED`, with the check's problems); it is an option of the call and not
  part of the operation, so a retry under the same op id is answered from the journal with or without it. `workspace
  batch` applies several operations as one: its final state is checked, so its steps may pass through overlapping
  arrangements also when the check is required.
* `workspace replace KEY MODULE` swaps an instance for another revision of its module at the same placement and keeps
  group memberships by source id; the instances it brought in from a composed module are replaced with it. Any instance
  can be replaced, also one of several instances of the same module. It is refused (`INSTANCE_EDITED`, listing the
  edits) when the instance or an instance nested in it has local edits -- members changed or deleted since the import,
  members a nested instance gained when it was replaced on its own, or objects added back that a revision lacks --
  unless `--discard-edits` (`edits_discarded` lists them; every member comes back exactly as the new revision has it,
  deleted members too, and of what the replaced instances held only the definitions that objects outside them use stay
  behind, as the next item states). `status` lists the same edits under each instance (`edited`, `change`: `EDITED`,
  `REMOVED` or `ADDED`).
* A definition (optical model, material, coating) of the replaced instances that an object outside them uses -- your
  own component pointed at an instance's mirror model, or another instance's -- is never changed or removed by a
  replacement: it stays with its current content, edits included, so its edits are not edits of the instance. The
  result lists it with the objects outside the instance that use it, directly or through another of its definitions
  (`kept_definitions`: `{kind, id, used_by: [{kind, id}]}`), and `status` lists the same per instance. A new revision
  with the same definition (same id and content) uses it as imported, so replacing an unedited instance by its own
  revision changes nothing here either. A new revision that lacks it leaves it to the instance (and to the nested
  instance it came from): no edit while those objects use it, a member added to the instance (`ADDED`) once nothing
  outside uses it, which replacing the instance with `--discard-edits` removes. A definition removed that way leaves
  the records at every depth, also those of the instances enclosing the replaced one (unless an enclosing module has it
  in its own import, which then records it as deleted), so a later revision of the nested instance may define it again
  (recorded as added in the enclosing instances, like any member a nested revision adds). The instances enclosing a nested
  instance hold such a definition as they imported it, also when the objects that use it belong to the enclosing
  instance's module: replacing the nested instance by a revision that changes no object leaves them unedited, and a
  composed module exported in this state is imported with the definition kept the same way. A new revision that
  defines the same id differently is refused (`ID_CONFLICT`, with `used_by`), with or without `--discard-edits`, since
  one id names one definition: point those objects at another definition first.
* Replacing an instance nested in another one on its own changes the enclosing instance, so it is recorded there: a
  member the new revision adds is recorded as added (`ADDED`) and is held by the enclosing instance like its other
  members (selected, moved and grouped with it), a member it no longer has as deleted, a changed one as edited. A member
  the new revision lacks stays part of the innermost instance whose revision lacks it, at any depth, and of every
  instance between that one and the replaced one: replacing any of them by a revision that has the member brings it
  back (also the unedited nested instance by its original revision), and an object added back with its id is a member
  added to them (`ADDED`, also an edit of the instances enclosing them). Only that part comes back this way: the id of
  a deleted member of the enclosing module itself stays reserved, also when it lies in the nested instance's namespace
  (`ID_CONFLICT`). The enclosing instance stays edited until it is replaced with `--discard-edits` or the nested
  instance goes back to the revision the enclosing module has: for each component or observation plane a nested
  revision changed or left out, the enclosing record names the module object and the placement the nested instance had
  (`nested_revision_digest`), and a later replacement that re-creates that object at that placement makes the member
  the enclosing instance's import content again, also when the enclosing instance was turned before the nested
  replacement (the re-created member is placed afresh, which after a rotation differs from the turned content by
  floating-point rounding alone). A nested replacement that changes no object (the same content under another revision
  name) is no edit of the enclosing instance; replacing the enclosing instance then puts the nested instance back at the
  revision the enclosing module has. A round trip with a whole move in between does not clear the edit: when the
  enclosing instance was moved or turned as a whole after the nested replacement, it stays edited after the nested
  instance is back at the revision the enclosing module has. Each component or observation plane the nested revision
  lacked or changed is listed as `EDITED` (the import content moved with the instance is not known), and replacing the
  enclosing instance needs `--discard-edits`, which clears it. The same holds when the nested instance was moved on its
  own.
* What a replacement writes is decided by exact identity, never by a numeric tolerance: every member gets the new
  revision's content, so a change of a pulse duration from 100 fs to 1 ps or of a power from 100 pW to 900 pW reaches
  the Project, and an edit as small as a sub-micrometre move is discarded. One exception keeps whole-instance moves
  from showing up as edits: an unedited component or observation plane whose module object the new revision has
  unchanged (the instance records the digest of the object each was copied from, and the new revision's object has the
  same digest) keeps its exact content, because after a whole-instance move or rotation a fresh placement of the same
  object differs from the moved content by floating-point rounding alone. So replacing an unedited instance by the
  revision it already has changes nothing (only the catalog binding may move catalog pins: a member that keeps its
  content takes the release pin the binding sets), the members a new revision keeps unchanged keep their bytes, also
  for the instances enclosing the replaced one, and a catalog re-pin stays no edit of the enclosing instance through
  nested replacements, also after a whole-instance rotation. The digest of a module object leaves its catalog release
  pin out, so a revision whose objects differ from the old one's only by their pin keeps every member exactly. A member
  a composed module's author edited before exporting it, and every member of a workspace written before instances
  recorded these digests, gets the revision's content.
* Replacing a nested instance re-creates its members from its revision at its unchanged placement. The instances
  enclosing it hold each of them as the nested instance imported it, so when the revision has the same module object
  (an edited or deleted member the replacement discards included), the member is their import content again:
  discarding an edit at any depth leaves every enclosing instance unedited, also after a whole-instance rotation, and
  also when the enclosing instance was moved as a whole after the edit (the nested instance moved with it). A nested
  instance moved on its own no longer sits where the enclosing import put it, so the enclosing instance stays edited.
* `workspace update-project WS PROJECT` takes a Project edited elsewhere (another tool, a browser host); groups drop the
  objects it no longer has. A member of an instance that it no longer has stays in the instance's record as a deleted
  member: `status` lists it under the instance's `edited` members (`change: REMOVED`), and the result reports it once
  (`removed_instance_members`, warning `INSTANCE_MEMBERS_REMOVED`). An object of the same kind and id added back is
  that member again, edited or not by its content, and rejoins the instance's group (`restored_instance_members`);
  meanwhile no import takes its id. Moving the whole instance is no edit of its unedited members; an edited or deleted
  component or observation plane stays an edit through such a move (its moved import content is not known), also when
  it is added back or a later move brings it back to its old place, until a replacement with `--discard-edits` of the
  instance (or of the nested instance it belongs to) re-creates it. Moving a catalog component to another catalog
  release with the same product is not an edit of its instance.

## 5. Validate the system

```
os-optics workspace run system.ws.json --out-dir run
os-optics workspace status system.ws.json
```

`run` simulates the whole Project through the normal local run (the same Engine and checks as any Project) and records
a system validation. `status` derives each validation's freshness: `CURRENT` while the Project hash and the Engine input
hash are the ones it ran on; otherwise `HISTORICAL` with the reasons `PHYSICS_CHANGED`, `TARGETS_CHANGED`,
`OTHER_PROJECT_CHANGED` (only another Project field changed, such as the design intent), `CATALOG_CHANGED` or
`ENGINE_CHANGED`. Regrouping keeps a validation current; moving an element or changing a target
makes it historical until the system is run again. A module's own validations are listed with its instance
(`MODULE_SCOPE`): a module that passes on its own says nothing about the system. Each instance also lists its module's
interface ports (`interfaces`: `point_mm`, `direction`, `wavelength_band_nm`, `aperture_radius_mm`, and `beam_reference`
as the Project component it became, `null` once that component is deleted) in the Project's design frame at the
instance's current placement, so after a transform of the whole instance they are where the instance now is; a composed
module's nested instances report theirs through both placements. `workspace record WS RUN_DIR` records a
run made elsewhere (for example in a browser host) when it ran on this exact Project.

The local trace budget (512 paths, 2048 trace steps, 64 interactions per path) is part of the Engine input and is the
same for composed systems: a system whose light needs more paths reports the incomplete evaluation (affected targets
`UNKNOWN`, a `MAX_PATHS` issue), never a pass for requirements it did not evaluate.

## Worked example (Python)

```python
import math
from os_optics.local import json_api as J

along_x = [0.5, 0.5, 0.5, 0.5]                  # local +Z -> design +X
fold_q = [0.2705980500730985, 0.2705980500730985, -0.6532814824381883, -0.6532814824381883]  # +X beam folded to +Y
laser = lambda iid, pos: {"instance_id": iid, "optical_role": "source",
    "custom_part": {"custom_part_key": "laser", "description": "632.8 nm laser"},
    "nominal_pose": {"position_mm": pos, "quaternion_wxyz": along_x},
    "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 632.8, "power_w": 0.001, "waist_radius_mm": 0.5,
                     "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}}

# a fold module: test laser, a native flat mirror, a monitor plane and a carried requirement
mirror = J.build_native_model("mirror.flat", {"diameter_mm": "25.4", "reflectance": "0.98", "valid_band_nm": [400, 700]})
fold = {"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM",
        "components": [laser("SRC", [-100, 0, 0]), {**mirror["component_template"], "instance_id": "M1",
                       "nominal_pose": {"position_mm": [0, 0, 0], "quaternion_wxyz": fold_q}}],
        "observations": [{"observation_id": "MON", "pose": {"position_mm": [0, 50, 0], "quaternion_wxyz": [0, 0, 0.7071067811865476, 0.7071067811865476]},
                          "aperture": {"shape": "circle", "radius_mm": 20}}],
        "targets": [{"target_id": "CENTER", "metric": "spot_centroid_u_mm", "observation_id": "MON", "target_value": 0, "tolerance_abs": 0.5}],
        "optical_models": [mirror["optical_model"]], "model_coatings": mirror["model_coatings"]}
descriptor = {"module_id": "fold-90", "title": "90 degree fold",
              "fixtures": [{"object_kind": "COMPONENT", "object_id": "SRC", "role": "TEST_SOURCE", "default_action": "REMOVE"}]}
module_ws = J.workspace_init(fold, {"module": descriptor})["workspace"]
module_ws = J.workspace_run(module_ws)["workspace"]                     # validated on its own
module = J.workspace_export_module(module_ws, descriptor)["module"]

# a system: its own laser, then two instances of the module; the second one 100 mm further along the folded beam
system = J.workspace_init({"schema": "opensci.project", "project_schema_version": "2.0", "design_intent": "OPTICAL_SYSTEM",
                           "components": [laser("LASER", [-100, 0, 0])]})["workspace"]
r = J.workspace_import(system, module, {"instance_key": "a", "fixtures": {"rebind": {"SRC": "LASER"}}, "op_id": "import-a"})
r = J.workspace_import(r["workspace"], module, {"instance_key": "b", "fixtures": {"rebind": {"SRC": "LASER"}}, "op_id": "import-b",
                       "placement": {"position_mm": [0, 100, 0], "quaternion_wxyz": [1, 0, 0, 0]}})
print(J.workspace_run(r["workspace"])["status"])  # the second fold faces the wrong way: its requirement fails
# turn the second fold by 90 degrees about its own mirror; a retry of the same op_id changes nothing
request = {"selector": {"instances": ["b"]}, "rotation": {"axis": [0, 0, 1], "angle_deg": 90}, "pivot": {"component": "b_M1"},
           "op_id": "turn-b", "expected_base": r["workspace_digest"]}
moved = J.workspace_transform(r["workspace"], request)
assert J.workspace_transform(moved["workspace"], request)["replayed"]
print([(a["id"], a["after"]["position_mm"]) for a in moved["result"]["affected"]])
run = J.workspace_run(moved["workspace"])
print(run["status"], [(t["target_id"], t["status"]) for t in run["summary"]["targets"]])
print([(v["freshness"], v["reasons"]) for v in J.workspace_status(run["workspace"])["status"]["validations"]])
```

## Reference

* Python: `os_optics.composition` (`import_layout`, `select`, `transform`, `group_create`, `group_dissolve`,
  `group_set_members`, `group_add`, `group_remove`, `replace_instance`, `update_project`, `batch`, `export_module`,
  `run_workspace`, `record_validation`, `workspace_status`, `history`); refusals raise `CompositionError` with
  `CompositionErrorCode` codes.
* JSON: `json_api.workspace_*` (module docstring of `os_optics.local.json_api`); `json_api.capabilities()["composition"]`
  lists the functions, vocabularies and codes.
* CLI: `os-optics workspace --help`.
