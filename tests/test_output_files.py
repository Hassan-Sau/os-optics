"""``os_optics.output_files``: the files of a regenerated output are written to temporary names and moved into place
together (``replace_files``, ``StagedFiles``)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from os_optics.output_files import StagedFiles, replace_files, staged_path


def _tree(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_replace_files_replaces_its_own_names_and_no_other(tmp_path):
    out = tmp_path / "out"
    assert replace_files(out, {"a.svg": b"one", "b.json": b"{}"}) == out
    (out / "notes.txt").write_bytes(b"mine")
    replace_files(out, {"a.svg": b"two", "c.pdf": b"%PDF"})
    assert _tree(out) == {"a.svg": b"two", "b.json": b"{}", "c.pdf": b"%PDF", "notes.txt": b"mine"}


def test_a_failure_while_the_set_is_written_replaces_no_file_of_it(tmp_path, monkeypatch):
    out = tmp_path / "out"
    replace_files(out, {"a.svg": b"one", "b.json": b"{}"})
    before = _tree(out)
    real_open = Path.open

    def open_(self, mode="r", *args, **kwargs):
        if self.name == ".b.json.partial":
            raise OSError(28, "No space left on device")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_)
    with pytest.raises(OSError):
        replace_files(out, {"a.svg": b"two", "b.json": b"[]"})
    monkeypatch.undo()
    assert _tree(out) == before  # (the earlier set, and no temporary file)


def test_stage_writes_beside_the_name_and_returns_size_and_sha256(tmp_path):
    staged = StagedFiles(tmp_path)
    (tmp_path / "result.json").write_bytes(b"earlier")
    pieces = [b"{", b'"a": 1', b"}"]
    assert staged.stage("result.json", iter(pieces)) == (8, hashlib.sha256(b"".join(pieces)).hexdigest())
    assert staged.stage("sub/deep/scene.glb", b"glTF") == (4, hashlib.sha256(b"glTF").hexdigest())
    assert staged.names == ("result.json", "sub/deep/scene.glb")
    assert staged_path(tmp_path / "result.json") == tmp_path / ".result.json.partial"
    assert _tree(tmp_path) == {"result.json": b"earlier", ".result.json.partial": b'{"a": 1}', "sub/deep/.scene.glb.partial": b"glTF"}
    staged.discard()
    assert _tree(tmp_path) == {"result.json": b"earlier"} and not (tmp_path / "sub").exists()  # (the directories it made go too)
    staged.discard()  # (nothing left to remove)


def test_discard_keeps_a_directory_that_was_there_or_holds_another_file(tmp_path):
    (tmp_path / "kept").mkdir()
    staged = StagedFiles(tmp_path)
    staged.stage("kept/a.json", b"1")
    staged.stage("made/b.json", b"2")
    (tmp_path / "made" / "other.txt").write_bytes(b"mine")
    staged.discard()
    assert (tmp_path / "kept").is_dir() and _tree(tmp_path) == {"made/other.txt": b"mine"}


def test_commit_removes_first_moves_in_order_then_removes_and_moves_the_last(tmp_path, monkeypatch):
    for name in ("manifest.json", "stopped.json", "a.json", "old.json", "old/x.json", "old/keep.txt"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"earlier")
    steps: list[str] = []
    real_replace, real_unlink, real_rmdir = Path.replace, Path.unlink, Path.rmdir

    def replace(self, target):
        steps.append("move " + Path(target).relative_to(tmp_path).as_posix())
        return real_replace(self, target)

    def unlink(self, missing_ok=False):
        if not self.name.endswith(".partial"):
            steps.append("remove " + self.relative_to(tmp_path).as_posix())
        return real_unlink(self, missing_ok=missing_ok)

    def rmdir(self):
        steps.append("rmdir " + self.relative_to(tmp_path).as_posix())
        return real_rmdir(self)

    staged = StagedFiles(tmp_path)
    for name in ("a.json", "b.json", "project.json", "manifest.json"):
        staged.stage(name, name.encode())
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(Path, "rmdir", rmdir)
    staged.commit(remove_first=["stopped.json", "manifest.json", "absent.json"], first="project.json",
                  remove=["old.json", "old/x.json", "old", "absent.json"], last="manifest.json")
    monkeypatch.undo()
    assert steps == ["remove stopped.json", "remove manifest.json", "remove absent.json",
                     "move project.json", "move a.json", "move b.json",
                     "remove old.json", "remove old/x.json", "rmdir old", "remove absent.json",
                     "move manifest.json"]
    assert _tree(tmp_path) == {"a.json": b"a.json", "b.json": b"b.json", "project.json": b"project.json",
                               "manifest.json": b"manifest.json", "old/keep.txt": b"earlier"}


def test_a_name_that_cannot_be_removed_first_stops_the_commit_before_any_move(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_bytes(b"earlier")
    (tmp_path / "a.json").write_bytes(b"earlier")
    staged = StagedFiles(tmp_path)
    staged.stage("a.json", b"new")
    staged.stage("manifest.json", b"new")
    real_unlink = Path.unlink

    def unlink(self, missing_ok=False):
        if self.name == "manifest.json":
            raise PermissionError(13, "delete refused", str(self))
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(PermissionError):
        staged.commit(remove_first=["manifest.json"], last="manifest.json")
    monkeypatch.undo()
    assert _tree(tmp_path) == {"manifest.json": b"earlier", "a.json": b"earlier"}  # (nothing moved, no temporary file)


def test_the_check_refuses_a_path_before_it_is_touched(tmp_path):
    class Refused(OSError):
        pass

    seen: list[str] = []

    def check(path: Path) -> None:
        seen.append(path.name)
        if path.name in ("link.json", ".tmp-link.json.partial", "gone.json"):
            raise Refused(str(path))

    staged = StagedFiles(tmp_path, check=check)
    with pytest.raises(Refused):
        staged.stage("link.json", b"x")
    with pytest.raises(Refused):
        staged.stage("tmp-link.json", b"x")
    assert _tree(tmp_path) == {} and seen == ["link.json", "tmp-link.json", ".tmp-link.json.partial"]
    (tmp_path / "gone.json").write_bytes(b"elsewhere")
    staged.stage("a.json", b"new")
    staged.commit(remove=["gone.json"])  # (a name the check refuses is left alone when it is only to be removed)
    assert _tree(tmp_path) == {"a.json": b"new", "gone.json": b"elsewhere"}
    staged.stage("a.json", b"newer")
    with pytest.raises(Refused):
        staged.commit(remove_first=["gone.json"])  # (and stops the commit when it had to go first)
    assert _tree(tmp_path) == {"a.json": b"new", "gone.json": b"elsewhere"}


def test_a_temporary_file_left_by_an_earlier_attempt_is_replaced_not_written_into(tmp_path):
    import os

    outside = tmp_path / "outside.json"
    outside.write_bytes(b"elsewhere")
    out = tmp_path / "out"
    out.mkdir()
    os.link(outside, out / ".a.json.partial")  # (a hard link: writing into it would change the file elsewhere)
    staged = StagedFiles(out)
    staged.stage("a.json", b"new")
    assert outside.read_bytes() == b"elsewhere" and (out / ".a.json.partial").read_bytes() == b"new"
    staged.commit()
    assert _tree(out) == {"a.json": b"new"} and outside.read_bytes() == b"elsewhere"
