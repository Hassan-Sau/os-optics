"""Writing the files of a deliverable that a command regenerates from the same inputs.

One rule for every output a command writes: a deliverable the same command regenerates from the same inputs (a
drawing, a layout, a mounting sheet, a downloaded artifact file, a run directory) replaces its own files; a credential
is never replaced without ``--overwrite``. Only the deliverable's own file names are replaced: other files in the
directory are left alone. Each file of the set is first written to a temporary name in the target directory, and the set
is moved into place (``Path.replace``) only after every file has been written, so a failure while writing leaves the
earlier set as it was.

``replace_files(directory, files)`` does this for a set held in memory. ``StagedFiles`` is the same rule in two steps,
for a set whose files are produced one at a time or whose last file states the others (a manifest with their sizes and
hashes): ``stage`` writes a file to its temporary name and returns its size and SHA-256, ``commit`` moves the staged
files into place in the order they were staged, ``discard`` removes them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

#: Bytes gathered before one write to a staged file.
WRITE_CHUNK_BYTES = 1 << 20


def staged_path(final: Path) -> Path:
    """The temporary name a file is written to before it is moved to ``final``: ``.<name>.partial`` beside it."""

    return final.with_name(f".{final.name}.partial")


class StagedFiles:
    """The files of one output set in ``directory``: written to temporary names (``stage``), then moved into place
    together (``commit``) or removed (``discard``). A name may hold ``/`` for a file in a directory below (made when the
    file is staged). ``check``: called with every path the set writes or replaces before it is touched (the final name
    and the temporary one); it raises to refuse the path -- a caller that never writes through a link passes that
    refusal."""

    def __init__(self, directory: str | Path, *, check: Callable[[Path], object] | None = None) -> None:
        self.directory = Path(directory)
        self._check = check
        self._staged: list[tuple[str, Path, Path]] = []
        self._made: list[Path] = []

    @property
    def names(self) -> tuple[str, ...]:
        """The names staged so far, in order."""

        return tuple(name for name, _, _ in self._staged)

    def stage(self, name: str, data: bytes | Iterable[bytes]) -> tuple[int, str]:
        """Write ``data`` (bytes, or pieces of bytes) to the temporary name of ``name`` and return the file's size and
        SHA-256. The file under ``name`` is not touched. A file not written to its end is removed (``OSError`` and
        whatever the pieces raise pass through)."""

        final = self.directory / name
        temporary = staged_path(final)
        for path in (final, temporary):
            if self._check is not None:
                self._check(path)
        missing = [parent for parent in (final.parent, *final.parent.parents) if not parent.exists()]
        final.parent.mkdir(parents=True, exist_ok=True)
        self._made += reversed(missing)
        try:
            temporary.unlink()  # (a temporary file an earlier attempt left, which may be another file's hard link)
        except OSError:
            pass  # (missing; one that cannot be deleted refuses the exclusive open below)
        self._staged.append((name, temporary, final))
        digest, size, pending, held = hashlib.sha256(), 0, [], 0
        with temporary.open("xb") as handle:
            for piece in (data,) if isinstance(data, (bytes, bytearray, memoryview)) else data:
                digest.update(piece)
                size += len(piece)
                pending.append(piece)
                held += len(piece)
                if held >= WRITE_CHUNK_BYTES:
                    handle.write(b"".join(pending))
                    pending, held = [], 0
            handle.write(b"".join(pending))
        return size, digest.hexdigest()

    def commit(self, *, remove_first: Iterable[str] = (), first: str | None = None, remove: Iterable[str] = (),
               last: str | None = None, keep: Iterable[str] = (), before_last: Callable[[], None] | None = None,
               after_last: Callable[[], None] | None = None) -> None:
        """Move every staged file to its name, in staging order. ``remove_first``: names deleted, in the order given,
        before any file is moved (what would otherwise describe the earlier set while the files change: a manifest); one
        that cannot be deleted raises ``OSError`` and nothing is moved. ``first``: the staged name moved before the
        others. ``remove``: names of the earlier set the new one does not write, deleted after the moves (a missing one
        is fine, a directory is removed when empty, a link is left alone). ``last``: the staged name moved after those
        removals (the file that states the set); ``before_last``: called after those removals, before ``last`` is moved (a
        record of what they could not remove, written while no statement of the new set is in place yet); ``after_last``:
        called the moment ``last`` is in place (the set is complete: what follows is clean-up). A failure of a
        move (a file held open elsewhere) can leave the names
        before it replaced; the temporary files are removed either way -- except, once ``remove_first`` is deleted, those of
        the names in ``keep`` still staged: the statements of the set the names moved so far belong to (its manifests),
        which a later attempt reads in place of the ones deleted, as it reads what a process killed there leaves."""

        removed = False
        try:
            for name in remove_first:
                path = self.directory / name
                if self._check is not None:
                    self._check(path)
                path.unlink(missing_ok=True)
            removed = True
            ordered = sorted(self._staged, key=lambda staged: (staged[0] == last, staged[0] != first))
            for name, temporary, final in ordered:
                if name != last:
                    temporary.replace(final)
            for name in remove:
                path = self.directory / name
                try:
                    if self._check is not None:
                        self._check(path)
                    if path.is_dir():
                        path.rmdir()
                    else:
                        path.unlink(missing_ok=True)
                except OSError:
                    pass  # (a link, a file that cannot be deleted, a directory that holds other files: left where it is)
            if before_last is not None:
                before_last()
            for name, temporary, final in ordered:
                if name == last:
                    temporary.replace(final)
                    if after_last is not None:
                        after_last()
        except BaseException:
            if removed:
                kept = set(keep)
                self._staged = [staged for staged in self._staged if staged[0] not in kept]  # (their temporary files stay)
            raise
        finally:
            self.discard()

    def discard(self) -> None:
        """Remove the temporary files (after a commit: the ones that were not moved) and the directories made for them
        that hold nothing. The files under the set's names are not touched."""

        for _, temporary, _ in self._staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        for made in reversed(self._made):
            try:
                made.rmdir()
            except OSError:
                pass  # (it holds a file of the set, or another file)
        self._staged, self._made = [], []


def replace_files(directory: str | Path, files: Mapping[str, bytes]) -> Path:
    """Write ``files`` (``{file name: bytes}``) into ``directory`` (created if needed), replacing those names only, and
    return the directory. Raises ``OSError`` when a file cannot be written; a failure while the set is being written
    replaces no file of it (a failure of the final move itself, e.g. a file held open elsewhere, can leave the names
    before it replaced)."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    staged = StagedFiles(target)
    try:
        for name, data in files.items():
            staged.stage(name, data)
    except BaseException:
        staged.discard()
        raise
    staged.commit()
    return target
