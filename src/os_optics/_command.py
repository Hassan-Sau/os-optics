"""Ctrl-C while the ``os-optics`` command is still starting (standard library only, imported first by the package).

Starting the command imports os-optics and its numerical libraries, which takes about a second. A Ctrl-C in that time
interrupts the import, before any argument is read and before any run starts: when this process is the command
(``os-optics`` / ``opensci-engine``, or ``python -m os_optics``), ``exit_interrupted_while_starting`` turns it into one line on
stderr saying that no run started and nothing was written, and exit code 130 (the code of an interrupted run), instead
of a Python traceback. A program that imports ``os_optics`` itself gets its ``KeyboardInterrupt`` unchanged. Once the
command runs, ``os_optics.cli`` handles Ctrl-C itself (``project run``: the stopped run's record).
"""

from __future__ import annotations

import sys

#: The names the command is installed under (console scripts: ``os-optics``, and the earlier name, which the alias
#: distribution installs) and the modules ``python -m`` starts it from. ``os_optics.cli`` uses the same names.
COMMAND_NAMES = ("os-optics", "opensci-engine")
COMMAND_MODULES = ("os_optics", "os_optics.cli", "opensci_engine", "opensci_engine.cli")
#: Exit code of a command interrupted by Ctrl-C (128 + SIGINT, as a shell reports it).
INTERRUPTED_EXIT = 130
#: What the command prints on stderr when Ctrl-C reaches it while it is still starting.
INTERRUPTED_WHILE_STARTING = ("os-optics: interrupted (Ctrl-C) while the command was starting: no run started and nothing "
                              "was written or removed")
#: Suffixes a console-script launcher adds to the command's name (Windows launchers, setuptools wrappers).
_LAUNCHER_SUFFIXES = (".exe", "-script.pyw", "-script.py")


def started_as_command(argv: list[str] | None = None, orig_argv: list[str] | None = None) -> bool:
    """Whether this process is the ``os-optics`` command: started by its console script (``argv[0]`` names it) or as
    ``python -m os_optics`` (``sys.orig_argv``; ``argv[0]`` is ``-m`` while the module is being imported)."""

    argv = sys.argv if argv is None else argv
    orig_argv = list(getattr(sys, "orig_argv", ())) if orig_argv is None else orig_argv
    name = (argv[0] if argv else "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in _LAUNCHER_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name in COMMAND_NAMES:
        return True
    args = orig_argv[1:]
    at = 0
    while at < len(args):
        arg = args[at]
        if arg == "-" or not arg.startswith("-"):  # (a script, or the standard input, runs: not a module)
            return False
        if arg.startswith("--"):
            at += 2 if arg == "--check-hash-based-pycs" else 1
            continue
        flags = arg[1:]
        for i, flag in enumerate(flags):
            if flag == "c":
                return False
            if flag == "m":
                module = flags[i + 1:] or (args[at + 1] if at + 1 < len(args) else "")
                return module in COMMAND_MODULES
            if flag in "XW":  # (an option that takes a value: the rest of this word, or the next word)
                if not flags[i + 1:]:
                    at += 1
                break
        at += 1
    return False


def exit_interrupted_while_starting() -> None:
    """Called while a ``KeyboardInterrupt`` interrupts the imports that start the command: when this process is the
    command, end it with ``INTERRUPTED_WHILE_STARTING`` on stderr and exit code 130 (``SystemExit``); otherwise return,
    and the caller raises the interrupt again."""

    if not started_as_command():
        return
    try:
        print(INTERRUPTED_WHILE_STARTING, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass
    raise SystemExit(INTERRUPTED_EXIT) from None


__all__ = ["COMMAND_MODULES", "COMMAND_NAMES", "INTERRUPTED_EXIT", "INTERRUPTED_WHILE_STARTING",
           "exit_interrupted_while_starting", "started_as_command"]
