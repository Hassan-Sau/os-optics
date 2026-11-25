"""The method registry of the JSON API (``os_optics.local.json_api``): which functions a host that is not a Python program
may call, and how it passes their arguments.

Every entry point of the JSON API is registered here exactly once (``@API_METHODS.method(...)`` in ``json_api``; the
composition functions by ``API_METHODS.add``). A registered method records its name, its category, the parameters a JSON
host passes (from the function's signature: keyword-only parameters are for Python hosts and are not part of the JSON
surface), which parameter takes a ``{file name: bytes}`` map (``file_map``) or raw bytes (``data``), and whether the call
changes the runtime's own state (``runtime_state``: ``install_catalog`` activates a catalog release for every later call).

Everything else follows from the registry: ``capabilities()["json_api_methods"]`` and the dispatch of a host's call
(``call_method``: a name that is not registered is never called, whatever the module also defines).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

#: The categories of the JSON API's methods (``ApiMethod.category``).
CATEGORIES = ("runtime", "catalog", "project", "run", "results", "modelling", "analysis", "assembly", "commerce", "workspace")

#: The error code of a call to a name that is not a method of this runtime.
METHOD_NOT_IN_RUNTIME = "METHOD_NOT_IN_RUNTIME"
#: The error code of a call whose parameters do not fit the method (names, a missing required parameter, binary inputs).
INPUT_INVALID = "INPUT_INVALID"


@dataclass(frozen=True)
class ApiParameter:
    """One parameter a JSON host passes: its name, whether it is required and its Python annotation (text)."""

    name: str
    required: bool
    annotation: str
    default: Any = None


@dataclass(frozen=True)
class ApiMethod:
    name: str
    function: Callable[..., dict[str, Any]]
    category: str
    parameters: tuple[ApiParameter, ...]
    #: The parameter that takes a ``{file name: bytes}`` map (a package, a run's files, drawing outputs), if any.
    file_map: str | None = None
    #: The parameter that takes raw bytes (a catalog release, an Assembly Detail Package), if any.
    data: str | None = None
    #: The call changes the runtime's state for later calls (the active catalog release).
    runtime_state: bool = False

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters if p.required)

    def signature(self) -> str:
        """``name(required, optional=default)``: the call as a host writes it."""

        args = [p.name if p.required else f"{p.name}={p.default!r}" for p in self.parameters]
        return f"{self.name}({', '.join(args)})"

    def describe(self) -> dict[str, Any]:
        """The JSON description of ``capabilities()["json_api_methods"][name]``: ``category``, ``parameters``,
        ``required`` and, only when the method has them, ``file_map``, ``data`` and ``runtime_state: true``."""

        described: dict[str, Any] = {"category": self.category, "parameters": list(self.parameter_names), "required": list(self.required)}
        if self.file_map is not None:
            described["file_map"] = self.file_map
        if self.data is not None:
            described["data"] = self.data
        if self.runtime_state:
            described["runtime_state"] = True
        return described


def _parameters(function: Callable[..., Any]) -> tuple[ApiParameter, ...]:
    """The parameters a JSON host passes, from the function's code: its positional-or-keyword parameters (keyword-only
    ones, e.g. ``catalog=``, are for Python hosts and not part of the JSON surface)."""

    code = function.__code__  # type: ignore[attr-defined]
    if code.co_flags & (0x04 | 0x08) or code.co_posonlyargcount:  # *args / **kwargs / positional-only
        raise TypeError(f"{function.__name__}: a JSON API method takes named parameters only")
    names = code.co_varnames[:code.co_argcount]
    defaults = function.__defaults__ or ()  # type: ignore[attr-defined]
    first_default = len(names) - len(defaults)
    annotations = getattr(function, "__annotations__", {})
    parameters = []
    for index, name in enumerate(names):
        annotation = annotations.get(name, "Any")
        annotation = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
        required = index < first_default
        parameters.append(ApiParameter(name, required, annotation, None if required else defaults[index - first_default]))
    return tuple(parameters)


class ApiRegistry(Mapping[str, ApiMethod]):
    """Name -> ``ApiMethod``, in registration order."""

    def __init__(self) -> None:
        self._methods: dict[str, ApiMethod] = {}

    def add(self, function: Callable[..., dict[str, Any]], category: str, *, file_map: str | None = None, data: str | None = None,
            runtime_state: bool = False) -> ApiMethod:
        name = function.__name__
        if category not in CATEGORIES:
            raise ValueError(f"{name}: unknown category {category!r}; known: {CATEGORIES}")
        if name in self._methods:
            raise ValueError(f"{name} is registered twice")
        parameters = _parameters(function)
        names = {p.name for p in parameters}
        for role, parameter in (("file_map", file_map), ("data", data)):
            if parameter is not None and parameter not in names:
                raise ValueError(f"{name}: {role} parameter {parameter!r} is not a parameter of the method")
        method = ApiMethod(name, function, category, parameters, file_map=file_map, data=data, runtime_state=runtime_state)
        self._methods[name] = method
        return method

    def method(self, category: str, *, file_map: str | None = None, data: str | None = None,
               runtime_state: bool = False) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
        """Decorator: register the function as a method of ``category``; the function itself is returned unchanged."""

        def register(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
            self.add(function, category, file_map=file_map, data=data, runtime_state=runtime_state)
            return function

        return register

    def __getitem__(self, name: str) -> ApiMethod:
        return self._methods[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._methods)

    def __len__(self) -> int:
        return len(self._methods)

    def of_category(self, category: str) -> list[str]:
        return sorted(name for name, method in self._methods.items() if method.category == category)


def _refused(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "problems": [{"code": code, "message": message}]}}


def _binary(value: Any) -> Any:
    return value.tobytes() if isinstance(value, memoryview) else value


def split_binary(reply: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bytes] | None]:
    """A method's reply as its JSON part and its bytes (``None`` when it has none): ``files`` (a run directory, a drawing's
    outputs, ``submission.json``), ``result_file`` (the local assembly check's ``assembly_local_validation.json``) or the
    ``data`` of a ``run_results`` export (under its ``file_name``). The reply is changed in place."""

    files = reply.pop("files", None)
    result_file = reply.pop("result_file", None)
    if result_file is not None:
        from ..client_artifacts import VALIDATION_FILENAME  # lazy: os_optics.client_artifacts imports os_optics.local

        files = {**(files or {}), VALIDATION_FILENAME: bytes(result_file)}
    answer = reply.get("result")
    if reply.get("ok") is True and isinstance(answer, dict) and isinstance(answer.get("data"), (bytes, bytearray)):
        files = {**(files or {}), str(answer["file_name"]): bytes(answer.pop("data"))}
    return reply, (dict(files) if files is not None else None)


def call_registered(registry: ApiRegistry, name: Any, params: Any = None, *, file_map: Mapping[str, Any] | None = None,
                    data: Any = None) -> tuple[dict[str, Any], dict[str, bytes] | None]:
    """One call of a registered method by a host that is not a Python program: ``params`` (a JSON object: the method's
    parameters by name), ``file_map`` (``{file name: bytes}``, bound to the method's ``file_map`` parameter) and ``data``
    (bytes, bound to its ``data`` parameter). Returns the reply's JSON part and its bytes (``split_binary``).

    A name that is not registered gives ``METHOD_NOT_IN_RUNTIME``; parameters the method does not take, a missing required
    parameter, or binary inputs it has no parameter for give ``INPUT_INVALID``. Nothing is called then."""

    method = registry.get(name) if isinstance(name, str) else None
    if method is None:
        return _refused(METHOD_NOT_IN_RUNTIME, f"{name!r} is not a method of this runtime's JSON API"), None
    params = {} if params is None else params
    if not isinstance(params, dict):
        return _refused(INPUT_INVALID, f"{method.name}: parameters are a JSON object"), None
    arguments = dict(params)
    for role, parameter, value in (("file_map", method.file_map, file_map), ("data", method.data, data)):
        if value is None:
            continue
        if parameter is None:
            return _refused(INPUT_INVALID, f"{method.name} takes no {'files' if role == 'file_map' else 'bytes'}"), None
        if parameter in arguments:
            return _refused(INPUT_INVALID, f"{method.name}: {parameter} is given twice"), None
        arguments[parameter] = {str(k): _binary(v) for k, v in value.items()} if role == "file_map" else _binary(value)
    unknown = sorted(set(arguments) - set(method.parameter_names))
    if unknown:
        return _refused(INPUT_INVALID, f"{method.name} takes {list(method.parameter_names)}, not {unknown}"), None
    missing = [p for p in method.required if p not in arguments]
    if missing:
        return _refused(INPUT_INVALID, f"{method.name} needs {missing}"), None
    return split_binary(method.function(**arguments))
