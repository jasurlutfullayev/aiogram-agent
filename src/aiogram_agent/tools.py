"""Turn plain Python functions into LLM tools.

@tool
async def find_product(query: str, max_price: int | None = None) -> str:
    '''Search the product catalog.

    Args:
        query: What the customer is looking for.
        max_price: Upper price limit in UZS.
    '''
"""

from __future__ import annotations

import inspect
import json
import re
import types
import typing
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union, get_args, get_origin

from .context import AgentContext

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _json_type(annotation: Any) -> dict[str, Any]:
    """Map a Python type hint to a JSON-schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = list(args)
        schema: dict[str, Any] = {"enum": values}
        if all(isinstance(v, str) for v in values):
            schema["type"] = "string"
        elif all(isinstance(v, int) for v in values):
            schema["type"] = "integer"
        return schema

    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return {"anyOf": [_json_type(a) for a in non_none]}

    if origin in (list, tuple, set, frozenset) or annotation in (list, tuple, set):
        item = _json_type(args[0]) if args else {}
        return {"type": "array", "items": item}

    if origin is dict or annotation is dict:
        return {"type": "object"}

    return {}


def _is_context(annotation: Any) -> bool:
    """True for ``AgentContext`` and ``AgentContext | None`` (Python 3.10 adds
    an implicit Optional when the default is None)."""
    if annotation is AgentContext:
        return True
    if get_origin(annotation) in (Union, types.UnionType):
        return any(a is AgentContext for a in get_args(annotation))
    return False


_ARG_LINE = re.compile(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description})."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    lines = doc.splitlines()
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if in_args:
            if stripped.endswith(":") and not line.startswith((" ", "\t")):
                in_args = False  # next section, e.g. "Returns:"
                continue
            match = _ARG_LINE.match(line)
            if match and line.startswith((" ", "\t")) and len(line) - len(line.lstrip()) <= 8:
                current = match.group(1)
                params[current] = match.group(2).strip()
            elif current and stripped:
                params[current] += " " + stripped
            continue
        if stripped and not stripped.endswith(":"):
            summary_lines.append(stripped)
        elif summary_lines and not stripped:
            summary_lines.append("")
    summary = " ".join(p for p in "\n".join(summary_lines).split("\n\n")[0].split()).strip()
    return summary, params


@dataclass
class Tool:
    """A callable the model may invoke."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    takes_context: str | None = None
    #: If True, the agent stops after this tool and does not ask the model again.
    final: bool = False
    _sig: inspect.Signature = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sig = inspect.signature(self.func)

    async def __call__(self, arguments: dict[str, Any], ctx: AgentContext) -> str:
        kwargs = {k: v for k, v in arguments.items() if k in self._sig.parameters}
        if self.takes_context:
            kwargs[self.takes_context] = ctx
        result = self.func(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        if result is None:
            return "ok"
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    final: bool = False,
) -> Any:
    """Decorator that turns a function into a :class:`Tool`.

    Parameters typed as :class:`AgentContext` are injected automatically and
    are hidden from the model.
    """

    def wrap(fn: Callable[..., Any | Awaitable[Any]]) -> Tool:
        summary, param_docs = _parse_docstring(fn.__doc__)
        try:
            hints = typing.get_type_hints(fn)
        except Exception:  # pragma: no cover - unresolved forward refs
            hints = {}
        sig = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []
        ctx_param: str | None = None
        for pname, param in sig.parameters.items():
            annotation = hints.get(pname, param.annotation)
            if _is_context(annotation):
                ctx_param = pname
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            schema = _json_type(annotation)
            if pname in param_docs:
                schema["description"] = param_docs[pname]
            if param.default is not inspect.Parameter.empty and param.default is not None:
                schema["default"] = param.default
            properties[pname] = schema
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        return Tool(
            name=name or fn.__name__,
            description=description or summary or fn.__name__,
            parameters=parameters,
            func=fn,
            takes_context=ctx_param,
            final=final,
        )

    if func is not None:
        return wrap(func)
    return wrap
