"""HX014: a private API-error predicate (registered in ``py_rules.CHECKERS``).

The auxiliary-client, compressor and turn paths each grew their own "is this a <kind> error?"
helpers that read an exception's ``status_code`` or grep its text. Each copy drifts from the shared
classifier (``agent.auxiliary_error_policy`` / ``agent.error_classifier.classify_api_error``), so
the same provider error is retried on one path and fatal on another. This flags the shape: a
function or lambda that takes an exception and decides from its status code or its text.

One finding per offending expression, reported on that expression's line: the ratchet keys a hit
by (rule, enclosing function, code text of the line), never by line number, so a predicate that
moves within or across files keeps its debt and a second probe added beside it is new.

A parameter typed as the classifier's verdict (``ClassifiedError``) is not an exception, whatever
its name: reading ``verdict.status_code`` is deciding from the shared classifier.

Known limits: an exception rebound to another name (``e2 = exc``) is not followed, and a
precompiled pattern (``PAT.search(str(exc))``) is not recognised as a text probe.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator

from scripts.code_health.py_ast import _FUNCS, Ctx, _call_name
from scripts.code_health.py_rules_sleep import _own

_EXC_NAMES = frozenset({"exc", "err", "error", "e", "ex", "exception"})
_EXC_SUFFIXES = ("_err", "_error", "_exc")
_EXC_TYPE_SUFFIXES = ("Exception", "Error")
_VERDICT_TYPE_SUFFIXES = ("ClassifiedError",)  # the classifier's output, not an exception
_CASE_FOLDS = frozenset({"lower", "upper", "casefold", "strip", "lstrip", "rstrip"})
_AFFIXES = frozenset({"startswith", "endswith"})
_REGEX_CALLS = frozenset({"re.search", "re.match", "re.fullmatch", "re.findall", "re.finditer"})
_CONTAINS_HELPERS = frozenset({"_contains_any"})
_WRAPPERS = frozenset({"Optional", "Union"})  # also as ``typing.Optional``


def _annotation_type_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _verdict_annotation(node: ast.expr | None) -> bool:
    """``ClassifiedError``, ``Optional[ClassifiedError]``, ``ClassifiedError | None`` or the same
    as a string."""
    if node is None:
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _verdict_annotation(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _verdict_annotation(node.left) or _verdict_annotation(node.right)
    if isinstance(node, ast.Subscript) and _annotation_type_name(node.value) in _WRAPPERS:
        parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return any(_verdict_annotation(part) for part in parts)
    name = _annotation_type_name(node)
    return name is not None and name.endswith(_VERDICT_TYPE_SUFFIXES)


def _exception_annotation(node: ast.expr | None) -> bool:
    """``Exception``, ``httpx.HTTPError``, ``Optional[BaseException]``, ``Exception | None`` or
    the same as a string. A container of exceptions (``list[Exception]``) is not one."""
    if node is None:
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _exception_annotation(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return False
    name = _annotation_type_name(node)
    if name is not None:
        return name.endswith(_EXC_TYPE_SUFFIXES) and not name.endswith(_VERDICT_TYPE_SUFFIXES)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _exception_annotation(node.left) or _exception_annotation(node.right)
    if isinstance(node, ast.Subscript) and isinstance(node.value, (ast.Name, ast.Attribute)):
        wrapper = node.value.id if isinstance(node.value, ast.Name) else node.value.attr
        if wrapper not in _WRAPPERS:
            return False
        parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return any(_exception_annotation(part) for part in parts)
    return False


def _exception_params(func: ast.AST) -> frozenset[str]:
    args = func.args  # type: ignore[attr-defined]
    params = (*args.posonlyargs, *args.args, *args.kwonlyargs)
    return frozenset(
        p.arg for p in params
        if not _verdict_annotation(p.annotation)
        and (p.arg in _EXC_NAMES or p.arg.endswith(_EXC_SUFFIXES) or _exception_annotation(p.annotation)))


def _is_param(node: ast.AST, params: frozenset[str]) -> bool:
    return isinstance(node, ast.Name) and node.id in params


def _getattr_of(node: ast.AST, attr: str) -> ast.expr | None:
    """``X`` for ``getattr(X, "<attr>", ...)``, else None."""
    if (isinstance(node, ast.Call) and _call_name(node) == "getattr" and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant) and node.args[1].value == attr):
        return node.args[0]
    return None


def _attr_of(node: ast.AST, attr: str) -> ast.expr | None:
    """``X`` for ``X.<attr>`` or ``getattr(X, "<attr>", ...)``, else None."""
    if isinstance(node, ast.Attribute) and node.attr == attr:
        return node.value
    return _getattr_of(node, attr)


def _status_read(node: ast.AST, params: frozenset[str]) -> bool:
    """``exc.status_code``, ``exc.response.status_code`` or their ``getattr`` spellings."""
    owner = _attr_of(node, "status_code")
    if owner is None:
        return False
    if _is_param(owner, params):
        return True
    response = _attr_of(owner, "response")
    return response is not None and _is_param(response, params)


def _exc_text(node: ast.AST, params: frozenset[str], texts: frozenset[str]) -> bool:
    """``str(exc)`` / ``repr(exc)``, case-folded (``.lower()``), or a local bound to one."""
    if isinstance(node, ast.Name):
        return node.id in texts
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute) and node.func.attr in _CASE_FOLDS:
        return _exc_text(node.func.value, params, texts)
    return (_call_name(node) in ("str", "repr") and len(node.args) == 1
            and _is_param(node.args[0], params))


def _text_names(nodes: list[ast.AST], params: frozenset[str]) -> frozenset[str]:
    """Locals bound (anywhere in the scope) to the exception's text: ``msg = str(exc).lower()``,
    including chains (``low = msg.lower()``)."""
    bindings: list[tuple[str, ast.AST]] = []
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            bindings.append((node.targets[0].id, node.value))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and isinstance(node.target, ast.Name) \
                and node.value is not None:
            bindings.append((node.target.id, node.value))
    texts: frozenset[str] = frozenset()
    while True:
        grown = texts | {name for name, value in bindings if _exc_text(value, params, texts)}
        if grown == texts:
            return texts
        texts = grown


def _text_probe(node: ast.AST, params: frozenset[str], texts: frozenset[str]) -> bool:
    """``"x" in <text>``, ``<text>.startswith(...)``, ``re.search(..., <text>)`` or
    ``_contains_any(<text>, ...)``."""
    if isinstance(node, ast.Compare):
        return any(isinstance(op, (ast.In, ast.NotIn)) and _exc_text(right, params, texts)
                   for op, right in zip(node.ops, node.comparators))
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute) and node.func.attr in _AFFIXES:
        return _exc_text(node.func.value, params, texts)
    name = _call_name(node)
    if name in _REGEX_CALLS or name.rsplit(".", 1)[-1] in _CONTAINS_HELPERS:
        return any(_exc_text(arg, params, texts) for arg in node.args)
    return False


def _functions(tree: ast.Module) -> Iterator[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, (*_FUNCS, ast.Lambda)):
            yield node


def private_error_predicate(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    """Each status-code read or text probe of an exception parameter in the body of the def or
    lambda that takes it; a nested def is judged on its own parameters."""
    for func in _functions(tree):
        params = _exception_params(func)
        if not params:
            continue
        nodes = list(_own(func))
        texts = _text_names(nodes, params)
        for node in nodes:
            if isinstance(node, ast.expr) and (_status_read(node, params) or _text_probe(node, params, texts)):
                yield node.lineno
