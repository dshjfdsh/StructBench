"""A tiny, dependency-free JSONPath subset used by the validation rules.

Only the forms StructBench datasets actually need are supported::

    $                        # the root
    $.user                   # object member
    $.user.profile.tags      # nested members
    $.items[0].id            # array index
    $['weird key'].value     # bracket-quoted member
    $["weird key"].value

``[*]`` wildcards, filters, slices and recursive descent are intentionally not
implemented -- a benchmark fixture should never need them, and keeping the
grammar small keeps the validator predictable.
"""

from __future__ import annotations

import re
from typing import Any

MISSING = object()
"""Sentinel returned by :func:`get` when a path does not resolve.

Distinct from ``None`` because a JSON field may legitimately hold ``null``, and
"absent" and "present but null" are very different verdicts for this benchmark.
"""

_TOKEN_RE = re.compile(
    r"""
      \[\s*(\d+)\s*\]                      # [0]
    | \[\s*'((?:[^'\\]|\\.)*)'\s*\]        # ['key']
    | \[\s*"((?:[^"\\]|\\.)*)"\s*\]        # ["key"]
    | \.([A-Za-z_][A-Za-z0-9_\-]*)         # .key
    | ^([A-Za-z_][A-Za-z0-9_\-]*)          # leading key, no dot
    """,
    re.VERBOSE,
)

_UNESCAPE_RE = re.compile(r"\\(.)")


class PathError(ValueError):
    """Raised when a rule path is not valid JSONPath-lite."""


def _unescape(text: str) -> str:
    return _UNESCAPE_RE.sub(r"\1", text)


def parse_path(path: str) -> list[str | int]:
    """Split a JSONPath-lite expression into tokens."""
    if not isinstance(path, str):  # pragma: no cover - defensive
        raise PathError(f"path must be a string, got {type(path).__name__}")

    text = path.strip()
    if text.startswith("$"):
        text = text[1:]
    elif text[:1] not in ("", ".", "["):
        # Bare ``a.b`` is accepted as a convenience.
        pass

    tokens: list[str | int] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if not match:
            raise PathError(
                f"invalid JSONPath {path!r}: unexpected {text[pos : pos + 12]!r} at offset {pos}"
            )
        index, single, double, dotted, bare = match.groups()
        if index is not None:
            tokens.append(int(index))
        elif single is not None:
            tokens.append(_unescape(single))
        elif double is not None:
            tokens.append(_unescape(double))
        elif dotted is not None:
            tokens.append(dotted)
        elif bare is not None:
            tokens.append(bare)
        pos = match.end()
    return tokens


def format_path(tokens: list[str | int]) -> str:
    """Render tokens back into canonical JSONPath-lite form."""
    out = "$"
    for token in tokens:
        if isinstance(token, int):
            out += f"[{token}]"
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*", token):
            out += f".{token}"
        else:
            out += "['" + token.replace("\\", "\\\\").replace("'", "\\'") + "']"
    return out


def get(value: Any, path: str | list[str | int], default: Any = MISSING) -> Any:
    """Resolve ``path`` against ``value``, returning ``default`` if absent."""
    tokens = parse_path(path) if isinstance(path, str) else list(path)
    current = value
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(current, list):
                return default
            if token >= len(current) or token < -len(current):
                return default
            current = current[token]
        else:
            if not isinstance(current, dict) or token not in current:
                return default
            current = current[token]
    return current


def exists(value: Any, path: str | list[str | int]) -> bool:
    """True when ``path`` resolves to a value, *including* an explicit ``null``."""
    return get(value, path) is not MISSING


def is_null(value: Any, path: str | list[str | int]) -> bool:
    """True when ``path`` exists and holds ``null``."""
    return get(value, path) is None


def iter_leaf_paths(
    value: Any, prefix: tuple[str | int, ...] = ()
) -> list[tuple[tuple[str | int, ...], Any]]:
    """Enumerate ``(path_tokens, leaf_value)`` for every non-container leaf.

    Empty objects/arrays count as leaves so that ``{"a": {}}`` still yields a
    path for ``a``.
    """
    leaves: list[tuple[tuple[str | int, ...], Any]] = []
    if isinstance(value, dict) and value:
        for key, child in value.items():
            leaves.extend(iter_leaf_paths(child, (*prefix, key)))
    elif isinstance(value, list) and value:
        for index, child in enumerate(value):
            leaves.extend(iter_leaf_paths(child, (*prefix, index)))
    else:
        leaves.append((prefix, value))
    return leaves


def set_(value: Any, path: str | list[str | int], new_value: Any) -> Any:
    """Set ``path`` inside ``value`` in place, creating missing containers.

    Returns the (possibly new) root, since setting a path on a non-container root
    replaces it.
    """
    tokens = parse_path(path) if isinstance(path, str) else list(path)
    if not tokens:
        return new_value

    if not isinstance(value, (dict, list)):
        value = [] if isinstance(tokens[0], int) else {}

    current = value
    for position, token in enumerate(tokens[:-1]):
        nxt = tokens[position + 1]
        if isinstance(token, int):
            if not isinstance(current, list):
                raise PathError(f"cannot index a non-list at {format_path(tokens[: position + 1])}")
            while len(current) <= token:
                current.append([] if isinstance(nxt, int) else {})
            if not isinstance(current[token], (dict, list)):
                current[token] = [] if isinstance(nxt, int) else {}
            current = current[token]
        else:
            if not isinstance(current, dict):
                raise PathError(
                    f"cannot set a key on a non-object at {format_path(tokens[: position + 1])}"
                )
            if not isinstance(current.get(token), (dict, list)):
                current[token] = [] if isinstance(nxt, int) else {}
            current = current[token]

    last = tokens[-1]
    if isinstance(last, int):
        if not isinstance(current, list):
            raise PathError(f"cannot index a non-list at {format_path(tokens)}")
        while len(current) <= last:
            current.append(None)
        current[last] = new_value
    else:
        if not isinstance(current, dict):
            raise PathError(f"cannot set a key on a non-object at {format_path(tokens)}")
        current[last] = new_value
    return value


def delete(value: Any, path: str | list[str | int]) -> bool:
    """Remove ``path`` if present; returns True when something was removed."""
    tokens = parse_path(path) if isinstance(path, str) else list(path)
    if not tokens:
        return False
    parent = get(value, tokens[:-1])
    last = tokens[-1]
    if isinstance(last, int) and isinstance(parent, list):
        if 0 <= last < len(parent):
            del parent[last]
            return True
    elif isinstance(parent, dict) and last in parent:
        del parent[last]
        return True
    return False


__all__ = [
    "MISSING",
    "PathError",
    "delete",
    "exists",
    "format_path",
    "get",
    "is_null",
    "iter_leaf_paths",
    "parse_path",
    "set_",
]
