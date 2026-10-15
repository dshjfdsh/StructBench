"""Generate a string that satisfies a regular expression.

Both JSON Schema ``pattern`` and StructBench's ``must_match`` rule are regular
expressions, and the dataset self-check has to *prove* a case is satisfiable by
constructing a value that matches. A fixed candidate list is not enough: a
perfectly reasonable pattern such as ``^CUST-[0-9]{4}$`` appears in no list, and a
case using it would be wrongly rejected as unsatisfiable.

So we generate one, with a deliberately small recursive-descent sampler.
Supported constructs: literals, ``.``, the escapes ``\\d \\D \\w \\W \\s \\S \\n
\\t \\\\ \\uXXXX`` and escaped punctuation, character classes (with ranges and
negation), groups (``(...)``, ``(?:...)``, ``(?P<name>...)``, ``(?i:...)``),
alternation ``|``, the quantifiers ``* + ? {n} {n,} {n,m}``, and the anchors
``^ $ \\b`` (ignored). Anything else raises :class:`RegexGenError`, which the
caller turns into an honest "cannot synthesise" diagnostic rather than a guess.

Note the two matching conventions, which callers must choose between: JSON Schema
(and Pydantic) ``pattern`` is a *partial* match, while StructBench ``must_match``
is a *full* match.
"""

from __future__ import annotations

import re

__all__ = ["RegexGenError", "sample_for_pattern"]

#: Characters tried when a negated class (``[^abc]``) has to be satisfied.
_NEGATED_POOL = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"

#: How far quantifier expansion is pushed while searching for a length-valid sample.
_PAD_STEPS = 8


class RegexGenError(ValueError):
    """The pattern uses a construct the sampler cannot handle."""


# --------------------------------------------------------------------------- #
# Tokenising helpers
# --------------------------------------------------------------------------- #


def _split_alternation(pattern: str) -> list[str]:
    """Split ``a|b|c`` at the top level, ignoring groups and character classes."""
    branches: list[str] = []
    depth = 0
    in_class = False
    start = 0
    pos = 0
    while pos < len(pattern):
        char = pattern[pos]
        if in_class:
            if char == "\\":
                pos += 2
                continue
            if char == "]":
                in_class = False
            pos += 1
            continue
        if char == "\\":
            pos += 2
            continue
        if char == "[":
            in_class = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise RegexGenError(f"unbalanced ')' in {pattern!r}")
        elif char == "|" and depth == 0:
            branches.append(pattern[start:pos])
            start = pos + 1
        pos += 1
    branches.append(pattern[start:])
    return branches


def _group_end(pattern: str, start: int) -> int:
    """Index of the ``)`` matching the ``(`` at ``start``."""
    depth = 0
    in_class = False
    pos = start
    while pos < len(pattern):
        char = pattern[pos]
        if in_class:
            if char == "\\":
                pos += 2
                continue
            if char == "]":
                in_class = False
            pos += 1
            continue
        if char == "\\":
            pos += 2
            continue
        if char == "[":
            in_class = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return pos
        pos += 1
    raise RegexGenError(f"unbalanced '(' in {pattern!r}")


def _class_end(pattern: str, start: int) -> int:
    """Index of the ``]`` closing the ``[`` at ``start``."""
    pos = start + 1
    if pos < len(pattern) and pattern[pos] == "^":
        pos += 1
    if pos < len(pattern) and pattern[pos] == "]":
        pos += 1  # a literal ']' first in the class
    while pos < len(pattern):
        char = pattern[pos]
        if char == "\\":
            pos += 2
            continue
        if char == "]":
            return pos
        pos += 1
    raise RegexGenError(f"unbalanced '[' in {pattern!r}")


def _strip_group_prefix(inner: str) -> str:
    """Remove ``?:``, ``?P<name>`` or ``?flags:`` from a group body."""
    if not inner.startswith("?"):
        return inner
    if inner.startswith("?:"):
        return inner[2:]
    named = re.match(r"\?P<[^>]*>", inner)
    if named:
        return inner[named.end() :]
    flagged = re.match(r"\?[aiLmsux]+:", inner)
    if flagged:
        return inner[flagged.end() :]
    raise RegexGenError(f"unsupported group construct in {inner!r}")


# --------------------------------------------------------------------------- #
# The sampler
# --------------------------------------------------------------------------- #


def _sample_escape(char: str) -> str:
    table = {
        "d": "7",
        "D": "x",
        "w": "a",
        "W": "-",
        "s": " ",
        "S": "x",
        "n": "\n",
        "t": "\t",
        "r": "\r",
        "f": "\f",
        "v": "\v",
        "0": "\0",
    }
    if char in table:
        return table[char]
    if char in ("b", "B", "A", "Z", "z"):
        raise RegexGenError(f"boundary escape \\{char} cannot be sampled")
    return char


def _expand_class(body: str) -> tuple[str, bool]:
    """Return ``(chars, negated)`` for a character-class body."""
    negated = body.startswith("^")
    if negated:
        body = body[1:]

    chars: list[str] = []
    pos = 0
    while pos < len(body):
        char = body[pos]
        if char == "\\" and pos + 1 < len(body):
            nxt = body[pos + 1]
            if nxt in "dDwWsS":
                chars.extend(_sample_escape(nxt) * (10 if nxt in "dDwW" else 1))
                pos += 2
                continue
            if nxt == "u" and pos + 5 < len(body):
                chars.append(chr(int(body[pos + 2 : pos + 6], 16)))
                pos += 6
                continue
            chars.append(_sample_escape(nxt))
            pos += 2
            continue
        if pos + 2 < len(body) and body[pos + 1] == "-" and body[pos + 2] != "]":
            low, high = ord(char), ord(body[pos + 2])
            if low > high:
                raise RegexGenError(f"reversed range {char}-{body[pos + 2]!r} in class")
            # Clamp to printable ASCII: sampling control characters helps nobody.
            for code in range(max(low, 0x20), min(high, 0x7E) + 1):
                chars.append(chr(code))
            pos += 3
            continue
        chars.append(char)
        pos += 1

    if negated:
        return "".join(chars), True
    if not chars:
        raise RegexGenError("empty character class")
    return "".join(chars), False


def _sample_class(body: str, pattern: str) -> str:
    chars, negated = _expand_class(body)
    if negated:
        for candidate in _NEGATED_POOL:
            if candidate not in chars:
                return candidate
        raise RegexGenError(f"negated class [{body}] excludes every candidate character")
    # Prefer a letter when the class offers one: the generated strings end up more
    # readable in self-check diagnostics ("AB12" beats "0012").
    for candidate in chars:
        if candidate.isalpha():
            return candidate
    return chars[0]


def _parse_quantifier(pattern: str, pos: int, pad: int) -> tuple[int, int]:
    char = pattern[pos]
    if char == "*":
        return pad, pos + 1
    if char == "+":
        return 1 + pad, pos + 1
    if char == "?":
        return 1, pos + 1
    if char == "{":
        end = pattern.find("}", pos)
        if end < 0:
            raise RegexGenError(f"unterminated quantifier in {pattern!r}")
        body = pattern[pos + 1 : end]
        parsed = re.fullmatch(r"(\d+)(?:,(\d*))?", body)
        if not parsed:
            raise RegexGenError(f"unsupported quantifier {{{body}}}")
        low = int(parsed.group(1))
        if parsed.group(2) is None:
            return low, end + 1
        if parsed.group(2) == "":
            return low + pad, end + 1
        return min(low + pad, int(parsed.group(2))), end + 1
    raise RegexGenError(f"unexpected {char!r} after an atom")


def _sample_sequence_strict(pattern: str, pad: int) -> str:
    """Sample one branch of a pattern (no top-level alternation expected)."""
    parts: list[str] = []
    pos = 0
    while pos < len(pattern):
        char = pattern[pos]

        if char in "^$":
            pos += 1
            continue

        if char == "\\":
            if pos + 1 >= len(pattern):
                raise RegexGenError("trailing backslash")
            nxt = pattern[pos + 1]
            if nxt in ("b", "B", "A", "Z", "z"):
                pos += 2
                continue
            if nxt == "u" and pos + 5 < len(pattern):
                atom = chr(int(pattern[pos + 2 : pos + 6], 16))
                pos += 6
            else:
                atom = _sample_escape(nxt)
                pos += 2
        elif char == "[":
            end = _class_end(pattern, pos)
            atom = _sample_class(pattern[pos + 1 : end], pattern)
            pos = end + 1
        elif char == "(":
            end = _group_end(pattern, pos)
            atom = _sample_sequence(_strip_group_prefix(pattern[pos + 1 : end]), pad)
            pos = end + 1
        elif char == ".":
            atom = "a"
            pos += 1
        elif char in "*+?{":
            raise RegexGenError(f"quantifier {char!r} with nothing to repeat in {pattern!r}")
        elif char == ")":
            raise RegexGenError(f"unbalanced ')' in {pattern!r}")
        else:
            atom = char
            pos += 1

        count = 1
        if pos < len(pattern) and pattern[pos] in "*+?{":
            count, pos = _parse_quantifier(pattern, pos, pad)
            # Skip laziness/possessive markers: they do not change a sample.
            if pos < len(pattern) and pattern[pos] in "?+":
                pos += 1
        parts.append(atom * max(count, 0))

    return "".join(parts)


def _sample_sequence(pattern: str, pad: int) -> str:
    """Sample ``pattern``, taking the first branch that can be generated.

    Alternation may appear at the top level *or* inside a group, so branch
    selection happens here rather than only at the entry point.
    """
    branches = _split_alternation(pattern)
    last_error: RegexGenError | None = None
    for branch in branches:
        try:
            return _sample_sequence_strict(branch, pad)
        except RegexGenError as exc:  # try the next alternative
            last_error = exc
    raise RegexGenError(
        f"no branch of {pattern!r} could be sampled" + (f" ({last_error})" if last_error else "")
    )


def sample_for_pattern(
    pattern: str,
    *,
    min_length: int = 0,
    max_length: int | None = None,
    full: bool = True,
) -> str | None:
    """Return a string matching ``pattern``, or ``None`` when impossible.

    ``full=True`` requires a full match (StructBench's ``must_match`` rule);
    ``full=False`` accepts a partial match, which is what JSON Schema's
    ``pattern`` means and what Pydantic enforces.
    """
    try:
        compiled = re.compile(pattern)
    except re.error:
        return None

    def fits(text: str) -> bool:
        if len(text) < min_length:
            return False
        return max_length is None or len(text) <= max_length

    def matches(text: str) -> bool:
        return compiled.fullmatch(text) is not None if full else compiled.search(text) is not None

    # Quantifier expansion is the only lever for reaching a `minLength`, so the
    # number of attempts scales with it.
    steps = max(_PAD_STEPS, min(min_length + 4, 64))
    for pad in range(steps):
        try:
            sample = _sample_sequence(pattern, pad)
        except RegexGenError:
            # A construct we cannot handle: give up rather than guess.
            return None
        except (ValueError, OverflowError):  # pragma: no cover - defensive
            continue
        if not full and len(sample) < min_length:
            # Partial-match semantics tolerate trailing filler, which is the
            # cheapest way to satisfy a minLength alongside a short pattern.
            # (It cannot rescue a `$`-anchored pattern, and `matches` will say so.)
            sample = sample + "x" * (min_length - len(sample))
        if fits(sample) and matches(sample):
            return sample
    return None
