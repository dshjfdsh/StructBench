"""Rasterise a StructBench SVG with no external tooling.

Why this exists: an SVG that is malformed, or whose geometry is wrong, fails
silently on GitHub -- you get a blank rectangle and no error. So the test suite
needs to *look* at the output, and a headless browser is not available in every
environment (nor in a locked-down CI sandbox).

This module therefore implements a deliberately tiny SVG subset renderer, covering
exactly the elements :mod:`structbench.reporter.svg` emits: ``rect``, ``line``,
``polygon``, ``circle`` and ``text``. It is not a general-purpose renderer and does
not try to be; it is a *verification* tool whose job is to prove that the committed
charts contain visible ink in the right places.

Output is a PNG written with :mod:`zlib` and :mod:`structwriter`-free code, so the
only dependency is the standard library.
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Canvas", "RenderError", "count_ink", "rasterise", "render_svg_to_png"]

_SVG_NS = "{http://www.w3.org/2000/svg}"


class RenderError(RuntimeError):
    """The SVG could not be rasterised."""


def _tag(element: ET.Element) -> str:
    return element.tag.replace(_SVG_NS, "")


def _floats(text: str | None, expected: int) -> list[float]:
    if not text:
        return []
    values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]
    return values if len(values) >= expected else []


def _hex_to_rgb(value: str) -> tuple[int, int, int] | None:
    text = value.strip()
    if text.startswith("#") and len(text) == 7:
        return (int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16))
    named = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "none": None,
    }
    return named.get(text.lower())


@dataclass
class Canvas:
    """A simple RGB pixel buffer with the handful of primitives we need."""

    width: int
    height: int
    background: tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        self.pixels = bytearray(bytes(self.background) * (self.width * self.height))

    def _set(self, x: int, y: int, colour: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset : offset + 3] = bytes(colour)

    def rect(self, x: float, y: float, w: float, h: float, colour: tuple[int, int, int]) -> None:
        x0, y0 = max(0, round(x)), max(0, round(y))
        x1, y1 = min(self.width, round(x + w)), min(self.height, round(y + h))
        for py in range(y0, y1):
            for px in range(x0, x1):
                self._set(px, py, colour)

    def line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        colour: tuple[int, int, int],
        width: float = 1.0,
    ) -> None:
        """Bresenham-ish line, thickened by stamping a small square."""
        steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0))))
        radius = max(0, int(width / 2))
        for step in range(steps + 1):
            t = step / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    self._set(round(x) + dx, round(y) + dy, colour)

    def polygon(
        self,
        points: list[tuple[float, float]],
        stroke: tuple[int, int, int] | None,
        fill: tuple[int, int, int] | None,
    ) -> None:
        if len(points) < 2:
            return
        if fill is not None and len(points) >= 3:
            self._fill_polygon(points, fill)
        if stroke is not None:
            for index in range(len(points)):
                x0, y0 = points[index]
                x1, y1 = points[(index + 1) % len(points)]
                self.line(x0, y0, x1, y1, stroke, 2.0)

    def _fill_polygon(
        self, points: list[tuple[float, float]], colour: tuple[int, int, int]
    ) -> None:
        """Scanline fill -- enough for convex radar polygons and bars."""
        ys = [p[1] for p in points]
        top, bottom = int(min(ys)), int(max(ys))
        for py in range(max(0, top), min(self.height, bottom + 1)):
            crossings: list[float] = []
            for index in range(len(points)):
                x0, y0 = points[index]
                x1, y1 = points[(index + 1) % len(points)]
                if (y0 <= py < y1) or (y1 <= py < y0):
                    t = (py - y0) / (y1 - y0)
                    crossings.append(x0 + (x1 - x0) * t)
            crossings.sort()
            for index in range(0, len(crossings) - 1, 2):
                for px in range(int(crossings[index]), int(crossings[index + 1]) + 1):
                    self._set(px, py, colour)

    def circle(
        self, cx: float, cy: float, r: float, colour: tuple[int, int, int], filled: bool = True
    ) -> None:
        radius = round(r)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    self._set(int(cx) + dx, int(cy) + dy, colour)

    def text_mark(self, x: float, y: float, size: float, colour: tuple[int, int, int]) -> None:
        """Approximate a text run as a small block.

        We cannot shape real glyphs without a font library, and we do not need to:
        the point is to prove that labels occupy space and are positioned inside
        the canvas, not to read them.
        """
        self.rect(x, y - size * 0.7, size * 0.55, size * 0.75, colour)


def rasterise(svg_text: str, *, scale: float = 1.0) -> Canvas:
    """Render the supported SVG subset onto a :class:`Canvas`."""
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise RenderError(f"malformed SVG: {exc}") from None

    width = float(root.get("width") or 0)
    height = float(root.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RenderError("SVG has no usable width/height")

    canvas = Canvas(int(width * scale), int(height * scale))

    def colour_of(element: ET.Element, attribute: str) -> tuple[int, int, int] | None:
        raw = element.get(attribute)
        return _hex_to_rgb(raw) if raw else None

    for element in root.iter():
        name = _tag(element)
        if name == "rect":
            values = _floats(element.get("x"), 1) + _floats(element.get("y"), 1)
            x = float(element.get("x") or 0)
            y = float(element.get("y") or 0)
            w = float(element.get("width") or 0)
            h = float(element.get("height") or 0)
            del values
            fill = colour_of(element, "fill")
            if fill is not None and w > 0 and h > 0:
                canvas.rect(x * scale, y * scale, w * scale, h * scale, fill)
        elif name == "line":
            stroke = colour_of(element, "stroke")
            if stroke is None:
                continue
            canvas.line(
                float(element.get("x1") or 0) * scale,
                float(element.get("y1") or 0) * scale,
                float(element.get("x2") or 0) * scale,
                float(element.get("y2") or 0) * scale,
                stroke,
                float(element.get("stroke-width") or 1.0) * scale,
            )
        elif name == "polygon":
            coords = _floats(element.get("points"), 6)
            if not coords:
                continue
            points = [
                (coords[i] * scale, coords[i + 1] * scale) for i in range(0, len(coords) - 1, 2)
            ]
            canvas.polygon(points, colour_of(element, "stroke"), colour_of(element, "fill"))
        elif name == "circle":
            fill = colour_of(element, "fill")
            stroke = colour_of(element, "stroke")
            cx = float(element.get("cx") or 0) * scale
            cy = float(element.get("cy") or 0) * scale
            r = float(element.get("r") or 0) * scale
            if fill is not None and fill != (255, 255, 255):
                canvas.circle(cx, cy, r, fill)
            if stroke is not None:
                canvas.circle(cx, cy, r, stroke, filled=False)
        elif name == "text":
            fill = colour_of(element, "fill")
            if fill is None:
                continue
            canvas.text_mark(
                float(element.get("x") or 0) * scale,
                float(element.get("y") or 0) * scale,
                float(element.get("font-size") or 12) * scale,
                fill,
            )

    return canvas


def count_ink(canvas: Canvas, background: tuple[int, int, int] = (255, 255, 255)) -> int:
    """Number of pixels differing from the background."""
    target = bytes(background)
    return sum(
        1
        for offset in range(0, len(canvas.pixels), 3)
        if canvas.pixels[offset : offset + 3] != target
    )


def _write_png(canvas: Canvas, path: Path) -> None:
    """Write an RGB PNG using only zlib + struct."""
    raw = bytearray()
    stride = canvas.width * 3
    for y in range(canvas.height):
        raw.append(0)  # filter type 0 (None)
        raw.extend(canvas.pixels[y * stride : (y + 1) * stride])

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", canvas.width, canvas.height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def render_svg_to_png(svg_text: str, path: str | Path, *, scale: float = 1.0) -> Path:
    """Rasterise ``svg_text`` and write a PNG. Returns the written path."""
    canvas = rasterise(svg_text, scale=scale)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_png(canvas, target)
    return target
