"""Dependency-free SVG chart rendering.

The interactive HTML report cannot help you on GitHub: a README cannot embed a
`<script>`, and neither can a Release description. So the charts that people
actually see when they land on the repository have to be *static images*.

GitHub renders SVG inline from a committed file, which makes SVG the right
format: it stays crisp at any zoom, diffs as text, needs no binary tooling, and
costs nothing to regenerate in CI. Everything here is hand-written SVG -- no
matplotlib, no plotting library, no font embedding -- so the renderer stays inside
the project's zero-heavyweight-dependency rule and the output is byte-stable
enough to commit.

Three chart types cover the story the benchmark tells:

* :func:`radar_chart`      -- per-dimension ability, the "jagged profile" view
* :func:`grouped_bar_chart` -- Pass@1 next to Heal@1, the headline comparison
* :func:`stacked_bar_chart` -- the five-way error taxonomy, i.e. *where* losses come from

All three are pure functions from numbers to a string, which makes them trivially
testable and keeps them usable from the CLI, the report and the test suite alike.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape

__all__ = [
    "PALETTE",
    "ChartError",
    "Series",
    "grouped_bar_chart",
    "legend_padding",
    "radar_chart",
    "stacked_bar_chart",
    "svg_header",
]

#: Model colours, in order. Chosen to stay distinguishable in the common forms of
#: colour-vision deficiency and to read on both light and dark GitHub themes.
PALETTE: tuple[str, ...] = (
    "#2f6fed",  # blue
    "#16a34a",  # green
    "#d97706",  # amber
    "#9333ea",  # purple
    "#dc2626",  # red
    "#0891b2",  # cyan
    "#65a30d",  # olive
    "#db2777",  # pink
)

_FONT = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', "
    "Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif"
)
_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


class ChartError(ValueError):
    """The requested chart cannot be drawn from the supplied data."""


@dataclass(slots=True)
class Series:
    """One named series of values (one model, one error category, ...)."""

    name: str
    values: list[float | None]
    colour: str | None = None
    #: Optional per-point labels, e.g. the raw counts behind a percentage.
    labels: list[str] | None = None


@dataclass(slots=True)
class _Layout:
    """Shared geometry, so the three charts agree on padding and typography."""

    width: int
    height: int
    pad_left: int = 70
    pad_right: int = 24
    pad_top: int = 56
    pad_bottom: int = 76
    legend_row_height: int = 20
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def plot_left(self) -> float:
        return self.pad_left

    @property
    def plot_right(self) -> float:
        return self.width - self.pad_right

    @property
    def plot_top(self) -> float:
        return self.pad_top

    @property
    def plot_bottom(self) -> float:
        return self.height - self.pad_bottom

    @property
    def plot_width(self) -> float:
        return self.plot_right - self.plot_left

    @property
    def plot_height(self) -> float:
        return self.plot_bottom - self.plot_top


def svg_header(width: int, height: int, title: str, description: str = "") -> list[str]:
    """Open an SVG document with accessible metadata.

    ``role="img"`` plus a ``<title>``/``<desc>`` pair means screen readers announce
    the chart instead of skipping it, which matters because the chart is often the
    only place a result is shown.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="sb-title sb-desc" font-family="{_FONT}">',
        f'<title id="sb-title">{escape(title)}</title>',
        f'<desc id="sb-desc">{escape(description or title)}</desc>',
    ]
    return parts


def _text_width(text: str, *, font_size: float = 12.5) -> float:
    """Estimate the rendered width of ``text`` in pixels.

    A CJK ideograph occupies roughly twice the advance width of a Latin character,
    so measuring every character as ASCII understates a Chinese label by ~2x. That
    is not cosmetic: the legend packs items onto rows by their estimated width, so
    underestimating makes labels overlap -- which is exactly what happened to the
    error-taxonomy legend, whose labels are Chinese.

    This remains an estimate (we embed no font metrics), but it is deliberately
    biased *upward*: a slightly-too-wide estimate wraps the legend onto an extra
    row, while a too-narrow one produces unreadable overlapping text.
    """
    ascii_width = font_size * 0.53
    wide_width = font_size * 1.0
    total = 0.0
    for char in text:
        code = ord(char)
        # CJK, Hangul, Kana, fullwidth forms and the CJK punctuation blocks all
        # render at full em width. Zero-width characters must not count at all.
        if code in (0x200B, 0x200C, 0x200D, 0xFEFF):
            continue
        if (
            0x1100 <= code <= 0x115F
            or 0x2E80 <= code <= 0xA4CF
            or 0xAC00 <= code <= 0xD7A3
            or 0xF900 <= code <= 0xFAFF
            or 0xFE30 <= code <= 0xFE6F
            or 0xFF00 <= code <= 0xFF60
            or 0xFFE0 <= code <= 0xFFE6
            or 0x1F300 <= code <= 0x1FAFF  # emoji
            or 0x20000 <= code <= 0x3FFFD
        ):
            total += wide_width
        else:
            total += ascii_width
    return total


def _legend_metrics(series: Sequence[Series], width: float) -> tuple[list[float], list[list[int]]]:
    """Return per-item widths and the wrapped row assignment.

    Shared by the drawing code and the layout code: the layout needs the row count
    *before* it can reserve enough bottom padding, and drawing needs the same
    wrapping. Computing it in one place keeps the two from disagreeing.
    """
    swatch = 12
    swatch_gap = 6
    item_gap = 22
    available = width - 32

    widths = [swatch + swatch_gap + _text_width(item.name) + item_gap for item in series]

    rows: list[list[int]] = [[]]
    row_width = 0.0
    for index, item_width in enumerate(widths):
        # A single item wider than the canvas still gets its own row rather than
        # being dropped.
        if row_width + item_width > available and rows[-1]:
            rows.append([])
            row_width = 0.0
        rows[-1].append(index)
        row_width += item_width
    return widths, rows


def legend_padding(
    series: Sequence[Series],
    width: float,
    *,
    base: int = 76,
    row_height: int = 20,
) -> int:
    """Bottom padding needed so the wrapped legend fits inside the canvas.

    Without this the legend simply overflows: a fixed 96px reserve fits one row, and
    a legend with long (e.g. Chinese) labels wraps to two or three.
    """
    if not series:
        return base
    _, rows = _legend_metrics(series, width)
    return base + (len(rows) - 1) * row_height


def _legend(
    series: Sequence[Series],
    layout: _Layout,
    *,
    y: float | None = None,
) -> list[str]:
    """Draw a wrapped legend centred at the bottom of the chart.

    Items are packed onto rows by their *measured* width, so a legend of long
    labels (Chinese ones, typically) wraps instead of overlapping.
    """
    items = [(item.name, item.colour or PALETTE[i % len(PALETTE)]) for i, item in enumerate(series)]
    if not items:
        return []

    swatch = 12
    swatch_gap = 6
    item_gap = 22
    widths, rows = _legend_metrics(series, layout.width)

    base_y = y if y is not None else layout.height - 14 - (len(rows) - 1) * layout.legend_row_height
    out: list[str] = []
    for row_index, row in enumerate(rows):
        row_total = sum(widths[i] for i in row) - item_gap  # no trailing gap
        x = max(16.0, (layout.width - row_total) / 2)
        y_pos = base_y + row_index * layout.legend_row_height
        for i in row:
            name, colour = items[i]
            out.append(
                f'<rect x="{x:.1f}" y="{y_pos - 8:.1f}" width="{swatch}" height="{swatch}" '
                f'rx="3" fill="{colour}"/>'
            )
            out.append(
                f'<text x="{x + swatch + swatch_gap:.1f}" y="{y_pos + 1:.1f}" '
                f'font-size="12.5" fill="#3d4756">{escape(name)}</text>'
            )
            x += widths[i]
    return out


def _nice_max(value: float) -> float:
    """Round an axis maximum up to a readable number (1, 2, 5, 10, 20, ...)."""
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    for factor in (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10):
        candidate = factor * magnitude
        if value <= candidate:
            return float(candidate)
    return float(10 * magnitude)


def _empty_chart(width: int, height: int, title: str, message: str) -> str:
    layout = _Layout(width=width, height=height)
    parts = svg_header(width, height, title, message)
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(
        f'<text x="{width / 2:.0f}" y="{layout.height / 2:.0f}" text-anchor="middle" '
        f'font-size="14" fill="#8a94a6">{escape(message)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Radar
# --------------------------------------------------------------------------- #


def radar_chart(
    axes: Sequence[str],
    series: Sequence[Series],
    *,
    title: str = "",
    width: int = 720,
    height: int = 560,
    max_value: float = 100.0,
    rings: int = 4,
) -> str:
    """Draw a multi-series radar chart over ``axes``.

    Values are expected on a 0..``max_value`` scale. ``None`` is treated as 0 for
    the shape but reported as a gap in the tooltip-free labels, because a model
    that was never scored on a dimension should not look like it scored zero.
    """
    if len(axes) < 3:
        return _empty_chart(
            width, height, title or "Radar", "A radar chart needs at least 3 dimensions"
        )
    if not series:
        return _empty_chart(width, height, title or "Radar", "No models to compare")

    layout = _Layout(
        width=width,
        height=height,
        pad_top=64,
        pad_bottom=legend_padding(series, width, base=96),
    )
    cx = width / 2
    cy = layout.plot_top + layout.plot_height / 2
    radius = min(layout.plot_width, layout.plot_height) / 2 - 34
    if radius <= 20:
        raise ChartError("chart is too small to draw a radar plot")

    count = len(axes)
    angles = [-math.pi / 2 + 2 * math.pi * i / count for i in range(count)]

    def point(index: int, fraction: float) -> tuple[float, float]:
        angle = angles[index]
        r = radius * fraction
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    parts = svg_header(
        width,
        height,
        title or "Radar chart",
        f"{len(series)} model(s) across {count} dimensions: {', '.join(axes)}",
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    if title:
        parts.append(
            f'<text x="{cx:.0f}" y="30" text-anchor="middle" font-size="15" '
            f'font-weight="600" fill="#1c2430">{escape(title)}</text>'
        )

    # Concentric grid + radial spokes.
    for ring in range(1, rings + 1):
        fraction = ring / rings
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(i, fraction) for i in range(count)))
        parts.append(f'<polygon points="{points}" fill="none" stroke="#e3e8ef" stroke-width="1"/>')
        # Ring value label, on the vertical axis just above each ring.
        _, ring_y = point(0, fraction)
        parts.append(
            f'<text x="{cx + 6:.1f}" y="{ring_y + 4:.1f}" font-size="10" fill="#9aa4b2">'
            f"{max_value * fraction:.0f}</text>"
        )

    for index in range(count):
        x, y = point(index, 1.0)
        parts.append(
            f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{x:.2f}" y2="{y:.2f}" '
            f'stroke="#e3e8ef" stroke-width="1"/>'
        )

    # Axis labels, pushed outward and anchored by quadrant.
    for index, label in enumerate(axes):
        x, y = point(index, 1.0)
        dx, dy = x - cx, y - cy
        norm = math.hypot(dx, dy) or 1.0
        lx = x + dx / norm * 20
        ly = y + dy / norm * 20
        anchor = "middle"
        if dx > 0.35 * radius:
            anchor = "start"
        elif dx < -0.35 * radius:
            anchor = "end"
        parts.append(
            f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="{anchor}" font-size="12.5" '
            f'font-weight="500" fill="#3d4756">{escape(label)}</text>'
        )

    # Series polygons, drawn back to front so the first model stays readable.
    for series_index, item in enumerate(series):
        colour = item.colour or PALETTE[series_index % len(PALETTE)]
        if len(item.values) != count:
            raise ChartError(f"series {item.name!r} has {len(item.values)} values but {count} axes")
        points = []
        for index, value in enumerate(item.values):
            fraction = 0.0 if value is None else max(0.0, min(float(value), max_value)) / max_value
            points.append(point(index, fraction))
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<polygon points="{path}" fill="{colour}" fill-opacity="0.10" '
            f'stroke="{colour}" stroke-width="2" stroke-linejoin="round"/>'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#ffffff" '
                f'stroke="{colour}" stroke-width="1.8"/>'
            )

    parts.extend(_legend(series, layout))
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Grouped bars
# --------------------------------------------------------------------------- #


def grouped_bar_chart(
    categories: Sequence[str],
    series: Sequence[Series],
    *,
    title: str = "",
    y_label: str = "",
    width: int = 860,
    height: int = 460,
    max_value: float | None = None,
    value_suffix: str = "%",
    show_values: bool = True,
) -> str:
    """Draw grouped vertical bars: one group per category, one bar per series."""
    if not categories:
        return _empty_chart(width, height, title or "Comparison", "No data to plot")
    if not series:
        return _empty_chart(width, height, title or "Comparison", "No series to plot")

    for item in series:
        if len(item.values) != len(categories):
            raise ChartError(
                f"series {item.name!r} has {len(item.values)} values but "
                f"{len(categories)} categories"
            )

    present = [v for item in series for v in item.values if v is not None]
    observed = max(present) if present else 1.0
    top = _nice_max(max(observed, 1.0)) if max_value is None else max_value
    if top <= 0:
        raise ChartError("axis maximum must be positive")

    layout = _Layout(
        width=width,
        height=height,
        pad_top=64,
        pad_bottom=legend_padding(series, width, base=96),
    )
    group_width = layout.plot_width / len(categories)
    bar_gap = 3.0
    # Leave a fifth of each group as breathing room between groups.
    bar_width = max(3.0, (group_width * 0.78 - bar_gap * (len(series) - 1)) / len(series))

    parts = svg_header(
        width,
        height,
        title or "Comparison",
        f"{len(series)} series across {len(categories)} categories",
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    if title:
        parts.append(
            f'<text x="{width / 2:.0f}" y="30" text-anchor="middle" font-size="15" '
            f'font-weight="600" fill="#1c2430">{escape(title)}</text>'
        )

    ticks = 5
    for tick in range(ticks + 1):
        value = top * tick / ticks
        y = layout.plot_bottom - (value / top) * layout.plot_height
        parts.append(
            f'<line x1="{layout.plot_left:.1f}" y1="{y:.1f}" x2="{layout.plot_right:.1f}" '
            f'y2="{y:.1f}" stroke="#eef2f7" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{layout.plot_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#8a94a6">{value:.0f}{escape(value_suffix)}</text>'
        )
    if y_label:
        parts.append(
            f'<text x="18" y="{(layout.plot_top + layout.plot_bottom) / 2:.0f}" '
            f'text-anchor="middle" font-size="11.5" fill="#8a94a6" '
            f'transform="rotate(-90 18 {(layout.plot_top + layout.plot_bottom) / 2:.0f})">'
            f"{escape(y_label)}</text>"
        )

    for group_index, category in enumerate(categories):
        group_left = layout.plot_left + group_index * group_width
        used = bar_width * len(series) + bar_gap * (len(series) - 1)
        start = group_left + (group_width - used) / 2

        for series_index, item in enumerate(series):
            colour = item.colour or PALETTE[series_index % len(PALETTE)]
            value = item.values[group_index]
            if value is None:
                continue
            fraction = max(0.0, min(float(value), top)) / top
            bar_height = fraction * layout.plot_height
            x = start + series_index * (bar_width + bar_gap)
            y = layout.plot_bottom - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{max(bar_height, 0.6):.2f}" fill="{colour}" rx="2.5"/>'
            )
            if show_values and bar_width >= 16:
                label = (
                    item.labels[group_index]
                    if item.labels and group_index < len(item.labels)
                    else f"{value:.1f}{value_suffix}"
                )
                parts.append(
                    f'<text x="{x + bar_width / 2:.2f}" y="{y - 5:.2f}" text-anchor="middle" '
                    f'font-size="10.5" font-weight="600" fill="#3d4756">{escape(label)}</text>'
                )

        parts.append(
            f'<text x="{group_left + group_width / 2:.2f}" y="{layout.plot_bottom + 18:.1f}" '
            f'text-anchor="middle" font-size="12" fill="#3d4756">{escape(category)}</text>'
        )

    parts.append(
        f'<line x1="{layout.plot_left:.1f}" y1="{layout.plot_bottom:.1f}" '
        f'x2="{layout.plot_right:.1f}" y2="{layout.plot_bottom:.1f}" '
        f'stroke="#cbd5e1" stroke-width="1.2"/>'
    )
    parts.extend(_legend(series, layout))
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Stacked bars
# --------------------------------------------------------------------------- #


def stacked_bar_chart(
    categories: Sequence[str],
    series: Sequence[Series],
    *,
    title: str = "",
    y_label: str = "cases",
    width: int = 860,
    height: int = 460,
    value_suffix: str = "",
    show_totals: bool = True,
) -> str:
    """Draw stacked vertical bars: one bar per category, segments per series.

    Used for the error taxonomy, where each bar is a model and the segments are the
    five verdict classes -- the reader wants both the total (how many cases) and the
    composition (which failure dominated).
    """
    if not categories:
        return _empty_chart(width, height, title or "Distribution", "No data to plot")
    if not series:
        return _empty_chart(width, height, title or "Distribution", "No series to plot")

    for item in series:
        if len(item.values) != len(categories):
            raise ChartError(
                f"series {item.name!r} has {len(item.values)} values but "
                f"{len(categories)} categories"
            )

    totals = [sum(item.values[index] or 0.0 for item in series) for index in range(len(categories))]
    top = _nice_max(max(totals) if totals else 1.0)

    layout = _Layout(
        width=width,
        height=height,
        pad_top=64,
        pad_bottom=legend_padding(series, width, base=96),
    )
    group_width = layout.plot_width / len(categories)
    bar_width = max(6.0, group_width * 0.52)

    parts = svg_header(
        width,
        height,
        title or "Distribution",
        f"{len(series)} segments across {len(categories)} categories",
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    if title:
        parts.append(
            f'<text x="{width / 2:.0f}" y="30" text-anchor="middle" font-size="15" '
            f'font-weight="600" fill="#1c2430">{escape(title)}</text>'
        )

    ticks = 5
    for tick in range(ticks + 1):
        value = top * tick / ticks
        y = layout.plot_bottom - (value / top) * layout.plot_height
        parts.append(
            f'<line x1="{layout.plot_left:.1f}" y1="{y:.1f}" x2="{layout.plot_right:.1f}" '
            f'y2="{y:.1f}" stroke="#eef2f7" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{layout.plot_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#8a94a6">{value:.0f}</text>'
        )
    if y_label:
        parts.append(
            f'<text x="18" y="{(layout.plot_top + layout.plot_bottom) / 2:.0f}" '
            f'text-anchor="middle" font-size="11.5" fill="#8a94a6" '
            f'transform="rotate(-90 18 {(layout.plot_top + layout.plot_bottom) / 2:.0f})">'
            f"{escape(y_label)}</text>"
        )

    for group_index, category in enumerate(categories):
        centre = layout.plot_left + group_index * group_width + group_width / 2
        x = centre - bar_width / 2
        cursor = layout.plot_bottom
        for series_index, item in enumerate(series):
            colour = item.colour or PALETTE[series_index % len(PALETTE)]
            value = item.values[group_index] or 0.0
            if value <= 0:
                continue
            segment = (value / top) * layout.plot_height
            cursor -= segment
            parts.append(
                f'<rect x="{x:.2f}" y="{cursor:.2f}" width="{bar_width:.2f}" '
                f'height="{max(segment, 0.4):.2f}" fill="{colour}"/>'
            )
            # Only label a segment wide enough to hold the number.
            if segment >= 15:
                label = (
                    item.labels[group_index]
                    if item.labels and group_index < len(item.labels)
                    else f"{value:.0f}{value_suffix}"
                )
                parts.append(
                    f'<text x="{centre:.2f}" y="{cursor + segment / 2 + 3.5:.2f}" '
                    f'text-anchor="middle" font-size="10.5" font-weight="600" '
                    f'fill="#ffffff">{escape(label)}</text>'
                )

        if show_totals and totals[group_index] > 0:
            parts.append(
                f'<text x="{centre:.2f}" y="{cursor - 6:.2f}" text-anchor="middle" '
                f'font-size="11" font-weight="600" fill="#3d4756">'
                f"{totals[group_index]:.0f}</text>"
            )
        parts.append(
            f'<text x="{centre:.2f}" y="{layout.plot_bottom + 18:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#3d4756">{escape(category)}</text>'
        )

    parts.append(
        f'<line x1="{layout.plot_left:.1f}" y1="{layout.plot_bottom:.1f}" '
        f'x2="{layout.plot_right:.1f}" y2="{layout.plot_bottom:.1f}" '
        f'stroke="#cbd5e1" stroke-width="1.2"/>'
    )
    parts.extend(_legend(series, layout))
    parts.append("</svg>")
    return "\n".join(parts)
