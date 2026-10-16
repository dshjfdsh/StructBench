"""Tests for the SVG chart renderer and the terminal visual summary.

Charts fail silently: a malformed SVG or wrong geometry renders as a blank
rectangle on GitHub with no error anywhere. So these tests do not merely check that
a string was produced -- they rasterise it with the standard-library renderer in
:mod:`tests.svg_raster` and assert that ink landed where it should.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from structbench.metrics import compute_run_result
from structbench.models import (
    ALL_LABELS,
    ErrorLabel,
    LatencyStats,
    ModelScore,
)
from structbench.reporter.charts import (
    CATEGORY_LABELS,
    build_charts,
    error_taxonomy_series,
    model_colours,
    ordered_models,
    pass_heal_series,
    radar_series,
    write_charts,
)
from structbench.reporter.svg import (
    PALETTE,
    ChartError,
    Series,
    grouped_bar_chart,
    radar_chart,
    stacked_bar_chart,
)
from tests.svg_raster import count_ink, rasterise, render_svg_to_png

SVG_NS = "{http://www.w3.org/2000/svg}"


def elements(svg: str, tag: str) -> list[ET.Element]:
    return list(ET.fromstring(svg).iter(f"{SVG_NS}{tag}"))


def data_rects(svg: str, *, height: float) -> list[ET.Element]:
    """Rectangles that are chart *data*, not legend swatches.

    The legend reuses the palette colours, so filtering on ``fill`` alone counts
    swatches as bars. Data rects are distinguishable by sitting above the legend
    band, which starts near the bottom of the canvas.
    """
    root = ET.fromstring(svg)
    legend_top = float(root.get("height") or 0) - 60
    return [
        rect
        for rect in root.iter(f"{SVG_NS}rect")
        if rect.get("fill") in PALETTE and float(rect.get("y") or 0) < legend_top
    ]


def ink_ratio(svg: str) -> float:
    canvas = rasterise(svg)
    return count_ink(canvas) / (canvas.width * canvas.height)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_score(
    name: str,
    *,
    pass_at_1: float,
    heal: float | None = None,
    by_category: dict[str, float] | None = None,
    errors: dict[ErrorLabel, int] | None = None,
) -> ModelScore:
    """A scorecard shaped like the real thing, with controllable rates."""
    categories = by_category or {}
    blocks = [
        {
            "category": category,
            "total": 22,
            "pass_at_1": int(rate * 22),
            "heal_at_1": int(rate * 22),
            "failed_at_1": 22 - int(rate * 22),
            "healed": 0,
            "pass_at_1_rate": rate,
            "heal_at_1_rate": rate,
            "repair_rate": None,
        }
        for category, rate in categories.items()
    ]
    heal_rate = pass_at_1 if heal is None else heal
    return ModelScore.model_validate(
        {
            "model": name,
            "total": 86,
            "pass_at_1": int(pass_at_1 * 86),
            "heal_at_1": int(heal_rate * 86),
            "failed_at_1": 86 - int(pass_at_1 * 86),
            "healed": int(heal_rate * 86) - int(pass_at_1 * 86),
            "pass_at_1_rate": pass_at_1,
            "heal_at_1_rate": heal_rate,
            "repair_rate": None,
            "latency": LatencyStats(p50=200.0, p95=400.0),
            "ttft": LatencyStats(p50=80.0),
            "tps": 150.0,
            "error_distribution": {
                label.value: (errors or {}).get(label, 0) for label in ALL_LABELS
            },
            "detail_distribution": {},
            "by_category": blocks,
        }
    )


@pytest.fixture
def two_model_result():
    """Two models whose weaknesses differ, so the radar shapes must differ."""
    return compute_run_result(
        version="0.0-test",
        records=[],
        models=[],
        datasets=["datasets/tiny.yaml"],
        case_count=86,
        concurrency=4,
        max_rounds=2,
        meta={},
    ).model_copy(
        update={
            "models": [
                make_score(
                    "strong",
                    pass_at_1=0.95,
                    heal=1.0,
                    by_category={
                        "deep_nesting": 0.95,
                        "escape_hell": 0.90,
                        "tool_hallucination": 0.98,
                        "edge_types": 0.97,
                    },
                    errors={ErrorLabel.VALID_PASS: 82, ErrorLabel.SYNTAX_TRUNCATED: 4},
                ),
                make_score(
                    "escape-weak",
                    pass_at_1=0.60,
                    heal=0.80,
                    by_category={
                        "deep_nesting": 0.90,
                        "escape_hell": 0.20,
                        "tool_hallucination": 0.88,
                        "edge_types": 0.92,
                    },
                    errors={
                        ErrorLabel.VALID_PASS: 52,
                        ErrorLabel.SYNTAX_TRUNCATED: 30,
                        ErrorLabel.SCHEMA_TYPE_MISMATCH: 4,
                    },
                ),
            ]
        }
    )


# --------------------------------------------------------------------------- #
# Radar
# --------------------------------------------------------------------------- #


class TestRadar:
    def test_produces_wellformed_svg_with_visible_ink(self) -> None:
        svg = radar_chart(
            ["A", "B", "C", "D"],
            [Series(name="m1", values=[90, 80, 70, 60])],
            title="radar",
        )
        root = ET.fromstring(svg)
        assert root.get("viewBox") == "0 0 720 560"
        # role/title/desc make the chart accessible rather than an unlabelled blob.
        assert root.get("role") == "img"
        assert elements(svg, "title") and elements(svg, "desc")
        assert ink_ratio(svg) > 0.02

    def test_one_polygon_per_series_plus_grid(self) -> None:
        svg = radar_chart(
            ["A", "B", "C"],
            [Series(name="m1", values=[90, 80, 70]), Series(name="m2", values=[10, 20, 30])],
            rings=4,
        )
        polygons = elements(svg, "polygon")
        # 4 grid rings + 2 data polygons.
        assert len(polygons) == 6

    def test_values_are_clamped_and_none_is_treated_as_zero(self) -> None:
        svg = radar_chart(
            ["A", "B", "C"],
            [Series(name="m", values=[None, 150.0, -20.0])],
        )
        # Must not raise, and must still emit a polygon.
        assert elements(svg, "polygon")

    def test_series_length_must_match_the_axis_count(self) -> None:
        with pytest.raises(ChartError, match="values but"):
            radar_chart(["A", "B", "C"], [Series(name="bad", values=[1, 2])])

    def test_too_few_axes_degrades_to_a_message(self) -> None:
        svg = radar_chart(["A", "B"], [Series(name="m", values=[1, 2])], title="t")
        assert "at least 3 dimensions" in svg
        assert ET.fromstring(svg) is not None

    def test_no_series_degrades_to_a_message(self) -> None:
        svg = radar_chart(["A", "B", "C"], [], title="t")
        assert "No models" in svg


# --------------------------------------------------------------------------- #
# Grouped bars
# --------------------------------------------------------------------------- #


class TestGroupedBars:
    def test_one_rect_per_non_null_value_plus_labels(self) -> None:
        svg = grouped_bar_chart(
            ["m1", "m2"],
            [
                Series(name="Pass@1", values=[90.0, 60.0]),
                Series(name="Heal@1", values=[100.0, 80.0]),
            ],
            title="t",
        )
        rects = data_rects(svg, height=460)
        assert len(rects) == 4
        assert ink_ratio(svg) > 0.02

    def test_null_values_are_skipped_not_drawn_as_zero(self) -> None:
        svg = grouped_bar_chart(
            ["m1", "m2"],
            [Series(name="Repair", values=[50.0, None])],
        )
        assert len(data_rects(svg, height=460)) == 1, (
            "a null must leave a gap, not a zero-height bar"
        )

    def test_bar_heights_are_proportional(self) -> None:
        svg = grouped_bar_chart(
            ["only"],
            [Series(name="s", values=[50.0])],
            max_value=100.0,
            height=460,
        )
        bar = data_rects(svg, height=460)[0]
        height = float(bar.get("height"))
        # Half of a 100% axis should be roughly half the plot height, not exact
        # because of padding, but definitely not full and not zero.
        assert 100 < height < 300

    def test_series_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ChartError, match="values but"):
            grouped_bar_chart(["a", "b"], [Series(name="s", values=[1.0])])

    def test_empty_inputs_degrade_to_a_message(self) -> None:
        assert "No data" in grouped_bar_chart([], [Series(name="s", values=[])])
        assert "No series" in grouped_bar_chart(["a"], [])


# --------------------------------------------------------------------------- #
# Stacked bars
# --------------------------------------------------------------------------- #


class TestStackedBars:
    def test_segments_stack_to_the_total(self) -> None:
        svg = stacked_bar_chart(
            ["m1"],
            [
                Series(name="a", values=[30.0]),
                Series(name="b", values=[20.0]),
            ],
            show_totals=True,
        )
        rects = data_rects(svg, height=460)
        assert len(rects) == 2
        # Segments must tile the stack without gaps or overlap. Sorting by top
        # edge puts the upper segment first in SVG coordinates (y grows downward).
        spans = sorted(
            (float(r.get("y")), float(r.get("y")) + float(r.get("height"))) for r in rects
        )
        (upper_top, upper_bottom), (lower_top, lower_bottom) = spans
        assert upper_bottom <= lower_top + 1, "segments overlap"
        assert lower_bottom >= upper_bottom, "segments are inverted"
        # 50 units on a 100-unit axis is half the plot, so the stack must be tall.
        assert (lower_bottom - upper_top) > 80

    def test_zero_segments_are_not_drawn(self) -> None:
        svg = stacked_bar_chart(
            ["m1"],
            [Series(name="a", values=[30.0]), Series(name="b", values=[0.0])],
        )
        assert len(data_rects(svg, height=460)) == 1

    def test_legend_lists_every_series(self) -> None:
        svg = stacked_bar_chart(
            ["m1"], [Series(name="alpha", values=[1.0]), Series(name="beta", values=[1.0])]
        )
        labels = [t.text for t in elements(svg, "text")]
        assert "alpha" in labels and "beta" in labels


# --------------------------------------------------------------------------- #
# RunResult -> charts
# --------------------------------------------------------------------------- #


class TestRunResultCharts:
    def test_model_order_is_stable_and_ranked(self, two_model_result) -> None:
        ordered = ordered_models(two_model_result)
        assert [s.model for s in ordered] == ["strong", "escape-weak"]
        # Colours are assigned by rank, so they must not shuffle between calls.
        assert model_colours(two_model_result) == model_colours(two_model_result)

    def test_radar_series_follow_the_shared_model_order(self, two_model_result) -> None:
        series = radar_series(two_model_result)
        assert [s.name for s in series] == ["strong", "escape-weak"]
        assert all(len(s.values) == 4 for s in series)

    def test_radar_series_reflect_the_jagged_profile(self, two_model_result) -> None:
        series = radar_series(two_model_result)
        weak = next(s for s in series if s.name == "escape-weak")
        # The escape_hell dimension is axis index 1 in the canonical order.
        assert weak.values[1] == pytest.approx(20.0)
        assert weak.values[3] == pytest.approx(92.0)

    def test_pass_heal_series_pairs_the_two_rates(self, two_model_result) -> None:
        series = pass_heal_series(two_model_result)
        assert [s.name for s in series] == ["Pass@1", "Heal@1"]
        assert series[0].values == [95.0, 60.0]
        assert series[1].values == [100.0, 80.0]

    def test_error_taxonomy_covers_every_label(self, two_model_result) -> None:
        series = error_taxonomy_series(two_model_result)
        assert len(series) == len(ALL_LABELS)
        # Labels are the Chinese display names, which is what the report shows.
        assert all(s.name for s in series)
        failures_only = error_taxonomy_series(two_model_result, include_pass=False)
        assert len(failures_only) == len(ALL_LABELS) - 1

    def test_build_charts_returns_all_three_named_charts(self, two_model_result) -> None:
        charts = build_charts(two_model_result)
        assert set(charts) == {"radar", "pass-heal", "taxonomy"}
        for name, svg in charts.items():
            ET.fromstring(svg)  # must parse
            assert ink_ratio(svg) > 0.02, f"{name} renders as (nearly) blank"

    def test_subtitle_is_recorded_in_every_chart(self, two_model_result) -> None:
        charts = build_charts(two_model_result, subtitle="reference run")
        for svg in charts.values():
            titles = " ".join(t.text or "" for t in elements(svg, "text"))
            assert "reference run" in titles

    def test_write_charts_creates_the_directory_and_files(self, two_model_result, workdir) -> None:
        target = workdir / "nested" / "charts"
        written = write_charts(two_model_result, target, subtitle="x")
        assert set(written) == {"radar", "pass-heal", "taxonomy"}
        for path in written.values():
            assert path.is_file()
            assert path.read_text(encoding="utf-8").startswith("<svg")

    def test_write_charts_can_select_a_subset(self, two_model_result, workdir) -> None:
        written = write_charts(two_model_result, workdir, names=["radar"])
        assert set(written) == {"radar"}
        assert not (workdir / "taxonomy.svg").exists()

    def test_category_labels_cover_the_dataset_dimensions(self) -> None:
        # If a dataset dimension is added without a label, its axis would show a
        # raw key like "deep_nesting"; this keeps the mapping honest.
        assert set(CATEGORY_LABELS) == {
            "deep_nesting",
            "escape_hell",
            "tool_hallucination",
            "edge_types",
        }


# --------------------------------------------------------------------------- #
# Rasteriser (the verification tool itself)
# --------------------------------------------------------------------------- #


class TestRasteriser:
    def test_rejects_malformed_svg(self) -> None:
        # Raw string: this is a regex alternation, and RUF043 wants that explicit.
        with pytest.raises(Exception, match=r"malformed|no usable"):
            rasterise("<svg><rect")

    def test_rejects_svg_without_dimensions(self) -> None:
        with pytest.raises(Exception, match=r"width/height"):
            rasterise("<svg></svg>")

    def test_writes_a_valid_png(self, workdir) -> None:
        svg = radar_chart(["A", "B", "C"], [Series(name="m", values=[1, 2, 3])])
        png = render_svg_to_png(svg, workdir / "chart.png")
        data = png.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        # IHDR carries the dimensions; check they match the SVG canvas.
        assert data[16:20] == (720).to_bytes(4, "big")
        assert data[20:24] == (560).to_bytes(4, "big")

    def test_blank_svg_has_no_ink(self) -> None:
        blank = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'
        assert count_ink(rasterise(blank)) == 0
