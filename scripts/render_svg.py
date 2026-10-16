"""Development helper: rasterise SVGs to PNG so a human can look at them.

Usage::

    python scripts/render_svg.py docs/charts/*.svg

Writes a sibling ``.png`` for each input. Uses the standard-library rasteriser in
``tests/svg_raster.py``, so it works with no browser, no ImageMagick and no
Cairo -- which is what makes it usable inside a locked-down sandbox where a
headless browser refuses to start.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tests.svg_raster import RenderError, count_ink, render_svg_to_png


def main(argv: list[str]) -> int:
    targets = [Path(arg) for arg in argv[1:]]
    if not targets:
        print("usage: python scripts/render_svg.py <file.svg> [...]")
        return 2

    failures = 0
    for svg_path in targets:
        if not svg_path.is_file():
            print(f"  MISSING {svg_path}")
            failures += 1
            continue
        png_path = svg_path.with_suffix(".png")
        try:
            render_svg_to_png(svg_path.read_text(encoding="utf-8"), png_path, scale=1.0)
        except RenderError as exc:
            print(f"  FAIL    {svg_path}: {exc}")
            failures += 1
            continue

        # Re-rasterise to report ink, so a "renders but is blank" SVG is caught.
        from tests.svg_raster import rasterise

        canvas = rasterise(svg_path.read_text(encoding="utf-8"))
        ink = count_ink(canvas)
        ratio = ink / (canvas.width * canvas.height)
        print(
            f"  OK      {svg_path} -> {png_path.name}  "
            f"{canvas.width}x{canvas.height}  ink={ink:,} ({ratio:.1%})"
        )
        if ratio < 0.01:
            print(f"          WARNING: only {ratio:.2%} of pixels are drawn")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
