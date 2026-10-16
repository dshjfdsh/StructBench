# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **The HTML report rendered as a blank page without a network.** It loaded ECharts
  from a CDN, so the file people download from a Release and double-click showed
  nothing when offline or when the CDN is unreachable. The library is now vendored
  into `reporter/vendor/` and inlined by default, so `report.html` is genuinely
  self-contained. `--echarts-src URL` opts back into the CDN for a much smaller
  artifact (about 27 KB instead of 1.2 MB) when network access is acceptable.
- The Release body linked charts at `/releases/download/...`, which GitHub serves
  as `application/octet-stream` with `Content-Disposition: attachment` — a
  broken image, not an inline one. Charts are now linked from the tagged commit via
  `raw.githubusercontent.com`, which serves `image/svg+xml`.
- Legend text overlapped in the error-taxonomy chart. Legend items were packed by
  `len(text) * 6.6px`, measuring every character as ASCII, but CJK ideographs render
  at roughly full-em width — so the Chinese labels were under-measured by 24–47px
  each and ran into one another. `_text_width` now measures per character, and the
  bottom padding grows with the wrapped legend row count instead of overflowing.

### Added

- A README section explaining which Release assets matter: the wheel installs the
  tool *and* the bundled 86-case suite, `report.html` is the offline visualisation,
  and the `run-*.json` archives are the raw data behind the charts.

## [0.1.0] — 2025-10-16

First public release: a complete, runnable micro-benchmark for structured output
and tool calling, usable with **zero API keys**.

### Added

**Static SVG charts** (`structbench.reporter.svg`, `structbench.reporter.charts`)
- Dependency-free SVG rendering: radar, grouped bar and stacked bar charts written
  by hand, so the project keeps its zero-heavyweight-dependency rule and the output
  is byte-stable enough to commit.
- `structbench chart <archive> -o DIR` and `run --charts DIR` write the chart set
  (`radar.svg`, `pass-heal.svg`, `taxonomy.svg`).
- These exist because the interactive HTML report cannot help on GitHub: a README
  or Release description cannot embed a `<script>`, so the charts a visitor
  actually sees have to be static images. All three are embedded in both READMEs.
- Charts record the run they came from in their title, so a committed image is
  never unverifiable.

**Terminal visualisation** (`structbench.reporter.console`)
- `render_report` prints bar-chart summaries alongside the tables: Pass@1, Heal@1,
  repair rate, and the weakest model's per-dimension profile. Tables answer "what is
  the number?"; bars answer "how do these compare?", which is the question a
  leaderboard is actually asked. Disable with `--no-charts`.

**Automated releases** (`.github/workflows/release.yml`, `scripts/release_notes.py`)
- Pushing a `v*` tag verifies the suite, regenerates the charts from a real run,
  builds sdist + wheel, install-tests the wheel, and publishes a GitHub Release via
  the `gh` CLI. A tag containing a hyphen is published as a pre-release.
- `scripts/release_notes.py` composes the release body from the CHANGELOG section
  for that version plus the generated charts, and refuses to run if a chart it
  would link to is missing.

**Test and verification tooling**
- `tests/svg_raster.py`: a standard-library SVG rasteriser, so the suite can prove
  a chart is not blank without a browser or a plotting library.
- `scripts/render_svg.py`: renders SVGs to PNG for human inspection.
- `scripts/make_charts.py`: regenerates the committed chart set from an archive.

**Deterministic validation sandbox** (`structbench.evaluator`)
- Tolerant JSON extraction: Markdown-fence stripping, prose trimming, and a
  string-aware balanced-span scanner that survives braces inside string literals.
- Root-cause attribution priority (syntax → wrapper → schema → values), so a fence
  wrapped around broken JSON is reported as truncation, not as a wrapper leak.
- Five-way pathology taxonomy with ~40 fine-grained detail codes:
  `Markdown_Wrapper_Leak`, `Syntax_Truncated`, `Schema_Type_Mismatch`,
  `Enum_Or_Value_Hallucination`, `Valid_Pass`.
- `ast.literal_eval` recovery pass that explains Python-literal output without ever
  turning a failure into a pass.
- Value-level ground-truth rules: `must_equal`, `must_be_null`, `must_contain`,
  `must_not_contain`, `must_match`, `forbidden_paths`.

**JSON-exact Pydantic v2 compiler** (`structbench.schema_builder`)
- Runtime JSON Schema → strict Pydantic model/`TypeAdapter` translation, so `"123"`
  is rejected where an integer is required and an optional string field rejects an
  explicit `null` — the failures this benchmark exists to measure.
- Local `$ref`/`$defs`, `allOf`, `anyOf`/`oneOf`, type arrays, `enum`/`const`,
  pattern/length/range constraints, and 12 `format` values.
- Exact `Decimal` arithmetic for `multipleOf`, because `2.5 % 0.01` is not `0` in
  binary floating point but JSON Schema says it is.
- Aliases for non-identifier property names (`user-id`, `2fa`).

**Async multi-model engine** (`structbench.runner`, `structbench.healer`)
- `asyncio.Semaphore` concurrency control with an optional `--rps` token bucket.
- Streaming by default so TTFT can be measured; tool-call deltas are merged across
  SSE chunks.
- Transport failures are recorded, never raised, and never "healed".
- Multi-turn self-healing: the model receives only the deterministic verdict plus
  the attributed field-level problems, which is what makes Heal@1 interpretable.
- Transport retries with `Retry-After` support and a fallback when a server rejects
  `stream_options`.

**Metrics** (`structbench.metrics`)
- Pass@1, Heal@1, repair rate (`healed / failed_at_1`, `None` when nothing failed),
  latency and TTFT P50/P95, throughput, token accounting, per-dimension and
  per-difficulty breakdowns, and error distributions over first attempts only.

**Reporting** (`structbench.reporter`)
- Rich terminal leaderboard, per-dimension matrix, taxonomy table, top-cause table
  and hardest-failure listing; column widths tuned for an 80-column terminal.
- Single-file interactive HTML report (Jinja2 + ECharts): radar chart,
  Pass@1/Heal@1 comparison, stacked error taxonomy, difficulty decay, latency
  comparison, and a filterable failure explorer showing each model's actual output
  next to the expected schema.
- Markdown leaderboard export for pasting into a README or pull request.

**Datasets** (`datasets/`)
- 86 hand-written adversarial cases across four dimensions: `deep_nesting` (22),
  `escape_hell` (22), `tool_hallucination` (22), `edge_types` (20).
- 62 JSON-mode, 13 tool-call, 9 clarify (anti-hallucination) and 2 explicit-null
  cases; difficulty spread 2–5.

**Dataset self-check** (`structbench.selfcheck`)
- Proves every case is *satisfiable* by synthesising the ideal response and
  asserting it passes, and *discriminating* by asserting that deterministic
  mutations (undeclared key, dropped required field, wrong primitive type) fail.

**Offline simulator** (`structbench.providers.mock`)
- Seven personas driving the full pipeline with no network and no API key, seeded
  deterministically so runs are reproducible across machines.

**Regex sampler** (`structbench.regexgen`)
- Recursive-descent generator that produces a value matching a regex, so a case
  using a pattern such as `^CUST-[0-9]{4}$` can still be proven satisfiable.

**CLI** (`structbench.cli`)
- `run`, `report`, `validate`, `list`, `personas`, `providers`; `--fail-under` for
  use as a CI quality gate.

**Tooling**
- 279 unit tests, `scripts/selfcheck.py`, GitHub Actions CI across Python
  3.10–3.13, ruff lint and format enforcement, MIT licence.

[Unreleased]: https://github.com/dshjfdsh/StructBench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dshjfdsh/StructBench/releases/tag/v0.1.0
