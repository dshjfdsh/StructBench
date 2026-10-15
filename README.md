<div align="center">

# StructBench

**Stress-testing LLMs on extreme JSON schemas, nested tool calls, and self-healing recovery.**

专治大模型「格式崩坏」与「参数幻觉」：面向 Agent 工具调用与极端 Schema 的轻量级评测基准。

[![CI](https://github.com/dshjfdsh/StructBench/actions/workflows/ci.yml/badge.svg)](https://github.com/dshjfdsh/StructBench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-279%20passing-brightgreen.svg)](tests/)
[![Cases](https://img.shields.io/badge/cases-86%20adversarial-orange.svg)](datasets/)
[![Judge model](https://img.shields.io/badge/judge%20model-not%20required-success.svg)](#why-no-llm-as-a-judge)

</div>

---

## Why another benchmark?

Because "the model is good at JSON" is not a measurable claim, and the failure
that actually breaks an agent in production is not a wrong answer — it is a
**missing closing brace**.

StructBench is a *micro*-benchmark with a deliberately narrow target: can a model
emit output that a program can consume? Strict JSON for a given schema, or a
well-formed tool call. It scores 86 hand-written adversarial cases in four
dimensions, attributes every failure to one of five root causes, and measures
whether a model can **repair its own output when handed a precise error message**.

| | |
| :--- | :--- |
| **No judge model** | Every verdict comes from a JSON parser plus a Pydantic v2 validator — 0 cost, millisecond latency, 100% reproducible. |
| **No API key required** | Seven offline simulator personas drive the entire pipeline, so you can clone, run, and read a real report in under a minute. |
| **Beyond Pass@1** | `Heal@1` quantifies the metric that actually matters for agent runtimes: recovery after a validation error. |
| **Actionable failures** | Not "failed", but `Syntax_Truncated / unclosed_bracket` at `$.user.tags` — a bug report, not a score. |

### The five pathologies

StructBench never reports a bare pass/fail. Every failed attempt is attributed to
exactly one root cause, with ~40 fine-grained sub-codes underneath:

| Label | 中文 | What it means |
| :--- | :--- | :--- |
| `Markdown_Wrapper_Leak` | Markdown 包裹泄漏 | Valid JSON, but wrapped in a ```` ```json ```` fence or padded with conversational filler. |
| `Syntax_Truncated` | 语法残缺/转义错误 | Unparseable: unbalanced brackets, unterminated strings, bad escapes, trailing commas. |
| `Schema_Type_Mismatch` | Schema 类型不匹配 | Valid JSON of the wrong shape: missing required field, `"123"` where an integer belongs. |
| `Enum_Or_Value_Hallucination` | 枚举/参数幻觉 | The right shape filled with invented values: a made-up enum, an undeclared field, a fabricated argument. |
| `Valid_Pass` | 合规通过 | Fully compliant. |

---

## The four dimensions

The suite contains **86 adversarial cases**, each verified to be *satisfiable*
(some response passes) and *discriminating* (an obvious mutation fails) — see
[Dataset integrity](#dataset-integrity).

| Dimension | Cases | What it attacks |
| :--- | ---: | :--- |
| `deep_nesting` | 22 | Object nesting ≥ 4 levels, `anyOf`/`oneOf` unions, strict `enum`/`const`, `pattern`, `format`, type arrays, `allOf` + `$defs`/`$ref`, numeric bounds. |
| `escape_hell` | 22 | Getting **hard content into a JSON string**: multi-line code, SQL with embedded quotes, regex backslashes, LaTeX, Markdown tables, Windows paths, JSON-inside-JSON, emoji/ZWJ/zero-width characters, PEM keys, JWTs. |
| `tool_hallucination` | 22 | Picking the right tool among 5 near-duplicate decoys; refusing to call when a required argument is missing; not inventing `api_key` fields or fabricating unknown data. |
| `edge_types` | 20 | The type system itself: `"123"` vs `123`, `1.0` as an integer, boolean traps, `multipleOf` on money, big integers, every `format`, `null` vs absent. |

Composition: 62 JSON-mode · 13 tool-call · 9 clarify (anti-hallucination) ·
2 explicit-null. Difficulty spread 2–5, with 16 cases at difficulty 5.

---

## Quick start

No API key needed — the offline simulator exercises the real evaluation path:

```bash
git clone https://github.com/dshjfdsh/StructBench
cd structbench
pip install -e ".[dev]"

# Validate the dataset integrity (every case solvable + discriminating)
structbench validate

# Run the whole suite against seven simulated models, no network at all
structbench run -m mock:good,mock:flaky,mock:escape_phobic,mock:typing_weak \
                --concurrency 8 --markdown -
```

`results/report.html` opens as one file with the whole run embedded as JSON — a
radar chart, a stacked error-taxonomy chart, latency comparison, and a filterable
explorer showing each model's actual output next to the expected schema. The chart
*library* loads from a CDN by default; add `--inline-echarts path/to/echarts.min.js`
for an artifact that renders with no network at all.

### Against real models

StructBench speaks the OpenAI chat-completions protocol, which covers DeepSeek,
DashScope/Qwen, Zhipu GLM, Moonshot, SiliconFlow, OpenAI, Ollama, vLLM and
LM Studio:

```bash
cp .env.example .env        # then export the keys you have
export STRUCTBENCH_DEEPSEEK_API_KEY=sk-...

structbench run \
  -m deepseek:deepseek-chat,qwen-plus,glm-4-flash,ollama:qwen2.5:7b \
  --concurrency 5 --rps 3 \
  --markdown leaderboard.md \
  -o results
```

`--concurrency` plus `--rps` keeps a free tier comfortable. `structbench providers`
lists every preset and the environment variable it reads.

---

## Reference run

Measured with the offline simulator on the full 86-case suite — reproducible
with `structbench run -m mock:perfect,mock:good,mock:flaky,mock:weak,mock:escape_phobic,mock:typing_weak,mock:tool_butcher`.

<!-- LEADERBOARD:START -->
| Model | Pass@1 | Heal@1 | Repair | P50 | TTFT p50 | TPS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mock/perfect` | **100.0%** | 100.0% | n/a | 189ms | 65ms | 284.6 |
| `mock/good` | **88.4%** | 100.0% | 100.0% | 253ms | 88ms | 212.4 |
| `mock/escape_phobic` | **77.9%** | 94.2% | 73.7% | 299ms | 104ms | 182.5 |
| `mock/typing_weak` | **70.9%** | 93.0% | 76.0% | 309ms | 108ms | 173.1 |
| `mock/tool_butcher` | **67.4%** | 91.9% | 75.0% | 269ms | 94ms | 193.0 |
| `mock/flaky` | **66.3%** | 95.3% | 86.2% | 402ms | 141ms | 141.3 |
| `mock/weak` | **34.9%** | 66.3% | 48.2% | 664ms | 237ms | 76.6 |

**Pass@1 by dimension** — the personas are built to be *jagged*, and the suite
locates the jaggedness precisely:

| Model | deep_nesting | escape_hell | tool_hallucination | edge_types |
| :--- | ---: | ---: | ---: | ---: |
| `mock/perfect` | 100.0% | 100.0% | 100.0% | 100.0% |
| `mock/good` | 86.4% | 90.9% | 81.8% | 95.0% |
| `mock/escape_phobic` | 90.9% | **40.9%** | 86.4% | 95.0% |
| `mock/typing_weak` | 63.6% | 90.9% | 86.4% | **40.0%** |
| `mock/tool_butcher` | 63.6% | 81.8% | **40.9%** | 85.0% |
| `mock/flaky` | 68.2% | **36.4%** | 86.4% | 75.0% |
| `mock/weak` | 36.4% | 27.3% | 36.4% | 40.0% |

**Error taxonomy (first attempts)** — where the losses actually come from:

| Model | 包裹泄漏 | 语法残缺 | 类型不匹配 | 枚举/幻觉 | 合规通过 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `mock/perfect` | 0 | 0 | 0 | 0 | 86 |
| `mock/good` | 2 | 2 | 2 | 4 | 76 |
| `mock/escape_phobic` | 1 | **9** | 4 | 5 | 67 |
| `mock/typing_weak` | 5 | 7 | 6 | 7 | 61 |
| `mock/tool_butcher` | 3 | 6 | 7 | **12** | 58 |
| `mock/flaky` | 6 | **9** | **9** | 5 | 57 |
| `mock/weak` | 7 | **26** | 13 | 10 | 30 |
<!-- LEADERBOARD:END -->

> **These numbers are simulated, not measured.** They validate the harness and
> give the report a shape to be read against; they are not claims about any real
> model. A real cross-vendor leaderboard requires API budget this repository does
> not ship — `structbench run` plus your keys produces one.

**Read the `Heal@1` column, not Pass@1.** Every persona gains 15–31 points once it
is told precisely what was wrong. That gap is the argument for putting a
validation-error retry loop in an agent runtime *before* spending money on
fine-tuning.

---

## How it works

```text
datasets/*.yaml  (86 declarative cases)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      StructBench evaluation engine                    │
│                                                                      │
│  ┌────────────────┐   ┌─────────────────────┐   ┌────────────────┐  │
│  │ 1. Async pool  │──▶│ 2. Deterministic    │──▶│ 3. Healing loop│  │
│  │  Semaphore +   │   │    sandbox          │   │  ValidationError│ │
│  │  token bucket  │   │  JSON AST + Pydantic│   │  replay        │  │
│  └────────────────┘   └──────────┬──────────┘   └───────┬────────┘  │
│                                  │ verdict + issues     │ repaired  │
│                                  ▼                      ▼           │
│                    [ 5-way error taxonomy + ~40 detail codes ]       │
└──────────────────────────────────┬───────────────────────────────────┘
                                   ▼
          [ Rich terminal leaderboard ]  +  [ single-file HTML radar report ]
```

### Deterministic validation pipeline

```text
raw model output
  ├─ extract_payload()   strip ```json fences, trim prose, scan a balanced
  │                      string-aware span (braces inside strings are ignored)
  ├─ parse_payload()     json.loads, plus an ast.literal_eval recovery pass that
  │                      *explains* Python-literal output without ever passing it
  ├─ validate_payload()  strict Pydantic v2 check against the case's JSON Schema
  ├─ check_ground_truth() value-level rules (must_be_null, forbidden_paths, ...)
  └─ Attempt             one of five labels + a fine-grained detail code
```

**Attribution follows root cause.** If the payload cannot be parsed at all, the
verdict is `Syntax_Truncated` even when the model *also* wrapped it in a fence — a
fence around broken JSON is not the interesting failure. Only once the JSON parses
cleanly does a wrapper violation become the headline.

---

<a id="why-no-llm-as-a-judge"></a>

## Design decisions

**Why no LLM-as-a-judge?**
Structured output and tool calling are *deterministic* tasks. Grading them with a
model wastes budget, adds latency, and — worst of all — the judge itself
hallucinates on exactly the inputs this benchmark stresses, like `(?<=\")secret`
buried in an escaped string. A JSON parser plus a schema validator is
zero-cost, millisecond-fast, and 100% reproducible across machines and runs.

**Why strict, JSON-exact validation?**
Pydantic's default (lax) mode quietly coerces `"123"` into `123`, which would
silently bless the precise failure this benchmark exists to catch. So every
generated field declares `strict=True`, with narrowly scoped shims only where JSON
and Python genuinely disagree — JSON `1.0` *is* an integer, and an ISO-8601 string
*is* a datetime. `multipleOf` is checked in exact `Decimal` arithmetic, because
`2.5 % 0.01` is not `0` in binary floating point but JSON Schema says it is.

**Why is `--json-mode` off by default?**
Sending `response_format: json_object` makes the *server* enforce valid JSON, which
is a real and useful engineering choice — but it hides the model's native
discipline, which is what you need to know before choosing a model for an agent.
Turn it on with `--json-mode` when you specifically want the constrained
comparison.

**Why does `repair_rate` sometimes read `n/a`?**
It is `healed / failed_at_1`. If a model failed nothing, the ratio is undefined,
and printing `0%` or `100%` would both be a lie.

### Metric definitions

| Metric | Definition |
| :--- | :--- |
| `Pass@1` | Cases whose **first** attempt was fully compliant. Raw, unaided reliability. |
| `Heal@1` | Cases that passed within the healing budget (first attempt plus error-replay retries). An *overall* pass rate, so always ≥ `Pass@1`. |
| `repair_rate` | `healed / failed_at_1` — of the cases it broke, how many did it fix once told exactly what was wrong? `n/a` when nothing failed. |
| `TTFT` | Time to first streamed token. Only measurable while streaming, so `--no-stream` reports it as unknown rather than guessing. |
| `TPS` | Completion tokens ÷ generation time (total latency minus TTFT). |

Transport failures (HTTP 429/5xx) are recorded separately and are **never**
counted as model pathologies, and never "healed" — the model never saw the
request, so retrying would measure the network.

---

## CLI

```text
structbench run        Run the suite against one or more models
structbench report     Re-render console/HTML/Markdown from a saved archive
structbench validate   Self-check the datasets (solvable + discriminating)
structbench list       Enumerate the cases
structbench personas   Describe the offline simulator personas
structbench providers  List provider presets and their environment variables
```

Useful flags for `run`:

| Flag | Purpose |
| :--- | :--- |
| `-m/--models` | Model specs, comma-separated: `deepseek:deepseek-chat,ollama:qwen2.5:7b`. |
| `-d/--datasets` | Your own dataset files or directories. |
| `-c/--categories`, `-t/--tags`, `--ids`, `-n/--limit` | Deterministic subsets — `-n 20` is a reproducible 20-case run for free tiers. |
| `--concurrency`, `--rps` | Respect a rate limit instead of tripping it. |
| `--max-rounds` | Attempt budget per case (`1` disables healing). |
| `--fail-under` | Exit non-zero below a Pass@1 threshold — a CI quality gate. |
| `--markdown` | Write the leaderboard as Markdown (`-` for stdout). |
| `--inline-echarts` | Embed a local `echarts.min.js` for a fully offline report. |

---

## Extending the suite

A case is a declarative YAML/JSONL record — no Python required:

```yaml
version: 1
category: escape_hell
description: Multi-language escaping inside JSON string values.
cases:
  - id: esc_900
    title: SQL with doubled apostrophes
    difficulty: 4
    prompt: |
      Produce a JSON object with the SQL shown below.
      SELECT id FROM users WHERE name = 'it''s fine';
    target_schema:
      type: object
      additionalProperties: false
      properties:
        query:
          type: string
          minLength: 10
      required: [query]
    ground_truth:
      must_contain: {"$.query": "it''s fine"}
      must_not_contain: {"$.query": "SELECT *"}
    notes: |
      中文说明：为什么这道题刁钻。
```

Then check it does what you think:

```bash
structbench validate -d my_cases.yaml   # must be solvable and discriminating
structbench run -m mock:perfect -d my_cases.yaml
```

<a id="dataset-integrity"></a>

### Dataset integrity

A benchmark fixture can fail silently in two ways, and both would poison the
leaderboard: a case **no response can pass** (every model scores zero on it), or a
case **any response passes** (it measures nothing while inflating scores).

`structbench validate` guards both directions. It synthesises the ideal response
from the schema and asserts it passes, then applies deterministic mutations that
*must* fail — an undeclared key, a dropped required field, a string substituted
for a number. `tests/test_datasets.py` enforces the same contract in CI, which is
why the shipped suite is clean at `86/86, 0 error(s), 0 warning(s)`.

---

## Project structure

```text
structbench/
├── pyproject.toml              # deps + the `structbench` console script
├── README.md · CHANGELOG.md · LICENSE (MIT)
├── .github/workflows/ci.yml    # ruff + pytest on 3.10–3.13 + offline demo run
├── datasets/                   # the benchmark asset itself
│   ├── 01_deep_nesting.yaml    #  22 cases
│   ├── 02_escape_hell.yaml     #  22 cases
│   ├── 03_tool_hallucination.yaml  #  22 cases
│   └── 04_edge_types.yaml      #  20 cases
├── scripts/selfcheck.py        # standalone dataset validator
├── src/structbench/
│   ├── cli.py                  # Typer CLI (run/report/validate/list/personas/providers)
│   ├── evaluator.py            # extraction → AST parsing → validation → attribution
│   ├── schema_builder.py       # JSON Schema → strict Pydantic v2 validators
│   ├── runner.py               # async pool, concurrency + rate limiting
│   ├── healer.py               # the multi-turn self-healing loop
│   ├── metrics.py              # Pass@1 / Heal@1 / latency / throughput
│   ├── loader.py               # declarative dataset loading
│   ├── models.py               # every cross-module Pydantic model
│   ├── selfcheck.py            # satisfiability + discrimination checks
│   ├── synth.py · regexgen.py  # conforming-value synthesis (drives mock + self-check)
│   ├── jsonpath.py             # the JSONPath-lite subset used by ground truth
│   ├── providers/              # OpenAI-compatible client + offline simulator
│   └── reporter/               # Rich console, Markdown, ECharts HTML template
└── tests/                      # 279 tests
```

---

## Development

```bash
pip install -e ".[dev]"
pytest -q                       # 279 tests
ruff check . && ruff format --check .
structbench validate            # dataset integrity
```

## Roadmap

- [ ] Publish a real cross-vendor leaderboard (requires API budget)
- [ ] Publish to PyPI
- [ ] An `--openai-tools/--legacy-functions` switch for older gateways
- [ ] A JSONL streaming output mode for long unattended runs
- [ ] Case-level regression tracking between runs (`structbench compare a.json b.json`)

## Contributing

New adversarial cases are the most valuable contribution — especially anything
that breaks a *frontier* model. Every submission must pass
`structbench validate`, which enforces the satisfiable-and-discriminating
contract automatically. Please include Chinese `notes` explaining the trap.

## License

[MIT](LICENSE)
