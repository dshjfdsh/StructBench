"""StructBench — stress-testing LLMs on extreme JSON schemas, nested tool calls,
and self-healing recovery.

StructBench is a lightweight micro-benchmark that measures how reliably a model
can emit *machine-consumable* output: strict JSON for a given JSON Schema, or a
well-formed tool call. Unlike MMLU-style benchmarks it needs no judge model --
every verdict is produced deterministically by a JSON parser plus a Pydantic v2
validator, which makes the whole suite free, millisecond-fast and 100%
reproducible.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
