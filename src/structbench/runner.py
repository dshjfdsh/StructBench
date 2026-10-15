"""The asynchronous multi-model benchmark runner.

Design constraints that shaped this module:

* **Bounded concurrency.** A single :class:`asyncio.Semaphore` caps in-flight
  requests so that a free API tier's rate limit is respected instead of tripped.
  An optional token-bucket rate limiter (``--rps``) smooths the burst further.
* **One broken call never kills the run.** Failures, whether transport-level or
  a malformed response, become recorded :class:`~structbench.models.Attempt`
  values. A 90-minute run must not die on request 812.
* **One task per (model, case).** Everything a case needs is independent of every
  other case, so the work is embarrassingly parallel and the results are
  order-independent.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from structbench import __version__
from structbench.evaluator import CompiledCase, compile_case, evaluate
from structbench.healer import build_conversation, should_heal
from structbench.metrics import compute_run_result
from structbench.models import (
    Attempt,
    Case,
    CaseResult,
    ModelSpec,
    RunResult,
)
from structbench.providers import (
    Provider,
    ProviderConfigError,
    ProviderOptions,
    ProviderRequest,
    build_provider,
)

__all__ = ["RateLimiter", "RunConfig", "Runner", "run_sync"]

#: Called after every case finishes, for progress rendering.
ProgressCallback = Callable[[str, Case, CaseResult], None]


class RateLimiter:
    """A minimal async token bucket: at most ``rps`` acquisitions per second."""

    def __init__(self, rps: float) -> None:
        if rps <= 0:
            raise ValueError("rps must be positive")
        self._interval = 1.0 / rps
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)


@dataclass
class RunConfig:
    """Everything needed to execute one benchmark run."""

    models: list[ModelSpec]
    cases: list[Case]
    concurrency: int = 5
    options: ProviderOptions = field(default_factory=ProviderOptions)
    #: Ask servers to constrain output to JSON (off by default -- see below).
    json_mode: bool = False
    tool_choice: str | None = None
    seed: int = 0
    #: Optional requests-per-second cap on top of the concurrency limit.
    rps: float | None = None
    dataset_files: list[str] = field(default_factory=list)
    progress: ProgressCallback | None = None

    @property
    def max_rounds(self) -> int:
        return max((case.max_rounds for case in self.cases), default=2)


class Runner:
    """Executes a :class:`RunConfig` and produces a :class:`RunResult`."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        # Compile once per case: building Pydantic validators per attempt would
        # dominate the runtime on a 100-case x 5-model sweep.
        self._compiled: dict[str, CompiledCase] = {}
        for case in config.cases:
            try:
                self._compiled[case.id] = compile_case(case)
            except Exception as exc:
                raise ValueError(f"case {case.id!r} could not be compiled: {exc}") from None

    # -- setup ------------------------------------------------------------- #

    def _build_providers(self) -> dict[str, Provider]:
        providers: dict[str, Provider] = {}
        problems: list[str] = []
        for spec in self.config.models:
            try:
                providers[spec.display] = build_provider(spec, self.config.options)
            except ProviderConfigError as exc:
                problems.append(f"  - {spec.display}: {exc}")
        if problems:
            raise ProviderConfigError(
                "no usable provider for the requested model(s):\n" + "\n".join(problems)
            )
        return providers

    # -- execution --------------------------------------------------------- #

    async def run(self) -> RunResult:
        providers = self._build_providers()
        semaphore = asyncio.Semaphore(max(self.config.concurrency, 1))
        limiter = RateLimiter(self.config.rps) if self.config.rps else None

        tasks: list[asyncio.Task[CaseResult]] = []
        try:
            for spec in self.config.models:
                provider = providers[spec.display]
                for case in self.config.cases:
                    tasks.append(
                        asyncio.create_task(
                            self._run_case(provider, spec, case, semaphore, limiter)
                        )
                    )
            records: list[CaseResult] = list(await asyncio.gather(*tasks))
        finally:
            for provider in providers.values():
                await provider.aclose()

        triplets = [(spec.display, spec.provider, spec.display) for spec in self.config.models]
        return compute_run_result(
            version=__version__,
            records=records,
            models=triplets,
            datasets=self.config.dataset_files,
            case_count=len(self.config.cases),
            concurrency=self.config.concurrency,
            max_rounds=self.config.max_rounds,
            meta={
                "json_mode": self.config.json_mode,
                "tool_choice": self.config.tool_choice,
                "seed": self.config.seed,
                "temperature": self.config.options.temperature,
                "max_tokens": self.config.options.max_tokens,
                "stream": self.config.options.stream,
                "rps": self.config.rps,
                "compiler_warnings": sorted(
                    {
                        warning
                        for compiled in self._compiled.values()
                        for warning in compiled.warnings
                    }
                ),
            },
        )

    async def _run_case(
        self,
        provider: Provider,
        spec: ModelSpec,
        case: Case,
        semaphore: asyncio.Semaphore,
        limiter: RateLimiter | None,
    ) -> CaseResult:
        compiled = self._compiled[case.id]
        attempts: list[Attempt] = []

        async with semaphore:
            while True:
                if limiter is not None:
                    await limiter.acquire()

                request = ProviderRequest(
                    messages=build_conversation(case, attempts),
                    tools=case.tools,
                    # JSON mode is only requested for JSON-shaped cases: combining
                    # response_format with tools is rejected by several servers.
                    json_mode=self.config.json_mode and not case.is_tool,
                    tool_choice=self.config.tool_choice,
                    case=case,
                    round_index=len(attempts) + 1,
                    seed=self.config.seed,
                )
                completion = await provider.complete(request)
                attempt = evaluate(compiled, completion, round_index=len(attempts) + 1)
                attempts.append(attempt)

                if not should_heal(case, attempt, len(attempts)):
                    break

        result = self._to_result(spec, case, attempts)
        if self.config.progress is not None:
            self.config.progress(spec.display, case, result)
        return result

    @staticmethod
    def _to_result(spec: ModelSpec, case: Case, attempts: list[Attempt]) -> CaseResult:
        first = attempts[0]
        last = attempts[-1]
        passed_at_1 = first.passed
        passed_after_heal = any(attempt.passed for attempt in attempts)
        return CaseResult(
            case_id=case.id,
            category=case.category,
            model=spec.display,
            title=case.title,
            difficulty=case.difficulty,
            attempts=attempts,
            passed_at_1=passed_at_1,
            passed_after_heal=passed_after_heal,
            healed=passed_after_heal and not passed_at_1,
            first_label=first.label,
            first_detail=first.detail,
            final_label=last.label,
            final_detail=last.detail,
            total_latency_ms=sum(attempt.latency_ms for attempt in attempts),
            total_tokens=sum(
                (attempt.prompt_tokens or 0) + (attempt.completion_tokens or 0)
                for attempt in attempts
            ),
        )


def run_sync(config: RunConfig) -> RunResult:
    """Blocking convenience wrapper for scripts and tests."""
    return asyncio.run(Runner(config).run())
