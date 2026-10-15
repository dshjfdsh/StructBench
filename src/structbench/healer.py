"""The multi-turn self-healing loop (StructBench's headline metric).

Heal@1 answers a question no Pass@1 number can: *when the model gets the format
wrong, does a precise error message let it fix itself?* That is exactly what an
agent runtime does when it validates a tool call and retries, so it predicts
real-world robustness far better than raw first-try accuracy.

The loop is deliberately austere. Between attempts the model receives **only**
:func:`structbench.evaluator.build_feedback` -- the deterministic verdict and the
attributed field-level problems. No hints, no re-prompting with extra
information, no judge model. That keeps the measurement reproducible and its
interpretation unambiguous.
"""

from __future__ import annotations

from structbench.models import Attempt, AttemptStatus, Case, Message, ToolCall

__all__ = [
    "MAX_REPLAY_ROUNDS",
    "build_conversation",
    "render_assistant_turn",
    "should_heal",
]

#: Cap on how many prior rounds are replayed, so that a long healing chain cannot
#: blow up the context window (and the API bill) on cases with ``max_rounds > 2``.
MAX_REPLAY_ROUNDS = 2

_FALLBACK_FEEDBACK = (
    "StructBench validation feedback\n"
    "Your previous response was rejected: it did not satisfy the required output "
    "contract. Reply with corrected output only."
)


def render_assistant_turn(attempt: Attempt) -> str:
    """Render a previous model turn back into the transcript.

    Tool calls are flattened to ``[tool call] name(arguments)`` rather than being
    replayed as a real ``tool_calls`` array: StructBench never executes a tool, so
    there is no tool result to supply, and OpenAI-compatible servers reject an
    assistant turn whose tool calls have no matching ``role: "tool"`` reply. The
    flattened form keeps the request valid on every endpoint while still showing
    the model exactly what it produced.
    """
    if attempt.tool_calls:
        rendered = "; ".join(
            f"[tool call] {call.name}({call.arguments_raw or '{}'})" for call in attempt.tool_calls
        )
        if attempt.raw_text:
            return f"{attempt.raw_text}\n{rendered}"
        return rendered
    return attempt.raw_text


def build_conversation(case: Case, attempts: list[Attempt]) -> list[Message]:
    """Assemble the message list for the next attempt.

    ``attempts`` holds every completed attempt so far; the newest one determines
    the repair instruction.
    """
    messages: list[Message] = []
    if case.system:
        messages.append(Message(role="system", content=case.system))
    messages.append(Message(role="user", content=case.prompt))

    for attempt in attempts[-MAX_REPLAY_ROUNDS:]:
        messages.append(Message(role="assistant", content=render_assistant_turn(attempt)))
        feedback = attempt.feedback or _FALLBACK_FEEDBACK
        messages.append(Message(role="user", content=feedback))

    return messages


def should_heal(case: Case, attempt: Attempt, rounds_used: int) -> bool:
    """Decide whether another attempt is worth making.

    Transport failures are *not* healed: the model never saw the request, so a
    retry would measure the network rather than the model. Those are recorded as
    provider failures and scored separately.
    """
    if attempt.passed:
        return False
    if attempt.status is not AttemptStatus.OK:
        return False
    if not attempt.feedback:
        return False
    return rounds_used < case.max_rounds


def tool_call_summary(calls: list[ToolCall]) -> str:
    """Compact one-line description of the tool calls in an attempt."""
    if not calls:
        return "(no tool call)"
    return ", ".join(f"{call.name}({call.arguments_raw or '{}'})" for call in calls)
