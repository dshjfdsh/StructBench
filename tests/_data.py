"""Shared test data and case factories.

Kept free of pytest imports so that both ``conftest.py`` (which turns these into
fixtures) and the test modules themselves can use them.
"""

from __future__ import annotations

from structbench.models import Case

#: A deliberately nasty schema: four levels of nesting, an enum, an array with a
#: minimum length, a nullable formatted string, and ``additionalProperties:
#: false`` so that invented keys are caught rather than silently dropped.
NESTED_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string", "minLength": 2},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "deleted_at": {"type": ["string", "null"], "format": "date-time"},
            },
            "required": ["id", "name", "tags"],
            "additionalProperties": False,
        },
        "status": {"enum": ["active", "banned"]},
        "note": {"type": ["string", "null"]},
    },
    "required": ["user", "status"],
    "additionalProperties": False,
}

#: The canonical compliant response for :data:`NESTED_SCHEMA`.
GOOD_PAYLOAD: dict = {
    "user": {"id": 8675309, "name": "Ada", "tags": ["x"]},
    "status": "active",
    "note": None,
}

WEATHER_TOOL: dict = {
    "name": "get_weather",
    "description": "Look up the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 1},
            "unit": {"enum": ["celsius", "fahrenheit"]},
            "days": {"type": "integer", "minimum": 1, "maximum": 7},
        },
        "required": ["city", "unit"],
        "additionalProperties": False,
    },
}

#: Three near-duplicate decoys -- exactly the "similar tools" trap from the
#: anti-hallucination dimension.
DECOY_TOOLS: list[dict] = [
    WEATHER_TOOL,
    {
        "name": "get_weather_forecast",
        "description": "Look up a multi-day weather forecast for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
            },
            "required": ["city", "start_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_weather_history",
        "description": "Look up historical weather records for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "year": {"type": "integer"}},
            "required": ["city", "year"],
            "additionalProperties": False,
        },
    },
]

#: A tool whose ``new_date`` is mandatory. The clarify prompt never supplies it,
#: so the only acceptable behaviour is to ask for it.
BOOKING_TOOL: dict = {
    "name": "rebook_flight",
    "description": "Move an existing flight booking to a different date.",
    "parameters": {
        "type": "object",
        "properties": {
            "booking_id": {"type": "string", "pattern": "^BK-[0-9]{4}$"},
            "new_date": {"type": "string", "format": "date"},
            "reason": {"type": ["string", "null"]},
        },
        "required": ["booking_id", "new_date"],
        "additionalProperties": False,
    },
}


def nested_case() -> Case:
    """A JSON-mode case with value-level ground truth."""
    return Case(
        id="nest_001",
        title="four-level nested profile",
        category="deep_nesting",
        prompt="Return the user profile as JSON.",
        target_schema=NESTED_SCHEMA,
        ground_truth={
            "must_equal": {"$.user.id": 8675309},
            "must_be_null": ["$.note"],
            "must_not_contain": {"$.user.name": "TODO"},
            "forbidden_paths": ["$.user.password"],
        },
    )


def tool_call_case() -> Case:
    """A tool-mode case where the model must emit a call."""
    return Case(
        id="tool_001",
        category="tool_hallucination",
        prompt="What is the weather in Beijing right now, in celsius?",
        tools=DECOY_TOOLS,
        expect={"behaviour": "call", "expected_tool": "get_weather"},
    )


def clarify_case() -> Case:
    """A tool-mode case where a required argument is missing on purpose."""
    return Case(
        id="tool_002",
        category="tool_hallucination",
        prompt="Rebook booking BK-4471 to a later date, please.",
        tools=[BOOKING_TOOL],
        expect={"behaviour": "clarify"},
    )
