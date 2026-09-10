"""
llm_provider.py
================
Talks to OpenRouter — a single API that can route requests to many AI
models (including free ones) — so the agent loop doesn't have to deal with
raw web requests or tool-calling formatting itself.

This file is "plumbing": it exists so the agent loop in app.py /
app_student.py can just say "call the model" and "package this tool
result" without worrying about HTTP details. It's given, not part of the
exercise.

Interface:
    call_model(api_key, model, system, messages)
        -> {
             "text": str,                # any plain-text reasoning this turn
             "tool_calls": [ {"id", "name", "arguments"} , ... ],
             "raw_assistant_message": <message to keep in conversation history>,
           }

    build_tool_result_messages(raw_assistant_message, tool_call_id, result_dict)
        -> list of message dicts to append to the conversation
           (the assistant's turn, followed by the tool's result)
"""

import json

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Two suggested defaults — students pick one in the sidebar.
# Paid: a strong, reliable model (small real cost per run).
# Free: OpenRouter's auto-router for its free-tier models — $0, but
# rate-limited and less consistent at tool calling.
PAID_MODEL_DEFAULT = "anthropic/claude-sonnet-5"
FREE_MODEL_DEFAULT = "openrouter/free"

TOOL_SPEC = {
    "name": "run_simulation",
    "description": (
        "Run a Monte Carlo retirement simulation for this client with a given "
        "monthly contribution and retirement age. Returns the resulting asset "
        "allocation, probability of hitting the target amount, and the "
        "distribution of simulated ending portfolio values. Age, risk "
        "tolerance, current savings, and the target dollar amount are fixed "
        "for this client and cannot be changed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "monthly_contribution": {
                "type": "number",
                "description": "Monthly contribution in dollars to test.",
            },
            "retirement_age": {
                "type": "integer",
                "description": "Retirement age to test.",
            },
        },
        "required": ["monthly_contribution", "retirement_age"],
    },
}


def _openai_tool():
    return {
        "type": "function",
        "function": {
            "name": TOOL_SPEC["name"],
            "description": TOOL_SPEC["description"],
            "parameters": TOOL_SPEC["parameters"],
        },
    }


def call_model(api_key: str, model: str, system: str, messages: list) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "tools": [_openai_tool()],
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        # Surface OpenRouter's actual error body — without this, hosting
        # services like Streamlit Cloud often hide the real reason from
        # anyone but the app owner digging through logs.
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"OpenRouter request failed ({resp.status_code}): {detail}")

    data = resp.json()
    message = data["choices"][0]["message"]

    tool_calls_raw = message.get("tool_calls") or []
    tool_calls = [
        {
            "id": tc["id"],
            "name": tc["function"]["name"],
            "arguments": json.loads(tc["function"]["arguments"]),
        }
        for tc in tool_calls_raw
    ]

    return {
        "text": message.get("content") or "",
        "tool_calls": tool_calls,
        "raw_assistant_message": message,
    }


def build_tool_result_messages(raw_assistant_message, tool_call_id, result_dict) -> list:
    return [
        raw_assistant_message,
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result_dict),
        },
    ]
