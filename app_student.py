
app_student.py

100%
"""
app_student.py
===============
STUDENT VERSION — run_agent() below is incomplete (see the numbered
comments and # TODOs inside it). You'll get an AI assistant to write the
missing code for you: see 04_guided_practice_class_24.md for the exact steps and a
ready-to-use prompt.

Everything else here (finance_tools.py, llm_provider.py, the Streamlit UI
below) is already built for you — mostly to save time in class. None of it
is hard; you could build it yourself with a bit more time. The interesting
part is the five steps already described in run_agent()'s comments: reason,
act, observe, repeat. That loop — the AI deciding what to do next rather
than following a fixed script — is what makes this an agent.

Run locally with:
    streamlit run app_student.py
(Not required for class — see 04_guided_practice_class_24.md for the browser-only path.)
"""

import streamlit as st

from finance_tools import run_full_simulation, dollar_allocation, compute_allocation
from llm_provider import call_model, build_tool_result_messages, TOOL_SPEC, PAID_MODEL_DEFAULT, FREE_MODEL_DEFAULT

MAX_AGENT_STEPS = 8  # safety cap: prevents a runaway tool-calling loop


def build_system_prompt(client: dict) -> str:
    return f"""You are a retirement-planning agent talking to a financial advisor.

CLIENT PROFILE (fixed — you cannot change these):
- Current age: {client['age']}
- Risk tolerance: {client['risk_tolerance']}
- Current savings: ${client['current_savings']:,.0f}
- Target retirement amount: ${client['target_amount']:,.0f}

STARTING PLAN (you CAN adjust these via the run_simulation tool):
- Monthly contribution: ${client['monthly_contribution']:,.0f}
- Retirement age: {client['target_age']}

GOAL:
Find a plan with a probability of success >= 80%, while minimizing
disruption to the client's life. Prefer the smallest reasonable increase
in monthly contribution and/or the smallest reasonable delay in retirement
age. You may adjust contribution alone, retirement age alone, or a blend
of both — use your judgment about what's a reasonable ask for a real
person, and explain your reasoning before each tool call.

If you cannot reach 80% within a monthly contribution of $5,000 or a
retirement age of 75, say so honestly instead of proposing an unrealistic
plan.

STOPPING RULE: As soon as a simulation you've run reaches the 80% target,
you may stop and give your final recommendation right away, or test at
most one more alternative if you have a specific reason to think a
meaningfully less disruptive option exists. Do not keep testing
indefinitely once you've found a plan that meets the goal — thoroughness
has a cost too. When you're ready to conclude, respond in plain text and
do not call the tool again; that is how this program knows you're finished.

When you're done, give a final recommendation in plain language: the
allocation, the dollar amounts, the final contribution/retirement age you
landed on, and the resulting probability of success. Do not call any more
tools once you've decided on a final recommendation — just respond with
text.
"""


# ----------------------------------------------------------------------------
# THE AGENT LOOP — this is what you're building.
# ----------------------------------------------------------------------------
def run_agent(client: dict, api_key: str, model: str):
    """
    Reasoning -> act -> observe -> repeat. Yields log entries as they happen
    so the Streamlit UI can render them incrementally.

    Each yielded entry: {"type": "reasoning"|"tool_call"|"tool_result"|"final",
                          "content": ..., "step": int}
    """
    system_prompt = build_system_prompt(client)
    messages = [
        {
            "role": "user",
            "content": (
                "Here is the client's current plan. Please evaluate it and, if "
                "needed, search for a plan that reaches an 80% success probability."
            ),
        }
    ]

    for step in range(MAX_AGENT_STEPS):
        # 1. REASON: call the model with the conversation so far, and get its
        #    turn back. If it produced any reasoning text, yield it as a
        #    "reasoning" log entry (type="reasoning", content=<the text>,
        #    step=step + 1).
        turn = call_model(
            messages=messages,
            system_prompt=system_prompt,
            api_key=api_key,
            model=model,
            tools=TOOL_SPEC,
        )
        if turn.get("content"):
            yield {"type": "reasoning", "content": turn["content"], "step": step + 1}

        # 2. Did the model decide it's done, or does it want to ACT (call a
        #    tool)? Check turn["tool_calls"]. If it's empty, the model is
        #    finished: yield a "final" entry with its text as content, and
        #    return from the generator (stop the loop).
        tool_calls = turn.get("tool_calls") or []
        if not tool_calls:
            yield {"type": "final", "content": turn.get("content"), "step": step + 1}
            return

        # 3. ACT: for each requested tool call in turn["tool_calls"], pull out
        #    its arguments (tool_call["arguments"]), yield a "tool_call" log
        #    entry, then actually run the simulation by calling
        #    run_full_simulation(...) with those arguments plus the fixed
        #    client fields (age, risk_tolerance, target_amount,
        #    current_savings, n_trials). Yield a "tool_result" entry with the
        #    result.
        #
        # 4. OBSERVE: package the result using build_tool_result_messages(...)
        #    and append the returned messages to `messages`, so the next loop
        #    iteration's call_model() sees what happened.
        for tool_call in tool_calls:
            args = tool_call["arguments"]
            yield {"type": "tool_call", "content": args, "step": step + 1}

            result = run_full_simulation(
                **args,
                age=client["age"],
                risk_tolerance=client["risk_tolerance"],
                target_amount=client["target_amount"],
                current_savings=client["current_savings"],
                n_trials=client["n_trials"],
            )
            yield {"type": "tool_result", "content": result, "step": step + 1}

            messages.extend(build_tool_result_messages(turn, tool_call, result))

        # 5. REPEAT: loop back to step 1 with the updated conversation history.
        #    (No code needed here — this just happens because it's a `for` loop.)

    yield {
        "type": "final",
        "content": (
            f"Hit the {MAX_AGENT_STEPS}-step safety cap without settling on a final "
            "answer. In a real deployment you'd investigate why it kept calling tools."
        ),
        "step": MAX_AGENT_STEPS,
    }


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Wealth Planning Agent", page_icon="💰", layout="centered")
st.title("💰 Wealth Planning Agent")
st.caption(
    "An LLM agent — not a hardcoded search — decides how to adjust your plan "
    "to reach an 80% chance of hitting your retirement goal."
)

with st.sidebar:
    st.header("Model")
    plan = st.selectbox(
        "OpenRouter plan",
        ["Paid (recommended)", "Free"],
        index=0,
        help="Paid: a strong, reliable model, small real cost per run (get a key "
             "and add ~$10 in credits at openrouter.ai). Free: no cost, but "
             "rate-limited and less reliable at using tools.",
    )
    default_model = PAID_MODEL_DEFAULT if plan.startswith("Paid") else FREE_MODEL_DEFAULT
    api_key = st.text_input(
        "OpenRouter API key",
        type="password",
        help="Get one free at openrouter.ai/keys. Falls back to st.secrets / "
             "the OPENROUTER_API_KEY environment variable if left blank.",
    )
    model = st.text_input(
        "Model",
        value=default_model,
        help="Check openrouter.ai/models for other options.",
    )

    st.header("Client inputs")
    age = st.number_input("Current age", min_value=18, max_value=74, value=35)
    risk_tolerance = st.selectbox("Risk tolerance", ["conservative", "moderate", "aggressive"], index=1)
    current_savings = st.number_input("Current savings ($)", min_value=0, value=80_000, step=1000)
    monthly_contribution = st.number_input("Current monthly contribution ($)", min_value=0, value=800, step=25)
    target_age = st.number_input("Target retirement age", min_value=int(age) + 1, max_value=90, value=65)
    target_amount = st.number_input("Target amount ($)", min_value=0, value=1_500_000, step=10_000)
    n_trials = st.select_slider(
        "Monte Carlo trials per simulation",
        options=[100, 200, 500, 1000, 2000, 5000],
        value=500,
        help="Lower = faster/noisier. Good for live class demos; raise it for a final run.",
    )
    run_clicked = st.button("Run agent", type="primary", use_container_width=True)

weights_preview = compute_allocation(age, risk_tolerance)
dollars_preview = dollar_allocation(current_savings, weights_preview)

st.subheader("Starting plan")
col1, col2, col3 = st.columns(3)
col1.metric("Equity (VTI)", f"{weights_preview['equity']*100:.0f}%", f"${dollars_preview['equity']:,.0f}")
col2.metric("Bonds (BND)", f"{weights_preview['bonds']*100:.0f}%", f"${dollars_preview['bonds']:,.0f}")
col3.metric("Cash", f"{weights_preview['cash']*100:.0f}%", f"${dollars_preview['cash']:,.0f}")

if run_clicked:
    import os

    resolved_key = api_key or st.secrets.get("OPENROUTER_API_KEY", None) or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_key:
        st.error("No OpenRouter API key found. Enter one in the sidebar, or set OPENROUTER_API_KEY.")
        st.stop()

    client = {
        "age": age,
        "risk_tolerance": risk_tolerance,
        "current_savings": current_savings,
        "monthly_contribution": monthly_contribution,
        "target_age": target_age,
        "target_amount": target_amount,
        "n_trials": n_trials,
    }

    st.subheader("Agent reasoning log")
    log_container = st.container()
    final_answer = None

    with log_container:
        try:
            for entry in run_agent(client, resolved_key, model):
                if entry["type"] == "reasoning":
                    with st.chat_message("assistant"):
                        st.markdown(f"**Step {entry['step']} — reasoning:**\n\n{entry['content']}")
                elif entry["type"] == "tool_call":
                    with st.chat_message("assistant", avatar="🛠️"):
                        st.markdown(f"**Step {entry['step']} — calling `run_simulation`**")
                        st.json(entry["content"])
                elif entry["type"] == "tool_result":
                    with st.chat_message("user", avatar="📊"):
                        st.markdown(f"**Step {entry['step']} — simulation result**")
                        st.json(entry["content"])
                elif entry["type"] == "final":
                    final_answer = entry["content"]
        except Exception as e:
            st.error(f"Agent run failed: {e}")
            st.stop()

    if final_answer:
        st.subheader("Final recommendation")
        st.success(final_answer)
Displaying app_student.py.
