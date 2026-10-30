
import json
import time
import uuid
from pathlib import Path

import pandas as pd
import mock_llm

# 1. Paths and data
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "raw"
CLEAN = BASE / "data" / "clean"
OUT = BASE / "outputs"
STATE = BASE / "state"

OUT.mkdir(exist_ok=True)
STATE.mkdir(exist_ok=True)

orders = pd.read_csv(CLEAN / "orders.csv")
policies = list((BASE / "data" / "policies").glob("*.txt"))

# 2. Tool specifications
TOOLS = [
    {"name": "get_order", "description": "Find an order by order ID"},
    {"name": "search_policy", "description": "Search refund policy"},
    {"name": "calculate_refund", "description": "Calculate refund amount"},
    {"name": "issue_refund", "description": "Issue an approved refund"},
    {"name": "escalate_to_human", "description": "Send refund to human support lead"}
]

# 3. Tool functions
def get_order(order_id):
    found = orders[orders["order_id"] == order_id]

    if found.empty:
        return {"error": "Order not found"}

    row = found.iloc[0]

    return {
        "order_id": str(row["order_id"]),
        "customer_id": str(row["customer_id"]),
        "status": str(row["status"]),
        "order_amount": float(row["order_amount"]),
        "delivery_minutes": (
            None if pd.isna(row["delivery_minutes"])
            else float(row["delivery_minutes"])
        )
    }


def search_policy(query):
    policy_text = ""

    for file in policies:
        policy_text += file.read_text(encoding="utf-8") + "\n"

    return {
        "query": query,
        "policy_found": "Refund Policy" in policy_text,
        "policy_documents": [file.name for file in policies]
    }


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None:
        refund = 0
    elif delivery_minutes > 45:
        refund = float(order_amount)
    else:
        refund = 0

    return {"refund_inr": round(refund, 2)}


def issue_refund(order_id, refund_inr):
    ledger_path = OUT / "refund_ledger.jsonl"

    # Prevent duplicate refunds for the same order
    if ledger_path.exists():
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                old_record = json.loads(line)

                if (
                    old_record.get("tool") == "issue_refund"
                    and old_record.get("order_id") == order_id
                ):
                    return {
                        "status": "already_issued",
                        "order_id": order_id,
                        "refund_inr": old_record["refund_inr"]
                    }

    record = {
        "timestamp": time.time(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": refund_inr
    }

    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {
        "status": "issued",
        "order_id": order_id,
        "refund_inr": refund_inr
    }


def escalate_to_human(order_id, refund_inr):
    record = {
        "timestamp": time.time(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": refund_inr
    }

    with open(OUT / "refund_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {
        "status": "escalated",
        "order_id": order_id,
        "refund_inr": refund_inr
    }


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# 4. Execute a tool call
def execute_tool(name, arguments):
    if name not in TOOL_MAP:
        return {"error": "Unknown tool"}

    try:
        result = TOOL_MAP[name](**arguments)
    except Exception as error:
        result = {"error": str(error)}

    return result


# 5. Save checkpoint
def save_checkpoint(run_id, scenario_id, messages, tokens, step, executed):
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens,
        "step": step,
        "tools_executed": executed
    }

    path = STATE / f"{run_id}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)

    return path


# 6. ReAct agent
def run_agent(
    scenario_id,
    user_message,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after=None,
    resume_data=None
):
    run_id = (
        resume_data["run_id"]
        if resume_data else str(uuid.uuid4())
    )

    if resume_data:
        messages = resume_data["messages"]
        tokens = resume_data["tokens_used"]
        executed = resume_data["tools_executed"]
        step = resume_data["step"]
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "AGENT. You are an Instamart refund support agent. "
                    "Use tools to verify orders and follow refund policy."
                )
            },
            {"role": "user", "content": user_message}
        ]
        tokens = 0
        executed = []
        step = 0

    start_time = time.time()
    final_answer = None
    status = "completed"
    llm_calls = 0

    while step < max_iterations:
        if time.time() - start_time > time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = "Human review required: time limit reached."
            break

        if tokens >= token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = "Human review required: token limit reached."
            break

        try:
            response = mock_llm.chat(messages, tools=TOOLS)
            llm_calls += 1
        except mock_llm.TransientLLMError:
            status = "FALLBACK_LLM_ERROR"
            final_answer = "Human review required: temporary LLM error."
            break

        tokens += response.usage.total_tokens

        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            final_answer = message.content

            if final_answer and final_answer.startswith("FALLBACK"):
                status = final_answer.split(":")[0]

            break

        for call in tool_calls:
            step += 1

            name = call.name
            arguments = (
                json.loads(call.arguments)
                if isinstance(call.arguments, str)
                else call.arguments
            )

            result = execute_tool(name, arguments)
            executed.append(name)

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call.id,
                    "name": name,
                    "arguments": call.arguments
                }]
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(result)
            })

            save_checkpoint(
                run_id, scenario_id, messages,
                tokens, step, executed
            )

            if crash_after is not None and step == crash_after:
                return {
                    "status": "SIMULATED_CRASH",
                    "run_id": run_id,
                    "final_answer": None,
                    "steps": step,
                    "llm_calls": llm_calls,
                    "tokens_used": tokens,
                    "tools_executed": executed
                }

    else:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = "Human review required: maximum iterations reached."

    if final_answer is None:
        final_answer = "Human review required."

    save_checkpoint(
        run_id, scenario_id, messages,
        tokens, step, executed
    )

    return {
        "status": status,
        "run_id": run_id,
        "final_answer": final_answer,
        "steps": step,
        "llm_calls": llm_calls,
        "tokens_used": tokens,
        "tools_executed": executed
    }


# 7. Resume an interrupted agent
def resume_agent(run_id, max_iterations=6, token_limit=6000,
                 time_limit_seconds=10):
    path = STATE / f"{run_id}.json"

    with open(path, encoding="utf-8") as f:
        checkpoint = json.load(f)

    return run_agent(
        checkpoint["scenario_id"],
        "",
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        resume_data=checkpoint
    )


# 8. Load and run seven scenarios
scenarios = pd.read_csv(
    DATA / "agent_scenarios.csv",
    keep_default_na=False
)

print("Total scenarios:", len(scenarios))

results = []

for _, row in scenarios.iterrows():
    mock_llm.reset()

    result = run_agent(
        str(row["scenario_id"]),
        str(row["message"])
    )

    print("\nScenario:", row["scenario_id"])
    print("Status:", result["status"])
    print("Answer:", result["final_answer"])
    print("Tools:", result["tools_executed"])

    results.append({
        "scenario_id": row["scenario_id"],
        "expected_outcome": row["expected_outcome"],
        "status": result["status"],
        "final_answer": result["final_answer"],
        "steps": result["steps"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "tools_executed": json.dumps(result["tools_executed"])
    })

pd.DataFrame(results).to_json(
    OUT / "agent_runs.jsonl",
    orient="records",
    lines=True
)


# 9. Token-limit test
mock_llm.reset()
token_test = run_agent(
    "TOKEN_TEST",
    "Please refund order ORD-100048.",
    token_limit=200
)

print("\nToken test:", token_test["status"])


# 10. Timeout test
mock_llm.reset()
timeout_test = run_agent(
    "TIMEOUT_TEST",
    "Please refund order ORD-100048.",
    time_limit_seconds=0
)

print("Timeout test:", timeout_test["status"])


# 11. Checkpoint and resume test using S1
s1 = scenarios[scenarios["scenario_id"] == "S1"].iloc[0]

mock_llm.reset()
crashed = run_agent(
    "S1_RESUME_TEST",
    s1["message"],
    crash_after=2
)

print("\nSimulated crash:", crashed["status"])

resumed = resume_agent(crashed["run_id"])

print("Resumed answer:", resumed["final_answer"])
print("Tools executed after resume:", resumed["tools_executed"])

print("\nPart D ReAct Agent completed!")
