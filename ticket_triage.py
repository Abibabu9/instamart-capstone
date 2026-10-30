
import pandas as pd
import json
import re
import time
import random
from pathlib import Path
import mock_llm

# 1. Project paths
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "raw"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# 2. Load ticket data
tickets = pd.read_csv(DATA / "tickets.csv", keep_default_na=False)
customers = pd.read_csv(BASE / "data" / "clean" / "customers.csv")
orders = pd.read_csv(BASE / "data" / "clean" / "orders.csv")

print("Total tickets:", len(tickets))
print("Ticket columns:", tickets.columns.tolist())

# 3. Find ticket text column
possible_text_columns = ["ticket_text", "message", "description", "text"]
text_column = next(
    (col for col in possible_text_columns if col in tickets.columns),
    None
)

if text_column is None:
    raise ValueError("Ticket text column not found. Check tickets.csv columns.")

# 4. Redact personal information
def redact_text(text):
    text = str(text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text)
    text = re.sub(r"\b[6-9]\d{9}\b", "[PHONE]", text)
    text = re.sub(r"ORD-\d{6}", "[ORDER_ID]", text)
    return text

tickets["redacted_text"] = tickets[text_column].apply(redact_text)

# Count redactions
original_text = " ".join(tickets[text_column].astype(str))
redacted_text = " ".join(tickets["redacted_text"])

print("\nREDACTION COUNTS")
print("Emails:", len(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", original_text)))
print("Phones:", len(re.findall(r"\b[6-9]\d{9}\b", original_text)))
print("Order IDs:", len(re.findall(r"ORD-\d{6}", original_text)))
print("Remaining emails:", len(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", redacted_text)))
print("Remaining phones:", len(re.findall(r"\b[6-9]\d{9}\b", redacted_text)))
print("Remaining order IDs:", len(re.findall(r"ORD-\d{6}", redacted_text)))

# 5. Prepare customer segment information
segments_path = OUT / "customer_segments.csv"
segments = pd.read_csv(segments_path)

# Get segment name column
segment_name_col = next(
    (c for c in ["segment_name", "segment_label"] if c in segments.columns),
    None
)

if segment_name_col is None:
    segments["segment_name"] = segments["segment_id"].astype(str)
    segment_name_col = "segment_name"

customer_info = customers[["customer_id"]].merge(
    segments[["customer_id", segment_name_col]],
    on="customer_id",
    how="left"
)

customer_info = customer_info.rename(
    columns={segment_name_col: "segment_name"}
)

# Premium customers: top 20% by completed-order spend
spend = (
    orders[orders["status"] == "completed"]
    .groupby("customer_id")["order_amount"]
    .sum()
)

premium_threshold = spend.quantile(0.80)
premium_ids = set(spend[spend >= premium_threshold].index)

# 6. Classification prompts
zero_shot_system = """
CLASSIFY customer support tickets.
Choose exactly one category: billing, delivery, technical, fraud.
Return JSON only in this format:
{"category": "one allowed category", "reason": "short explanation"}
"""

few_shot_system = """
CLASSIFY customer support tickets.
Choose exactly one category: billing, delivery, technical, fraud.
Return JSON only in this format:
{"category": "one allowed category", "reason": "short explanation"}

EXAMPLES:
Ticket: Someone stole my account and made an unauthorized payment.
Answer: {"category": "fraud", "reason": "unauthorized account activity"}

Ticket: My order arrived late and the rider did not call.
Answer: {"category": "delivery", "reason": "late delivery"}
"""

ALLOWED = {"billing", "delivery", "technical", "fraud"}

# 7. Retry LLM call
def call_with_retry(messages, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            response = mock_llm.chat(messages)
            return response
        except mock_llm.TransientLLMError:
            if attempt == max_retries:
                raise
            delay = 0.05 * (2 ** attempt) * random.uniform(0.9, 1.15)
            print(f"Retry {attempt + 1}: temporary LLM error")
            time.sleep(delay)

# 8. Validate LLM output
def parse_response(content):
    try:
        result = json.loads(content)
        category = result.get("category", "").lower()

        if category in ALLOWED:
            return category, "classified_json"

    except (json.JSONDecodeError, AttributeError):
        pass

    words = set(re.findall(r"[a-z]+", str(content).lower()))
    found = words.intersection(ALLOWED)

    if len(found) == 1:
        return found.pop(), "classified_keyword"

    return None, "flagged_invalid_output"

# 9. Classify one ticket
def classify_ticket(text, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    response = call_with_retry(messages)
    category, status = parse_response(
        response.choices[0].message.content
    )

    tokens = response.usage.total_tokens
    return category, status, tokens

# 10. Ticket routing rules
def route_ticket(row):
    category = row["category"]
    status = row["status"]

    if status in [
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    ]:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if row["customer_id"] in premium_ids:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"

# 11. Run a classification batch with token budget
def run_batch(system_prompt, token_budget, output_file):
    mock_llm.reset()
    results = []
    tokens_used = 0

    for _, row in tickets.iterrows():
        text = row["redacted_text"]
        customer_id = row["customer_id"]

        result = {
            "ticket_id": row["ticket_id"],
            "customer_id": customer_id,
            "status": "",
            "category": "",
            "segment_name": "",
            "priority": "",
            "route": "",
            "tokens_used": 0
        }

        customer_row = customer_info[
            customer_info["customer_id"] == customer_id
        ]

        if len(customer_row) > 0:
            result["segment_name"] = customer_row.iloc[0]["segment_name"]

        if not text.strip():
            result["status"] = "skipped_empty"

        elif len(text.split()) * 1.3 > 400:
            result["status"] = "flagged_long"

        elif tokens_used >= token_budget:
            result["status"] = "budget_exhausted"

        else:
            try:
                category, status, used = classify_ticket(
                    text, system_prompt
                )

                if tokens_used + used > token_budget:
                    result["status"] = "budget_exhausted"
                else:
                    tokens_used += used
                    result["status"] = status
                    result["category"] = category or ""

            except mock_llm.TransientLLMError:
                result["status"] = "flagged_invalid_output"

        result["tokens_used"] = tokens_used
        result["route"], result["priority"] = route_ticket(result)
        results.append(result)

    output = pd.DataFrame(results)
    output.to_csv(OUT / output_file, index=False)

    print("\nSaved:", output_file)
    print("Tokens used:", tokens_used)
    print("\nStatus counts:")
    print(output["status"].value_counts())
    print("\nCategory counts:")
    print(output["category"].value_counts())

    return output

# 12. Small-budget demonstration
print("\n===== BUDGET DEMONSTRATION =====")
budget_demo = run_batch(
    few_shot_system,
    token_budget=6000,
    output_file="triage_budget_demo.csv"
)

# 13. Final few-shot classification run
print("\n===== FINAL FEW-SHOT RUN =====")
final_results = run_batch(
    few_shot_system,
    token_budget=100000,
    output_file="triage_results.csv"
)

# 14. Final zero-shot classification run
print("\n===== FINAL ZERO-SHOT RUN =====")
zero_shot_results = run_batch(
    zero_shot_system,
    token_budget=100000,
    output_file="triage_zero_shot_results.csv"
)

print("\nPart C ticket classification completed!")
# 15. Compare zero-shot and few-shot results
zero = pd.read_csv(OUT / "triage_zero_shot_results.csv")
few = pd.read_csv(OUT / "triage_results.csv")

comparison = pd.DataFrame({
    "Method": ["Zero-shot", "Few-shot"],
    "Total Tickets": [len(zero), len(few)],
    "JSON Classified": [
        (zero["status"] == "classified_json").sum(),
        (few["status"] == "classified_json").sum()
    ],
    "Keyword Classified": [
        (zero["status"] == "classified_keyword").sum(),
        (few["status"] == "classified_keyword").sum()
    ],
    "Invalid Outputs": [
        (zero["status"] == "flagged_invalid_output").sum(),
        (few["status"] == "flagged_invalid_output").sum()
    ]
})

comparison.to_csv(OUT / "triage_comparison.csv", index=False)

print("\nZERO-SHOT VS FEW-SHOT COMPARISON")
print(comparison.to_string(index=False))
print("Saved: triage_comparison.csv")