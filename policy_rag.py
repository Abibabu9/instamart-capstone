
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Project paths
BASE = Path(__file__).resolve().parent
POLICY_DIR = BASE / "data" / "policies"
DATA = BASE / "data" / "raw"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# 2. Read policy documents and split into chunks
chunks = []

for file_path in sorted(POLICY_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")
    words = text.split()

    start = 0
    while start < len(words):
        end = min(start + 40, len(words))

        chunk_text = " ".join(words[start:end])

        chunks.append({
            "source": file_path.name,
            "text": chunk_text
        })

        if end == len(words):
            break

        start += 30  # 40 words with 10 words overlap

print("Total policy chunks:", len(chunks))

# 3. Create TF-IDF vectors
chunk_texts = [chunk["text"] for chunk in chunks]

vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(chunk_texts)

# 4. Retrieval function
def retrieve(query, k=3):
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    top_indices = scores.argsort()[::-1][:k]

    results = []

    for index in top_indices:
        results.append({
            "source": chunks[index]["source"],
            "score": round(float(scores[index]), 4),
            "text": chunks[index]["text"]
        })

    return results

# 5. Test retrieval
test_query = "refund policy for late delivery"

results = retrieve(test_query, k=3)

print("\nRETRIEVAL TEST")
for item in results:
    print("\nSource:", item["source"])
    print("Score:", item["score"])
    print("Text:", item["text"])

# 6. Save chunks for inspection
pd.DataFrame(chunks).to_csv(
    OUT / "policy_chunks.csv",
    index=False
)

print("\nPolicy RAG retrieval setup completed!")

# 7. LLM-based policy answer function
import mock_llm

def answer(question):
    retrieved = retrieve(question, k=3)

    context_parts = []

    for item in retrieved:
        context_parts.append(
            f"[{item['source']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer using only the supplied "
                "policy context. If the answer is not available, "
                "say: I could not find that information in the policy documents."
            )
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        }
    ]

    response = mock_llm.chat(messages)
    answer_text = response.choices[0].message.content

    sources = list(dict.fromkeys(
        item["source"] for item in retrieved
    ))

    return {
        "answer": answer_text,
        "sources": sources
    }


# 8. Test policy question answering
print("\n===== POLICY QUESTION ANSWERING =====")

questions = [
    "What is the refund policy?",
    "What happens when delivery is late?",
    "How can I escalate a customer complaint?"
]

for question in questions:
    result = answer(question)

    print("\nQuestion:", question)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])


# 9. Evaluate using policy evaluation dataset
evaluation_file = DATA / "policy_eval.csv"

if evaluation_file.exists():
    evaluation = pd.read_csv(
        evaluation_file,
        keep_default_na=False
    )

    print("\n===== POLICY EVALUATION =====")
    print("Evaluation questions:", len(evaluation))
    print("Evaluation columns:", evaluation.columns.tolist())

print("\nPolicy question answering completed!")

# 10. Full policy evaluation
evaluation = pd.read_csv(
    DATA / "policy_eval.csv",
    keep_default_na=False
)

print("\n===== FULL POLICY EVALUATION =====")
print("Columns:", evaluation.columns.tolist())

evaluation_results = []

for _, row in evaluation.iterrows():
    question = row["question"]
    result = answer(question)

    evaluation_results.append({
        "question": question,
        "answer": result["answer"],
        "sources": ", ".join(result["sources"])
    })

evaluation_output = pd.DataFrame(evaluation_results)
evaluation_output.to_csv(
    OUT / "policy_evaluation_results.csv",
    index=False
)

print("Questions evaluated:", len(evaluation_output))
print("Saved: policy_evaluation_results.csv")

# 11. Evaluate retrieval and answer quality

evaluation = pd.read_csv(
    DATA / "policy_eval.csv",
    keep_default_na=False
)

evaluation_rows = []

for _, row in evaluation.iterrows():
    question = row["question"]
    expected_doc = row["expected_doc"]
    expected_fact = row["expected_fact"]

    retrieved = retrieve(question, k=3)
    result = answer(question)

    top_docs = [item["source"] for item in retrieved]

    expected_doc_found = any(
        expected_doc.lower().replace(" ", "_")
        in doc.lower().replace(" ", "_")
        for doc in top_docs
    )

    expected_fact_found = (
        expected_fact in result["answer"]
        if expected_fact else True
    )

    unanswerable_correct = (
        "I could not find" in result["answer"]
        if not expected_fact else True
    )

    evaluation_rows.append({
        "question": question,
        "expected_doc": expected_doc,
        "retrieved_docs": ", ".join(top_docs),
        "expected_doc_found": expected_doc_found,
        "expected_fact_found": expected_fact_found,
        "answer": result["answer"],
        "unanswerable_correct": unanswerable_correct
    })

evaluation_df = pd.DataFrame(evaluation_rows)

evaluation_df.to_csv(
    OUT / "policy_evaluation_results.csv",
    index=False
)

print("\nPOLICY EVALUATION RESULTS")
print("Total questions:", len(evaluation_df))
print(
    "Expected document found:",
    evaluation_df["expected_doc_found"].sum(),
    "/", len(evaluation_df)
)
print(
    "Expected fact found:",
    evaluation_df["expected_fact_found"].sum(),
    "/", len(evaluation_df)
)
print(
    "Unanswerable fallback correct:",
    evaluation_df["unanswerable_correct"].sum(),
    "/", len(evaluation_df)
)

# 12. Zero-overlap comparison
zero_chunks = []

for file_path in sorted(POLICY_DIR.glob("*.txt")):
    words = file_path.read_text(encoding="utf-8").split()

    for start in range(0, len(words), 40):
        chunk_text = " ".join(words[start:start + 40])

        if chunk_text:
            zero_chunks.append({
                "source": file_path.name,
                "text": chunk_text
            })

zero_vectorizer = TfidfVectorizer()
zero_vectors = zero_vectorizer.fit_transform(
    [item["text"] for item in zero_chunks]
)

def retrieve_zero_overlap(query, k=3):
    query_vector = zero_vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector, zero_vectors
    ).flatten()

    indices = scores.argsort()[::-1][:k]

    return [zero_chunks[i] for i in indices]

zero_doc_hits = 0

for _, row in evaluation.iterrows():
    zero_results = retrieve_zero_overlap(row["question"], k=3)

    if any(
        row["expected_doc"].lower().replace(" ", "_")
        in item["source"].lower().replace(" ", "_")
        for item in zero_results
    ):
        zero_doc_hits += 1

print("\nZERO OVERLAP COMPARISON")
print("Expected document found:", zero_doc_hits, "/", len(evaluation))

print("\nPolicy evaluation completed!")
