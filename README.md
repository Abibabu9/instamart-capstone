# Instamart Customer Intelligence and Support Co-Pilot

## Overview
This project analyses Instamart customer data, predicts customer lapse, classifies support tickets, and builds a policy-based support agent.

## Part A – Data Preparation and EDA
- Cleaned customer, product, and order datasets.
- Removed 119 duplicate rows.
- Prepared 12,309 unique orders.
- Performed SQL analysis and EDA.
- Created and validated a SQLite database.

## Part B – Customer Intelligence
- Performed RFM analysis.
- Applied K-Means clustering with K = 4.
- Built Logistic Regression and Random Forest lapse prediction models.

## Part C – LLM Ticket Triage
- Implemented zero-shot and few-shot ticket classification.
- Added JSON parsing, retry handling, PII redaction, and token-budget checks.

## Part D – Policy RAG and ReAct Agent
- Built policy retrieval using TF-IDF and cosine similarity.
- Created an agent with order lookup, refund calculation, refund processing, and human escalation tools.
- Tested checkpoint resume, timeout, and token limits.

## Project Structure
- `data/` – Raw, cleaned data, and policies
- `sql/` – SQL queries
- `outputs/` – Analysis and evaluation results
- `state/` – Agent checkpoints
- Python files – Project implementation

## Limitations and Next Steps
- Uses simulated data and mock LLM responses.
- Policy retrieval and model performance can be improved.
- Future work includes real-world data and support-system integration.
