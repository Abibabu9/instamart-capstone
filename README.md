# Instamart Customer Intelligence and Support Co-Pilot

## Overview

This project builds a customer intelligence and support co-pilot for a simulated Instamart dataset. It covers data cleaning, SQL analysis, customer segmentation, lapse prediction, LLM-based ticket triage, policy retrieval, and a ReAct support agent.

## Reproduce

### Requirements
- Python 3.11
- pandas
- numpy
- scikit-learn
- joblib

### Setup

```bash
git clone https://github.com/Abibabu9/instamart-capstone.git
cd instamart-capstone

python -m venv .venv
```

Activate the virtual environment:

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the project scripts from the repository root:

```bash
python clean_data_new.py
python validate_data.py
python run_sql.py
python eda.py
python rfm_analysis.py
python customer_segmentation.py
python lapse_prediction.py
python ticket_triage.py
python policy_rag.py
python agent_workflow.py
```

The scripts use the supplied data and mock LLM implementation. Generated results are saved in the `outputs/` directory.

## Part A - Data Preparation and EDA

- Cleaned customer, product, and order datasets.
- Removed 119 duplicate order rows.
- Prepared 12,309 unique orders.
- Final order status counts: 11,110 completed and 1,199 cancelled.
- Completed order revenue: INR 2,998,668.80.
- Performed seven SQL analyses, including customer spending, city/category revenue, ranking, and daily running revenue.
- Created a SQLite database and validated the tables, IDs, order amounts, and order statuses.
- All nine validation checkpoints passed.
- EDA showed a right-skewed completed order amount distribution, with skewness of 4.61.

## Part B - Customer Intelligence

### RFM Analysis
- Calculated Recency, Frequency, and Monetary values for customers with completed orders.
- Saved the results in `outputs/rfm_analysis.csv`.

### Customer Segmentation
- Applied StandardScaler and K-Means clustering.
- Compared K values from 2 to 8.
- Selected K = 4, which produced the highest silhouette score of 0.526.
- PCA explained 85.2% of the variance using the first two components.
- Saved customer segments in `outputs/customer_segments.csv`.

### Lapse Prediction
- Built Logistic Regression and Random Forest models.
- Used a time-based lapse definition and a stratified train/test split.
- Applied GridSearchCV and training-data oversampling.
- Saved the retention list and model output in the `outputs/` directory.

## Part C - LLM Ticket Triage

- Implemented zero-shot and few-shot ticket classification.
- Used four ticket categories: billing, delivery, technical, and fraud.
- Added JSON response parsing and keyword fallback.
- Added retry handling for invalid responses.
- Redacted email addresses, phone numbers, and order IDs.
- Added token-budget checks and handling for empty or long tickets.
- Processed 164 tickets.
- Few-shot run status counts: 135 JSON classifications, 12 keyword classifications, and 9 invalid-output flags.
- Saved triage results and comparison outputs in `outputs/`.

## Part D - Policy RAG and ReAct Agent

### Policy RAG
- Loaded policy documents and created 15 text chunks.
- Used TF-IDF and cosine similarity for policy retrieval.
- Evaluated retrieval against 14 questions.
- The expected document was retrieved in the top results for 12 of 14 questions (85.7%).
- Saved policy chunks and evaluation results in `outputs/`.

### ReAct Support Agent
- Implemented order lookup, policy search, refund calculation, refund processing, and human escalation tools.
- Tested seven customer scenarios.
- Tested token-limit and timeout handling.
- Tested simulated crash and checkpoint resume.
- Added refund-ledger checking to prevent repeated refund issuance.

## Project Structure

```text
instamart-capstone/
├── data/
│   ├── raw/
│   ├── clean/
│   └── policies/
├── outputs/
├── sql/
├── state/
├── generate_data.py
├── mock_llm.py
├── clean_data_new.py
├── validate_data.py
├── run_sql.py
├── eda.py
├── rfm_analysis.py
├── customer_segmentation.py
├── lapse_prediction.py
├── ticket_triage.py
├── policy_rag.py
├── agent_workflow.py
├── requirements.txt
└── README.md
```

## Limitations and Next Steps

- The project uses simulated customer and order data.
- LLM responses are generated using a mock LLM implementation, not a live production model.
- Policy retrieval can return irrelevant context for some questions.
- Policy answer evaluation needs stronger semantic checks.
- Model performance should be evaluated further before real-world use.
- Future work includes improved retrieval, stronger evaluation, real-world data integration, and connection to a production support system.