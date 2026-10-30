
import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent
data_path = base_path / "data" / "clean"

customers = pd.read_csv(data_path / "customers.csv")
orders = pd.read_csv(data_path / "orders.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

completed = orders[orders["status"] == "completed"].copy()

# Analysis date
snapshot_date = pd.Timestamp("2025-10-01")

# RFM calculation
rfm = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum")
).reset_index()

rfm["recency"] = (
    snapshot_date - rfm["last_order"]
).dt.days

rfm = rfm[[
    "customer_id", "recency", "frequency", "monetary"
]]

print("\n--- RFM SUMMARY ---")
print(rfm[["recency", "frequency", "monetary"]].describe())

print("\nTotal RFM customers:", len(rfm))

# Save RFM output
output_dir = base_path / "outputs"
output_dir.mkdir(exist_ok=True)

rfm.to_csv(output_dir / "rfm_analysis.csv", index=False)

print("\nRFM analysis saved successfully!")
