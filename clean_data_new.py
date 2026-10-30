
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
CLEAN_DIR = PROJECT_DIR / "data" / "clean"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load customer data
df = pd.read_csv(RAW_DIR / "customers.csv")

# 1. Clean city names
df["city"] = df["city"].astype("string").str.strip().str.title()

# 2. Clean gender spellings
df["gender"] = (
    df["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
)

gender_map = {
    "m": "male",
    "male": "male",
    "man": "male",
    "f": "female",
    "female": "female",
    "woman": "female",
}

df["gender"] = df["gender"].map(gender_map).fillna("unknown")

# 3. Convert invalid ages to missing
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df.loc[~df["age"].between(10, 100), "age"] = pd.NA

# 4. Fill missing ages using each city's median
city_median = df.groupby("city")["age"].transform("median")
df["age"] = df["age"].fillna(city_median)

# 5. Convert signup date to datetime
df["signup_date"] = pd.to_datetime(
    df["signup_date"], errors="coerce"
)

# Save cleaned customers
output_path = CLEAN_DIR / "customers.csv"
df.to_csv(output_path, index=False)

print("Customer cleaning completed!")
print("Rows:", len(df))
print("Missing ages:", df["age"].isna().sum())
print("Missing genders:", df["gender"].isna().sum())
print("Cities:", sorted(df["city"].dropna().unique()))
print("Gender values:", sorted(df["gender"].unique()))
print("Saved to:", output_path)

# Load products data
products = pd.read_csv(RAW_DIR / "products.csv")

print("\nPRODUCTS AUDIT")
print("Rows:", len(products))
print("Columns:", products.columns.tolist())
print("Missing values:")
print(products.isna().sum())

print("\nProduct sample:")
print(products.head())

# Save products data
products["category"] = (
    products["category"].astype("string").str.strip().str.title()
)

products["product_name"] = (
    products["product_name"].astype("string").str.strip()
)

products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

products.to_csv(CLEAN_DIR / "products.csv", index=False)

print("Products cleaning completed!")
print("Products saved:", len(products))

# Load raw orders
orders = pd.read_csv(RAW_DIR / "orders.csv")

print("\nORDERS CLEANING STARTED")
print("Original rows:", len(orders))
print("Columns:", orders.columns.tolist())
print(orders.head())

# 1. Check and clean column names
orders.columns = orders.columns.str.strip()

# 2. Convert order timestamp to datetime
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# 3. Remove exact duplicate rows
before = len(orders)
orders = orders.drop_duplicates().copy()

print("Exact duplicate rows removed:", before - len(orders))
print("Rows after duplicate removal:", len(orders))
print("Status column exists:", "status" in orders.columns)

# Keep the latest record for each order_id
orders = orders.sort_values("order_ts")

orders = orders.drop_duplicates(
    subset="order_id",
    keep="last"
).copy()

print("Rows after order_id cleaning:", len(orders))
print("Unique order IDs:", orders["order_id"].nunique())
print("Status counts:")
print(orders["status"].value_counts())

# 1. Convert order_amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# 2. Calculate missing order amounts using product price
orders = orders.merge(
    products[["product_id", "price_inr"]],
    on="product_id",
    how="left"
)

orders["order_amount"] = orders["order_amount"].fillna(
    (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)
)

# 3. Check the result
print("Missing order amounts:", orders["order_amount"].isna().sum())
print("Total orders:", len(orders))

# 1. Convert delivery_minutes to numeric
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# 2. Make negative delivery times missing
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

# 3. Calculate median delivery time by product category
orders["category"] = orders["product_id"].map(
    products.set_index("product_id")["category"]
)

completed_mask = orders["status"] == "completed"

category_median = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(category_median)
)

# 4. Keep delivery time missing for cancelled orders
orders.loc[
    orders["status"] == "cancelled",
    "delivery_minutes"
] = pd.NA

# 5. Check the results
print("Missing order amounts:", orders["order_amount"].isna().sum())
print(
    "Completed orders missing delivery time:",
    orders.loc[completed_mask, "delivery_minutes"].isna().sum()
)
print(
    "Cancelled orders missing delivery time:",
    orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().sum()
)

# Remove the temporary category column
orders = orders.drop(columns=["category"])

# Save cleaned orders
orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

# Save cleaned customers and products again
df.to_csv(CLEAN_DIR / "customers.csv", index=False)
products.to_csv(CLEAN_DIR / "products.csv", index=False)

print("\nALL CLEANED CSV FILES SAVED!")
print("Customers:", len(df))
print("Products:", len(products))
print("Orders:", len(orders))
print("Saved folder:", CLEAN_DIR)

import sqlite3

# Create database
DB_PATH = PROJECT_DIR / "data" / "instamart.db"

conn = sqlite3.connect(DB_PATH)

# Store cleaned CSV data as database tables
df.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)

conn.close()

print("\nDATABASE CREATED SUCCESSFULLY!")
print("Database:", DB_PATH)
print("Tables: customers, products, orders")
