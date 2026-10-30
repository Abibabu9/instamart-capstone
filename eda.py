
import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent
data_path = base_path / "data" / "clean"

customers = pd.read_csv(data_path / "customers.csv")
products = pd.read_csv(data_path / "products.csv")
orders = pd.read_csv(data_path / "orders.csv")

print("\n--- DATASET SHAPES ---")
print("Customers:", customers.shape)
print("Products:", products.shape)
print("Orders:", orders.shape)

print("\n--- ORDER STATUS ---")
print(orders["status"].value_counts())

completed = orders[orders["status"] == "completed"].copy()

print("\n--- COMPLETED ORDER AMOUNT ---")
print("Mean:", round(completed["order_amount"].mean(), 2))
print("Median:", round(completed["order_amount"].median(), 2))
print("Minimum:", round(completed["order_amount"].min(), 2))
print("Maximum:", round(completed["order_amount"].max(), 2))

print("\n--- ORDER AMOUNT STATISTICS ---")
print(completed["order_amount"].describe())

print("\n--- CITY-WISE CUSTOMERS ---")
print(customers["city"].value_counts())

print("\n--- CATEGORY-WISE REVENUE ---")
category_revenue = completed.merge(
    products[["product_id", "category"]],
    on="product_id"
).groupby("category")["order_amount"].sum()

print(category_revenue.sort_values(ascending=False).round(2))

print("\n--- MISSING VALUES ---")
print("Customers:", customers.isnull().sum().sum())
print("Products:", products.isnull().sum().sum())
print("Orders:", orders.isnull().sum().sum())
print("\n--- MISSING VALUES BY ORDER COLUMN ---")
print(orders.isnull().sum())

print("\n--- OUTLIER ANALYSIS ---")

Q1 = completed["order_amount"].quantile(0.25)
Q3 = completed["order_amount"].quantile(0.75)

IQR = Q3 - Q1
upper_limit = Q3 + (1.5 * IQR)
lower_limit = Q1 - (1.5 * IQR)

outliers = completed[
    (completed["order_amount"] < lower_limit) |
    (completed["order_amount"] > upper_limit)
]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)
print("Outlier count:", len(outliers))

print("\n--- SKEWNESS ANALYSIS ---")

skew_value = completed["order_amount"].skew()

print("Order Amount Skewness:", round(skew_value, 2))

if skew_value > 0:
    print("Distribution is Right Skewed")
elif skew_value < 0:
    print("Distribution is Left Skewed")
else:
    print("Distribution is Symmetrical")
print("\n--- CORRELATION ANALYSIS ---")

correlation = completed[
    ["order_amount", "discount_pct", "quantity"]
].corr()

print(correlation)

print(
    "Order amount vs discount:",
    round(correlation.loc["order_amount", "discount_pct"], 3)
)

output_dir = base_path / "outputs"
output_dir.mkdir(exist_ok=True)

with open(output_dir / "eda_summary.txt", "w") as file:
    file.write("INSTAMART EDA SUMMARY\n")
    file.write("=====================\n")
    file.write(f"Customers: {len(customers)}\n")
    file.write(f"Products: {len(products)}\n")
    file.write(f"Orders: {len(orders)}\n")
    file.write(f"Completed orders: {len(completed)}\n")
    file.write(f"Cancelled orders: {(orders['status'] == 'cancelled').sum()}\n")
    file.write(f"Mean order amount: {completed['order_amount'].mean():.2f}\n")
    file.write(f"Median order amount: {completed['order_amount'].median():.2f}\n")
    file.write(f"Outlier count: {len(outliers)}\n")
    file.write(f"Skewness: {skew_value:.2f}\n")
    file.write(f"Order amount vs discount correlation: {correlation.loc['order_amount', 'discount_pct']:.3f}\n")

print("EDA summary saved successfully!")
