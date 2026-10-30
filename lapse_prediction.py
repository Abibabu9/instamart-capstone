import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.base import clone
import joblib

# 1. Load data
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "clean"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

customers = pd.read_csv(DATA / "customers.csv")
orders = pd.read_csv(DATA / "orders.csv")
products = pd.read_csv(DATA / "products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
cutoff = pd.Timestamp("2025-09-01")
end_date = pd.Timestamp("2025-10-01")

# 2. Use only orders before September 1 for model features
before = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] < cutoff)
].copy()

all_before = orders[orders["order_ts"] < cutoff].copy()

# Customers who had at least one completed order before cutoff
cohort_ids = before["customer_id"].unique()

# 3. Create customer features
rfm = before.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean")
).reset_index()

rfm["recency"] = (cutoff - rfm["last_order"]).dt.days
rfm = rfm.drop(columns=["last_order"])

cancel = all_before.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    cancelled_orders=("status", lambda x: (x == "cancelled").sum())
).reset_index()

cancel["cancellation_rate"] = (
    cancel["cancelled_orders"] / cancel["total_orders"]
)

# Favourite product category
category_orders = before.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite = (
    category_orders.groupby(["customer_id", "category"])
    .size()
    .reset_index(name="count")
    .sort_values(["customer_id", "count"], ascending=[True, False])
    .drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

features = (
    customers[customers["customer_id"].isin(cohort_ids)]
    .merge(rfm, on="customer_id", how="left")
    .merge(cancel[["customer_id", "cancellation_rate"]], on="customer_id", how="left")
    .merge(favourite, on="customer_id", how="left")
)

features["signup_date"] = pd.to_datetime(features["signup_date"])
features["tenure_days"] = (cutoff - features["signup_date"]).dt.days
features = features.drop(columns=["signup_date"])

# 4. Create churn label: no completed order during September
september_customers = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] >= cutoff) &
    (orders["order_ts"] < end_date)
]["customer_id"].unique()

features["lapsed"] = (
    ~features["customer_id"].isin(september_customers)
).astype(int)

print("Cohort:", len(features))
print("Lapsed:", int(features["lapsed"].sum()))
print("Lapsed percentage:", round(features["lapsed"].mean() * 100, 2), "%")

# 5. Select features and target
numeric_features = [
    "age", "recency", "frequency", "monetary",
    "avg_discount", "cancellation_rate",
    "avg_delivery", "tenure_days"
]
categorical_features = [
    "city", "gender", "favourite_category"
]

X = features[numeric_features + categorical_features]
y = features["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Train rows:", len(X_train))
print("Test rows:", len(X_test))

# 6. Baseline: predict every customer as not lapsed
baseline = np.zeros(len(y_test), dtype=int)
print("\nBASELINE")
print("Accuracy:", round(accuracy_score(y_test, baseline), 3))
print("Lapsed F1:", round(f1_score(y_test, baseline, zero_division=0), 3))

# 7. Preprocessing: fit only on training data
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# 8. Logistic Regression with GridSearchCV
lr_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", LogisticRegression(max_iter=1000))
])

lr_grid = GridSearchCV(
    lr_pipeline,
    {"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
    n_jobs=-1
)
lr_grid.fit(X_train, y_train)

lr_model = lr_grid.best_estimator_
lr_predictions = lr_model.predict(X_test)

print("\nLOGISTIC REGRESSION")
print("Best parameters:", lr_grid.best_params_)
print("Confusion matrix:\n", confusion_matrix(y_test, lr_predictions))
print("Precision:", round(precision_score(y_test, lr_predictions, zero_division=0), 3))
print("Recall:", round(recall_score(y_test, lr_predictions, zero_division=0), 3))
print("F1:", round(f1_score(y_test, lr_predictions, zero_division=0), 3))

# 9. Random Forest with GridSearchCV
rf_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=100, random_state=42
    ))
])

rf_grid = GridSearchCV(
    rf_pipeline,
    {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)

rf_model = rf_grid.best_estimator_
rf_predictions = rf_model.predict(X_test)

print("\nRANDOM FOREST")
print("Best parameters:", rf_grid.best_params_)
print("Confusion matrix:\n", confusion_matrix(y_test, rf_predictions))
print("Precision:", round(precision_score(y_test, rf_predictions, zero_division=0), 3))
print("Recall:", round(recall_score(y_test, rf_predictions, zero_division=0), 3))
print("F1:", round(f1_score(y_test, rf_predictions, zero_division=0), 3))

# 10. Oversampling: duplicate lapsed training rows only
train_data = X_train.copy()
train_data["target"] = y_train

majority = train_data[train_data["target"] == 0]
minority = train_data[train_data["target"] == 1]

minority_oversampled = minority.sample(
    n=len(majority), replace=True, random_state=42
)

balanced = pd.concat([majority, minority_oversampled])
X_balanced = balanced.drop(columns="target")
y_balanced = balanced["target"]

print("\nTraining class counts after oversampling:")
print(y_balanced.value_counts())

oversampled_model = clone(rf_model)
oversampled_model.fit(X_balanced, y_balanced)
oversampled_predictions = oversampled_model.predict(X_test)

print("\nOVERSAMPLED RANDOM FOREST")
print("Precision:", round(precision_score(y_test, oversampled_predictions, zero_division=0), 3))
print("Recall:", round(recall_score(y_test, oversampled_predictions, zero_division=0), 3))
print("F1:", round(f1_score(y_test, oversampled_predictions, zero_division=0), 3))

# 11. Show top 5 Random Forest feature importances
feature_names = (
    oversampled_model.named_steps["preprocessor"].get_feature_names_out()
)
importances = oversampled_model.named_steps["model"].feature_importances_

importance_table = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTOP 5 RANDOM FOREST FEATURES")
print(importance_table.head(5).to_string(index=False))

# 12. Show Logistic Regression coefficients
lr_feature_names = lr_model.named_steps["preprocessor"].get_feature_names_out()
lr_coefficients = lr_model.named_steps["model"].coef_[0]

coefficient_table = pd.DataFrame({
    "feature": lr_feature_names,
    "coefficient": lr_coefficients
}).sort_values("coefficient", ascending=False)

print("\nTOP LOGISTIC REGRESSION COEFFICIENTS")
print(coefficient_table.head(5).to_string(index=False))

# 13. Save and reload model; check predictions
model_path = OUT / "lapse_model.joblib"
joblib.dump(oversampled_model, model_path)

loaded_model = joblib.load(model_path)
reloaded_predictions = loaded_model.predict(X_test)

print("\nMODEL RELOAD CHECK:")
print("Predictions identical:", np.array_equal(
    oversampled_predictions, reloaded_predictions
))

# 14. Create retention list: top 10 Premium customers in test set
# Premium is a descriptive label based on total completed spend;
# it is not used as a model feature.
whole_window_spend = (
    orders[orders["status"] == "completed"]
    .groupby("customer_id")["order_amount"]
    .sum()
)

premium_threshold = whole_window_spend.quantile(0.80)
premium_ids = whole_window_spend[
    whole_window_spend >= premium_threshold
].index

retention = X_test.copy()
retention["customer_id"] = features.loc[X_test.index, "customer_id"]
retention["actual_lapsed"] = y_test
retention["lapse_probability"] = loaded_model.predict_proba(X_test)[:, 1]
retention["premium"] = retention["customer_id"].isin(premium_ids)

retention_list = (
    retention[retention["premium"]]
    .sort_values("lapse_probability", ascending=False)
    .head(10)
    [["customer_id", "lapse_probability", "actual_lapsed"]]
)

retention_list.to_csv(OUT / "retention_list.csv", index=False)

print("\nRETENTION LIST")
print(retention_list.to_string(index=False))

print("\nPart B lapse prediction completed!")