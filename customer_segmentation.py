
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

base_path = Path(__file__).resolve().parent
rfm = pd.read_csv(base_path / "outputs" / "rfm_analysis.csv")

features = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
X = scaler.fit_transform(rfm[features])

print("\n--- K VALUE COMPARISON ---")

for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)

    print(
        "K:", k,
        "| Inertia:", round(model.inertia_, 1),
        "| Silhouette:", round(silhouette_score(X, labels), 3)
    )

# Final model: K=4
final_model = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["segment_id"] = final_model.fit_predict(X)

# Segment summary
summary = rfm.groupby("segment_id").agg(
    customers=("customer_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean")
).round(2)

print("\n--- SEGMENT SUMMARY ---")
print(summary)

# Segment names based on business interpretation
rfm["segment_name"] = rfm["segment_id"].map({
    0: "Regular Customers",
    1: "At Risk Customers",
    2: "High Value Customers",
    3: "New or Occasional Customers"
})

rfm["is_premium"] = (
    rfm["segment_name"] == "High Value Customers"
)

output_dir = base_path / "outputs"
output_dir.mkdir(exist_ok=True)

rfm.to_csv(output_dir / "customer_segments.csv", index=False)

print("\nCustomer segmentation saved successfully!")

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

print("\n--- PCA ANALYSIS ---")
print("PC1 explained variance:", round(pca.explained_variance_ratio_[0], 3))
print("PC2 explained variance:", round(pca.explained_variance_ratio_[1], 3))
print("Total explained variance:", round(pca.explained_variance_ratio_.sum(), 3))
