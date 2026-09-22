#modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the raw dataset
df = pd.read_excel('Online Retail.xlsx')

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nQuantity - min/max:", df["Quantity"].min(), df["Quantity"].max())
print("UnitPrice - min/max:", df["UnitPrice"].min(), df["UnitPrice"].max())
print("\nUnique customers:", df["CustomerID"].nunique())
print("Countries:\n", df["Country"].value_counts().head(10))


#cleaning the data
print("Rows before cleaning:", df.shape[0])

#remove rows with no CustomerID - can't segment a customer we can't identify
df = df.dropna(subset=["CustomerID"])

#remove exact duplicate rows
df = df.drop_duplicates()

#remove returns/cancellations (negative quantity) and invalid prices
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

print("Rows after cleaning:", df.shape[0])

#CustomerID is really an ID, not a decimal number - convert to int
df["CustomerID"] = df["CustomerID"].astype(int)

print("\nRemaining unique customers:", df["CustomerID"].nunique())
print("Quantity - min/max:", df["Quantity"].min(), df["Quantity"].max())
print("UnitPrice - min/max:", df["UnitPrice"].min(), df["UnitPrice"].max())


#creating a TotalPrice column per transaction line
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

#reference date: one day after the last transaction in the dataset
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print("Reference date:", reference_date)

#aggregating to one row per customer
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum")
).reset_index()

print(rfm.shape)
print(rfm.head())
print(rfm.describe())

#investigating outliers
print("\nTop 5 customers by Monetary:")
print(rfm.sort_values("Monetary", ascending=False).head())

print("\nTop 5 customers by Frequency:")
print(rfm.sort_values("Frequency", ascending=False).head())

#visualizing distributions to see the skew
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(rfm["Recency"], bins=30, color="#4C72B0")
axes[0].set_title("Recency Distribution")
axes[1].hist(rfm["Frequency"], bins=30, color="#55A868")
axes[1].set_title("Frequency Distribution")
axes[2].hist(rfm["Monetary"], bins=30, color="#C44E52")
axes[2].set_title("Monetary Distribution")
plt.tight_layout()
plt.show()


#log-transforming to reduce skew (add 1 to avoid log(0) issues)
rfm["Recency_log"] = np.log1p(rfm["Recency"])
rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
rfm["Monetary_log"] = np.log1p(rfm["Monetary"])


#visualizing the transformed distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(rfm["Recency_log"], bins=30, color="#4C72B0")
axes[0].set_title("Recency (log) Distribution")
axes[1].hist(rfm["Frequency_log"], bins=30, color="#55A868")
axes[1].set_title("Frequency (log) Distribution")
axes[2].hist(rfm["Monetary_log"], bins=30, color="#C44E52")
axes[2].set_title("Monetary (log) Distribution")
plt.tight_layout()
plt.show()


#selecting the log-transformed features for clustering
features = ["Recency_log", "Frequency_log", "Monetary_log"]
X = rfm[features]

#scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#elbow method + silhouette score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_range = range(2, 11)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil_scores[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(k_range), inertias, "bo-")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")

axes[1].plot(list(k_range), sil_scores, "go-")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score by k")

plt.tight_layout()
plt.show()


#fitting final model with k=4
best_k = 4
final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm["Cluster"] = final_km.fit_predict(X_scaled)

#explicit confirmation of the final chosen model's quality
print(f"\nFinal chosen k={best_k}, Silhouette Score: {silhouette_score(X_scaled, rfm['Cluster']):.4f}")

print("\nCluster sizes:\n", rfm["Cluster"].value_counts().sort_index())
print("\nCluster profiles (original, non-log scale):")
print(rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(1))

#cluster size bar chart
plt.figure(figsize=(6, 4))
rfm["Cluster"].value_counts().sort_index().plot(kind="bar", color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
plt.title("Customer Count per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#boxplots of RFM by cluster - shows spread, not just the mean
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
    sns.boxplot(data=rfm, x="Cluster", y=col, ax=ax)
    ax.set_title(f"{col} by Cluster")
plt.tight_layout()
plt.show()

#visualizing with PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
rfm["pca1"], rfm["pca2"] = X_pca[:, 0], X_pca[:, 1]

print(f"\nExplained variance by 2 PCA components: {pca.explained_variance_ratio_.sum():.2%}")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=rfm, x="pca1", y="pca2", hue="Cluster", palette="tab10", s=40, alpha=0.7)
centers_pca = pca.transform(final_km.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c="black", marker="X", s=200, label="Centroids")
plt.title(f"Customer Segments (k={best_k}) Visualized with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

#mapping clusters to human-readable segment labels
cluster_labels = {
    0: "Champions",
    1: "Lost/Churned",
    2: "New/Recent",
    3: "Regulars"
}
rfm["Segment"] = rfm["Cluster"].map(cluster_labels)

print("\nFinal segment counts:\n", rfm["Segment"].value_counts())

#save the clustered data to a new file in the same directory
rfm.to_csv("rfm_clustered_customers.csv", index=False)
print("\nSaved to rfm_clustered_customers.csv")

