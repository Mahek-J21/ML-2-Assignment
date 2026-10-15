
# ============================================================
# CUSTOMER SEGMENTATION USING CLUSTERING
# Dataset: Mall_Customers.csv
# Algorithms: K-Means, Hierarchical, DBSCAN
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)
from sklearn.metrics import silhouette_score

os.makedirs("results", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("Datasets/Mall_Customers.csv")
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# 2. Select features
feature_cols = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[feature_cols].copy()

# Handle missing values
X = X.fillna(X.median())

# 3. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Elbow and silhouette analysis for K-Means
inertias = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )

# Elbow graph
plt.figure(figsize=(7, 4))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/clustering_elbow.png",
    dpi=150
)

plt.show()
plt.close()

# Silhouette graph
plt.figure(figsize=(7, 4))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by K")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/clustering_silhouette_by_k.png",
    dpi=150
)

plt.show()
plt.close()

# Choose K with the highest silhouette score
best_k = list(k_values)[
    int(np.argmax(silhouette_scores))
]

print("\nBest K:", best_k)
print(
    "Best K-Means silhouette score:",
    round(max(silhouette_scores), 4)
)

# 5. Compare three clustering algorithms
labels_by_model = {}

# K-Means
km = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

labels_by_model["K-Means"] = km.fit_predict(X_scaled)

# Hierarchical / Agglomerative clustering
agg = AgglomerativeClustering(
    n_clusters=best_k
)

labels_by_model["Hierarchical"] = agg.fit_predict(
    X_scaled
)

# DBSCAN
db = DBSCAN(
    eps=0.5,
    min_samples=5
)

labels_by_model["DBSCAN"] = db.fit_predict(
    X_scaled
)

# 6. Evaluate clustering algorithms
results = []

for name, labels in labels_by_model.items():

    # Exclude DBSCAN noise label -1 from cluster count
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    # Silhouette requires at least 2 clusters
    # and fewer clusters than samples
    valid_labels = labels != -1

    if (
        n_clusters >= 2
        and np.sum(valid_labels) > n_clusters
    ):
        score = silhouette_score(
            X_scaled[valid_labels],
            labels[valid_labels]
        )
    else:
        score = np.nan

    results.append({
        "Algorithm": name,
        "Clusters (excluding noise)": n_clusters,
        "Noise Points": n_noise,
        "Silhouette Score": score
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Silhouette Score",
    ascending=False,
    na_position="last"
)

print("\n========== CLUSTERING COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "results/clustering_comparison.csv",
    index=False
)

# 7. Silhouette score comparison graph
comparison = results_df.dropna(
    subset=["Silhouette Score"]
)

if not comparison.empty:

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        comparison["Algorithm"],
        comparison["Silhouette Score"]
    )

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )

    plt.xlabel("Clustering Algorithm")
    plt.ylabel("Silhouette Score")
    plt.title("Comparison of Clustering Algorithms")
    plt.tight_layout()

    plt.savefig(
        "results/clustering_algorithm_comparison.png",
        dpi=150
    )

    plt.show()
    plt.close()

# 8. Choose the best algorithm with a valid score
if comparison.empty:
    raise ValueError(
        "No algorithm produced a valid silhouette score. "
        "Try adjusting the DBSCAN parameters."
    )

best_algorithm = comparison.iloc[0]["Algorithm"]
best_labels = labels_by_model[best_algorithm]

print("\n========== BEST CLUSTERING MODEL ==========")
print("Selected algorithm:", best_algorithm)

# Add cluster labels to the dataset
df["Cluster"] = best_labels

print("\nCustomers in each cluster (-1 means noise):")
print(df["Cluster"].value_counts().sort_index())

df.to_csv(
    "results/customer_clusters_comparison.csv",
    index=False
)

# 9. Visualize all three algorithms
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 5)
)

for ax, (name, labels) in zip(
    axes,
    labels_by_model.items()
):

    for cluster in sorted(set(labels)):

        mask = labels == cluster

        label = (
            "Noise"
            if cluster == -1
            else f"Cluster {cluster}"
        )

        ax.scatter(
            X.loc[mask, feature_cols[0]],
            X.loc[mask, feature_cols[1]],
            label=label,
            s=35
        )

    ax.set_title(name)
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend(fontsize=7)

plt.suptitle("Comparison of Clustering Algorithms")
plt.tight_layout()

plt.savefig(
    "results/clustering_all_algorithms.png",
    dpi=150
)

plt.show()
plt.close()

# 10. Plot best algorithm separately
plt.figure(figsize=(8, 5))

for cluster in sorted(set(best_labels)):

    mask = best_labels == cluster

    label = (
        "Noise"
        if cluster == -1
        else f"Cluster {cluster}"
    )

    plt.scatter(
        X.loc[mask, feature_cols[0]],
        X.loc[mask, feature_cols[1]],
        label=label,
        s=45
    )

# Show X-shaped centres only for K-Means
if best_algorithm == "K-Means":

    centres = scaler.inverse_transform(
        km.cluster_centers_
    )

    plt.scatter(
        centres[:, 0],
        centres[:, 1],
        marker="X",
        s=220,
        c="black",
        edgecolors="white",
        linewidths=1.5,
        label="Cluster Centres"
    )

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"Customer Segmentation - {best_algorithm}")
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/clustering_best_model.png",
    dpi=150
)

plt.show()
plt.close()

# 11. Summarize customer characteristics
print("\nAVERAGE CUSTOMER CHARACTERISTICS")

print(
    df[df["Cluster"] != -1]
    .groupby("Cluster")[feature_cols]
    .mean()
    .round(2)
)

print("\nClustering analysis completed!")
print("All results saved in the results folder.")
