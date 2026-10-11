
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)

warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs("results", exist_ok=True)

# ==================================================
# TASK 1: BOOSTING - ADABOOST REGRESSION
# ==================================================

print("\n" + "=" * 50)
print("TASK 1: ADABOOST REGRESSION")
print("=" * 50)

insurance = pd.read_csv("Datasets/insurance_pre.csv")

X = insurance.drop(columns=["charges"])
y = insurance["charges"]

categorical = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numerical = X.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical),
    ("numerical", "passthrough", numerical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

boost_model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=100,
        random_state=42
    ))
])

boost_model.fit(X_train, y_train)
boost_predictions = boost_model.predict(X_test)

boost_mae = mean_absolute_error(y_test, boost_predictions)
boost_rmse = np.sqrt(mean_squared_error(y_test, boost_predictions))
boost_r2 = r2_score(y_test, boost_predictions)

print(f"MAE: {boost_mae:.2f}")
print(f"RMSE: {boost_rmse:.2f}")
print(f"R2 Score: {boost_r2:.4f}")

pd.DataFrame({
    "Actual Charges": y_test.to_numpy(),
    "Predicted Charges": boost_predictions
}).to_csv("insurance_predictions.csv", index=False)

# ==================================================
# TASK 2: CLASSIFICATION - LOGISTIC REGRESSION
# ==================================================

print("\n" + "=" * 50)
print("TASK 2: CLASSIFICATION")
print("=" * 50)

ads = pd.read_csv("Datasets/Social_Network_Ads.csv")

X = ads[["Age", "EstimatedSalary"]]
y = ads["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42))
])

class_model.fit(X_train, y_train)
class_predictions = class_model.predict(X_test)

class_accuracy = accuracy_score(y_test, class_predictions)

print(f"Accuracy: {class_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, class_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, class_predictions))

classification_results = X_test.copy()
classification_results["Actual Purchased"] = y_test.to_numpy()
classification_results["Predicted Purchased"] = class_predictions

classification_results.to_csv(
    "classification_predictions.csv", index=False
)

# ==================================================
# TASK 3: CLUSTERING - K-MEANS
# ==================================================

print("\n" + "=" * 50)
print("TASK 3: K-MEANS CLUSTERING")
print("=" * 50)

customers = pd.read_csv("Datasets/Mall_Customers.csv")

features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]
X = customers[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertias = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias, marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_plot.png", dpi=150)
plt.close()

# Use five clusters for comparison with the earlier run
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

customers["Cluster"] = clusters
silhouette = silhouette_score(X_scaled, clusters)

print("Number of clusters:", len(np.unique(clusters)))
print(f"Silhouette Score: {silhouette:.4f}")
print("\nCustomers in each cluster:")
print(customers["Cluster"].value_counts().sort_index())

customers.to_csv("customer_clusters.csv", index=False)

# Plot customer groups
plt.figure(figsize=(8, 5))

for cluster in sorted(customers["Cluster"].unique()):
    group = customers[customers["Cluster"] == cluster]
    plt.scatter(
        group["Annual Income (k$)"],
        group["Spending Score (1-100)"],
        label=f"Cluster {cluster}"
    )

plt.title("Mall Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.tight_layout()
plt.savefig("customer_clusters.png", dpi=150)
plt.close()

# ==================================================
# SAVE SUMMARY OF ALL THREE ALGORITHMS
# ==================================================

summary = pd.DataFrame([
    {
        "Algorithm": "AdaBoost Regression",
        "Metric": "MAE",
        "Value": round(boost_mae, 4)
    },
    {
        "Algorithm": "AdaBoost Regression",
        "Metric": "RMSE",
        "Value": round(boost_rmse, 4)
    },
    {
        "Algorithm": "AdaBoost Regression",
        "Metric": "R2 Score",
        "Value": round(boost_r2, 4)
    },
    {
        "Algorithm": "Logistic Regression",
        "Metric": "Accuracy",
        "Value": round(class_accuracy, 4)
    },
    {
        "Algorithm": "K-Means",
        "Metric": "Silhouette Score",
        "Value": round(silhouette, 4)
    }
])

summary.to_csv("results/model_results.csv", index=False)

print("\n" + "=" * 50)
print("ALL THREE TASKS COMPLETED SUCCESSFULLY")
print("Prediction files, graphs, and model_results.csv saved.")
print("=" * 50)
