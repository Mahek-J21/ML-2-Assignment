
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    silhouette_score
)


# --------------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

model_results = []


# ==================================================
# 1. ADABOOST REGRESSION - INSURANCE CHARGES
# ==================================================

print("\n" + "=" * 50)
print("1. ADABOOST REGRESSION")
print("=" * 50)

insurance = pd.read_csv("Datasets/insurance_pre.csv")

# Remove rows with missing values, if any
insurance = insurance.dropna()

# Target column
target_column = "charges"

if target_column not in insurance.columns:
    raise ValueError(
        f"'charges' column not found. Available columns: "
        f"{list(insurance.columns)}"
    )

X = insurance.drop(columns=[target_column])
y = insurance[target_column]

# Identify numerical and categorical columns
numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Preprocess categorical data
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        ("numerical", "passthrough", numeric_columns)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

adaboost = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=4),
    n_estimators=100,
    random_state=42
)

boost_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", adaboost)
    ]
)

boost_model.fit(X_train, y_train)

boost_predictions = boost_model.predict(X_test)

mae = mean_absolute_error(y_test, boost_predictions)
rmse = mean_squared_error(
    y_test, boost_predictions
) ** 0.5
r2 = r2_score(y_test, boost_predictions)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

model_results.append({
    "Algorithm": "AdaBoost Regression",
    "Metric": "R2 Score",
    "Value": round(r2, 4)
})

# Save predictions
boost_output = pd.DataFrame({
    "Actual Charges": y_test.to_numpy(),
    "Predicted Charges": boost_predictions
})

boost_output.to_csv(
    "results/insurance_predictions.csv",
    index=False
)

# GRAPH 1: Actual vs Predicted
plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    boost_predictions,
    alpha=0.7
)

low = min(y_test.min(), boost_predictions.min())
high = max(y_test.max(), boost_predictions.max())

plt.plot(
    [low, high],
    [low, high],
    "r--",
    label="Ideal Prediction"
)

plt.xlabel("Actual Insurance Charges")
plt.ylabel("Predicted Insurance Charges")
plt.title("AdaBoost: Actual vs Predicted")
plt.legend()
plt.tight_layout()

plt.savefig(
    "insurance_actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ==================================================
# 2. LOGISTIC REGRESSION - PURCHASE CLASSIFICATION
# ==================================================

print("\n" + "=" * 50)
print("2. LOGISTIC REGRESSION CLASSIFICATION")
print("=" * 50)

ads = pd.read_csv("Datasets/Social_Network_Ads.csv")

required_columns = [
    "Age",
    "EstimatedSalary",
    "Purchased"
]

for column in required_columns:
    if column not in ads.columns:
        raise ValueError(
            f"'{column}' column not found in Social_Network_Ads.csv"
        )

ads = ads.dropna(subset=required_columns)

X_class = ads[["Age", "EstimatedSalary"]]
y_class = ads["Purchased"]

X_train_class, X_test_class, y_train_class, y_test_class = (
    train_test_split(
        X_class,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )
)

# Scale the input features
scaler_class = StandardScaler()

X_train_scaled = scaler_class.fit_transform(X_train_class)
X_test_scaled = scaler_class.transform(X_test_class)

class_model = LogisticRegression(random_state=42)

class_model.fit(X_train_scaled, y_train_class)

class_predictions = class_model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test_class,
    class_predictions
)

cm = confusion_matrix(
    y_test_class,
    class_predictions,
    labels=[0, 1]
)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Results:")
print("True Negatives:", cm[0, 0])
print("False Positives:", cm[0, 1])
print("False Negatives:", cm[1, 0])
print("True Positives:", cm[1, 1])

model_results.append({
    "Algorithm": "Logistic Regression",
    "Metric": "Accuracy",
    "Value": round(accuracy, 4)
})

# Save predictions
class_output = pd.DataFrame({
    "Actual Purchased": y_test_class.to_numpy(),
    "Predicted Purchased": class_predictions
})

class_output.to_csv(
    "results/classification_predictions.csv",
    index=False
)

# GRAPH 2: Confusion Matrix
plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    ["Not Purchased", "Purchased"]
)

plt.yticks(
    [0, 1],
    ["Not Purchased", "Purchased"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=12
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "classification_confusion_matrix.png",
    dpi=150
)

plt.close()


# ==================================================
# 3. K-MEANS CLUSTERING - MALL CUSTOMERS
# ==================================================

print("\n" + "=" * 50)
print("3. K-MEANS CLUSTERING")
print("=" * 50)

customers = pd.read_csv("Datasets/Mall_Customers.csv")

income_column = "Annual Income (k$)"
spending_column = "Spending Score (1-100)"

for column in [income_column, spending_column]:
    if column not in customers.columns:
        raise ValueError(
            f"'{column}' column not found in Mall_Customers.csv"
        )

customers = customers.dropna(
    subset=[income_column, spending_column]
).copy()

X_cluster = customers[
    [income_column, spending_column]
]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# GRAPH 3: Elbow Method
inertia_values = []

for k in range(1, 11):
    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans_test.fit(X_scaled)
    inertia_values.append(kmeans_test.inertia_)

plt.figure(figsize=(7, 5))

plt.plot(
    range(1, 11),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.xticks(range(1, 11))
plt.tight_layout()

plt.savefig("elbow_plot.png", dpi=150)
plt.close()

# Train final K-Means model with 5 clusters
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

customers["Cluster"] = clusters

silhouette = silhouette_score(
    X_scaled,
    clusters
)

print(f"Silhouette Score: {silhouette:.4f}")
print("\nNumber of Customers in Each Cluster:")
print(customers["Cluster"].value_counts().sort_index())

model_results.append({
    "Algorithm": "K-Means Clustering",
    "Metric": "Silhouette Score",
    "Value": round(silhouette, 4)
})

# Save customer cluster assignments
customers.to_csv(
    "results/customer_clusters.csv",
    index=False
)

# GRAPH 4: Customer clusters with X-shaped centres
plt.figure(figsize=(8, 5))

for cluster in sorted(customers["Cluster"].unique()):
    group = customers[customers["Cluster"] == cluster]

    plt.scatter(
        group[income_column],
        group[spending_column],
        label=f"Cluster {cluster}"
    )

# Convert centres to the original data scale
centres = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centres[:, 0],
    centres[:, 1],
    marker="X",
    s=250,
    c="black",
    edgecolors="white",
    linewidths=1.5,
    label="Cluster Centres"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segmentation using K-Means")
plt.legend()
plt.tight_layout()

plt.savefig(
    "customer_clusters.png",
    dpi=150
)

plt.close()


# ==================================================
# SAVE OVERALL MODEL RESULTS
# ==================================================

results_df = pd.DataFrame(model_results)

results_df.to_csv(
    "results/model_results.csv",
    index=False
)

print("\n" + "=" * 50)
print("ALL MODELS COMPLETED SUCCESSFULLY!")
print("=" * 50)

print("\nGenerated Graphs:")
print("1. insurance_actual_vs_predicted.png")
print("2. classification_confusion_matrix.png")
print("3. elbow_plot.png")
print("4. customer_clusters.png")

print("\nGenerated CSV Files:")
print("1. results/insurance_predictions.csv")
print("2. results/classification_predictions.csv")
print("3. results/customer_clusters.csv")
print("4. results/model_results.csv")
