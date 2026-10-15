
# ============================================================
# COMPARISON OF BOOSTING ALGORITHMS
# Dataset: insurance_pre.csv
# Algorithms: AdaBoost, Gradient Boosting,
#             Histogram Gradient Boosting
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# 1. Create output folder
os.makedirs("results", exist_ok=True)


# 2. Load dataset
df = pd.read_csv("Datasets/insurance_pre.csv")

df.columns = df.columns.str.strip()
df = df.dropna()

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 3. Separate features and target
X = df.drop(columns=["charges"])
y = df["charges"]

# 4. Identify categorical columns
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# 5. Preprocess categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ],
    remainder="passthrough"
)

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. Define boosting algorithms
models = {
    "AdaBoost": AdaBoostRegressor(
        estimator=DecisionTreeRegressor(
            max_depth=3,
            random_state=42
        ),
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),

    "Histogram Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
}

# 8. Train and evaluate each model
results = []
predictions = {}

for name, regressor in models.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Algorithm": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"{name} completed!")

# 9. Display comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

print("\n========== BOOSTING COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

# Save metrics
results_df.to_csv(
    "results/boosting_comparison.csv",
    index=False
)

# 10. Identify the best model using RMSE
best_name = results_df.iloc[0]["Algorithm"]
best_r2 = results_df.iloc[0]["R2 Score"]
best_rmse = results_df.iloc[0]["RMSE"]

print("\n========== BEST MODEL ON TEST SET ==========")
print("Algorithm:", best_name)
print(f"RMSE: {best_rmse:.2f}")
print(f"R2 Score: {best_r2:.4f}")

# 11. Plot RMSE comparison
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Algorithm"],
    results_df["RMSE"]
)

plt.ylabel("RMSE (lower is better)")
plt.title("Boosting Algorithms: RMSE Comparison")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "results/boosting_rmse_comparison.png",
    dpi=150
)

plt.show()
plt.close()

# 12. Plot R2 comparison
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Algorithm"],
    results_df["R2 Score"]
)

plt.ylabel("R2 Score (higher is better)")
plt.title("Boosting Algorithms: R2 Comparison")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "results/boosting_r2_comparison.png",
    dpi=150
)

plt.show()
plt.close()

# 13. Actual vs predicted for the best model
best_pred = predictions[best_name]

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    best_pred,
    alpha=0.7
)

lower = min(y_test.min(), best_pred.min())
upper = max(y_test.max(), best_pred.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    "r--",
    label="Ideal Prediction"
)

plt.xlabel("Actual Insurance Charges")
plt.ylabel("Predicted Insurance Charges")
plt.title(f"Actual vs Predicted: {best_name}")
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/boosting_actual_vs_predicted.png",
    dpi=150
)

plt.show()
plt.close()

# 14. Save predictions from the best model
prediction_output = pd.DataFrame({
    "Actual Charges": y_test.to_numpy(),
    "Predicted Charges": best_pred
})

prediction_output.to_csv(
    "results/boosting_predictions.csv",
    index=False
)

print("\nComparison completed successfully!")
print("Results saved in the results folder.")
