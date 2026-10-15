
# ============================================================
# COMPARISON OF CLASSIFICATION ALGORITHMS
# Dataset: Social_Network_Ads.csv
# Algorithms: KNN, Logistic Regression, Decision Tree
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

os.makedirs("results", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("Datasets/Social_Network_Ads.csv")
df.columns = df.columns.str.strip()

df = df.drop_duplicates()
df = df.drop(columns=["User ID"], errors="ignore")
df = df.dropna()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# 2. Separate features and target
X = df.drop(columns=["Purchased"])
y = df["Purchased"]

numeric_cols = ["Age", "EstimatedSalary"]
categorical_cols = ["Gender"] if "Gender" in X.columns else []

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Define three classification algorithms
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        min_samples_leaf=3
    )
}

results = []
predictions_dict = {}

# 5. Train and compare models
for name, classifier in models.items():

    print(f"\nTraining {name}...")

    # Scale numerical features for KNN and Logistic Regression
    if name in ["KNN", "Logistic Regression"]:
        numeric_transformer = StandardScaler()
    else:
        numeric_transformer = "passthrough"

    transformers = [
        ("numeric", numeric_transformer, numeric_cols)
    ]

    if categorical_cols:
        transformers.append(
            (
                "gender",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore"
                ),
                categorical_cols
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions_dict[name] = (model, y_pred)

    results.append({
        "Algorithm": name,
        "Accuracy (%)": accuracy_score(
            y_test, y_pred
        ) * 100,

        "Precision (%)": precision_score(
            y_test, y_pred, zero_division=0
        ) * 100,

        "Recall (%)": recall_score(
            y_test, y_pred, zero_division=0
        ) * 100,

        "F1 Score (%)": f1_score(
            y_test, y_pred, zero_division=0
        ) * 100
    })

    print(f"{name} completed!")

# 6. Display comparison table
results_df = pd.DataFrame(results)

print("\n========== CLASSIFICATION COMPARISON ==========")
print(results_df.round(2).to_string(index=False))

results_df.to_csv(
    "results/classification_comparison.csv",
    index=False
)

# Select best model based on F1 score
best_name = results_df.loc[
    results_df["F1 Score (%)"].idxmax(),
    "Algorithm"
]

best_model, best_pred = predictions_dict[best_name]

print("\n========== BEST CLASSIFICATION MODEL ==========")
print("Best algorithm:", best_name)

print("\nClassification report:")
print(classification_report(
    y_test,
    best_pred,
    target_names=["Not Purchased", "Purchased"],
    zero_division=0
))

# 7. Accuracy comparison graph
plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="Algorithm",
    y="Accuracy (%)",
    hue="Algorithm",
    legend=False
)

plt.title("Accuracy Comparison of Classifiers")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(
    "results/classification_accuracy_comparison.png",
    dpi=150
)

plt.show()
plt.close()

# 8. F1-score comparison graph
plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="Algorithm",
    y="F1 Score (%)",
    hue="Algorithm",
    legend=False
)

plt.title("F1-Score Comparison of Classifiers")
plt.ylabel("F1 Score (%)")
plt.ylim(0, 100)
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(
    "results/classification_f1_comparison.png",
    dpi=150
)

plt.show()
plt.close()

# 9. Confusion matrix for each model
for name, (model, y_pred) in predictions_dict.items():

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Purchased", "Purchased"],
        yticklabels=["Not Purchased", "Purchased"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")

    plt.savefig(
        f"results/confusion_matrix_{safe_name}.png",
        dpi=150
    )

    plt.show()
    plt.close()

# 10. Age vs salary graph
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Age",
    y="EstimatedSalary",
    hue="Purchased",
    palette="Set1"
)

plt.title("Age vs Estimated Salary by Purchase")
plt.tight_layout()

plt.savefig(
    "results/classification_age_salary.png",
    dpi=150
)

plt.show()
plt.close()

# 11. Save predictions from the best model
prediction_output = pd.DataFrame({
    "Actual Purchased": y_test.to_numpy(),
    "Predicted Purchased": best_pred
})

prediction_output.to_csv(
    "results/classification_predictions.csv",
    index=False
)

print("\nClassification analysis completed!")
print("All results saved in the results folder.")
