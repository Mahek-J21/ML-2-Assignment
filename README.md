# Machine Learning Algorithms Assignment

## Overview
This project demonstrates three machine learning techniques: boosting, classification, and clustering using Python and scikit-learn.

## 1. Boosting — AdaBoost Regression
- **Dataset:** `insurance_pre.csv`
- **Objective:** Predict insurance charges.
- **Algorithm:** AdaBoost Regression with a Decision Tree base estimator.
- **Evaluation metrics:**
  - MAE: 4677.38
  - RMSE: 5549.03
  - R² Score: 0.8017
- **Output:** `insurance_predictions.csv`

## 2. Classification — Logistic Regression
- **Dataset:** `Social_Network_Ads.csv`
- **Objective:** Predict whether a person purchases a product.
- **Algorithm:** Logistic Regression.
- **Accuracy:** 83.75%
- **Output:** `classification_predictions.csv`

## 3. Clustering — K-Means
- **Dataset:** `Mall_Customers.csv`
- **Objective:** Group customers based on annual income and spending score.
- **Algorithm:** K-Means with five clusters.
- **Silhouette Score:** 0.5547
- **Cluster sizes:** 81, 39, 22, 35, 23
- **Outputs:** `customer_clusters.csv`, `customer_clusters.png`, `elbow_plot.png`

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run
1. Install Python 3.11 or a compatible version.
2. Install the required libraries using `pip install pandas numpy scikit-learn matplotlib seaborn`.
3. Run `python main.py` from the project folder.

## Conclusion
This project demonstrates how boosting can predict numerical values, classification can predict categories, and clustering can identify groups in unlabeled data.