import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Path settings
TRAIN_PATH = r"C:\Users\asus\Downloads\SEEM3650\raw data\train_2000_2025_80.csv"
TEST_PATH  = r"C:\Users\asus\Downloads\SEEM3650\raw data\test_2000_2025_20.csv"
OUTPUT_PATH = r"C:\Users\asus\Downloads\SEEM3650\raw data\tree_bagging_results.csv"

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

feature_cols = [
    "rainfall_total",
    "relative_humidity_mean",
    "solar_radiation_total",
    "temp_max",
    "temp_mean",
    "temp_min",
    "wind_speed_mean",
    "amount_of_cloud_mean"
]
target_col = "target_next_day"

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

# Base tree + bagging
base_tree = DecisionTreeClassifier(
    max_depth=None,          # bagging usually works well with strong/deep trees
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

bagging_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", BaggingClassifier(
        estimator=base_tree,
        n_estimators=200,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    ))
])

# Train
bagging_model.fit(X_train, y_train)

# Predict
y_pred = bagging_model.predict(X_test)
y_prob = bagging_model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

print("=" * 60)
print("Tree Bagging Results")
print("=" * 60)
print(f"AUC       : {auc:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1        : {f1:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print("\nConfusion Matrix:")
print(cm)

# Output
results_df = pd.DataFrame([{
    "model": "Bagging + Decision Tree",
    "AUC": auc,
    "Recall": recall,
    "Precision": precision,
    "F1": f1,
    "Accuracy": accuracy,
    "TN": cm[0, 0],
    "FP": cm[0, 1],
    "FN": cm[1, 0],
    "TP": cm[1, 1]
}])

results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nOutput saved: {OUTPUT_PATH}")