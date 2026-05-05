import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
OUTPUT_PATH = r"C:\Users\asus\Downloads\SEEM3650\raw data\logistic_threshold_tuning_results.csv"

# Tunable parameters
N_SPLITS = 5
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)
OPTIMIZE_METRIC = "f1"   # can be changed to "precision" / "recall" / "accuracy"

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

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

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df[target_col].copy()

# Logistic pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# 1. Use TimeSeriesSplit to generate out-of-fold probabilities
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

oof_probs = np.full(shape=len(X_train), fill_value=np.nan)

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    model.fit(X_tr, y_tr)
    val_prob = model.predict_proba(X_val)[:, 1]
    oof_probs[val_idx] = val_prob

    print(f"Fold {fold}: train={len(tr_idx)}, valid={len(val_idx)}")

valid_mask = ~np.isnan(oof_probs)
y_oof = y_train[valid_mask].values
p_oof = oof_probs[valid_mask]

# 2. Sweep thresholds, find best threshold
def compute_metric(y_true, y_pred, metric_name):
    if metric_name == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    elif metric_name == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    elif metric_name == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    elif metric_name == "accuracy":
        return accuracy_score(y_true, y_pred)
    else:
        raise ValueError("Unsupported metric name.")

best_threshold = None
best_score = -np.inf

threshold_records = []

for thr in THRESHOLD_GRID:
    y_pred_thr = (p_oof >= thr).astype(int)

    acc = accuracy_score(y_oof, y_pred_thr)
    prec = precision_score(y_oof, y_pred_thr, zero_division=0)
    rec = recall_score(y_oof, y_pred_thr, zero_division=0)
    f1 = f1_score(y_oof, y_pred_thr, zero_division=0)

    score_to_optimize = compute_metric(y_oof, y_pred_thr, OPTIMIZE_METRIC)

    threshold_records.append({
        "threshold": thr,
        "cv_accuracy": acc,
        "cv_precision": prec,
        "cv_recall": rec,
        "cv_f1": f1,
        "cv_score_optimized": score_to_optimize
    })

    if score_to_optimize > best_score:
        best_score = score_to_optimize
        best_threshold = thr

threshold_df = pd.DataFrame(threshold_records)

print("\nBest threshold found from CV:")
print(f"Optimize metric : {OPTIMIZE_METRIC}")
print(f"Best threshold  : {best_threshold:.2f}")
print(f"Best CV score   : {best_score:.4f}")

# 3. Retrain on full training set
model.fit(X_train, y_train)

# 4. Evaluate on test set with optimal threshold
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
auc = roc_auc_score(y_test, test_prob)

cm = confusion_matrix(y_test, test_pred)

print("\n" + "=" * 60)
print("Logistic Regression + Threshold Tuning")
print("=" * 60)
print(f"Selected Threshold : {best_threshold:.2f}")
print(f"AUC                : {auc:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"F1                 : {f1:.4f}")
print(f"Accuracy           : {accuracy:.4f}")
print("\nConfusion Matrix:")
print(cm)

# 5. Output
summary_df = pd.DataFrame([{
    "model": "Logistic Regression (Threshold Tuned)",
    "selected_threshold": best_threshold,
    "optimize_metric": OPTIMIZE_METRIC,
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

# Save two separate CSVs
summary_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
threshold_df.to_csv(
    r"C:\Users\asus\Downloads\SEEM3650\raw data\logistic_threshold_grid_search.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSummary saved: {OUTPUT_PATH}")
print(r"Threshold grid saved: C:\Users\asus\Downloads\SEEM3650\raw data\logistic_threshold_grid_search.csv")