import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load data
file_path = r"C:\Users\asus\Downloads\SEEM3650\raw data\model_ready_daily_data.csv"
df = pd.read_csv(file_path)

# Convert date format
df["date"] = pd.to_datetime(df["date"])

# 2. Keep only data from 2000-01-01 to 2025-12-31
df = df[(df["date"] >= "2000-01-01") & (df["date"] <= "2025-12-31")].copy()

# Sort by date (very important)
df = df.sort_values("date").reset_index(drop=True)

print("Date range of data:")
print(df["date"].min(), "to", df["date"].max())
print("Total rows:", len(df))

# 3. Select features
#    only drop sea_temp_mean (because missing > 80%)
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

# Keep only required columns
df = df[["date"] + feature_cols + [target_col]].copy()

print("\nFeatures used:")
print(feature_cols)

# 4. X / y
X = df[feature_cols]
y = df[target_col]

# Keep dates for checking train/test range later
dates = df["date"]

# 5. Use train_test_split for 80/20 split
#    Note: shuffle=False to preserve time order
X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
    X,
    y,
    dates,
    test_size=0.2,
    shuffle=False
)

# 6. Rebuild train / test DataFrames
train_df = pd.concat(
    [date_train.reset_index(drop=True),
     X_train.reset_index(drop=True),
     y_train.reset_index(drop=True)],
    axis=1
)

test_df = pd.concat(
    [date_test.reset_index(drop=True),
     X_test.reset_index(drop=True),
     y_test.reset_index(drop=True)],
    axis=1
)

# 7. Check split results
print("\nTrain size:", len(train_df))
print("Test size :", len(test_df))

print("\nTrain date range:")
print(train_df["date"].min(), "to", train_df["date"].max())

print("\nTest date range:")
print(test_df["date"].min(), "to", test_df["date"].max())

print("\nTrain target distribution:")
print(train_df[target_col].value_counts(dropna=False))
print(train_df[target_col].value_counts(normalize=True, dropna=False))

print("\nTest target distribution:")
print(test_df[target_col].value_counts(dropna=False))
print(test_df[target_col].value_counts(normalize=True, dropna=False))

# 8. Output train / test
output_dir = r"C:\Users\asus\Downloads\SEEM3650\raw data"

train_df.to_csv(f"{output_dir}\\train_2000_2025_80.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(f"{output_dir}\\test_2000_2025_20.csv", index=False, encoding="utf-8-sig")

print("\nOutput files saved:")
print(f"{output_dir}\\train_2000_2025_80.csv")
print(f"{output_dir}\\test_2000_2025_20.csv")