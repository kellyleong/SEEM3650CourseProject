import pandas as pd
import numpy as np
from pathlib import Path
from functools import reduce

# 0. Modify here: your raw data root folder
ROOT_DIR = Path(r"C:\Users\asus\Downloads\SEEM3650\raw data")

# 1. Feature mapping of folders/files
FEATURE_SOURCES = {
    "rainfall_total": "rainfall_total (daily)",
    "relative_humidity_mean": "relative humidity_mean (daily)",
    "solar_radiation_total": "solar radiation_total (daily)",
    "temp_max": "temp_max (daily)",
    "temp_mean": "temp_mean (daily)",
    "temp_min": "temp_min (daily)",
    "wind_speed_mean": "wind speed_mean (daily)",
    "amount_of_cloud_mean": "amount of cloud_mean (daily).csv",
    "sea_temp_mean": "sea temp_mean (daily).csv"
}

VERY_HOT_THRESHOLD = 33.0

# 2. Function to read HKO daily csv
def read_hko_daily_csv(file_path: Path, value_col_name="value"):
    """
    For HKO daily file format:
    row1: Chinese title
    row2: English title
    row3: actual header
    then data
    """
    # skip first two description rows
    df = pd.read_csv(file_path, skiprows=2, encoding="utf-8-sig")

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Automatically find columns
    year_col = [c for c in df.columns if ("Year" in c or "年" in c)][0]
    month_col = [c for c in df.columns if ("Month" in c or "月" in c)][0]
    day_col = [c for c in df.columns if ("Day" in c or "日" in c)][0]
    value_col = [c for c in df.columns if ("Value" in c or "數值" in c)][0]

    completeness_candidates = [c for c in df.columns if ("Completeness" in c or "完整" in c)]
    completeness_col = completeness_candidates[0] if completeness_candidates else None

    keep_cols = [year_col, month_col, day_col, value_col]
    if completeness_col is not None:
        keep_cols.append(completeness_col)

    df = df[keep_cols].copy()

    rename_map = {
        year_col: "year",
        month_col: "month",
        day_col: "day",
        value_col: value_col_name
    }
    if completeness_col is not None:
        rename_map[completeness_col] = "completeness"

    df.rename(columns=rename_map, inplace=True)

    # Strip whitespace from strings
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # Clean value
    df[value_col_name] = (
        df[value_col_name]
        .replace({
            "***": np.nan,
            "": np.nan,
            "nan": np.nan,
            "None": np.nan
        })
    )

    df[value_col_name] = pd.to_numeric(df[value_col_name], errors="coerce")

    # Create date
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # completeness flag
    if "completeness" in df.columns:
        df["is_complete"] = (df["completeness"] == "C").astype(int)

    # Deduplicate and sort
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    return df

# 3. List all files under data source
def list_all_files(source_path: Path):
    if not source_path.exists():
        raise FileNotFoundError(f"Path not found: {source_path}")

    if source_path.is_file():
        return [source_path]

    if source_path.is_dir():
        files = [p for p in source_path.iterdir() if p.is_file()]
        files = sorted(files)
        if len(files) == 0:
            raise ValueError(f"Folder exists but no files: {source_path}")
        return files

    raise ValueError(f"Not a valid file or folder: {source_path}")

# 4. Build one feature series from a source
#    - If multiple station files: average same day
#    - If only one file: use directly
def build_feature_series(source_path: Path, feature_name: str):
    files = list_all_files(source_path)

    station_dfs = []

    for fp in files:
        station_name = fp.stem  # use filename as station label

        try:
            temp = read_hko_daily_csv(fp, value_col_name=station_name)
            temp = temp[["date", station_name]].copy()
            station_dfs.append(temp)
        except Exception as e:
            print(f"[WARNING] Failed to read, skipping file: {fp.name}")
            print(f"Reason: {e}")

    if len(station_dfs) == 0:
        raise ValueError(f"{feature_name} all files failed to read")

    # merge all stations
    merged_station = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"),
        station_dfs
    )

    merged_station = merged_station.sort_values("date").reset_index(drop=True)

    station_cols = [c for c in merged_station.columns if c != "date"]

    # Average across stations for same day
    merged_station[feature_name] = merged_station[station_cols].mean(axis=1, skipna=True)

    # Record number of stations with value on that day
    merged_station[f"{feature_name}_n_stations"] = merged_station[station_cols].notna().sum(axis=1)

    # Output only date + feature + station count
    result = merged_station[["date", feature_name, f"{feature_name}_n_stations"]].copy()

    return result

# 5. Read all features
feature_tables = {}

for feature_name, relative_path in FEATURE_SOURCES.items():
    source_path = ROOT_DIR / relative_path

    print("\n" + "=" * 60)
    print(f"Processing feature: {feature_name}")
    print(f"Source: {source_path}")

    ft = build_feature_series(source_path, feature_name)
    feature_tables[feature_name] = ft

    print(f"Finished {feature_name}, rows = {len(ft)}")
    print(ft.head())

# 6. Merge all features into master table
master_df = reduce(
    lambda left, right: pd.merge(left, right, on="date", how="outer"),
    feature_tables.values()
)

master_df = master_df.sort_values("date").reset_index(drop=True)

print("\n" + "=" * 30)
print("Master data shape:", master_df.shape)
print(master_df.head())
print("=" * 30)

# 7. Check missing rate
missing_rate = master_df.isna().mean().sort_values(ascending=False)
missing_rate_df = missing_rate.reset_index()
missing_rate_df.columns = ["column", "missing_rate"]

print("\nMissing rate per column:")
print(missing_rate_df)

# 8. Create target
#    very_hot_today = temp_max >= 33
#    target_next_day = whether next day is very hot day
if "temp_max" not in master_df.columns:
    raise ValueError("master_df has no temp_max, cannot create target")

master_df["very_hot_today"] = (master_df["temp_max"] >= VERY_HOT_THRESHOLD).astype(float)
master_df["target_next_day"] = master_df["very_hot_today"].shift(-1)

# Last day has no next_day label
master_df = master_df.dropna(subset=["target_next_day"]).copy()
master_df["target_next_day"] = master_df["target_next_day"].astype(int)

print("\nvery_hot_today distribution:")
print(master_df["very_hot_today"].value_counts(dropna=False))

print("\ntarget_next_day distribution:")
print(master_df["target_next_day"].value_counts(dropna=False))

# 9. Build model-ready dataset
#    Rule: remove features with missing > 80%
# First check missing rate
feature_missing = master_df.isna().mean()

# Manually exclude columns
exclude_cols = ["date", "very_hot_today", "target_next_day"]

# Find feature columns
candidate_feature_cols = [
    c for c in master_df.columns
    if c not in exclude_cols and not c.endswith("_n_stations")
]

# Keep only features with missing <= 80%
selected_feature_cols = [
    c for c in candidate_feature_cols
    if feature_missing[c] <= 0.8
]

print("\nRetained features (missing <= 80%):")
print(selected_feature_cols)

print("\nDropped features (missing > 80%):")
dropped_feature_cols = [
    c for c in candidate_feature_cols
    if feature_missing[c] > 0.8
]
print(dropped_feature_cols)

# Create model_df
model_df = master_df[["date"] + selected_feature_cols + ["very_hot_today", "target_next_day"]].copy()

print("\nModel-ready data shape:", model_df.shape)
print(model_df.head())

# 10. Output results
master_df.to_csv(ROOT_DIR / "merged_daily_all_features.csv", index=False, encoding="utf-8-sig")
missing_rate_df.to_csv(ROOT_DIR / "missing_rate_summary.csv", index=False, encoding="utf-8-sig")
model_df.to_csv(ROOT_DIR / "model_ready_daily_data.csv", index=False, encoding="utf-8-sig")

print("\nOutput files saved:")
print(ROOT_DIR / "merged_daily_all_features.csv")
print(ROOT_DIR / "missing_rate_summary.csv")
print(ROOT_DIR / "model_ready_daily_data.csv")