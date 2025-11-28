# src/data_prep.py
from pathlib import Path
import pandas as pd

# --- Paths: resolve from project root (parent of src) ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# --- Read input splits ---
train_data = pd.read_csv(RAW_DIR / "train_data.csv")
test_data  = pd.read_csv(RAW_DIR / "test_data.csv")

# --- Utilities (no chained assignment) ---
def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    # compute medians only for numeric columns
    medians = df.select_dtypes(include="number").median()
    # return a new frame with NA filled (no inplace, avoids FutureWarning)
    return df.copy().fillna(medians)

# --- Process ---
train_processed = fill_missing_with_median(train_data)
test_processed  = fill_missing_with_median(test_data)

# --- Write outputs ---
train_processed.to_csv(PROC_DIR / "train_processed.csv", index=False)
test_processed.to_csv(PROC_DIR / "test_processed.csv", index=False)

print(f"Saved: {PROC_DIR / 'train_processed.csv'}")
print(f"Saved: {PROC_DIR / 'test_processed.csv'}")


