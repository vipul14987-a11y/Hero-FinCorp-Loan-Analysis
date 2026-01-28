import pandas as pd
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw"

def load_csv(name):
    path = DATA_PATH / f"{name}.csv"
    df = pd.read_csv(path)
    print(f"{name}: loaded with shape {df.shape}")
    return df

customers = load_csv("customers")
loans = load_csv("loans")
applications = load_csv("applications")
transactions = load_csv("transactions")
defaults = load_csv("defaults")
branches = load_csv("branches")
datasets = {
    "customers": customers,
    "loans": loans,
    "applications": applications,
    "transactions": transactions,
    "defaults": defaults,
    "branches": branches,
}

print("\n--- Dataset Info ---")
for name, df in datasets.items():
    print(f"\n{name.upper()}")
    print(df.info())

print("\n--- Missing Values ---")
for name, df in datasets.items():
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"\n{name}")
        print(missing)
    else:
        print(f"\n{name}: No missing values")

print("\n--- Duplicate Records ---")
for name, df in datasets.items():
    print(f"{name}: {df.duplicated().sum()} duplicates")

date_columns = {
    "applications": ["Application_Date"],
    "transactions": ["Transaction_Date"],
    "defaults": ["Default_Date"],
}

print("\n--- Date Parsing ---")
for dataset_name, cols in date_columns.items():
    df = datasets[dataset_name]
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        print(f"{dataset_name}.{col} converted to datetime")

print("\nData quality checks completed successfully.")
