import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

print(">>> Script started")

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH.mkdir(exist_ok=True)

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
customers = pd.read_csv(DATA_PATH / "customers.csv")
loans = pd.read_csv(DATA_PATH / "loans.csv")
applications = pd.read_csv(DATA_PATH / "applications.csv")
transactions = pd.read_csv(DATA_PATH / "transactions.csv")
defaults = pd.read_csv(DATA_PATH / "defaults.csv")

# --------------------------------------------------
# Date parsing
# --------------------------------------------------
date_parsing_tasks = [
    (loans, ["Disbursal_Date", "Repayment_Start_Date", "Repayment_End_Date"]),
    (applications, ["Application_Date", "Approval_Date"]),
    (transactions, ["Transaction_Date"]),
    (defaults, ["Default_Date"]),
]

for df, cols in date_parsing_tasks:
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# --------------------------------------------------
# Default information
# --------------------------------------------------
defaults["Default_Flag"] = 1

loan_defaults = defaults[
    [
        "Loan_ID",
        "Default_Flag",
        "Default_Amount",
        "Recovery_Amount",
        "Recovery_Status",
        "Legal_Action",
        "Default_Date",
    ]
]

# --------------------------------------------------
# Create loan master (BASE TABLE)
# --------------------------------------------------
loan_master = loans.merge(
    loan_defaults,
    on="Loan_ID",
    how="left"
)

loan_master["Default_Flag"] = (
    loan_master["Default_Flag"]
    .fillna(0)
    .astype(int)
)

# --------------------------------------------------
# Recovery metrics
# --------------------------------------------------
loan_master["Recovery_Rate"] = np.where(
    loan_master["Default_Flag"] == 1,
    loan_master["Recovery_Amount"] / loan_master["Default_Amount"],
    np.nan
)

# --------------------------------------------------
# Application features (APPROVED ONLY)
# --------------------------------------------------
applications_approved = applications[
    applications["Approval_Status"] == "Approved"
]

loan_master = loan_master.merge(
    applications_approved[
        [
            "Loan_ID",
            "Application_Date",
            "Approval_Date",
            "Loan_Purpose",
            "Source_Channel",
            "Processing_Fee",
        ]
    ],
    on="Loan_ID",
    how="left"
)

loan_master["Processing_Time_Days"] = (
    loan_master["Approval_Date"] - loan_master["Application_Date"]
).dt.days

# --------------------------------------------------
# Customer features
# --------------------------------------------------
loan_master = loan_master.merge(
    customers,
    on="Customer_ID",
    how="left"
)

loan_master["Income_Band"] = pd.cut(
    loan_master["Annual_Income"],
    bins=[0, 300000, 600000, 1000000, np.inf],
    labels=["Low", "Medium", "High", "Very High"]
)

loan_master["Credit_Score_Band"] = pd.cut(
    loan_master["Credit_Score"],
    bins=[0, 600, 700, 800, np.inf],
    labels=["Poor", "Average", "Good", "Excellent"]
)

loan_master["EMI_to_Income_Ratio"] = (
    loan_master["EMI_Amount"] * 12 / loan_master["Annual_Income"]
)

# --------------------------------------------------
# Time-based features
# --------------------------------------------------
loan_master["Loan_Age_Months"] = (
    (pd.Timestamp.today() - loan_master["Disbursal_Date"])
    .dt.days / 30
).round(1)

# --------------------------------------------------
# Save output (robust & explicit)
# --------------------------------------------------
print("Feature engineering completed.")
print("Final dataset shape:", loan_master.shape)

output_file = OUTPUT_PATH / "loan_master_table.csv"
print(">>> Saving file to:", output_file)

try:
    loan_master.to_csv(output_file, index=False)
    print(">>> File write completed successfully")
except Exception as e:
    print(">>> ERROR while saving file:", e)

print(">>> Script finished execution")
