"""
clean_data.py  -  Cleans the 50-column Telco churn dataset.

Run from your project root:
    python clean_data.py

Input : data/raw/telco_churn.csv
Output: data/processed/telco_clean.csv
"""
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/telco_churn.csv")
OUT = Path("data/processed/telco_clean.csv")


def to_snake(name: str) -> str:
    """'Avg Monthly GB Download' -> 'avg_monthly_gb_download'"""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Tidy column names
    df.columns = [to_snake(c) for c in df.columns]

    # 2. Create the target: 1 = churned, 0 = stayed/joined
    df["churn"] = (df["churn_label"] == "Yes").astype(int)

    # 3. Remove LEAKAGE columns (they reveal the answer or only exist after leaving)
    leakage = [
        "churn_label",       # the target itself (we made 'churn' from it)
        "customer_status",   # Churned / Stayed / Joined  -> same as target
        "churn_score",       # a score already computed by IBM's own churn model
        "churn_category",    # only filled in for customers who left
        "churn_reason",      # only filled in for customers who left
        "satisfaction_score",  # scores 1-2 = 100% churn, 4-5 = 0% churn (too perfect)
        "cltv",              # computed by IBM using churn information
    ]
    df = df.drop(columns=leakage)

    # 4. Remove columns with no predictive value
    useless = [
        "customer_id",   # unique ID
        "country",       # always 'United States'
        "state",         # always 'California'
        "quarter",       # always 'Q3'
        "city", "zip_code", "latitude", "longitude",  # location detail (too granular)
    ]
    df = df.drop(columns=useless)

    # 5. Remove redundant columns (exactly derivable from other columns)
    redundant = [
        "under_30",          # derived from age
        "senior_citizen",    # derived from age (>= 65)
        "dependents",        # derived from number_of_dependents
        "internet_service",  # internet_type already says 'No Internet'
        "total_revenue",     # = total_charges - refunds + extra data + long distance
    ]
    df = df.drop(columns=redundant)

    # 6. Missing values: they mean "not applicable", not "unknown"
    df["offer"] = df["offer"].fillna("No Offer")
    df["internet_type"] = df["internet_type"].fillna("No Internet")

    # 7. Convert Yes/No columns to 1/0
    yes_no = [c for c in df.columns
              if df[c].dtype == "object" or str(df[c].dtype).startswith("str")]
    for c in yes_no:
        if set(df[c].dropna().unique()) <= {"Yes", "No"}:
            df[c] = (df[c] == "Yes").astype(int)

    return df


if __name__ == "__main__":
    raw = pd.read_csv(RAW)
    print("Raw shape  :", raw.shape)

    clean_df = clean(raw)

    assert clean_df.isnull().sum().sum() == 0, "Still have missing values!"
    assert clean_df["churn"].isin([0, 1]).all()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUT, index=False)

    print("Clean shape:", clean_df.shape)
    print("Churn rate :", round(clean_df["churn"].mean(), 3))
    print("Saved to   :", OUT)