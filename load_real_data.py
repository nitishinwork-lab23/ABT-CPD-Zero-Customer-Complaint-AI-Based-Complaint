"""Load the organisers' REAL ABT-CPD complaint file into the prototype database.

Usage:  python load_real_data.py path/to/complaints.xlsx  [units_dispatched.csv]

1. Edit COLUMN_MAP so the keys are YOUR column names and the values are the prototype's names.
2. Only date, customer, product, complaint_text and category are mandatory; every other column is optional
   (root_cause, corrective_action, action_effective [Y/N], closure_days, batch_no, line, severity, ...).
3. If `category` is missing, run the file through the NLP classifier first (see engine.TextClassifier) or map manually.
4. Optional second file with columns product, month (YYYY-MM), units_dispatched enables complaints-per-million-units.
"""
import sys, sqlite3, pandas as pd, numpy as np
import generate_data as G

COLUMN_MAP = {            # YOUR column name  ->  prototype column name
    "Complaint Date": "date", "Customer Name": "customer", "Product": "product",
    "Complaint Description": "complaint_text", "Complaint Category": "category",
    "Batch No": "batch_no", "Root Cause": "root_cause", "Corrective Action": "corrective_action",
    "Action Effective": "action_effective",
}
REQUIRED = ["date", "customer", "product", "complaint_text", "category"]
OPTIONAL = ["complaint_id", "customer_type", "batch_no", "line", "root_cause", "corrective_action", "action_effective",
            "closure_days", "severity", "seal_temp_c", "moisture_pct", "fill_weight_dev_g", "test_result"]

def main(path, vol_path=None):
    raw = pd.read_excel(path) if path.lower().endswith(("xlsx", "xls")) else pd.read_csv(path)
    df = raw.rename(columns=COLUMN_MAP)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss: sys.exit(f"Missing required columns after mapping: {miss}. Edit COLUMN_MAP.")
    df["date"] = pd.to_datetime(df["date"]); df = df.sort_values("date").reset_index(drop=True)
    for c in OPTIONAL:
        if c not in df.columns: df[c] = None
    if df["complaint_id"].isna().all(): df["complaint_id"] = [f"C{i+1:05d}" for i in range(len(df))]
    df["customer_type"] = df["customer_type"].fillna("Unknown"); df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    cols = ["complaint_id", "date", "customer", "customer_type", "product", "category", "complaint_text", "batch_no", "line", "root_cause",
            "corrective_action", "action_effective", "closure_days", "severity", "seal_temp_c", "moisture_pct", "fill_weight_dev_g", "test_result"]
    df = df[cols]
    if vol_path:
        vols = pd.read_csv(vol_path)[["product", "month", "units_dispatched"]]
    else:   # no volume data: assume constant volume so rate-based charts still work (clearly approximate)
        months = pd.period_range(df["date"].min(), df["date"].max(), freq="M").astype(str)
        vols = pd.DataFrame([(p, m, 100_000) for p in df["product"].unique() for m in months], columns=["product", "month", "units_dispatched"])
        print("WARNING: no volume file given – using a constant 100,000 units/month placeholder.")
    G.write_db("data/complaints.db", df, vols); print(f"Loaded {len(df):,} complaints.  Restart the app (and press 'Retrain models').")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
