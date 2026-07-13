from __future__ import annotations
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_wc2026_results() -> pd.DataFrame:
    path = DATA_DIR / "wc2026_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()

def merge_wc2026_results(existing_results: pd.DataFrame) -> pd.DataFrame:
    wc = load_wc2026_results()
    if wc.empty:
        return existing_results
    
    existing = existing_results.copy()
    existing["date"] = pd.to_datetime(existing["date"])
    wc["date"] = pd.to_datetime(wc["date"])
    
    combined = pd.concat([existing, wc], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "country", "opposition_country"], keep="last"
    )
    combined = combined.sort_values(["country", "date"], ascending=[True, False])
    combined = combined.groupby("country", group_keys=False).head(20)
    combined = combined.reset_index(drop=True)
    return combined
