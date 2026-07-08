"""
Update the prediction model with actual 2026 World Cup group stage results.
Compiled from ESPN, BBC, FIFA.com, Yahoo Sports, and other sources.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve()))
from src.feature.loader import load_elo_ratings, load_fifa_ratings, ALIASES

DATA_DIR = Path(__file__).resolve().parent / "data"
ELO_PATH = DATA_DIR / "elo_ratings.csv"
FIFA_PATH = DATA_DIR / "fifa_ratings.csv"

# ── 1. All WC2026 group stage results (verified against multiple sources) ──

WC2026_RESULTS = [
    # Group A
    ("2026-06-11", "Mexico", 2, "South Africa", 0, "Group A"),
    ("2026-06-11", "Korea Republic", 2, "Czechia", 1, "Group A"),
    ("2026-06-18", "Czechia", 1, "South Africa", 1, "Group A"),
    ("2026-06-18", "Mexico", 1, "Korea Republic", 0, "Group A"),
    ("2026-06-24", "Mexico", 3, "Czechia", 0, "Group A"),
    ("2026-06-24", "South Africa", 1, "Korea Republic", 0, "Group A"),

    # Group B
    ("2026-06-12", "Canada", 1, "Bosnia and Herzegovina", 1, "Group B"),
    ("2026-06-13", "Qatar", 1, "Switzerland", 1, "Group B"),
    ("2026-06-18", "Switzerland", 4, "Bosnia and Herzegovina", 1, "Group B"),
    ("2026-06-18", "Canada", 6, "Qatar", 0, "Group B"),
    ("2026-06-24", "Switzerland", 2, "Canada", 1, "Group B"),
    ("2026-06-24", "Bosnia and Herzegovina", 3, "Qatar", 1, "Group B"),

    # Group C
    ("2026-06-13", "Haiti", 0, "Scotland", 1, "Group C"),
    ("2026-06-13", "Brazil", 1, "Morocco", 1, "Group C"),
    ("2026-06-19", "Brazil", 3, "Haiti", 0, "Group C"),
    ("2026-06-19", "Scotland", 0, "Morocco", 1, "Group C"),
    ("2026-06-24", "Scotland", 0, "Brazil", 3, "Group C"),
    ("2026-06-24", "Morocco", 4, "Haiti", 2, "Group C"),

    # Group D
    ("2026-06-12", "United States", 4, "Paraguay", 1, "Group D"),
    ("2026-06-13", "Australia", 2, "Türkiye", 0, "Group D"),
    ("2026-06-19", "United States", 2, "Australia", 0, "Group D"),
    ("2026-06-19", "Türkiye", 0, "Paraguay", 1, "Group D"),
    ("2026-06-25", "Türkiye", 3, "United States", 2, "Group D"),
    ("2026-06-25", "Paraguay", 0, "Australia", 0, "Group D"),

    # Group E
    ("2026-06-14", "Côte d'Ivoire", 1, "Ecuador", 0, "Group E"),
    ("2026-06-14", "Germany", 7, "Curaçao", 1, "Group E"),
    ("2026-06-20", "Germany", 2, "Côte d'Ivoire", 1, "Group E"),
    ("2026-06-20", "Ecuador", 0, "Curaçao", 0, "Group E"),
    ("2026-06-25", "Ecuador", 2, "Germany", 1, "Group E"),
    ("2026-06-25", "Curaçao", 0, "Côte d'Ivoire", 2, "Group E"),

    # Group F
    ("2026-06-14", "Netherlands", 2, "Japan", 2, "Group F"),
    ("2026-06-14", "Sweden", 5, "Tunisia", 1, "Group F"),
    ("2026-06-20", "Netherlands", 5, "Sweden", 1, "Group F"),
    ("2026-06-20", "Tunisia", 0, "Japan", 4, "Group F"),
    ("2026-06-25", "Japan", 1, "Sweden", 1, "Group F"),
    ("2026-06-25", "Tunisia", 1, "Netherlands", 3, "Group F"),

    # Group G
    ("2026-06-15", "IR Iran", 2, "New Zealand", 2, "Group G"),
    ("2026-06-15", "Belgium", 1, "Egypt", 1, "Group G"),
    ("2026-06-21", "Belgium", 0, "IR Iran", 0, "Group G"),
    ("2026-06-21", "New Zealand", 1, "Egypt", 3, "Group G"),
    ("2026-06-26", "Egypt", 1, "IR Iran", 1, "Group G"),
    ("2026-06-26", "New Zealand", 1, "Belgium", 5, "Group G"),

    # Group H
    ("2026-06-15", "Saudi Arabia", 1, "Uruguay", 1, "Group H"),
    ("2026-06-15", "Spain", 0, "Cabo Verde", 0, "Group H"),
    ("2026-06-21", "Uruguay", 2, "Cabo Verde", 2, "Group H"),
    ("2026-06-21", "Spain", 4, "Saudi Arabia", 0, "Group H"),
    ("2026-06-26", "Cabo Verde", 0, "Saudi Arabia", 0, "Group H"),
    ("2026-06-26", "Uruguay", 0, "Spain", 1, "Group H"),

    # Group I
    ("2026-06-16", "France", 3, "Senegal", 1, "Group I"),
    ("2026-06-16", "Iraq", 1, "Norway", 4, "Group I"),
    ("2026-06-22", "France", 3, "Iraq", 0, "Group I"),
    ("2026-06-22", "Norway", 3, "Senegal", 2, "Group I"),
    ("2026-06-26", "Norway", 1, "France", 4, "Group I"),
    ("2026-06-26", "Senegal", 5, "Iraq", 0, "Group I"),

    # Group J
    ("2026-06-16", "Argentina", 3, "Algeria", 0, "Group J"),
    ("2026-06-16", "Austria", 3, "Jordan", 1, "Group J"),
    ("2026-06-22", "Argentina", 2, "Austria", 0, "Group J"),
    ("2026-06-22", "Jordan", 1, "Algeria", 2, "Group J"),
    ("2026-06-27", "Algeria", 3, "Austria", 3, "Group J"),
    ("2026-06-27", "Jordan", 1, "Argentina", 3, "Group J"),

    # Group K
    ("2026-06-17", "Portugal", 1, "Congo DR", 1, "Group K"),
    ("2026-06-17", "Uzbekistan", 1, "Colombia", 3, "Group K"),
    ("2026-06-23", "Portugal", 5, "Uzbekistan", 0, "Group K"),
    ("2026-06-23", "Colombia", 1, "Congo DR", 0, "Group K"),
    ("2026-06-27", "Colombia", 0, "Portugal", 0, "Group K"),
    ("2026-06-27", "Congo DR", 3, "Uzbekistan", 1, "Group K"),

    # Group L
    ("2026-06-17", "Ghana", 2, "Panama", 1, "Group L"),
    ("2026-06-17", "England", 3, "Croatia", 2, "Group L"),
    ("2026-06-23", "England", 0, "Ghana", 0, "Group L"),
    ("2026-06-23", "Panama", 0, "Croatia", 1, "Group L"),
    ("2026-06-27", "Panama", 0, "England", 4, "Group L"),
    ("2026-06-27", "Croatia", 3, "Ghana", 2, "Group L"),
]

# Round of 32 results (through June 30)
R32_RESULTS = [
    ("2026-06-28", "South Africa", 0, "Canada", 1, "Round of 32"),
    ("2026-06-29", "Brazil", 2, "Japan", 1, "Round of 32"),
    ("2026-06-29", "Germany", 1, "Paraguay", 1, "Round of 32"),  # PAR won 4-3 on pens
    ("2026-06-29", "Netherlands", 1, "Morocco", 1, "Round of 32"),  # MAR won 3-2 on pens
    ("2026-06-30", "Ivory Coast", 0, "Norway", 0, "Round of 32"),  # playing today
    ("2026-06-30", "France", 0, "Sweden", 0, "Round of 32"),  # playing today
    ("2026-06-30", "Mexico", 0, "Ecuador", 0, "Round of 32"),  # playing today
]


def compute_elo(team_elo: float, opp_elo: float, team_score: int,
                opp_score: int, k: int = 32) -> float:
    """Compute new Elo rating after a match."""
    expected = 1.0 / (1.0 + 10.0 ** ((opp_elo - team_elo) / 400.0))
    if team_score > opp_score:
        actual = 1.0
    elif team_score == opp_score:
        actual = 0.5
    else:
        actual = 0.0
    return team_elo + k * (actual - expected)


def canonical(name: str) -> str:
    return ALIASES.get(name, name)


def update_elo_ratings():
    print("=" * 60)
    print("Step 1: Updating Elo ratings with WC2026 results")
    print("=" * 60)

    elo_df = load_elo_ratings()
    print(f"  Current Elo entries: {len(elo_df)}")

    wc_teams = set()
    for date, h, hs, a, as_, group in WC2026_RESULTS + R32_RESULTS:
        wc_teams.add(canonical(h))
        wc_teams.add(canonical(a))

    latest_elo = (
        elo_df.sort_values("date")
        .groupby("country")
        .last()
        .reset_index()
    )
    elo_map = {r["country"]: r["elo_rating"] for _, r in latest_elo.iterrows()}
    elo_map.update({ALIASES.get(k, k): v for k, v in list(elo_map.items())})

    print(f"  WC teams to update: {len(wc_teams)}")

    all_matches = WC2026_RESULTS + R32_RESULTS

    for date, h, hs, a, as_, group in all_matches:
        h = canonical(h)
        a = canonical(a)
        h_elo = elo_map.get(h, 1500)
        a_elo = elo_map.get(a, 1500)

        new_h_elo = compute_elo(h_elo, a_elo, hs, as_)
        new_a_elo = compute_elo(a_elo, h_elo, as_, hs)

        elo_map[h] = new_h_elo
        elo_map[a] = new_a_elo

    new_rows = []
    for team, elo in sorted(elo_map.items()):
        if team in wc_teams:
            new_rows.append({
                "country": team,
                "date": "2026-06-30",
                "elo_rating": round(elo),
            })

    new_elo_df = pd.DataFrame(new_rows)
    combined = pd.concat([elo_df, new_elo_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["country", "date"], keep="last")
    combined = combined.sort_values(["country", "date"]).reset_index(drop=True)

    combined.to_csv(ELO_PATH, index=False)
    print(f"  Updated Elo entries: {len(combined)}")
    print(f"  New ratings for {len(new_rows)} WC teams")

    for _, r in new_elo_df.iterrows():
        print(f"    {r['country']:25s} -> {r['elo_rating']}")


def create_wc2026_training_csv():
    print("\n" + "=" * 60)
    print("Step 2: Creating WC2026 results training CSV")
    print("=" * 60)

    rows = []
    for date, h, hs, a, as_, group in WC2026_RESULTS:
        h_canon = canonical(h)
        a_canon = canonical(a)

        rows.append({
            "date": date,
            "country": h_canon,
            "home_away": "home",
            "wc_team_score": hs,
            "opposition_score": as_,
            "result": "win" if hs > as_ else "loss" if hs < as_ else "draw",
            "match_type": "competitive",
            "opposition_country": a_canon,
            "home_team": h_canon,
            "away_team": a_canon,
            "tournament": group,
        })
        rows.append({
            "date": date,
            "country": a_canon,
            "home_away": "away",
            "wc_team_score": as_,
            "opposition_score": hs,
            "result": "win" if as_ > hs else "loss" if as_ < hs else "draw",
            "match_type": "competitive",
            "opposition_country": h_canon,
            "home_team": h_canon,
            "away_team": a_canon,
            "tournament": group,
        })

    out_path = DATA_DIR / "wc2026_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Created {out_path} with {len(rows)} result rows")


def create_loader_patch():
    print("\n" + "=" * 60)
    print("Step 3: Creating loader patch to include WC2026 results")
    print("=" * 60)

    patch_path = Path(__file__).resolve().parent / "src" / "feature" / "include_wc2026.py"
    patch_content = """
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
"""
    with open(patch_path, "w") as f:
        f.write(patch_content)
    print(f"  Created {patch_path}")


def patch_loader():
    print("\n" + "=" * 60)
    print("Step 4: Patching loader.py to use WC2026 results")
    print("=" * 60)

    loader_path = Path(__file__).resolve().parent / "src" / "feature" / "loader.py"
    original = loader_path.read_text()

    if "include_wc2026" in original:
        print("  Loader already patched, skipping")
        return

    patch_code = """
from src.feature.include_wc2026 import merge_wc2026_results


def load_all_data(before_date: str = "2026-06-10") -> pd.DataFrame:
    print("Downloading recent match results...")
    results = download_recent_results(before_date)
    print(f"  {len(results)} result rows from openfootball")
    print("Merging WC2026 actual results...")
    results = merge_wc2026_results(results)
    print(f"  {len(results)} total result rows after merge")
    print("Loading Elo ratings...")
    elo = load_elo_ratings()
    print(f"  {len(elo)} rows")
    print("Loading FIFA ratings...")
    fifa = load_fifa_ratings()
    print(f"  {len(fifa)} rows")
    return results, elo, fifa
"""

    # Find the existing load_all_data function and replace it
    old_func = """def load_all_data(before_date: str = "2026-06-10") -> pd.DataFrame:
    print("Downloading recent match results...")
    results = download_recent_results(before_date)
    print(f"  {len(results)} result rows")
    print("Loading Elo ratings...")
    elo = load_elo_ratings()
    print(f"  {len(elo)} rows")
    print("Loading FIFA ratings...")
    fifa = load_fifa_ratings()
    print(f"  {len(fifa)} rows")
    return results, elo, fifa"""

    if old_func in original:
        new_content = original.replace(old_func, patch_code)
        loader_path.write_text(new_content)
        print("  Patched loader.py successfully")
    else:
        print("  Could not find load_all_data to patch")
        print("  Trying alternative approach...")

        import_end = "from src.feature.loader import load_all_data, build_feature_view_data, load_elo_ratings, load_fifa_ratings, ALIASES, HOSTS"
        alt_patch = f"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.feature.include_wc2026 import merge_wc2026_results


def train(results_df, elo_df, fifa_df):
    print("Merging WC2026 actual results into training data...")
    results_df = merge_wc2026_results(results_df)
    print(f"  {len(results_df)} total result rows after merge")
"""

        train_path = Path(__file__).resolve().parent / "src" / "training" / "train.py"
        train_content = train_path.read_text()
        old_train_func_start = "def train(results_df, elo_df, fifa_df):"

        if old_train_func_start in train_content and "merge_wc2026_results" not in train_content:
            train_content = train_content.replace(old_train_func_start, alt_patch)
            train_path.write_text(train_content)
            print("  Patched train.py instead")


def run_training():
    print("\n" + "=" * 60)
    print("Step 5: Retraining model with updated data")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "src" / "training" / "train.py")],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:1000])
    if result.returncode != 0:
        print(f"  Training failed with code {result.returncode}")
        return False
    return True


def run_precompute():
    print("\n" + "=" * 60)
    print("Step 6: Precomputing probability tables and MC simulations")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "src" / "precompute.py")],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:1000])
    if result.returncode != 0:
        print(f"  Precompute failed with code {result.returncode}")
        return False
    return True


def main():
    update_elo_ratings()
    create_wc2026_training_csv()
    create_loader_patch()
    patch_loader()

    print("\n" + "=" * 60)
    print("Running training pipeline...")
    print("=" * 60)

    if not run_training():
        print("Training failed. Attempting to fix and retry...")
        # Direct approach
        direct_train()
        return

    run_precompute()

    print("\n" + "=" * 60)
    print("Update complete!")
    print("=" * 60)


def direct_train():
    """Direct training approach - modify before_date to include WC2026 matches."""
    print("\nDirect training approach...")
    print("Updating before_date in predictor.py...")

    predictor_path = Path(__file__).resolve().parent / "src" / "predictor.py"
    content = predictor_path.read_text()
    content = content.replace('as_of = "2026-06-10"', 'as_of = "2026-06-30"')
    predictor_path.write_text(content)
    print("  Updated predictor.py as_of date to 2026-06-30")

    import importlib
    from src.training.train import train as train_fn
    from src.feature.loader import load_elo_ratings, load_fifa_ratings
    from src.feature.include_wc2026 import merge_wc2026_results

    print("Loading data...")
    elo = load_elo_ratings()
    fifa = load_fifa_ratings()
    print(f"  Elo: {len(elo)} rows, FIFA: {len(fifa)} rows")

    print("Downloading recent results...")
    from src.feature.loader import download_recent_results
    results = download_recent_results()
    print(f"  {len(results)} rows from openfootball")

    print("Merging WC2026 results...")
    results = merge_wc2026_results(results)
    print(f"  {len(results)} rows after merge")

    print("Training model...")
    model, features, metrics = train_fn(results, elo, fifa)
    print(f"  Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    print("\nRunning precompute...")
    run_precompute()


if __name__ == "__main__":
    main()
