"""
Update the predictive model with actual 2026 World Cup match results.
- Updates Elo ratings based on WC2026 group stage + R32 results
- Retrains the XGBoost model  
- Re-runs precompute (probability tables + Monte Carlo)
"""
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.feature.loader import load_elo_ratings, load_fifa_ratings, load_all_data, ALIASES
from src.training.train import train as train_fn
from src.predictor import (
    load_model, get_ratings, get_schedule, parse_schedule,
    build_proba_table, simulate_once, run_monte_carlo, N_ROLLOUTS,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "model"

# ── All verified WC2026 results (group stage + Round of 32 completed) ──

WC2026_RESULTS = [
    # Group A
    ("2026-06-11", "Mexico", 2, "South Africa", 0),
    ("2026-06-11", "Korea Republic", 2, "Czechia", 1),
    ("2026-06-18", "Czechia", 1, "South Africa", 1),
    ("2026-06-18", "Mexico", 1, "Korea Republic", 0),
    ("2026-06-24", "Mexico", 3, "Czechia", 0),
    ("2026-06-24", "South Africa", 1, "Korea Republic", 0),

    # Group B
    ("2026-06-12", "Canada", 1, "Bosnia and Herzegovina", 1),
    ("2026-06-13", "Qatar", 1, "Switzerland", 1),
    ("2026-06-18", "Switzerland", 4, "Bosnia and Herzegovina", 1),
    ("2026-06-18", "Canada", 6, "Qatar", 0),
    ("2026-06-24", "Switzerland", 2, "Canada", 1),
    ("2026-06-24", "Bosnia and Herzegovina", 3, "Qatar", 1),

    # Group C
    ("2026-06-13", "Haiti", 0, "Scotland", 1),
    ("2026-06-13", "Brazil", 1, "Morocco", 1),
    ("2026-06-19", "Brazil", 3, "Haiti", 0),
    ("2026-06-19", "Scotland", 0, "Morocco", 1),
    ("2026-06-24", "Scotland", 0, "Brazil", 3),
    ("2026-06-24", "Morocco", 4, "Haiti", 2),

    # Group D
    ("2026-06-12", "United States", 4, "Paraguay", 1),
    ("2026-06-13", "Australia", 2, "Türkiye", 0),
    ("2026-06-19", "United States", 2, "Australia", 0),
    ("2026-06-19", "Türkiye", 0, "Paraguay", 1),
    ("2026-06-25", "Türkiye", 3, "United States", 2),
    ("2026-06-25", "Paraguay", 0, "Australia", 0),

    # Group E
    ("2026-06-14", "Côte d'Ivoire", 1, "Ecuador", 0),
    ("2026-06-14", "Germany", 7, "Curaçao", 1),
    ("2026-06-20", "Germany", 2, "Côte d'Ivoire", 1),
    ("2026-06-20", "Ecuador", 0, "Curaçao", 0),
    ("2026-06-25", "Ecuador", 2, "Germany", 1),
    ("2026-06-25", "Curaçao", 0, "Côte d'Ivoire", 2),

    # Group F
    ("2026-06-14", "Netherlands", 2, "Japan", 2),
    ("2026-06-14", "Sweden", 5, "Tunisia", 1),
    ("2026-06-20", "Netherlands", 5, "Sweden", 1),
    ("2026-06-20", "Tunisia", 0, "Japan", 4),
    ("2026-06-25", "Japan", 1, "Sweden", 1),
    ("2026-06-25", "Tunisia", 1, "Netherlands", 3),

    # Group G
    ("2026-06-15", "IR Iran", 2, "New Zealand", 2),
    ("2026-06-15", "Belgium", 1, "Egypt", 1),
    ("2026-06-21", "Belgium", 0, "IR Iran", 0),
    ("2026-06-21", "New Zealand", 1, "Egypt", 3),
    ("2026-06-26", "Egypt", 1, "IR Iran", 1),
    ("2026-06-26", "New Zealand", 1, "Belgium", 5),

    # Group H
    ("2026-06-15", "Saudi Arabia", 1, "Uruguay", 1),
    ("2026-06-15", "Spain", 0, "Cabo Verde", 0),
    ("2026-06-21", "Uruguay", 2, "Cabo Verde", 2),
    ("2026-06-21", "Spain", 4, "Saudi Arabia", 0),
    ("2026-06-26", "Cabo Verde", 0, "Saudi Arabia", 0),
    ("2026-06-26", "Uruguay", 0, "Spain", 1),

    # Group I
    ("2026-06-16", "France", 3, "Senegal", 1),
    ("2026-06-16", "Iraq", 1, "Norway", 4),
    ("2026-06-22", "France", 3, "Iraq", 0),
    ("2026-06-22", "Norway", 3, "Senegal", 2),
    ("2026-06-26", "Norway", 1, "France", 4),
    ("2026-06-26", "Senegal", 5, "Iraq", 0),

    # Group J
    ("2026-06-16", "Argentina", 3, "Algeria", 0),
    ("2026-06-16", "Austria", 3, "Jordan", 1),
    ("2026-06-22", "Argentina", 2, "Austria", 0),
    ("2026-06-22", "Jordan", 1, "Algeria", 2),
    ("2026-06-27", "Algeria", 3, "Austria", 3),
    ("2026-06-27", "Jordan", 1, "Argentina", 3),

    # Group K
    ("2026-06-17", "Portugal", 1, "Congo DR", 1),
    ("2026-06-17", "Uzbekistan", 1, "Colombia", 3),
    ("2026-06-23", "Portugal", 5, "Uzbekistan", 0),
    ("2026-06-23", "Colombia", 1, "Congo DR", 0),
    ("2026-06-27", "Colombia", 0, "Portugal", 0),
    ("2026-06-27", "Congo DR", 3, "Uzbekistan", 1),

    # Group L
    ("2026-06-17", "Ghana", 2, "Panama", 1),
    ("2026-06-17", "England", 3, "Croatia", 2),
    ("2026-06-23", "England", 0, "Ghana", 0),
    ("2026-06-23", "Panama", 0, "Croatia", 1),
    ("2026-06-27", "Panama", 0, "England", 4),
    ("2026-06-27", "Croatia", 3, "Ghana", 2),
]

R32_RESULTS = [
    ("2026-06-28", "South Africa", 0, "Canada", 1),                       # Match 73
    ("2026-06-29", "Germany", 1, "Paraguay", 1),                          # Match 74 (PAR 4-3 pens)
    ("2026-06-29", "Netherlands", 1, "Morocco", 1),                       # Match 75 (MAR 3-2 pens)
    ("2026-06-29", "Brazil", 2, "Japan", 1),                              # Match 76
    ("2026-06-30", "France", 3, "Sweden", 0),                             # Match 77
    ("2026-06-30", "Ivory Coast", 1, "Norway", 2),                        # Match 78
    ("2026-06-30", "Mexico", 2, "Ecuador", 0),                            # Match 79
    ("2026-07-01", "England", 2, "DR Congo", 1),                          # Match 80
    ("2026-07-01", "United States", 2, "Bosnia and Herzegovina", 0),      # Match 81
    ("2026-07-01", "Belgium", 3, "Senegal", 2),                           # Match 82 (AET)
    ("2026-07-02", "Portugal", 2, "Croatia", 1),                          # Match 83
    ("2026-07-02", "Spain", 3, "Austria", 0),                             # Match 84
    ("2026-07-02", "Switzerland", 2, "Algeria", 0),                       # Match 85
    ("2026-07-03", "Argentina", 3, "Cape Verde", 2),                      # Match 86 (AET)
    ("2026-07-03", "Colombia", 1, "Ghana", 0),                            # Match 87
    ("2026-07-03", "Australia", 1, "Egypt", 1),                           # Match 88 (EGY 4-2 pens)
]

# Round of 16 matches (completed after July 4)
R16_RESULTS = [
    ("2026-07-04", "France", 1, "Paraguay", 0),                             # Match 89
    ("2026-07-04", "Morocco", 3, "Canada", 0),                               # Match 90
    ("2026-07-05", "Norway", 2, "Brazil", 1),                                # Match 91
    ("2026-07-05", "England", 3, "Mexico", 2),                               # Match 92
    ("2026-07-06", "Spain", 1, "Portugal", 0),                               # Match 93
    ("2026-07-06", "Belgium", 4, "United States", 1),                        # Match 94
    ("2026-07-07", "Argentina", 3, "Egypt", 2),                              # Match 95
    ("2026-07-07", "Switzerland", 0, "Colombia", 0),                         # Match 96 (SUI 4-3 pens)
]


def compute_elo(team_elo: float, opp_elo: float, team_score: int,
                opp_score: int, k: int = 32) -> float:
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
    print("Updating Elo ratings with WC2026 results")
    print("=" * 60)

    elo_df = load_elo_ratings()
    print(f"  Current Elo entries: {len(elo_df)}")

    latest_elo = (
        elo_df.sort_values("date")
        .groupby("country")
        .last()
        .reset_index()
    )
    elo_map = {}
    for _, r in latest_elo.iterrows():
        elo_map[r["country"]] = r["elo_rating"]
        elo_map[ALIASES.get(r["country"], r["country"])] = r["elo_rating"]

    all_matches = WC2026_RESULTS + R32_RESULTS + R16_RESULTS

    for date, h, hs, a, as_ in all_matches:
        h_c = canonical(h)
        a_c = canonical(a)
        h_elo = elo_map.get(h_c, 1500)
        a_elo = elo_map.get(a_c, 1500)

        new_h_elo = compute_elo(h_elo, a_elo, hs, as_)
        new_a_elo = compute_elo(a_elo, h_elo, as_, hs)

        elo_map[h_c] = new_h_elo
        elo_map[a_c] = new_a_elo

    wc_teams = set()
    for date, h, hs, a, as_ in all_matches:
        wc_teams.add(canonical(h))
        wc_teams.add(canonical(a))

    new_rows = []
    for team in sorted(wc_teams):
        new_rows.append({
            "country": team,
            "date": "2026-07-04",
            "elo_rating": round(elo_map.get(team, 1500)),
        })

    new_elo_df = pd.DataFrame(new_rows)
    combined = pd.concat([elo_df, new_elo_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["country", "date"], keep="last")
    combined = combined.sort_values(["country", "date"]).reset_index(drop=True)

    combined.to_csv(DATA_DIR / "elo_ratings.csv", index=False)
    print(f"  Saved {len(combined)} Elo entries")
    print("  New ratings for WC teams (as of June 30):")
    for _, r in new_elo_df.iterrows():
        print(f"    {r['country']:25s} {r['elo_rating']}")


def create_wc2026_training_data():
    print("\n" + "=" * 60)
    print("Creating WC2026 results training CSV")
    print("=" * 60)

    rows = []
    for date, h, hs, a, as_ in WC2026_RESULTS + R32_RESULTS:
        h_c = canonical(h)
        a_c = canonical(a)
        for team, score, opp_score, ha in [(h_c, hs, as_, "home"), (a_c, as_, hs, "away")]:
            opp = a_c if ha == "home" else h_c
            result = "win" if score > opp_score else "loss" if score < opp_score else "draw"
            rows.append({
                "date": date,
                "country": team,
                "home_away": ha,
                "wc_team_score": score,
                "opposition_score": opp_score,
                "result": result,
                "match_type": "competitive",
                "opposition_country": opp,
                "home_team": h_c,
                "away_team": a_c,
                "tournament": "FIFA World Cup 2026",
            })

    out_path = DATA_DIR / "wc2026_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Created {out_path} with {len(rows)} rows")

    return pd.DataFrame(rows)


def train_with_wc2026(wc_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Training model with historical + WC2026 data")
    print("=" * 60)

    elo = load_elo_ratings()
    fifa = load_fifa_ratings()
    print(f"  Elo: {len(elo)} rows, FIFA: {len(fifa)} rows")

    print("  Downloading historical data from openfootball...")
    from src.feature.loader import download_recent_results
    historical = download_recent_results(before_date=None)
    print(f"  Historical: {len(historical)} rows")

    combined = pd.concat([historical, wc_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "country", "opposition_country"], keep="last"
    )
    combined["date_sort"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["country", "date_sort"], ascending=[True, False])
    combined = combined.groupby("country", group_keys=False).head(20)
    combined = combined.drop(columns=["date_sort"]).reset_index(drop=True)
    print(f"  Combined: {len(combined)} rows (top 20 per team)")

    print("  Training XGBoost model...")
    model, features, metrics = train_fn(combined, elo, fifa)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")

    return model, features, metrics


def update_predictor_as_of():
    """Update the as_of date in predictor.py to use latest ratings."""
    pred_path = Path(__file__).resolve().parent / "src" / "predictor.py"
    content = pred_path.read_text()
    content = content.replace('as_of = "2026-06-10"', 'as_of = "2026-07-08"')
    content = content.replace('as_of = "2026-06-30"', 'as_of = "2026-07-08"')
    content = content.replace('as_of = "2026-07-04"', 'as_of = "2026-07-08"')
    pred_path.write_text(content)
    print("  Updated predictor.py as_of date to 2026-07-08")


def run_precompute():
    print("\n" + "=" * 60)
    print("Precomputing probability tables and MC simulations")
    print("=" * 60)

    model = load_model()
    elo_map, rank_map, pts_map = get_ratings()
    sched = get_schedule()
    group_teams, gfixtures, kf = parse_schedule(sched)
    all_teams = sorted(set(
        t for teams in group_teams.values() for t in teams
    ))

    print(f"  Building {len(all_teams) * (len(all_teams) - 1):,} pair probabilities...")
    proba = build_proba_table(model, group_teams, elo_map, rank_map, pts_map)

    with open(MODEL_DIR / "proba_table.pkl", "wb") as f:
        pickle.dump(proba, f)
    print(f"  Saved proba_table.pkl ({len(proba)} entries)")

    print(f"  Running {N_ROLLOUTS:,} MC simulations...")
    champ_probs, reach_probs = run_monte_carlo(group_teams, gfixtures, kf, proba, N_ROLLOUTS)
    with open(MODEL_DIR / "mc_results.pkl", "wb") as f:
        pickle.dump({"champ_probs": champ_probs, "reach_probs": reach_probs}, f)
    print(f"  Saved mc_results.pkl ({len(champ_probs)} unique champions)")

    print("  Computing group phase probabilities...")
    group_data = {}
    for g, teams in group_teams.items():
        pos_counts = {t: [0, 0, 0, 0] for t in teams}
        for i in range(2000):
            rng = np.random.default_rng(999 + i)
            gres, _, _, _ = simulate_once(group_teams, gfixtures, kf, proba, rng)
            rank = gres[g]
            for pos, t in enumerate(rank):
                pos_counts[t][pos] += 1
        group_data[g] = {t: [c / 20 for c in counts] for t, counts in pos_counts.items()}
    with open(MODEL_DIR / "group_results.pkl", "wb") as f:
        pickle.dump(group_data, f)
    print("  Saved group_results.pkl")

    print("\nTop 10 champion probabilities:")
    for team, prob in list(champ_probs.items())[:10]:
        print(f"  {team:20s} {prob*100:5.2f}%")


def main():
    # Step 1: Update Elo ratings based on actual results
    update_elo_ratings()

    # Step 2: Create WC2026 training data
    wc_df = create_wc2026_training_data()

    # Step 3: Update predictor to use latest ratings
    update_predictor_as_of()

    # Step 4: Train model with merged data
    train_with_wc2026(wc_df)

    # Step 5: Precompute
    run_precompute()

    print("\n" + "=" * 60)
    print("Done! Model updated with WC2026 results.")
    print("Run `streamlit run src/app/app.py` to view.")
    print("=" * 60)


if __name__ == "__main__":
    main()
