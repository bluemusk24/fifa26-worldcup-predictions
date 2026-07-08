from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.predictor import (
    load_model, get_ratings, get_schedule, parse_schedule,
    build_proba_table, simulate_once, run_monte_carlo,
    N_ROLLOUTS,
)

model_dir = Path(__file__).resolve().parents[1] / "model"

print("Loading model and data...")
model = load_model()
elo_map, rank_map, pts_map = get_ratings()
sched = get_schedule()
group_teams, gfixtures, kf = parse_schedule(sched)

all_teams = sorted(set(itertools.chain.from_iterable(group_teams.values())))
n_pairs = len(all_teams) * (len(all_teams) - 1)
print(f"Building probability table for {n_pairs:,} ordered pairs ({len(all_teams)} teams)...")
proba = build_proba_table(model, group_teams, elo_map, rank_map, pts_map)

with open(model_dir / "proba_table.pkl", "wb") as f:
    pickle.dump(proba, f)
print(f"Saved -> model/proba_table.pkl ({len(proba)} entries)")

print(f"Running {N_ROLLOUTS:,} Monte-Carlo simulations...")
champ_probs, reach_probs = run_monte_carlo(group_teams, gfixtures, kf, proba, N_ROLLOUTS)
with open(model_dir / "mc_results.pkl", "wb") as f:
    pickle.dump({"champ_probs": champ_probs, "reach_probs": reach_probs}, f)
print(f"Saved -> model/mc_results.pkl ({len(champ_probs)} unique champions)")

print("Computing group phase probabilities (2000 sims per group)...")
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
with open(model_dir / "group_results.pkl", "wb") as f:
    pickle.dump(group_data, f)
print(f"Saved -> model/group_results.pkl")

print("\nDone! All precomputed files ready.")
