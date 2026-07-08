from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import itertools
from src.predictor import (
    load_model, get_ratings, get_schedule, parse_schedule,
    predict_proba, simulate_once, run_monte_carlo
)
from src.app.app import draw_bracket


def test_all():
    print("Loading model...")
    model = load_model()
    print("  OK")

    print("Loading ratings...")
    elo_map, rank_map, pts_map = get_ratings()
    print(f"  {len(elo_map)} teams with Elo")

    print("Loading schedule...")
    sched = get_schedule()
    group_teams, gfixtures, kf = parse_schedule(sched)
    print(f"  {len(group_teams)} groups, {len(gfixtures)} group fixtures, {len(kf)} KO fixtures")

    all_teams = sorted(set(itertools.chain.from_iterable(group_teams.values())))
    print(f"  {len(all_teams)} teams total")
    assert len(all_teams) == 48, f"Expected 48 teams, got {len(all_teams)}"

    print("Building probability table...")
    proba = {}
    for a, b in itertools.permutations(all_teams, 2):
        proba[(a, b)] = predict_proba(model, a, b, elo_map, rank_map, pts_map)
    print(f"  {len(proba)} ordered pairs computed")

    # check probabilities sum to 1
    for (a, b), (pw, pd_, pl) in list(proba.items())[:10]:
        assert abs(pw + pd_ + pl - 1.0) < 0.01, f"Probs don't sum to 1: {pw}+{pd_}+{pl}"

    print("Testing single simulation...")
    rng = np.random.default_rng(42)
    gres, ko_res, champ, best_thirds = simulate_once(group_teams, gfixtures, kf, proba, rng)
    for g, rank in gres.items():
        assert len(rank) == 4, f"Group {g} has {len(rank)} teams (expected 4)"
    print(f"  Groups OK, champion: {champ}")

    # check bracket
    mid_nums = sorted(int(k.split()[-1]) for k in ko_res.keys())
    for mn in [73, 89, 97, 101, 104]:
        assert mn in mid_nums, f"Missing knockout match {mn}"
    print(f"  {len(ko_res)} KO matches resolved")

    print("Testing Monte Carlo (100 rollouts)...")
    champ_probs, reach_probs = run_monte_carlo(group_teams, gfixtures, kf, proba, 100)
    total_champ_p = sum(champ_probs.values())
    assert abs(total_champ_p - 1.0) < 0.01, f"Champion probs sum to {total_champ_p}"
    print(f"  {len(champ_probs)} unique champions, total prob={total_champ_p:.2f}")

    for t, r in reach_probs.items():
        assert r["R32"] >= r["R16"] >= r["QF"] >= r["SF"] >= r["Final"] >= r["Win"]
    print("  Reach probabilities are monotonic (OK)")

    print("Drawing bracket...")
    fig = draw_bracket(ko_res, champ, group_teams, "Test Bracket")
    print("  Bracket rendered to figure")

    print("\n All tests passed!")


if __name__ == "__main__":
    test_all()
