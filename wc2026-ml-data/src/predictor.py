from __future__ import annotations

import itertools
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature.loader import load_elo_ratings, load_fifa_ratings, load_schedule, ALIASES, HOSTS

CLASS_NAMES = ["loss", "draw", "win"]
N_ROLLOUTS = 10000
GROUPS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def load_model():
    p = Path(__file__).resolve().parents[1] / "model" / "wc2026_match_predictor.joblib"
    return joblib.load(p)


def get_ratings():
    elo = load_elo_ratings()
    fifa = load_fifa_ratings()
    as_of = "2026-07-08"
    elo_latest = elo[elo["date"] <= as_of].sort_values("date").groupby("country").last().reset_index()
    fifa_latest = fifa[fifa["date"] <= as_of].sort_values("date").groupby("country").last().reset_index()
    elo_map = {}
    for _, r in elo_latest.iterrows():
        elo_map[ALIASES.get(r["country"], r["country"])] = r["elo_rating"]
    rank_map, pts_map = {}, {}
    for _, r in fifa_latest.iterrows():
        n = ALIASES.get(r["country"], r["country"])
        rank_map[n] = r["ranking"]
        pts_map[n] = r["points"]
    return elo_map, rank_map, pts_map


def get_schedule():
    return load_schedule()


def _norm(n):
    return ALIASES.get(n, n)


def parse_schedule(sched):
    gm = sched[sched["group"].notna() & (sched["group"] != "")].copy()
    gteams = {}
    gfixtures = []
    for _, r in gm.iterrows():
        g = r["group"]
        parts = [_norm(t.strip()) for t in re.split(r"\s+v\s+", r["teams"])]
        gteams.setdefault(g, set()).update(parts)
        gfixtures.append(dict(match=r["match_number"], group=g,
                              home=parts[0], away=parts[1],
                              host_adv=parts[0] if parts[0] in HOSTS else None))
    ko = sched[sched["group"].isna() | (sched["group"] == "")].copy()
    kf = []
    for _, r in ko.iterrows():
        parts = r["teams"].split(" v ")
        kf.append(dict(match=r["match_number"],
                        slot_a=parts[0].strip(), slot_b=parts[1].strip() if len(parts) > 1 else ""))
    return {g: list(ts) for g, ts in gteams.items()}, gfixtures, kf


def predict_proba(model, a, b, elo_map, rank_map, pts_map):
    fa = _feats(a, b, elo_map, rank_map, pts_map)
    fb = _feats(b, a, elo_map, rank_map, pts_map)
    cols = ["home", "is_friendly", "team_elo", "opp_elo", "elo_diff",
            "team_rank", "opp_rank", "rank_diff", "team_points", "opp_points", "points_diff"]
    pa = model.predict_proba(pd.DataFrame([fa], columns=cols))[0]
    pb = model.predict_proba(pd.DataFrame([fb], columns=cols))[0]
    p_w = pa[2]
    p_d = (pa[1] + pb[1]) / 2
    p_l = pb[2]
    s = p_w + p_d + p_l
    return p_w / s, p_d / s, p_l / s


def predict_batch(model, pairs, elo_map, rank_map, pts_map):
    rows = []
    keys = []
    for a, b in pairs:
        rows.append(_feats(a, b, elo_map, rank_map, pts_map))
        keys.append((a, b, 0))
        rows.append(_feats(b, a, elo_map, rank_map, pts_map))
        keys.append((b, a, 1))
    cols = ["home", "is_friendly", "team_elo", "opp_elo", "elo_diff",
            "team_rank", "opp_rank", "rank_diff", "team_points", "opp_points", "points_diff"]
    X = pd.DataFrame(rows, columns=cols)
    probas = model.predict_proba(X)
    result = {}
    for i in range(0, len(keys), 2):
        a, b, _ = keys[i]
        pa = probas[i]
        pb = probas[i + 1]
        p_w = pa[2]
        p_d = (pa[1] + pb[1]) / 2
        p_l = pb[2]
        s = p_w + p_d + p_l
        result[(a, b)] = (p_w / s, p_d / s, p_l / s)
    return result


def _feats(team, opp, elo_map, rank_map, pts_map, home=0):
    te = elo_map.get(team, 1500)
    oe = elo_map.get(opp, 1500)
    return dict(home=home, is_friendly=0, team_elo=te, opp_elo=oe,
                elo_diff=te - oe, team_rank=rank_map.get(team, 100),
                opp_rank=rank_map.get(opp, 100),
                rank_diff=rank_map.get(opp, 100) - rank_map.get(team, 100),
                team_points=pts_map.get(team, 1000), opp_points=pts_map.get(opp, 1000),
                points_diff=pts_map.get(team, 1000) - pts_map.get(opp, 1000))


def _match_outcome(team_a, team_b, proba_table, rng):
    entry = proba_table.get((team_a, team_b))
    if entry is None:
        entry = proba_table.get((team_b, team_a))
        if entry:
            pw, pd_, pl = entry[2], entry[1], entry[0]
        else:
            pw = pd_ = pl = 1 / 3
    else:
        pw, pd_, pl = entry
    r = rng.random()
    if r < pw + 0.5 * pd_:
        return team_a, team_b
    else:
        return team_b, team_a


def simulate_once(group_teams, gfixtures, kf, proba_table, rng):
    gres = {}
    for g, teams in group_teams.items():
        pts = {t: 0 for t in teams}
        gf = {t: 0.0 for t in teams}
        ga = {t: 0.0 for t in teams}
        for f in gfixtures:
            if f["group"] != g:
                continue
            h, a = f["home"], f["away"]
            pw, pd_, pl = proba_table.get((h, a), (1 / 3, 1 / 3, 1 / 3))
            r = rng.random()
            if r < pw:
                pts[h] += 3; gf[h] += 1.5; ga[a] += 0.8
            elif r < pw + pd_:
                pts[h] += 1; pts[a] += 1; gf[h] += 1; ga[h] += 1; gf[a] += 1; ga[a] += 1
            else:
                pts[a] += 3; gf[h] += 0.8; ga[a] += 1.5; gf[a] += 1.5; ga[h] += 0.8
        gd = {t: gf[t] - ga[t] for t in teams}
        rank = sorted(teams, key=lambda t: (-pts[t], -gd[t], gf[t]))
        gres[g] = rank

    third_ranked = []
    for g, rank in gres.items():
        third_team = rank[2]
        third_pts = 0
        third_gf = 0.0
        third_ga = 0.0
        for f in gfixtures:
            if f["group"] != g:
                continue
            h, a = f["home"], f["away"]
            pw, pd_, pl = proba_table.get((h, a), (1 / 3, 1 / 3, 1 / 3))
            r = rng.random()
            if h == third_team or a == third_team:
                is_h = h == third_team
                if r < pw:
                    if is_h: third_gf += 1.5; third_ga += 0.8
                    else: third_ga += 1.5; third_gf += 0.8
                elif r < pw + pd_:
                    third_gf += 1; third_ga += 1
                else:
                    if is_h: third_ga += 1.5; third_gf += 0.8
                    else: third_gf += 1.5; third_ga += 0.8
                if r < pw:
                    if is_h: third_pts += 3
                elif r < pw + pd_:
                    third_pts += 1
                else:
                    if not is_h: third_pts += 3
        third_ranked.append((g, third_team, third_pts, third_gf - third_ga))
    third_ranked.sort(key=lambda x: (-x[2], -x[3]))
    best_thirds = {t for _, t, _, _ in third_ranked[:8]}

    ko_results = {}
    for km in kf:
        def _resolve(slot, ko_res=ko_results, gr=gres, tr=third_ranked, bt=best_thirds):
            s = slot.strip()
            m = re.match(r"Group ([A-L]) (winners|runners[- ]up)", s)
            if m:
                g = f"Group {m.group(1)}"
                return gr[g][0] if m.group(2).startswith("winner") else gr[g][1]
            m = re.match(r"Group ([A-L /]+) third place", s)
            if m:
                allowed = {f"Group {x}" for x in re.findall(r"[A-L]", m.group(1))}
                cand = [t for g, t, _, _ in tr if g in allowed and t in bt]
                return rng.choice(cand) if cand else "?"
            m = re.match(r"Winner match (\d+)", s)
            if m:
                mid = f"Match {m.group(1)}"
                return ko_res.get(mid, ("?", "?"))[0]
            m = re.match(r"Runner-up match (\d+)", s)
            if m:
                mid = f"Match {m.group(1)}"
                return ko_res.get(mid, ("?", "?"))[1]
            return s
        a = _resolve(km["slot_a"])
        b = _resolve(km["slot_b"])
        if a == "?" or b == "?":
            ko_results[km["match"]] = ("?", "?")
        else:
            winner, loser = _match_outcome(a, b, proba_table, rng)
            ko_results[km["match"]] = (winner, loser)

    champ = "?"
    for mid in ["Match 104", "Match 101", "Match 102"]:
        if mid in ko_results:
            champ = ko_results[mid][0]
    if champ == "?" and ko_results:
        champ = list(ko_results.values())[-1][0]

    return gres, ko_results, champ, best_thirds


def run_monte_carlo(group_teams, gfixtures, kf, proba_table, n_rollouts):
    champ_counts = {}
    reach = {}
    all_teams = set(itertools.chain.from_iterable(group_teams.values()))
    for t in all_teams:
        reach[t] = {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "Final": 0, "Win": 0}

    for i in range(n_rollouts):
        rng = np.random.default_rng(42 + i)
        _, ko_results, champ, _ = simulate_once(group_teams, gfixtures, kf, proba_table, rng)
        champ_counts[champ] = champ_counts.get(champ, 0) + 1
        for mid, (winner, loser) in ko_results.items():
            for t in (winner, loser):
                if t == "?":
                    continue
                mn = int(re.search(r"\d+", mid).group())
                reach[t]["R32"] += 1
                if mn >= 89:
                    reach[t]["R16"] += 1
                if mn >= 97:
                    reach[t]["QF"] += 1
                if mn >= 101:
                    reach[t]["SF"] += 1
                if mn in (103, 104):
                    reach[t]["Final"] += 1
                if mid == "Match 104" and t == winner:
                    reach[t]["Win"] += 1

    champ_probs = {t: c / n_rollouts for t, c in champ_counts.items()}
    champ_probs = dict(sorted(champ_probs.items(), key=lambda x: -x[1]))
    reach_probs = {t: {k: v / n_rollouts for k, v in r.items()} for t, r in reach.items()}
    return champ_probs, reach_probs


def build_proba_table(model, group_teams, elo_map, rank_map, pts_map):
    all_teams = sorted(set(itertools.chain.from_iterable(group_teams.values())))
    pairs = [(a, b) for a in all_teams for b in all_teams if a != b]
    return predict_batch(model, pairs, elo_map, rank_map, pts_map)
