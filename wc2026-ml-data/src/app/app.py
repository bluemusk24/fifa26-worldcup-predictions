from __future__ import annotations

import itertools
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.predictor import (
    load_model, get_ratings, get_schedule, parse_schedule,
    predict_proba, simulate_once, ACTUAL_BRACKET,
    CLASS_NAMES, GROUPS,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"

# Actual QF results for display
ACTUAL_KO = {
    "Match 73": ("Canada", "South Africa"), "Match 74": ("Brazil", "Japan"),
    "Match 75": ("Morocco", "Netherlands"), "Match 76": ("Paraguay", "Germany"),
    "Match 77": ("Norway", "Ivory Coast"), "Match 78": ("France", "Sweden"),
    "Match 79": ("Mexico", "Ecuador"), "Match 80": ("England", "DR Congo"),
    "Match 81": ("Belgium", "Senegal"), "Match 82": ("United States", "Bosnia and Herzegovina"),
    "Match 83": ("Croatia", "Portugal"), "Match 84": ("Spain", "Austria"),
    "Match 85": ("Switzerland", "Algeria"), "Match 86": ("Argentina", "Cape Verde"),
    "Match 87": ("Colombia", "Ghana"), "Match 88": ("Egypt", "Australia"),
    "Match 89": ("Morocco", "Canada"), "Match 90": ("France", "Paraguay"),
    "Match 91": ("Norway", "Brazil"), "Match 92": ("England", "Mexico"),
    "Match 93": ("Spain", "Portugal"), "Match 94": ("Belgium", "United States"),
    "Match 95": ("Argentina", "Egypt"), "Match 96": ("Switzerland", "Colombia"),
    "Match 97": ("France", "Morocco"), "Match 98": ("Spain", "Belgium"),
    "Match 99": ("England", "Norway"), "Match 100": ("Argentina", "Switzerland"),
}

st.set_page_config(page_title="FIFA WC 2026 Predictor", layout="wide", page_icon="")

# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------

@st.cache_resource
def load_precomputed():
    data = {}
    for name in ["proba_table", "mc_results", "group_results"]:
        p = MODEL_DIR / f"{name}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                data[name] = pickle.load(f)
    return data


@st.cache_resource
def load_model_and_schedule():
    model = load_model()
    elo_map, rank_map, pts_map = get_ratings()
    sched = get_schedule()
    group_teams, gfixtures, kf = parse_schedule(sched)
    return model, elo_map, rank_map, pts_map, group_teams, gfixtures, kf

# ---------------------------------------------------------------------------
# Bracket drawing
# ---------------------------------------------------------------------------

def draw_bracket(ko_results, champion, group_results, title):
    fig, ax = plt.subplots(figsize=(22, 30))
    fig.patch.set_facecolor("#0a0e27")
    ax.set_facecolor("#0a0e27")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 30)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color="#f0f0f0")

    def draw_box(x, y, ta, tb, winner, w=3.5, h=0.7):
        a_w = ta == winner
        b_w = tb == winner
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                              facecolor="#12163a", edgecolor="#3a3a7a", linewidth=0.8))
        for i, (tm, is_w) in enumerate([(ta, a_w), (tb, b_w)]):
            bg = "#1a3a2a" if is_w else "#1a1a2e"
            bd = "#4caf50" if is_w else "#3a3a6a"
            c = "#4caf50" if is_w else "#c0c0e0"
            ax.add_patch(mpatches.FancyBboxPatch((x + 0.05, y + 0.02 + i * 0.33), w - 0.1, 0.3,
                                                  boxstyle="round,pad=0.02", facecolor=bg,
                                                  edgecolor=bd, linewidth=0.5 if is_w else 0.3))
            ax.text(x + 0.2, y + 0.2 + i * 0.33, tm, fontsize=6.5,
                    fontweight="bold" if is_w else "normal", color=c, va="center", ha="left")

    mid_info = {}
    for mid, (winner, loser) in ko_results.items():
        mid_info[mid] = (winner, loser)

    def y_pos(m):
        n = int(re.search(r"\d+", m).group())
        if n <= 88: return 13.7 - (n - 73) * 0.7
        if n <= 96: return 13.0 - (n - 89) * 1.4
        if n <= 100: return 12.0 - (n - 97) * 2.8
        if n <= 102: return 10.0 - (n - 101) * 5.6
        if n == 103: return 2.0
        return 6.0

    def x_col(n):
        if n <= 88: return 0.5
        if n <= 96: return 5.0
        if n <= 100: return 9.5
        if n <= 102: return 14.0
        return 18.5

    labels = {73: "Round of 32", 89: "Round of 16", 97: "Quarter-finals",
              101: "Semi-finals", 104: "FINAL"}
    for n, lbl in labels.items():
        ax.text(x_col(n) + 1.75, 27.5, lbl, fontsize=10, fontweight="bold",
                ha="center", color="#9a9ac0")

    for mid, (winner, loser) in sorted(mid_info.items(),
                                         key=lambda kv: int(re.search(r"\d+", kv[0]).group())):
        n = int(re.search(r"\d+", mid).group())
        if n == 103: continue
        x, y = x_col(n), y_pos(mid)
        draw_box(x, y, winner if winner != "?" else "TBD", loser if loser != "?" else "TBD",
                 winner if winner != "?" else "TBD")

    if "Match 103" in mid_info:
        w, l = mid_info["Match 103"]
        ax.text(13, 3.5, "Third Place", fontsize=8, fontstyle="italic", ha="center", color="#7a7aaa")
        draw_box(13, 2.0, w if w != "?" else "TBD", l if l != "?" else "TBD",
                 w if w != "?" else "TBD")

    ax.add_patch(mpatches.FancyBboxPatch((18.5, 6.3), 3.5, 0.9, boxstyle="round,pad=0.1",
                                          facecolor="#ffd700", edgecolor="#b8860b", linewidth=2))
    ax.text(20.25, 6.75, f"\u2605 {champion} \u2605", fontsize=10, fontweight="bold",
            color="#8B4513", ha="center", va="center")
    ax.text(20.25, 8.5, "CHAMPION", fontsize=8, fontweight="bold", ha="center", color="#ffd700")
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

L = "#1a1f4e"  # card bg
B = "#3a3a7a"  # border
T = "#e0e0e0"  # text
M = "#6c63ff"  # primary
G = "#4caf50"  # green

def card(inner, extra=""):
    return f'<div style="background:{L};border:1px solid {B};border-radius:12px;padding:16px;{extra}">{inner}</div>'

def section(title, content, col=None):
    html = f"<h3 style='color:#ffffff;margin:0 0 12px 0'>{title}</h3>{content}"
    return html

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.title("  FIFA World Cup 2026")
    st.markdown(f"<p style='color:{B};margin-top:-10px;font-size:16px'>Match Predictor & Tournament Simulator</p>", unsafe_allow_html=True)

    with st.spinner("Loading model and data..."):
        model, elo_map, rank_map, pts_map, group_teams, gfixtures, kf = \
            load_model_and_schedule()
        pre = load_precomputed()

    all_teams = sorted(set(itertools.chain.from_iterable(group_teams.values())))

    # Betting session state
    if "wallet" not in st.session_state:
        st.session_state.wallet = 10000.0
    if "bets" not in st.session_state:
        st.session_state.bets = []
    if "bet_id" not in st.session_state:
        st.session_state.bet_id = 0
    if "settled_up_to" not in st.session_state:
        st.session_state.settled_up_to = 0

    proba_table = pre.get("proba_table")
    mc_data = pre.get("mc_results")
    group_data = pre.get("group_results")

    if proba_table is None:
        from src.predictor import build_proba_table
        proba_table = build_proba_table(model, group_teams, elo_map, rank_map, pts_map)
    if mc_data is None:
        from src.predictor import run_monte_carlo
        with st.spinner("Running Monte Carlo..."):
            champ_probs, reach_probs = run_monte_carlo(group_teams, gfixtures, kf, proba_table, 10000)
    else:
        champ_probs = mc_data["champ_probs"]
        reach_probs = mc_data["reach_probs"]
    if group_data is None:
        group_data = {}
        for g, teams in group_teams.items():
            pos_counts = {t: [0, 0, 0, 0] for t in teams}
            for i in range(500):
                rng = np.random.default_rng(999 + i)
                gres, _, _, _ = simulate_once(group_teams, gfixtures, kf, proba_table, rng)
                rank = gres[g]
                for pos, t in enumerate(rank):
                    pos_counts[t][pos] += 1
            group_data[g] = {t: [c / 5 for c in counts] for t, counts in pos_counts.items()}

    # Load model metrics from metadata
    meta_path = MODEL_DIR / "model_metadata.json"
    if meta_path.exists():
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        acc = meta.get("metrics", {}).get("accuracy", 0.51)
        auc = meta.get("metrics", {}).get("roc_auc", 0.68)
    else:
        acc, auc = 0.51, 0.68

    tab1, tab2, tab3 = st.tabs([" Tournament Simulation", " Match Odds & Betting", " Virtual Betting"])

    # ============ TAB 1 ============
    with tab1:
        st.markdown(card(
            f"<span style='color:#9a9ac0'>{len(all_teams)} Teams  | 12 Groups  | "
            f"Model: XGBoost (Accuracy {acc*100:.0f}%, ROC-AUC {auc:.2f})</span>",
            extra="margin-bottom:20px"
        ), unsafe_allow_html=True)

        col_l, col_m, col_r = st.columns([1, 1, 1.5])

        with col_l:
            st.markdown("<h3 style='color:#ffffff'> Champion Probabilities</h3>", unsafe_allow_html=True)
            for rank, (team, p) in enumerate(list(champ_probs.items())[:15], 1):
                bw = int(p * 40)
                bar = "" + "\u2588" * bw
                st.markdown(
                    f"<div style='display:flex;align-items:center;margin:2px 0;padding:4px 10px;"
                    f"background:#12163a;border-radius:8px;border:1px solid #2a2a5a'>"
                    f"<span style='width:28px;color:#7a7aaa'>{rank}.</span>"
                    f"<span style='width:180px;color:#e0e0e0'>{team}</span>"
                    f"<span style='color:{M};font-weight:bold'>{p*100:.1f}%</span>"
                    f"<span style='flex:1;text-align:right;color:#3a3a6a;font-size:11px'>{bar}</span>"
                    f"</div>", unsafe_allow_html=True)

        with col_m:
            st.markdown("<h3 style='color:#ffffff'> Group Stage</h3>", unsafe_allow_html=True)
            g_sel = st.selectbox("Group", list(GROUPS), key="g_sel_t", label_visibility="collapsed")
            sel_g = f"Group {g_sel}"
            if sel_g in group_data:
                rows = []
                for t in group_teams.get(sel_g, []):
                    c = group_data[sel_g].get(t, [0, 0, 0, 0])
                    rows.append({"Team": t, "1st": f"{c[0]:.0f}%", "2nd": f"{c[1]:.0f}%",
                                 "3rd": f"{c[2]:.0f}%", "4th": f"{c[3]:.0f}%"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        with col_r:
            st.markdown("<h3 style='color:#ffffff'> Route to the Final</h3>", unsafe_allow_html=True)
            top_t = [t for t, _ in list(champ_probs.items())[:10]]
            rows = []
            for t in top_t:
                r = reach_probs.get(t, {})
                rows.append({"Team": t, "R32": f"{r.get('R32',0)*100:.0f}%",
                             "R16": f"{r.get('R16',0)*100:.0f}%",
                             "QF": f"{r.get('QF',0)*100:.0f}%",
                             "SF": f"{r.get('SF',0)*100:.0f}%",
                             "Final": f"{r.get('Final',0)*100:.0f}%",
                             "Win": f"{r.get('Win',0)*100:.0f}%"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.divider()
        view = st.radio("Bracket view", ["Most Likely (Chalk)", "Random Simulation"],
                        horizontal=True, key="bv")

        if view == "Most Likely (Chalk)":
            if "chalk" not in st.session_state:
                rng = np.random.default_rng(0)
                gres, ko_res, champ, _ = simulate_once(group_teams, gfixtures, kf, proba_table, rng)
                st.session_state["chalk"] = (ko_res, champ)
            ko_res, champ = st.session_state["chalk"]
            merged_ko = dict(ACTUAL_KO)
            merged_ko.update(ko_res)
            ko_res = merged_ko
            if "Match 104" in ACTUAL_KO:
                champ = ACTUAL_KO["Match 104"][0]
            title = f"Tournament Bracket \u2014 \u2605 {champ} Champion"
        else:
            key = "rseed"
            if key not in st.session_state:
                st.session_state[key] = np.random.default_rng().integers(0, 100000)
            _, ko_res, champ, _ = simulate_once(group_teams, gfixtures, kf, proba_table,
                                                np.random.default_rng(st.session_state[key]))
            merged_ko = dict(ACTUAL_KO)
            merged_ko.update(ko_res)
            ko_res = merged_ko
            if "Match 104" in ACTUAL_KO:
                champ = ACTUAL_KO["Match 104"][0]
            title = f"Random Simulation \u2014 \u2605 {champ} Champion"

        fig = draw_bracket(ko_res, champ, group_teams, title)
        st.pyplot(fig)
        plt.close(fig)

        if st.button(" Run New Simulation"):
            for k in list(st.session_state.keys()):
                if k in ("chalk", "rseed", "mc_results"):
                    del st.session_state[k]
            st.rerun()

    # ============ TAB 2 ============
    with tab2:
        st.markdown("<h2 style='color:#ffffff'> Head-to-Head Match Odds</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#7a7aaa'>Predictions based on Elo ratings, FIFA rankings & historical results</p>",
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 0.5, 1])
        with c1:
            ta = st.selectbox("Home Team", all_teams,
                              index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)
        with c2:
            st.markdown(f"<div style='text-align:center;padding-top:28px;font-size:28px;font-weight:bold;color:{M}'>VS</div>",
                        unsafe_allow_html=True)
        with c3:
            tb = st.selectbox("Away Team", all_teams,
                              index=all_teams.index("Argentina") if "Argentina" in all_teams else 1)

        if ta != tb:
            pw, pd_, pl = proba_table.get((ta, tb), predict_proba(model, ta, tb, elo_map, rank_map, pts_map))
            ow, od, ol = 1 / pw, 1 / pd_, 1 / pl
            mx = max(pw, pd_, pl)

            html = f"<h3 style='text-align:center;color:#ffffff;margin:16px 0'>{ta} vs {tb}</h3>"
            for label, p, odds, color in [
                (f"{ta} Win", pw, ow, G),
                ("Draw", pd_, od, "#ff9800"),
                (f"{tb} Win", pl, ol, "#f44336"),
            ]:
                bp = p / mx * 100 if mx > 0 else 0
                html += f"""
                <div style="margin:12px 0">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="font-weight:600;color:{T}">{label}</span>
                        <span style="color:{color};font-weight:700">{odds:.2f}</span>
                    </div>
                    <div style="background:#0a0e27;border-radius:8px;height:36px;overflow:hidden;border:1px solid #2a2a5a">
                        <div style="width:{bp}%;background:linear-gradient(90deg,{color}66,{color});height:36px;border-radius:8px;
                                   display:flex;align-items:center;padding-left:12px;font-weight:bold;font-size:15px;color:#fff">
                            {p*100:.1f}%
                        </div>
                    </div>
                </div>"""
            st.markdown(card(html), unsafe_allow_html=True)

            st.markdown("<h3 style='color:#ffffff;margin-top:20px'> Betting Slip</h3>", unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric(f"{ta} Win", f"{ow:.2f}")
            with sc2:
                st.metric("Draw", f"{od:.2f}")
            with sc3:
                st.metric(f"{tb} Win", f"{ol:.2f}")

            st.markdown(card(
                "<span style='color:#7a7aaa'>\u2022 Decimal odds: bet $10, win $10 \u00d7 odds<br>"
                "\u2022 Higher odds = less likely / bigger payout</span>",
                extra="margin-top:16px"
            ), unsafe_allow_html=True)

    # ============ TAB 3 ============
    with tab3:
        st.markdown("<h2 style='color:#ffffff'> Virtual Betting</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#7a7aaa'>Place mock bets using model-generated odds. Each visitor starts with $10,000.</p>",
                    unsafe_allow_html=True)

        bal = st.session_state.wallet
        total_staked = sum(b["stake"] for b in st.session_state.bets if not b["settled"])
        total_pnl = sum(b.get("pnl", 0) for b in st.session_state.bets if b["settled"])
        wins = sum(1 for b in st.session_state.bets if b.get("pnl", 0) > 0)
        losses = sum(1 for b in st.session_state.bets if b.get("pnl", 0) < 0)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Balance", f"${bal:,.2f}", delta=f"${total_pnl:+,.2f}" if total_pnl != 0 else None)
        with mc2:
            st.metric("Active Bets", sum(1 for b in st.session_state.bets if not b["settled"]))
        with mc3:
            st.metric("Settled", f"{wins + losses} ({wins}W / {losses}L)")
        with mc4:
            if wins + losses > 0:
                roi = total_pnl / sum(b["stake"] for b in st.session_state.bets if b["settled"]) * 100
                st.metric("ROI", f"{roi:+.1f}%")
            else:
                st.metric("ROI", "0.0%")

        st.divider()

        # ---- Place bets ----
        st.markdown("<h3 style='color:#ffffff'> Place Bets</h3>", unsafe_allow_html=True)
        bg = st.selectbox("Group", [f"Group {g}" for g in GROUPS], label_visibility="collapsed")
        bg_fixtures = [f for f in gfixtures if f["group"] == bg]

        for f in bg_fixtures:
            h, a = f["home"], f["away"]
            pw, pd_, pl = proba_table.get((h, a), (1/3, 1/3, 1/3))
            ow, od, ol = 1 / pw, 1 / pd_, 1 / pl
            cols = st.columns([2, 1, 1, 1, 1.5])
            with cols[0]:
                st.markdown(f"<div style='padding:6px 0'><b>{h}</b> vs <b>{a}</b></div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<div style='text-align:center;padding:6px 0'><span style='color:{G}'>{ow:.2f}</span></div>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<div style='text-align:center;padding:6px 0'><span style='color:#ff9800'>{od:.2f}</span></div>", unsafe_allow_html=True)
            with cols[3]:
                st.markdown(f"<div style='text-align:center;padding:6px 0'><span style='color:#f44336'>{ol:.2f}</span></div>", unsafe_allow_html=True)
            with cols[4]:
                pick = st.selectbox("Pick", ["", f"{h} (Win)", "Draw", f"{a} (Win)"],
                                    key=f"pk_{f['match']}", label_visibility="collapsed")
                if pick:
                    stake = st.number_input("$", min_value=1.0, max_value=min(bal, 5000.0),
                                            value=10.0, key=f"st_{f['match']}", label_visibility="collapsed",
                                            step=5.0)
                    if pick == f"{h} (Win)":
                        odds = ow
                    elif pick == "Draw":
                        odds = od
                    else:
                        odds = ol
                    payout = stake * odds
                    if st.button("Place", key=f"bt_{f['match']}", type="primary", use_container_width=True):
                        if stake <= st.session_state.wallet:
                            st.session_state.bet_id += 1
                            st.session_state.bets.append(dict(
                                id=st.session_state.bet_id,
                                match=f["match"],
                                home=h, away=a,
                                pick=pick,
                                odds=odds,
                                stake=stake,
                                payout=payout,
                                settled=False,
                                result=None,
                                pnl=0,
                            ))
                            st.session_state.wallet -= stake
                            st.rerun()

        st.divider()

        # ---- Active bets ----
        active = [b for b in st.session_state.bets if not b["settled"]]
        if active:
            st.markdown("<h3 style='color:#ffffff'> Active Bets</h3>", unsafe_allow_html=True)
            rows = []
            for b in active:
                rows.append(dict(
                    Match=b["match"], Home=b["home"], Away=b["away"],
                    Pick=b["pick"], Odds=f"{b['odds']:.2f}",
                    Stake=f"${b['stake']:.0f}", Payout=f"${b['payout']:.2f}"
                ))
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            if st.button(" Settle All Bets", type="primary", use_container_width=True):
                rng = np.random.default_rng(0)
                gres, ko_results, champ, _ = simulate_once(
                    group_teams, gfixtures, kf, proba_table, rng)
                match_results = {}
                for f in gfixtures:
                    h, a = f["home"], f["away"]
                    pw, pd_, pl = proba_table.get((h, a), (1/3, 1/3, 1/3))
                    r = rng.random()
                    if r < pw:
                        match_results[f["match"]] = (f"{h} (Win)", h)
                    elif r < pw + pd_:
                        match_results[f["match"]] = ("Draw", None)
                    else:
                        match_results[f["match"]] = (f"{a} (Win)", a)
                for bid in [b["id"] for b in active]:
                    for b in st.session_state.bets:
                        if b["id"] == bid:
                            mr = match_results.get(b["match"])
                            if mr is None:
                                continue
                            if b["pick"] == mr[0]:
                                b["result"] = "Won"
                                b["pnl"] = b["payout"] - b["stake"]
                                st.session_state.wallet += b["payout"]
                            else:
                                b["result"] = "Lost"
                                b["pnl"] = -b["stake"]
                            b["settled"] = True
                st.rerun()
        else:
            st.info("No active bets. Pick a match above to get started.")

        # ---- Bet history ----
        settled = [b for b in st.session_state.bets if b["settled"]]
        if settled:
            st.divider()
            st.markdown("<h3 style='color:#ffffff'> Bet History</h3>", unsafe_allow_html=True)
            rows = []
            for b in settled[-20:]:
                rows.append(dict(
                    Match=b["match"], Pick=b["pick"], Odds=f"{b['odds']:.2f}",
                    Stake=f"${b['stake']:.0f}", Result=b["result"],
                    PnL=f"${b['pnl']:+,.0f}" if b["pnl"] != 0 else "$0"
                ))
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            if st.button(" Reset Bets", use_container_width=True):
                st.session_state.wallet = 10000.0
                st.session_state.bets = []
                st.session_state.bet_id = 0
                st.rerun()

    st.markdown(f"""
    <hr style="border-color:#2a2a5a;margin-top:40px">
    <div style="text-align:center;color:#5a5a8a;font-size:13px">
        Built with XGBoost  |  Data: Elo ratings, FIFA rankings & historical matches  |  Monte Carlo simulations
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
