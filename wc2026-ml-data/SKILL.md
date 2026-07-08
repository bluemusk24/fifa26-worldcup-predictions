# WC2026 ML Data — Prediction System

This project builds a **FIFA World Cup 2026 match-prediction system**:
XGBoost classifier trained on Elo ratings, FIFA rankings, and historical
match results, deployed as a Streamlit app with Monte-Carlo tournament
simulation and match-odds betting interface.

## Project layout

```
src/
  feature/     Data loading + team-name alias resolution
  training/    Model training pipeline
  app/         Streamlit app + self-test
  predictor.py Shared simulation logic (Monte-Carlo, bracket, odds)
  precompute.py Generate cached probability table + MC results
data/          Source CSVs (Elo, FIFA ratings, schedule)
model/         Trained model + precomputed .pkl files
.streamlit/    Streamlit Cloud config (dark theme)
```

## Pipeline

### 1. Data loading (`src/feature/loader.py`)

- Downloads 20 most-recent matches per WC2026 team from openfootball
- Loads `elo_ratings.csv`, `fifa_ratings.csv`, `FIFA2026_schedule_Fixtures.csv`
- Resolves team-name aliases (e.g. `USA` → `United States`, `Congo DR` → `DR Congo`)
- Resolves 6 playoff slots in the schedule to concrete teams
- Hosts (USA, Canada, Mexico) flagged for home advantage in group stage

### 2. Training (`src/training/train.py`)

- Builds feature matrix: `home`, `is_friendly`, `team_elo`, `opp_elo`,
  `elo_diff`, `team_rank`, `opp_rank`, `rank_diff`, `team_points`,
  `opp_points`, `points_diff`
- Label: 3-class (`loss=0, draw=1, win=2`)
- Filters last 20 games per team, includes friendlies
- Holds out most recent 2 games per team as test set
- Trains `XGBClassifier` (multi:softprob, max_depth=6, n_estimators=200)
- Saves model + evaluation images

### 3. Precomputation (`src/precompute.py`)

Batches all 2256 ordered-pair predictions into a single model call,
then runs 10,000 Monte-Carlo rollouts and 2000 group-phase simulations.
Caches as `.pkl` files for instant app startup.

### 4. App (`src/app/app.py`)

Three-tab Streamlit interface:

- **Tournament Simulation** — champion probabilities, group-stage standings,
  route-to-final table, knockout bracket (chalk / random mode), re-run button
- **Match Odds & Betting** — head-to-head probabilities with decimal odds,
  visual bars, betting slip
- **Virtual Betting** — $10,000 paper-trading wallet, place mock bets on
  any group fixture at model-derived decimal odds, settle against chalk
  simulation results, track P&L and ROI

## Quickstart

```bash
uv pip install -r requirements.txt
# Train model (uses cached data if available)
python src/training/train.py
# Precompute for fast app startup
python src/precompute.py
# Run locally
streamlit run src/app/app.py
```

## Model performance

- Accuracy: ~51% (vs 33% random baseline for 3-class)
- ROC-AUC: ~0.68 (one-vs-rest)
- 48 teams × 20 matches = 960 training rows

## Deployment

Deployed on Streamlit Cloud from GitHub:
`src/app/app.py` as main file, `.streamlit/config.toml` sets dark theme.

App URL: https://fifa26-worldcup-predictions.streamlit.app

## Alias map

```
USA → United States         Türkiye → Turkey
Korea Republic → South Korea  IR Iran → Iran
Czechia → Czech Republic    Côte d'Ivoire → Ivory Coast
Cabo Verde → Cape Verde     Congo DR → DR Congo
```

## Schedule structure

12 groups A–L × 4 teams (matches 1–72). Knockout:
**R32 = 73–88, R16 = 89–96, QF = 97–100, SF = 101–102,
3rd-place = 103, FINAL = 104.**

Qualification: top 2 of each group + 8 best third-placed teams.
