# FIFA World Cup 2026 Prediction System

An XGBoost-based machine learning system that predicts match outcomes for the FIFA World Cup 2026, with a Monte Carlo tournament simulator and head-to-head betting odds interface.

## Features

- **Match outcome prediction** (win/draw/loss) using Elo ratings, FIFA rankings, and historical match results
- **Full tournament simulation** via Monte Carlo rollouts (10,000 iterations)
- **Champion probabilities** for all 48 teams, with group-stage standings and route-to-final tables
- **Knockout bracket visualization** with chalk (most likely) and random simulation modes
- **Match odds & betting interface** with decimal odds based on predicted probabilities
- **Virtual betting platform** — $10,000 paper-trading wallet, place mock bets on 72 group fixtures, settle against chalk simulation, track P&L and ROI
- **Instant startup** via precomputed probability tables and cached simulations

## Architecture

```
src/
├── feature/loader.py      Data loading + feature engineering (Elo, FIFA, matches)
├── training/train.py      XGBoost training pipeline (last 20 games/team, inc. friendlies)
├── predictor.py           Shared match prediction + Monte Carlo + bracket logic
├── precompute.py          Batch-predicts all 2256 ordered pairs, caches MC results
└── app/
    ├── app.py             Streamlit app (Tournament + Odds + Virtual Betting tabs)
    ├── _selftest.py       Offline test suite
    └── .streamlit/        Streamlit Cloud config
data/                      Source CSV files
model/                     Trained model + cached .pkl files
```

## Setup

```bash
pip install -r requirements.txt
python src/training/train.py
python src/precompute.py
streamlit run src/app/app.py
```

## Data sources

- Match results from openfootball (20 most recent per team)
- Elo ratings and FIFA rankings (historical, as-of match dates)
- WC 2026 schedule fixtures with 6 playoff slots resolved to concrete teams

## Deployment

Hosted on Streamlit Cloud at https://fifa26-worldcup-predictions.streamlit.app

No API keys or external services required — runs entirely from local CSV files and a trained XGBoost model.

## Model

- **Algorithm:** XGBoost multiclass classifier (loss / draw / win)
- **Training samples:** 864 (48 teams × ~18 games after 2 held out per team)
- **Features:** home/away flag, friendly flag, Elo rating, FIFA rank, FIFA points, plus differentials
- **Performance:** ~51% accuracy, ~0.68 ROC-AUC
