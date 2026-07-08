from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

import joblib
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.feature.loader import load_all_data, build_feature_view_data, load_elo_ratings, load_fifa_ratings, ALIASES, HOSTS

N_GAMES = 20
INCLUDE_FRIENDLY = True
N_TEST = 2
MODEL_NAME = "wc2026_match_predictor"
CLASS_NAMES = ["loss", "draw", "win"]


def train(results_df, elo_df, fifa_df):
    df, X, y, test_mask = build_feature_view_data(
        results_df, elo_df, fifa_df,
        n_games=N_GAMES, include_friendly=INCLUDE_FRIENDLY, n_test=N_TEST,
    )

    X_train, y_train = X[~test_mask], y[~test_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"Class distribution (train): loss={sum(y_train==0)}, draw={sum(y_train==1)}, win={sum(y_train==2)}")
    print(f"Class distribution (test): loss={sum(y_test==0)}, draw={sum(y_test==1)}, win={sum(y_test==2)}")

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        max_depth=6,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=20,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = (y_pred == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")

    try:
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
        print(f"ROC-AUC (OVR): {auc:.4f}")
    except Exception as e:
        print(f"AUC compute warning: {e}")
        auc = 0.0

    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": float(auc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_games_per_team": N_GAMES,
        "include_friendly": INCLUDE_FRIENDLY,
        "n_test_per_team": N_TEST,
    }

    model_dir = Path(__file__).resolve().parents[2] / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    images_dir = model_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title("Confusion Matrix (Test Set)")

    for i, label in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        axes[1].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_test_bin[:, i], y_proba[:, i]):.2f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curves (One-vs-Rest)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(images_dir / "evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    model_path = model_dir / "wc2026_match_predictor.joblib"
    joblib.dump(model, model_path)

    with open(model_dir / "model_metadata.json", "w") as f:
        json.dump({
            "feature_names": list(X.columns),
            "class_names": CLASS_NAMES,
            "n_games_per_team": N_GAMES,
            "include_friendly": INCLUDE_FRIENDLY,
            "n_test_per_team": N_TEST,
            "metrics": metrics,
            "hosts": list(HOSTS),
            "aliases": ALIASES,
        }, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Images saved to {images_dir}")

    return model, list(X.columns), metrics


def main():
    print("Loading data...")
    results, elo, fifa = load_all_data()
    print(f"Results: {len(results)} rows")
    print(f"Training model with: N_GAMES={N_GAMES}, INCLUDE_FRIENDLY={INCLUDE_FRIENDLY}, N_TEST={N_TEST}")
    model, features, metrics = train(results, elo, fifa)
    print(f"\nDone! Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
