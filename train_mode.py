"""
train_mode.py
-------------
Trains a Random Forest classifier to predict comfort vs budget mode.

The model learns from the exact same category totals that compute_summary
receives at runtime, so training and inference use identical numbers with
zero inconsistency.

Features (5 absolute monthly costs):
    bills          → total utility bills
    transport      → full-price transport total
    groceries      → grocery basket total
    entertainment  → full entertainment total
    accommodation  → average rent across all options

Label:
    "comfort"  →  total_expenses ≤ salary
    "budget"   →  total_expenses > salary

Usage:
    python train_mode.py

Outputs (saved to models/):
    mode_model.pkl
    mode_importance.png
    Mode model: RandomForestClassifier_report.txt
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

from data_preprocessing import (
    preprocess_data,
    compute_bills,
    compute_transportation,
    compute_groceries,
    compute_entertainment,
    compute_accommodations,
)

os.makedirs("models", exist_ok=True)

FEATURE_COLS = [
    "bills",
    "transport",
    "groceries",
    "entertainment",
    "accommodation",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE & LABEL CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each city row, call the same compute_* functions used at inference
    time and derive a label from the rule: budget if expenses exceed salary.
    """
    records = []

    for _, row in df.iterrows():
        salary = float(row["x54"]) or 1e-9

        _, bills,          _     = compute_bills(row)
        _, transport,  _, _      = compute_transportation(row)
        _, groceries,      _     = compute_groceries(row)
        _, entertainment,  _     = compute_entertainment(row)
        _, accommodation, _, _   = compute_accommodations(row)

        total_expenses   = bills + transport + groceries + entertainment + accommodation
        remaining_income = salary - total_expenses

        records.append({
            "bills":         bills,
            "transport":     transport,
            "groceries":     groceries,
            "entertainment": entertainment,
            "accommodation": accommodation,
            "mode":          "comfort" if remaining_income >= 0 else "budget",
        })

    result = pd.DataFrame(records)
    print(f"Built {len(result)} feature rows")
    print("Label distribution:\n", result["mode"].value_counts())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(df_features: pd.DataFrame):
    """
    Tune a Random Forest with GridSearchCV (cv=3, scoring=f1_macro).

    f1_macro averages F1 across both classes without requiring a pos_label,
    which is ideal for a two-string-label problem.
    """
    X = df_features[FEATURE_COLS]
    y = df_features["mode"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [3, 5, 10, None],
        "min_samples_leaf":  [5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features":      ["sqrt", "log2"],
        "class_weight":      ["balanced"],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(f"Best params: {grid.best_params_}")
    return best_model, X, y, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluation(name: str, y_test, y_pred) -> None:
    """Print metrics to stdout and save a text report to models/."""
    cm   = confusion_matrix(y_test, y_pred, labels=["comfort", "budget"])
    acc  = round(accuracy_score (y_test, y_pred), 3)
    prec = round(precision_score(y_test, y_pred, average="macro"), 3)
    rec  = round(recall_score   (y_test, y_pred, average="macro"), 3)
    f1   = round(f1_score       (y_test, y_pred, average="macro"), 3)

    print(f"\n── {name} ──────────────────────────────────────────────────────")
    print(f"Accuracy:  {acc}")
    print(f"Precision: {prec}")
    print(f"Recall:    {rec}")
    print(f"F1-score:  {f1}")
    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"               comfort   budget")
    print(f"  comfort       {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"  budget        {cm[1,0]:5d}    {cm[1,1]:5d}")

    report_path = f"models/{name}_report.txt"
    lines = [
        "=" * 60,
        f"MODE CLASSIFIER — {name} Report",
        "=" * 60,
        f"Accuracy:  {acc}",
        f"Precision: {prec}",
        f"Recall:    {rec}",
        f"F1-score:  {f1}",
        "",
        "Confusion matrix (rows=actual, cols=predicted):",
        "               comfort   budget",
        f"  comfort       {cm[0,0]:5d}    {cm[0,1]:5d}",
        f"  budget        {cm[1,0]:5d}    {cm[1,1]:5d}",
    ]
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Report saved → {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE CHART
# ══════════════════════════════════════════════════════════════════════════════

def save_importance_chart(model) -> None:
    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8, 4))
    importances.plot(kind="barh", ax=ax, color="#4A90D9")
    ax.set_title("What drives comfort vs budget mode?", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("models/mode_importance.png", dpi=150)
    plt.close()
    print("Feature importance chart saved → models/mode_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading data …")
    df, error = preprocess_data("data/cost-of-living.csv", "data/cost-of-living_v2.csv")
    if error:
        raise RuntimeError(error)

    print("\nComputing features and labels …")
    df_features = build_features_and_labels(df)

    print("\nTraining model …")
    model, X, y, X_test, y_test = train(df_features)
    y_pred = model.predict(X_test)

    evaluation("Mode model: RandomForestClassifier", y_test, y_pred)
    save_importance_chart(model)

    joblib.dump(model, "models/mode_model.pkl")
    print("\n✓  Model saved → models/mode_model.pkl")


if __name__ == "__main__":
    main()
