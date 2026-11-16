"""Shared helpers for the Credit Card Fraud Detection project."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
MODELS = ["Logistic Regression", "Decision Tree", "Random Forest"]
STRATEGIES = ["none", "class_weight", "smote", "undersample"]
STRATEGY_LABELS = {
    "none": "No handling (baseline)",
    "class_weight": "Class weights",
    "smote": "SMOTE (train only)",
    "undersample": "Random undersampling (train only)",
}
RANDOM_STATE = 42
SAMPLING_RATIO = 0.1  # minority:majority ratio after SMOTE / undersampling


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


# ----------------------------------------------------------------- data
def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load the CSV, validate the schema and drop duplicate / null rows."""
    df = pd.read_csv(path)
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    info = {
        "rows_raw": int(len(df)),
        "missing_values": int(df[FEATURES + [TARGET]].isnull().sum().sum()),
        "duplicates_removed": int(df.duplicated().sum()),
    }
    df = df.dropna(subset=FEATURES + [TARGET]).drop_duplicates().reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df, info


# ---------------------------------------------------------------- models
def make_preprocessor() -> ColumnTransformer:
    """Scale Time and Amount (robust to outliers); V1-V28 are already PCA outputs."""
    return ColumnTransformer(
        [("scale", RobustScaler(), ["Time", "Amount"])], remainder="passthrough"
    )


def transformed_feature_names() -> list[str]:
    return ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def build_model(name: str, strategy: str) -> Pipeline:
    """Pipeline = scaling -> (optional resampler) -> classifier.

    Because the sampler lives inside the pipeline it is only applied when fitting,
    i.e. on training data. It is never applied to validation / test data.
    """
    use_cw = strategy == "class_weight"
    if name == "Logistic Regression":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced" if use_cw else None,
            random_state=RANDOM_STATE,
        )
    elif name == "Decision Tree":
        clf = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced" if use_cw else None,
            random_state=RANDOM_STATE,
        )
    elif name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample" if use_cw else None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    steps = [("prep", make_preprocessor())]
    if strategy == "smote":
        steps.append(("sampler", SMOTE(sampling_strategy=SAMPLING_RATIO, random_state=RANDOM_STATE)))
    elif strategy == "undersample":
        steps.append(
            ("sampler", RandomUnderSampler(sampling_strategy=SAMPLING_RATIO, random_state=RANDOM_STATE))
        )
    elif strategy not in ("none", "class_weight"):
        raise ValueError(f"Unknown strategy: {strategy}")
    steps.append(("clf", clf))
    return Pipeline(steps)


# --------------------------------------------------------------- metrics
def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Threshold that maximises F1 (use on a validation set, never on test)."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[int(np.argmax(f1))]) if len(t) else 0.5


def predict_with_bundle(bundle: dict, df: pd.DataFrame, threshold: float | None = None):
    """Return (probabilities, predicted classes) for a saved model bundle."""
    thr = bundle["threshold"] if threshold is None else threshold
    proba = bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1]
    return proba, (proba >= thr).astype(int)
