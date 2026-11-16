"""End-to-end training + evaluation pipeline.

Usage (from the project root):
    python src/train.py                 # uses data/creditcard.csv (Kaggle dataset)
    python src/train.py --demo          # uses SYNTHETIC data to test the pipeline

Protocol (no test-set leakage):
  1. Drop duplicates, stratified 80/20 train/test split.
  2. Carve a validation set (20%) out of the training portion.
  3. Compare 3 models x 4 imbalance strategies on the VALIDATION set.
  4. For each model, keep its best strategy, tune the threshold on validation,
     refit on the full training set, and evaluate on the TEST set once.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

import fraud_lib as fl

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"Logistic Regression": "#4C78A8", "Decision Tree": "#F58518", "Random Forest": "#54A24B"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ EDA
def eda_figures(df: pd.DataFrame) -> None:
    counts = df[fl.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["#4C78A8", "#E45756"])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Class distribution")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / len(df):.3%})", ha="center", va="bottom")
    ax.set_ylim(top=counts.max() * 5)
    fig.tight_layout()
    fig.savefig(fl.FIG_DIR / "eda_class_distribution.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_df = df.assign(Label=df[fl.TARGET].map({0: "Legitimate", 1: "Fraud"}), logAmount=np.log1p(df["Amount"]))
    sns.boxplot(data=plot_df, x="Label", y="logAmount", palette=["#4C78A8", "#E45756"], hue="Label", legend=False, ax=axes[0])
    axes[0].set_title("Transaction amount by class")
    axes[0].set_ylabel("log(1 + Amount)")
    for lab, col in [("Legitimate", "#4C78A8"), ("Fraud", "#E45756")]:
        sns.kdeplot(plot_df.loc[plot_df.Label == lab, "logAmount"], ax=axes[1], label=lab, color=col, fill=True, alpha=0.3, common_norm=False)
    axes[1].set_title("Amount density (each class normalised)")
    axes[1].set_xlabel("log(1 + Amount)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fl.FIG_DIR / "eda_amount_by_class.png", dpi=140)
    plt.close(fig)

    corr = df[fl.FEATURES].corrwith(df[fl.TARGET]).sort_values(key=np.abs, ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(corr.index[::-1], corr.values[::-1], color=["#E45756" if v > 0 else "#4C78A8" for v in corr.values[::-1]])
    ax.set_title("Top 12 features by correlation with Class")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(fl.FIG_DIR / "eda_feature_correlation.png", dpi=140)
    plt.close(fig)


# -------------------------------------------------------- result figures
def result_figures(test_scores: pd.DataFrame, results: dict, winner_pipe) -> None:
    y = test_scores["y_true"].values
    names = fl.MODELS

    # confusion matrices (tuned threshold)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, n in zip(axes, names):
        m = results["test"][n]["tuned"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred legit", "Pred fraud"], yticklabels=["Actual legit", "Actual fraud"])
        ax.set_title(f"{n}\n(threshold = {m['threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(fl.FIG_DIR / "confusion_matrices.png", dpi=140)
    plt.close(fig)

    # PR + ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for n in names:
        p, r, _ = precision_recall_curve(y, test_scores[n])
        axes[0].plot(r, p, label=f"{n} (AP={results['test'][n]['tuned']['pr_auc']:.3f})", color=PALETTE[n], lw=2)
        fpr, tpr, _ = roc_curve(y, test_scores[n])
        axes[1].plot(fpr, tpr, label=f"{n} (AUC={results['test'][n]['tuned']['roc_auc']:.3f})", color=PALETTE[n], lw=2)
    axes[0].axhline(y.mean(), ls="--", color="grey", label=f"No-skill ({y.mean():.4f})")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves (test set)")
    axes[1].plot([0, 1], [0, 1], ls="--", color="grey", label="No-skill")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    for ax in axes:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fl.FIG_DIR / "pr_roc_curves.png", dpi=140)
    plt.close(fig)

    # metric comparison bars
    rows = []
    for n in names:
        for k, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("pr_auc", "PR-AUC")]:
            rows.append({"Model": n, "Metric": label, "Value": results["test"][n]["tuned"][k]})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=pd.DataFrame(rows), x="Metric", y="Value", hue="Model", palette=PALETTE, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the held-out test set")
    fig.tight_layout()
    fig.savefig(fl.FIG_DIR / "model_comparison.png", dpi=140)
    plt.close(fig)

    # winner: feature importance
    clf = winner_pipe.named_steps["clf"]
    imp = getattr(clf, "feature_importances_", None)
    label = "Feature importance"
    if imp is None and hasattr(clf, "coef_"):
        imp, label = np.abs(clf.coef_[0]), "|Coefficient| (scaled features)"
    if imp is not None:
        s = pd.Series(imp, index=fl.transformed_feature_names()).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.barh(s.index[::-1], s.values[::-1], color="#54A24B")
        ax.set_title(f"{results['winner']}: top 15 features")
        ax.set_xlabel(label)
        fig.tight_layout()
        fig.savefig(fl.FIG_DIR / "feature_importance.png", dpi=140)
        plt.close(fig)


def threshold_figure(y_val, proba_val, name: str, thr: float) -> None:
    p, r, t = precision_recall_curve(y_val, proba_val)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t, p[:-1], label="Precision", color="#4C78A8", lw=2)
    ax.plot(t, r[:-1], label="Recall", color="#E45756", lw=2)
    ax.plot(t, f1, label="F1", color="#54A24A", lw=2)
    ax.axvline(thr, ls="--", color="black", label=f"Chosen threshold = {thr:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Score", title=f"{name}: precision / recall trade-off (validation set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fl.FIG_DIR / "threshold_tradeoff.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(fl.DATA_DIR / "creditcard.csv"), help="path to creditcard.csv")
    ap.add_argument("--demo", action="store_true", help="use synthetic demo data (pipeline test only)")
    ap.add_argument("--strategies", nargs="+", default=fl.STRATEGIES, choices=fl.STRATEGIES,
                    help="imbalance strategies to compare (default: all four)")
    args = ap.parse_args()

    for d in (fl.MODEL_DIR, fl.REPORT_DIR, fl.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if args.demo:
        from make_demo_data import generate

        data_path = fl.DATA_DIR / "creditcard_demo.csv"
        generate().to_csv(data_path, index=False)
        log(f"SYNTHETIC demo mode: generated {data_path.name}. Results are NOT valid for your report.")
    else:
        data_path = Path(args.data)
        if not data_path.exists():
            raise SystemExit(
                f"Dataset not found at {data_path}.\n"
                "Download creditcard.csv from Kaggle (mlg-ulb/creditcardfraud) into data/, "
                "or run with --demo to test the pipeline on synthetic data."
            )

    # ---- load + split
    df, info = fl.load_dataset(data_path)
    X, y = df[fl.FEATURES], df[fl.TARGET]
    log(f"Loaded {len(df):,} rows (removed {info['duplicates_removed']:,} duplicates); fraud = {y.sum()} ({y.mean():.4%})")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=fl.RANDOM_STATE, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=fl.RANDOM_STATE, stratify=y_trainval)
    log(f"Split -> train {len(X_tr):,} | validation {len(X_val):,} | test {len(X_test):,} "
        f"(fraud: {y_tr.sum()} / {y_val.sum()} / {y_test.sum()})")

    eda_figures(df)
    log("EDA figures saved")

    # ---- stage 1: model x strategy comparison on validation
    experiments = []
    for name in fl.MODELS:
        for strat in args.strategies:
            t0 = time.time()
            pipe = fl.build_model(name, strat).fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_val)[:, 1]
            thr = fl.best_f1_threshold(y_val, proba)
            m = fl.compute_metrics(y_val, proba, thr)
            m.update(model=name, strategy=strat, fit_seconds=round(time.time() - t0, 1),
                     default_threshold_f1=fl.compute_metrics(y_val, proba, 0.5)["f1"],
                     default_threshold_recall=fl.compute_metrics(y_val, proba, 0.5)["recall"])
            experiments.append(m)
            log(f"  [val] {name:<19} {strat:<12} PR-AUC={m['pr_auc']:.4f} F1@best-thr={m['f1']:.4f} ({m['fit_seconds']}s)")

    exp_df = pd.DataFrame(experiments)
    exp_df.to_csv(fl.REPORT_DIR / "validation_experiments.csv", index=False)

    # ---- stage 2: refit best strategy per model on full train, evaluate once on test
    results = {
        "dataset": {
            "name": "SYNTHETIC demo data" if args.demo else "Kaggle Credit Card Fraud Detection (ULB)",
            "synthetic": bool(args.demo),
            **info,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "fraud": int(y.sum()),
            "legit": int((y == 0).sum()),
            "fraud_pct": float(y.mean() * 100),
            "majority_class_accuracy": float(1 - y.mean()),
            "split": {"train": int(len(X_tr)), "validation": int(len(X_val)), "test": int(len(X_test)),
                      "test_fraud": int(y_test.sum()), "train_fraud": int(y_tr.sum()), "val_fraud": int(y_val.sum())},
        },
        "validation_experiments": experiments,
        "test": {},
    }
    test_scores = pd.DataFrame({"y_true": y_test.values})
    pipes, val_proba = {}, {}

    for name in fl.MODELS:
        sub = exp_df[exp_df.model == name].sort_values("pr_auc", ascending=False).iloc[0]
        strat, thr = sub["strategy"], float(sub["threshold"])
        log(f"Refitting {name} with best strategy '{strat}' on full training set")
        # threshold is tuned on validation predictions of the train-subset model
        pipe_sub = fl.build_model(name, strat).fit(X_tr, y_tr)
        val_proba[name] = pipe_sub.predict_proba(X_val)[:, 1]

        pipe = fl.build_model(name, strat).fit(X_trainval, y_trainval)
        p_test = pipe.predict_proba(X_test)[:, 1]
        p_train = pipe.predict_proba(X_trainval)[:, 1]
        test_scores[name] = p_test
        pipes[name] = pipe

        tr = fl.compute_metrics(y_trainval, p_train, thr)
        results["test"][name] = {
            "strategy": strat,
            "strategy_label": fl.STRATEGY_LABELS[strat],
            "threshold": thr,
            "val_pr_auc": float(sub["pr_auc"]),
            "default": fl.compute_metrics(y_test, p_test, 0.5),
            "tuned": fl.compute_metrics(y_test, p_test, thr),
            "train": {"f1": tr["f1"], "recall": tr["recall"], "precision": tr["precision"], "pr_auc": tr["pr_auc"]},
        }
        t = results["test"][name]["tuned"]
        log(f"  [test] {name:<19} P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} PR-AUC={t['pr_auc']:.3f}")

    winner = max(fl.MODELS, key=lambda n: results["test"][n]["val_pr_auc"])
    results["winner"] = winner
    log(f"Selected model (highest validation PR-AUC): {winner}")

    # ---- persist artefacts
    for name in fl.MODELS:
        bundle = {"pipeline": pipes[name], "threshold": results["test"][name]["threshold"],
                  "features": fl.FEATURES, "name": name, "strategy": results["test"][name]["strategy"],
                  "synthetic": bool(args.demo)}
        joblib.dump(bundle, fl.MODEL_DIR / f"{fl.slug(name)}.joblib", compress=3)
    joblib.dump({"pipeline": pipes[winner], "threshold": results["test"][winner]["threshold"],
                 "features": fl.FEATURES, "name": winner, "strategy": results["test"][winner]["strategy"],
                 "synthetic": bool(args.demo)}, fl.MODEL_DIR / "fraud_model.joblib", compress=3)

    test_scores.round(6).to_csv(fl.REPORT_DIR / "test_scores.csv", index=False)

    rng = np.random.default_rng(fl.RANDOM_STATE)
    test_df = X_test.assign(**{fl.TARGET: y_test.values})
    fraud_s = test_df[test_df[fl.TARGET] == 1]
    fraud_s = fraud_s.sample(min(50, len(fraud_s)), random_state=fl.RANDOM_STATE)
    legit_s = test_df[test_df[fl.TARGET] == 0].sample(100, random_state=fl.RANDOM_STATE)
    samples = pd.concat([fraud_s, legit_s]).sample(frac=1, random_state=fl.RANDOM_STATE).reset_index(drop=True)
    samples.to_csv(fl.REPORT_DIR / "test_samples.csv", index=False)

    result_figures(test_scores, results, pipes[winner])
    threshold_figure(y_val, val_proba[winner], winner, results["test"][winner]["threshold"])

    with open(fl.REPORT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("Saved models/, reports/results.json, reports/figures/. Done.")


if __name__ == "__main__":
    main()
