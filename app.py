"""FraudShield - Credit Card Fraud Detection dashboard.

Run:  streamlit run app.py   (after `python src/train.py`)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import fraud_lib as fl  # noqa: E402

st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="wide")

DISCLAIMER = (
    "The output is a **model prediction** based on patterns in historical, anonymised data. "
    "It is not a verified fraud determination."
)


# ------------------------------------------------------------- loaders
@st.cache_data
def load_results():
    p = fl.REPORT_DIR / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource
def load_bundle(name: str):
    return joblib.load(fl.MODEL_DIR / f"{fl.slug(name)}.joblib")


@st.cache_data
def load_csv(name: str):
    p = fl.REPORT_DIR / name
    return pd.read_csv(p) if p.exists() else None


results = load_results()
if results is None:
    st.title("🛡️ FraudShield")
    st.error("No trained model found. Run `python src/train.py` first (or `python src/train.py --demo`).")
    st.stop()

ds, winner = results["dataset"], results["winner"]
samples, scores = load_csv("test_samples.csv"), load_csv("test_scores.csv")

# -------------------------------------------------------------- header
st.title("🛡️ FraudShield: Credit Card Fraud Detection System")
st.caption("Compare Logistic Regression, Decision Tree and Random Forest on imbalanced transaction data.")
if ds["synthetic"]:
    st.warning("These results come from **synthetic demo data** used to test the pipeline. "
               "Retrain on the real Kaggle dataset before using any number in your report.")

tab_over, tab_check, tab_cmp, tab_ins = st.tabs(
    ["📊 Overview", "🔎 Transaction checker", "⚖️ Model comparison", "📈 Insights"])

# ------------------------------------------------------------ overview
with tab_over:
    t = results["test"][winner]["tuned"]
    c = st.columns(4)
    c[0].metric("Total transactions", f"{ds['rows']:,}")
    c[1].metric("Fraudulent transactions", f"{ds['fraud']:,}")
    c[2].metric("Fraud percentage", f"{ds['fraud_pct']:.3f}%")
    c[3].metric("Flagged in test set", f"{t['tp'] + t['fp']:,}",
                help=f"{winner}: {t['tp']} true frauds caught, {t['fp']} false alarms")
    st.markdown(f"**Dataset:** {ds['name']} · duplicates removed: {ds['duplicates_removed']:,} · "
                f"missing values: {ds['missing_values']:,}")
    st.info(
        f"**Why accuracy is misleading:** predicting *every* transaction as legitimate scores "
        f"**{ds['majority_class_accuracy']:.2%}** accuracy while catching **0** frauds. "
        "That is why this project is judged on precision, recall, F1 and PR-AUC.")
    left, right = st.columns(2)
    left.image(str(fl.FIG_DIR / "eda_class_distribution.png"), width="stretch")
    right.image(str(fl.FIG_DIR / "eda_amount_by_class.png"), width="stretch")
    st.markdown(f"**Selected model:** {winner} (highest validation PR-AUC, "
                f"strategy: {results['test'][winner]['strategy_label']})")

# --------------------------------------------------- transaction checker
with tab_check:
    st.subheader("Check a transaction")
    st.caption("V1-V28 are anonymised PCA components, so merchant names, card numbers or locations "
               "cannot be entered. Use a held-out example, edit its values, or upload a CSV.")
    model_name = st.selectbox("Model", fl.MODELS, index=fl.MODELS.index(winner))
    bundle = load_bundle(model_name)
    thr = st.slider("Decision threshold", 0.01, 0.99, float(np.clip(bundle["threshold"], 0.01, 0.99)), 0.01,
                    help="Transactions with fraud probability >= threshold are flagged.")
    mode = st.radio("Input mode", ["Held-out example", "Edit values", "Upload CSV"], horizontal=True)

    def show_single(row: pd.DataFrame, actual=None):
        proba, _ = fl.predict_with_bundle(bundle, row, thr)
        p = float(proba[0])
        flagged = p >= thr
        a, b, c3 = st.columns(3)
        a.metric("Predicted class", "🚨 Fraud" if flagged else "✅ Legitimate")
        b.metric("Fraud probability", f"{p:.1%}")
        if actual is not None:
            c3.metric("Actual label (ground truth)", "Fraud" if int(actual) == 1 else "Legitimate")
        st.progress(min(max(p, 0.0), 1.0))
        st.caption(DISCLAIMER)

    if mode in ("Held-out example", "Edit values"):
        if samples is None:
            st.error("reports/test_samples.csv is missing - re-run training.")
        else:
            labels = [f"#{i} - {'FRAUD' if r.Class == 1 else 'legit'} - amount {r.Amount:,.2f}"
                      for i, r in samples.iterrows()]
            pick = st.selectbox("Held-out test transaction", range(len(samples)), format_func=lambda i: labels[i])
            row = samples.loc[[pick], fl.FEATURES].copy()
            actual = samples.loc[pick, fl.TARGET]
            if mode == "Edit values":
                with st.expander("Edit feature values", expanded=True):
                    cols = st.columns(4)
                    for i, f in enumerate(fl.FEATURES):
                        row[f] = cols[i % 4].number_input(f, value=float(row[f].iloc[0]), format="%.4f",
                                                          key=f"{f}_{pick}")
                actual = None
            show_single(row, actual)
    else:
        up = st.file_uploader("CSV with columns Time, V1-V28, Amount (optional: Class)", type="csv")
        if up is not None:
            data = pd.read_csv(up)
            missing = [c for c in fl.FEATURES if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                proba, pred = fl.predict_with_bundle(bundle, data, thr)
                out = data.assign(fraud_probability=proba.round(4), predicted_class=pred)
                n_flag = int(pred.sum())
                m = st.columns(3)
                m[0].metric("Rows scored", f"{len(out):,}")
                m[1].metric("Flagged as fraud", f"{n_flag:,}")
                m[2].metric("Flag rate", f"{n_flag / len(out):.2%}")
                if fl.TARGET in data.columns and data[fl.TARGET].nunique() > 1:
                    mm = fl.compute_metrics(data[fl.TARGET], proba, thr)
                    st.write(f"Against the uploaded labels: precision **{mm['precision']:.3f}**, "
                             f"recall **{mm['recall']:.3f}**, F1 **{mm['f1']:.3f}**, PR-AUC **{mm['pr_auc']:.3f}**")
                st.dataframe(out.sort_values("fraud_probability", ascending=False).head(500), width="stretch")
                st.download_button("Download predictions", out.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")
                st.caption(DISCLAIMER)

# -------------------------------------------------- model comparison
with tab_cmp:
    st.subheader("Held-out test results")
    view = st.radio("Threshold", ["Tuned on validation set", "Default (0.5)"], horizontal=True)
    key = "tuned" if view.startswith("Tuned") else "default"
    rows = []
    for n in fl.MODELS:
        r = results["test"][n]
        m = r[key]
        rows.append({"Model": n + (" ⭐" if n == winner else ""), "Imbalance strategy": r["strategy_label"],
                     "Threshold": round(m["threshold"], 3), "Precision": m["precision"], "Recall": m["recall"],
                     "F1-score": m["f1"], "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"],
                     "Accuracy": m["accuracy"], "Frauds missed (FN)": m["fn"], "False alarms (FP)": m["fp"]})
    tbl = pd.DataFrame(rows).set_index("Model")
    st.dataframe(tbl.style.format({c: "{:.4f}" for c in ["Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC", "Accuracy"]}),
                 width="stretch")
    st.image(str(fl.FIG_DIR / "model_comparison.png"), width="stretch")

    st.subheader("Overfitting check (train vs test)")
    of = pd.DataFrame([{"Model": n, "Train F1": results["test"][n]["train"]["f1"],
                        "Test F1": results["test"][n]["tuned"]["f1"],
                        "F1 gap": results["test"][n]["train"]["f1"] - results["test"][n]["tuned"]["f1"],
                        "Train PR-AUC": results["test"][n]["train"]["pr_auc"],
                        "Test PR-AUC": results["test"][n]["tuned"]["pr_auc"]} for n in fl.MODELS]).set_index("Model")
    st.dataframe(of.style.format("{:.4f}"), width="stretch")
    st.caption("A large positive gap means the model memorised the training data.")

    st.subheader("Imbalance-handling experiment (validation set)")
    ex = pd.DataFrame(results["validation_experiments"])
    ex["strategy"] = ex["strategy"].map(fl.STRATEGY_LABELS)
    ex = ex[["model", "strategy", "pr_auc", "roc_auc", "precision", "recall", "f1", "threshold"]]
    ex.columns = ["Model", "Strategy", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Best threshold"]
    st.dataframe(ex.style.format({c: "{:.4f}" for c in ex.columns[2:]}), width="stretch", hide_index=True)

# ---------------------------------------------------------- insights
with tab_ins:
    st.subheader("Confusion matrices (test set)")
    st.image(str(fl.FIG_DIR / "confusion_matrices.png"), width="stretch")
    st.subheader("Precision-Recall and ROC curves")
    st.image(str(fl.FIG_DIR / "pr_roc_curves.png"), width="stretch")

    st.subheader("Interactive threshold explorer")
    if scores is not None:
        cA, cB = st.columns([1, 2])
        mn = cA.selectbox("Model", fl.MODELS, index=fl.MODELS.index(winner), key="ins_model")
        t2 = cA.slider("Threshold", 0.01, 0.99, float(np.clip(results["test"][mn]["threshold"], 0.01, 0.99)), 0.01, key="ins_thr")
        mm = fl.compute_metrics(scores["y_true"], scores[mn], t2)
        cA.metric("Precision", f"{mm['precision']:.3f}")
        cA.metric("Recall", f"{mm['recall']:.3f}")
        cA.metric("F1-score", f"{mm['f1']:.3f}")
        cm = pd.DataFrame([[mm["tn"], mm["fp"]], [mm["fn"], mm["tp"]]],
                          index=["Actual legit", "Actual fraud"], columns=["Pred legit", "Pred fraud"])
        cB.dataframe(cm, width="stretch")
        cB.caption("Lower the threshold -> more frauds caught (recall up) but more false alarms (precision down).")

    l, r = st.columns(2)
    if (fl.FIG_DIR / "feature_importance.png").exists():
        l.subheader("What drives the selected model")
        l.image(str(fl.FIG_DIR / "feature_importance.png"), width="stretch")
    r.subheader("Threshold trade-off")
    r.image(str(fl.FIG_DIR / "threshold_tradeoff.png"), width="stretch")
    st.subheader("Features most correlated with fraud")
    st.image(str(fl.FIG_DIR / "eda_feature_correlation.png"), width=520)

st.divider()
st.caption("Limitations: anonymised PCA features, a single historical dataset and a very small number of fraud cases. "
           "A real system would need streaming data, concept-drift monitoring and human review.")
