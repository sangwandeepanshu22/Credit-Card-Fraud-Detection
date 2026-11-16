"""Generate reports/project_report.md from the ACTUAL results of the last training run.

Usage:  python src/generate_report.py
Every number in the report is read from reports/results.json - nothing is assumed.
Edit the placeholders marked  [EDIT]  (names, roll no., guide) before submitting.
"""
import json

import fraud_lib as fl

R = json.loads((fl.REPORT_DIR / "results.json").read_text())
ds, winner = R["dataset"], R["winner"]
T = R["test"]
sp = ds["split"]


def pct(x):
    return f"{x:.4f}"


def table(key):
    lines = ["| Model | Strategy | Precision | Recall | F1-score | PR-AUC | ROC-AUC |",
             "|---|---|---|---|---|---|---|"]
    for n in fl.MODELS:
        m = T[n][key]
        lines.append(f"| {n} | {T[n]['strategy_label']} | {pct(m['precision'])} | {pct(m['recall'])} | "
                     f"{pct(m['f1'])} | {pct(m['pr_auc'])} | {pct(m['roc_auc'])} |")
    return "\n".join(lines)


def cm_table():
    lines = ["| Model | TN | FP | FN | TP |", "|---|---|---|---|---|"]
    for n in fl.MODELS:
        m = T[n]["tuned"]
        lines.append(f"| {n} | {m['tn']:,} | {m['fp']:,} | {m['fn']:,} | {m['tp']:,} |")
    return "\n".join(lines)


def exp_table():
    lines = ["| Model | Strategy | Val PR-AUC | Val F1 (best thr.) | Val recall (best thr.) |", "|---|---|---|---|---|"]
    for e in R["validation_experiments"]:
        lines.append(f"| {e['model']} | {fl.STRATEGY_LABELS[e['strategy']]} | {pct(e['pr_auc'])} | "
                     f"{pct(e['f1'])} | {pct(e['recall'])} |")
    return "\n".join(lines)


def overfit_table():
    lines = ["| Model | Train F1 | Test F1 | Gap | Train PR-AUC | Test PR-AUC |", "|---|---|---|---|---|---|"]
    for n in fl.MODELS:
        tr, te = T[n]["train"], T[n]["tuned"]
        lines.append(f"| {n} | {pct(tr['f1'])} | {pct(te['f1'])} | {pct(tr['f1'] - te['f1'])} | "
                     f"{pct(tr['pr_auc'])} | {pct(te['pr_auc'])} |")
    return "\n".join(lines)


w = T[winner]["tuned"]
best_recall = max(fl.MODELS, key=lambda n: T[n]["tuned"]["recall"])
best_prec = max(fl.MODELS, key=lambda n: T[n]["tuned"]["precision"])
largest_gap = max(fl.MODELS, key=lambda n: T[n]["train"]["f1"] - T[n]["tuned"]["f1"])

banner = ""
if ds["synthetic"]:
    banner = ("> **WARNING: this report was generated from SYNTHETIC demo data. "
              "Re-run `python src/train.py` on the real Kaggle dataset, then re-run this script.**\n\n")

report = f""" {banner}# FraudShield: Credit Card Fraud Detection using Machine Learning

**Course:** Machine Learning Lab (AIM3101), Manipal University Jaipur
**Student:** [EDIT: name, registration no., section]  |  **Guide:** [EDIT]  |  **Date:** [EDIT]

---

## 1. Abstract and Problem Statement

Credit card fraud causes direct financial loss and erodes customer trust, but fraudulent transactions are rare
among millions of legitimate ones, so manual review does not scale. This project trains and compares three
supervised classifiers (Logistic Regression, Decision Tree, Random Forest) that estimate whether a transaction
resembles known fraud patterns. Because fraud is extremely rare ({ds['fraud_pct']:.3f}% of transactions), the study
focuses on imbalance-aware training and on metrics that remain meaningful under imbalance (precision, recall,
F1-score, PR-AUC). On a held-out test set the selected model, **{winner}**, achieved precision {pct(w['precision'])},
recall {pct(w['recall'])}, F1-score {pct(w['f1'])} and PR-AUC {pct(w['pr_auc'])}. A Streamlit dashboard,
*FraudShield*, demonstrates the models.

## 2. Objectives

1. Build a leakage-free ML pipeline for binary fraud classification.
2. Compare Logistic Regression, Decision Tree and Random Forest.
3. Study how imbalance handling (class weights, SMOTE, undersampling) changes results.
4. Evaluate with precision, recall, F1, confusion matrix, PR-AUC and ROC-AUC rather than accuracy alone.
5. Deliver an interactive demonstration of the trained model.

## 3. Dataset Description

* **Source:** {ds['name']} (Kaggle `mlg-ulb/creditcardfraud`): European cardholder transactions.
* **Size:** {ds['rows_raw']:,} rows x {ds['columns']} columns before cleaning; {ds['rows']:,} rows after removing
  {ds['duplicates_removed']:,} duplicate rows. Missing values found: {ds['missing_values']:,}.
* **Features:** `Time` (seconds since the first transaction), `V1`-`V28` (anonymised PCA components),
  `Amount` (transaction value). **Label:** `Class` (0 = legitimate, 1 = fraud).
* **Class imbalance:** {ds['fraud']:,} frauds vs {ds['legit']:,} legitimate transactions
  ({ds['fraud_pct']:.3f}% fraud). A model that predicts "legitimate" for everything would reach
  {ds['majority_class_accuracy']:.2%} accuracy while detecting no fraud, which is why accuracy is not used for model selection.
* **Limitation:** the data are historical and anonymised; a model trained on them estimates similarity to past
  fraud patterns, not whether a real transaction is actually fraudulent.

![Class distribution](figures/eda_class_distribution.png)
![Amount by class](figures/eda_amount_by_class.png)
![Feature correlation](figures/eda_feature_correlation.png)

## 4. Methodology

1. **Cleaning:** duplicate rows removed; schema validated.
2. **Splitting:** stratified 80/20 train/test split (random_state=42). A further stratified 20% of the training
   portion forms a validation set. Sizes: train {sp['train']:,} (fraud {sp['train_fraud']}), validation
   {sp['validation']:,} (fraud {sp['val_fraud']}), test {sp['test']:,} (fraud {sp['test_fraud']}).
3. **Preprocessing:** `Time` and `Amount` scaled with `RobustScaler`; fitted inside a pipeline on training data only.
4. **Imbalance handling (compared on the validation set):** no handling (baseline), class weights, SMOTE and
   random undersampling (both to a 1:10 minority:majority ratio). Resamplers sit inside the pipeline, so they are
   applied **only to training data** and never touch validation or test data (no information leakage).
5. **Models:** Logistic Regression (baseline, interpretable); Decision Tree (max_depth=8, min_samples_leaf=5; rule-based);
   Random Forest (100 trees, max_depth=12, min_samples_leaf=2; ensemble).
6. **Model selection:** for each algorithm the strategy with the highest validation PR-AUC was chosen, and the
   decision threshold was tuned to maximise F1 on the validation set. Each chosen configuration was then refitted on
   the full training portion and evaluated **once** on the test set. The final model is the one with the highest
   validation PR-AUC.

## 5. Implementation

| File | Role |
|---|---|
| `src/fraud_lib.py` | data loading, pipeline builder, metrics, threshold tuning |
| `src/train.py` | EDA, experiments, final evaluation, saving models and figures |
| `src/generate_report.py` | builds this report from measured results |
| `app.py` | Streamlit dashboard (overview, transaction checker, comparison, insights) |
| `models/*.joblib` | saved pipelines with their tuned thresholds |

## 6. Results

### 6.1 Imbalance-handling experiment (validation set)

{exp_table()}

### 6.2 Final comparison on the held-out test set (tuned threshold)

{table('tuned')}

### 6.3 Same models at the default 0.5 threshold

{table('default')}

### 6.4 Confusion matrices (tuned threshold; test set has {sp['test_fraud']} frauds)

{cm_table()}

![Confusion matrices](figures/confusion_matrices.png)
![PR and ROC curves](figures/pr_roc_curves.png)
![Model comparison](figures/model_comparison.png)

### 6.5 Overfitting check

{overfit_table()}

The largest train-to-test F1 gap is for **{largest_gap}** ({pct(T[largest_gap]['train']['f1'] - T[largest_gap]['tuned']['f1'])}).

### 6.6 What drives the selected model, and threshold trade-off

![Feature importance](figures/feature_importance.png)
![Threshold trade-off](figures/threshold_tradeoff.png)

## 7. Discussion, Conclusion and Limitations

* **Selected model:** {winner} (strategy: {T[winner]['strategy_label']}), chosen by validation PR-AUC.
  On the test set it caught {w['tp']} of {w['tp'] + w['fn']} frauds (recall {pct(w['recall'])}) with
  {w['fp']} false alarms (precision {pct(w['precision'])}).
* **Highest recall:** {best_recall} ({pct(T[best_recall]['tuned']['recall'])}). **Highest precision:** {best_prec}
  ({pct(T[best_prec]['tuned']['precision'])}). The right trade-off depends on the cost of a missed fraud versus a
  blocked genuine customer; the threshold lets a bank move along this curve.
* **Accuracy** ranges from {min(T[n]['tuned']['accuracy'] for n in fl.MODELS):.4%} to {max(T[n]['tuned']['accuracy'] for n in fl.MODELS):.4%}
  across the three models, against {ds['majority_class_accuracy']:.4%} for a model that never flags fraud, so accuracy
  alone cannot show how many frauds are actually detected.
* **Limitations:** the test set contains only {sp['test_fraud']} fraud cases, so metrics have wide uncertainty and a
  single split may be optimistic or pessimistic; features are anonymised, so business explanations are limited; the
  original dataset covers only two days of European card transactions, and fraud patterns drift over time.
* **Future work:** cross-validation / repeated splits, gradient boosting, cost-sensitive threshold selection,
  probability calibration, and monitoring for concept drift.

## 8. References

1. Dal Pozzolo, A., Caelen, O., Johnson, R. A., Bontempi, G. (2015). *Calibrating Probability with Undersampling for
   Unbalanced Classification.* IEEE CIDM. (Dataset: ULB Machine Learning Group, Kaggle `mlg-ulb/creditcardfraud`.)
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
3. Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR 16.
4. Lemaitre, G., Nogueira, F., Aridas, C. (2017). *Imbalanced-learn.* JMLR 18.
5. Course handout AIM3101: Logistic Regression, Decision Trees, Random Forests, evaluation metrics.
"""

out = fl.REPORT_DIR / "project_report.md"
out.write_text(report)
print(f"Wrote {out}")
