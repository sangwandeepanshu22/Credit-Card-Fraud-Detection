> **WARNING: this report was generated from SYNTHETIC demo data. Re-run `python src/train.py` on the real Kaggle dataset, then re-run this script.**

# FraudShield: Credit Card Fraud Detection using Machine Learning

**Course:** Machine Learning Lab (AIM3101), Manipal University Jaipur
**Student:** [EDIT: name, registration no., section]  |  **Guide:** [EDIT]  |  **Date:** [EDIT]

---

## 1. Abstract and Problem Statement

Credit card fraud causes direct financial loss and erodes customer trust, but fraudulent transactions are rare
among millions of legitimate ones, so manual review does not scale. This project trains and compares three
supervised classifiers (Logistic Regression, Decision Tree, Random Forest) that estimate whether a transaction
resembles known fraud patterns. Because fraud is extremely rare (0.600% of transactions), the study
focuses on imbalance-aware training and on metrics that remain meaningful under imbalance (precision, recall,
F1-score, PR-AUC). On a held-out test set the selected model, **Random Forest**, achieved precision 1.0000,
recall 0.6667, F1-score 0.8000 and PR-AUC 0.7826. A Streamlit dashboard,
*FraudShield*, demonstrates the models.

## 2. Objectives

1. Build a leakage-free ML pipeline for binary fraud classification.
2. Compare Logistic Regression, Decision Tree and Random Forest.
3. Study how imbalance handling (class weights, SMOTE, undersampling) changes results.
4. Evaluate with precision, recall, F1, confusion matrix, PR-AUC and ROC-AUC rather than accuracy alone.
5. Deliver an interactive demonstration of the trained model.

## 3. Dataset Description

* **Source:** SYNTHETIC demo data (Kaggle `mlg-ulb/creditcardfraud`): European cardholder transactions.
* **Size:** 40,000 rows x 31 columns before cleaning; 40,000 rows after removing
  0 duplicate rows. Missing values found: 0.
* **Features:** `Time` (seconds since the first transaction), `V1`-`V28` (anonymised PCA components),
  `Amount` (transaction value). **Label:** `Class` (0 = legitimate, 1 = fraud).
* **Class imbalance:** 240 frauds vs 39,760 legitimate transactions
  (0.600% fraud). A model that predicts "legitimate" for everything would reach
  99.40% accuracy while detecting no fraud, which is why accuracy is not used for model selection.
* **Limitation:** the data are historical and anonymised; a model trained on them estimates similarity to past
  fraud patterns, not whether a real transaction is actually fraudulent.

![Class distribution](figures/eda_class_distribution.png)
![Amount by class](figures/eda_amount_by_class.png)
![Feature correlation](figures/eda_feature_correlation.png)

## 4. Methodology

1. **Cleaning:** duplicate rows removed; schema validated.
2. **Splitting:** stratified 80/20 train/test split (random_state=42). A further stratified 20% of the training
   portion forms a validation set. Sizes: train 25,600 (fraud 154), validation
   6,400 (fraud 38), test 8,000 (fraud 48).
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

| Model | Strategy | Val PR-AUC | Val F1 (best thr.) | Val recall (best thr.) |
|---|---|---|---|---|
| Logistic Regression | No handling (baseline) | 0.8711 | 0.9296 | 0.8684 |
| Logistic Regression | Class weights | 0.8679 | 0.9143 | 0.8421 |
| Logistic Regression | SMOTE (train only) | 0.8698 | 0.9143 | 0.8421 |
| Logistic Regression | Random undersampling (train only) | 0.8701 | 0.9143 | 0.8421 |
| Decision Tree | No handling (baseline) | 0.6715 | 0.7941 | 0.7105 |
| Decision Tree | Class weights | 0.3748 | 0.5500 | 0.5789 |
| Decision Tree | SMOTE (train only) | 0.6257 | 0.7500 | 0.7105 |
| Decision Tree | Random undersampling (train only) | 0.4200 | 0.6154 | 0.7368 |
| Random Forest | No handling (baseline) | 0.8635 | 0.8857 | 0.8158 |
| Random Forest | Class weights | 0.8812 | 0.9296 | 0.8684 |
| Random Forest | SMOTE (train only) | 0.8960 | 0.9143 | 0.8421 |
| Random Forest | Random undersampling (train only) | 0.8967 | 0.9143 | 0.8421 |

### 6.2 Final comparison on the held-out test set (tuned threshold)

| Model | Strategy | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | No handling (baseline) | 1.0000 | 0.6875 | 0.8148 | 0.7748 | 0.9082 |
| Decision Tree | No handling (baseline) | 0.9565 | 0.4583 | 0.6197 | 0.4949 | 0.7395 |
| Random Forest | Random undersampling (train only) | 1.0000 | 0.6667 | 0.8000 | 0.7826 | 0.9819 |

### 6.3 Same models at the default 0.5 threshold

| Model | Strategy | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | No handling (baseline) | 1.0000 | 0.6667 | 0.8000 | 0.7748 | 0.9082 |
| Decision Tree | No handling (baseline) | 0.8276 | 0.5000 | 0.6234 | 0.4949 | 0.7395 |
| Random Forest | Random undersampling (train only) | 0.9211 | 0.7292 | 0.8140 | 0.7826 | 0.9819 |

### 6.4 Confusion matrices (tuned threshold; test set has 48 frauds)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 7,952 | 0 | 15 | 33 |
| Decision Tree | 7,951 | 1 | 26 | 22 |
| Random Forest | 7,952 | 0 | 16 | 32 |

![Confusion matrices](figures/confusion_matrices.png)
![PR and ROC curves](figures/pr_roc_curves.png)
![Model comparison](figures/model_comparison.png)

### 6.5 Overfitting check

| Model | Train F1 | Test F1 | Gap | Train PR-AUC | Test PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8596 | 0.8148 | 0.0448 | 0.8286 | 0.7748 |
| Decision Tree | 0.7548 | 0.6197 | 0.1351 | 0.7308 | 0.4949 |
| Random Forest | 0.8870 | 0.8000 | 0.0870 | 0.9153 | 0.7826 |

The largest train-to-test F1 gap is for **Decision Tree** (0.1351).

### 6.6 What drives the selected model, and threshold trade-off

![Feature importance](figures/feature_importance.png)
![Threshold trade-off](figures/threshold_tradeoff.png)

## 7. Discussion, Conclusion and Limitations

* **Selected model:** Random Forest (strategy: Random undersampling (train only)), chosen by validation PR-AUC.
  On the test set it caught 32 of 48 frauds (recall 0.6667) with
  0 false alarms (precision 1.0000).
* **Highest recall:** Logistic Regression (0.6875). **Highest precision:** Logistic Regression
  (1.0000). The right trade-off depends on the cost of a missed fraud versus a
  blocked genuine customer; the threshold lets a bank move along this curve.
* **Accuracy** ranges from 99.6625% to 99.8125%
  across the three models, against 99.4000% for a model that never flags fraud, so accuracy
  alone cannot show how many frauds are actually detected.
* **Limitations:** the test set contains only 48 fraud cases, so metrics have wide uncertainty and a
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
