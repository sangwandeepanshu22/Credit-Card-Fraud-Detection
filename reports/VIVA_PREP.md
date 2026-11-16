# Viva preparation

**1. Why is accuracy misleading here?** Only ~0.17% of transactions are fraud, so predicting "legitimate" for
everything gives ~99.8% accuracy with zero frauds detected. Use precision, recall, F1 and PR-AUC.

**2. Precision vs recall?** Precision = TP/(TP+FP): of flagged transactions, how many were really fraud.
Recall = TP/(TP+FN): of all real frauds, how many we caught. Fraud teams usually want high recall at acceptable precision.

**3. Why these three algorithms?** Logistic Regression: simple, interpretable baseline. Decision Tree: readable rules.
Random Forest: ensemble of trees, reduces variance, usually captures non-linear patterns. All are in the AIM3101 handout.

**4. What is overfitting and how did you check?** The model memorises training data and fails on new data. I compared
train vs test F1 / PR-AUC (see the overfitting table), limited tree depth and leaf size, and selected on a validation set.

**5. What is SMOTE and why only on training data?** It creates synthetic minority samples by interpolating between
neighbouring fraud cases. Applying it before splitting puts synthetic copies of test-set neighbours into training
(information leakage) and inflates test scores. It is inside the pipeline, so it runs only when fitting.

**6. What does a confusion matrix tell us?** Counts of TN, FP, FN, TP. FN = missed frauds, FP = false alarms.

**7. Can the model guarantee a transaction is fraud?** No. It outputs a probability based on patterns in historical,
anonymised data; real systems add human review and rules.

**8. Why PR-AUC over ROC-AUC?** With extreme imbalance ROC-AUC can look excellent because true negatives dominate;
PR-AUC focuses on the rare positive class.

**9. Why did you tune the threshold on validation, not test?** Tuning on test would leak test information and give
optimistic results. The test set is used once.

**10. Why RobustScaler, and why not scale V1-V28?** Amount has heavy outliers; the median/IQR scaling resists them.
V1-V28 are already PCA outputs on comparable scales.

**11. Limitations?** Few frauds, anonymised features, two days of data, concept drift, one split (add cross-validation).

**12. How would you improve it?** Gradient boosting (XGBoost/LightGBM), cross-validation, cost-sensitive threshold,
calibration, drift monitoring.
