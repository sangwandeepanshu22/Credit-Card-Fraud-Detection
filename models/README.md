# Trained model artifacts

This directory is populated locally by `python src/train.py`.

The training script saves:
- `logistic_regression.joblib`
- `decision_tree.joblib`
- `random_forest.joblib`
- `fraud_model.joblib` (the selected model bundle)

These binary artifacts are intentionally not tracked in Git. Generate them locally with the demo data or the Kaggle dataset as described in the root README.
