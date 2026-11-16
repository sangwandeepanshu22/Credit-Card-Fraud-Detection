# FraudShield: Credit Card Fraud Detection

Machine Learning Lab mini-project comparing Logistic Regression, Decision Tree, and Random Forest for highly imbalanced credit-card transaction data. The Streamlit dashboard supports transaction scoring, model comparison, and evaluation insights.

> **Important:** The tracked repository contains the source code and documentation. Generated datasets, trained `.joblib` models, CSV evaluation outputs, and plots are not committed; the training script creates them locally.

## Project structure

```text
.
├── app.py
├── requirements.txt
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   └── fraud_detection.ipynb
├── src/
│   ├── fraud_lib.py
│   ├── train.py
│   ├── make_demo_data.py
│   └── generate_report.py
└── reports/
    ├── figures/
    │   └── README.md
    ├── project_report.md
    └── VIVA_PREP.md
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the project with synthetic demo data

This is the quickest way to test the complete training pipeline and dashboard. The script generates synthetic data and writes model files, metrics, CSV samples, and plots into the local `data/`, `models/`, and `reports/` directories.

```bash
python src/train.py --demo
streamlit run app.py
```

Demo metrics are only for testing the pipeline. Do not present them as real-world performance.

## Run with the Kaggle dataset

1. Download the Kaggle **Credit Card Fraud Detection** dataset (`mlg-ulb/creditcardfraud`).
2. Put the file at `data/creditcard.csv`.
3. Train and generate the report:

```bash
python src/train.py
python src/generate_report.py
streamlit run app.py
```

The training process compares three classifiers across four imbalance strategies: baseline, class weighting, SMOTE, and random undersampling. Preprocessing and resampling are placed inside the model pipeline to prevent validation/test leakage. The best strategy and decision threshold for each classifier are selected using the validation set; final results are then calculated on the held-out test set.

## Evaluation

The pipeline reports precision, recall, F1-score, PR-AUC, ROC-AUC, and confusion-matrix counts. Accuracy alone can be misleading when fraud transactions are rare.

## Notes and limitations

- `V1`-`V28` are anonymised PCA components; the dashboard cannot accept card numbers, merchant names, or locations.
- Predictions are model estimates based on historical data, not verified fraud determinations.
- The original dataset is not included. See `data/README.md` for setup.
- Generated files are excluded by `.gitignore`; create them by running the training command.
- `reports/project_report.md` is an example report generated from synthetic demo data. Regenerate it after training on the real Kaggle data before using its metrics.
- See `notebooks/fraud_detection.ipynb` for a step-by-step exploration of the modelling workflow.

## Academic viva notes

See [reports/VIVA_PREP.md](reports/VIVA_PREP.md).
