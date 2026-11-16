# Dataset

The project expects the Kaggle Credit Card Fraud Detection dataset as `data/creditcard.csv`.

The original dataset is not bundled in this repository. Download it from Kaggle and place the CSV at that path. To test the pipeline without the real dataset, run:

```bash
python src/train.py --demo
```

This command generates a synthetic dataset locally. Synthetic demo metrics are for pipeline testing only and must not be presented as real-world model performance.
