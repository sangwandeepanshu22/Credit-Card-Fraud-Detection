"""Generate a SYNTHETIC dataset with the same schema as the Kaggle creditcard.csv.

Purpose: lets you test the whole pipeline (training, dashboard) without the real data.
Metrics produced on this data are NOT meaningful for your report - use the real dataset.
"""
import numpy as np
import pandas as pd

from fraud_lib import DATA_DIR, FEATURES, TARGET


def generate(n: int = 40000, fraud_rate: float = 0.006, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud
    scales = np.linspace(2.0, 0.6, 28)

    legit = rng.normal(0, scales, size=(n_legit, 28))
    fraud = rng.normal(0, scales * 1.4, size=(n_fraud, 28))
    shifts = {1: -3.0, 3: -3.5, 4: 2.5, 7: -2.5, 10: -3.5, 11: 2.5, 12: -3.5, 14: -4.5, 16: -3.0, 17: -3.5}
    weak = rng.random(n_fraud) < 0.2  # 20% of frauds are subtle -> overlap with legit
    for k, s in shifts.items():
        fraud[:, k - 1] += np.where(weak, s * 0.25, s)

    X = np.vstack([legit, fraud])
    y = np.r_[np.zeros(n_legit, int), np.ones(n_fraud, int)]
    amount = np.r_[rng.lognormal(3.5, 1.3, n_legit), rng.lognormal(3.0, 1.8, n_fraud)].round(2)
    time = rng.uniform(0, 172792, n).round(0)

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df[TARGET] = y
    df = df.sample(frac=1, random_state=seed).sort_values("Time").reset_index(drop=True)
    return df[FEATURES + [TARGET]]


if __name__ == "__main__":
    out = DATA_DIR / "creditcard_demo.csv"
    generate().to_csv(out, index=False)
    print(f"Wrote synthetic demo data to {out}")
