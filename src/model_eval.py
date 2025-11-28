# src/model_eval.py
from pathlib import Path
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# resolve project root (parent of src)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "model.pkl"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# load data & model
test_data = pd.read_csv(DATA_DIR / "test_processed.csv")
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# predict & metrics
y_pred = model.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted"),
    "recall": recall_score(y_test, y_pred, average="weighted"),
    "f1_score": f1_score(y_test, y_pred, average="weighted"),
}

# save
with open(REPORTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Evaluation metrics saved to {REPORTS_DIR / 'metrics.json'}")

