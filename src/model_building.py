# src/model_building.py
from pathlib import Path
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Paths relative to project root (parent of src)
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load training data
train_data = pd.read_csv(PROC_DIR / "train_processed.csv")
#X_train = train_data.iloc[:, :-1].values
#y_train = train_data.iloc[:, -1].values

X_train = train_data.drop(columns=['Potability'],axis=1)
y_train = train_data['Potability']


# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model where DVC expects it
with open(MODEL_DIR / "model.pkl", "wb") as f:
    pickle.dump(clf, f)

print(f"Saved model to {MODEL_DIR / 'model.pkl'}")
