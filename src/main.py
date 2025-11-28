
'''

from fastapi import FastAPI
from routers import items, users       # or `.routers` if running from root
from data_model import Water
from pathlib import Path
import pickle
import numpy as np

app = FastAPI(title="ML Pipeline API")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"

model = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

app.include_router(items.router)
app.include_router(users.router)

@app.get("/")
def home():
    return {
        "message": "🚀 Welcome to Parimal's ML Pipeline API!",
        "model_status": "Loaded ✅" if model else "Not loaded ❌",
        "instructions": "POST /predict with JSON body to get a prediction."
    }

@app.post("/predict")
def model_predict(data: Water):
    if not model:
        return {"error": "Model not loaded."}
    
    input_data = np.array([[
        data.ph,
        data.Hardness,
        data.Solids,
        data.Chloramines,
        data.Sulfate,
        data.Conductivity,
        data.Organic_carbon,
        data.Trihalomethanses,
        data.Turbidity
    ]])
    
    prediction = model.predict(input_data)
    

    if predicted_value == 1:
        result = "The water is safe to drink."
    else:
        result = "The water is not safe to drink."
    

'''



from fastapi import FastAPI, HTTPException
from src.routers import items, users
from src.data_model import Water              # your Pydantic model
from pathlib import Path
import pickle
import numpy as np


# App setup

app = FastAPI(title="ML Pipeline API")

# Project root:  ml_pipeline/
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"


# Load model once at startup

model = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Routers
app.include_router(items.router)
app.include_router(users.router) 

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def home():
    """
    Health-check / welcome endpoint.
    """
    return {
        "message": "🚀 Welcome to Parimal's ML Pipeline API!",
        "model_status": "Loaded ✅" if model is not None else "Not loaded ❌",
        "instructions": "POST /predict with JSON body to get a prediction."
    }


@app.post("/predict")
def predict(data: Water):
    """
    Run the water potability model.
    """
    if model is None:
        # better than returning plain dict → shows as 500 error in docs/clients
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Keep feature order EXACTLY as in training
    X = np.array([[
        data.ph,
        data.Hardness,
        data.Solids,
        data.Chloramines,
        data.Sulfate,
        data.Conductivity,
        data.Organic_carbon,
        data.Trihalomethanes,
        data.Turbidity,
    ]], dtype=float)

    # Prediction
    y = model.predict(X)[0]
    y_int = int(y)

    # Optional probabilities
    proba_out = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        proba_out = proba.tolist()

    # Human-readable message
    if y_int == 1:
        result_msg = "The water is safe to drink."
    else:
        result_msg = "The water is not safe to drink."

    return {
        "prediction": y_int,
        "result": result_msg,
        "proba": proba_out
    }


'''

pandas==2.2.2
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
six==1.16.0
threadpoolctl==3.5.0
tzdata==2024.1
fastapi==0.115.6
uvicorn[standard]==0.32.0
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
scipy==1.13.1s



'''