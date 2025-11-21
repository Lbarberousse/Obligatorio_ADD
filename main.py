from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

STATIC_DIR = Path(__file__).with_name("web")
STATIC_DIR.mkdir(exist_ok=True)

# Ruta al modelo ganador (el que guarda tu model.py)
MODEL_PATH = "modelo_obesidad.pkl"

# Cargar el pipeline completo (preprocesado + modelo)
model = joblib.load(MODEL_PATH)

# Campos de entrada: deben coincidir con CAT_COLS + NUM_COLS
class ObesityInput(BaseModel):
    # Categóricas
    Gender: str
    family_history_with_overweight: str
    FAVC: str
    CAEC: str
    SMOKE: str
    SCC: str
    CALC: str
    MTRANS: str

    # Numéricas
    Age: float
    Weight: float
    Height: float
    FCVC: float
    NCP: float
    CH2O: float
    FAF: float
    TUE: float

app = FastAPI(
    title="API Clasificación Obesidad",
    description="Clasifica en Obeso / No_Obeso usando el mejor modelo entrenado.",
    version="1.0.0",
)
app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True), name="app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod, restringí
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "API de clasificación de obesidad",
        "model_loaded": type(model).__name__,
    }

@app.post("/predict")
def predict_obesity(input_data: ObesityInput):
    # Convertir el input a DataFrame con las mismas columnas que el pipeline espera
    df = pd.DataFrame([input_data.dict()])

    # Predicción (el pipeline ya hace OneHot, escalado, PCA si corresponde, etc.)
    y_pred = model.predict(df)[0]

    # Probabilidades por clase (si el modelo las expone)
    proba_dict = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        labels = model.classes_
        proba_dict = {str(label): float(p) for label, p in zip(labels, probs)}

    return {
        "prediction": y_pred,   # p.ej. "Obeso" o "No_Obeso"
        "probabilities": proba_dict,
    }
