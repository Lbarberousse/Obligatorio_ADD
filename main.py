from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from model import load_or_train, CAT_COLS, NUM_COLS  
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).with_name("web")
STATIC_DIR.mkdir(exist_ok=True)

# Montamos el front en /app (evita pisar /predict)

CSV_PATH = "obesity_dataset_clean.csv"   
PKL_PATH = "modelo_obesidad.pkl"        

app = FastAPI()
app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True), name="app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod, restringí
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estructura de entrada: los mismos features que usaste al entrenar
class ObsIn(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

# Carga o entrena al iniciar
res = load_or_train(CSV_PATH, save_path=PKL_PATH, random_state=42)
if isinstance(res, tuple) and len(res) == 2:
    clf, METRICS = res
else:
    clf = res
    METRICS = {}

# Imprimir métricas básicas al iniciar (si están disponibles)
if METRICS:
    try:
        print("-------------- Metricas --------------")
        for k, v in METRICS.items():
            if k == 'confusion_matrix':
                print(f"{k}: (matrix {len(v)}x{len(v[0]) if v else 0})")
            else:
                print(f"{k}: {v}")
    except Exception:
        pass

@app.post("/predict")
def predict(obs: ObsIn):
    df = pd.DataFrame([obs.model_dump()])

    # sanity-check: columnas necesarias
    expected = set(CAT_COLS + NUM_COLS)
    missing = expected - set(df.columns)
    if missing:
        return {"error": f"Faltan columnas: {sorted(missing)}"}

    pred = clf.predict(df)[0]
    # obtiene el estimador final del pipeline y sus clases
    last = list(clf.named_steps.keys())[-1]
    classes = clf.named_steps[last].classes_
    probs = clf.predict_proba(df)[0]
    probs_dict = {c: float(p) for c, p in zip(classes, probs)}
    return {"pred": pred, "probs": probs_dict}

