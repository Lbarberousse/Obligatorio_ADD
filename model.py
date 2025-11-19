# model.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
import joblib
from pathlib import Path
import json

CAT_COLS = [
    'Gender','family_history_with_overweight','FAVC','CAEC',
    'SMOKE','SCC','CALC','MTRANS'
]
NUM_COLS = ['Age','Weight','Height','FCVC','NCP','CH2O','FAF','TUE']
TARGET = 'NObeyesdad'

def make_preprocess():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )

def make_estimator(random_state=42):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight=None,
        random_state=random_state,
    )

def make_pipeline(random_state=42):
    return Pipeline(steps=[
        ("prep", make_preprocess()),
        ("model", make_estimator(random_state=random_state)),
    ])

def train_from_df(df, random_state=42):
    df = df.dropna(subset=[TARGET]).copy()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    pipe = make_pipeline(random_state=random_state)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average='weighted', zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }
    return pipe, metrics

def load_or_train(csv_path: str, save_path: str | None = None, random_state=42):
    """
    Si existe save_path (.pkl), lo carga. Si no, entrena desde el CSV y opcionalmente guarda.
    """
    pkl = Path(save_path) if save_path else None

    if pkl and pkl.exists():
        modelo = joblib.load(pkl)
        metrics_path = pkl.with_name(pkl.stem + '_metrics.json')
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
                return modelo, metrics
            except Exception:
                pass

        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path, sep=';')
            modelo, metricas = train_from_df(df, random_state=random_state)
            try:
                joblib.dump(modelo, pkl)
                metrics_path.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding='utf-8')
            except Exception:
                pass
            return modelo, metricas

        return modelo

    df = pd.read_csv(csv_path, sep=';')
    modelo, metricas = train_from_df(df, random_state=random_state)

    if pkl:
        try:
            joblib.dump(modelo, pkl)
            metrics_path = pkl.with_name(pkl.stem + '_metrics.json')
            metrics_path.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass

    return modelo, metricas