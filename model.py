# model.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

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
    # asegurate de que no haya NaN en TARGET
    df = df.dropna(subset=[TARGET]).copy()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # opcional: estratificar para estabilidad
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipe = make_pipeline(random_state=random_state)
    pipe.fit(X_train, y_train)
    return pipe

def load_or_train(csv_path: str, save_path: str | None = None, random_state=42):
    """
    Si existe save_path (.pkl), lo carga. Si no, entrena desde el CSV y opcionalmente guarda.
    """
    pkl = Path(save_path) if save_path else None
    if pkl and pkl.exists():
        return joblib.load(pkl)

    # leé tu CSV limpio (ajustá separador si es ';')
    df = pd.read_csv(csv_path, sep=';')
    pipe = train_from_df(df, random_state=random_state)

    if pkl:
        joblib.dump(pipe, pkl)
    return pipe
