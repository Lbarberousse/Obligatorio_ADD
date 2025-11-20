# model.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
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
        # Añadimos PCA como un paso de transformación de características.
        # n_components=0.95 significa que PCA seleccionará automáticamente el número
        # de componentes necesarios para explicar el 95% de la varianza.
        ("pca", PCA(n_components=0.95, random_state=random_state)),
        ("model", make_estimator(random_state=random_state))
    ])

def make_pipeline_without_pca(random_state=42):
    """Pipeline de referencia sin el paso de PCA."""
    return Pipeline(steps=[
        ("prep", make_preprocess()),
        ("model", make_estimator(random_state=random_state))
    ])

def train_from_df(df, random_state=42):
    # chequeamos que no haya NaN en TARGET
    df = df.dropna(subset=[TARGET]).copy()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # --- Entrenamiento y evaluación de ambos pipelines ---

    # Modelo CON PCA
    pipe_with_pca = make_pipeline(random_state=random_state)
    pipe_with_pca.fit(X_train, y_train)
    y_pred_pca = pipe_with_pca.predict(X_val)
    accuracy_pca = accuracy_score(y_val, y_pred_pca)
    precision_pca, recall_pca, f1_pca, _ = precision_recall_fscore_support(
        y_val, y_pred_pca, average='weighted', zero_division=0
    )

    print("\nRendimiento del Modelo CON PCA")
    print("================================")
    print(classification_report(y_val, y_pred_pca))
    metrics_pca = {
        "accuracy": accuracy_pca,
        "precision_weighted": precision_pca,
        "recall_weighted": recall_pca,
        "f1_weighted": f1_pca,
    }

    # Modelo SIN PCA
    pipe_without_pca = make_pipeline_without_pca(random_state=random_state)
    pipe_without_pca.fit(X_train, y_train)
    y_pred_no_pca = pipe_without_pca.predict(X_val)
    accuracy_no_pca = accuracy_score(y_val, y_pred_no_pca)
    precision_no_pca, recall_no_pca, f1_no_pca, _ = precision_recall_fscore_support(
        y_val, y_pred_no_pca, average='weighted', zero_division=0
    )
    
    print("\nRendimiento del Modelo SIN PCA")
    print("================================")
    print(classification_report(y_val, y_pred_no_pca))
    metrics_no_pca = {
        "accuracy": accuracy_no_pca,
        "precision_weighted": precision_no_pca,
        "recall_weighted": recall_no_pca,
        "f1_weighted": f1_no_pca,
    }

    # Comparar modelos y devolver el mejor
    if metrics_no_pca["f1_weighted"] >= metrics_pca["f1_weighted"]:
        print("\n Seleccionamos el modelo SIN PCA debido a su mayor rendimiento.")
        return pipe_without_pca, metrics_no_pca
    else:
        print("\n Seleccionamos el modelo CON PCA debido a su mayor rendimiento.")
        return pipe_with_pca, metrics_pca

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
        return modelo, {} # Devuelve métricas vacías si el json no existe o falla

    
    df = pd.read_csv(csv_path, sep=';')
    modelo, metricas = train_from_df(df, random_state=random_state)

    if pkl:
        try:
            joblib.dump(modelo, pkl)
            metrics_path = pkl.with_name(pkl.stem + '_metrics.json')
            metrics_path.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding='utf-8')
        except Exception as e:
            print(f"[ERROR] No se pudo guardar el modelo o las métricas: {e}")

    return modelo, metricas

if __name__ == "__main__":
    df = pd.read_csv("obesity_dataset_clean.csv", sep=';')
    train_from_df(df, random_state=42)