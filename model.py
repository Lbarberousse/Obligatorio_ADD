import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
import json
from pathlib import Path

TRAIN_PATH = "data_train_clean.csv"
TEST_PATH  = "data_test_clean.csv"
MODEL_OUT  = "modelo_obesidad.pkl"
METRICS_OUT = "modelo_obesidad_metrics.json"

CAT_COLS = [
    'Gender','family_history_with_overweight','FAVC','CAEC',
    'SMOKE','SCC','CALC','MTRANS'
]
NUM_COLS = ['Age','Weight','Height','FCVC','NCP','CH2O','FAF','TUE']
TARGET = 'NObeyesdad'

# Construcores de pieline
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
        n_jobs=-1,
        random_state=random_state
    )

def make_pipeline(random_state=42):
    return Pipeline(steps=[
        ("prep", make_preprocess()),
        ("pca", PCA(n_components=0.95, random_state=random_state)),
        ("model", make_estimator(random_state))
    ])

def make_pipeline_without_pca(random_state=42):
    return Pipeline(steps=[
        ("prep", make_preprocess()),
        ("model", make_estimator(random_state))
    ])


def train_and_evaluate(train_df, test_df, random_state=42):

    # Separar X e y
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Modelo CON PCA
    pipe_pca = make_pipeline(random_state)
    pipe_pca.fit(X_train, y_train)

    y_pred_pca = pipe_pca.predict(X_test)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    prec_pca, rec_pca, f1_pca, _ = precision_recall_fscore_support(
        y_test, y_pred_pca, average='weighted', zero_division=0
    )

    print("\n========== MODELO CON PCA ==========")
    print(classification_report(y_test, y_pred_pca))

    metrics_pca = {
        "accuracy": acc_pca,
        "precision_weighted": prec_pca,
        "recall_weighted": rec_pca,
        "f1_weighted": f1_pca
    }

    # Modelo SIN PCA
    pipe_no_pca = make_pipeline_without_pca(random_state)
    pipe_no_pca.fit(X_train, y_train)

    y_pred_no_pca = pipe_no_pca.predict(X_test)
    acc_no_pca = accuracy_score(y_test, y_pred_no_pca)
    prec_no_pca, rec_no_pca, f1_no_pca, _ = precision_recall_fscore_support(
        y_test, y_pred_no_pca, average='weighted', zero_division=0
    )

    print("\n========== MODELO SIN PCA ==========")
    print(classification_report(y_test, y_pred_no_pca))

    metrics_no_pca = {
        "accuracy": acc_no_pca,
        "precision_weighted": prec_no_pca,
        "recall_weighted": rec_no_pca,
        "f1_weighted": f1_no_pca
    }

    # Seleccionar mejor modelo
    if metrics_no_pca["f1_weighted"] >= metrics_pca["f1_weighted"]:
        print("\n Usamos el modelo SIN PCA (mejor F1)")
        return pipe_no_pca, metrics_no_pca
    else:
        print("\n Usamos el modelo CON PCA (mejor F1)")
        return pipe_pca, metrics_pca

# MAIN
if __name__ == "__main__":
    print("Cargando datasets limpio de TRAIN y TEST...")
    train_df = pd.read_csv(TRAIN_PATH, sep=";")
    test_df  = pd.read_csv(TEST_PATH, sep=";")

    print("Entrenando y evaluando modelo...")
    modelo, metricas = train_and_evaluate(train_df, test_df, random_state=42)

    # Guardar modelo
    joblib.dump(modelo, MODEL_OUT)
    print(f"\nModelo guardado en {MODEL_OUT}")

    # Guardar métricas
    Path(METRICS_OUT).write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Métricas guardadas en {METRICS_OUT}")
