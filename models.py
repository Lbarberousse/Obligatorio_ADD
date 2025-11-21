import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
import joblib
import json
from pathlib import Path

DATA_PATH   = "data_clean.csv"
MODEL_OUT   = "modelo_obesidad.pkl"
METRICS_OUT = "modelo_obesidad_metrics.json"

CAT_COLS = [
    'Gender','family_history_with_overweight','FAVC','CAEC',
    'SMOKE','SCC','CALC','MTRANS'
]
NUM_COLS = ['Age','Weight','Height','FCVC','NCP','CH2O','FAF','TUE']
TARGET   = 'NObeyesdad'
#TARGET = 'ObesityBinary'

RANDOM_STATE = 42

def make_preprocess():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )

def make_estimator(model_type: str, random_state: int = RANDOM_STATE):
    """
    Devuelve el estimador según el tipo:
      - 'rf'     - RandomForest
      - 'knn'    - KNN
      - 'logreg' - Regresión Logística
    """
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=17,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced"
        )
    elif model_type == "knn":
        return KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski"
        )
    elif model_type == "logreg":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced"
        )
    else:
        raise ValueError(f"model_type desconocido: {model_type}")

def build_pipeline(model_type: str, use_pca: bool, random_state: int = RANDOM_STATE) -> Pipeline:
    """
    Construye un pipeline:
      prep -> (opcional PCA) -> modelo
    """
    steps = [("prep", make_preprocess())]

    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=random_state)))

    steps.append(("model", make_estimator(model_type, random_state)))
    return Pipeline(steps=steps)

def evaluate_pipeline(name, pipe, X, y, cv):
    print(f"\n ------------ {name.upper()} ------------")

    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted"
    }

    scores = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    acc_mean = scores["test_accuracy"].mean()
    acc_std  = scores["test_accuracy"].std()
    f1_mean  = scores["test_f1_weighted"].mean()
    f1_std   = scores["test_f1_weighted"].std()

    print(f"Accuracy CV (media ± std): {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"F1_weighted CV (media ± std): {f1_mean:.3f} ± {f1_std:.3f}")

    # Classification report usando predicciones de CV
    print("\nClassification report (CV):")
    y_pred_cv = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    print(classification_report(y, y_pred_cv, zero_division=0))

    metrics = {
        "name": name,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "f1_weighted_mean": f1_mean,
        "f1_weighted_std": f1_std
    }

    return metrics

if __name__ == "__main__":
    print("=== ENTRENAMIENTO BINARIO: OBESO vs NO_OBESO ===")
    print(f"Leyendo dataset limpio: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=";")

    # Asegurarnos de que la variable original no tenga NaN
    print("Nulos en NObeyesdad ANTES de filtrar:", df["NObeyesdad"].isna().sum())
    df = df.dropna(subset=["NObeyesdad"])
    print("Nulos en NObeyesdad DESPUÉS de filtrar:", df["NObeyesdad"].isna().sum())

    # ---- NUEVO: crear target binario ----
    obese_classes = [
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    ]

    def to_binary_label(label):
        if label in obese_classes:
            return "Obeso"
        else:
            return "No_Obeso"

    df["ObesityBinary"] = df["NObeyesdad"].apply(to_binary_label)

    # Por si acaso:
    print(df["ObesityBinary"].value_counts())

    # Armamos X e y
    X = df[CAT_COLS + NUM_COLS]
    y = df[TARGET]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Definimos qué modelos vamos a probar
    model_configs = {
        "rf_con_pca": {
            "desc": "RandomForest con PCA",
            "model_type": "rf",
            "use_pca": True,
        },
        "rf_sin_pca": {
            "desc": "RandomForest sin PCA",
            "model_type": "rf",
            "use_pca": False,
        },
        "knn_sin_pca": {
            "desc": "KNN (k=7) sin PCA",
            "model_type": "knn",
            "use_pca": False,
        },
        "logreg_sin_pca": {
            "desc": "Regresión Logística sin PCA",
            "model_type": "logreg",
            "use_pca": False,
        },
    }

    all_metrics = {}
    best_key = None
    best_f1 = -1.0

    # Evaluar cada modelo
    for key, cfg in model_configs.items():
        print(f"\n>>> Evaluando modelo: {cfg['desc']} ({key})")
        pipe = build_pipeline(cfg["model_type"], cfg["use_pca"], RANDOM_STATE)
        metrics = evaluate_pipeline(cfg["desc"], pipe, X, y, cv)

        all_metrics[key] = {
            **cfg,
            **metrics
        }

        # Elegimos mejor F1 weighted
        if metrics["f1_weighted_mean"] > best_f1:
            best_f1 = metrics["f1_weighted_mean"]
            best_key = key

    # Resumen del mejor modelo
    best_cfg = all_metrics[best_key]
    print(f"\n>>> MEJOR MODELO SEGÚN F1 WEIGHTED: {best_cfg['desc']} ({best_key}) <<<")
    print(f"F1_weighted_mean = {best_cfg['f1_weighted_mean']:.3f}")

    # Entrenar mejor modelo sobre TODO el dataset limpio
    print(f"\nEntrenando modelo final ({best_cfg['desc']}) sobre todo el dataset...")
    final_model = build_pipeline(
        best_cfg["model_type"],
        best_cfg["use_pca"],
        RANDOM_STATE
    )
    final_model.fit(X, y)

    # Guardar modelo
    joblib.dump(final_model, MODEL_OUT)
    print(f"Modelo final guardado en: {MODEL_OUT}")

    # Guardar métricas de todos los modelos y el ganador
    output_metrics = {
        "models": all_metrics,
        "best_model_key": best_key,
        "best_model_desc": best_cfg["desc"],
    }

    Path(METRICS_OUT).write_text(
        json.dumps(output_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Métricas de cross-validation guardadas en: {METRICS_OUT}")

    # ===== PODIO DE MODELOS (según F1 weighted) =====
    print("\n===== PODIO DE MODELOS (según F1 weighted) =====")
    ordenados = sorted(
        all_metrics.items(),
        key=lambda kv: kv[1]["f1_weighted_mean"],
        reverse=True
    )

    for i, (key, info) in enumerate(ordenados, start=1):
        print(f"{i}º lugar: {info['desc']} ({key})")
        print(f"   F1 weighted medio: {info['f1_weighted_mean']:.3f}")
        print(f"   Accuracy medio:    {info['accuracy_mean']:.3f}")
        print()