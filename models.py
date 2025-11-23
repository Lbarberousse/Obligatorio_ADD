import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import joblib

# --- CONSTANTES ---
DATA_PATH   = "data_clean.csv" # Asegurarse que el nombre es el correcto
OUTPUTS_DIR = Path("outputs")

CAT_COLS = [
    'Gender','family_history_with_overweight','FAVC','CAEC',
    'SMOKE','SCC','CALC','MTRANS'
]
NUM_COLS = ['Age','Weight','Height','FCVC','NCP','CH2O','FAF','TUE']

RANDOM_STATE = 42

def make_preprocess():
    # Pipeline para preprocesar las columnas numéricas: imputar con mediana y luego escalar
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # Pipeline para preprocesar las columnas categóricas: imputar con moda y luego OneHotEncoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS)
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
            n_estimators=500,
            max_depth=15,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced"
        )
    elif model_type == "knn":
        return KNeighborsClassifier(
            n_neighbors=3,
            weights="distance",
            metric="minkowski"
        )
    elif model_type == "logreg":
        return LogisticRegression(
            C=100,
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

def plot_model_comparison(all_metrics: dict, output_path: Path, plot_title: str):
    """
    Genera y guarda un gráfico de barras comparando el F1-Score de los modelos.
    """
    # Convertir el diccionario de métricas a un DataFrame para facilitar el ploteo
    metrics_list = []
    for key, data in all_metrics.items():
        metrics_list.append({
            "Model": data["desc"],
            "F1-Score": data["f1_weighted_mean"]
        })
    
    df_plot = pd.DataFrame(metrics_list).sort_values("F1-Score", ascending=False)

    # Crear la gráfica
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    bars = sns.barplot(
        x="F1-Score",
        y="Model",
        data=df_plot,
        ax=ax,
        palette="viridis"
    )

    # Añadir etiquetas de valor en cada barra
    for bar in bars.patches:
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.3f}',
            va='center'
        )

    ax.set_title(plot_title, fontsize=16, weight='bold')
    ax.set_xlabel("F1-Score Ponderado (CV)", fontsize=12)
    ax.set_ylabel("") # El nombre de los modelos ya está en el eje y
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Gráfica de comparación de modelos guardada en: {output_path}")

def train_and_evaluate_for_target(target_name: str, df: pd.DataFrame):
    """
    Ejecuta el ciclo completo de entrenamiento y evaluación para un target específico.
    """
    target_suffix = "multiclass" if target_name == "NObeyesdad" else "binary"
    MODEL_OUT = f"modelo_obesidad_{target_suffix}.pkl"
    METRICS_OUT = f"modelo_obesidad_metrics_{target_suffix}.json"

    print(f"\n{'='*20} ENTRENAMIENTO DEL MODELO ({target_suffix.upper()}) {'='*20}")
    
    # Armamos X e y
    X = df[CAT_COLS + NUM_COLS]
    y = df[target_name]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

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
            "desc": "KNN (k=3) sin PCA",
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

    # Generar y guardar la gráfica de comparación
    plot_model_comparison(
        all_metrics=all_metrics,
        output_path=OUTPUTS_DIR / f"model_comparison_{target_suffix}.png",
        plot_title=f"Comparación de Modelos - Tarea {target_suffix.upper()}"
    )

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

if __name__ == "__main__":
    print("Leyendo y preparando el dataset...")
    # Asegurarse de que el directorio de salida exista
    OUTPUTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep=";")

    # 1. Limpiar filas donde la variable objetivo original ('NObeyesdad') es nula.
    #    Esto es crucial para asegurar la consistencia de ambos targets.
    initial_rows = len(df)
    df = df.dropna(subset=["NObeyesdad"])
    print(f"Se eliminaron {initial_rows - len(df)} filas con target 'NObeyesdad' nulo.")

    # 2. Crear la variable objetivo binaria a partir de la multiclase ya limpia.
    obese_classes = [
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    ]
    df["ObesityBinary"] = df["NObeyesdad"].apply(
        lambda label: "Obeso" if label in obese_classes else "No_Obeso"
    )

    # Lista de targets a entrenar
    targets_to_train = ["NObeyesdad", "ObesityBinary"]

    # Ejecutar el proceso para cada target
    for target in targets_to_train:
        train_and_evaluate_for_target(target_name=target, df=df.copy())

    print("\nProceso de entrenamiento para todos los targets completado.")