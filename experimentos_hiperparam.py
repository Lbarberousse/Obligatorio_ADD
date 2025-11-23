import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

DATA_PATH = "data_clean.csv"
OUTPUTS_DIR = Path("outputs")

CAT_COLS = [
    'Gender','family_history_with_overweight','FAVC','CAEC',
    'SMOKE','SCC','CALC','MTRANS'
]
NUM_COLS = ['Age','Weight','Height','FCVC','NCP','CH2O','FAF','TUE']

TARGET = "NObeyesdad"      # multiclase (7 clases)
#TARGET = "ObesityBinary"    # binario (Obeso / No_Obeso) si lo creaste

RANDOM_STATE = 42
N_SPLITS = 5

# Hiperparámetros a explorar
RF_N_ESTIMATORS = [200, 300, 400, 500]
RF_MAX_DEPTHS   = [5, 10, 15]

KNN_K_VALUES = [3, 5, 7, 9, 11, 13, 15]

LOGREG_C_VALUES = [0.01, 0.1, 1, 10, 100]

def make_preprocess():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )

print(f"Leyendo dataset limpio: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, sep=";")

if TARGET not in df.columns:
    raise ValueError(f"La columna TARGET '{TARGET}' no existe en el CSV. Verificá el nombre.")

print(f"Nulos en {TARGET} antes de filtrar:", df[TARGET].isna().sum())
df = df.dropna(subset=[TARGET])
print(f"Nulos en {TARGET} después de filtrar:", df[TARGET].isna().sum())

X = df[CAT_COLS + NUM_COLS]
y = df[TARGET]

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

prep = make_preprocess()

rf_results = {}

print("\n=== EXPERIMENTO RANDOM FOREST ===")
for max_depth in RF_MAX_DEPTHS:
    f1s = []
    for n_estimators in RF_N_ESTIMATORS:
        print(f"  RF: n_estimators={n_estimators}, max_depth={max_depth}")

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )

        pipe = Pipeline(steps=[
            ("prep", prep),
            ("model", rf)
        ])

        scores = cross_validate(
            pipe, X, y,
            cv=cv,
            scoring={"f1_weighted": "f1_weighted"},
            n_jobs=-1,
            return_train_score=False
        )

        f1_mean = scores["test_f1_weighted"].mean()
        f1s.append(f1_mean)
        print(f"    -> F1_weighted medio: {f1_mean:.3f}")

    rf_results[max_depth] = f1s

plt.figure(figsize=(8, 5))
for max_depth, f1s in rf_results.items():
    plt.plot(RF_N_ESTIMATORS, f1s, marker="o", label=f"max_depth={max_depth}")

plt.xlabel("n_estimators")
plt.ylabel("F1 weighted (CV)")
plt.title("Random Forest: F1 vs n_estimators (curvas por max_depth)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR/"rf_hyperparams_f1.png", dpi=150)

print("\n=== EXPERIMENTO KNN ===")
knn_f1s = []

for k in KNN_K_VALUES:
    print(f"  KNN: k={k}")

    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        metric="minkowski"
    )

    pipe = Pipeline(steps=[
        ("prep", prep),
        ("model", knn)
    ])

    scores = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring={"f1_weighted": "f1_weighted"},
        n_jobs=-1,
        return_train_score=False
    )

    f1_mean = scores["test_f1_weighted"].mean()
    knn_f1s.append(f1_mean)
    print(f"    -> F1_weighted medio: {f1_mean:.3f}")

# Graficar KNN
plt.figure(figsize=(8, 5))
plt.plot(KNN_K_VALUES, knn_f1s, marker="o")
plt.xlabel("k (n_neighbors)")
plt.ylabel("F1 weighted (CV)")
plt.title("KNN: F1 vs k")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR/"knn_hyperparams_f1.png", dpi=150)

logreg_f1s = []

print("\n=== EXPERIMENTO REGRESIÓN LOGÍSTICA (C) ===")
for C in LOGREG_C_VALUES:
    print(f"  LogReg: C={C}")

    logreg = LogisticRegression(
        C=C,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced"
    )

    pipe = Pipeline(steps=[
        ("prep", prep),
        ("model", logreg)
    ])

    scores = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring={"f1_weighted": "f1_weighted"},
        n_jobs=-1,
        return_train_score=False
    )

    f1_mean = scores["test_f1_weighted"].mean()
    logreg_f1s.append(f1_mean)
    print(f"    -> F1_weighted medio: {f1_mean:.3f}")

#Graficar LogReg
plt.figure(figsize=(8, 5))
plt.semilogx(LOGREG_C_VALUES, logreg_f1s, marker="o")  # escala log en el eje de C
plt.xlabel("C (inversa de regularización)")
plt.ylabel("F1 weighted (CV)")
plt.title("Regresión Logística: F1 vs C")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR/"logreg_hyperparams_f1.png", dpi=150)

print("\n=== Experimentos terminados ===")
print("Gráficos guardados como:")
print("  - rf_hyperparams_f1.png")
print("  - knn_hyperparams_f1.png")
