import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

INPUT_CSV  = "Tema_6.csv"
OUTPUT_CSV = "obesity_hw_imputed.csv"
N_NEIGHBORS = 7

# Rangos razonables
H_MIN, H_MAX = 1.20, 2.20   # m
W_MIN, W_MAX = 30.0, 250.0  # kg

df = pd.read_csv(INPUT_CSV, sep=";")


# --- 1) Normalizar nombres problemáticos conocidos ---
rename_map = {
    'Wight': 'Weight',
    'weight': 'Weight',
    'height': 'Height',
    'age': 'Age',
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# --- 2) Definir candidatas numéricas y quedarme con las que existan ---
candidate_feats = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']
present_feats = [c for c in candidate_feats if c in df.columns]

# Asegurar que al menos estén Height/Weight
for must in ['Height', 'Weight']:
    if must not in present_feats:
        present_feats.append(must)


# Si no existe ninguna de estas columnas en el CSV, abortamos con diagnóstico claro
really_present = [c for c in present_feats if c in df.columns]
if len(really_present) == 0:
    raise ValueError(
        "No se encontró ninguna de las columnas esperadas. "
        f"Columnas del CSV: {list(df.columns)}"
    )

print("Usando como features para KNN:", really_present)

# --- 3) Marcar como NaN valores fuera de rango en H/W (prelimpieza) ---
height_before_missing = df['Height'].isna().sum() if 'Height' in df.columns else 0
weight_before_missing = df['Weight'].isna().sum() if 'Weight' in df.columns else 0

if 'Height' in df.columns:
    out_of_range_height = ~(df['Height'].between(H_MIN, H_MAX)) & df['Height'].notna()
    df.loc[out_of_range_height, 'Height'] = np.nan
else:
    df['Height'] = np.nan  # si no estaba, la creamos

if 'Weight' in df.columns:
    out_of_range_weight = ~(df['Weight'].between(W_MIN, W_MAX)) & df['Weight'].notna()
    df.loc[out_of_range_weight, 'Weight'] = np.nan
else:
    df['Weight'] = np.nan  # si no estaba, la creamos

# --- 4) Preparar matriz numérica ---
X = df[really_present].astype(float).copy()

print("Columnas CSV:", list(df.columns))
print(df[['Height','Weight']].head(10))
print("Shape de X:", X.shape)


# Guardar máscaras de NaN (después del rango)
missing_h_mask = df['Height'].isna()
missing_w_mask = df['Weight'].isna()

# --- 5) Estandarizar -> KNNImputer -> Desestandarizar ---
# Si por algún motivo no queda ninguna columna (no debería pasar), avisar
if X.shape[1] == 0:
    raise ValueError("No hay columnas numéricas disponibles para el KNNImputer.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

imputer = KNNImputer(n_neighbors=N_NEIGHBORS, weights='distance')
X_imp_scaled = imputer.fit_transform(X_scaled)

# OJO: solo invertimos si hubo al menos 1 feature
X_imp = scaler.inverse_transform(X_imp_scaled)

X_imp_df = pd.DataFrame(X_imp, columns=really_present, index=df.index)

# --- 6) Escribir SOLO donde faltaba (o se marcó fuera de rango) ---
if 'Height' in X_imp_df.columns:
    df.loc[missing_h_mask, 'Height'] = X_imp_df.loc[missing_h_mask, 'Height']
if 'Weight' in X_imp_df.columns:
    df.loc[missing_w_mask, 'Weight'] = X_imp_df.loc[missing_w_mask, 'Weight']

# --- 7) Clip de seguridad ---
df['Height'] = df['Height'].clip(H_MIN, H_MAX)
df['Weight'] = df['Weight'].clip(W_MIN, W_MAX)

# --- 8) Reporte ---
height_after_missing = df['Height'].isna().sum()
weight_after_missing = df['Weight'].isna().sum()

print("=== Diagnóstico imputación KNN (Height/Weight) ===")
print(f"Height NaN antes: {height_before_missing}  | después: {height_after_missing}")
print(f"Weight NaN antes: {weight_before_missing}  | después: {weight_after_missing}")

print("\nHeight (m) -> min/median/max:",
      float(df['Height'].min()), float(df['Height'].median()), float(df['Height'].max()))
print("Weight (kg) -> min/median/max:",
      float(df['Weight'].min()), float(df['Weight'].median()), float(df['Weight'].max()))

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Guardado: {OUTPUT_CSV}")
