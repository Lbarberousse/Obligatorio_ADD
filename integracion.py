import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import math

INPUT_CSV  = "Tema_6.csv"
OUTPUT_CSV = "obesity_hw_imputed.csv"
N_NEIGHBORS = 7

# Rangos razonables
H_MIN, H_MAX = 1.20, 2.20   # m
W_MIN, W_MAX = 30.0, 250.0  # kg

# Fallback para filas con BOTH NaN (opcional)
DO_FALLBACK_MEDIAN = True

df = pd.read_csv(INPUT_CSV, sep=";")


def to_numeric_clean(s):
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(',', '.', regex=False),
        errors='coerce'
    )

df['Height'] = to_numeric_clean(df['Height'])
df['Weight'] = to_numeric_clean(df['Weight'])

# 2) Prelimpieza: marcar fuera de rango como NaN
h_before = df['Height'].isna().sum()
w_before = df['Weight'].isna().sum()

out_h = ~(df['Height'].between(H_MIN, H_MAX)) & df['Height'].notna()
out_w = ~(df['Weight'].between(W_MIN, W_MAX)) & df['Weight'].notna()
df.loc[out_h, 'Height'] = np.nan
df.loc[out_w, 'Weight'] = np.nan

# Máscaras de dónde FALTABA (o se marcó) para reescribir sólo allí
missing_h_mask = df['Height'].isna()
missing_w_mask = df['Weight'].isna()

# 3) Matriz SOLO con Height/Weight
X = df[['Height', 'Weight']].astype(float).copy()

# Escalar → KNN → desescalar (solo 2 columnas)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

imputer = KNNImputer(n_neighbors=N_NEIGHBORS, weights='distance')
X_imp_scaled = imputer.fit_transform(X_scaled)

X_imp = scaler.inverse_transform(X_imp_scaled)
X_imp_df = pd.DataFrame(X_imp, columns=['Height', 'Weight'], index=df.index)

def fix_format(val):
    if pd.notna(val):
        return float(f"{val:.2f}")
    return val

# Corrige solo los valores fuera de rango en Height imputado
X_imp_df['Height'] = X_imp_df['Height'].apply(fix_format)
X_imp_df['Weight'] = X_imp_df['Weight'].apply(fix_format)


print(X_imp_df["Height"].head(25))
# Si ves el mismo problema en Weight, puedes aplicar una función similar

# 4) Escribir SOLO donde faltaba
df.loc[missing_h_mask, 'Height'] = X_imp_df.loc[missing_h_mask, 'Height']
df.loc[missing_w_mask, 'Weight'] = X_imp_df.loc[missing_w_mask, 'Weight']

# 5) Fallback opcional: si una fila quedó con ambas NaN, imputar con mediana (o dejarlas NaN si preferís)
if DO_FALLBACK_MEDIAN:
    both_nan = df['Height'].isna() & df['Weight'].isna()
    if both_nan.any():
        df.loc[both_nan, 'Height'] = df['Height'].median(skipna=True)
        df.loc[both_nan, 'Weight'] = df['Weight'].median(skipna=True)

# 6) Clip final
df['Height'] = df['Height'].clip(H_MIN, H_MAX)
df['Weight'] = df['Weight'].clip(W_MIN, W_MAX)

# 7) Reporte
h_after = df['Height'].isna().sum()
w_after = df['Weight'].isna().sum()

print("=== KNN con SOLO Height/Weight ===")
print(f"Height NaN antes: {h_before} | después: {h_after}")
print(f"Weight NaN antes: {w_before} | después: {w_after}")
print("Height (m) min/median/max:", float(df['Height'].min()), float(df['Height'].median()), float(df['Height'].max()))
print("Weight (kg) min/median/max:", float(df['Weight'].min()), float(df['Weight'].median()), float(df['Weight'].max()))

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Guardado: {OUTPUT_CSV}")
