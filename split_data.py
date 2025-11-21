import pandas as pd
from sklearn.model_selection import train_test_split

df_original = pd.read_csv("Tema_6.csv", sep=";", na_values=['', ' '], skipinitialspace=True)
df = df_original.copy()

df.info()

filas, columnas = df.shape

faltantes = df.isna().sum()
faltantes_porcentaje = (faltantes/filas) * 100
print('Los datos faltantes por columna son \n\n',faltantes)
print("\nPorcentaje de datos faltantes por columna:")
print(faltantes_porcentaje)

# Calcular el porcentaje total de celdas faltantes
total_celdas = df.size
faltantes_totales = df.isnull().sum().sum()  # Total de celdas faltantes
porcentaje_faltantes_totales = (faltantes_totales / total_celdas) * 100

print(f"\n\nPorcentaje total de celdas faltantes en el dataset: {porcentaje_faltantes_totales:.2f}%")

duplicados = df.duplicated().sum()
duplicados_porcentaje = (duplicados/filas) * 100
print('\nLos datos duplicados por columna son:',duplicados)
print("\n\nPorcentaje de datos duplicados del dataset:")
print(duplicados_porcentaje)

# Eliminamos duplicados exactos
df = df.drop_duplicates()

print(df.dtypes)

df = df.dropna(subset=['NObeyesdad'])

train, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df['NObeyesdad']  # si hay NObeyesdad
)

train.to_csv("data_train.csv", index=False, sep=";")
test.to_csv("data_test.csv", index=False, sep=";")

print("Archivos generados: data_train.csv, data_test.csv")
