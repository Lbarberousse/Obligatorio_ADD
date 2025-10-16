import pandas as pd
import numpy as np

df = pd.read_csv("Tema_6.csv", sep=";")
df.info()
df.head()

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

df = df.drop_duplicates()

print(df.dtypes)

#Limpiar los datos antes de convertir

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

num = ['Age',
       'Weight',
       'Height',
       'FCVC',
       'NCP',
       'CH2O',
       'FAF',
       'TUE']
for col in num:
    df[col] = df[col].replace({r'\.': ''}, regex=True)  # Eliminamos todos los puntos

    # Convertir a valores numéricos
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convertir a números, ignorando los errores

# Convertir Age a int quedandonos con el piso
df['Age'] = np.floor(df['Age']).astype('Int64') 

print(df.isnull().sum() )




#Evaluar las variables discretas y los posibles valores unicos
variables_discretas = [
    'Gender',
    'family_history_with_overweight',
    'FAVC',
    'CAEC',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS',
    'NObeyesdad'
]

# Evaluar valores únicos para cada una
for var in variables_discretas:
    print(f"=== {var} ===")
    if var in df.columns:
        unique_values = df[var].dropna().unique()
        print(f"Número de valores únicos: {len(unique_values)}")
        print(f"Valores únicos: {sorted(map(str, unique_values))}\n")

# --------- RELLENAR VALORES FALTANTES --------------

### RELLENAR WEIGHT, HEIGHT Y NOBEYSDAD COMBIANANDO LOS TRES ###
print(df[['Weight', 'Height', 'NObeyesdad']].isna().sum())

#Paso 1: Limpiar espacios en NObeyesdad para evitar problemas de mapeo
df['NObeyesdad'] = df['NObeyesdad'].astype(str).str.strip()

#Paso 2: Definir rangos medios de BMI por categoría
bmi_promedios = {
    'Insufficient_Weight': 17,
    'Normal_Weight': 21.5,
    'Overweight_Level_I': 25.5,
    'Overweight_Level_II': 28.5,
    'Obesity_Type_I': 32,
    'Obesity_Type_II': 37,
    'Obesity_Type_III': 42,
}

#Función para clasificar según BMI
def clasificar_bmi(bmi):
    if pd.isna(bmi):
        return np.nan
    elif bmi < 18.5:
        return 'Insufficient_Weight'
    elif bmi < 25:
        return 'Normal_Weight'
    elif bmi < 27:
        return 'Overweight_Level_I'
    elif bmi < 30:
        return 'Overweight_Level_II'
    elif bmi < 35:
        return 'Obesity_Type_I'
    elif bmi < 40:
        return 'Obesity_Type_II'
    else:
        return 'Obesity_Type_III'

#Paso 3: Rellenar NObeyesdad si falta, usando BMI
mask = df['NObeyesdad'].isna() & df['Weight'].notna() & df['Height'].notna()
bmi = df.loc[mask, 'Weight'] / (df.loc[mask, 'Height'] ** 2)
df.loc[mask, 'NObeyesdad'] = bmi.apply(clasificar_bmi)

#Paso 4: Rellenar Weight si falta
mask = df['Weight'].isna() & df['Height'].notna() & df['NObeyesdad'].isin(bmi_promedios.keys())
df.loc[mask, 'Weight'] = df.loc[mask].apply(
    lambda row: bmi_promedios[row['NObeyesdad']] * (row['Height'] ** 2),
    axis=1
)

#Paso 5: Rellenar Height si falta
mask = df['Height'].isna() & df['Weight'].notna() & df['NObeyesdad'].isin(bmi_promedios.keys())
df.loc[mask, 'Height'] = df.loc[mask].apply(
    lambda row: np.sqrt(row['Weight'] / bmi_promedios[row['NObeyesdad']]),
    axis=1
)

print(df[['Weight', 'Height', 'NObeyesdad']].isna().sum())

### RELLENAR LOS VALORES FALTANTES Y REEMPLAZAR LOS VALORES INCORRECTOS DE GENDER POR UNKNOWN ##
df['Gender'] = df['Gender'].apply(lambda x: 'Unknown' if pd.isna(x) or x not in ['Male', 'Female'] else x)

### RELLENAR FAMILY_HISTORY_WITH_OVERWEIGHT ###

#print(df['family_history_with_overweight'].isna().sum())

#Agrupar por 'NObeyesdad' para calcular la moda de 'family_history_with_overweight'
family_history_mode_by_obesity = df.groupby('NObeyesdad')['family_history_with_overweight'].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

#Función para rellenar 'family_history_with_overweight' según el nivel de obesidad
def fill_family_history(row):
    if pd.isna(row['family_history_with_overweight']):
        return family_history_mode_by_obesity.get(row['NObeyesdad'], np.nan)
    else:
        return row['family_history_with_overweight']

#Aplicamos la función para rellenar los valores faltantes
df['family_history_with_overweight'] = df.apply(fill_family_history, axis=1)

#print(df['family_history_with_overweight'].isna().sum())