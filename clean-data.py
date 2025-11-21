import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import sys

tasks = [
    ("data_train.csv", "data_train_clean.csv", "train"),
    ("data_test.csv",  "data_test_clean.csv",  "test")
]

print("=== Ejecutando clean-data.py en modo AUTOMÁTICO ===")
print("Se generarán:")
print(" - data_train_clean.csv")
print(" - data_test_clean.csv")

for input_file, output_file, mode in tasks:
    print("\n----------------------------------------")
    print(f"Procesando archivo: {input_file}")
    print(f"Salida:             {output_file}")
    print(f"Modo:               {mode}")
    print("----------------------------------------\n")

    # --- LECTURA DE DATOS ---
    # Leemos el CSV original y el limpio para comparar si es necesario
    df_original = pd.read_csv(input_file, sep=";", na_values=['', ' '], skipinitialspace=True)
    df = df_original.copy() # Trabajamos sobre una copia

    # --- LIMPIEZA Y CONVERSIÓN DE TIPOS ---

    # 1. Limpieza general de strings y valores conocidos
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    df['Gender'] = df['Gender'].replace({'12345': np.nan, 'nan': np.nan})
    df['NObeyesdad'] = df['NObeyesdad'].replace('nan', np.nan)

    # 2. Función robusta para limpiar y convertir columnas numéricas
    def clean_and_convert_numeric(df_in, column_name):
        # Convertir a string para poder manipular
        series = df_in[column_name].astype(str)

        # Función para aplicar a cada valor
        def clean_value(val):
            if pd.isna(val) or val.lower() in ['nan', 'none']:
                return np.nan
            # Si hay más de un punto, son separadores de miles -> los quitamos
            if val.count('.') > 1:
                val = val.replace('.', '')
            # Truncar a 2 dígitos para 'Age'
            if column_name == 'Age' and len(val) > 2:
                val = val[:2]
            return val

        # Aplicar la limpieza
        cleaned = series.apply(clean_value)
        numeric = pd.to_numeric(cleaned, errors='coerce')

        # Lógica específica para Altura (unificar a metros)
        if column_name == 'Height':
            numeric.loc[numeric > 100] = numeric.loc[numeric > 100] / 100

        return numeric

    # 3. Aplicar la limpieza a todas las columnas numéricas
    numeric_cols = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']
    for col in numeric_cols:
        df[col] = clean_and_convert_numeric(df, col)

    # Testeo sin imputar
    if mode == "test":
        print("Modo TEST: limpieza + imputación segura sin leakage")

        # Imputación mínima para evitar NaNs
        for col in numeric_cols:
            df[col] = df[col].fillna(-1)

        cat_cols = ['Gender','family_history_with_overweight','FAVC','CAEC',
                    'SMOKE','SCC','CALC','MTRANS']
        for col in cat_cols:
            df[col] = df[col].fillna("Unknown")

        df.to_csv(output_file, index=False, sep=";")
        print(f"Archivo TEST generado: {output_file}")
        continue

    # 4. Corregir valores extremos/imposibles después de la conversión
    df.loc[df['Age'] > 100, 'Age'] = np.nan
    df.loc[df['Height'] > 2.5, 'Height'] = np.nan
    df.loc[df['Height'] < 1.0, 'Height'] = np.nan # Alturas menores a 1m son improbables
    df.loc[df['Weight'] > 200, 'Weight'] = np.nan
    df.loc[df['Weight'] < 20, 'Weight'] = np.nan # Pesos menores a 20kg son improbables

    '''# Usamos el método del Rango Intercuartílico (IQR) para detectar y tratar outliers
    # en las columnas numéricas donde tiene sentido.
    print("\nLimpiando outliers con el método IQR...")
    cols_for_outlier_check = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for col in cols_for_outlier_check:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_superior = Q3 + 1.5 * IQR
            limite_inferior = Q1 - 1.5 * IQR

            # Reemplazamos outliers con NaN para que luego sean imputados
            outliers_mask = (df[col] < limite_inferior) | (df[col] > limite_superior)
            num_outliers = outliers_mask.sum()
            if num_outliers > 0:
                print(f"Encontrados y reemplazados {num_outliers} outliers en '{col}'.")
                df.loc[outliers_mask, col] = np.nan'''
    
    # 5. Convertir 'Age' a entero (ahora que está limpio)
    df['Age'] = np.floor(df['Age']).astype('Int64')

    print(df.isnull().sum() )

    # --- ELIMINAR FILAS CON EDAD FALTANTE ---
    # De acuerdo a la instrucción, no imputamos 'Age', sino que eliminamos las filas.
    age_missing_before = df['Age'].isna().sum()
    if age_missing_before > 0:
        print(f"\nEliminando {age_missing_before} filas donde 'Age' es nulo...")
        df.dropna(subset=['Age'], inplace=True)
        print(f"Verificación de nulos en 'Age' después de eliminar: {df['Age'].isna().sum()}")

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
    bmi_prom = {
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
        if pd.isna(bmi): return np.nan
        elif bmi < 18.5: return 'Insufficient_Weight'
        elif bmi < 25: return 'Normal_Weight'
        elif bmi < 27: return 'Overweight_Level_I'
        elif bmi < 30: return 'Overweight_Level_II'
        elif bmi < 35: return 'Obesity_Type_I'
        elif bmi < 40: return 'Obesity_Type_II'
        else: return 'Obesity_Type_III'

    #Paso 3: Rellenar NObeyesdad si falta, usando BMI
    mask = df['NObeyesdad'].isna() & df['Weight'].notna() & df['Height'].notna()
    bmi = df.loc[mask, 'Weight'] / (df.loc[mask, 'Height'] ** 2)
    df.loc[mask, 'NObeyesdad'] = bmi.apply(clasificar_bmi)

    #Paso 4: Rellenar Weight si falta
    mask = df['Weight'].isna() & df['Height'].notna() & df['NObeyesdad'].isin(bmi_prom.keys())
    # La altura debe estar en metros para el cálculo
    df.loc[mask, 'Weight'] = df.loc[mask].apply(lambda r: bmi_prom[r['NObeyesdad']] * (r['Height'] ** 2), axis=1)

    #Paso 5: Rellenar Height si falta
    mask = df['Height'].isna() & df['Weight'].notna() & df['NObeyesdad'].isin(bmi_prom.keys())
    # La altura debe estar en metros para el cálculo
    df.loc[mask, 'Height'] = df.loc[mask].apply(lambda r: np.sqrt(r['Weight'] / bmi_prom[r['NObeyesdad']]), axis=1)

    # --- Fallback para Weight, Height y NObeyesdad ---
    # Si después de la lógica anterior aún quedan nulos (ej. una fila tenía NaN en 2 de 3 columnas),
    # usamos una imputación por grupo como plan B.

    # Para NObeyesdad, si aún falta, no podemos hacer mucho más, así que lo eliminamos.
    df.dropna(subset=['NObeyesdad'], inplace=True)

    # Para Weight y Height, usamos la mediana del grupo de NObeyesdad.
    for col in ['Weight', 'Height']:
        if df[col].isna().any():
            # Usamos transform para rellenar por grupo y luego un fillna global para los grupos sin mediana
            df[col] = df[col].fillna(df.groupby('NObeyesdad')[col].transform('median'))
            df[col] = df[col].fillna(df[col].median())

    print(df[['Weight', 'Height', 'NObeyesdad']].isna().sum())

    # Rellenar Gender usando un clasificador Bayesiano
    df['Gender'] = df['Gender'].astype(str).str.capitalize()
    valid = {'Male','Female'}

    mask_valid = df['Gender'].isin(valid)
    mask_pred  = ~df['Gender'].isin(valid)

    X_train = df.loc[mask_valid, ['Height','Weight']]
    y_train = df.loc[mask_valid, 'Gender']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    X_test = df.loc[mask_pred, ['Height','Weight']]
    X_test_scaled = scaler.transform(X_test)

    df.loc[mask_pred, 'Gender'] = model.predict(X_test_scaled)

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

    ### RELLENAR COLUMNAS NUMÉRICAS CON MEDIANA POR GRUPO (NObeyesdad) ###

    # Crear rangos de edad para una imputación más precisa
    age_bins = [13, 18, 30, 45, 60, 101] # Ajustado al rango real de datos (mínimo 14)
    age_labels = ['14-18', '19-30', '31-45', '46-60', '61+'] # El límite superior es exclusivo por defecto
    df['Age_Range'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)

    numeric_cols_to_fill = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

    print("\n=== Imputando con Mediana por Grupo (NObeyesdad) ===")
    for col in numeric_cols_to_fill:
        if col in df.columns and df[col].isna().any():
            # Agrupamos por NObeyesdad, Género y Rango de Edad para una imputación más robusta.
            # Ya no es necesario imputar 'Age', pero usamos 'Age_Range' para mejorar la imputación de otras columnas.
            df[col] = df[col].fillna(df.groupby(['NObeyesdad', 'Gender', 'Age_Range'], observed=False)[col].transform('median'))
            
            # Fallback: si después de agrupar aún queda algún NaN, rellenamos con la mediana global.
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    ### RELLENAR COLUMNAS CATEGÓRICAS CON MODA POR GRUPO (NObeyesdad) ###

    # Limpieza específica para 'CALC' antes de imputar
    if 'CALC' in df.columns:
        df['CALC'] = df['CALC'].replace({'No': 'no'})

    categorical_cols_to_fill = ['FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    print("\n=== Imputando con Moda por Grupo (NObeyesdad) ===")
    for col in categorical_cols_to_fill:
        if col in df.columns and df[col].isna().any():
            # La lógica es la misma, pero con mode(). Usamos [0] porque mode() puede devolver varios valores.
            df[col] = df[col].fillna(df.groupby('NObeyesdad')[col].transform(lambda x: x.mode()[0] if not x.mode().empty else "Unknown"))

            # Fallback con la moda global
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])

    # Eliminar la columna auxiliar de rango de edad antes de guardar
    df.drop(columns=['Age_Range'], inplace=True)

    df.to_csv(output_file, index=False, sep=';')
    print(f"Archivo TRAIN generado: {output_file}")

print("\n=== ARCHIVOS GENERADOS CORRECTAMENTE ===")
