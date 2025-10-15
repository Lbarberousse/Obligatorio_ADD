import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Cargar dataset
df = pd.read_csv("obesity_hw_imputed.csv")

df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize() # Normalizamos Gender de la forma Aaaaa
df.loc[~df['Gender'].isin(['Male', 'Female']), 'Gender'] = np.nan  # Reemplazamos todo lo que no sea Male/Female por NaN (elimino los 12345) 

# Esta funcion convierte a string, quita espacios, cambia coma por punto y pasa el dato a numérico
def to_numeric_clean(s):
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(',', '.', regex=False),
        errors='coerce'
    )

df['Height'] = to_numeric_clean(df['Height'])
df['Weight'] = to_numeric_clean(df['Weight'])

# Recortamos valores imposibles para ENTRENAR (no se borran filas, solo para X_train)
# Usamos rangos razonables
height_ok = df['Height'].between(1.2, 2.2)   # metros
weight_ok = df['Weight'].between(30, 400)    # kg

# Entrenamos con filas válidas
valid_mask = df['Gender'].notna() & height_ok & weight_ok # Gender no NaN y altura y peso dentro de los rangos
validos = df.loc[valid_mask, ['Height', 'Weight', 'Gender']].dropna()

faltantes_mask = df['Gender'].isna() & df['Height'].notna() & df['Weight'].notna() & height_ok & weight_ok # Gender NaN y el resto de valores validos
faltantes = df.loc[faltantes_mask, ['Height', 'Weight']]

X_train = validos[['Height', 'Weight']].values # Entrenamos el modelo sobre Height y Weight (validos)
y_train = validos['Gender'].values # El valor que el modelo debe predecir

scaler = StandardScaler()   # Normalizamos los valores z = (x - u)/s
X_train_scaled = scaler.fit_transform(X_train)

model = GaussianNB()    # Instanciamos el modelo
model.fit(X_train_scaled, y_train)  # Entrenamos al modelo

# Testeo basico del modelo
X_test = faltantes[['Height', 'Weight']].values
X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)
df.loc[faltantes.index, 'Gender'] = y_pred

# Guardado de resultados
print("Distribución final de Gender:")
print(df['Gender'].value_counts(dropna=False))
print(f"Faltantes de Gender: {df['Gender'].isna().sum()}")

df.to_csv("gender_clean.csv", index=False)

# Poloteo del modelo 
plt.figure(figsize=(6,4))
for gender, color in [('Male', 'blue'), ('Female', 'red')]:
    subset = df[df['Gender'] == gender]
    plt.scatter(subset['Height'], subset['Weight'], alpha=0.6, label=gender, c=color)

plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Distribución de Altura y Peso por Género')
plt.legend()
plt.show()
