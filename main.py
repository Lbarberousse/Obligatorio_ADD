from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import subprocess
import os
from pathlib import Path

# Inicializamos la app Flask, especificando explícitamente las carpetas
# para plantillas (templates) y archivos estáticos (static).
# Esto asegura que Flask siempre encuentre los archivos, sin importar desde dónde se ejecute el script.
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Carga de Modelos ---
# Cargamos ambos modelos al iniciar la app para mayor eficiencia.
try:
    model_multiclass = joblib.load('modelo_obesidad_multiclass.pkl')
    model_binary = joblib.load('modelo_obesidad_binary.pkl')
    print("Modelos cargados correctamente.")
except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo del modelo: {e.filename}")
    print("Asegúrate de haber entrenado ambos modelos (con --target NObeyesdad y --target ObesityBinary) y que los archivos .pkl estén en la misma carpeta.")
    model_multiclass = None
    model_binary = None

@app.route('/')
def home():
    """Página de inicio que renderiza el formulario."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones."""
    if not model_multiclass or not model_binary:
        return jsonify({'error': 'Los modelos no están disponibles. Revisa la consola del servidor.'}), 500

    try:
        # Obtener los datos del formulario
        data = request.form.to_dict()
        
        # Extraer el tipo de predicción y eliminarlo de los datos del modelo
        prediction_type = data.pop('prediction_type', 'multiclass')

        # Convertir los datos a los tipos numéricos correctos
        for key, value in data.items():
            try:
                data[key] = float(value)
            except (ValueError, TypeError):
                # Si no se puede convertir a float, lo dejamos como está (para las categóricas)
                pass

        # Definir el orden de las columnas que espera el modelo
        expected_cols = [
            'Gender','family_history_with_overweight','FAVC','CAEC',
            'SMOKE','SCC','CALC','MTRANS', 'Age','Weight','Height',
            'FCVC','NCP','CH2O','FAF','TUE'
        ]

        # Crear un DataFrame de una sola fila para la predicción
        df_to_predict = pd.DataFrame([data])
        df_to_predict = df_to_predict[expected_cols] # Forzar el orden correcto de las columnas

        # Seleccionar el modelo basado en la elección del usuario
        if prediction_type == 'binary':
            model = model_binary
            model_name = "Binario (Obeso/No Obeso)"
        else: # 'multiclass' es el default
            model = model_multiclass
            model_name = "Multiclase (Niveles de Obesidad)"

        # Realizar la predicción
        prediction = model.predict(df_to_predict)
        prediction_proba = model.predict_proba(df_to_predict)

        # Preparar la respuesta
        response = {
            'model_used': model_name,
            'prediction': prediction[0],
            'probabilities': dict(zip(model.classes_, prediction_proba[0]))
        }

        return jsonify(response)

    except Exception as e:
        # Capturar cualquier error durante el proceso
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Antes de iniciar el servidor, verificar si los modelos existen.
    # Si no existen, ejecutar el script de entrenamiento.
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        model_multiclass_path = Path('modelo_obesidad_multiclass.pkl')
        model_binary_path = Path('modelo_obesidad_binary.pkl')

        # Solo entrenar si alguno de los modelos no existe.
        if not model_multiclass_path.exists() or not model_binary_path.exists():
            print("="*50)
            print("Uno o más modelos no encontrados. INICIANDO ENTRENAMIENTO...")
            print("="*50)
            try:
                subprocess.run(["python", "models.py"], check=True)
                print("="*50)
                print("ENTRENAMIENTO COMPLETADO.")
                print("="*50)
            except subprocess.CalledProcessError as e:
                print(f"Error durante el entrenamiento: {e}")
                print("El servidor no se iniciará.")
                exit(1) # Detiene la ejecución si el entrenamiento falla

    # Iniciar la aplicación en modo debug
    app.run(debug=True)