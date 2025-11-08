from ydata_profiling import ProfileReport
import pandas as pd
import os

# Leer datos (ruta relativa al directorio de trabajo o absoluta según corresponda)
df = pd.read_csv("Tema_6.csv", sep=";")

# Generar el perfil de los datos
profile = ProfileReport(df, title="Perfil del Dataset", explorative=True)

# Guardar el informe HTML dentro de la carpeta data_profile (la misma carpeta que este script)
here = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(here, "data_profile_obligatorio.html")
profile.to_file(out_path)
print(f"Informe guardado en: {out_path}")
