from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("Tema_6.csv", sep=";")
# Generar el perfil de los datos
profile = ProfileReport(df, title="Perfil del Dataset", explorative=True)

# Guardar el informe como un archivo HTML
profile.to_file("data_profile_obligatorio.html")

# Para ver el perfil en el notebook
profile.to_notebook_iframe()