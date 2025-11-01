import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression  # o el clasificador que prefieras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df_original = pd.read_csv("obesity_dataset_clean.csv", sep=";", na_values=['', ' '], skipinitialspace=True)
df = df_original.copy() 
df = df.dropna(subset=["NObeyesdad"])

print(df)

cat_cols = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']
num_cols = ['Age','Weight','Height','FCVC','NCP','CH2O','FAF','TUE']

X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols),
    ]
)

clf = Pipeline(steps=[
    ('prep', preprocess),
    ('model', LogisticRegression(max_iter=1000))  # o RandomForestClassifier(), etc.
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

nuevo = pd.DataFrame([{
    "Gender": "Male",
    "Age": 28,
    "Height": 1.78,
    "Weight": 86,
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "FCVC": 2.0,
    "NCP": 3.0,
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "CH2O": 2.0,
    "SCC": "no",
    "FAF": 2.0,
    "TUE": 1.0,
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation"
}])

pred_clase = clf.predict(nuevo)[0]
pred_probs = clf.predict_proba(nuevo)[0]   # probas por clase
clases = clf.named_steps["model"].classes_

# ejemplo: top-3 clases más probables
top = sorted(zip(clases, pred_probs), key=lambda x: x[1], reverse=True)[:3]
print(pred_clase)
print(top)



