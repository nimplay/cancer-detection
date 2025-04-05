import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Para guardar el modelo

# Cargar datos
cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')

# Corregir nombres de columnas con espacios
cancer = cancer.rename(columns={
    'concave points_mean': 'concave_points_mean',
    'concave points_se': 'concave_points_se',
    'concave points_worst': 'concave_points_worst'
})

y = cancer['diagnosis'].map({'B': 0, 'M': 1})  # Convertir a 0 y 1
X = cancer.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)

# Verificar que tenemos todas las columnas esperadas
expected_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

assert all(col in X.columns for col in expected_columns), "Faltan columnas en los datos"

# Escalar datos (importante para modelos de ML)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Entrenar modelo
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar modelo y scaler para producción
joblib.dump(model, 'cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Guardar también la lista de características
import json
with open('feature_names.json', 'w') as f:
    json.dump(expected_columns, f)
