import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Para guardar el modelo

# Cargar datos
cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')
y = cancer['diagnosis'].map({'B': 0, 'M': 1})  # Convertir a 0 y 1
X = cancer.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)

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
