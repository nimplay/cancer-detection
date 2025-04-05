from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
# Configuración CORS más robusta para desarrollo
CORS(app, resources={
    r"/predict": {
        "origins": ["*"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Carga segura del modelo con manejo de errores
def load_model():
    try:
        model = joblib.load('cancer_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        raise RuntimeError(f"Error cargando archivos del modelo: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error inesperado al cargar el modelo: {str(e)}")

model, scaler = load_model()

# Lista completa de características requeridas
REQUIRED_FEATURES = [
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


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verificar datos recibidos
        if not request.is_json:
            return jsonify({"error": "El contenido debe ser JSON"}), 400

        data = request.get_json()

          # Convertir nombres alternativos (para compatibilidad)
        if 'concave points_mean' in data:
            data['concave_points_mean'] = data.pop('concave points_mean')
        if 'concave points_se' in data:
            data['concave_points_se'] = data.pop('concave points_se')
        if 'concave points_worst' in data:
            data['concave_points_worst'] = data.pop('concave points_worst')

        # Validar campos requeridos
        missing_fields = [field for field in REQUIRED_FEATURES if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Campos faltantes",
                "missing": missing_fields
            }), 400

        # Convertir datos a array numpy
        try:
            features = np.array([float(data[field]) for field in REQUIRED_FEATURES])
        except ValueError as e:
            return jsonify({
                "error": "Valores inválidos",
                "details": str(e)
            }), 400

        # Preprocesamiento y predicción
        scaled_features = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        return jsonify({
            'prediction': 'Maligno' if prediction == 1 else 'Benigno',
            'probability': float(probability),
            'status': 'success'
        })

    except Exception as e:
        app.logger.error(f"Error en predicción: {str(e)}")
        return jsonify({
            "error": "Error interno del servidor",
            "details": str(e)
        }), 500

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({
        "status": "ready",
        "model_loaded": True,
        "required_features": REQUIRED_FEATURES
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
