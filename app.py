from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y scaler
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['radius_mean'], data['texture_mean'], data['perimeter_mean'],
        data['area_mean'], data['smoothness_mean'], data['compactness_mean'],
        data['concavity_mean'], data['concave_points_mean'], data['symmetry_mean'],
        data['fractal_dimension_mean'], data['radius_se'], data['texture_se'],
        data['perimeter_se'], data['area_se'], data['smoothness_se'],
        data['compactness_se'], data['concavity_se'], data['concave_points_se'],
        data['symmetry_se'], data['fractal_dimension_se'], data['radius_worst'],
        data['texture_worst'], data['perimeter_worst'], data['area_worst'],
        data['smoothness_worst'], data['compactness_worst'], data['concavity_worst'],
        data['concave_points_worst'], data['symmetry_worst'], data['fractal_dimension_worst']
    ]])

    # Escalar y predecir
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]  # Probabilidad de maligno

    return jsonify({
        'prediction': 'Maligno' if prediction == 1 else 'Benigno',
        'probability': float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True)
