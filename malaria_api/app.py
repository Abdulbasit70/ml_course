# app.py
import os
from functools import wraps
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialise Flask app
app = Flask(__name__)

# In production store this in environment variables — never hardcode
VALID_API_KEYS = {
    'hospital_lagos':  'key_abc123xyz',
    'clinic_abuja':    'key_def456uvw',
    'admin':           'key_ghi789rst'
}

# ─────────────────────────────────────────
# Authentication decorator
# ─────────────────────────────────────────
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check for API key in request headers
        api_key = request.headers.get('X-API-Key')

        if not api_key:
            return jsonify({
                'error': 'No API key provided',
                'hint':  'Include X-API-Key in your request headers'
            }), 401   # 401 = Unauthorised

        if api_key not in VALID_API_KEYS.values():
            return jsonify({
                'error': 'Invalid API key'
            }), 403   # 403 = Forbidden

        return f(*args, **kwargs)
    return decorated

# Load model ONCE when server starts — not on every request
with open('model/malaria_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")

# ─────────────────────────────────────────
# ROUTE 1 — Health check
# ─────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Malaria Detection API is running',
        'status': 'healthy',
        'version': '1.0'
    })

# Apply authentication to predict route
@app.route('/predict', methods=['POST'])
@require_api_key                           # ← one line adds security
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract and validate features
        features = np.array([[
            data['age'],
            data['fever'],
            data['chills'],
            data['headache'],
            data['temperature']
        ]])

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Build response
        result = {
            'prediction': int(prediction),
            'diagnosis': 'Malaria Detected' if prediction == 1
                         else 'No Malaria Detected',
            'confidence': f"{max(probability) * 100:.1f}%",
            'risk_level': 'HIGH' if probability[1] > 0.7
                          else 'MEDIUM' if probability[1] > 0.4
                          else 'LOW'
        }

        return jsonify(result), 200

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────
# Run the server
# ─────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
    pass



# def validate_patient_data(data):
#     errors = []

#     # Check all required fields exist
#     required_fields = ['age', 'fever', 'chills', 'headache', 'temperature']
#     for field in required_fields:
#         if field not in data:
#             errors.append(f"Missing required field: '{field}'")

#     if errors:
#         return errors   # Return early if fields are missing

#     # Validate data types and ranges
#     if not isinstance(data['age'], (int, float)):
#         errors.append("'age' must be a number")
#     elif not (0 < data['age'] < 120):
#         errors.append("'age' must be between 1 and 120")

#     if data['fever'] not in [0, 1]:
#         errors.append("'fever' must be 0 or 1")

#     if data['chills'] not in [0, 1]:
#         errors.append("'chills' must be 0 or 1")

#     if data['headache'] not in [0, 1]:
#         errors.append("'headache' must be 0 or 1")

#     if not isinstance(data['temperature'], (int, float)):
#         errors.append("'temperature' must be a number")
#     elif not (35.0 <= data['temperature'] <= 45.0):
#         errors.append("'temperature' must be between 35.0 and 45.0°C")

#     return errors

# # Updated predict route with validation
# @app.route('/predict', methods=['POST'])
# @require_api_key
# def predict():
#     data = request.get_json()

#     if not data:
#         return jsonify({'error': 'No JSON data provided'}), 400

#     # Validate before touching the model
#     errors = validate_patient_data(data)
#     if errors:
#         return jsonify({
#             'error': 'Validation failed',
#             'details': errors          # tell the caller exactly what's wrong
#         }), 400

#     # Safe to proceed — data is clean
#     features = np.array([[
#         data['age'], data['fever'],
#         data['chills'], data['headache'],
#         data['temperature']
#     ]])

#     prediction   = model.predict(features)[0]
#     probability  = model.predict_proba(features)[0]

#     return jsonify({
#         'prediction':  int(prediction),
#         'diagnosis':   'Malaria Detected' if prediction == 1
#                        else 'No Malaria Detected',
#         'confidence':  f"{max(probability) * 100:.1f}%",
#         'risk_level':  'HIGH'   if probability[1] > 0.7
#                        else 'MEDIUM' if probability[1] > 0.4
#                        else 'LOW'
#     }), 200
















# batch predict
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Get list of patients from request
        data = request.get_json()
        patients = data['patients']

        # Build feature matrix — one row per patient
        features = np.array([[
            p['age'],
            p['fever'],
            p['chills'],
            p['headache'],
            p['temperature']
        ] for p in patients])

        # Predict all at once — one model call, not a loop
        predictions  = model.predict(features)
        probabilities = model.predict_proba(features)

        # Build result for each patient
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'patient_index': i,
                'prediction':    int(pred),
                'diagnosis':     'Malaria Detected' if pred == 1
                                 else 'No Malaria Detected',
                'confidence':    f"{max(prob) * 100:.1f}%",
                'risk_level':    'HIGH'   if prob[1] > 0.7
                                 else 'MEDIUM' if prob[1] > 0.4
                                 else 'LOW'
            })

        return jsonify({
            'total_patients': len(patients),
            'results': results
        }), 200

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500