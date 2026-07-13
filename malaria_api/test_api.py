# test_api.py
import requests

BASE_URL = "http://127.0.0.1:5000"

# Test 1 — Health check
response = requests.get(f"{BASE_URL}/")
print("Health check:", response.json())

# Test 2 — Prediction
patient_data = {
    "age": 25,
    "fever": 1,
    "chills": 1,
    "headache": 1,
    "temperature": 39.5
}

response = requests.post(
    f"{BASE_URL}/predict",
    json=patient_data
)

print("\nPrediction result:")
print(response.json())