import json
import requests

# URL of the FastAPI endpoint
url = 'http://127.0.0.1:8000/diabetes_prediction'

# Input data for the model
input_data_for_model = {
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 3.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50
}

# Send POST request with JSON data
response = requests.post(url, json=input_data_for_model)

# Print the response from the server
print(response.text)
