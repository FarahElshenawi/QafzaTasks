import json
import requests

def predict_diabetes(input_data: dict) -> str:
    """
    Sends a POST request to the diabetes prediction API and returns the prediction result.

    Args:
        input_data (dict): A dictionary containing input features for prediction.

    Returns:
        str: The prediction result as a string.
    """
    # URL of the FastAPI endpoint
    url = 'http://127.0.0.1:8000/diabetes_prediction'

    try:
        # Send POST request with JSON data
        response = requests.post(url, json=input_data)

        # Raise an exception for HTTP errors
        response.raise_for_status()  

        # Return the server response
        return response.text

    except requests.exceptions.RequestException as e:
        # Handle any request errors (e.g., connection errors)
        return f"An error occurred: {e}"


if __name__ == "__main__":
    # Input data for the model
    input_data_for_model = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,  # Corrected from 3.6 to 33.6
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }

    # Get the prediction result
    prediction_result = predict_diabetes(input_data_for_model)

    # Print the prediction result
    print(prediction_result)
