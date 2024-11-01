from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

class InputData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

class DiabetesPredictor:
    """
    A class to handle diabetes prediction using a trained model.
    
    Attributes:
        model: The trained model for predicting diabetes.
    """
    
    def __init__(self, model_path: str):
        """
        Initializes the predictor by loading the model.

        Args:
            model_path (str): Path to the trained model file.
        """
        self.model = self.load_model(model_path)

    @staticmethod
    def load_model(model_path: str):
        """Load the model from the given path."""
        with open(model_path, 'rb') as file:
            return pickle.load(file)

    def predict(self, input_data: InputData) -> str:
        """
        Predicts whether a person is diabetic based on input data.

        Args:
            input_data (InputData): An instance of InputData containing features.

        Returns:
            str: Prediction result ("not diabetic" or "diabetic").
        """
        input_list = [
            input_data.Pregnancies,
            input_data.Glucose,
            input_data.BloodPressure,
            input_data.SkinThickness,
            input_data.Insulin,
            input_data.BMI,
            input_data.DiabetesPedigreeFunction,
            input_data.Age
        ]
        prediction = self.model.predict([input_list])
        return "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"


app = FastAPI()
predictor = DiabetesPredictor('D:\\MLOps\\diabetes_prediction_project\\data\\diabetes_model.sav')


@app.get("/")
async def welcome():
    return {"message": "Hello! Here you can predict diabetes at 'http://127.0.0.1:8000/diabetes_prediction'"}


@app.post('/diabetes_prediction')
async def diabetes_pred(input_parameters: InputData):
    """
    Endpoint to predict diabetes based on input parameters.

    Args:
        input_parameters (InputData): Input features for prediction.

    Returns:
        dict: A dictionary with the prediction result.
    """
    prediction_result = predictor.predict(input_parameters)
    return {"prediction": prediction_result}
