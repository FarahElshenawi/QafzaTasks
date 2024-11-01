from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    
@app.get("/")
async def welcome():
    return {"Hello, Here you can predict diabetes 'http://127.0.0.1:8000/predict'"}


# Load the model
model = pickle.load(open('D:\MLOps\diabetes_prediction_project\data\diabetes_model.sav', 'rb'))

@app.get('/diabetes_prediction')
async def diabetes_pred(input_parameters: InputData):
    # Convert the input to JSON format
    input_data_json = input_parameters.model_dump_json()
    
    # Optionally, you can log or print the JSON data
    print("Input Data (JSON):", input_data_json)

    # Convert the JSON string back to a dictionary for prediction
    input_data = json.loads(input_data_json)

    # Extract input parameters
    input_list = [
        input_data['Pregnancies'],
        input_data['Glucose'],
        input_data['BloodPressure'],
        input_data['SkinThickness'],
        input_data['Insulin'],
        input_data['BMI'],
        input_data['DiabetesPedigreeFunction'],
        input_data['Age']
    ]
    
    # Make the prediction
    prediction = model.predict([input_list])

    # Return the result as JSON
    if prediction[0] == 0:
        return {"prediction": "The person is not diabetic"}
    else:
        return {"prediction": "The person is diabetic"}
