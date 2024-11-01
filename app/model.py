import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle


class DiabetesModel:
    def __init__(self, model_path: str = 'diabetes_model.sav'):
        self.model_path = model_path
        self.model = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads the dataset from the specified CSV file.
        """
        return pd.read_csv(file_path)

    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepares the features and target variable from the dataset.
        """
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        return X, y

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains a logistic regression model on the training data.
        """
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Evaluates the trained model and prints the classification report.
        """
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

    def save_model(self) -> None:
        """
        Saves the trained model to the specified file.
        """
        with open(self.model_path, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self) -> None:
        """
        Loads a model from the specified file.
        """
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained model.
        """
        if self.model is None:
           raise ValueError("Model must be loaded before prediction.")
        return self.model.predict(input_data)


if __name__ == "__main__":
    # Instantiate the DiabetesModel class
    diabetes_model = DiabetesModel()

    # Load the dataset
    df = diabetes_model.load_data('data/diabetes.csv')

    # Preprocess the data
    X, y = diabetes_model.preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    diabetes_model.train(X_train, y_train)

    # Evaluate the model
    diabetes_model.evaluate(X_test, y_test)

    # Save the trained classifier
    diabetes_model.save_model()

    # Load the model (if needed later)
    diabetes_model.load_model()

    # Make a sample prediction (for demonstration)
    sample_data = X_test.iloc[140:154].values
    sample_predictions = diabetes_model.predict(sample_data)
    print("Sample predictions from the loaded model:", sample_predictions)
