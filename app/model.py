import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle  # Import pickle for saving/loading the model

# Load the dataset
df = pd.read_csv('data/diabetes.csv')

# Split the dataset into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the classifier
classifier = LogisticRegression(max_iter=200)  # Increased max_iter to avoid convergence warnings
classifier.fit(X_train, y_train)

# Predict the outcomes for the test set
y_pred = classifier.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Save the trained classifier with .sav extension
with open('diabetes_model.sav', 'wb') as file:
    pickle.dump(classifier, file)

# Load the model (if needed later)
with open('diabetes_model.sav', 'rb') as file:
    loaded_classifier = pickle.load(file)


# (Just for demonstration; this step is not necessary in the final implementation)
sample_prediction = loaded_classifier.predict(X_test[:5])
print("Sample predictions from the loaded model:", sample_prediction)
