# Diabetes Prediction Web App

This repository contains a simple web application built with Flask to predict diabetes based on user input features using a K-Nearest Neighbors (KNN) classification model trained on the Pima Indians Diabetes dataset.

---

## Project Overview

The web app allows users to input health-related parameters and predicts whether the person is diabetic or not. The prediction is made using a KNN machine learning model that was trained on the well-known diabetes dataset from Kaggle.

---

## Features

- Input form for eight health parameters:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- Model predicts whether the user is **Diabetic** or **Not Diabetic**
- Simple and clean user interface

---

## How to Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/diabetes-prediction-flask.git
   cd diabetes-prediction-flask
Install required packages:

It's recommended to use a virtual environment:

python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Run the Flask app:

python app.py


Open your browser and navigate to http://127.0.0.1:5000/

Input your data and get the diabetes prediction!

Model Training

The model is trained using the K-Nearest Neighbors algorithm from scikit-learn on the Pima Indians Diabetes dataset. The dataset contains various health metrics and a target variable indicating if the person has diabetes (1) or not (0).

Training code snippet:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

data = pd.read_csv("diabetes.csv")
X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open("model.pkl", "wb"))

Dataset

The dataset used is the Pima Indians Diabetes Dataset
 from Kaggle.

Folder Structure
.
├── app.py                # Flask application code
├── model.pkl             # Pickled trained KNN model
├── templates/
│   └── index.html        # HTML template for the web app
├── diabetes.csv          # Dataset used for training (optional)
└── README.md       
