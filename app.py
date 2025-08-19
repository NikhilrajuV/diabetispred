import pickle
import numpy as np
from flask import Flask, request, render_template

# Load the saved model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the HTML form
    features = [
        float(request.form['Pregnancies']),
        float(request.form['Glucose']),
        float(request.form['BloodPressure']),
        float(request.form['SkinThickness']),
        float(request.form['Insulin']),
        float(request.form['BMI']),
        float(request.form['DiabetesPedigreeFunction']),
        float(request.form['Age'])
    ]

    # Convert to numpy array for prediction
    input_array = np.array([features])
    prediction = model.predict(input_array)[0]

    # Map output
    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
