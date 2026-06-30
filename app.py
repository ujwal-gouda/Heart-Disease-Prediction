from pathlib import Path

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Load the trained scikit-learn pipeline only once
model = joblib.load(BASE_DIR / "model.joblib")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        # ==========================
        # Get Form Data
        # ==========================

        age = int(request.form["age"])
        sex = request.form["sex"]
        restingBP = int(request.form["restingBP"])
        cholesterol = int(request.form["cholesterol"])
        fastingBS = int(request.form["fastingBS"])
        maxHR = int(request.form["maxHR"])
        exerciseAngina = request.form["exerciseAngina"]
        oldPeak = float(request.form["oldPeak"])
        stSlope = request.form["stSlope"]
        chestPainType = request.form["chestPainType"]
        restingECG = request.form["restingECG"]

        # ==========================
        # Build Feature DataFrame
        # ==========================

        sampleDF = pd.DataFrame([
            {
                "Age": age,
                "Sex": 1 if sex == "Male" else 0,
                "RestingBP": restingBP,
                "Cholesterol": cholesterol,
                "FastingBS": fastingBS,
                "MaxHR": maxHR,
                "ExerciseAngina": 1 if exerciseAngina == "Yes" else 0,
                "Oldpeak": oldPeak,
                "ST_Slope": {
                    "Down": 0,
                    "Flat": 1,
                    "Up": 2
                }[stSlope],
                "ChestPainType": chestPainType,
                "RestingECG": restingECG
            }
        ])

        # ==========================
        # Predict
        # ==========================

        predict = model.predict(sampleDF)
        probability = round(float(model.predict_proba(sampleDF)[0][1]) * 100, 2)

        if predict[0] == 1:
            prediction = "Have Heart Disease"
            prediction_class = "danger"
        else:
            prediction = "Don't Have Heart Disease"
            prediction_class = "success"

        return render_template(
            "index.html",
            prediction=prediction,
            probability=probability,
            prediction_class=prediction_class
        )

    return render_template(
        "index.html",
        prediction=None,
        probability=None
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)