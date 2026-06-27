from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler only once
model = load_model("heart.keras")
scaler = joblib.load("scaler.joblib")


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
        # ST Slope Encoding
        # ==========================

        stSlope_num = {
            "Down": 0,
            "Flat": 1,
            "Up": 2
        }[stSlope]

        # ==========================
        # Chest Pain One-Hot Encoding
        # ==========================

        chestPain_arr = {
            "ASY": [1, 0, 0, 0],
            "ATA": [0, 1, 0, 0],
            "NAP": [0, 0, 1, 0],
            "TA":  [0, 0, 0, 1]
        }[chestPainType]

        # ==========================
        # Resting ECG One-Hot Encoding
        # ==========================

        restingECG_arr = {
            "LVH": [1, 0, 0],
            "Normal": [0, 1, 0],
            "ST": [0, 0, 1]
        }[restingECG]

        # ==========================
        # Feature Vector
        # ==========================

        features = [

            age,

            1 if sex == "Male" else 0,

            restingBP,

            cholesterol,

            fastingBS,

            maxHR,

            1 if exerciseAngina == "No" else 0,

            oldPeak,

            stSlope_num

        ]

        features += chestPain_arr
        features += restingECG_arr

        # ==========================
        # DataFrame
        # ==========================

        sampleDF = pd.DataFrame(
            [features],
            columns=[
                "Age",
                "Sex",
                "RestingBP",
                "Cholesterol",
                "FastingBS",
                "MaxHR",
                "ExerciseAngina",
                "Oldpeak",
                "ST_Slope",
                "ChestPainType_ASY",
                "ChestPainType_ATA",
                "ChestPainType_NAP",
                "ChestPainType_TA",
                "RestingECG_LVH",
                "RestingECG_Normal",
                "RestingECG_ST"
            ]
        )

        # ==========================
        # Scale Features
        # ==========================

        sampleDF = scaler.transform(sampleDF)

        # ==========================
        # Predict
        # ==========================

        predict = model.predict(sampleDF, verbose=0)

        probability = round(float(predict[0][0]) * 100, 2)

        if predict[0][0] >= 0.5:
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