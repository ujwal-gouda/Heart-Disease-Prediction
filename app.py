import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}
</style>
""", unsafe_allow_html=True)

model = load_model("heart.keras")
df = pd.read_csv("cleanedDataFrame.csv")

st.title("❤️ Heart Disease Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, step=1)
    restingBP = st.number_input("Resting BP", min_value=0)
    cholesterol = st.number_input("Cholesterol", min_value=0)
    fastingBS = st.number_input("Fasting Blood Sugar", min_value=0)

with col2:
    sex = st.selectbox("Gender", ["Male", "Female"])
    maxHR = st.number_input("Maximum Heart Rate", min_value=0)
    exerciseAngina = st.selectbox("Exercise Angina", ["Yes", "No"])
    oldPeak = st.number_input("Oldpeak", min_value=0.0, step=0.1)

with col3:
    stSlope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])
    chestPainType = st.selectbox(
        "Chest Pain Type",
        ["ASY", "NAP", "ATA", "TA"]
    )
    restingECG = st.selectbox(
        "Resting ECG",
        ["Normal", "LVH", "ST"]
    )

st.divider()

if st.button("Predict", use_container_width=True):
    # Map ST Slope to numeric (match training encoding)
    if stSlope == 'Down':
        stSlope_num = 0
    elif stSlope == 'Flat':
        stSlope_num = 1
    elif stSlope == 'Up':
        stSlope_num = 2

    # Preserve original strings for display
    chestPain_str = chestPainType
    restingECG_str = restingECG

    # One-hot encode chest pain into the same column order used during training
    if chestPain_str == 'ASY':
        chestPain_arr = np.array([1, 0, 0, 0])
    elif chestPain_str == 'ATA':
        chestPain_arr = np.array([0, 1, 0, 0])
    elif chestPain_str == 'NAP':
        chestPain_arr = np.array([0, 0, 1, 0])
    elif chestPain_str == 'TA':  # 'TA'
        chestPain_arr = np.array([0, 0, 0, 1])

    # One-hot encode resting ECG
    if restingECG_str == 'LVH':
        restingECG_arr = np.array([1, 0, 0])
    elif restingECG_str == 'Normal':
        restingECG_arr = np.array([0, 1, 0])
    elif restingECG_str == 'ST':  # 'ST'
        restingECG_arr = np.array([0, 0, 1])

    # Build flattened feature vector in the same order used for training
    features = [
        age,
        1 if sex == 'Male' else 0,
        restingBP,
        cholesterol,
        fastingBS,
        maxHR,
        1 if exerciseAngina == 'No' else 0,
        oldPeak,
        stSlope_num,
    ]

    features += chestPain_arr.tolist()
    features += restingECG_arr.tolist()

    predictArray = np.array([features]).astype(float)
    
    sampleDF = pd.DataFrame(predictArray, columns=[
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
    ])
    st.dataframe(sampleDF)

    scaler = joblib.load('scaler.joblib')
    sampleDF = scaler.transform(sampleDF)
    
    predict = model.predict(sampleDF)
    
    st.write(f'{predict[0][0] * 100}%')
    if predict[0][0] >= 0.5: 
        st.error("Have Heart Disease")
    else: 
        st.success("Dont have heart Disease")