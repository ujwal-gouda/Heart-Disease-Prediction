# Heart Disease Prediction

A Machine Learning-powered web application that predicts the likelihood of heart disease based on a patient's medical information. This project demonstrates an end-to-end machine learning workflow, from data preprocessing and model training to deployment as a live web application using Flask.

## Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help individuals seek timely medical attention. This application uses a trained machine learning model to estimate the likelihood of heart disease based on user-provided health parameters.

> **Disclaimer:** This project is intended for educational purposes only and should not be used as a substitute for professional medical advice or diagnosis.

## Features

* Predicts heart disease risk in real time.
* Simple and responsive web interface.
* User-friendly input form.
* Machine learning model integrated with Flask.
* Fast prediction results.
* Easily deployable on cloud platforms.

## Tech Stack

**Frontend**

* HTML
* CSS
* JavaScript

**Backend**

* Python
* Flask

**Machine Learning**

* Scikit-learn
* Pandas
* NumPy

## Project Structure

```text
heart-disease-prediction/
│── static/                 # CSS, JavaScript, Images
│── templates/              # HTML templates
│── model/                  # Saved ML model
│── dataset/                # Training dataset
│── app.py                  # Flask application
│── train_model.py          # Model training script
│── requirements.txt
│── README.md
```

> Folder names may differ slightly depending on your project.

## Getting Started

### Clone the repository

```bash
git clone https://github.com/ujwal-gouda/heart-disease-prediction.git
```

### Move into the project directory

```bash
cd heart-disease-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Input Parameters

The model predicts heart disease using medical attributes such as:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate Achieved
* Exercise-Induced Angina
* ST Depression (Oldpeak)
* ST Slope
* Number of Major Vessels
* Thalassemia

## Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Model Serialization
7. Flask Integration
8. Web Deployment

## Future Improvements

* Improve prediction accuracy using advanced ML algorithms.
* Compare multiple machine learning models.
* Add prediction probability scores.
* Store prediction history.
* Add authentication for users.
* Deploy using Docker and CI/CD.
* Improve UI/UX with charts and visualizations.

## Author

**Ujwal Gouda**

GitHub: https://github.com/ujwal-gouda

## Support

If you found this project helpful, consider giving it a **⭐ Star** on GitHub.

## License

This project is intended for educational and learning purposes.
