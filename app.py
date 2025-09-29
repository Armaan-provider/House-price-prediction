from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import requests
import gdown

app = Flask(__name__)

MODEL_FILE = "Model.pkl"
PIPELINE_FILE = "Pipeline.pkl"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1Zdyz3l-tvRwvsEyXs8ddkhhmAkhZj0sD"
PIPELINE_URL = "https://drive.google.com/uc?export=download&id=1MrNa7XtTapMOBQLRixyDCWxzKj9l6dlh"

if not os.path.exists(MODEL_FILE):
    try:
        print("Downloading pretrained model...")
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
        gdown.download(PIPELINE_URL, PIPELINE_FILE, quiet=False)
        print("Pretrained model downloaded successfully.")
    except Exception as e:
        print("Download failed. Training locally (if dataset exists)...")


model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        input_data = {
        'longitude': float(request.form['longitude']),
        'latitude': float(request.form['latitude']),
        'housing_median_age': float(request.form['housing_median_age']),
        'total_rooms': float(request.form['total_rooms']),
        'total_bedrooms': float(request.form['total_bedrooms']),
        'population': float(request.form['population']),
        'households': float(request.form['households']),
        'median_income': float(request.form['median_income']),
        'ocean_proximity': request.form['ocean_proximity']
         }
    
        df = pd.DataFrame([input_data])

        prepared_input = pipeline.transform(df)

        prediction = model.predict(prepared_input)[0]

        return render_template('index.html', prediction_text = f'The predicted house value is: ${prediction:,.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text = f'error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)

