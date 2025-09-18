from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('Model.pkl')
pipeline = joblib.load('Pipeline.pkl')


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

