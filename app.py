import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))



@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the features and solar rating from the form
        feature_names = [
            'latitude', 'longitude', 'altitude', 'humidity', 
            'ambient_temp', 'wind_speed', 'pressure', 
            'cloud_ceiling','month', 'day'
        ]
        features = [float(request.form[name]) for name in feature_names]
        solar_rating = float(request.form['solar_rating'])

        # Convert features to numpy array
        features_array = np.array(features).reshape(1, -1)

        # Predict the output power
        prediction = model.predict(features_array)[0]

        # Multiply by the solar rating
        final_output = prediction * solar_rating

        return render_template('index.html', prediction_text=f'Predicted Solar Output Energy: {final_output:.2f} kW/h')
    except ValueError as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)