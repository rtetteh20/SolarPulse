import numpy as np
from flask import Flask, request, jsonify
import pickle
import logging


# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))



@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type != 'application/json':
            return jsonify({"status": False, "message": "Content-Type must be application/json", "data": None}), 415
        # Get the features and solar rating from the request body
        data = request.json
        feature_names = [
            'latitude', 'longitude', 'altitude', 'humidity', 
            'ambient_temp', 'wind_speed', 'pressure', 
            'cloud_ceiling', 'month', 'day'
        ]
        features = [float(data[name]) for name in feature_names]
        print(features)
        solar_rating = float(data['solar_rating'])

        # Convert features to numpy array
        features_array = np.array(features).reshape(1, -1)

        # Predict the output power
        prediction = model.predict(features_array)[0]

        # Multiply by the solar rating
        final_output = prediction * solar_rating

        return jsonify({"status": True, "message": "Prediction provided successfully", "data": final_output}), 200
    
    except ValueError as e:
        logging.error(f"ValueError: {e}")
        return jsonify({"status": False, "message": f"An error occurred: {e}", "data": None}), 500
    
    except Exception as e:
        logging.error(f"Exception: {e}")
        return jsonify({"status": False, "message": "An unexpected error occurred", "data": None}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)