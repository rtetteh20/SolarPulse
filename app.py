import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
     # Extract features from form
    features = [float(request.form[f'feature {i}']) for i in range(1, 15)]
    features = np.array(features).reshape(1, -1)

    
    # Predict using the model
    prediction = model.predict(features)[0]
    
    # Render the result
    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
