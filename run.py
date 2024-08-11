from flask import Flask, request, jsonify
import joblib
import logging
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# The Random Forest Model is be loaded here
model = joblib.load("RandomForest_Best_fraud_detection_model.pkl")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/', methods=['GET'])
def home():
	return "Welcome To Fraud Detection System"
	
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)

        # Convert JSON data to a 2D array for the model
        input_data = np.array([[
            data["V1"], data["V2"], data["V3"], data["V4"], data["V5"], data["V6"],
            data["V7"], data["V8"], data["V9"], data["V10"], data["V11"], data["V12"],
            data["V13"], data["V14"], data["V15"], data["V16"], data["V17"], data["V18"],
            data["V19"], data["V20"], data["V21"], data["V22"], data["V23"], data["V24"],
            data["V25"], data["V26"], data["V27"], data["V28"], data["scaled_amount"],
            data["scaled_time"]
        ]])

        # Ensure the model is the correct type
        if hasattr(model, 'predict'):
            # Make a prediction using the model
            prediction = model.predict(input_data)
            return jsonify({
                "prediction": f"{int(prediction[0])}; fraud" if int(prediction[0]) == 1 else f"{int(prediction[0])}; not fraud"
            })
        else:
            return jsonify({"error": "Model does not support prediction"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Change the port here

