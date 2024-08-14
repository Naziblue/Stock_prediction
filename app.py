from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

app = Flask(__name__)
CORS(app)

# Define the directory where the model and scaler are saved
save_dir = '/tmp'
model_path = os.path.join(save_dir, 'my_model.keras')
scaler_path = os.path.join(save_dir, 'scaler.pkl')

# Load your model and scaler
model = load_model(model_path)
scaler = joblib.load(scaler_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive data in the original price range
        data = request.json['data']
        
        # Convert input data to numpy array
        input_data = np.array(data).reshape(-1, 1)
        
        # Ensure the input data is within the scaler's min and max range
        print(f"Input data before scaling: {input_data.flatten()}")

        # Scale the input data
        input_data_scaled = scaler.transform(input_data).reshape(1, -1, 1)

        # Ensure the scaled data is within the expected range
        print(f"Input data after scaling: {input_data_scaled.flatten()}")

        # Make a prediction
        prediction_scaled = model.predict(input_data_scaled)
        prediction = scaler.inverse_transform(prediction_scaled).flatten()

        print("Input data:", input_data.flatten())
        print("Scaled data:", input_data_scaled.flatten())
        print("Prediction (scaled):", prediction_scaled.flatten())
        print("Prediction (inverse scaled):", prediction)

        return jsonify(prediction=prediction.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
