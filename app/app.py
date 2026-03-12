"""
app.py — Flask API for Loan Default Prediction

WHAT THIS DOES:
- Loads the trained model, scaler, and feature names from files
- Creates a web server with endpoints:
    /           → Health check (is the API alive?)
    /predict    → Takes customer data, returns default prediction
    /features   → Returns what input fields are expected

Think of it like:
- The KITCHEN in a restaurant
- Receives orders (POST requests with customer data)
- Uses the recipe (trained model) to prepare the dish (prediction)
- Sends back the plate (JSON response with prediction + probability)
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# ============================================================
# Initialize Flask App
# ============================================================
# Flask is a lightweight web framework for Python
# It turns your Python script into a web server that can receive HTTP requests

app = Flask(__name__)

# ============================================================
# Load the Trained Model, Scaler, and Feature Names
# ============================================================
# These were saved by train_model.py
# We load them ONCE when the server starts (not on every request — that would be slow)

MODEL_DIR = os.environ.get("MODEL_DIR", "model")  # Can be overridden by Docker env variable

print("🔄 Loading model artifacts...")
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
print(f"✅ Model loaded! Expected features: {feature_names}")


# ============================================================
# Route 1: Health Check (GET /)
# ============================================================
# This is like asking "Hey kitchen, are you open?"
# Useful for Docker health checks and monitoring

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "Loan Default Prediction",
        "version": "1.0",
        "features_expected": feature_names
    })


# ============================================================
# Route 2: Get Feature Info (GET /features)
# ============================================================
# Returns what fields the model expects — helpful for the dashboard

@app.route("/features", methods=["GET"])
def get_features():
    feature_info = {
        "income": {"description": "Annual income in USD", "example": 75000, "range": "20000-150000"},
        "loan_amount": {"description": "Requested loan amount in USD", "example": 15000, "range": "1000-50000"},
        "credit_score": {"description": "Credit score", "example": 680, "range": "300-850"},
        "months_employed": {"description": "Months at current job", "example": 48, "range": "0-360"},
        "num_credit_lines": {"description": "Number of open credit lines", "example": 5, "range": "0-20"},
        "interest_rate": {"description": "Loan interest rate (%)", "example": 12.5, "range": "5.0-25.0"},
        "loan_term": {"description": "Loan term in months", "example": 36, "range": "12, 24, 36, 48, 60"},
        "dti_ratio": {"description": "Debt-to-income ratio", "example": 0.35, "range": "0.05-0.80"},
    }
    return jsonify(feature_info)


# ============================================================
# Route 3: Predict (POST /predict)
# ============================================================
# This is the MAIN endpoint — the reason this API exists
# It receives customer data and returns a prediction
#
# Example request body (JSON):
# {
#   "income": 75000,
#   "loan_amount": 15000,
#   "credit_score": 680,
#   "months_employed": 48,
#   "num_credit_lines": 5,
#   "interest_rate": 12.5,
#   "loan_term": 36,
#   "dti_ratio": 0.35
# }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate: Make sure all required features are present
        missing = [f for f in feature_names if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing fields: {missing}",
                "required_fields": feature_names
            }), 400  # 400 = Bad Request

        # Extract features in the correct order
        # (Order matters! The model was trained with features in a specific order)
        input_values = [data[feature] for feature in feature_names]
        input_array = np.array([input_values])

        # Scale the input (same preprocessing as during training)
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        # Build response
        result = {
            "prediction": int(prediction),
            "label": "Default" if prediction == 1 else "No Default",
            "confidence": {
                "no_default_probability": round(float(probability[0]), 4),
                "default_probability": round(float(probability[1]), 4),
            },
            "input_received": data
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # 500 = Internal Server Error


# ============================================================
# Run the Server
# ============================================================
# host='0.0.0.0' means "accept connections from anywhere" (needed inside Docker)
# port=5000 is the default Flask port

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
