from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models
models = {}
for fname in os.listdir("models"):
    if fname.endswith(".pkl"):
        model_name = fname.replace(".pkl", "")
        models[model_name] = joblib.load(os.path.join("models", fname))

@app.route("/")
def home():
    return jsonify({"message": "✅ SuccessLG API running on Vercel"})

@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if model_name not in models:
        return jsonify({"error": "Model not found"}), 404
    
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = models[model_name].predict(features)
    return jsonify({
        "model": model_name,
        "prediction": prediction.tolist()
    })
