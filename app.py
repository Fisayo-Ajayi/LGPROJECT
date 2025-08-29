from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Cache for loaded models
models = {}

@app.route("/")
def home():
    logger.info("Home endpoint called")
    return jsonify({"message": "✅ SuccessLG API running on Vercel"})

@app.route("/models", methods=["GET"])
def list_models():
    """Return a list of all available .pkl models in the models/ directory"""
    try:
        model_dir = "models"
        if not os.path.exists(model_dir):
            logger.warning("⚠️ Models directory not found")
            return jsonify({"models": [], "message": "No models directory found"}), 404
        
        available_models = [
            f.replace(".pkl", "")
            for f in os.listdir(model_dir)
            if f.endswith(".pkl")
        ]

        logger.info(f"📦 Available models: {available_models}")
        return jsonify({"models": available_models})
    
    except Exception as e:
        logger.exception("🚨 Error listing models")
        return jsonify({"error": str(e)}), 500

@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    try:
        # Load model only if not already cached
        if model_name not in models:
            model_path = os.path.join("models", f"{model_name}.pkl")
            if not os.path.exists(model_path):
                logger.error(f"❌ Model not found: {model_name}")
                return jsonify({"error": "Model not found"}), 404
            
            models[model_name] = joblib.load(model_path)
            logger.info(f"✅ Loaded model: {model_name}")
        else:
            logger.info(f"♻️ Using cached model: {model_name}")

        # Parse input JSON
        data = request.get_json()
        if not data or "features" not in data:
            logger.error("❌ Invalid request: 'features' missing")
            return jsonify({"error": "Request must contain 'features' key"}), 400
        
        features = data["features"]

        # Validate that features is a list of numbers
        if not isinstance(features, (list, tuple)) or not all(isinstance(x, (int, float)) for x in features):
            logger.error("❌ Invalid request: 'features' must be a list of numbers")
            return jsonify({"error": "'features' must be a list of numbers"}), 400

        # Convert to numpy array for prediction
        features = np.array(features).reshape(1, -1)

        # Run prediction
        prediction = models[model_name].predict(features)
        logger.info(f"Prediction from {model_name}: {prediction.tolist()}")

        return jsonify({
            "model": model_name,
            "prediction": prediction.tolist()
        })
    
    except Exception as e:
        logger.exception("🚨 Error during prediction")
        return jsonify({"error": str(e)}), 500
