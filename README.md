
LG – Flask API on Vercel

This project provides a Flask API for running machine learning models.
It is deployed on Vercel, making it easy to test and share with others (professor, teammates, etc.).

 Project Structure
SuccessLG/
│── app.py              # Flask app
│── requirements.txt    # Python dependencies
│── runtime.txt         # Python version for Vercel
│── vercel.json         # Vercel configuration
│── models/             # Folder containing .pkl trained models
│── README.md           # Project documentation

 Required Files

requirements.txt

Flask==3.1.1
gunicorn==21.2.0
pandas==2.2.2
numpy==2.0.2
scikit-learn==1.6.1
joblib==1.5.1


runtime.txt

python-3.11.9


vercel.json

{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}


app.py

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

Deployment Guide (Vercel)

Push project to GitHub

Include all files above

Ensure models/ contains your .pkl files

Login to Vercel

Go to https://vercel.com

Click New Project → Import GitHub Repo

Configure project

Vercel auto-detects Python runtime

Ensure it points to app.py

Deploy

Vercel builds and deploys the app

You’ll get a public link like:

https://successlg.vercel.app

 Testing the API

Check home endpoint

curl https://successlg.vercel.app/


Response:

{"message": "✅ SuccessLG API running on Vercel"}


Send a prediction request

curl -X POST https://successlg.vercel.app/predict/your_model \
     -H "Content-Type: application/json" \
     -d '{"features": [45, 60000, 70, 0.85]}'


Response:

{
  "model": "your_model",
  "prediction": [1]
}

 Notes for Professor

The project is live on Vercel (no local setup required).

All models are preloaded from the models/ folder.

API supports multiple models → call /predict/<model_name> with JSON features.
