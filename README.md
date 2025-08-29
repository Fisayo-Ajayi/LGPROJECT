# 📊 LGPROJECT API

A lightweight **Flask API** deployed on **Vercel** for running predictions and transformations using pre-trained machine learning models, such as **KMeans** and **Scalers**.

---

## 🚀 Features
- ✅ Supports multiple models (`.pkl` files stored in `/models/`)
- ✅ Loads models only when requested (saves memory and keeps deployment size small)
- ✅ `/models` endpoint → lists available models
- ✅ `/predict/<model_name>` endpoint → accepts JSON payload and returns predictions or transformations
- ✅ CORS enabled for smooth frontend integration
- ✅ Clear error handling and logging

---

## 📂 Project Structure
LGPROJECT/
│── app.py # Main Flask application
│── models/ # Directory containing your .pkl models
│── requirements.txt # Python dependencies
│── vercel.json # Deployment configuration
│── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Endpoints

### **GET /**  
Returns API health status and available models.
```json
{
  "message": "✅ SuccessLG API running on Vercel",
  "available_models": ["kmeans_model", "scaler"]
}
GET /models
Lists all models currently available in /models/.

json
Copy code
{
  "models": ["kmeans_model", "scaler"]
}
POST /predict/<model_name>
Run prediction or transformation with the selected model.

Request Example:

json
Copy code
{
  "features": [25, 55000, 80, 0.65]
}
Response (KMeans example):

json
Copy code
{
  "model": "kmeans_model",
  "prediction": [2]
}
Response (Scaler example):

json
Copy code
{
  "model": "scaler",
  "prediction": [[-0.75, 0.23, 1.12, -0.65]]
}
🛠️ Local Development
To run the project locally:

bash
Copy code
# Install dependencies
pip install -r requirements.txt

# Start Flask app
python app.py
🌐 Deployment on Vercel
Vercel uses vercel.json to configure builds and routing.

Models must be small (each under ~24 MB unzipped) to fit within Vercel’s function size limits.

Recommended: only keep essential .pkl models in /models.

📜 Dependencies
The project uses the following Python libraries (see requirements.txt):

Flask

Flask-Cors

Gunicorn

Numpy

Scikit-learn

Joblib
