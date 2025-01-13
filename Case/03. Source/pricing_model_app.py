import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fastapi import FastAPI
import joblib
import uvicorn
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  

from fastapi.middleware.cors import CORSMiddleware


# Construção da API para utilização dos modelos construídos
import pandas as pd
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/keepalive', methods=['GET'])
def api_health():
    return jsonify(Message="Success")

@app.route("/predict", methods=["POST"])
def predict():
    json_ = request.json
    df = pd.DataFrame(json_)
    prediction = model.predict(df)
    return jsonify({"Prediction": list(prediction)})

if __name__ == "__main__":
    app.run(debug=True)