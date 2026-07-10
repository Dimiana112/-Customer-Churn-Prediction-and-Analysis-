from flask import Flask, request, jsonify
import pandas as pd
import joblib



#home_Endpoint
app = Flask(__name__)

model = joblib.load("../models/best_model.pkl")
@app.route("/")
def home():

    return "Customer Churn Prediction API is Running!"


#Prediction Endpoint

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    probability = model.predict_proba(df)

    return jsonify({

        "Prediction": int(prediction[0]),

        "Probability": float(probability[0][1])

    })

#Run Server
if __name__ == "__main__":

    app.run(debug=True)


