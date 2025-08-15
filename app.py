from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")
USD_TO_INR = 83.5  

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
   
    features = {
        'GrLivArea': float(request.form['GrLivArea']),
        'BedroomAbvGr': int(request.form['BedroomAbvGr']),
        'FullBath': int(request.form['FullBath']),
        'HalfBath': int(request.form['HalfBath']),
        'TotRmsAbvGrd': int(request.form['TotRmsAbvGrd'])
    }
    df = pd.DataFrame([features])
    prediction_usd = model.predict(df)[0]
    prediction_inr = prediction_usd * USD_TO_INR
    return render_template("index.html", prediction=f"₹{prediction_inr:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)
