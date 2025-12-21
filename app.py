from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("credit_card_fraud_random_forest.pkl")

# Feature order (VERY IMPORTANT – must match training)
FEATURES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount_log','Hour'
]
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()

    input_df = pd.DataFrame([data], columns=FEATURES)

    prob = model.predict_proba(input_df)[0][1]
    pred = int(prob >= 0.5)

    return jsonify({
        "prediction": pred,
        "fraud_probability": float(prob)
    })

if __name__ == "__main__":
    app.run(debug=True)