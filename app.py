# app.py
from flask import Flask, render_template, request, jsonify, send_file
import joblib
import os
import io
import csv
import pandas as pd
import numpy as np
import logging
import traceback

app = Flask(__name__, static_folder='static', template_folder='templates')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_model_no_preg.pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Make sure diabetes_model_no_preg.pkl is present.")


model_bundle = joblib.load(MODEL_PATH)
if isinstance(model_bundle, dict) and 'model' in model_bundle and 'features' in model_bundle:
    model = model_bundle['model']
    features = model_bundle['features']
else:
    model = model_bundle
    features = getattr(model, "feature_names_in_", None)
    if features is None:
        features = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Insulin']
    logger.warning("Model bundle didn't contain explicit 'model'/'features'. Using inferred/fallback features: %s", features)

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        text = request.form.get('feedback', '').strip()
        if text:
            with open('feedback.txt', 'a', encoding='utf-8') as f:
                f.write(text.replace('\n', ' ') + '\n')
        return render_template('feedback_thanks.html')
    return render_template('feedback.html')

@app.route('/debug_model', methods=['GET'])
def debug_model():
    try:
        info = {
            "model_type": str(type(model)),
            "n_features_in": getattr(model, "n_features_in_", None),
            "feature_names_in": getattr(model, "feature_names_in_", None),
            "bundle_features": features
        }
        logger.info("debug_model: %s", info)
        return jsonify(info)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("debug_model failed")
        return jsonify({"error": "debug_failed", "detail": str(e), "trace": tb}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON payload with keys matching feature names (features variable).
    Builds a single-row pandas DataFrame with those columns and passes to pipeline.
    Returns probability and binary outcome.
    Also saves the last prediction to last_prediction.csv for download.
    """
    data = request.get_json(silent=True) or request.json or {}
    try:
        row = {}
        for f in features:
            val = data.get(f, 0)
            if val == "" or val is None:
                val = 0
            row[f] = float(val)
        X_df = pd.DataFrame([row], columns=features)
        logger.info("Predict input row: %s", row)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Invalid input values")
        return jsonify({'error': 'Invalid input values', 'details': str(e), 'trace': tb}), 400

    # Predict
    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_df)[0, 1])
        else:
            pred = int(model.predict(X_df)[0])
            prob = 1.0 if pred == 1 else 0.0
        outcome = int(prob >= 0.5)
        logger.info("Prediction result: outcome=%s prob=%s", outcome, prob)
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Model prediction failed")
        return jsonify({'error': 'Model prediction failed', 'details': str(e), 'trace': tb}), 500

    try:
        out_path = 'last_prediction.csv'
        with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(features + ['Probability'])
            writer.writerow([row[f] for f in features] + [round(prob, 6)])
    except Exception:
        logger.exception("Failed to save last_prediction.csv (ignored)")

    return jsonify({'probability': prob, 'outcome': outcome})

@app.route('/download/sample')
def download_sample():
    sample = "Glucose,BloodPressure,BMI,DiabetesPedigreeFunction,Age,Prediction\n148,72,33.6,0.627,50,0.85\n110,74,25.6,0.351,30,0.12\n"
    return send_file(io.BytesIO(sample.encode()), mimetype='text/csv', as_attachment=True, download_name='sample_predictions.csv')

@app.route('/download/last')
def download_last():
    p = 'last_prediction.csv'
    if os.path.exists(p):
        return send_file(p, as_attachment=True)
    return ('No saved prediction', 404)

if __name__ == '__main__':
    # Use host 0.0.0.0 so container platforms can reach it; port from env or 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
