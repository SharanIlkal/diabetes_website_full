# app.py
from flask import Flask, render_template, request, jsonify, send_file
import joblib
import os
import io
import csv
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')

# Path to the model file (same directory as app.py)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_model_no_preg.pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Make sure diabetes_model_no_preg.pkl is present.")

model_bundle = joblib.load(MODEL_PATH)
model = model_bundle['model']
features = model_bundle['features']  # list of feature names in order

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

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON payload with keys matching feature names (features variable).
    Builds a single-row pandas DataFrame with those columns and passes to pipeline.
    Returns probability and binary outcome.
    Also saves the last prediction to last_prediction.csv for download.
    """
    data = request.json or {}
    try:
        # Build row dict in exact features order and convert to float
        row = {}
        for f in features:
            # Accept empty or missing as 0; caller may wish to change
            val = data.get(f, 0)
            # If value is empty string, treat as 0
            if val == "" or val is None:
                val = 0
            row[f] = float(val)
        # Create DataFrame so transformers that expect feature names work correctly
        X_df = pd.DataFrame([row], columns=features)

    except Exception as e:
        return jsonify({'error': 'Invalid input values', 'details': str(e)}), 400

    # Predict
    try:
        prob = float(model.predict_proba(X_df)[0, 1])
    except Exception as e:
        return jsonify({'error': 'Model prediction failed', 'details': str(e)}), 500

    outcome = int(prob >= 0.5)

    # Save last prediction as CSV for quick download
    try:
        out_path = 'last_prediction.csv'
        with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(features + ['Probability'])
            writer.writerow([row[f] for f in features] + [round(prob, 6)])
    except Exception:
        # do not fail prediction if saving fails; just ignore
        pass

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
    # Use host 0.0.0.0 so container platforms can reach it; default port 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
