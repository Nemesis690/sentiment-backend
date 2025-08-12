# backend/app.py
import os
import io
import csv
import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# thresholds for mapping
POS_TH = 0.60
NEG_TH = 0.40

if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    raise RuntimeError("Model files not found. Run training/train.py first to create model_files.")

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# determine maxlen from model input shape
try:
    MAX_LEN = model.input_shape[1]
except Exception:
    MAX_LEN = 200

app = Flask(__name__)
CORS(app)

# in-memory history
HISTORY = []

def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    score = float(model.predict(pad, verbose=0)[0][0])  # [0,1] where higher == positive
    if score >= POS_TH:
        sentiment = "Positive"
    elif score <= NEG_TH:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, round(score * 100, 2)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text or not text.strip():
        return jsonify({"error": "No text provided"}), 400

    sentiment, confidence = predict_text(text)
    item = {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    HISTORY.insert(0, item)
    return jsonify(item)

@app.route("/analyze_csv", methods=["POST"])
def analyze_csv():
    """
    Expects multipart/form-data with a file field named 'file' (CSV).
    CSV must contain a column named one of: 'text', 'sentence', 'tweet'.
    Returns JSON { results: [ {text, sentiment, confidence} ] }
    """
    if "file" not in request.files:
        return jsonify({"error": "Please upload a CSV file in field 'file'"}), 400

    file = request.files["file"]
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {str(e)}"}), 400

    # accept various column names
    candidate_cols = ["text", "sentence", "tweet", "review"]
    col = None
    for c in candidate_cols:
        if c in df.columns:
            col = c
            break
    if col is None:
        # try first column if it's not obviously headers-only
        if len(df.columns) >= 1:
            col = df.columns[0]
        else:
            return jsonify({"error": "CSV must contain a text column (e.g. 'text')"}), 400

    results = []
    for _, row in df.iterrows():
        raw_text = str(row[col]) if not pd.isna(row[col]) else ""
        sentiment, confidence = predict_text(raw_text)
        item = {"text": raw_text, "sentiment": sentiment, "confidence": confidence}
        results.append(item)
        HISTORY.insert(0, item)

    return jsonify({"results": results})

@app.route("/history", methods=["GET"])
def history():
    return jsonify({"history": HISTORY[:200]})

@app.route("/history/download", methods=["GET"])
def download_history():
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(["timestamp", "text", "sentiment", "confidence"])
    for it in HISTORY:
        writer.writerow([it.get("timestamp",""), it.get("text",""), it.get("sentiment",""), it.get("confidence","")])
    mem = io.BytesIO()
    mem.write(si.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="history.csv")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
