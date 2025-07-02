from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from newspaper import Article
import os
from hashlib import sha256
import nltk
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Summary cache
summary_cache = {}

# ✅ BART model setup
try:
    model_path = os.path.join(os.path.dirname(__file__), 'bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path, weights_only=False)
    logger.info("✅ BART model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load BART model: {e}")
    model = None
    tokenizer = None

# ✅ Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

@app.route('/bart_summarize', methods=['POST'])
def bart_summarize():
    if not model or not tokenizer:
        return jsonify({"error": "Model not available"}), 503

    data = request.get_json()
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        url_hash = sha256(url.encode()).hexdigest()
        if url_hash in summary_cache:
            logger.info("✅ Returning cached summary")
            return jsonify({"summary": summary_cache[url_hash]})

        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        if not text.strip():
            return jsonify({"error": "No readable content found"}), 400

        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        summary_cache[url_hash] = summary
        return jsonify({"summary": summary})

    except Exception as e:
        logger.error(f"BART summarization failed: {e}")
        return jsonify({"error": "Summarization failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
