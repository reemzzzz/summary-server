from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BartTokenizer, BartForConditionalGeneration
from newspaper import Article
import torch
import nltk
import logging
from hashlib import sha256

# Setup Flask and logging
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('punkt')

# Load BART model and tokenizer from local directory
try:
    model_path = "./bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    logger.info("✅ Loaded BART model from local path")
except Exception as e:
    logger.error(f"❌ Failed to load BART model: {e}")
    tokenizer = None
    model = None

# Constants
MAX_INPUT_TOKENS = 1024
summary_cache = {}

def summarize_text(text, max_output=150):
    inputs = tokenizer(text, return_tensors="pt", max_length=MAX_INPUT_TOKENS, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_output,
        min_length=60,
        length_penalty=2.0,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def chunk_text_by_token_limit(text):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sent in sentences:
        tentative_chunk = current_chunk + " " + sent if current_chunk else sent
        input_ids = tokenizer.encode(tentative_chunk, truncation=False)
        if len(input_ids) < MAX_INPUT_TOKENS:
            current_chunk = tentative_chunk
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk.strip())

    logger.info(f"📚 Split into {len(chunks)} chunks")
    return chunks


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


@app.route('/healthz')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    logger.info("🚀 Running summarization server...")
    app.run(host='0.0.0.0', port=5000)
