from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from newspaper import Article
import nltk
import logging
from hashlib import sha256

# Setup Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')

# Load BART summarization pipeline
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
    logger.info("✅ Loaded BART model from Hugging Face Hub")
except Exception as e:
    logger.exception("❌ Failed to load BART model")
    summarizer = None

MAX_INPUT_TOKENS = 1024
summary_cache = {}  # In-memory cache


def summarize_long_article(text):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(summarizer.tokenizer.encode(current_chunk + sentence)) < MAX_INPUT_TOKENS:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    logger.info(f"📚 Article split into {len(chunks)} chunks")

    summaries = []
    for chunk in chunks:
        try:
            input_ids = summarizer.tokenizer.encode(chunk, return_tensors="pt", max_length=MAX_INPUT_TOKENS, truncation=True)
            summary_ids = summarizer.model.generate(
                input_ids,
                max_length=150,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = summarizer.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        except Exception as e:
            logger.error(f"❌ Failed to summarize chunk: {e}")

    return " ".join(summaries)


@app.route('/bart-summarize', methods=['POST'])
def bart_summarize():
    if not summarizer:
        return jsonify({"error": "Model not available"}), 503

    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    url_hash = sha256(url.encode()).hexdigest()
    if url_hash in summary_cache:
        logger.info("✅ Returning cached summary")
        return jsonify({"summary": summary_cache[url_hash]})

    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        if not text or not text.strip():
            return jsonify({"error": "No text extracted from URL"}), 400

        summary = summarize_long_article(text)
        summary_cache[url_hash] = summary
        return jsonify({"summary": summary})

    except Exception as e:
        logger.exception("❌ Error during summarization")
        return jsonify({"error": "Summarization failed", "details": str(e)}), 500


@app.route('/healthz')
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    logger.info("🚀 Running summarization server...")
    app.run(host='0.0.0.0', port=5000)
