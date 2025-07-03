from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from newspaper import Article
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

# Load summarization pipeline from Hugging Face Hub
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
    logger.info("✅ Loaded BART model from Hugging Face Hub")
except Exception as e:
    logger.error(f"❌ Failed to load BART model: {e}")
    summarizer = None

# Constants
MAX_INPUT_TOKENS = 1024
summary_cache = {}

def chunk_text_by_token_limit(text):
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    
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

@app.route('/bart-summarize', methods=['POST'])
def bart_summarize():
    if not summarizer:
        return jsonify({"error": "Model not available"}), 503

    data = request.get_json()
    url = data.get("url", "").strip()
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

        chunks = chunk_text_by_token_limit(text)
        summaries = []

        for chunk in chunks:
            result = summarizer(chunk, max_length=150, min_length=60, do_sample=False)
            summaries.append(result[0]['summary_text'])

        summary = " ".join(summaries)
        summary_cache[url_hash] = summary
        return jsonify({"summary": summary})

    except Exception as e:
        logger.exception("❌ BART summarization failed")
        return jsonify({"error": "Summarization failed", "details": str(e)}), 500

@app.route('/healthz')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    logger.info("🚀 Running summarization server...")
    app.run(host='0.0.0.0', port=5000)
