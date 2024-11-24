from flask import Flask, request, jsonify
from .utils import SummaryGenerator
import os
import logging
from typing import Dict, Any, Union


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ----------------------------------------------bart section----------------------------------------------
@app.route('/bart-summarize', methods=['POST'])
def bert_summarize():    # Nama fungsi diubah menjadi bert_summarize
    data = request.json
    text = data.get('text', '')
    
    # Di sini nanti kita akan memanggil fungsi summarize untuk BERT
    summary = "Ringkasan BART akan dihasilkan di sini"
    
    return jsonify({
        'model': 'BART',
        'input_text': text,
        'summary': summary
    })


# ----------------------------------------------mbart section----------------------------------------------
# Initialize model mbart
model_path = os.path.join(os.path.dirname(__file__), "model", "indonesian-summarizer-mbart")
generator = SummaryGenerator()

# load model
with app.app_context():
    """Load model before first request"""
    logger.info(f"Loading model from: {model_path}")
    success = generator.load_model(model_path)
    if not success:
        logger.error("Failed to load model")
        raise RuntimeError("Failed to load model")
    logger.info("Model loaded successfully")
# health check mbart model
@app.route('/mbart-health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': generator.model is not None
    })

# evaluation mbart model

# summarize with mbart model
@app.route("/mbart-summarize", methods=["POST"])
def summarize() -> tuple[Dict[str, Any], int]:
    """
    Endpoint for text summarization
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return {"error": "No text provided"}, 400

        text = data["text"]
        if not isinstance(text, str):
            return {"error": "Text must be a string"}, 400

        if not text.strip():
            return {"error": "Text cannot be empty"}, 400

        summary = generator.generate_summary(text)
        
        if summary is None:
            return {"error": "Failed to generate summary"}, 500

        return {"summary": summary}, 200

    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}", exc_info=True)
        return {"error": str(e)}, 500

@app.errorhandler(Exception)
def handle_error(error: Exception) -> tuple[Dict[str, str], int]:
    """Global error handler"""
    logger.error(f"Unhandled error: {str(error)}", exc_info=True)
    return {"error": "Internal server error"}, 500

if __name__ == '__main__':
    app.run(debug=True)
