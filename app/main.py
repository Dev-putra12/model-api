from flask import Flask, request, jsonify
from .utils import SummaryGenerator, BartSummaryGenerator, SummaryEvaluator
from flask_cors import CORS
import os
import logging
import nltk
from typing import Dict, Any, Union, List


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Di main.py, bagian download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Tambahkan ini
    nltk.download('stopwords')
    nltk.download('indonesian')
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {str(e)}")


app = Flask(__name__)
CORS(app)

# ----------------------------------------------bart section----------------------------------------------

# Initialize model bart
bart_model_path = os.path.join(os.path.dirname(__file__), "model", "indonesian-summarizer-bart")
bart_generator = BartSummaryGenerator()

# load model
with app.app_context():
    """Load model bart before first request"""
    logger.info(f"Loading model from: {bart_model_path}")
    success = bart_generator.load_model(bart_model_path)  # Ubah dari load_bart_model
    if not success:
        logger.error("Failed to load model")
        raise RuntimeError("Failed to load model")
    logger.info("Model loaded successfully")

@app.route('/bart-summarize', methods=['POST'])
def bart_summarize() -> tuple[Dict[str, Any], int]:  # Ubah nama function
    """
    Endpoint for text summarization
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        data = request.get_json()
        if not data or "text" not in data:  # Ubah dari bart_text
            return {"error": "No text provided"}, 400

        text = data["text"]  # Ubah dari bart_text
        if not isinstance(text, str):
            return {"error": "Text must be a string"}, 400

        if not text.strip():
            return {"error": "Text cannot be empty"}, 400

        summary = bart_generator.generate_summary(text)  # Ubah dari generate_bart_summary
        
        if summary is None:
            return {"error": "Failed to generate summary"}, 500

        return {"summary": summary}, 200

    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}", exc_info=True)
        return {"error": str(e)}, 500


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

# ------------------------------evaluation section----------------------------------------------
# Tambahkan endpoint evaluasi untuk BART
@app.route('/bart-evaluate', methods=['POST'])
def bart_evaluate() -> tuple[Dict[str, Any], int]:
    """
    Endpoint for evaluating BART summary against reference summary
    
    Expected JSON input:
    {
        "reference_text": "text of reference summary",
        "generated_summary": "text of generated summary"
    }
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        data = request.get_json()
        if not data or "reference_text" not in data or "generated_summary" not in data:
            return {"error": "Both reference_text and generated_summary must be provided"}, 400

        reference_text = data["reference_text"]
        generated_summary = data["generated_summary"]

        # Validate inputs
        if not all(isinstance(text, str) for text in [reference_text, generated_summary]):
            return {"error": "Both texts must be strings"}, 400

        if not all(text.strip() for text in [reference_text, generated_summary]):
            return {"error": "Texts cannot be empty"}, 400

        # Get evaluation metrics
        metrics = bart_generator.evaluate_summary(reference_text, generated_summary)
        
        return {
            "metrics": metrics
            # "reference_text": reference_text,
            # "generated_summary": generated_summary
        }, 200

    except Exception as e:
        logger.error(f"Error in bart_evaluate endpoint: {str(e)}", exc_info=True)
        return {"error": str(e)}, 500

# Tambahkan endpoint evaluasi untuk mBART
@app.route('/mbart-evaluate', methods=['POST'])
def mbart_evaluate() -> tuple[Dict[str, Any], int]:
    """
    Endpoint for evaluating mBART summary against reference summary
    
    Expected JSON input:
    {
        "reference_text": "text of reference summary",
        "generated_summary": "text of generated summary"
    }
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        data = request.get_json()
        if not data or "reference_text" not in data or "generated_summary" not in data:
            return {"error": "Both reference_text and generated_summary must be provided"}, 400

        reference_text = data["reference_text"]
        generated_summary = data["generated_summary"]

        # Validate inputs
        if not all(isinstance(text, str) for text in [reference_text, generated_summary]):
            return {"error": "Both texts must be strings"}, 400

        if not all(text.strip() for text in [reference_text, generated_summary]):
            return {"error": "Texts cannot be empty"}, 400

        # Get evaluation metrics
        metrics = generator.evaluate_summary(reference_text, generated_summary)
        
        return {
            "metrics": metrics
            # "reference_text": reference_text,
            # "generated_summary": generated_summary
        }, 200

    except Exception as e:
        logger.error(f"Error in mbart_evaluate endpoint: {str(e)}", exc_info=True)
        return {"error": str(e)}, 500

# Tambahkan endpoint untuk evaluasi batch/multiple summaries
@app.route('/bart-evaluate-batch', methods=['POST'])
def bart_evaluate_batch() -> tuple[Dict[str, Any], int]:
    """
    Endpoint for evaluating multiple BART summaries
    
    Expected JSON input:
    {
        "reference_texts": ["reference1", "reference2", ...],
        "generated_summaries": ["summary1", "summary2", ...]
    }
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        data = request.get_json()
        if not data or "reference_texts" not in data or "generated_summaries" not in data:
            return {"error": "Both reference_texts and generated_summaries must be provided"}, 400

        reference_texts = data["reference_texts"]
        generated_summaries = data["generated_summaries"]

        # Validate inputs
        if not isinstance(reference_texts, list) or not isinstance(generated_summaries, list):
            return {"error": "Both inputs must be lists"}, 400

        if len(reference_texts) != len(generated_summaries):
            return {"error": "Number of reference and generated summaries must match"}, 400

        # Get evaluation metrics
        results = bart_generator.evaluator.evaluate_multiple_summaries(
            reference_texts, 
            generated_summaries
        )
        
        return {
            "results": results
        }, 200

    except Exception as e:
        logger.error(f"Error in bart_evaluate_batch endpoint: {str(e)}", exc_info=True)
        return {"error": str(e)}, 500

# Tambahkan endpoint untuk evaluasi batch/multiple summaries mBART
@app.route('/mbart-evaluate-batch', methods=['POST'])
def mbart_evaluate_batch() -> tuple[Dict[str, Any], int]:
    """
    Endpoint for evaluating multiple mBART summaries
    
    Expected JSON input:
    {
        "reference_texts": ["reference1", "reference2", ...],
        "generated_summaries": ["summary1", "summary2", ...]
    }
    
    Returns:
        tuple: (response_dict, status_code)
    """
    try:
        data = request.get_json()
        if not data or "reference_texts" not in data or "generated_summaries" not in data:
            return {"error": "Both reference_texts and generated_summaries must be provided"}, 400

        reference_texts = data["reference_texts"]
        generated_summaries = data["generated_summaries"]

        # Validate inputs
        if not isinstance(reference_texts, list) or not isinstance(generated_summaries, list):
            return {"error": "Both inputs must be lists"}, 400

        if len(reference_texts) != len(generated_summaries):
            return {"error": "Number of reference and generated summaries must match"}, 400

        # Get evaluation metrics
        results = generator.evaluator.evaluate_multiple_summaries(
            reference_texts, 
            generated_summaries
        )
        
        return {
            "results": results
        }, 200

    except Exception as e:
        logger.error(f"Error in mbart_evaluate_batch endpoint: {str(e)}", exc_info=True)
        return {"error": str(e)}, 500
# end
@app.errorhandler(Exception)
def handle_error(error: Exception) -> tuple[Dict[str, str], int]:
    """Global error handler"""
    logger.error(f"Unhandled error: {str(error)}", exc_info=True)
    return {"error": "Internal server error"}, 500

if __name__ == '__main__':
    app.run(debug=True)
