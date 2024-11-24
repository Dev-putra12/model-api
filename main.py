from flask import Flask, request, jsonify
from .utils import SummaryGenerator
import os

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
mbart_model_path = os.path.join(os.path.dirname(__file__), "model/indonesian-summarizer-mbart")
mbart_generator = SummaryGenerator()

@app.before_first_request
def load_model():
    """Load model before first request"""
    success = mbart_generator.load_model(mbart_model_path)
    if not success:
        raise RuntimeError("Failed to load model")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': mbart_generator.model is not None
    })

@app.route('/mbart-summarize', methods=['POST'])
def summarize():
    """Endpoint for text summarization"""
    try:
        # Get input text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided'
            }), 400

        text = data['text']
        
        # Generate summary
        summary = mbart_generator.generate_summary(text)
        
        if summary is None:
            return jsonify({
                'error': 'Failed to generate summary'
            }), 500

        return jsonify({
            'original_text': text,
            'summary': summary
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
