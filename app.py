from flask import Flask, request, jsonify
from .utils import SummaryGenerator
import os

app = Flask(__name__)

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

# Initialize model mbart
model_path = os.path.join(os.path.dirname(__file__), "model/indonesian-summarizer-mbart")
generator = SummaryGenerator()

@app.route('/mbart-summarize', methods=['POST'])
def mbert_summarize():    # Nama fungsi diubah menjadi mbert_summarize
    data = request.json
    text = data.get('text', '')
    
    # Di sini nanti kita akan memanggil fungsi summarize untuk MBERT
    summary = "Ringkasan MBART akan dihasilkan di sini"
    
    return jsonify({
        'model': 'MBART',
        'input_text': text,
        'summary': summary
    })

if __name__ == '__main__':
    app.run(debug=True)
