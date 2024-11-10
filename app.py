from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/bert-summarize', methods=['POST'])
def bert_summarize():    # Nama fungsi diubah menjadi bert_summarize
    data = request.json
    text = data.get('text', '')
    
    # Di sini nanti kita akan memanggil fungsi summarize untuk BERT
    summary = "Ringkasan BERT akan dihasilkan di sini"
    
    return jsonify({
        'model': 'BERT',
        'input_text': text,
        'summary': summary
    })

@app.route('/mbert-summarize', methods=['POST'])
def mbert_summarize():    # Nama fungsi diubah menjadi mbert_summarize
    data = request.json
    text = data.get('text', '')
    
    # Di sini nanti kita akan memanggil fungsi summarize untuk MBERT
    summary = "Ringkasan MBERT akan dihasilkan di sini"
    
    return jsonify({
        'model': 'MBERT',
        'input_text': text,
        'summary': summary
    })

if __name__ == '__main__':
    app.run(debug=True)
