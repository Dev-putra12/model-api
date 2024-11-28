# 🚀 Model API with Flask, BART Base & mBART

A Flask-based API service implementing BART Base and mBART (Multilingual BART) model for online article text summarization with customizable generation parameters. This API provides two different models for text summarization with extensive parameter customization options.

## 🌟 Features

- **Flask REST API** for text summarization
- **BART Model Integration** for test the quality betwen BART base and mBART
- **mBART Model Integration** for high-quality multilingual summarization
- **Customizable Generation Parameters** for different summarization styles
- **Dynamic Length Control** based on input text
- **Two Generation Modes** for different use cases
- **Evaluation Metrics** for summary quality assessment
- **Batch Processing** support for multiple summaries
- **Error Handling** with detailed error messages

## 🛠️ API Endpoints

### BART Model Endpoints

#### 1. Generate Summary (BART)
```http
POST /bart-summarize
```

**Request Body:**
```json
{
  "text": "Your long text to summarize",
  "params": {
    "length_penalty": 2.0,
    "num_beams": 4,
    "no_repeat_ngram_size": 2,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9
  }
}
```
> Note: `params` object is optional. Default values will be used if not provided.

**Response:**
```json
{
  "summary": "Generated summary text"
}
```

#### 2. Evaluate Summary (BART)
```http
POST /bart-evaluate
```

**Request Body:**
```json
{
  "reference_text": "Original reference summary",
  "generated_summary": "Generated summary to evaluate"
}
```

**Response:**
```json
{
  "metrics": {
    "precision": 85.5,
    "recall": 78.3,
    "f1": 81.7,
    "bleu": 72.4
  }
}
```

### mBART Model Endpoints

#### 1. Generate Summary (mBART)
```http
POST /mbart-summarize
```

**Request Body:**
```json
{
  "text": "Your long text to summarize",
  "params": {
    "num_beams": 4,
    "length_penalty": 1.5,
    "num_beam_groups": 4,
    "diversity_penalty": 0.5
  }
}
```

#### 2. Evaluate Summary (mBART)
```http
POST /mbart-evaluate
```

**Request Body:** Same as BART evaluate endpoint

### Batch Processing Endpoints

#### 1. Batch Evaluation (BART)
```http
POST /bart-evaluate-batch
```

**Request Body:**
```json
{
  "reference_texts": ["reference1", "reference2", "..."],
  "generated_summaries": ["summary1", "summary2", "..."]
}
```

#### 2. Batch Evaluation (mBART)
```http
POST /mbart-evaluate-batch
```

## 🛠️ Generation Parameters

### Basic Parameters

| Parameter | Description | Default Value | Valid Range |
|-----------|-------------|---------------|-------------|
| `min_length` | Minimum length of summary | Dynamic (1/4 of input) | 5-15 words |
| `max_length` | Maximum length of summary | Dynamic (1/2 of input) | 30-150 words |
| `num_beams` | Number of beams for search | 4 | 1-10 |
| `early_stopping` | Stop when valid output found | True | boolean |
| `temperature` | Randomness in generation | 0.6-0.8 | 0.0-1.0 |
| `top_k` | Top K sampling parameter | 50 | 1-100 |
| `top_p` | Nucleus sampling parameter | 0.9 | 0.0-1.0 |
| `repetition_penalty` | Penalty for repeating words | 1.5-2.5 | 1.0-3.0 |

### Advanced Parameters

#### BART Specific
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `length_penalty` | Controls summary length | 2.0 |
| `no_repeat_ngram_size` | Prevents repetition | 2 |
| `do_sample` | Enable sampling | True |

#### mBART Specific
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `num_beam_groups` | Groups for diverse beams | 4 |
| `diversity_penalty` | Penalty for similar beams | 0.5 |
| `forced_bos_token_id` | Beginning token ID | None |

## 🎮 Usage Examples

### Basic API Call
```python
import requests

url = "http://your-api-endpoint/bart-summarize"  # or mbart-summarize
text = "Your long text here..."

response = requests.post(url, json={
    "text": text
})

print(response.json()["summary"])
```

### Advanced Parameters
```python
response = requests.post(url, json={
    "text": text,
    "params": {
        "length_penalty": 2.0,
        "num_beams": 6,
        "no_repeat_ngram_size": 3,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 2.5
    }
})
```

### Evaluation Example
```python
eval_url = "http://your-api-endpoint/bart-evaluate"
eval_response = requests.post(eval_url, json={
    "reference_text": "Original summary",
    "generated_summary": "Generated summary"
})

print(eval_response.json()["metrics"])
```

## 🔧 Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/model-api.git
cd model-api
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the Flask server
```bash
python app.py
```

## 📝 Notes

- The API automatically adjusts parameters based on input text length
- Two generation modes available for different use cases:
  - Deterministic mode for consistent results
  - Creative mode for more varied outputs
- Parameters can be fine-tuned based on specific needs
- Error handling includes detailed messages for troubleshooting
- Evaluation metrics provide quantitative quality assessment
- Batch processing available for multiple documents

## 🔍 Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400`: Bad Request (invalid parameters or input)
- `500`: Internal Server Error (model processing error)
- `404`: Endpoint Not Found

Example error response:
```json
{
  "error": "Text cannot be empty",
  "status": 400
}
```

## 🤝 Contributing

Feel free to open issues and pull requests for:
- Bug fixes
- New features
- Documentation improvements
- Parameter optimization suggestions
- Test cases and examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [BART Paper](https://arxiv.org/abs/1910.13461)
- [mBART Paper](https://arxiv.org/abs/2001.08210)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

---
Made with ❤️ for text summarization