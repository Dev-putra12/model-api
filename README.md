# 🚀 Model API with Flask, BART Base & mBART

A Flask-based API service implementing BART Base and mBART (Multilingual BART) model for online article text summarization with customizable generation parameters.

## 🌟 Features

- **Flask REST API** for text summarization
- **BART Model Integration** for test the quality betwen BART base and mBART
- **mBART Model Integration** for high-quality multilingual summarization
- **Customizable Generation Parameters** for different summarization styles
- **Dynamic Length Control** based on input text
- **Two Generation Modes** for different use cases

## 🛠️ Generation Parameters

### Basic Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `min_length` | Minimum length of summary | Dynamic (1/4 of input) |
| `max_length` | Maximum length of summary | Dynamic (1/2 of input) |
| `num_beams` | Number of beams for search | 4 |
| `early_stopping` | Stop when valid output found | True |

### 🌎 BART Base Section
comming soon

### 🌎 mBART Section

### 🎯 Mode 1: Deterministic Beam Search
```python
# More deterministic but diverse outputs
summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=4,
    min_length=min_length,
    max_length=max_length,
    early_stopping=True,
    no_repeat_ngram_size=3,
    length_penalty=1.5,
    num_beam_groups=4,
    diversity_penalty=0.5,
    do_sample=False
)
```

### 🎨 Mode 2: Creative Sampling
```python
# More creative and varied outputs
summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=4,
    temperature=0.8,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    repetition_penalty=2.5
)
```

## ⚙️ Parameter Tuning Guide for mBART

### Length Control
```python
# For longer summaries
length_penalty = 2.0  # > 1.0

# For shorter summaries
length_penalty = 0.8  # < 1.0
```

### Diversity Control
```python
# For more diverse outputs
num_beam_groups = 4
diversity_penalty = 0.5

# For more focused outputs
num_beams = 6
no_repeat_ngram_size = 2
```

## 🎮 Usage Examples

### Basic API Call
```python
import requests

url = "http://your-api-endpoint/summarize"
text = "Your long text here..."

response = requests.post(url, json={
    "text": text
})

print(response.json()["summary"])
```

### Customizing Parameters
```python
response = requests.post(url, json={
    "text": text,
    "params": {
        "length_penalty": 2.0,
        "num_beams": 6,
        "no_repeat_ngram_size": 3
    }
})
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

## 🤝 Contributing

Feel free to open issues and pull requests for:
- Bug fixes
- New features
- Documentation improvements
- Parameter optimization suggestions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Made with ❤️ for text summarization
