from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import torch

class SummaryGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_input_length = 1024
        self.max_target_length = 128
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self, model_path):
        """Load model and tokenizer"""
        try:
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
            self.model = MBartForConditionalGeneration.from_pretrained(model_path)
            
            # Set language
            self.tokenizer.src_lang = "id_ID"
            self.tokenizer.tgt_lang = "id_ID"
            
            # Move model to GPU if available
            self.model = self.model.to(self.device)
            
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def generate_summary(self, text):
        """Generate summary from input text"""
        try:
            # Tokenize
            inputs = self.tokenizer(text, 
                                  return_tensors="pt", 
                                  max_length=self.max_input_length, 
                                  truncation=True)
            
            # Move inputs to GPU if available
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=4,
                min_length=20,
                max_length=self.max_target_length,
                early_stopping=True,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["id_ID"]
            )

            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return summary
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return None
