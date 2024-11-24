from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import torch
from typing import Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SummaryGenerator:
    def __init__(self):
        self.model: Optional[MBartForConditionalGeneration] = None
        self.tokenizer: Optional[MBart50TokenizerFast] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_model(self, model_path: str) -> bool:
        """
        Load the model and tokenizer from the specified path
        
        Args:
            model_path (str): Path to the model directory
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Load tokenizer
            self.tokenizer = MBart50TokenizerFast.from_pretrained(
                model_path,
                src_lang="id_ID",
                tgt_lang="id_ID",
                use_fast=True
            )
            
            # Load model
            self.model = MBartForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="auto"  # Automatically handle device placement
            )
            
            # Move model to device if not using device_map="auto"
            if self.model.device.type != self.device.type:
                self.model.to(self.device)
            
            logger.info("Model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False

    def generate_summary(self, text: str) -> Optional[str]:
        """
        Generate summary from input text
        
        Args:
            text (str): Input text to summarize
            
        Returns:
            Optional[str]: Generated summary or None if error occurs
        """
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("Model or tokenizer not initialized")

            # Tokenize input
            inputs = self.tokenizer(
                text, 
                max_length=1024, 
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    min_length=15,
                    max_length=150,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=2.0,
                    temperature=0.7
                )

            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return None
