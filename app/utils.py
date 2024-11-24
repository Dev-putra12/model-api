from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ----------------------------------------------bart section----------------------------------------------
class BartSummaryGenerator:
    def __init__(self):
        self.model: Optional[AutoModelForSeq2SeqLM] = None  # Ubah dari bart_model
        self.tokenizer: Optional[AutoTokenizer] = None  # Ubah dari bart_tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Ubah dari bart_device
        logger.info(f"Using device for BART: {self.device}")

    def load_model(self, model_path: str) -> bool:  # Ubah dari load_bart_model
        """
        Load the BART model and tokenizer from the specified path
        
        Args:
            model_path (str): Path to the BART model directory
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            logger.info(f"Loading BART model from: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            
            # Move model to device if not using device_map="auto"
            if self.model.device.type != self.device.type:
                self.model.to(self.device)
            
            logger.info("BART model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading BART model: {str(e)}", exc_info=True)
            return False

    def generate_summary(self, text: str) -> Optional[str]:  # Ubah dari generate_bart_summary
        """
        Generate summary from input text with dynamic parameters using BART
        
        Args:
            text (str): Input text to summarize
            
        Returns:
            Optional[str]: Generated summary or None if error occurs
        """
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("BART model or tokenizer not initialized")

            # Preprocessing text
            text = text.strip()
            
            # Count words in input text
            word_count = len(text.split())
            
            # Calculate dynamic parameters based on input length
            min_length = min(15, max(5, word_count // 4))
            max_length = min(150, max(30, word_count // 2))
            
            # Adjust repetition penalty based on text length
            repetition_penalty = 2.5 if word_count > 100 else 1.5
            
            # Adjust temperature based on text complexity
            temperature = 0.8 if word_count > 100 else 0.6
            
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                max_length=512,
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
                    min_length=min_length,
                    max_length=max_length,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=2.0,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=repetition_penalty
                )

            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Post-processing
            summary = summary.strip()
            
            # Validasi hasil
            if len(summary.split()) < min_length:
                logger.warning(f"Generated BART summary too short: {len(summary.split())} words")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating BART summary: {str(e)}", exc_info=True)
            return None


# ----------------------------------------------mbart section----------------------------------------------
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
        Generate summary from input text with dynamic parameters
        
        Args:
            text (str): Input text to summarize
            
        Returns:
            Optional[str]: Generated summary or None if error occurs
        """
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("Model or tokenizer not initialized")

            # Preprocessing text
            text = text.strip()
            
            # Count words in input text
            word_count = len(text.split())
            
            # Calculate dynamic parameters based on input length
            min_length = min(15, max(5, word_count // 4))  # Minimal 5 kata, maksimal 15 kata
            max_length = min(150, max(30, word_count // 2))  # Minimal 30 kata, maksimal 150 kata
            
            # Adjust repetition penalty based on text length
            repetition_penalty = 2.5 if word_count > 100 else 1.5
            
            # Adjust temperature based on text complexity
            temperature = 0.8 if word_count > 100 else 0.6
            
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
                # Opsi 1: Menggunakan beam search dengan diversity
                # summary_ids = self.model.generate(
                #     inputs["input_ids"],
                #     num_beams=4,
                #     min_length=min_length,
                #     max_length=max_length,
                #     early_stopping=True,
                #     no_repeat_ngram_size=3,
                #     length_penalty=1.5,
                #     num_beam_groups=4,
                #     diversity_penalty=0.5,
                #     do_sample=False  # Harus False ketika menggunakan diversity_penalty
                # )

                # # Opsi 2: Menggunakan sampling (uncommment jika ingin menggunakan ini)
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    min_length=min_length,
                    max_length=max_length,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=2.0,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=repetition_penalty
                )

            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Post-processing
            summary = summary.strip()
            
            # Validasi hasil
            if len(summary.split()) < min_length:
                logger.warning(f"Generated summary too short: {len(summary.split())} words")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return None

