from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Optional
import logging
# matriks evaluasi
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords
from typing import Dict, List, Tuple
import numpy as np
import nltk
import re
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Di utils.py, setelah import nltk
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Tambahkan ini
    nltk.download('stopwords')
    nltk.download('indonesian')
except:
    pass



# ----------------------------------------------evaluator section----------------------------------------------
class SummaryEvaluator:
    def __init__(self):
        try:
            self.stopwords = set(stopwords.words('indonesian'))
        except:
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('indonesian'))
            
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text dengan penanganan yang lebih baik
        """
        # Lowercase the text
        text = text.lower()
        
        # Standardisasi whitespace
        text = ' '.join(text.split())
        
        # Hapus tanda baca tapi pertahankan beberapa karakter penting
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Hapus angka tapi pertahankan angka yang merupakan bagian dari kata
        text = re.sub(r'\b\d+\b', ' ', text)
        
        # Tokenisasi dengan mempertimbangkan kata majemuk
        words = word_tokenize(text)
        
        # Filter stopwords dan token kosong dengan lebih selektif
        words = [word for word in words if word and word not in self.stopwords and len(word) > 1]
        
        return words


    def calculate_precision(self, reference_tokens: List[str], generated_tokens: List[str]) -> float:
        """
        Hitung precision score dengan mempertimbangkan frekuensi kata
        """
        if not generated_tokens:
            return 0.0
        
        ref_counter = Counter(reference_tokens)
        gen_counter = Counter(generated_tokens)
        
        # Hitung overlap dengan mempertimbangkan frekuensi
        overlap = sum((ref_counter & gen_counter).values())
        total = sum(gen_counter.values())
        
        return overlap / total if total > 0 else 0.0

    def calculate_recall(self, reference_tokens: List[str], generated_tokens: List[str]) -> float:
        """
        Hitung recall score dengan mempertimbangkan frekuensi kata
        """
        if not reference_tokens:
            return 0.0
        
        ref_counter = Counter(reference_tokens)
        gen_counter = Counter(generated_tokens)
        
        # Hitung overlap dengan mempertimbangkan frekuensi
        overlap = sum((ref_counter & gen_counter).values())
        total = sum(ref_counter.values())
        
        return overlap / total if total > 0 else 0.0

    def calculate_bleu(self, reference_tokens: List[str], generated_tokens: List[str]) -> float:
        """
        Hitung BLEU score dengan smoothing yang lebih baik
        """
        from nltk.translate.bleu_score import SmoothingFunction
        
        if not reference_tokens or not generated_tokens:
            return 0.0
            
        # Gunakan method7 (Average of methods 1-4) untuk smoothing yang lebih baik
        smoothing = SmoothingFunction().method7
        
        # Gunakan weights yang lebih seimbang untuk n-gram yang lebih pendek
        weights = (0.4, 0.3, 0.2, 0.1)  # Memberikan bobot lebih tinggi pada 1-gram dan 2-gram
        
        try:
            return sentence_bleu(
                [reference_tokens],
                generated_tokens,
                weights=weights,
                smoothing_function=smoothing
            )
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {str(e)}")
            return 0.0


    def evaluate_summary(self, reference_text: str, generated_summary: str) -> Dict[str, float]:
        """
        Evaluasi ringkasan dengan penanganan kasus khusus
        """
        try:
            # Validasi input
            if not reference_text or not generated_summary:
                return {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'bleu': 0.0
                }

            # Preprocess kedua teks
            reference_tokens = self.preprocess_text(reference_text)
            generated_tokens = self.preprocess_text(generated_summary)

            # Penanganan kasus teks terlalu pendek
            if len(reference_tokens) < 4 or len(generated_tokens) < 4:
                logger.warning("Text too short for reliable evaluation")
                
            # Hitung metrik
            precision = self.calculate_precision(reference_tokens, generated_tokens)
            recall = self.calculate_recall(reference_tokens, generated_tokens)
            
            # Hitung F1 score dengan penanganan division by zero
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            # Hitung BLEU score
            bleu = self.calculate_bleu(reference_tokens, generated_tokens)

            # Format scores
            metrics = {
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1': round(f1 * 100, 2),
                'bleu': round(bleu * 100, 2)
            }

            return metrics

        except Exception as e:
            logger.error(f"Error in evaluate_summary: {str(e)}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'bleu': 0.0
            }


    def evaluate_multiple_summaries(self, reference_texts: List[str], generated_summaries: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple summaries and provide average and individual scores
        
        Args:
            reference_texts (List[str]): List of reference summaries
            generated_summaries (List[str]): List of generated summaries
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing average and individual scores
        """
        try:
            if len(reference_texts) != len(generated_summaries):
                raise ValueError("Number of reference and generated summaries must match")

            individual_scores = []
            for ref, gen in zip(reference_texts, generated_summaries):
                scores = self.evaluate_summary(ref, gen)
                individual_scores.append(scores)

            # Calculate average scores
            avg_scores = {
                'precision': round(np.mean([s['precision'] for s in individual_scores]), 2),
                'recall': round(np.mean([s['recall'] for s in individual_scores]), 2),
                'f1': round(np.mean([s['f1'] for s in individual_scores]), 2),
                'bleu': round(np.mean([s['bleu'] for s in individual_scores]), 2)
            }

            return {
                'average_scores': avg_scores,
                'individual_scores': individual_scores
            }

        except Exception as e:
            logger.error(f"Error in evaluate_multiple_summaries: {str(e)}", exc_info=True)
            return {
                'average_scores': {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'bleu': 0.0
                },
                'individual_scores': []
            }
# ----------------------------------------------bart section----------------------------------------------
class BartSummaryGenerator:
    def __init__(self):
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for BART: {self.device}")
        self.evaluator = SummaryEvaluator()

    def load_model(self, model_path: str) -> bool:
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

    def generate_summary(self, text: str) -> Optional[str]: 
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

    def evaluate_summary(self, reference_text: str, generated_summary: str) -> Dict[str, float]:
            return self.evaluator.evaluate_summary(reference_text, generated_summary)

# ----------------------------------------------mbart section----------------------------------------------
class SummaryGenerator:
    def __init__(self):
        self.model: Optional[MBartForConditionalGeneration] = None
        self.tokenizer: Optional[MBart50TokenizerFast] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.evaluator = SummaryEvaluator()

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

    def evaluate_summary(self, reference_text: str, generated_summary: str) -> Dict[str, float]:
        return self.evaluator.evaluate_summary(reference_text, generated_summary)