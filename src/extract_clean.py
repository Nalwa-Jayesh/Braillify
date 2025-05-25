"""
Text Extraction and Cleaning Module for Braille ETL Pipeline
Handles OCR processing and text cleaning for both scanned images and digital text
"""

import os
import json
import re
import logging
from typing import List, Dict, Optional
import unicodedata
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextExtractor:
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Text cleaning patterns
        self.cleaning_patterns = {
            'multiple_spaces': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s\.\,\!\?\;\:\-\'\"\(\)\[\]]'),
            'extra_newlines': re.compile(r'\n{3,}'),
            'leading_trailing': re.compile(r'^\s+|\s+$'),
            'quotes': re.compile(r'[""''`]'),
            'dashes': re.compile(r'[—–]'),
        }
        
        # Language-specific patterns
        self.hindi_pattern = re.compile(r'[\u0900-\u097F]+')
        self.english_pattern = re.compile(r'[a-zA-Z]+')

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR results
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None

    def extract_text_from_image(self, image_path: str, language: str = 'eng') -> Optional[str]:
        """
        Extract text from image using OCR
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            if processed_img is None:
                return None
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6'
            if language == 'hindi':
                ocr_lang = 'hin+eng'
            else:
                ocr_lang = 'eng'
            
            # Perform OCR
            text = pytesseract.image_to_string(
                processed_img, 
                lang=ocr_lang, 
                config=custom_config
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return None

    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily Hindi or English
        """
        hindi_chars = len(self.hindi_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        
        if hindi_chars > english_chars:
            return 'hindi'
        else:
            return 'english'

    def clean_text(self, text: str, language: str = 'english') -> str:
        """
        Clean and normalize text for training
        """
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Language-specific cleaning
        if language == 'hindi':
            text = self._clean_hindi_text(text)
        else:
            text = self._clean_english_text(text)
        
        # Common cleaning
        text = self.cleaning_patterns['multiple_spaces'].sub(' ', text)
        text = self.cleaning_patterns['extra_newlines'].sub('\n\n', text)
        text = text.strip()
        
        return text

    def _clean_english_text(self, text: str) -> str:
        """
        Clean English text specifically
        """
        # Normalize quotes
        text = self.cleaning_patterns['quotes'].sub('"', text)
        
        # Normalize dashes
        text = self.cleaning_patterns['dashes'].sub('-', text)
        
        # Remove excessive special characters but keep basic punctuation
        # This pattern preserves common punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"\(\)\[\]]+', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'\bl\b', 'I', text)  # Single 'l' often should be 'I'
        text = re.sub(r'\bm\b', 'in', text)  # Single 'm' often should be 'in'
        
        return text

    def _clean_hindi_text(self, text: str) -> str:
        """
        Clean Hindi text specifically
        """
        # Remove unwanted characters but preserve Devanagari
        # Keep basic punctuation and Devanagari characters
        text = re.sub(r'[^\u0900-\u097F\w\s\.\,\!\?\;\:\-\'\"\(\)\[\]]+', ' ', text)
        
        # Fix spacing around Devanagari characters
        text = re.sub(r'([।])\s*', r'\1 ', text)  # Devanagari sentence separator
        
        return text

    def validate_extraction(self, text: str, min_words: int = 3) -> bool:
        """
        Validate if extracted text is meaningful
        """
        if not text or len(text.strip()) < 10:
            return False
        
        words = text.split()
        if len(words) < min_words:
            return False
        
        # Check if text has reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.3:  # At least 30% alphabetic characters
            return False
        
        return True

    def process_collected_data(self) -> None:
        """
        Process all collected data (both text files and images)
        """
        logger.info("Starting text extraction and cleaning...")
        
        # Load collected data
        collected_file = os.path.join(self.input_dir, 'collected_data.json')
        if not os.path.exists(collected_file):
            logger.error(f"Collected data file not found: {collected_file}")
            return
        
        with open(collected_file, 'r', encoding='utf-8') as f:
            collected_data = json.load(f)
        
        processed_data = []
        
        for item in collected_data:
            try:
                text = item.get('text', '')
                
                # If no text, try to read from individual file
                if not text:
                    text_file = os.path.join(self.input_dir, f"{item['id']}.txt")
                    if os.path.exists(text_file):
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text = f.read()
                
                # Detect language if not specified
                language = item.get('language', self.detect_language(text))
                
                # Clean the text
                cleaned_text = self.clean_text(text, language)
                
                # Validate extraction
                if self.validate_extraction(cleaned_text):
                    processed_item = {
                        'id': item['id'],
                        'original_text': text,
                        'cleaned_text': cleaned_text,
                        'language': language,
                        'word_count': len(cleaned_text.split()),
                        'char_count': len(cleaned_text),
                        'source': item.get('source', 'unknown'),
                        'type': item.get('type', 'text'),
                        'extraction_method': 'direct_text',
                        'quality_score': self._calculate_quality_score(cleaned_text),
                        'processing_notes': []
                    }
                    
                    processed_data.append(processed_item)
                    logger.info(f"Processed {item['id']} ({language})")
                else:
                    logger.warning(f"Skipped {item['id']} - failed validation")
                    
            except Exception as e:
                logger.error(f"Error processing {item.get('id', 'unknown')}: {e}")
        
        # Process any image files in the raw directory
        self._process_image_files(processed_data)
        
        # Save processed data
        output_file = os.path.join(self.output_dir, 'extracted_cleaned.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # Save individual cleaned text files
        for item in processed_data:
            filename = f"{item['id']}_cleaned.txt"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(item['cleaned_text'])
        
        logger.info(f"Processed {len(processed_data)} items")
        logger.info(f"Results saved to {output_file}")
        
        self._print_processing_summary(processed_data)

    def _process_image_files(self, processed_data: List[Dict]) -> None:
        """
        Process any image files found in the input directory
        """
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
        for filename in os.listdir(self.input_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(self.input_dir, filename)
                logger.info(f"Processing image: {filename}")
                
                # Extract text from image
                extracted_text = self.extract_text_from_image(image_path)
                
                if extracted_text and self.validate_extraction(extracted_text):
                    language = self.detect_language(extracted_text)
                    cleaned_text = self.clean_text(extracted_text, language)
                    
                    if self.validate_extraction(cleaned_text):
                        item_id = f"ocr_{filename.split('.')[0]}"
                        processed_item = {
                            'id': item_id,
                            'original_text': extracted_text,
                            'cleaned_text': cleaned_text,
                            'language': language,
                            'word_count': len(cleaned_text.split()),
                            'char_count': len(cleaned_text),
                            'source': filename,
                            'type': 'ocr_image',
                            'extraction_method': 'tesseract_ocr',
                            'quality_score': self._calculate_quality_score(cleaned_text),
                            'processing_notes': ['extracted_from_image']
                        }
                        
                        processed_data.append(processed_item)
                        logger.info(f"Successfully processed image: {filename}")

    def _calculate_quality_score(self, text: str) -> float:
        """
        Calculate a quality score for the extracted text
        """
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length score (normalized)
        length_score = min(len(text) / 500, 1.0) * 0.3
        score += length_score
        
        # Word count score
        words = text.split()
        word_score = min(len(words) / 50, 1.0) * 0.3
        score += word_score
        
        # Character diversity score
        unique_chars = len(set(text.lower()))
        char_score = min(unique_chars / 30, 1.0) * 0.2
        score += char_score
        
        # Sentence structure score
        sentences = re.split(r'[.!?]+', text)
        sentence_score = min(len(sentences) / 10, 1.0) * 0.2
        score += sentence_score
        
        return round(score, 2)

    def _print_processing_summary(self, processed_data: List[Dict]) -> None:
        """
        Print summary of processing results
        """
        if not processed_data:
            print("No data was successfully processed.")
            return
        
        english_count = sum(1 for item in processed_data if item['language'] == 'english')
        hindi_count = sum(1 for item in processed_data if item['language'] == 'hindi')
        total_words = sum(item['word_count'] for item in processed_data)
        avg_quality = sum(item['quality_score'] for item in processed_data) / len(processed_data)
        
        print(f"\nProcessing Summary:")
        print(f"Total processed items: {len(processed_data)}")
        print(f"English items: {english_count}")
        print(f"Hindi items: {hindi_count}")
        print(f"Total words: {total_words}")
        print(f"Average quality score: {avg_quality:.2f}")
        
        # Quality distribution
        high_quality = sum(1 for item in processed_data if item['quality_score'] >= 0.7)
        medium_quality = sum(1 for item in processed_data if 0.4 <= item['quality_score'] < 0.7)
        low_quality = sum(1 for item in processed_data if item['quality_score'] < 0.4)
        
        print(f"Quality distribution:")
        print(f"  High quality (≥0.7): {high_quality}")
        print(f"  Medium quality (0.4-0.7): {medium_quality}")
        print(f"  Low quality (<0.4): {low_quality}")

def main():
    extractor = TextExtractor()
    extractor.process_collected_data()

if __name__ == "__main__":
    main()