"""
Amharic Text Preprocessing Module
Handles tokenization, normalization, and cleaning of Amharic text data
"""

import re
import string
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from loguru import logger
import nltk
from nltk.tokenize import word_tokenize

class AmharicTextProcessor:
    """Processor for Amharic text preprocessing and normalization"""
    
    def __init__(self):
        """Initialize the text processor"""
        self.setup_nltk()
        
        # Amharic Unicode ranges
        self.amharic_range = (0x1200, 0x137F)  # Ethiopic Unicode block
        
        # Common Amharic punctuation and symbols
        self.amharic_punctuation = '።፣፤፥፦፧፨'
        
        # Price patterns in Amharic
        self.price_patterns = [
            r'ዋጋ\s*\d+',  # ዋጋ followed by numbers
            r'\d+\s*ብር',  # numbers followed by ብር
            r'በ\s*\d+\s*ብር',  # በ X ብር pattern
            r'\d+\s*birr',  # English birr
            r'ETB\s*\d+',  # ETB currency
        ]
        
        # Location indicators in Amharic
        self.location_indicators = [
            'አዲስ አበባ', 'አዲስ', 'አበባ', 'ቦሌ', 'ፒያሳ', 'መርካቶ',
            'ካዛንቺስ', 'ጎፋ', 'ኮልፌ', 'ኪርኮስ', 'ላፍቶ'
        ]
        
        logger.info("AmharicTextProcessor initialized")
    
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def is_amharic_text(self, text: str) -> bool:
        """Check if text contains Amharic characters"""
        if not text:
            return False
        
        amharic_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                if self.amharic_range[0] <= ord(char) <= self.amharic_range[1]:
                    amharic_chars += 1
        
        if total_chars == 0:
            return False
        
        # Consider text as Amharic if more than 30% of alphabetic chars are Amharic
        return (amharic_chars / total_chars) > 0.3
    
    def normalize_amharic_text(self, text: str) -> str:
        """Normalize Amharic text by handling common variations"""
        if not text:
            return ""
        
        # Normalize common Amharic character variations
        normalizations = {
            'ሀ': 'ሃ',  # Normalize ha variations
            'ሁ': 'ሁ',
            'ሂ': 'ሂ',
            'ሄ': 'ሄ',
            'ህ': 'ህ',
            'ሆ': 'ሆ',
            # Add more normalizations as needed
        }
        
        normalized_text = text
        for old_char, new_char in normalizations.items():
            normalized_text = normalized_text.replace(old_char, new_char)
        
        return normalized_text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Normalize Amharic text
        text = self.normalize_amharic_text(text)
        
        return text.strip()
    
    def tokenize_amharic(self, text: str) -> List[str]:
        """Tokenize Amharic text"""
        if not text:
            return []
        
        # Clean text first
        text = self.clean_text(text)
        
        # Use NLTK word tokenizer as base
        tokens = word_tokenize(text)
        
        # Post-process tokens for Amharic-specific cases
        processed_tokens = []
        for token in tokens:
            # Skip empty tokens
            if not token.strip():
                continue
            
            # Handle Amharic punctuation
            if token in self.amharic_punctuation:
                processed_tokens.append(token)
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def extract_entities_hints(self, text: str) -> Dict[str, List[str]]:
        """Extract potential entity hints from text"""
        hints = {
            'potential_prices': [],
            'potential_locations': [],
            'potential_products': []
        }
        
        # Extract price patterns
        for pattern in self.price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            hints['potential_prices'].extend(matches)
        
        # Extract location hints
        for location in self.location_indicators:
            if location in text:
                hints['potential_locations'].append(location)
        
        # Simple product detection (words that might be products)
        tokens = self.tokenize_amharic(text)
        for token in tokens:
            # Skip very short tokens and common words
            if len(token) > 2 and self.is_amharic_text(token):
                # This is a simple heuristic - in practice, you'd use more sophisticated methods
                if not any(char.isdigit() for char in token):
                    hints['potential_products'].append(token)
        
        return hints
    
    def preprocess_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single message"""
        text = message_data.get('text', '')
        
        # Clean and tokenize text
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_amharic(cleaned_text)
        
        # Extract entity hints
        entity_hints = self.extract_entities_hints(cleaned_text)
        
        # Add preprocessing results
        processed_data = message_data.copy()
        processed_data.update({
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'token_count': len(tokens),
            'is_amharic': self.is_amharic_text(cleaned_text),
            'entity_hints': entity_hints,
            'has_price_indicators': len(entity_hints['potential_prices']) > 0,
            'has_location_indicators': len(entity_hints['potential_locations']) > 0,
        })
        
        return processed_data
    
    def preprocess_dataset(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Preprocess entire dataset"""
        logger.info(f"Starting preprocessing of {input_file}")
        
        # Load data
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.json'):
            df = pd.read_json(input_file)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # Preprocess each message
        processed_messages = []
        for idx, row in df.iterrows():
            processed_msg = self.preprocess_message(row.to_dict())
            processed_messages.append(processed_msg)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} messages")
        
        # Create processed DataFrame
        processed_df = pd.DataFrame(processed_messages)
        
        # Save processed data
        if output_file is None:
            output_file = input_file.replace('.csv', '_processed.csv').replace('.json', '_processed.json')
        
        if output_file.endswith('.csv'):
            processed_df.to_csv(output_file, index=False, encoding='utf-8')
        else:
            processed_df.to_json(output_file, orient='records', force_ascii=False, indent=2)
        
        logger.info(f"Preprocessing completed. Saved to {output_file}")
        return processed_df

def main():
    """Main function for testing the preprocessor"""
    processor = AmharicTextProcessor()
    
    # Test with sample text
    sample_text = "ሰላም! የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።"
    
    print("Original text:", sample_text)
    print("Cleaned text:", processor.clean_text(sample_text))
    print("Tokens:", processor.tokenize_amharic(sample_text))
    print("Entity hints:", processor.extract_entities_hints(sample_text))

if __name__ == "__main__":
    main()
