"""
CoNLL Format Labeling System for Amharic NER
Handles manual and semi-automatic labeling of Amharic text for Named Entity Recognition
"""

import json
import re
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import pandas as pd

from loguru import logger

class CoNLLLabeler:
    """System for labeling Amharic text in CoNLL format for NER"""
    
    def __init__(self):
        """Initialize the CoNLL labeler"""
        self.entity_types = {
            'PRODUCT': ['B-PRODUCT', 'I-PRODUCT'],
            'PRICE': ['B-PRICE', 'I-PRICE'],
            'LOCATION': ['B-LOC', 'I-LOC'],
            'DELIVERY_FEE': ['B-DELIVERY_FEE', 'I-DELIVERY_FEE'],
            'CONTACT_INFO': ['B-CONTACT_INFO', 'I-CONTACT_INFO']
        }
        
        # Amharic price patterns for automatic detection
        self.price_patterns = [
            (r'ዋጋ\s*(\d+)', 'PRICE'),
            (r'(\d+)\s*ብር', 'PRICE'),
            (r'በ\s*(\d+)\s*ብር', 'PRICE'),
            (r'(\d+)\s*birr', 'PRICE'),
            (r'ETB\s*(\d+)', 'PRICE'),
            (r'(\d+)\s*ብር\s*ነው', 'PRICE'),
        ]
        
        # Amharic location patterns
        self.location_patterns = [
            (r'አዲስ\s*አበባ', 'LOCATION'),
            (r'ቦሌ', 'LOCATION'),
            (r'ፒያሳ', 'LOCATION'),
            (r'መርካቶ', 'LOCATION'),
            (r'ካዛንቺስ', 'LOCATION'),
            (r'ጎፋ', 'LOCATION'),
            (r'ኮልፌ', 'LOCATION'),
            (r'ኪርኮስ', 'LOCATION'),
            (r'ላፍቶ', 'LOCATION'),
        ]
        
        # Contact info patterns
        self.contact_patterns = [
            (r'09\d{8}', 'CONTACT_INFO'),  # Ethiopian phone numbers
            (r'@\w+', 'CONTACT_INFO'),     # Telegram usernames
            (r'\+251\d{9}', 'CONTACT_INFO'), # International format
        ]
        
        logger.info("CoNLL Labeler initialized")
    
    def tokenize_amharic_for_conll(self, text: str) -> List[str]:
        """Tokenize text specifically for CoNLL format"""
        if not text:
            return []
        
        # Simple tokenization - split by whitespace and punctuation
        # Keep punctuation as separate tokens
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in '።፣፤፥፦፧፨.,!?;:()[]{}':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return [token for token in tokens if token.strip()]
    
    def auto_label_entities(self, tokens: List[str]) -> List[str]:
        """Automatically label obvious entities using patterns"""
        labels = ['O'] * len(tokens)
        text = ' '.join(tokens)
        
        # Label prices
        for pattern, entity_type in self.price_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                start_pos = len(text[:match.start()].split())
                end_pos = start_pos + len(match.group().split())
                
                if start_pos < len(labels):
                    labels[start_pos] = f'B-{entity_type}'
                    for i in range(start_pos + 1, min(end_pos, len(labels))):
                        labels[i] = f'I-{entity_type}'
        
        # Label locations
        for pattern, entity_type in self.location_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                start_pos = len(text[:match.start()].split())
                end_pos = start_pos + len(match.group().split())
                
                if start_pos < len(labels):
                    labels[start_pos] = f'B-{entity_type}'
                    for i in range(start_pos + 1, min(end_pos, len(labels))):
                        labels[i] = f'I-{entity_type}'
        
        # Label contact info
        for pattern, entity_type in self.contact_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                start_pos = len(text[:match.start()].split())
                end_pos = start_pos + len(match.group().split())
                
                if start_pos < len(labels):
                    labels[start_pos] = f'B-{entity_type}'
                    for i in range(start_pos + 1, min(end_pos, len(labels))):
                        labels[i] = f'I-{entity_type}'
        
        return labels
    
    def create_conll_format(self, tokens: List[str], labels: List[str]) -> str:
        """Create CoNLL format string from tokens and labels"""
        if len(tokens) != len(labels):
            raise ValueError("Tokens and labels must have the same length")
        
        conll_lines = []
        for token, label in zip(tokens, labels):
            conll_lines.append(f"{token}\t{label}")
        
        return '\n'.join(conll_lines)
    
    def parse_conll_format(self, conll_text: str) -> Tuple[List[str], List[str]]:
        """Parse CoNLL format text into tokens and labels"""
        lines = conll_text.strip().split('\n')
        tokens = []
        labels = []
        
        for line in lines:
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    labels.append(parts[1])
        
        return tokens, labels
    
    def label_message(self, message_text: str, auto_label: bool = True) -> Dict[str, Any]:
        """Label a single message"""
        tokens = self.tokenize_amharic_for_conll(message_text)
        
        if auto_label:
            labels = self.auto_label_entities(tokens)
        else:
            labels = ['O'] * len(tokens)
        
        conll_format = self.create_conll_format(tokens, labels)
        
        return {
            'text': message_text,
            'tokens': tokens,
            'labels': labels,
            'conll_format': conll_format,
            'token_count': len(tokens)
        }
    
    def create_sample_labeled_data(self) -> List[Dict[str, Any]]:
        """Create sample labeled data for demonstration"""
        sample_messages = [
            "የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።",
            "ስልክ ቁጥር 0911234567 ላይ ይደውሉ። አዲስ አበባ ውስጥ ነን።",
            "የሴቶች ጫማ በ 800 ብር። ፒያሳ አካባቢ ይገኛል።",
            "ላፕቶፕ ዋጋ 25000 ብር። ዴሊቨሪ ነፃ ነው።",
            "የወንዶች ሸሚዝ 300 ብር። @ethioshop ላይ ይገናኙን።",
            "የቤት እቃዎች በመርካቶ። ዋጋ 1500 ብር ነው።",
            "ስማርት ፎን ETB 15000። ካዛንቺስ አካባቢ።",
            "የሴቶች ቦርሳ 450 ብር። ዴሊቨሪ ክፍያ 50 ብር።",
            "የመጽሐፍ መደብር በጎፋ። ዋጋ 200 ብር ነው።",
            "የኮምፒውተር አክሰሰሪዎች በ 1200 ብር። ኪርኮስ አካባቢ።"
        ]
        
        labeled_data = []
        for i, message in enumerate(sample_messages):
            labeled_msg = self.label_message(message, auto_label=True)
            labeled_msg['message_id'] = f"sample_{i+1}"
            labeled_data.append(labeled_msg)
        
        return labeled_data
    
    def save_conll_file(self, labeled_data: List[Dict[str, Any]], output_file: str):
        """Save labeled data in CoNLL format"""
        conll_content = []
        
        for data in labeled_data:
            # Add message separator comment
            conll_content.append(f"# Message ID: {data.get('message_id', 'unknown')}")
            conll_content.append(f"# Text: {data['text']}")
            conll_content.append("")
            
            # Add tokens and labels
            conll_content.append(data['conll_format'])
            conll_content.append("")  # Empty line between messages
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(conll_content))
        
        logger.info(f"CoNLL format data saved to {output_file}")
    
    def load_conll_file(self, input_file: str) -> List[Dict[str, Any]]:
        """Load CoNLL format file"""
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by empty lines to get individual messages
        messages = content.split('\n\n')
        labeled_data = []
        
        for message_block in messages:
            if not message_block.strip():
                continue
            
            lines = message_block.strip().split('\n')
            message_id = None
            text = None
            tokens = []
            labels = []
            
            for line in lines:
                if line.startswith('# Message ID:'):
                    message_id = line.replace('# Message ID:', '').strip()
                elif line.startswith('# Text:'):
                    text = line.replace('# Text:', '').strip()
                elif line.strip() and not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        tokens.append(parts[0])
                        labels.append(parts[1])
            
            if tokens and labels:
                labeled_data.append({
                    'message_id': message_id,
                    'text': text,
                    'tokens': tokens,
                    'labels': labels,
                    'token_count': len(tokens)
                })
        
        return labeled_data
    
    def validate_labels(self, labels: List[str]) -> List[str]:
        """Validate and fix label sequence"""
        valid_labels = []
        
        for i, label in enumerate(labels):
            if label == 'O':
                valid_labels.append(label)
            elif label.startswith('B-'):
                valid_labels.append(label)
            elif label.startswith('I-'):
                # Check if previous label is compatible
                if i > 0:
                    prev_label = valid_labels[-1]
                    entity_type = label[2:]  # Remove 'I-'
                    
                    if prev_label == f'B-{entity_type}' or prev_label == f'I-{entity_type}':
                        valid_labels.append(label)
                    else:
                        # Convert to B- if no compatible previous label
                        valid_labels.append(f'B-{entity_type}')
                else:
                    # Convert to B- if it's the first token
                    valid_labels.append(f'B-{label[2:]}')
            else:
                # Invalid label, mark as O
                valid_labels.append('O')
        
        return valid_labels
    
    def get_entity_statistics(self, labeled_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about labeled entities"""
        entity_counts = {}
        total_tokens = 0
        total_entities = 0
        
        for data in labeled_data:
            labels = data['labels']
            total_tokens += len(labels)
            
            for label in labels:
                if label != 'O':
                    entity_type = label[2:]  # Remove B- or I-
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                    if label.startswith('B-'):
                        total_entities += 1
        
        return {
            'total_messages': len(labeled_data),
            'total_tokens': total_tokens,
            'total_entities': total_entities,
            'entity_counts': entity_counts,
            'entity_coverage': total_entities / total_tokens if total_tokens > 0 else 0
        }

def main():
    """Main function for testing the labeler"""
    labeler = CoNLLLabeler()
    
    # Create sample labeled data
    logger.info("Creating sample labeled data...")
    sample_data = labeler.create_sample_labeled_data()
    
    # Save to CoNLL file
    output_file = "data/labeled/sample_conll_labeled.txt"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    labeler.save_conll_file(sample_data, output_file)
    
    # Get statistics
    stats = labeler.get_entity_statistics(sample_data)
    logger.info(f"Labeling statistics: {stats}")
    
    # Test loading
    loaded_data = labeler.load_conll_file(output_file)
    logger.info(f"Successfully loaded {len(loaded_data)} labeled messages")

if __name__ == "__main__":
    main()
