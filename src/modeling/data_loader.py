"""
Data loader for CoNLL format datasets
Handles loading, tokenization, and label alignment for NER training
"""

import json
import pandas as pd
from typing import List, Dict, Tuple, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset
from loguru import logger

class CoNLLDataLoader:
    """Data loader for CoNLL format NER datasets"""
    
    def __init__(self, tokenizer_name: str = "xlm-roberta-base"):
        """Initialize the data loader with tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label_to_id = {}
        self.id_to_label = {}
        self.max_length = 512
        
        logger.info(f"Initialized CoNLL data loader with tokenizer: {tokenizer_name}")
    
    def load_conll_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load CoNLL format file and parse into structured data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by empty lines to get individual messages
        messages = content.split('\n\n')
        parsed_data = []
        
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
                parsed_data.append({
                    'id': message_id or f'msg_{len(parsed_data)}',
                    'text': text or ' '.join(tokens),
                    'tokens': tokens,
                    'labels': labels
                })
        
        logger.info(f"Loaded {len(parsed_data)} messages from {file_path}")
        return parsed_data
    
    def create_label_mappings(self, data: List[Dict[str, Any]]):
        """Create label to ID mappings"""
        all_labels = set()
        for item in data:
            all_labels.update(item['labels'])
        
        # Sort labels to ensure consistent mapping
        sorted_labels = sorted(list(all_labels))
        
        self.label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        logger.info(f"Created label mappings for {len(sorted_labels)} labels: {sorted_labels}")
    
    def align_labels_with_tokens(self, tokens: List[str], labels: List[str], 
                                tokenized_inputs) -> List[int]:
        """Align original labels with tokenizer's subword tokens"""
        aligned_labels = []
        word_ids = tokenized_inputs.word_ids()
        
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss calculation)
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the original label
                if word_idx < len(labels):
                    aligned_labels.append(self.label_to_id[labels[word_idx]])
                else:
                    aligned_labels.append(-100)
            else:
                # Subsequent subwords get -100 (ignored)
                aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        return aligned_labels
    
    def tokenize_and_align_labels(self, data: List[Dict[str, Any]]) -> HFDataset:
        """Tokenize data and align labels for training"""
        tokenized_data = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'id': []
        }
        
        for item in data:
            tokens = item['tokens']
            labels = item['labels']
            
            # Join tokens to create text for tokenization
            text = ' '.join(tokens)
            
            # Tokenize
            tokenized_inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_offsets_mapping=False,
                is_split_into_words=False
            )
            
            # For proper alignment, we need to tokenize with is_split_into_words=True
            tokenized_inputs = self.tokenizer(
                tokens,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                is_split_into_words=True
            )
            
            # Align labels
            aligned_labels = self.align_labels_with_tokens(tokens, labels, tokenized_inputs)
            
            tokenized_data['input_ids'].append(tokenized_inputs['input_ids'])
            tokenized_data['attention_mask'].append(tokenized_inputs['attention_mask'])
            tokenized_data['labels'].append(aligned_labels)
            tokenized_data['id'].append(item['id'])
        
        # Convert to HuggingFace Dataset
        dataset = HFDataset.from_dict(tokenized_data)
        logger.info(f"Tokenized {len(dataset)} examples")
        
        return dataset
    
    def prepare_datasets(self, train_file: str, val_split: float = 0.2) -> Tuple[HFDataset, HFDataset]:
        """Prepare training and validation datasets"""
        # Load data
        data = self.load_conll_file(train_file)
        
        # Create label mappings
        self.create_label_mappings(data)
        
        # Split data
        split_idx = int(len(data) * (1 - val_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        logger.info(f"Split data: {len(train_data)} train, {len(val_data)} validation")
        
        # Tokenize and align labels
        train_dataset = self.tokenize_and_align_labels(train_data)
        val_dataset = self.tokenize_and_align_labels(val_data)
        
        return train_dataset, val_dataset
    
    def get_label_info(self) -> Dict[str, Any]:
        """Get label mapping information"""
        return {
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label,
            'num_labels': len(self.label_to_id)
        }

class NERDataset(Dataset):
    """PyTorch Dataset for NER data"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

def main():
    """Test the data loader"""
    loader = CoNLLDataLoader()
    
    # Test with sample data
    train_file = "data/labeled/amharic_ner_sample_50_messages.txt"
    if Path(train_file).exists():
        train_dataset, val_dataset = loader.prepare_datasets(train_file)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Label mappings: {loader.get_label_info()}")
    else:
        print(f"Training file not found: {train_file}")

if __name__ == "__main__":
    main()
