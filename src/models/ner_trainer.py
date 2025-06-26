"""
NER Model Training Module for Amharic Text
Trains transformer-based models for Named Entity Recognition
"""

import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from torch.utils.data import Dataset
from loguru import logger


@dataclass
class NERConfig:
    """Configuration for NER training"""
    model_name: str = "distilbert-base-multilingual-cased"  # Lighter model for memory efficiency
    max_length: int = 64  # Reduced sequence length
    learning_rate: float = 3e-5
    num_epochs: int = 2  # Reduced epochs
    batch_size: int = 4  # Much smaller batch size
    warmup_steps: int = 100  # Reduced warmup
    weight_decay: float = 0.01
    output_dir: str = "models/ner_model"
    gradient_accumulation_steps: int = 4  # Simulate larger batch size
    

class NERDataset(Dataset):
    """Dataset class for NER training"""
    
    def __init__(self, texts: List[List[str]], labels: List[List[str]], 
                 tokenizer, label_to_id: Dict[str, int], max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize and align labels
        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels with tokenized input
        aligned_labels = self._align_labels(labels, encoding.word_ids())
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }
    
    def _align_labels(self, labels: List[str], word_ids: List[Optional[int]]) -> List[int]:
        """Align labels with tokenized input"""
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss calculation)
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word gets the label
                if word_idx < len(labels):
                    aligned_labels.append(self.label_to_id[labels[word_idx]])
                else:
                    aligned_labels.append(self.label_to_id['O'])
            else:
                # Subsequent tokens of the same word get -100
                aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        return aligned_labels


class NERTrainer:
    """Trainer for Amharic NER models"""
    
    def __init__(self, config: NERConfig = None):
        self.config = config or NERConfig()
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        
        # Setup logging
        logger.add("logs/ner_training.log", rotation="1 day")
    
    def load_data(self, data_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """Load labeled data from JSON file"""
        logger.info(f"Loading data from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for item in data:
            if 'tokens' in item and 'labels' in item:
                texts.append(item['tokens'])
                labels.append(item['labels'])
        
        logger.info(f"Loaded {len(texts)} samples")
        return texts, labels
    
    def prepare_labels(self, all_labels: List[List[str]]) -> None:
        """Create label mappings"""
        unique_labels = set()
        for label_list in all_labels:
            unique_labels.update(label_list)
        
        # Sort labels to ensure consistent mapping
        sorted_labels = sorted(list(unique_labels))
        
        self.label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        logger.info(f"Found {len(unique_labels)} unique labels: {sorted_labels}")
    
    def initialize_model(self) -> None:
        """Initialize tokenizer and model"""
        logger.info(f"Initializing model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.label_to_id),
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )
    
    def train(self, data_path: str, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the NER model"""
        logger.info("Starting NER model training")
        
        # Load and prepare data
        texts, labels = self.load_data(data_path)
        self.prepare_labels(labels)
        self.initialize_model()
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        
        # Create datasets
        train_dataset = NERDataset(
            train_texts, train_labels, self.tokenizer, 
            self.label_to_id, self.config.max_length
        )
        val_dataset = NERDataset(
            val_texts, val_labels, self.tokenizer, 
            self.label_to_id, self.config.max_length
        )
        
        # Setup training arguments with memory optimizations
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Memory optimizations
            dataloader_pin_memory=False,
            fp16=True,  # Use mixed precision
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, padding=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save label mappings
        label_mapping_path = Path(self.config.output_dir) / "label_mapping.json"
        with open(label_mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training completed. Model saved to {self.config.output_dir}")
        
        return {
            'train_loss': train_result.training_loss,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'num_labels': len(self.label_to_id),
            'model_path': self.config.output_dir
        }
    
    def load_trained_model(self, model_path: str) -> None:
        """Load a trained model"""
        logger.info(f"Loading trained model from {model_path}")
        
        # Load label mappings
        label_mapping_path = Path(model_path) / "label_mapping.json"
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            self.label_to_id = mappings['label_to_id']
            self.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        logger.info("Model loaded successfully")
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        """Predict entities in text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_trained_model() first.")
        
        # Tokenize text
        tokens = text.split()
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Align predictions with original tokens
        word_ids = encoding.word_ids()
        predicted_labels = []
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and i < len(predictions[0]):
                label_id = predictions[0][i].item()
                if label_id in self.id_to_label:
                    predicted_labels.append(self.id_to_label[label_id])
                else:
                    predicted_labels.append('O')
        
        # Combine tokens with predictions
        result = []
        for i, token in enumerate(tokens):
            if i < len(predicted_labels):
                result.append((token, predicted_labels[i]))
            else:
                result.append((token, 'O'))
        
        return result


def main():
    """Main function for testing the trainer"""
    config = NERConfig(
        num_epochs=1,  # Quick test
        batch_size=2,  # Very small for testing
        max_length=32  # Very short sequences
    )
    
    trainer = NERTrainer(config)
    
    # Train model
    data_path = "data/labeled/amharic_ner_sample_50_messages.json"
    if Path(data_path).exists():
        results = trainer.train(data_path)
        logger.info(f"Training results: {results}")
        
        # Test prediction
        test_text = "የሕፃናት ልብስ ዋጋ 500 ብር ነው"
        predictions = trainer.predict(test_text)
        logger.info(f"Test prediction: {predictions}")
    else:
        logger.error(f"Data file not found: {data_path}")


if __name__ == "__main__":
    main()
