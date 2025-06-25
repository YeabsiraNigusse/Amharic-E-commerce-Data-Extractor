"""
NER Model Trainer for Amharic E-commerce Entity Extraction
Supports multiple transformer models including XLM-RoBERTa, mBERT, and others
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset
import evaluate
from loguru import logger

from .data_loader import CoNLLDataLoader

class AmharicNERTrainer:
    """Trainer for Amharic NER models"""
    
    def __init__(self, model_name: str = "xlm-roberta-base", output_dir: str = "models"):
        """Initialize the trainer with model configuration"""
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.data_loader = None
        self.trainer = None
        
        # Metrics
        self.seqeval = evaluate.load("seqeval")
        
        logger.info(f"Initialized NER trainer with model: {model_name}")
    
    def setup_model_and_tokenizer(self, num_labels: int, label_to_id: Dict[str, int]):
        """Setup model and tokenizer with label configuration"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model for token classification
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            label2id=label_to_id,
            id2label={v: k for k, v in label_to_id.items()},
            ignore_mismatched_sizes=True
        )
        
        logger.info(f"Setup model with {num_labels} labels")
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.model.config.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def train(self,
              train_dataset: Dataset,
              val_dataset: Dataset,
              learning_rate: float = 2e-5,
              num_epochs: int = 3,
              batch_size: int = 16,
              warmup_steps: int = 500,
              weight_decay: float = 0.01,
              save_strategy: str = "epoch",
              eval_strategy: str = "epoch",
              early_stopping_patience: int = 3):
        """Train the NER model"""
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / f"{self.model_name.replace('/', '_')}_finetuned"),
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            eval_strategy=eval_strategy,  # Updated parameter name
            save_strategy=save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            save_total_limit=2,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=False,  # Avoid memory issues
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        logger.info("Starting training...")
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save the model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        metrics_file = Path(training_args.output_dir) / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training completed. Model saved to {training_args.output_dir}")
        logger.info(f"Training metrics: {metrics}")
        
        return train_result
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model"""
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        eval_results = self.trainer.evaluate(eval_dataset)
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def predict(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Make predictions on new texts"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call setup_model_and_tokenizer() first.")
        
        predictions = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=2)
            
            # Convert to labels
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_labels = [self.model.config.id2label[id.item()] for id in predicted_ids[0]]
            
            # Create prediction results
            text_predictions = []
            for token, label in zip(tokens, predicted_labels):
                if token not in ['<s>', '</s>', '<pad>']:
                    text_predictions.append({
                        'token': token,
                        'label': label
                    })
            
            predictions.append(text_predictions)
        
        return predictions
    
    def save_model_info(self, label_info: Dict[str, Any], metrics: Dict[str, float]):
        """Save model information and metrics"""
        model_info = {
            'model_name': self.model_name,
            'num_labels': len(label_info['label_to_id']),
            'label_to_id': label_info['label_to_id'],
            'id_to_label': label_info['id_to_label'],
            'metrics': metrics,
            'training_completed': True
        }
        
        info_file = self.output_dir / f"{self.model_name.replace('/', '_')}_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model info saved to {info_file}")

def main():
    """Test the trainer"""
    # Initialize trainer
    trainer = AmharicNERTrainer("xlm-roberta-base")
    
    # Load data
    data_loader = CoNLLDataLoader("xlm-roberta-base")
    train_file = "data/labeled/amharic_ner_sample_50_messages.txt"
    
    if Path(train_file).exists():
        train_dataset, val_dataset = data_loader.prepare_datasets(train_file, val_split=0.2)
        label_info = data_loader.get_label_info()
        
        # Setup model
        trainer.setup_model_and_tokenizer(
            num_labels=label_info['num_labels'],
            label_to_id=label_info['label_to_id']
        )
        
        # Train model (with small parameters for testing)
        train_result = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=1,
            batch_size=8
        )
        
        # Evaluate
        eval_results = trainer.evaluate(val_dataset)
        
        # Save model info
        trainer.save_model_info(label_info, eval_results)
        
        print("Training completed successfully!")
    else:
        print(f"Training file not found: {train_file}")

if __name__ == "__main__":
    main()
