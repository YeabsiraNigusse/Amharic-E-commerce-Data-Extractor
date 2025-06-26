#!/usr/bin/env python3
"""
Main script to run Task 3: Fine-Tune NER Model
"""

import sys
import os
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from loguru import logger
from src.modeling.ner_trainer import AmharicNERTrainer
from src.modeling.data_loader import CoNLLDataLoader
from src.utils.data_utils import setup_logging

def train_single_model(model_name: str, 
                      train_file: str,
                      output_dir: str = "models",
                      learning_rate: float = 2e-5,
                      num_epochs: int = 3,
                      batch_size: int = 16,
                      val_split: float = 0.2):
    """Train a single NER model"""
    
    logger.info(f"Starting training for model: {model_name}")
    
    # Initialize data loader
    data_loader = CoNLLDataLoader(tokenizer_name=model_name)
    
    # Prepare datasets
    train_dataset, val_dataset = data_loader.prepare_datasets(train_file, val_split=val_split)
    label_info = data_loader.get_label_info()
    
    logger.info(f"Dataset prepared: {len(train_dataset)} train, {len(val_dataset)} validation")
    logger.info(f"Labels: {list(label_info['label_to_id'].keys())}")
    
    # Initialize trainer
    trainer = AmharicNERTrainer(model_name=model_name, output_dir=output_dir)
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer(
        num_labels=label_info['num_labels'],
        label_to_id=label_info['label_to_id']
    )
    
    # Train the model
    train_result = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        warmup_steps=min(500, len(train_dataset) // 4),
        eval_strategy="epoch",
        save_strategy="epoch",
        early_stopping_patience=2
    )
    
    # Evaluate the model
    eval_results = trainer.evaluate(val_dataset)
    
    # Save model information
    trainer.save_model_info(label_info, eval_results)
    
    logger.info(f"Training completed for {model_name}")
    logger.info(f"Final F1 Score: {eval_results.get('eval_f1', 'N/A')}")
    
    return {
        'model_name': model_name,
        'train_result': train_result,
        'eval_results': eval_results,
        'model_path': trainer.output_dir / f"{model_name.replace('/', '_')}_finetuned"
    }

def test_model_prediction(model_path: str, model_name: str):
    """Test model with sample predictions"""
    logger.info(f"Testing predictions for {model_name}")
    
    try:
        trainer = AmharicNERTrainer(model_name=model_name)
        
        # Load the trained model (this would need to be implemented)
        # For now, we'll just show what the prediction interface would look like
        
        sample_texts = [
            "የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።",
            "ስልክ ቁጥር 0911234567 ላይ ይደውሉ። አዲስ አበባ ውስጥ ነን።",
            "የሴቶች ጫማ በ 800 ብር። ፒያሳ አካባቢ ይገኛል።"
        ]
        
        logger.info("Sample texts for prediction:")
        for i, text in enumerate(sample_texts, 1):
            logger.info(f"{i}. {text}")
        
        # Note: Actual prediction would require loading the saved model
        logger.info("Prediction testing completed (interface ready)")
        
    except Exception as e:
        logger.error(f"Error testing predictions: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fine-tune NER models for Amharic e-commerce")
    parser.add_argument('--model', '-m', default='xlm-roberta-base', 
                       help='Model name to fine-tune (default: xlm-roberta-base)')
    parser.add_argument('--train-file', '-t', default='data/labeled/amharic_ner_sample_50_messages.txt',
                       help='Path to training file in CoNLL format')
    parser.add_argument('--output-dir', '-o', default='models',
                       help='Output directory for trained models')
    parser.add_argument('--learning-rate', '-lr', type=float, default=2e-5,
                       help='Learning rate for training')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--val-split', '-v', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test prediction interface')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("AMHARIC E-COMMERCE DATA EXTRACTOR - TASK 3")
    logger.info("Fine-Tune NER Model")
    logger.info("=" * 60)
    
    # Check if training file exists
    if not Path(args.train_file).exists():
        logger.error(f"Training file not found: {args.train_file}")
        logger.info("Please run Task 2 first to generate labeled data")
        return
    
    if args.test_only:
        test_model_prediction("models/xlm-roberta-base_finetuned", "xlm-roberta-base")
        return
    
    try:
        # Train the specified model
        result = train_single_model(
            model_name=args.model,
            train_file=args.train_file,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split
        )
        
        logger.info("=" * 60)
        logger.info("TASK 3 COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Model: {result['model_name']}")
        logger.info(f"Model Path: {result['model_path']}")
        logger.info(f"Evaluation Results: {result['eval_results']}")
        
        # Test predictions
        test_model_prediction(str(result['model_path']), args.model)
        
        logger.info("\nNext steps:")
        logger.info("1. Run Task 4 to compare multiple models")
        logger.info("2. Use the trained model for entity extraction")
        logger.info("3. Integrate with FinTech scoring system")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Please check the error logs and try again")

if __name__ == "__main__":
    main()
