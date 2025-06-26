#!/usr/bin/env python3
"""
Advanced Fine-tuning Script for Amharic NER Models
Demonstrates comprehensive model fine-tuning with hyperparameter optimization
"""

import sys
import os
import argparse
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from loguru import logger
from src.modeling.advanced_ner_trainer import AdvancedNERTrainer
from src.modeling.data_loader import CoNLLDataLoader
from src.utils.data_utils import setup_logging

def run_comprehensive_fine_tuning(
    model_name: str = "xlm-roberta-base",
    train_file: str = "data/labeled/amharic_ner_sample_50_messages.txt",
    optimization_method: str = "grid_search",
    max_trials: int = 20,
    cv_folds: int = 5,
    output_dir: str = "models/advanced_training"
):
    """Run comprehensive fine-tuning with all advanced features"""
    
    logger.info("=" * 80)
    logger.info("ADVANCED NER MODEL FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Training file: {train_file}")
    logger.info(f"Optimization method: {optimization_method}")
    logger.info(f"Max trials: {max_trials}")
    logger.info(f"CV folds: {cv_folds}")
    logger.info("=" * 80)
    
    # Check if training file exists
    if not Path(train_file).exists():
        logger.error(f"Training file not found: {train_file}")
        logger.info("Please run the data labeling step first")
        return False
    
    try:
        # Initialize advanced trainer
        trainer = AdvancedNERTrainer(model_name=model_name, output_dir=output_dir)
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data_loader = CoNLLDataLoader(tokenizer_name=model_name)
        trainer.data_loader = data_loader
        
        train_dataset, val_dataset = data_loader.prepare_datasets(train_file, val_split=0.2)
        label_info = data_loader.get_label_info()
        
        logger.info(f"Dataset prepared:")
        logger.info(f"  - Training samples: {len(train_dataset)}")
        logger.info(f"  - Validation samples: {len(val_dataset)}")
        logger.info(f"  - Number of labels: {label_info['num_labels']}")
        logger.info(f"  - Labels: {list(label_info['label_to_id'].keys())}")
        
        # Step 1: Hyperparameter Optimization
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 60)
        
        if optimization_method == "grid_search":
            logger.info("Performing Grid Search optimization...")
            
            # Define search space for grid search
            param_grid = {
                'learning_rate': [1e-5, 2e-5, 3e-5],
                'batch_size': [8, 16],
                'num_epochs': [3, 5],
                'warmup_ratio': [0.1, 0.2],
                'weight_decay': [0.01, 0.1],
                'dropout': [0.1, 0.2],
                'scheduler_type': ['linear', 'cosine']
            }
            
            optimization_results = trainer.grid_search(
                train_dataset, val_dataset, param_grid, max_trials
            )
            
        elif optimization_method == "bayesian":
            logger.info("Performing Bayesian optimization...")
            
            optimization_results = trainer.bayesian_optimization(
                train_dataset, val_dataset, max_trials
            )
            
        else:
            logger.info("Skipping hyperparameter optimization, using default parameters...")
            
            # Train with default parameters
            default_params = {
                'learning_rate': 2e-5,
                'batch_size': 16,
                'num_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'dropout': 0.1,
                'scheduler_type': 'linear'
            }
            
            optimization_results = trainer.train_with_hyperparams(
                train_dataset, val_dataset, default_params, "default_training"
            )
        
        logger.info("Hyperparameter optimization completed!")
        if trainer.best_hyperparams:
            logger.info(f"Best hyperparameters: {trainer.best_hyperparams}")
        
        # Step 2: Cross-Validation
        if cv_folds > 1:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 2: CROSS-VALIDATION")
            logger.info("=" * 60)
            
            # Combine train and validation for cross-validation
            combined_data = {
                'input_ids': [ex['input_ids'] for ex in train_dataset] + [ex['input_ids'] for ex in val_dataset],
                'attention_mask': [ex['attention_mask'] for ex in train_dataset] + [ex['attention_mask'] for ex in val_dataset],
                'labels': [ex['labels'] for ex in train_dataset] + [ex['labels'] for ex in val_dataset]
            }
            
            from datasets import Dataset
            combined_dataset = Dataset.from_dict(combined_data)
            
            cv_results = trainer.cross_validate(
                combined_dataset, 
                hyperparams=trainer.best_hyperparams,
                n_folds=cv_folds
            )
            
            logger.info("Cross-validation completed!")
            if 'cv_summary' in cv_results and 'error' not in cv_results['cv_summary']:
                cv_summary = cv_results['cv_summary']
                logger.info(f"CV Results - Mean F1: {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")
        
        # Step 3: Save Best Model
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: SAVING BEST MODEL")
        logger.info("=" * 60)
        
        best_model_path = trainer.save_best_model()
        if best_model_path:
            logger.info(f"Best model saved to: {best_model_path}")
        
        # Step 4: Generate Comprehensive Report
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: GENERATING TRAINING REPORT")
        logger.info("=" * 60)
        
        training_report = trainer.generate_training_report()
        
        # Create summary for display
        summary = {
            'model_name': model_name,
            'optimization_method': optimization_method,
            'total_experiments': len(trainer.experiment_history),
            'best_model_path': best_model_path,
            'training_file': train_file
        }
        
        if trainer.best_hyperparams:
            summary['best_hyperparams'] = trainer.best_hyperparams
        
        if 'summary_stats' in training_report:
            summary['best_f1_score'] = training_report['summary_stats']['best_f1']
            summary['mean_f1_score'] = training_report['summary_stats']['mean_f1']
        
        if cv_folds > 1 and 'cv_summary' in cv_results and 'error' not in cv_results['cv_summary']:
            summary['cv_mean_f1'] = cv_results['cv_summary']['mean_f1']
            summary['cv_std_f1'] = cv_results['cv_summary']['std_f1']
        
        # Save summary
        summary_file = Path(output_dir) / "fine_tuning_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Display final results
        logger.info("\n" + "=" * 80)
        logger.info("FINE-TUNING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("SUMMARY:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
        
        logger.info("\nNext steps:")
        logger.info("1. Review the training report and model performance")
        logger.info("2. Test the best model on new data")
        logger.info("3. Deploy the model for production use")
        logger.info("4. Monitor model performance and retrain as needed")
        
        return True
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Advanced Fine-tuning for Amharic NER Models")
    
    parser.add_argument('--model', '-m', default='xlm-roberta-base',
                       help='Model name to fine-tune (default: xlm-roberta-base)')
    parser.add_argument('--train-file', '-t', default='data/labeled/amharic_ner_sample_50_messages.txt',
                       help='Path to training file in CoNLL format')
    parser.add_argument('--optimization', '-opt', choices=['grid_search', 'bayesian', 'none'],
                       default='grid_search', help='Optimization method')
    parser.add_argument('--max-trials', '-mt', type=int, default=20,
                       help='Maximum number of optimization trials')
    parser.add_argument('--cv-folds', '-cv', type=int, default=5,
                       help='Number of cross-validation folds (0 to disable)')
    parser.add_argument('--output-dir', '-o', default='models/advanced_training',
                       help='Output directory for models and results')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Run fine-tuning
    success = run_comprehensive_fine_tuning(
        model_name=args.model,
        train_file=args.train_file,
        optimization_method=args.optimization,
        max_trials=args.max_trials,
        cv_folds=args.cv_folds,
        output_dir=args.output_dir
    )
    
    if success:
        logger.info("Advanced fine-tuning completed successfully!")
        sys.exit(0)
    else:
        logger.error("Advanced fine-tuning failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
