"""
Model Comparison and Selection System
Trains multiple models and provides comprehensive comparison
"""

import os
import json
import yaml
from typing import Dict, List, Any, Tuple
from pathlib import Path
import pandas as pd
import concurrent.futures
from datetime import datetime

from loguru import logger
from .ner_trainer import AmharicNERTrainer
from .data_loader import CoNLLDataLoader
from .model_evaluator import ModelEvaluator

class ModelComparison:
    """System for training and comparing multiple NER models"""
    
    def __init__(self, 
                 config_file: str = "config/model_configs.yaml",
                 output_dir: str = "models",
                 results_dir: str = "comparison_results"):
        """Initialize the comparison system"""
        self.config_file = config_file
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(str(self.results_dir))
        
        # Results storage
        self.training_results = {}
        self.comparison_results = {}
        
        logger.info(f"Initialized model comparison system")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_file}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file not found"""
        return {
            'models': {
                'xlm-roberta-base': {
                    'name': 'xlm-roberta-base',
                    'recommended_batch_size': 16,
                    'recommended_lr': 2e-5,
                    'recommended_epochs': 3
                },
                'bert-base-multilingual-cased': {
                    'name': 'bert-base-multilingual-cased',
                    'recommended_batch_size': 16,
                    'recommended_lr': 2e-5,
                    'recommended_epochs': 4
                },
                'distilbert-base-multilingual-cased': {
                    'name': 'distilbert-base-multilingual-cased',
                    'recommended_batch_size': 32,
                    'recommended_lr': 3e-5,
                    'recommended_epochs': 4
                }
            },
            'training': {
                'default': {
                    'learning_rate': 2e-5,
                    'num_epochs': 3,
                    'batch_size': 16,
                    'warmup_steps': 500,
                    'weight_decay': 0.01
                }
            }
        }
    
    def train_single_model(self, 
                          model_name: str, 
                          train_file: str,
                          val_split: float = 0.2,
                          training_config: str = "default") -> Dict[str, Any]:
        """Train a single model with specified configuration"""
        
        logger.info(f"Training model: {model_name}")
        
        try:
            # Get model and training configurations
            model_config = self.config['models'].get(model_name, {})
            train_config = self.config['training'].get(training_config, self.config['training']['default'])
            
            # Use model-specific parameters if available
            learning_rate = model_config.get('recommended_lr', train_config['learning_rate'])
            num_epochs = model_config.get('recommended_epochs', train_config['num_epochs'])
            batch_size = model_config.get('recommended_batch_size', train_config['batch_size'])
            
            # Initialize data loader
            data_loader = CoNLLDataLoader(tokenizer_name=model_name)
            
            # Prepare datasets
            train_dataset, val_dataset = data_loader.prepare_datasets(train_file, val_split=val_split)
            label_info = data_loader.get_label_info()
            
            # Initialize trainer
            trainer = AmharicNERTrainer(model_name=model_name, output_dir=str(self.output_dir))
            
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
                warmup_steps=min(train_config['warmup_steps'], len(train_dataset) // 4),
                weight_decay=train_config['weight_decay'],
                eval_strategy="epoch",
                save_strategy="epoch",
                early_stopping_patience=2
            )
            
            # Evaluate the model
            eval_results = trainer.evaluate(val_dataset)
            
            # Save model information
            trainer.save_model_info(label_info, eval_results)
            
            model_path = trainer.output_dir / f"{model_name.replace('/', '_')}_finetuned"
            
            result = {
                'model_name': model_name,
                'model_path': str(model_path),
                'train_result': train_result,
                'eval_results': eval_results,
                'training_config': {
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'warmup_steps': train_config['warmup_steps'],
                    'weight_decay': train_config['weight_decay']
                },
                'label_info': label_info,
                'status': 'success'
            }
            
            logger.info(f"Successfully trained {model_name}")
            logger.info(f"F1 Score: {eval_results.get('eval_f1', 'N/A')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            return {
                'model_name': model_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def train_multiple_models(self, 
                             model_names: List[str], 
                             train_file: str,
                             val_split: float = 0.2,
                             parallel: bool = False) -> Dict[str, Any]:
        """Train multiple models"""
        
        logger.info(f"Training {len(model_names)} models: {model_names}")
        
        results = {}
        
        if parallel and len(model_names) > 1:
            # Parallel training (be careful with GPU memory)
            logger.info("Training models in parallel")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_model = {
                    executor.submit(self.train_single_model, model_name, train_file, val_split): model_name
                    for model_name in model_names
                }
                
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        results[model_name] = result
                    except Exception as e:
                        logger.error(f"Error training {model_name}: {e}")
                        results[model_name] = {'status': 'failed', 'error': str(e)}
        else:
            # Sequential training
            logger.info("Training models sequentially")
            for model_name in model_names:
                result = self.train_single_model(model_name, train_file, val_split)
                results[model_name] = result
        
        self.training_results = results
        return results
    
    def evaluate_trained_models(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all trained models"""
        
        logger.info("Evaluating trained models")
        
        evaluation_results = []
        
        for model_name, training_result in self.training_results.items():
            if training_result.get('status') == 'success':
                model_path = training_result['model_path']
                
                if Path(model_path).exists():
                    eval_result = self.evaluator.evaluate_model(
                        model_path=model_path,
                        test_data=test_data,
                        model_name=model_name
                    )
                    evaluation_results.append(eval_result)
                else:
                    logger.warning(f"Model path not found: {model_path}")
        
        # Create comprehensive comparison report
        if evaluation_results:
            comparison_report = self.evaluator.create_evaluation_report(evaluation_results)
            self.comparison_results = comparison_report
            
            logger.info("Model evaluation completed")
            return comparison_report
        else:
            logger.warning("No models available for evaluation")
            return {}
    
    def select_best_model(self, 
                         criteria: str = "f1",
                         min_f1: float = 0.0,
                         max_inference_time: float = float('inf')) -> Dict[str, Any]:
        """Select the best model based on specified criteria"""
        
        if not self.comparison_results:
            logger.error("No comparison results available. Run evaluation first.")
            return {}
        
        model_results = self.comparison_results.get('model_results', [])
        
        if not model_results:
            logger.error("No model results found")
            return {}
        
        # Filter models based on constraints
        filtered_models = []
        for result in model_results:
            if 'error' not in result:
                f1_score = result.get('overall_f1', 0)
                inference_time = result.get('inference_time_avg', float('inf'))
                
                if f1_score >= min_f1 and inference_time <= max_inference_time:
                    filtered_models.append(result)
        
        if not filtered_models:
            logger.warning("No models meet the specified criteria")
            return {}
        
        # Select best model based on criteria
        if criteria == "f1":
            best_model = max(filtered_models, key=lambda x: x.get('overall_f1', 0))
        elif criteria == "speed":
            best_model = min(filtered_models, key=lambda x: x.get('inference_time_avg', float('inf')))
        elif criteria == "balanced":
            # Balanced score: F1 / (inference_time + 1)
            best_model = max(filtered_models, 
                           key=lambda x: x.get('overall_f1', 0) / (x.get('inference_time_avg', 1) + 1))
        else:
            logger.warning(f"Unknown criteria: {criteria}. Using F1 score.")
            best_model = max(filtered_models, key=lambda x: x.get('overall_f1', 0))
        
        logger.info(f"Selected best model: {best_model['model_name']}")
        logger.info(f"F1 Score: {best_model.get('overall_f1', 'N/A')}")
        logger.info(f"Inference Time: {best_model.get('inference_time_avg', 'N/A')}s")
        
        # Save selection results
        selection_result = {
            'selected_model': best_model,
            'selection_criteria': criteria,
            'constraints': {
                'min_f1': min_f1,
                'max_inference_time': max_inference_time
            },
            'selection_timestamp': datetime.now().isoformat(),
            'total_models_evaluated': len(model_results),
            'models_meeting_criteria': len(filtered_models)
        }
        
        selection_file = self.results_dir / "best_model_selection.json"
        with open(selection_file, 'w', encoding='utf-8') as f:
            json.dump(selection_result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Model selection results saved to {selection_file}")
        
        return selection_result
    
    def get_model_recommendations(self) -> Dict[str, str]:
        """Get model recommendations for different use cases"""
        
        if not self.comparison_results:
            return {"error": "No comparison results available"}
        
        model_results = self.comparison_results.get('model_results', [])
        valid_models = [r for r in model_results if 'error' not in r]
        
        if not valid_models:
            return {"error": "No valid model results found"}
        
        recommendations = {}
        
        # Best overall performance
        best_f1 = max(valid_models, key=lambda x: x.get('overall_f1', 0))
        recommendations['best_performance'] = {
            'model': best_f1['model_name'],
            'reason': f"Highest F1 score: {best_f1.get('overall_f1', 'N/A'):.4f}"
        }
        
        # Fastest inference
        fastest = min(valid_models, key=lambda x: x.get('inference_time_avg', float('inf')))
        recommendations['fastest_inference'] = {
            'model': fastest['model_name'],
            'reason': f"Fastest inference: {fastest.get('inference_time_avg', 'N/A'):.4f}s"
        }
        
        # Most balanced
        balanced_scores = []
        for model in valid_models:
            f1 = model.get('overall_f1', 0)
            time = model.get('inference_time_avg', 1)
            balanced_score = f1 / (time + 1)  # Avoid division by zero
            balanced_scores.append((model, balanced_score))
        
        best_balanced = max(balanced_scores, key=lambda x: x[1])
        recommendations['most_balanced'] = {
            'model': best_balanced[0]['model_name'],
            'reason': f"Best F1/speed ratio: {best_balanced[1]:.4f}"
        }
        
        # Production recommendation
        production_models = [m for m in valid_models if m.get('overall_f1', 0) > 0.7]
        if production_models:
            prod_model = min(production_models, key=lambda x: x.get('inference_time_avg', float('inf')))
            recommendations['production_ready'] = {
                'model': prod_model['model_name'],
                'reason': f"Good F1 ({prod_model.get('overall_f1', 'N/A'):.4f}) with reasonable speed"
            }
        
        return recommendations

def main():
    """Test the comparison system"""
    comparison = ModelComparison()
    print("Model comparison system initialized successfully!")
    print("Available models:", list(comparison.config['models'].keys()))

if __name__ == "__main__":
    main()
