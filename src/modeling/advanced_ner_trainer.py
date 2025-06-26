"""
Advanced NER Model Fine-tuning with Hyperparameter Optimization
Comprehensive fine-tuning implementation for Amharic E-commerce Entity Extraction
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
import pickle

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset
import evaluate
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import optuna
from loguru import logger

from .data_loader import CoNLLDataLoader
from .ner_trainer import AmharicNERTrainer

class AdvancedNERTrainer(AmharicNERTrainer):
    """Advanced NER trainer with hyperparameter optimization and cross-validation"""
    
    def __init__(self, model_name: str = "xlm-roberta-base", output_dir: str = "models"):
        super().__init__(model_name, output_dir)
        
        # Advanced training configurations
        self.best_hyperparams = None
        self.cv_results = []
        self.experiment_history = []
        
        # Hyperparameter search space
        self.hyperparameter_space = {
            'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
            'batch_size': [8, 16, 32],
            'num_epochs': [3, 5, 8],
            'warmup_ratio': [0.1, 0.2, 0.3],
            'weight_decay': [0.01, 0.1, 0.2],
            'dropout': [0.1, 0.2, 0.3],
            'scheduler_type': ['linear', 'cosine', 'polynomial']
        }
        
        logger.info(f"Initialized Advanced NER trainer with model: {model_name}")
    
    def setup_model_with_dropout(self, num_labels: int, label_to_id: Dict[str, int], dropout: float = 0.1):
        """Setup model with custom dropout rate"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with custom dropout
        config = AutoModelForTokenClassification.from_pretrained(self.model_name).config
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        config.num_labels = num_labels
        config.label2id = label_to_id
        config.id2label = {v: k for k, v in label_to_id.items()}
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        logger.info(f"Setup model with {num_labels} labels and dropout={dropout}")
    
    def create_optimizer_and_scheduler(self, 
                                     num_training_steps: int,
                                     learning_rate: float = 2e-5,
                                     warmup_ratio: float = 0.1,
                                     weight_decay: float = 0.01,
                                     scheduler_type: str = 'linear'):
        """Create custom optimizer and learning rate scheduler"""
        
        # Create optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        # Create scheduler
        warmup_steps = int(num_training_steps * warmup_ratio)
        
        if scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )
        elif scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )
        else:  # polynomial
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )
        
        return optimizer, scheduler
    
    def train_with_hyperparams(self,
                              train_dataset: Dataset,
                              val_dataset: Dataset,
                              hyperparams: Dict[str, Any],
                              trial_name: str = None) -> Dict[str, Any]:
        """Train model with specific hyperparameters"""
        
        if trial_name is None:
            trial_name = f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Training with hyperparameters: {hyperparams}")
        
        # Setup model with custom dropout
        label_info = self.data_loader.get_label_info() if self.data_loader else {}
        if not label_info:
            # Fallback: extract from dataset
            all_labels = set()
            for example in train_dataset:
                all_labels.update(example['labels'])
            label_to_id = {f"LABEL_{i}": i for i in sorted(all_labels)}
            label_info = {
                'label_to_id': label_to_id,
                'num_labels': len(label_to_id)
            }
        
        self.setup_model_with_dropout(
            num_labels=label_info['num_labels'],
            label_to_id=label_info['label_to_id'],
            dropout=hyperparams.get('dropout', 0.1)
        )
        
        # Calculate training steps
        num_training_steps = len(train_dataset) * hyperparams['num_epochs'] // hyperparams['batch_size']
        
        # Setup training arguments
        output_dir = self.output_dir / f"{self.model_name.replace('/', '_')}_{trial_name}"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=hyperparams['learning_rate'],
            per_device_train_batch_size=hyperparams['batch_size'],
            per_device_eval_batch_size=hyperparams['batch_size'],
            num_train_epochs=hyperparams['num_epochs'],
            weight_decay=hyperparams['weight_decay'],
            warmup_ratio=hyperparams['warmup_ratio'],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=str(output_dir / "logs"),
            logging_steps=10,
            save_total_limit=2,
            report_to=None,
            dataloader_pin_memory=False,
            gradient_accumulation_steps=1,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Evaluate the model
        eval_results = trainer.evaluate()
        
        # Save model and results
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # Save hyperparameters and results
        experiment_data = {
            'trial_name': trial_name,
            'hyperparams': hyperparams,
            'train_results': train_result.metrics,
            'eval_results': eval_results,
            'model_path': str(output_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save experiment data
        experiment_file = output_dir / "experiment_data.json"
        with open(experiment_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        self.experiment_history.append(experiment_data)
        
        logger.info(f"Trial {trial_name} completed. F1 Score: {eval_results.get('eval_f1', 'N/A')}")
        
        return experiment_data
    
    def grid_search(self,
                   train_dataset: Dataset,
                   val_dataset: Dataset,
                   param_grid: Dict[str, List] = None,
                   max_trials: int = 20) -> Dict[str, Any]:
        """Perform grid search for hyperparameter optimization"""
        
        if param_grid is None:
            param_grid = self.hyperparameter_space
        
        logger.info(f"Starting grid search with {len(list(ParameterGrid(param_grid)))} combinations")
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        
        # Limit trials if too many combinations
        if len(param_combinations) > max_trials:
            param_combinations = param_combinations[:max_trials]
            logger.info(f"Limited to {max_trials} trials")
        
        best_score = 0
        best_params = None
        best_results = None
        
        for i, params in enumerate(param_combinations):
            trial_name = f"grid_search_trial_{i+1}"
            
            try:
                results = self.train_with_hyperparams(
                    train_dataset, val_dataset, params, trial_name
                )
                
                current_score = results['eval_results'].get('eval_f1', 0)
                
                if current_score > best_score:
                    best_score = current_score
                    best_params = params
                    best_results = results
                
                logger.info(f"Trial {i+1}/{len(param_combinations)} - F1: {current_score:.4f}")
                
            except Exception as e:
                logger.error(f"Trial {trial_name} failed: {e}")
                continue
        
        self.best_hyperparams = best_params
        
        # Save grid search results
        grid_search_results = {
            'best_params': best_params,
            'best_score': best_score,
            'best_results': best_results,
            'all_experiments': self.experiment_history,
            'search_space': param_grid
        }
        
        results_file = self.output_dir / "grid_search_results.json"
        with open(results_file, 'w') as f:
            json.dump(grid_search_results, f, indent=2, default=str)
        
        logger.info(f"Grid search completed. Best F1 Score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return grid_search_results

    def bayesian_optimization(self,
                            train_dataset: Dataset,
                            val_dataset: Dataset,
                            n_trials: int = 50) -> Dict[str, Any]:
        """Perform Bayesian optimization using Optuna"""

        def objective(trial):
            # Define hyperparameter search space
            hyperparams = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
                'num_epochs': trial.suggest_int('num_epochs', 3, 8),
                'warmup_ratio': trial.suggest_float('warmup_ratio', 0.1, 0.3),
                'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.2),
                'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                'scheduler_type': trial.suggest_categorical('scheduler_type', ['linear', 'cosine'])
            }

            trial_name = f"optuna_trial_{trial.number}"

            try:
                results = self.train_with_hyperparams(
                    train_dataset, val_dataset, hyperparams, trial_name
                )

                f1_score = results['eval_results'].get('eval_f1', 0)

                # Report intermediate value for pruning
                trial.report(f1_score, step=0)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                return f1_score

            except Exception as e:
                logger.error(f"Trial {trial_name} failed: {e}")
                return 0

        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        )

        logger.info(f"Starting Bayesian optimization with {n_trials} trials")

        # Optimize
        study.optimize(objective, n_trials=n_trials)

        # Get best results
        best_params = study.best_params
        best_score = study.best_value

        self.best_hyperparams = best_params

        # Save optimization results
        optuna_results = {
            'best_params': best_params,
            'best_score': best_score,
            'study_trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
            ],
            'optimization_history': study.trials_dataframe().to_dict('records')
        }

        results_file = self.output_dir / "bayesian_optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(optuna_results, f, indent=2, default=str)

        logger.info(f"Bayesian optimization completed. Best F1 Score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return optuna_results

    def cross_validate(self,
                      dataset: Dataset,
                      hyperparams: Dict[str, Any] = None,
                      n_folds: int = 5,
                      stratify: bool = True) -> Dict[str, Any]:
        """Perform k-fold cross-validation"""

        if hyperparams is None:
            hyperparams = self.best_hyperparams or {
                'learning_rate': 2e-5,
                'batch_size': 16,
                'num_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01,
                'dropout': 0.1,
                'scheduler_type': 'linear'
            }

        logger.info(f"Starting {n_folds}-fold cross-validation")

        # Convert dataset to pandas for easier manipulation
        data_dict = {
            'input_ids': [example['input_ids'] for example in dataset],
            'attention_mask': [example['attention_mask'] for example in dataset],
            'labels': [example['labels'] for example in dataset]
        }

        # Create fold indices
        if stratify:
            # Use first label of each sequence for stratification
            y_stratify = [labels[0] if labels else 0 for labels in data_dict['labels']]
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_indices = list(kfold.split(range(len(dataset)), y_stratify))
        else:
            # Simple k-fold
            fold_size = len(dataset) // n_folds
            fold_indices = []
            for i in range(n_folds):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(dataset)
                test_indices = list(range(start_idx, end_idx))
                train_indices = list(range(0, start_idx)) + list(range(end_idx, len(dataset)))
                fold_indices.append((train_indices, test_indices))

        cv_results = []

        for fold, (train_indices, val_indices) in enumerate(fold_indices):
            logger.info(f"Training fold {fold + 1}/{n_folds}")

            # Create fold datasets
            train_fold_data = {
                'input_ids': [data_dict['input_ids'][i] for i in train_indices],
                'attention_mask': [data_dict['attention_mask'][i] for i in train_indices],
                'labels': [data_dict['labels'][i] for i in train_indices]
            }

            val_fold_data = {
                'input_ids': [data_dict['input_ids'][i] for i in val_indices],
                'attention_mask': [data_dict['attention_mask'][i] for i in val_indices],
                'labels': [data_dict['labels'][i] for i in val_indices]
            }

            train_fold_dataset = Dataset.from_dict(train_fold_data)
            val_fold_dataset = Dataset.from_dict(val_fold_data)

            # Train on fold
            trial_name = f"cv_fold_{fold + 1}"

            try:
                fold_results = self.train_with_hyperparams(
                    train_fold_dataset, val_fold_dataset, hyperparams, trial_name
                )

                cv_results.append({
                    'fold': fold + 1,
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'results': fold_results['eval_results']
                })

                logger.info(f"Fold {fold + 1} F1 Score: {fold_results['eval_results'].get('eval_f1', 'N/A')}")

            except Exception as e:
                logger.error(f"Fold {fold + 1} failed: {e}")
                cv_results.append({
                    'fold': fold + 1,
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'results': {'error': str(e)}
                })

        # Calculate cross-validation statistics
        valid_results = [r for r in cv_results if 'error' not in r['results']]

        if valid_results:
            f1_scores = [r['results'].get('eval_f1', 0) for r in valid_results]
            precision_scores = [r['results'].get('eval_precision', 0) for r in valid_results]
            recall_scores = [r['results'].get('eval_recall', 0) for r in valid_results]

            cv_summary = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'mean_precision': np.mean(precision_scores),
                'std_precision': np.std(precision_scores),
                'mean_recall': np.mean(recall_scores),
                'std_recall': np.std(recall_scores),
                'successful_folds': len(valid_results),
                'total_folds': n_folds
            }
        else:
            cv_summary = {'error': 'All folds failed'}

        cv_final_results = {
            'hyperparams': hyperparams,
            'cv_summary': cv_summary,
            'fold_results': cv_results,
            'n_folds': n_folds
        }

        # Save cross-validation results
        cv_file = self.output_dir / "cross_validation_results.json"
        with open(cv_file, 'w') as f:
            json.dump(cv_final_results, f, indent=2, default=str)

        if 'error' not in cv_summary:
            logger.info(f"Cross-validation completed. Mean F1: {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")
        else:
            logger.error("Cross-validation failed")

        self.cv_results = cv_final_results
        return cv_final_results

    def save_best_model(self, output_path: str = None) -> str:
        """Save the best model found during optimization"""

        if self.best_hyperparams is None:
            logger.error("No best hyperparameters found. Run optimization first.")
            return None

        if output_path is None:
            output_path = str(self.output_dir / "best_model")

        # Find the best experiment
        best_experiment = None
        best_score = 0

        for exp in self.experiment_history:
            f1_score = exp['eval_results'].get('eval_f1', 0)
            if f1_score > best_score:
                best_score = f1_score
                best_experiment = exp

        if best_experiment:
            # Copy best model to final location
            import shutil
            best_model_path = best_experiment['model_path']

            if Path(best_model_path).exists():
                if Path(output_path).exists():
                    shutil.rmtree(output_path)
                shutil.copytree(best_model_path, output_path)

                # Save best model info
                best_model_info = {
                    'model_name': self.model_name,
                    'best_hyperparams': self.best_hyperparams,
                    'best_score': best_score,
                    'best_experiment': best_experiment,
                    'saved_at': datetime.now().isoformat()
                }

                info_file = Path(output_path) / "best_model_info.json"
                with open(info_file, 'w') as f:
                    json.dump(best_model_info, f, indent=2, default=str)

                logger.info(f"Best model saved to {output_path} with F1 score: {best_score:.4f}")
                return output_path

        logger.error("No valid best model found to save")
        return None

    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""

        report = {
            'model_name': self.model_name,
            'training_timestamp': datetime.now().isoformat(),
            'hyperparameter_space': self.hyperparameter_space,
            'best_hyperparams': self.best_hyperparams,
            'experiment_history': self.experiment_history,
            'cv_results': self.cv_results,
            'total_experiments': len(self.experiment_history)
        }

        # Calculate summary statistics
        if self.experiment_history:
            f1_scores = [exp['eval_results'].get('eval_f1', 0) for exp in self.experiment_history]
            report['summary_stats'] = {
                'best_f1': max(f1_scores),
                'worst_f1': min(f1_scores),
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'total_trials': len(f1_scores)
            }

        # Save report
        report_file = self.output_dir / "training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate markdown report
        self._generate_markdown_report(report)

        logger.info(f"Training report saved to {report_file}")
        return report

    def _generate_markdown_report(self, report: Dict[str, Any]):
        """Generate markdown training report"""

        md_content = f"""# Advanced NER Training Report

## Model Information
- **Model Name**: {report['model_name']}
- **Training Date**: {report['training_timestamp']}
- **Total Experiments**: {report['total_experiments']}

## Best Results
"""

        if report.get('best_hyperparams'):
            md_content += f"""
### Best Hyperparameters
```json
{json.dumps(report['best_hyperparams'], indent=2)}
```
"""

        if report.get('summary_stats'):
            stats = report['summary_stats']
            md_content += f"""
### Summary Statistics
- **Best F1 Score**: {stats['best_f1']:.4f}
- **Worst F1 Score**: {stats['worst_f1']:.4f}
- **Mean F1 Score**: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}
- **Total Trials**: {stats['total_trials']}
"""

        if report.get('cv_results') and 'cv_summary' in report['cv_results']:
            cv_summary = report['cv_results']['cv_summary']
            if 'error' not in cv_summary:
                md_content += f"""
### Cross-Validation Results
- **Mean F1 Score**: {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}
- **Mean Precision**: {cv_summary['mean_precision']:.4f} ± {cv_summary['std_precision']:.4f}
- **Mean Recall**: {cv_summary['mean_recall']:.4f} ± {cv_summary['std_recall']:.4f}
- **Successful Folds**: {cv_summary['successful_folds']}/{cv_summary.get('total_folds', 'N/A')}
"""

        md_content += """
## Experiment History
| Trial | F1 Score | Learning Rate | Batch Size | Epochs | Dropout |
|-------|----------|---------------|------------|--------|---------|
"""

        for i, exp in enumerate(report.get('experiment_history', [])[:10]):  # Show top 10
            f1 = exp['eval_results'].get('eval_f1', 0)
            params = exp['hyperparams']
            md_content += f"| {i+1} | {f1:.4f} | {params.get('learning_rate', 'N/A')} | {params.get('batch_size', 'N/A')} | {params.get('num_epochs', 'N/A')} | {params.get('dropout', 'N/A')} |\n"

        # Save markdown report
        md_file = self.output_dir / "training_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Markdown report saved to {md_file}")


def main():
    """Test the advanced trainer"""
    # Initialize trainer
    trainer = AdvancedNERTrainer("xlm-roberta-base")

    # Load data
    data_loader = CoNLLDataLoader("xlm-roberta-base")
    trainer.data_loader = data_loader

    train_file = "data/labeled/amharic_ner_sample_50_messages.txt"

    if Path(train_file).exists():
        train_dataset, val_dataset = data_loader.prepare_datasets(train_file, val_split=0.2)

        logger.info("Starting hyperparameter optimization...")

        # Perform grid search (small grid for testing)
        small_grid = {
            'learning_rate': [2e-5, 3e-5],
            'batch_size': [8, 16],
            'num_epochs': [2, 3],
            'warmup_ratio': [0.1],
            'weight_decay': [0.01],
            'dropout': [0.1],
            'scheduler_type': ['linear']
        }

        grid_results = trainer.grid_search(train_dataset, val_dataset, small_grid, max_trials=4)

        # Perform cross-validation with best parameters
        cv_results = trainer.cross_validate(train_dataset, n_folds=3)

        # Save best model
        best_model_path = trainer.save_best_model()

        # Generate report
        report = trainer.generate_training_report()

        print("Advanced training completed successfully!")
        print(f"Best model saved to: {best_model_path}")
        print(f"Best F1 Score: {grid_results.get('best_score', 'N/A')}")

    else:
        print(f"Training file not found: {train_file}")


if __name__ == "__main__":
    main()
