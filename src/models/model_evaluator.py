"""
Model Evaluation Module for NER Models
Provides comprehensive evaluation metrics and analysis
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from .ner_trainer import NERTrainer


class ModelEvaluator:
    """Comprehensive evaluation for NER models"""
    
    def __init__(self, model_trainer: NERTrainer):
        self.trainer = model_trainer
        self.evaluation_results = {}
        
        # Setup logging
        logger.add("logs/model_evaluation.log", rotation="1 day")
    
    def evaluate_on_dataset(self, test_data_path: str) -> Dict[str, Any]:
        """Evaluate model on test dataset"""
        logger.info(f"Evaluating model on {test_data_path}")
        
        # Load test data
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        all_true_labels = []
        all_pred_labels = []
        token_level_results = []
        
        for item in test_data:
            text = item['text']
            true_tokens = item['tokens']
            true_labels = item['labels']

            # Get predictions
            predictions = self.trainer.predict(text)
            pred_tokens = [pred[0] for pred in predictions]
            pred_labels = [pred[1] for pred in predictions]

            # Align tokens (handle tokenization differences)
            aligned_true, aligned_pred = self._align_predictions(
                true_tokens, true_labels, pred_tokens, pred_labels
            )

            # Debug: Log lengths for each sample
            logger.debug(f"Sample {len(token_level_results)}: true={len(aligned_true)}, pred={len(aligned_pred)}")

            # Only add if both have the same length
            if len(aligned_true) == len(aligned_pred):
                all_true_labels.extend(aligned_true)
                all_pred_labels.extend(aligned_pred)
            else:
                logger.warning(f"Skipping sample due to length mismatch: {len(aligned_true)} vs {len(aligned_pred)}")
                logger.warning(f"True tokens: {len(true_tokens)}, True labels: {len(true_labels)}")
                logger.warning(f"Pred tokens: {len(pred_tokens)}, Pred labels: {len(pred_labels)}")
            
            # Store token-level results
            token_level_results.append({
                'text': text,
                'true_tokens': true_tokens,
                'true_labels': true_labels,
                'pred_tokens': pred_tokens,
                'pred_labels': pred_labels,
                'aligned_true': aligned_true,
                'aligned_pred': aligned_pred
            })
        
        # Debug: Check lengths before calculating metrics
        logger.info(f"Total true labels: {len(all_true_labels)}, Total pred labels: {len(all_pred_labels)}")

        # Ensure equal lengths before calculating metrics
        if len(all_true_labels) != len(all_pred_labels):
            min_length = min(len(all_true_labels), len(all_pred_labels))
            logger.warning(f"Length mismatch detected. Truncating to {min_length} samples.")
            all_true_labels = all_true_labels[:min_length]
            all_pred_labels = all_pred_labels[:min_length]

        # Calculate metrics
        evaluation_results = self._calculate_metrics(all_true_labels, all_pred_labels)
        evaluation_results['token_level_results'] = token_level_results
        evaluation_results['total_samples'] = len(test_data)
        
        self.evaluation_results = evaluation_results
        logger.info("Evaluation completed")
        
        return evaluation_results
    
    def _align_predictions(self, true_tokens: List[str], true_labels: List[str],
                          pred_tokens: List[str], pred_labels: List[str]) -> Tuple[List[str], List[str]]:
        """Align true and predicted labels when tokenization differs"""

        # First, ensure true tokens and labels match
        if len(true_tokens) != len(true_labels):
            min_true_len = min(len(true_tokens), len(true_labels))
            true_tokens = true_tokens[:min_true_len]
            true_labels = true_labels[:min_true_len]

        # Ensure pred tokens and labels match
        if len(pred_tokens) != len(pred_labels):
            min_pred_len = min(len(pred_tokens), len(pred_labels))
            pred_tokens = pred_tokens[:min_pred_len]
            pred_labels = pred_labels[:min_pred_len]

        # Now align the two sequences
        min_length = min(len(true_labels), len(pred_labels))

        aligned_true = true_labels[:min_length]
        aligned_pred = pred_labels[:min_length]

        return aligned_true, aligned_pred
    
    def _calculate_metrics(self, true_labels: List[str], pred_labels: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        
        # Overall metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        class_report = classification_report(
            true_labels, pred_labels, output_dict=True, zero_division=0
        )
        
        # Entity-level metrics (B- tags only)
        entity_metrics = self._calculate_entity_metrics(true_labels, pred_labels)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        
        return {
            'overall_metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': np.mean([t == p for t, p in zip(true_labels, pred_labels)])
            },
            'per_class_metrics': class_report,
            'entity_level_metrics': entity_metrics,
            'confusion_matrix': conf_matrix.tolist(),
            'label_names': sorted(list(set(true_labels + pred_labels)))
        }
    
    def _calculate_entity_metrics(self, true_labels: List[str], pred_labels: List[str]) -> Dict[str, Any]:
        """Calculate entity-level precision, recall, and F1"""
        true_entities = self._extract_entities(true_labels)
        pred_entities = self._extract_entities(pred_labels)
        
        # Calculate metrics per entity type
        entity_types = set()
        for entities in [true_entities, pred_entities]:
            for entity_type, _ in entities:
                entity_types.add(entity_type)
        
        entity_metrics = {}
        
        for entity_type in entity_types:
            true_type_entities = set([span for etype, span in true_entities if etype == entity_type])
            pred_type_entities = set([span for etype, span in pred_entities if etype == entity_type])
            
            tp = len(true_type_entities & pred_type_entities)
            fp = len(pred_type_entities - true_type_entities)
            fn = len(true_type_entities - pred_type_entities)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            entity_metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_count': len(true_type_entities),
                'pred_count': len(pred_type_entities),
                'correct_count': tp
            }
        
        return entity_metrics
    
    def _extract_entities(self, labels: List[str]) -> List[Tuple[str, Tuple[int, int]]]:
        """Extract entity spans from BIO labels"""
        entities = []
        current_entity = None
        start_idx = None
        
        for i, label in enumerate(labels):
            if label.startswith('B-'):
                # Start of new entity
                if current_entity is not None:
                    entities.append((current_entity, (start_idx, i - 1)))
                current_entity = label[2:]
                start_idx = i
            elif label.startswith('I-'):
                # Continuation of entity
                if current_entity is None or label[2:] != current_entity:
                    # Invalid sequence - treat as new entity
                    if current_entity is not None:
                        entities.append((current_entity, (start_idx, i - 1)))
                    current_entity = label[2:]
                    start_idx = i
            else:
                # Outside entity
                if current_entity is not None:
                    entities.append((current_entity, (start_idx, i - 1)))
                    current_entity = None
                    start_idx = None
        
        # Handle entity at end of sequence
        if current_entity is not None:
            entities.append((current_entity, (start_idx, len(labels) - 1)))
        
        return entities
    
    def analyze_difficult_cases(self, threshold: float = 0.5) -> Dict[str, Any]:
        """Analyze cases where the model struggles"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_on_dataset first.")
        
        difficult_cases = []
        token_results = self.evaluation_results['token_level_results']
        
        for result in token_results:
            true_labels = result['aligned_true']
            pred_labels = result['aligned_pred']
            
            # Calculate accuracy for this sample
            correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
            accuracy = correct / len(true_labels) if true_labels else 0
            
            if accuracy < threshold:
                # Analyze what went wrong
                error_analysis = self._analyze_errors(true_labels, pred_labels)
                
                difficult_cases.append({
                    'text': result['text'],
                    'accuracy': accuracy,
                    'true_labels': true_labels,
                    'pred_labels': pred_labels,
                    'error_analysis': error_analysis
                })
        
        # Categorize common error patterns
        error_patterns = self._categorize_error_patterns(difficult_cases)
        
        return {
            'difficult_cases': difficult_cases,
            'error_patterns': error_patterns,
            'total_difficult_cases': len(difficult_cases)
        }
    
    def _analyze_errors(self, true_labels: List[str], pred_labels: List[str]) -> Dict[str, Any]:
        """Analyze specific errors in a prediction"""
        errors = {
            'missed_entities': [],  # True entities not predicted
            'false_entities': [],  # Predicted entities that don't exist
            'wrong_type': [],      # Entities with wrong type
            'boundary_errors': []   # Entities with wrong boundaries
        }
        
        true_entities = self._extract_entities(true_labels)
        pred_entities = self._extract_entities(pred_labels)
        
        # Find missed entities
        for entity_type, span in true_entities:
            if (entity_type, span) not in pred_entities:
                errors['missed_entities'].append((entity_type, span))
        
        # Find false entities
        for entity_type, span in pred_entities:
            if (entity_type, span) not in true_entities:
                errors['false_entities'].append((entity_type, span))
        
        return errors
    
    def _categorize_error_patterns(self, difficult_cases: List[Dict]) -> Dict[str, Any]:
        """Categorize common error patterns"""
        patterns = {
            'entity_types_confused': defaultdict(int),
            'common_missed_entities': defaultdict(int),
            'boundary_issues': 0,
            'ambiguous_contexts': []
        }
        
        for case in difficult_cases:
            error_analysis = case['error_analysis']
            
            # Count missed entities by type
            for entity_type, span in error_analysis['missed_entities']:
                patterns['common_missed_entities'][entity_type] += 1
        
        return dict(patterns)
    
    def generate_evaluation_report(self, output_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_on_dataset first.")
        
        report_lines = []
        results = self.evaluation_results
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("AMHARIC NER MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Overall metrics
        overall = results['overall_metrics']
        report_lines.append("OVERALL PERFORMANCE:")
        report_lines.append(f"  Accuracy:  {overall['accuracy']:.4f}")
        report_lines.append(f"  Precision: {overall['precision']:.4f}")
        report_lines.append(f"  Recall:    {overall['recall']:.4f}")
        report_lines.append(f"  F1-Score:  {overall['f1_score']:.4f}")
        report_lines.append("")
        
        # Entity-level metrics
        report_lines.append("ENTITY-LEVEL PERFORMANCE:")
        entity_metrics = results['entity_level_metrics']
        for entity_type, metrics in entity_metrics.items():
            report_lines.append(f"  {entity_type}:")
            report_lines.append(f"    Precision: {metrics['precision']:.4f}")
            report_lines.append(f"    Recall:    {metrics['recall']:.4f}")
            report_lines.append(f"    F1-Score:  {metrics['f1_score']:.4f}")
            report_lines.append(f"    Count:     {metrics['true_count']} true, {metrics['pred_count']} predicted")
        report_lines.append("")
        
        # Difficult cases analysis
        if hasattr(self, 'difficult_analysis'):
            difficult = self.difficult_analysis
            report_lines.append("DIFFICULT CASES ANALYSIS:")
            report_lines.append(f"  Total difficult cases: {difficult['total_difficult_cases']}")
            report_lines.append("  Common missed entities:")
            for entity_type, count in difficult['error_patterns']['common_missed_entities'].items():
                report_lines.append(f"    {entity_type}: {count} times")
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report_content
    
    def plot_confusion_matrix(self, output_path: str = None) -> None:
        """Plot confusion matrix"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available.")
        
        conf_matrix = np.array(self.evaluation_results['confusion_matrix'])
        labels = self.evaluation_results['label_names']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - NER Model')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {output_path}")
        
        plt.show()


def main():
    """Main function for testing the evaluator"""
    from .ner_trainer import NERTrainer, NERConfig
    
    # Load trained model
    model_path = "models/ner_model"
    if Path(model_path).exists():
        config = NERConfig()
        trainer = NERTrainer(config)
        trainer.load_trained_model(model_path)
        
        # Evaluate model
        evaluator = ModelEvaluator(trainer)
        test_data_path = "data/labeled/amharic_ner_sample_50_messages.json"
        
        if Path(test_data_path).exists():
            results = evaluator.evaluate_on_dataset(test_data_path)
            
            # Analyze difficult cases
            difficult_analysis = evaluator.analyze_difficult_cases()
            evaluator.difficult_analysis = difficult_analysis
            
            # Generate report
            report = evaluator.generate_evaluation_report("reports/evaluation_report.txt")
            print(report)
            
            # Plot confusion matrix
            evaluator.plot_confusion_matrix("reports/confusion_matrix.png")
        else:
            logger.error(f"Test data not found: {test_data_path}")
    else:
        logger.error(f"Trained model not found: {model_path}")


if __name__ == "__main__":
    main()
