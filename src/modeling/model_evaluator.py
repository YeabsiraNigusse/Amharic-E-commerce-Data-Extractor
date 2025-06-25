"""
Model Evaluator for comparing different NER models
Provides comprehensive evaluation metrics and comparison tools
"""

import json
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report, confusion_matrix
import evaluate
from loguru import logger

class ModelEvaluator:
    """Evaluator for comparing NER models"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """Initialize the evaluator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.seqeval = evaluate.load("seqeval")
        
        # Results storage
        self.evaluation_results = {}
        
        logger.info(f"Initialized model evaluator, output dir: {output_dir}")
    
    def load_model(self, model_path: str) -> Tuple[Any, Any]:
        """Load a trained model and tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            
            logger.info(f"Loaded model from {model_path}")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None, None
    
    def predict_batch(self, model, tokenizer, texts: List[str]) -> List[List[str]]:
        """Make predictions on a batch of texts"""
        all_predictions = []
        
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=2)
            
            # Convert to labels
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_labels = [model.config.id2label[id.item()] for id in predicted_ids[0]]
            
            # Filter out special tokens
            filtered_labels = []
            for token, label in zip(tokens, predicted_labels):
                if token not in ['<s>', '</s>', '<pad>', '<unk>']:
                    filtered_labels.append(label)
            
            all_predictions.append(filtered_labels)
        
        return all_predictions
    
    def evaluate_model(self, 
                      model_path: str, 
                      test_data: List[Dict[str, Any]], 
                      model_name: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        logger.info(f"Evaluating model: {model_name}")
        
        # Load model
        model, tokenizer = self.load_model(model_path)
        if model is None:
            return {"error": f"Failed to load model from {model_path}"}
        
        # Prepare test data
        test_texts = [' '.join(item['tokens']) for item in test_data]
        true_labels = [item['labels'] for item in test_data]
        
        # Measure inference time
        start_time = time.time()
        predictions = self.predict_batch(model, tokenizer, test_texts)
        inference_time = time.time() - start_time
        
        # Align predictions with true labels (handle length differences)
        aligned_predictions = []
        aligned_true_labels = []
        
        for pred, true in zip(predictions, true_labels):
            min_len = min(len(pred), len(true))
            aligned_predictions.append(pred[:min_len])
            aligned_true_labels.append(true[:min_len])
        
        # Calculate metrics
        try:
            seqeval_results = self.seqeval.compute(
                predictions=aligned_predictions, 
                references=aligned_true_labels
            )
            
            # Entity-level metrics
            entity_metrics = self._calculate_entity_metrics(aligned_predictions, aligned_true_labels)
            
            # Speed metrics
            avg_inference_time = inference_time / len(test_texts)
            
            results = {
                'model_name': model_name,
                'model_path': model_path,
                'overall_precision': seqeval_results['overall_precision'],
                'overall_recall': seqeval_results['overall_recall'],
                'overall_f1': seqeval_results['overall_f1'],
                'overall_accuracy': seqeval_results['overall_accuracy'],
                'entity_metrics': entity_metrics,
                'inference_time_total': inference_time,
                'inference_time_avg': avg_inference_time,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'test_samples': len(test_data)
            }
            
            # Store results
            self.evaluation_results[model_name] = results
            
            logger.info(f"Evaluation completed for {model_name}")
            logger.info(f"F1 Score: {results['overall_f1']:.4f}")
            logger.info(f"Avg Inference Time: {avg_inference_time:.4f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation of {model_name}: {e}")
            return {"error": str(e)}
    
    def _calculate_entity_metrics(self, predictions: List[List[str]], 
                                 true_labels: List[List[str]]) -> Dict[str, Dict[str, float]]:
        """Calculate per-entity type metrics"""
        # Flatten predictions and labels
        flat_predictions = [label for seq in predictions for label in seq]
        flat_true_labels = [label for seq in true_labels for label in seq]
        
        # Get unique labels (excluding 'O')
        unique_labels = set(flat_true_labels + flat_predictions)
        entity_labels = [label for label in unique_labels if label != 'O']
        
        # Calculate metrics per entity type
        entity_metrics = {}
        
        for entity_type in ['PRODUCT', 'PRICE', 'LOCATION', 'CONTACT_INFO', 'DELIVERY_FEE']:
            # Get B- and I- labels for this entity type
            b_label = f'B-{entity_type}'
            i_label = f'I-{entity_type}'
            
            if b_label in unique_labels or i_label in unique_labels:
                # Binary classification for this entity type
                binary_true = [1 if label.endswith(entity_type) else 0 for label in flat_true_labels]
                binary_pred = [1 if label.endswith(entity_type) else 0 for label in flat_predictions]
                
                # Calculate precision, recall, F1
                tp = sum(1 for t, p in zip(binary_true, binary_pred) if t == 1 and p == 1)
                fp = sum(1 for t, p in zip(binary_true, binary_pred) if t == 0 and p == 1)
                fn = sum(1 for t, p in zip(binary_true, binary_pred) if t == 1 and p == 0)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                entity_metrics[entity_type] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': sum(binary_true)
                }
        
        return entity_metrics
    
    def compare_models(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple models and create comparison table"""
        comparison_data = []
        
        for result in results_list:
            if 'error' not in result:
                comparison_data.append({
                    'Model': result['model_name'],
                    'F1 Score': result['overall_f1'],
                    'Precision': result['overall_precision'],
                    'Recall': result['overall_recall'],
                    'Accuracy': result['overall_accuracy'],
                    'Avg Inference Time (s)': result['inference_time_avg'],
                    'Parameters (M)': result['num_parameters'] / 1_000_000,
                })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1 Score', ascending=False)
        
        # Save comparison table
        comparison_file = self.output_dir / "model_comparison.csv"
        df.to_csv(comparison_file, index=False)
        
        logger.info(f"Model comparison saved to {comparison_file}")
        return df
    
    def create_evaluation_report(self, results_list: List[Dict[str, Any]]):
        """Create comprehensive evaluation report"""
        # Model comparison
        comparison_df = self.compare_models(results_list)
        
        # Create visualizations
        self._create_comparison_plots(comparison_df)
        
        # Create detailed report
        report = {
            'evaluation_summary': {
                'total_models_evaluated': len(results_list),
                'best_model': comparison_df.iloc[0]['Model'] if len(comparison_df) > 0 else None,
                'best_f1_score': comparison_df.iloc[0]['F1 Score'] if len(comparison_df) > 0 else None,
            },
            'model_results': results_list,
            'comparison_table': comparison_df.to_dict('records')
        }
        
        # Save report
        report_file = self.output_dir / "evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Evaluation report saved to {report_file}")
        
        return report
    
    def _create_comparison_plots(self, comparison_df: pd.DataFrame):
        """Create comparison visualizations"""
        if len(comparison_df) == 0:
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
        
        # F1 Score comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['F1 Score'])
        axes[0, 0].set_title('F1 Score Comparison')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall
        axes[0, 1].scatter(comparison_df['Precision'], comparison_df['Recall'], s=100)
        for i, model in enumerate(comparison_df['Model']):
            axes[0, 1].annotate(model, (comparison_df['Precision'].iloc[i], comparison_df['Recall'].iloc[i]))
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Precision vs Recall')
        
        # Inference Time comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['Avg Inference Time (s)'])
        axes[1, 0].set_title('Average Inference Time')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Model Size vs Performance
        axes[1, 1].scatter(comparison_df['Parameters (M)'], comparison_df['F1 Score'], s=100)
        for i, model in enumerate(comparison_df['Model']):
            axes[1, 1].annotate(model, (comparison_df['Parameters (M)'].iloc[i], comparison_df['F1 Score'].iloc[i]))
        axes[1, 1].set_xlabel('Parameters (Millions)')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Model Size vs Performance')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "model_comparison_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plots saved to {plot_file}")

def main():
    """Test the evaluator"""
    evaluator = ModelEvaluator()
    
    # This would be used with actual trained models
    print("Model evaluator initialized successfully!")
    print("Use this class to evaluate and compare trained NER models.")

if __name__ == "__main__":
    main()
