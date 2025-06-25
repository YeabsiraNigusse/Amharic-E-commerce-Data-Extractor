"""
LIME (Local Interpretable Model-agnostic Explanations) Analyzer for Amharic NER
Provides local explanations by perturbing input text
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
import random

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from loguru import logger

# Import LIME with error handling
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")

class LIMEAnalyzer:
    """LIME-based explainer for NER models"""
    
    def __init__(self, model, tokenizer, label_mapping: Dict[str, str]):
        """Initialize LIME analyzer"""
        self.model = model
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        
        if not LIME_AVAILABLE:
            logger.error("LIME is not available. Please install it to use this analyzer.")
            return
        
        # Initialize LIME explainer
        self.explainer = LimeTextExplainer(
            class_names=list(label_mapping.values()),
            mode='classification'
        )
        
        logger.info("LIME analyzer initialized")
    
    def _create_prediction_function(self, target_token_idx: int) -> Callable:
        """Create prediction function for a specific token position"""
        
        def predict_proba(texts: List[str]) -> np.ndarray:
            """Prediction function for LIME"""
            results = []
            
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                
                # Extract probabilities for the target token
                if target_token_idx < probabilities.size(1):
                    token_probs = probabilities[0, target_token_idx].numpy()
                else:
                    # If token index is out of bounds, return uniform distribution
                    token_probs = np.ones(len(self.label_mapping)) / len(self.label_mapping)
                
                results.append(token_probs)
            
            return np.array(results)
        
        return predict_proba
    
    def explain_text(self, text: str, num_features: int = 10, num_samples: int = 100) -> Dict[str, Any]:
        """Explain a text using LIME"""
        
        if not LIME_AVAILABLE:
            return {'error': 'LIME not available'}
        
        try:
            # Get base prediction
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_ids = torch.argmax(outputs.logits, dim=-1)
            
            # Convert predictions to labels
            predicted_labels = [self.label_mapping.get(str(id.item()), f"LABEL_{id.item()}") 
                              for id in predicted_ids[0]]
            
            # Explain important tokens (entities)
            token_explanations = []
            
            for i, (token, label) in enumerate(zip(tokens, predicted_labels)):
                if token not in ['<s>', '</s>', '<pad>', '<unk>'] and label != 'O':
                    # Create prediction function for this token
                    predict_fn = self._create_prediction_function(i)
                    
                    try:
                        # Get LIME explanation
                        explanation = self.explainer.explain_instance(
                            text,
                            predict_fn,
                            num_features=min(num_features, len(text.split())),
                            num_samples=num_samples
                        )
                        
                        # Extract feature importance
                        feature_importance = explanation.as_list()
                        
                        token_explanations.append({
                            'token': token,
                            'position': i,
                            'predicted_label': label,
                            'feature_importance': feature_importance,
                            'prediction_probability': float(probabilities[0, i, predicted_ids[0, i]].item())
                        })
                        
                    except Exception as e:
                        logger.warning(f"LIME explanation failed for token {token}: {e}")
                        # Fallback to simple explanation
                        token_explanations.append({
                            'token': token,
                            'position': i,
                            'predicted_label': label,
                            'feature_importance': [(token, 1.0)],
                            'prediction_probability': float(probabilities[0, i, predicted_ids[0, i]].item()),
                            'error': str(e)
                        })
            
            # Create simplified explanation using word-level perturbation
            simplified_explanation = self._create_simplified_explanation(text, predicted_labels, tokens)
            
            return {
                'text': text,
                'method': 'lime_text',
                'token_explanations': token_explanations,
                'simplified_explanation': simplified_explanation,
                'summary': self._summarize_lime_explanation(token_explanations),
                'visualization_data': self._prepare_lime_visualization_data(token_explanations, text)
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {'error': str(e)}
    
    def _create_simplified_explanation(self, text: str, predicted_labels: List[str], tokens: List[str]) -> Dict[str, Any]:
        """Create simplified explanation by word-level perturbation"""
        
        words = text.split()
        word_importance = {}
        
        # Get baseline prediction
        baseline_prediction = self._get_prediction_confidence(text)
        
        # Test importance of each word by removing it
        for i, word in enumerate(words):
            # Create perturbed text without this word
            perturbed_words = words[:i] + words[i+1:]
            perturbed_text = ' '.join(perturbed_words)
            
            if perturbed_text.strip():
                perturbed_prediction = self._get_prediction_confidence(perturbed_text)
                
                # Calculate importance as difference in confidence
                importance = baseline_prediction - perturbed_prediction
                word_importance[word] = importance
            else:
                word_importance[word] = baseline_prediction
        
        return {
            'word_importance': word_importance,
            'baseline_confidence': baseline_prediction,
            'method': 'word_removal_perturbation'
        }
    
    def _get_prediction_confidence(self, text: str) -> float:
        """Get average prediction confidence for a text"""
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
        
        # Calculate average confidence for entity predictions
        confidences = []
        for i in range(probabilities.size(1)):
            predicted_class = predicted_ids[0, i]
            confidence = probabilities[0, i, predicted_class].item()
            
            # Only consider entity predictions (not 'O')
            if predicted_class != 0:  # Assuming 'O' is class 0
                confidences.append(confidence)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _summarize_lime_explanation(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize LIME explanations"""
        
        if not explanations:
            return {}
        
        # Extract most important features across all tokens
        all_features = {}
        for exp in explanations:
            if 'feature_importance' in exp:
                for feature, importance in exp['feature_importance']:
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
        
        # Calculate average importance per feature
        avg_feature_importance = {}
        for feature, importances in all_features.items():
            avg_feature_importance[feature] = np.mean(importances)
        
        # Sort by importance
        sorted_features = sorted(avg_feature_importance.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'most_important_features': sorted_features[:10],
            'least_important_features': sorted_features[-5:],
            'total_features_analyzed': len(all_features),
            'average_prediction_confidence': np.mean([exp['prediction_probability'] for exp in explanations])
        }
    
    def _prepare_lime_visualization_data(self, explanations: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """Prepare data for LIME visualization"""
        
        words = text.split()
        word_importance = {}
        
        # Aggregate importance scores by word
        for exp in explanations:
            if 'feature_importance' in exp:
                for feature, importance in exp['feature_importance']:
                    if feature in word_importance:
                        word_importance[feature] += importance
                    else:
                        word_importance[feature] = importance
        
        return {
            'words': list(word_importance.keys()),
            'importance_scores': list(word_importance.values()),
            'original_text': text,
            'word_positions': {word: i for i, word in enumerate(words)}
        }
    
    def visualize_explanation(self, explanation: Dict[str, Any], save_path: str = None) -> None:
        """Create visualization of LIME explanation"""
        
        if 'error' in explanation:
            logger.error(f"Cannot visualize explanation with error: {explanation['error']}")
            return
        
        viz_data = explanation['visualization_data']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Word importance scores
        words = viz_data['words'][:15]  # Limit to first 15 words
        scores = viz_data['importance_scores'][:15]
        
        # Color code by importance (positive = green, negative = red)
        colors = ['green' if score > 0 else 'red' for score in scores]
        
        bars = ax1.bar(range(len(words)), scores, color=colors, alpha=0.7)
        ax1.set_xlabel('Words')
        ax1.set_ylabel('LIME Importance Score')
        ax1.set_title('Word Importance Scores (LIME Analysis)')
        ax1.set_xticks(range(len(words)))
        ax1.set_xticklabels(words, rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 2: Summary statistics
        summary = explanation['summary']
        if 'most_important_features' in summary:
            features = [item[0] for item in summary['most_important_features'][:10]]
            importances = [item[1] for item in summary['most_important_features'][:10]]
            
            ax2.barh(range(len(features)), importances, color='skyblue')
            ax2.set_ylabel('Features')
            ax2.set_xlabel('Average Importance')
            ax2.set_title('Top 10 Most Important Features')
            ax2.set_yticks(range(len(features)))
            ax2.set_yticklabels(features)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"LIME visualization saved to {save_path}")
        
        plt.close()

        return fig
    
    def batch_explain(self, texts: List[str], num_features: int = 10) -> List[Dict[str, Any]]:
        """Explain multiple texts using LIME"""
        
        explanations = []
        for i, text in enumerate(texts):
            logger.info(f"LIME explaining text {i+1}/{len(texts)}")
            explanation = self.explain_text(text, num_features=num_features)
            explanations.append(explanation)
        
        return explanations

def main():
    """Test the LIME analyzer"""
    if not LIME_AVAILABLE:
        print("LIME not available. Install with: pip install lime")
        return
    
    print("LIME analyzer initialized successfully!")
    print("Use this class to explain NER model predictions with LIME.")

if __name__ == "__main__":
    main()
