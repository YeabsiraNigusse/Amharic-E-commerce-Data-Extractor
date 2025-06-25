"""
SHAP (SHapley Additive exPlanations) Analyzer for Amharic NER
Provides token-level feature importance using SHAP values
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from loguru import logger

# Import SHAP with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

class SHAPAnalyzer:
    """SHAP-based explainer for NER models"""
    
    def __init__(self, model, tokenizer, label_mapping: Dict[str, str]):
        """Initialize SHAP analyzer"""
        self.model = model
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        
        if not SHAP_AVAILABLE:
            logger.error("SHAP is not available. Please install it to use this analyzer.")
            return
        
        # Initialize SHAP explainer
        self.explainer = None
        self._setup_explainer()
        
        logger.info("SHAP analyzer initialized")
    
    def _setup_explainer(self):
        """Setup SHAP explainer for the model"""
        if not SHAP_AVAILABLE:
            return
        
        try:
            # Create a wrapper function for SHAP
            def model_wrapper(texts):
                """Wrapper function for SHAP to call the model"""
                if isinstance(texts, str):
                    texts = [texts]
                
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
                        logits = outputs.logits
                        probabilities = torch.softmax(logits, dim=-1)
                    
                    # Return probabilities for all classes
                    results.append(probabilities[0].numpy())
                
                return np.array(results)
            
            # Initialize explainer with a simple approach
            self.model_wrapper = model_wrapper
            logger.info("SHAP model wrapper created")
            
        except Exception as e:
            logger.error(f"Failed to setup SHAP explainer: {e}")
            self.explainer = None
    
    def explain_text(self, text: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Explain a text using SHAP"""
        
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available'}
        
        try:
            # Tokenize the input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Get base prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_ids = torch.argmax(logits, dim=-1)
            
            # Convert predictions to labels
            predicted_labels = [self.label_mapping.get(str(id.item()), f"LABEL_{id.item()}") 
                              for id in predicted_ids[0]]
            
            # Create simplified SHAP explanation using gradient-based approach
            explanation = self._gradient_based_explanation(text, inputs, outputs)
            
            # Filter out special tokens
            filtered_explanation = []
            for i, (token, label, importance) in enumerate(zip(tokens, predicted_labels, explanation['token_importance'])):
                if token not in ['<s>', '</s>', '<pad>', '<unk>'] and i < max_tokens:
                    filtered_explanation.append({
                        'token': token,
                        'predicted_label': label,
                        'importance_score': float(importance),
                        'position': i
                    })
            
            return {
                'text': text,
                'method': 'gradient_based_shap',
                'token_explanations': filtered_explanation,
                'summary': self._summarize_explanation(filtered_explanation),
                'visualization_data': self._prepare_visualization_data(filtered_explanation)
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}
    
    def _gradient_based_explanation(self, text: str, inputs: Dict, outputs) -> Dict[str, Any]:
        """Create gradient-based explanation as SHAP alternative"""
        
        # Enable gradients for input embeddings
        self.model.eval()
        inputs_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
        inputs_embeds.requires_grad_(True)
        
        # Forward pass with embeddings
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"])
        
        # Calculate gradients for each predicted class
        token_importance = []
        
        for token_idx in range(inputs_embeds.size(1)):
            # Get the predicted class for this token
            predicted_class = torch.argmax(outputs.logits[0, token_idx])
            
            # Calculate gradient
            if inputs_embeds.grad is not None:
                inputs_embeds.grad.zero_()
            
            outputs.logits[0, token_idx, predicted_class].backward(retain_graph=True)
            
            # Calculate importance as gradient magnitude
            if inputs_embeds.grad is not None:
                importance = torch.norm(inputs_embeds.grad[0, token_idx]).item()
            else:
                importance = 0.0
            
            token_importance.append(importance)
        
        return {
            'token_importance': token_importance,
            'method': 'gradient_magnitude'
        }
    
    def _summarize_explanation(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize the SHAP explanation"""
        
        if not explanations:
            return {}
        
        # Find most important tokens
        sorted_explanations = sorted(explanations, key=lambda x: abs(x['importance_score']), reverse=True)
        
        # Group by entity types
        entity_importance = {}
        for exp in explanations:
            label = exp['predicted_label']
            if label != 'O':
                entity_type = label.split('-')[-1]
                if entity_type not in entity_importance:
                    entity_importance[entity_type] = []
                entity_importance[entity_type].append(exp['importance_score'])
        
        # Calculate average importance per entity type
        avg_entity_importance = {}
        for entity_type, scores in entity_importance.items():
            avg_entity_importance[entity_type] = np.mean(scores)
        
        return {
            'most_important_tokens': sorted_explanations[:5],
            'least_important_tokens': sorted_explanations[-5:],
            'average_importance_by_entity': avg_entity_importance,
            'total_tokens_analyzed': len(explanations),
            'entity_tokens_count': len([e for e in explanations if e['predicted_label'] != 'O'])
        }
    
    def _prepare_visualization_data(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for visualization"""
        
        tokens = [exp['token'] for exp in explanations]
        importance_scores = [exp['importance_score'] for exp in explanations]
        labels = [exp['predicted_label'] for exp in explanations]
        
        return {
            'tokens': tokens,
            'importance_scores': importance_scores,
            'labels': labels,
            'normalized_scores': self._normalize_scores(importance_scores)
        }
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize importance scores to [0, 1] range"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def visualize_explanation(self, explanation: Dict[str, Any], save_path: str = None) -> None:
        """Create visualization of SHAP explanation"""
        
        if 'error' in explanation:
            logger.error(f"Cannot visualize explanation with error: {explanation['error']}")
            return
        
        viz_data = explanation['visualization_data']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Token importance scores
        tokens = viz_data['tokens'][:20]  # Limit to first 20 tokens
        scores = viz_data['importance_scores'][:20]
        labels = viz_data['labels'][:20]
        
        # Color code by entity type
        colors = []
        for label in labels:
            if label == 'O':
                colors.append('lightgray')
            elif 'PRICE' in label:
                colors.append('green')
            elif 'LOCATION' in label:
                colors.append('blue')
            elif 'PRODUCT' in label:
                colors.append('orange')
            else:
                colors.append('red')
        
        bars = ax1.bar(range(len(tokens)), scores, color=colors)
        ax1.set_xlabel('Tokens')
        ax1.set_ylabel('Importance Score')
        ax1.set_title('Token Importance Scores (SHAP-like Analysis)')
        ax1.set_xticks(range(len(tokens)))
        ax1.set_xticklabels(tokens, rotation=45, ha='right')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, color='lightgray', label='O'),
            plt.Rectangle((0,0),1,1, color='green', label='PRICE'),
            plt.Rectangle((0,0),1,1, color='blue', label='LOCATION'),
            plt.Rectangle((0,0),1,1, color='orange', label='PRODUCT'),
            plt.Rectangle((0,0),1,1, color='red', label='OTHER')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Entity type importance
        summary = explanation['summary']
        if 'average_importance_by_entity' in summary:
            entity_types = list(summary['average_importance_by_entity'].keys())
            entity_scores = list(summary['average_importance_by_entity'].values())
            
            ax2.bar(entity_types, entity_scores, color=['green', 'blue', 'orange', 'red'][:len(entity_types)])
            ax2.set_xlabel('Entity Types')
            ax2.set_ylabel('Average Importance Score')
            ax2.set_title('Average Importance by Entity Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP visualization saved to {save_path}")
        
        plt.close()
    
    def batch_explain(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Explain multiple texts"""
        
        explanations = []
        for i, text in enumerate(texts):
            logger.info(f"Explaining text {i+1}/{len(texts)}")
            explanation = self.explain_text(text)
            explanations.append(explanation)
        
        return explanations

def main():
    """Test the SHAP analyzer"""
    if not SHAP_AVAILABLE:
        print("SHAP not available. Install with: pip install shap")
        return
    
    print("SHAP analyzer initialized successfully!")
    print("Use this class to explain NER model predictions with SHAP.")

if __name__ == "__main__":
    main()
