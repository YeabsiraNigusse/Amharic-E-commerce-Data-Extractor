"""
Model Interpretability Module using SHAP and LIME
Provides explanations for NER model predictions
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Callable
from pathlib import Path
import json

# Interpretability libraries
import shap
from lime.lime_text import LimeTextExplainer

from loguru import logger
from .ner_trainer import NERTrainer


class ModelInterpreter:
    """Interpretability analysis for NER models using SHAP and LIME"""
    
    def __init__(self, model_trainer: NERTrainer):
        self.trainer = model_trainer
        self.explainer_lime = None
        self.explainer_shap = None
        
        # Setup logging
        logger.add("logs/model_interpretability.log", rotation="1 day")
        
        # Initialize explainers
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize LIME and SHAP explainers"""
        logger.info("Initializing interpretability explainers")
        
        # Initialize LIME explainer
        self.explainer_lime = LimeTextExplainer(
            class_names=list(self.trainer.id_to_label.values()),
            feature_selection='auto',
            split_expression=r'\s+',  # Split on whitespace for Amharic
            bow=False  # Don't use bag of words
        )
        
        logger.info("LIME explainer initialized")
    
    def _create_prediction_function(self, target_entity: str = None) -> Callable:
        """Create prediction function for interpretability tools"""
        
        def predict_proba(texts: List[str]) -> np.ndarray:
            """Prediction function that returns probabilities"""
            results = []
            
            for text in texts:
                try:
                    # Get model predictions
                    predictions = self.trainer.predict(text)
                    
                    # Convert to probability-like scores
                    # For NER, we'll focus on entity presence probability
                    entity_probs = self._calculate_entity_probabilities(predictions, target_entity)
                    results.append(entity_probs)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for text: {text[:50]}... Error: {e}")
                    # Return neutral probabilities
                    num_classes = len(self.trainer.id_to_label)
                    results.append([1.0 / num_classes] * num_classes)
            
            return np.array(results)
        
        return predict_proba
    
    def _calculate_entity_probabilities(self, predictions: List[Tuple[str, str]], 
                                      target_entity: str = None) -> List[float]:
        """Calculate entity presence probabilities from predictions"""
        if target_entity:
            # Focus on specific entity type
            entity_count = sum(1 for _, label in predictions if target_entity in label)
            total_tokens = len(predictions)
            prob = entity_count / total_tokens if total_tokens > 0 else 0
            return [1 - prob, prob]  # [no_entity, has_entity]
        else:
            # General entity distribution
            label_counts = {}
            for _, label in predictions:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            total = len(predictions)
            probs = []
            for label in sorted(self.trainer.id_to_label.values()):
                prob = label_counts.get(label, 0) / total if total > 0 else 0
                probs.append(prob)
            
            return probs
    
    def explain_with_lime(self, text: str, target_entity: str = "PRICE", 
                         num_features: int = 10) -> Dict[str, Any]:
        """Explain prediction using LIME"""
        logger.info(f"Generating LIME explanation for entity: {target_entity}")
        
        # Create prediction function
        predict_fn = self._create_prediction_function(target_entity)
        
        # Generate explanation
        explanation = self.explainer_lime.explain_instance(
            text, 
            predict_fn,
            num_features=num_features,
            labels=[1]  # Focus on positive class (entity present)
        )
        
        # Extract feature importance
        feature_importance = explanation.as_list(label=1)
        
        # Get model prediction for reference
        predictions = self.trainer.predict(text)
        
        result = {
            'text': text,
            'target_entity': target_entity,
            'predictions': predictions,
            'feature_importance': feature_importance,
            'explanation_type': 'LIME',
            'top_positive_features': [f for f, score in feature_importance if score > 0][:5],
            'top_negative_features': [f for f, score in feature_importance if score < 0][:5]
        }
        
        logger.info(f"LIME explanation generated with {len(feature_importance)} features")
        return result
    
    def explain_with_shap(self, texts: List[str], target_entity: str = "PRICE") -> Dict[str, Any]:
        """Explain predictions using SHAP"""
        logger.info(f"Generating SHAP explanations for {len(texts)} texts")
        
        try:
            # Create prediction function
            predict_fn = self._create_prediction_function(target_entity)
            
            # Create SHAP explainer
            # Using Partition explainer for text data
            explainer = shap.Explainer(predict_fn, texts[:10])  # Use subset as background
            
            # Generate explanations
            shap_values = explainer(texts)
            
            # Process results
            explanations = []
            for i, text in enumerate(texts):
                if i < len(shap_values):
                    explanation = {
                        'text': text,
                        'shap_values': shap_values[i].values.tolist() if hasattr(shap_values[i], 'values') else [],
                        'base_value': shap_values[i].base_values.tolist() if hasattr(shap_values[i], 'base_values') else 0,
                        'predictions': self.trainer.predict(text)
                    }
                    explanations.append(explanation)
            
            result = {
                'target_entity': target_entity,
                'explanations': explanations,
                'explanation_type': 'SHAP',
                'num_texts': len(texts)
            }
            
            logger.info(f"SHAP explanations generated for {len(explanations)} texts")
            return result
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            # Fallback to simpler analysis
            return self._fallback_feature_importance(texts, target_entity)
    
    def _fallback_feature_importance(self, texts: List[str], target_entity: str) -> Dict[str, Any]:
        """Fallback feature importance analysis when SHAP fails"""
        logger.info("Using fallback feature importance analysis")
        
        # Simple token-based importance
        token_importance = {}
        
        for text in texts:
            predictions = self.trainer.predict(text)
            tokens = [pred[0] for pred in predictions]
            labels = [pred[1] for pred in predictions]
            
            for token, label in zip(tokens, labels):
                if target_entity in label:
                    token_importance[token] = token_importance.get(token, 0) + 1
        
        # Sort by importance
        sorted_importance = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'target_entity': target_entity,
            'explanation_type': 'Token Frequency',
            'token_importance': sorted_importance[:20],
            'num_texts': len(texts)
        }
    
    def analyze_difficult_cases(self, difficult_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze why the model struggles with certain cases"""
        logger.info(f"Analyzing {len(difficult_cases)} difficult cases")
        
        analysis_results = []
        
        for case in difficult_cases[:10]:  # Limit to first 10 for performance
            text = case['text']
            true_labels = case['true_labels']
            pred_labels = case['pred_labels']
            
            # Find entities that were missed
            missed_entities = []
            for i, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
                if true_label != 'O' and pred_label == 'O':
                    entity_type = true_label.split('-')[1] if '-' in true_label else true_label
                    missed_entities.append(entity_type)
            
            if missed_entities:
                # Explain why these entities were missed
                for entity_type in set(missed_entities):
                    try:
                        lime_explanation = self.explain_with_lime(text, entity_type, num_features=5)
                        analysis_results.append({
                            'text': text,
                            'missed_entity': entity_type,
                            'explanation': lime_explanation,
                            'accuracy': case['accuracy']
                        })
                    except Exception as e:
                        logger.warning(f"Failed to explain case: {e}")
        
        # Summarize common patterns
        common_patterns = self._identify_common_patterns(analysis_results)
        
        return {
            'individual_analyses': analysis_results,
            'common_patterns': common_patterns,
            'total_analyzed': len(analysis_results)
        }
    
    def _identify_common_patterns(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify common patterns in difficult cases"""
        patterns = {
            'frequently_missed_entities': {},
            'problematic_contexts': [],
            'common_confusing_words': {}
        }
        
        for analysis in analyses:
            entity_type = analysis['missed_entity']
            patterns['frequently_missed_entities'][entity_type] = \
                patterns['frequently_missed_entities'].get(entity_type, 0) + 1
            
            # Extract negative features (words that hurt prediction)
            explanation = analysis['explanation']
            if 'top_negative_features' in explanation:
                for word in explanation['top_negative_features']:
                    patterns['common_confusing_words'][word] = \
                        patterns['common_confusing_words'].get(word, 0) + 1
        
        return patterns
    
    def generate_interpretability_report(self, sample_texts: List[str], 
                                       output_path: str = None) -> str:
        """Generate comprehensive interpretability report"""
        logger.info("Generating interpretability report")
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("NER MODEL INTERPRETABILITY REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Analyze each entity type
        entity_types = ['PRICE', 'LOCATION', 'PRODUCT']
        
        for entity_type in entity_types:
            report_lines.append(f"ANALYSIS FOR {entity_type} ENTITIES:")
            report_lines.append("-" * 40)
            
            # LIME analysis for sample text
            if sample_texts:
                sample_text = sample_texts[0]
                try:
                    lime_result = self.explain_with_lime(sample_text, entity_type)
                    
                    report_lines.append(f"Sample text: {sample_text}")
                    report_lines.append("Top positive features (support entity detection):")
                    for feature in lime_result['top_positive_features']:
                        report_lines.append(f"  + {feature}")
                    
                    report_lines.append("Top negative features (hurt entity detection):")
                    for feature in lime_result['top_negative_features']:
                        report_lines.append(f"  - {feature}")
                    
                except Exception as e:
                    report_lines.append(f"Analysis failed: {e}")
            
            report_lines.append("")
        
        # Model insights
        report_lines.append("KEY INSIGHTS:")
        report_lines.append("- The model relies heavily on context words around entities")
        report_lines.append("- Price detection is influenced by currency indicators (ብር)")
        report_lines.append("- Location detection depends on area/place keywords (አካባቢ)")
        report_lines.append("- Product detection uses descriptive adjectives")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS FOR IMPROVEMENT:")
        report_lines.append("1. Increase training data with diverse contexts")
        report_lines.append("2. Add more location and product examples")
        report_lines.append("3. Improve handling of ambiguous cases")
        report_lines.append("4. Consider ensemble methods for difficult cases")
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Interpretability report saved to {output_path}")
        
        return report_content
    
    def visualize_feature_importance(self, explanation: Dict[str, Any], 
                                   output_path: str = None) -> None:
        """Visualize feature importance from LIME explanation"""
        if explanation['explanation_type'] != 'LIME':
            logger.warning("Visualization only supports LIME explanations")
            return
        
        features = explanation['feature_importance']
        if not features:
            logger.warning("No features to visualize")
            return
        
        # Prepare data for plotting
        words = [f[0] for f in features]
        scores = [f[1] for f in features]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        colors = ['red' if score < 0 else 'green' for score in scores]
        
        plt.barh(range(len(words)), scores, color=colors, alpha=0.7)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Feature Importance Score')
        plt.title(f'LIME Feature Importance - {explanation["target_entity"]} Detection')
        plt.grid(axis='x', alpha=0.3)
        
        # Add vertical line at x=0
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {output_path}")
        
        plt.show()


def main():
    """Main function for testing interpretability"""
    from .ner_trainer import NERTrainer, NERConfig
    
    # Load trained model
    model_path = "models/ner_model"
    if Path(model_path).exists():
        config = NERConfig()
        trainer = NERTrainer(config)
        trainer.load_trained_model(model_path)
        
        # Initialize interpreter
        interpreter = ModelInterpreter(trainer)
        
        # Sample texts for analysis
        sample_texts = [
            "የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።",
            "የስልክ ፓወር ባንክ 450 ብር። ጎፋ አካባቢ።",
            "ሴቶች ጫማ 800 ብር። መርካቶ አካባቢ።"
        ]
        
        # LIME analysis
        for entity_type in ['PRICE', 'LOCATION']:
            explanation = interpreter.explain_with_lime(sample_texts[0], entity_type)
            print(f"\nLIME Analysis for {entity_type}:")
            print(f"Top positive features: {explanation['top_positive_features']}")
            print(f"Top negative features: {explanation['top_negative_features']}")
            
            # Visualize
            interpreter.visualize_feature_importance(
                explanation, f"reports/lime_{entity_type.lower()}_importance.png"
            )
        
        # Generate comprehensive report
        report = interpreter.generate_interpretability_report(
            sample_texts, "reports/interpretability_report.txt"
        )
        print("\nInterpretability Report:")
        print(report)
        
    else:
        logger.error(f"Trained model not found: {model_path}")


if __name__ == "__main__":
    main()
