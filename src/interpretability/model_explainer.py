"""
Main Model Explainer for Amharic NER
Coordinates SHAP, LIME, and other interpretability methods
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from loguru import logger

from .shap_analyzer import SHAPAnalyzer
from .lime_analyzer import LIMEAnalyzer

class ModelExplainer:
    """Main explainer class for Amharic NER model interpretability"""
    
    def __init__(self, 
                 model_path: str,
                 output_dir: str = "interpretability_results"):
        """Initialize the model explainer"""
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self.label_mapping = {}
        
        # Initialize analyzers
        self.shap_analyzer = None
        self.lime_analyzer = None
        
        # Results storage
        self.explanation_results = {}
        
        logger.info(f"Initialized model explainer for: {model_path}")
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            
            # Load label mapping if available
            config_file = Path(self.model_path) / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.label_mapping = config.get('id2label', {})
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Labels: {list(self.label_mapping.values())}")
            
            # Initialize analyzers
            self.shap_analyzer = SHAPAnalyzer(self.model, self.tokenizer, self.label_mapping)
            self.lime_analyzer = LIMEAnalyzer(self.model, self.tokenizer, self.label_mapping)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        """Make prediction with confidence scores"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
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
            predicted_ids = torch.argmax(logits, dim=-1)
        
        # Convert to readable format
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.label_mapping.get(str(id.item()), f"LABEL_{id.item()}") 
                          for id in predicted_ids[0]]
        confidence_scores = [prob.max().item() for prob in probabilities[0]]
        
        # Filter out special tokens
        filtered_results = []
        for token, label, confidence in zip(tokens, predicted_labels, confidence_scores):
            if token not in ['<s>', '</s>', '<pad>', '<unk>']:
                filtered_results.append({
                    'token': token,
                    'label': label,
                    'confidence': confidence
                })
        
        return {
            'text': text,
            'predictions': filtered_results,
            'raw_logits': logits.numpy(),
            'probabilities': probabilities.numpy()
        }
    
    def explain_prediction(self, 
                          text: str, 
                          methods: List[str] = ['shap', 'lime']) -> Dict[str, Any]:
        """Explain a single prediction using specified methods"""
        
        logger.info(f"Explaining prediction for: {text[:50]}...")
        
        # Get base prediction
        prediction = self.predict_with_confidence(text)
        
        explanations = {
            'text': text,
            'prediction': prediction,
            'explanations': {}
        }
        
        # SHAP explanation
        if 'shap' in methods and self.shap_analyzer:
            try:
                shap_explanation = self.shap_analyzer.explain_text(text)
                explanations['explanations']['shap'] = shap_explanation
                logger.info("SHAP explanation completed")
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")
                explanations['explanations']['shap'] = {'error': str(e)}
        
        # LIME explanation
        if 'lime' in methods and self.lime_analyzer:
            try:
                lime_explanation = self.lime_analyzer.explain_text(text)
                explanations['explanations']['lime'] = lime_explanation
                logger.info("LIME explanation completed")
            except Exception as e:
                logger.error(f"LIME explanation failed: {e}")
                explanations['explanations']['lime'] = {'error': str(e)}
        
        return explanations
    
    def analyze_difficult_cases(self, test_texts: List[str]) -> Dict[str, Any]:
        """Analyze difficult cases where the model might struggle"""
        
        logger.info(f"Analyzing {len(test_texts)} difficult cases")
        
        difficult_cases = []
        
        for i, text in enumerate(test_texts):
            prediction = self.predict_with_confidence(text)
            
            # Identify potential difficulties
            difficulties = self._identify_difficulties(prediction)
            
            if difficulties['is_difficult']:
                case_analysis = {
                    'text': text,
                    'prediction': prediction,
                    'difficulties': difficulties,
                    'explanation': self.explain_prediction(text, methods=['shap'])
                }
                difficult_cases.append(case_analysis)
                
                logger.info(f"Difficult case {i+1}: {difficulties['reasons']}")
        
        analysis_result = {
            'total_cases': len(test_texts),
            'difficult_cases_count': len(difficult_cases),
            'difficult_cases': difficult_cases,
            'summary': self._summarize_difficulties(difficult_cases)
        }
        
        return analysis_result
    
    def _identify_difficulties(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Identify potential difficulties in a prediction"""
        
        predictions = prediction['predictions']
        
        difficulties = {
            'is_difficult': False,
            'reasons': [],
            'low_confidence_tokens': [],
            'conflicting_predictions': [],
            'entity_boundaries': []
        }
        
        # Check for low confidence predictions
        low_confidence_threshold = 0.7
        for pred in predictions:
            if pred['confidence'] < low_confidence_threshold and pred['label'] != 'O':
                difficulties['low_confidence_tokens'].append(pred)
                difficulties['is_difficult'] = True
        
        if difficulties['low_confidence_tokens']:
            difficulties['reasons'].append('Low confidence predictions')
        
        # Check for potential entity boundary issues
        prev_label = 'O'
        for i, pred in enumerate(predictions):
            current_label = pred['label']
            
            # Check for I- tags without preceding B- tags
            if current_label.startswith('I-') and not prev_label.startswith(('B-', 'I-')):
                difficulties['entity_boundaries'].append({
                    'position': i,
                    'issue': 'I-tag without B-tag',
                    'token': pred['token']
                })
                difficulties['is_difficult'] = True
            
            prev_label = current_label
        
        if difficulties['entity_boundaries']:
            difficulties['reasons'].append('Entity boundary issues')
        
        # Check for overlapping or conflicting entities
        entity_spans = self._extract_entity_spans(predictions)
        for i, span1 in enumerate(entity_spans):
            for span2 in entity_spans[i+1:]:
                if self._spans_overlap(span1, span2):
                    difficulties['conflicting_predictions'].append({
                        'span1': span1,
                        'span2': span2
                    })
                    difficulties['is_difficult'] = True
        
        if difficulties['conflicting_predictions']:
            difficulties['reasons'].append('Overlapping entities')
        
        return difficulties
    
    def _extract_entity_spans(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract entity spans from predictions"""
        spans = []
        current_span = None
        
        for i, pred in enumerate(predictions):
            label = pred['label']
            
            if label.startswith('B-'):
                # Start new entity
                if current_span:
                    spans.append(current_span)
                current_span = {
                    'start': i,
                    'end': i,
                    'entity_type': label[2:],
                    'tokens': [pred['token']],
                    'confidence': pred['confidence']
                }
            elif label.startswith('I-') and current_span and label[2:] == current_span['entity_type']:
                # Continue current entity
                current_span['end'] = i
                current_span['tokens'].append(pred['token'])
                current_span['confidence'] = min(current_span['confidence'], pred['confidence'])
            else:
                # End current entity
                if current_span:
                    spans.append(current_span)
                    current_span = None
        
        # Add final span if exists
        if current_span:
            spans.append(current_span)
        
        return spans
    
    def _spans_overlap(self, span1: Dict[str, Any], span2: Dict[str, Any]) -> bool:
        """Check if two spans overlap"""
        return not (span1['end'] < span2['start'] or span2['end'] < span1['start'])
    
    def _summarize_difficulties(self, difficult_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize common difficulties across cases"""
        
        summary = {
            'common_issues': {},
            'entity_types_affected': {},
            'average_confidence': 0.0,
            'recommendations': []
        }
        
        if not difficult_cases:
            return summary
        
        # Count common issues
        for case in difficult_cases:
            for reason in case['difficulties']['reasons']:
                summary['common_issues'][reason] = summary['common_issues'].get(reason, 0) + 1
        
        # Count affected entity types
        for case in difficult_cases:
            for pred in case['prediction']['predictions']:
                if pred['label'] != 'O':
                    entity_type = pred['label'].split('-')[-1]
                    summary['entity_types_affected'][entity_type] = \
                        summary['entity_types_affected'].get(entity_type, 0) + 1
        
        # Calculate average confidence
        all_confidences = []
        for case in difficult_cases:
            for pred in case['prediction']['predictions']:
                if pred['label'] != 'O':
                    all_confidences.append(pred['confidence'])
        
        if all_confidences:
            summary['average_confidence'] = np.mean(all_confidences)
        
        # Generate recommendations
        if 'Low confidence predictions' in summary['common_issues']:
            summary['recommendations'].append(
                "Consider additional training data for low-confidence entity types"
            )
        
        if 'Entity boundary issues' in summary['common_issues']:
            summary['recommendations'].append(
                "Review and improve entity boundary labeling in training data"
            )
        
        if 'Overlapping entities' in summary['common_issues']:
            summary['recommendations'].append(
                "Consider using a different tagging scheme or post-processing rules"
            )
        
        return summary
    
    def generate_interpretability_report(self, 
                                       test_texts: List[str],
                                       output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive interpretability report"""
        
        logger.info("Generating comprehensive interpretability report")
        
        if output_file is None:
            output_file = self.output_dir / "interpretability_report.json"
        
        # Analyze difficult cases
        difficult_analysis = self.analyze_difficult_cases(test_texts)
        
        # Get sample explanations
        sample_explanations = []
        for text in test_texts[:5]:  # Explain first 5 texts
            explanation = self.explain_prediction(text)
            sample_explanations.append(explanation)
        
        # Compile report
        report = {
            'model_path': self.model_path,
            'analysis_summary': {
                'total_texts_analyzed': len(test_texts),
                'difficult_cases_found': difficult_analysis['difficult_cases_count'],
                'difficulty_rate': difficult_analysis['difficult_cases_count'] / len(test_texts)
            },
            'difficult_cases_analysis': difficult_analysis,
            'sample_explanations': sample_explanations,
            'model_insights': self._generate_model_insights(difficult_analysis),
            'recommendations': difficult_analysis['summary']['recommendations']
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Interpretability report saved to {output_file}")
        
        return report
    
    def _generate_model_insights(self, difficult_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about model behavior"""
        
        insights = {
            'strengths': [],
            'weaknesses': [],
            'entity_performance': {},
            'confidence_analysis': {}
        }
        
        # Analyze entity performance
        entity_counts = difficult_analysis['summary']['entity_types_affected']
        total_difficult = difficult_analysis['difficult_cases_count']
        
        for entity_type, count in entity_counts.items():
            difficulty_rate = count / total_difficult if total_difficult > 0 else 0
            insights['entity_performance'][entity_type] = {
                'difficulty_rate': difficulty_rate,
                'status': 'challenging' if difficulty_rate > 0.3 else 'good'
            }
        
        # Generate strengths and weaknesses
        if total_difficult / difficult_analysis['total_cases'] < 0.2:
            insights['strengths'].append("Low overall difficulty rate - model performs well")
        
        common_issues = difficult_analysis['summary']['common_issues']
        if 'Low confidence predictions' in common_issues:
            insights['weaknesses'].append("Confidence calibration needs improvement")
        
        if 'Entity boundary issues' in common_issues:
            insights['weaknesses'].append("Entity boundary detection needs refinement")
        
        return insights

def main():
    """Test the model explainer"""
    # This would be used with an actual trained model
    print("Model explainer initialized successfully!")
    print("Use this class to explain NER model predictions.")

if __name__ == "__main__":
    main()
