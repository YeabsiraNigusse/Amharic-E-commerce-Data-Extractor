#!/usr/bin/env python3
"""
Lightweight NER Model for Memory-Constrained Environments
Uses simpler models and techniques for demonstration
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import re
from loguru import logger


class LightweightNER:
    """Lightweight NER model using sklearn"""
    
    def __init__(self, max_features: int = 1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            analyzer='char_wb'  # Character n-grams work better for Amharic
        )
        self.model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
        self.label_to_id = {}
        self.id_to_label = {}
        self.is_trained = False
        
        # Setup logging
        logger.add("logs/lightweight_ner.log", rotation="1 day")
    
    def _prepare_features(self, texts: List[str]) -> np.ndarray:
        """Extract features from texts"""
        # Simple feature extraction
        features = []
        
        for text in texts:
            # Basic features
            text_features = {
                'length': len(text),
                'has_digits': bool(re.search(r'\d', text)),
                'has_currency': bool(re.search(r'ብር|birr|etb', text.lower())),
                'has_location_words': bool(re.search(r'አካባቢ|ቦሌ|መርካቶ|ፒያሳ', text.lower())),
                'starts_with_capital': text[0].isupper() if text else False,
                'word_count': len(text.split()),
            }
            
            # Convert to feature vector
            feature_vector = [
                text_features['length'],
                int(text_features['has_digits']),
                int(text_features['has_currency']),
                int(text_features['has_location_words']),
                int(text_features['starts_with_capital']),
                text_features['word_count']
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _tokenize_and_label(self, data: List[Dict]) -> Tuple[List[str], List[List[int]]]:
        """Convert data to tokens and label sequences"""
        all_tokens = []
        all_labels = []
        
        for item in data:
            tokens = item['tokens']
            labels = item['labels']
            
            # Convert labels to IDs
            label_ids = [self.label_to_id[label] for label in labels]
            
            all_tokens.extend(tokens)
            all_labels.append(label_ids)
        
        return all_tokens, all_labels
    
    def train(self, data_path: str) -> Dict[str, Any]:
        """Train the lightweight NER model"""
        logger.info(f"Training lightweight NER model from {data_path}")
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create label mappings
        all_labels = set()
        for item in data:
            all_labels.update(item['labels'])
        
        sorted_labels = sorted(list(all_labels))
        self.label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        logger.info(f"Found {len(sorted_labels)} labels: {sorted_labels}")
        
        # Prepare training data
        all_tokens = []
        all_token_labels = []
        
        for item in data:
            tokens = item['tokens']
            labels = item['labels']
            
            for token, label in zip(tokens, labels):
                all_tokens.append(token)
                # Create one-hot encoding for each label
                label_vector = [0] * len(self.label_to_id)
                label_vector[self.label_to_id[label]] = 1
                all_token_labels.append(label_vector)
        
        # Extract features
        logger.info("Extracting features...")
        
        # Use TF-IDF for text features
        tfidf_features = self.vectorizer.fit_transform(all_tokens).toarray()
        
        # Combine with manual features
        manual_features = self._prepare_features(all_tokens)
        
        # Combine features
        X = np.hstack([tfidf_features, manual_features])
        y = np.array(all_token_labels)
        
        logger.info(f"Training with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        logger.info("Training completed")
        
        return {
            'num_samples': len(all_tokens),
            'num_features': X.shape[1],
            'num_labels': len(self.label_to_id),
            'labels': sorted_labels
        }
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        """Predict entities in text"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        tokens = text.split()
        if not tokens:
            return []
        
        # Extract features for each token
        tfidf_features = self.vectorizer.transform(tokens).toarray()
        manual_features = self._prepare_features(tokens)
        
        # Combine features
        X = np.hstack([tfidf_features, manual_features])
        
        # Predict
        predictions = self.model.predict(X)
        
        # Convert predictions to labels
        result = []
        for i, (token, pred_vector) in enumerate(zip(tokens, predictions)):
            # Get the label with highest probability
            label_id = np.argmax(pred_vector)
            label = self.id_to_label[label_id]
            result.append((token, label))
        
        return result
    
    def evaluate(self, test_data_path: str) -> Dict[str, Any]:
        """Evaluate the model"""
        logger.info(f"Evaluating model on {test_data_path}")
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        all_true_labels = []
        all_pred_labels = []
        
        for item in test_data:
            text = item['text']
            true_labels = item['labels']
            
            # Get predictions
            predictions = self.predict(text)
            pred_labels = [pred[1] for pred in predictions]
            
            # Align lengths
            min_len = min(len(true_labels), len(pred_labels))
            all_true_labels.extend(true_labels[:min_len])
            all_pred_labels.extend(pred_labels[:min_len])
        
        # Calculate metrics
        report = classification_report(
            all_true_labels, all_pred_labels, 
            output_dict=True, zero_division=0
        )
        
        return {
            'classification_report': report,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        }
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label,
            'is_trained': self.is_trained
        }
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.label_to_id = model_data['label_to_id']
        self.id_to_label = model_data['id_to_label']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {model_path}")


def create_rule_based_ner():
    """Create a simple rule-based NER for demonstration"""
    
    class RuleBasedNER:
        def __init__(self):
            self.price_patterns = [
                r'\d+\s*ብር',
                r'\d+\s*birr',
                r'\d+\s*ETB',
                r'ዋጋ\s*\d+',
            ]
            
            self.location_patterns = [
                r'ቦሌ\s*አካባቢ?',
                r'መርካቶ\s*አካባቢ?',
                r'ፒያሳ\s*አካባቢ?',
                r'ጎፋ\s*አካባቢ?',
                r'አዲስ\s*አበባ',
            ]
            
            self.contact_patterns = [
                r'\+251\d+',
                r'09\d{8}',
                r'07\d{8}',
            ]
        
        def predict(self, text: str) -> List[Tuple[str, str]]:
            """Simple rule-based prediction"""
            tokens = text.split()
            labels = ['O'] * len(tokens)
            
            # Check for prices
            for pattern in self.price_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    start, end = match.span()
                    # Find which tokens this spans
                    char_pos = 0
                    for i, token in enumerate(tokens):
                        if char_pos <= start < char_pos + len(token):
                            labels[i] = 'B-PRICE'
                        elif char_pos < end <= char_pos + len(token):
                            if labels[i] == 'O':
                                labels[i] = 'I-PRICE'
                        char_pos += len(token) + 1  # +1 for space
            
            # Check for locations
            for pattern in self.location_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    start, end = match.span()
                    char_pos = 0
                    for i, token in enumerate(tokens):
                        if char_pos <= start < char_pos + len(token):
                            if labels[i] == 'O':
                                labels[i] = 'B-LOCATION'
                        elif char_pos < end <= char_pos + len(token):
                            if labels[i] == 'O':
                                labels[i] = 'I-LOCATION'
                        char_pos += len(token) + 1
            
            # Check for contacts
            for pattern in self.contact_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    start, end = match.span()
                    char_pos = 0
                    for i, token in enumerate(tokens):
                        if char_pos <= start < char_pos + len(token):
                            if labels[i] == 'O':
                                labels[i] = 'B-CONTACT_INFO'
                        char_pos += len(token) + 1
            
            return list(zip(tokens, labels))
    
    return RuleBasedNER()


def main():
    """Test the lightweight NER"""
    data_path = "data/labeled/amharic_ner_sample_50_messages.json"
    
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Test lightweight model
    logger.info("Testing Lightweight NER Model")
    lightweight_ner = LightweightNER(max_features=500)
    
    try:
        # Train model
        results = lightweight_ner.train(data_path)
        logger.info(f"Training results: {results}")
        
        # Save model
        model_path = "models/lightweight_ner.pkl"
        lightweight_ner.save_model(model_path)
        
        # Test prediction
        test_text = "የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።"
        predictions = lightweight_ner.predict(test_text)
        logger.info(f"Lightweight prediction: {predictions}")
        
        # Evaluate
        eval_results = lightweight_ner.evaluate(data_path)
        logger.info(f"Evaluation results: {eval_results}")
        
    except Exception as e:
        logger.error(f"Lightweight model failed: {e}")
    
    # Test rule-based model
    logger.info("Testing Rule-Based NER Model")
    rule_ner = create_rule_based_ner()
    
    test_text = "የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።"
    rule_predictions = rule_ner.predict(test_text)
    logger.info(f"Rule-based prediction: {rule_predictions}")
    
    print("\n" + "="*60)
    print("LIGHTWEIGHT NER MODELS TESTED")
    print("="*60)
    print("✅ Lightweight sklearn-based NER")
    print("✅ Rule-based NER for comparison")
    print("Both models can be used for interpretability analysis")


if __name__ == "__main__":
    main()
