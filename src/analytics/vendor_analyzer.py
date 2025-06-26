"""
Vendor Analytics Engine for Micro-Lending Assessment
Analyzes vendor performance and business activity from Telegram data
"""

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict

from loguru import logger
from ..models.ner_trainer import NERTrainer


class VendorAnalyzer:
    """Analyzes vendor performance for micro-lending assessment"""
    
    def __init__(self, ner_model_path: str = None):
        self.ner_trainer = None
        self.vendor_data = {}
        self.analysis_results = {}
        
        # Setup logging
        logger.add("logs/vendor_analysis.log", rotation="1 day")
        
        # Load NER model if provided
        if ner_model_path and Path(ner_model_path).exists():
            self._load_ner_model(ner_model_path)
    
    def _load_ner_model(self, model_path: str):
        """Load trained NER model for entity extraction"""
        try:
            from ..models.ner_trainer import NERConfig
            config = NERConfig()
            self.ner_trainer = NERTrainer(config)
            self.ner_trainer.load_trained_model(model_path)
            logger.info(f"NER model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load NER model: {e}")
            self.ner_trainer = None
    
    def load_telegram_data(self, data_path: str) -> pd.DataFrame:
        """Load Telegram data for analysis"""
        logger.info(f"Loading Telegram data from {data_path}")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path, encoding='utf-8')
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Loaded {len(df)} messages from {df['channel_name'].nunique()} channels")
        return df
    
    def extract_entities_from_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract entities from message text using NER model"""
        logger.info("Extracting entities from messages")
        
        if self.ner_trainer is None:
            logger.warning("NER model not available, using fallback extraction")
            return self._fallback_entity_extraction(df)
        
        # Add entity columns
        df['extracted_entities'] = None
        df['prices'] = None
        df['locations'] = None
        df['products'] = None
        
        for idx, row in df.iterrows():
            text = row.get('text', '')
            if not text:
                continue
            
            try:
                # Get NER predictions
                predictions = self.ner_trainer.predict(text)
                entities = self._parse_ner_predictions(predictions)
                
                df.at[idx, 'extracted_entities'] = entities
                df.at[idx, 'prices'] = entities.get('PRICE', [])
                df.at[idx, 'locations'] = entities.get('LOCATION', [])
                df.at[idx, 'products'] = entities.get('PRODUCT', [])
                
            except Exception as e:
                logger.warning(f"Entity extraction failed for message {idx}: {e}")
        
        logger.info("Entity extraction completed")
        return df
    
    def _parse_ner_predictions(self, predictions: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Parse NER predictions into entity groups"""
        entities = defaultdict(list)
        current_entity = None
        current_tokens = []
        
        for token, label in predictions:
            if label.startswith('B-'):
                # Save previous entity
                if current_entity and current_tokens:
                    entities[current_entity].append(' '.join(current_tokens))
                
                # Start new entity
                current_entity = label[2:]
                current_tokens = [token]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                # Continue current entity
                current_tokens.append(token)
                
            else:
                # End current entity
                if current_entity and current_tokens:
                    entities[current_entity].append(' '.join(current_tokens))
                current_entity = None
                current_tokens = []
        
        # Save final entity
        if current_entity and current_tokens:
            entities[current_entity].append(' '.join(current_tokens))
        
        return dict(entities)
    
    def _fallback_entity_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback entity extraction using regex patterns"""
        logger.info("Using fallback regex-based entity extraction")
        
        # Price patterns
        price_pattern = r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:ብር|birr|ETB)'
        
        # Location patterns (common Ethiopian places)
        location_patterns = [
            r'(?:በ)?(?:ቦሌ|መርካቶ|ፒያሳ|ጎፋ|ካዛንቺስ|ሰሚት|አራት\s*ኪሎ|ሽሮ\s*መዳ|ጀሞ|ሳሪስ)\s*(?:አካባቢ)?',
            r'(?:አዲስ\s*አበባ|አ\.አ)',
        ]
        
        df['prices'] = df['text'].str.extractall(price_pattern)[0].groupby(level=0).apply(list)
        df['prices'] = df['prices'].fillna('').apply(lambda x: x if isinstance(x, list) else [])
        
        # Extract locations
        df['locations'] = None
        for idx, text in df['text'].items():
            locations = []
            for pattern in location_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                locations.extend(matches)
            df.at[idx, 'locations'] = locations
        
        # Simple product extraction (nouns before price)
        df['products'] = df['text'].apply(self._extract_products_simple)
        
        return df
    
    def _extract_products_simple(self, text: str) -> List[str]:
        """Simple product extraction"""
        # Look for common product words
        product_words = ['ልብስ', 'ጫማ', 'ስልክ', 'ፓወር', 'ባንክ', 'መጽሐፍ', 'ቦርሳ']
        products = []
        
        for word in product_words:
            if word in text:
                products.append(word)
        
        return products
    
    def analyze_vendor_performance(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze performance metrics for each vendor/channel"""
        logger.info("Analyzing vendor performance")
        
        vendor_metrics = {}
        
        # Group by channel/vendor
        for channel_name, channel_data in df.groupby('channel_name'):
            metrics = self._calculate_vendor_metrics(channel_data)
            vendor_metrics[channel_name] = metrics
        
        self.analysis_results = vendor_metrics
        logger.info(f"Analyzed {len(vendor_metrics)} vendors")
        
        return vendor_metrics
    
    def _calculate_vendor_metrics(self, channel_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a single vendor"""
        
        # Basic activity metrics
        total_posts = len(channel_data)
        date_range = (channel_data['date'].max() - channel_data['date'].min()).days
        posting_frequency = total_posts / max(date_range / 7, 1)  # Posts per week
        
        # Engagement metrics
        avg_views = channel_data['views'].mean() if 'views' in channel_data.columns else 0
        max_views = channel_data['views'].max() if 'views' in channel_data.columns else 0
        total_views = channel_data['views'].sum() if 'views' in channel_data.columns else 0
        
        # Find top performing post
        top_post = None
        if 'views' in channel_data.columns and not channel_data.empty:
            top_idx = channel_data['views'].idxmax()
            top_post = {
                'text': channel_data.loc[top_idx, 'text'],
                'views': channel_data.loc[top_idx, 'views'],
                'date': channel_data.loc[top_idx, 'date'].isoformat() if pd.notna(channel_data.loc[top_idx, 'date']) else None
            }
        
        # Business profile metrics
        prices = []
        products = []
        locations = []
        
        for _, row in channel_data.iterrows():
            if 'prices' in row and row['prices']:
                # Extract numeric values from price strings
                for price_str in row['prices']:
                    price_num = self._extract_price_number(price_str)
                    if price_num:
                        prices.append(price_num)
            
            if 'products' in row and row['products']:
                products.extend(row['products'])
            
            if 'locations' in row and row['locations']:
                locations.extend(row['locations'])
        
        # Price analysis
        avg_price = np.mean(prices) if prices else 0
        min_price = min(prices) if prices else 0
        max_price = max(prices) if prices else 0
        price_range = max_price - min_price if prices else 0
        
        # Product diversity
        unique_products = len(set(products)) if products else 0
        
        # Location coverage
        unique_locations = len(set(locations)) if locations else 0
        
        # Consistency metrics
        posts_with_prices = sum(1 for _, row in channel_data.iterrows() 
                               if 'prices' in row and row['prices'])
        price_listing_rate = posts_with_prices / total_posts if total_posts > 0 else 0
        
        # Recent activity (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_posts = channel_data[channel_data['date'] > recent_cutoff] if 'date' in channel_data.columns else pd.DataFrame()
        recent_activity = len(recent_posts)
        
        return {
            # Activity & Consistency
            'total_posts': total_posts,
            'posting_frequency_per_week': round(posting_frequency, 2),
            'date_range_days': date_range,
            'recent_activity_30d': recent_activity,
            
            # Market Reach & Engagement
            'avg_views_per_post': round(avg_views, 2),
            'max_views': max_views,
            'total_views': total_views,
            'top_performing_post': top_post,
            
            # Business Profile
            'avg_price_etb': round(avg_price, 2),
            'min_price_etb': min_price,
            'max_price_etb': max_price,
            'price_range_etb': round(price_range, 2),
            'total_products_mentioned': len(products),
            'unique_products': unique_products,
            'unique_locations': unique_locations,
            'price_listing_rate': round(price_listing_rate, 2),
            
            # Raw data for further analysis
            'all_prices': prices,
            'all_products': products,
            'all_locations': locations
        }
    
    def _extract_price_number(self, price_str: str) -> Optional[float]:
        """Extract numeric value from price string"""
        try:
            # Remove currency symbols and extract number
            number_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d{2})?)', str(price_str))
            if number_match:
                number_str = number_match.group(1).replace(',', '')
                return float(number_str)
        except (ValueError, AttributeError):
            pass
        return None
    
    def calculate_lending_scores(self, vendor_metrics: Dict[str, Dict[str, Any]], 
                               weights: Dict[str, float] = None) -> Dict[str, float]:
        """Calculate lending scores for vendors"""
        logger.info("Calculating lending scores")
        
        if weights is None:
            weights = {
                'avg_views': 0.3,
                'posting_frequency': 0.2,
                'price_consistency': 0.2,
                'recent_activity': 0.15,
                'business_diversity': 0.15
            }
        
        lending_scores = {}
        
        # Normalize metrics across all vendors
        all_avg_views = [metrics['avg_views_per_post'] for metrics in vendor_metrics.values()]
        all_posting_freq = [metrics['posting_frequency_per_week'] for metrics in vendor_metrics.values()]
        all_recent_activity = [metrics['recent_activity_30d'] for metrics in vendor_metrics.values()]
        
        max_views = max(all_avg_views) if all_avg_views else 1
        max_freq = max(all_posting_freq) if all_posting_freq else 1
        max_recent = max(all_recent_activity) if all_recent_activity else 1
        
        for vendor_name, metrics in vendor_metrics.items():
            # Normalize metrics (0-1 scale)
            norm_views = metrics['avg_views_per_post'] / max_views if max_views > 0 else 0
            norm_freq = metrics['posting_frequency_per_week'] / max_freq if max_freq > 0 else 0
            norm_recent = metrics['recent_activity_30d'] / max_recent if max_recent > 0 else 0
            
            # Price consistency (higher is better)
            price_consistency = metrics['price_listing_rate']
            
            # Business diversity
            diversity_score = min(metrics['unique_products'] / 5, 1.0)  # Cap at 5 products
            
            # Calculate weighted score
            lending_score = (
                norm_views * weights['avg_views'] +
                norm_freq * weights['posting_frequency'] +
                price_consistency * weights['price_consistency'] +
                norm_recent * weights['recent_activity'] +
                diversity_score * weights['business_diversity']
            )
            
            # Scale to 0-100
            lending_scores[vendor_name] = round(lending_score * 100, 2)
        
        logger.info(f"Calculated lending scores for {len(lending_scores)} vendors")
        return lending_scores
    
    def get_vendor_summary(self, vendor_name: str) -> Dict[str, Any]:
        """Get comprehensive summary for a specific vendor"""
        if vendor_name not in self.analysis_results:
            raise ValueError(f"Vendor {vendor_name} not found in analysis results")
        
        metrics = self.analysis_results[vendor_name]
        
        # Calculate lending score for this vendor
        lending_scores = self.calculate_lending_scores({vendor_name: metrics})
        lending_score = lending_scores[vendor_name]
        
        # Risk assessment
        risk_level = self._assess_risk_level(metrics, lending_score)
        
        return {
            'vendor_name': vendor_name,
            'lending_score': lending_score,
            'risk_level': risk_level,
            'key_metrics': {
                'avg_views_per_post': metrics['avg_views_per_post'],
                'posting_frequency_per_week': metrics['posting_frequency_per_week'],
                'avg_price_etb': metrics['avg_price_etb'],
                'price_listing_rate': metrics['price_listing_rate']
            },
            'business_profile': {
                'total_posts': metrics['total_posts'],
                'unique_products': metrics['unique_products'],
                'unique_locations': metrics['unique_locations'],
                'price_range': f"{metrics['min_price_etb']}-{metrics['max_price_etb']} ETB"
            },
            'top_performing_post': metrics['top_performing_post'],
            'recommendations': self._generate_recommendations(metrics, lending_score)
        }
    
    def _assess_risk_level(self, metrics: Dict[str, Any], lending_score: float) -> str:
        """Assess risk level for lending"""
        if lending_score >= 70:
            return "Low Risk"
        elif lending_score >= 50:
            return "Medium Risk"
        elif lending_score >= 30:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _generate_recommendations(self, metrics: Dict[str, Any], lending_score: float) -> List[str]:
        """Generate recommendations for vendor improvement"""
        recommendations = []
        
        if metrics['posting_frequency_per_week'] < 1:
            recommendations.append("Increase posting frequency to at least 1 post per week")
        
        if metrics['price_listing_rate'] < 0.5:
            recommendations.append("Include prices in more posts to improve transparency")
        
        if metrics['avg_views_per_post'] < 100:
            recommendations.append("Improve content quality to increase engagement")
        
        if metrics['unique_products'] < 3:
            recommendations.append("Diversify product offerings to reduce risk")
        
        if lending_score < 50:
            recommendations.append("Focus on building consistent business activity before loan application")
        
        return recommendations


def main():
    """Main function for testing vendor analysis"""
    # Create sample data for testing
    sample_data = [
        {
            'channel_name': 'Test Vendor 1',
            'text': 'የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።',
            'views': 150,
            'date': '2024-01-15T10:00:00'
        },
        {
            'channel_name': 'Test Vendor 1',
            'text': 'ሴቶች ጫማ 800 ብር። መርካቶ አካባቢ።',
            'views': 200,
            'date': '2024-01-16T14:30:00'
        }
    ]
    
    # Save sample data
    sample_file = "data/sample_vendor_data.json"
    Path(sample_file).parent.mkdir(parents=True, exist_ok=True)
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # Test analyzer
    analyzer = VendorAnalyzer()
    df = analyzer.load_telegram_data(sample_file)
    df = analyzer.extract_entities_from_messages(df)
    
    vendor_metrics = analyzer.analyze_vendor_performance(df)
    lending_scores = analyzer.calculate_lending_scores(vendor_metrics)
    
    print("Vendor Analysis Results:")
    for vendor, score in lending_scores.items():
        print(f"{vendor}: {score}")
        summary = analyzer.get_vendor_summary(vendor)
        print(f"  Risk Level: {summary['risk_level']}")
        print(f"  Key Metrics: {summary['key_metrics']}")


if __name__ == "__main__":
    main()
