"""
Metrics Calculator for Vendor Performance Analysis
Calculates various business and engagement metrics for vendor assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import re

from loguru import logger


class MetricsCalculator:
    """Calculates comprehensive metrics for vendor performance analysis"""
    
    def __init__(self):
        # Setup logging
        logger.add("logs/metrics_calculation.log", rotation="1 day")
    
    def calculate_activity_metrics(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate activity and consistency metrics"""
        logger.info("Calculating activity metrics")
        
        if vendor_data.empty:
            return self._empty_metrics()
        
        # Basic counts
        total_posts = len(vendor_data)
        
        # Date range analysis
        if 'date' in vendor_data.columns:
            vendor_data['date'] = pd.to_datetime(vendor_data['date'])
            date_range = (vendor_data['date'].max() - vendor_data['date'].min()).days
            
            # Posting frequency
            posting_frequency_daily = total_posts / max(date_range, 1)
            posting_frequency_weekly = posting_frequency_daily * 7
            
            # Consistency analysis
            consistency_metrics = self._calculate_posting_consistency(vendor_data)
        else:
            date_range = 0
            posting_frequency_daily = 0
            posting_frequency_weekly = 0
            consistency_metrics = {}
        
        # Recent activity (last 30 days)
        recent_activity = self._calculate_recent_activity(vendor_data)
        
        return {
            'total_posts': total_posts,
            'date_range_days': date_range,
            'posting_frequency_daily': round(posting_frequency_daily, 3),
            'posting_frequency_weekly': round(posting_frequency_weekly, 2),
            'recent_activity_30d': recent_activity['posts_30d'],
            'recent_activity_7d': recent_activity['posts_7d'],
            'consistency_score': consistency_metrics.get('consistency_score', 0),
            'posting_pattern': consistency_metrics.get('posting_pattern', 'irregular')
        }
    
    def calculate_engagement_metrics(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market reach and engagement metrics"""
        logger.info("Calculating engagement metrics")
        
        if vendor_data.empty:
            return self._empty_engagement_metrics()
        
        # Views analysis
        views_metrics = self._calculate_views_metrics(vendor_data)
        
        # Engagement rate (if available)
        engagement_metrics = self._calculate_engagement_rates(vendor_data)
        
        # Content performance
        content_metrics = self._calculate_content_performance(vendor_data)
        
        # Top performing content
        top_posts = self._identify_top_performing_posts(vendor_data)
        
        return {
            **views_metrics,
            **engagement_metrics,
            **content_metrics,
            'top_performing_posts': top_posts
        }
    
    def calculate_business_profile_metrics(self, vendor_data: pd.DataFrame, 
                                         extracted_entities: Dict[str, List] = None) -> Dict[str, Any]:
        """Calculate business profile metrics from extracted entities"""
        logger.info("Calculating business profile metrics")
        
        if vendor_data.empty:
            return self._empty_business_metrics()
        
        # Price analysis
        price_metrics = self._calculate_price_metrics(vendor_data, extracted_entities)
        
        # Product diversity
        product_metrics = self._calculate_product_metrics(vendor_data, extracted_entities)
        
        # Location coverage
        location_metrics = self._calculate_location_metrics(vendor_data, extracted_entities)
        
        # Business maturity indicators
        maturity_metrics = self._calculate_business_maturity(vendor_data)
        
        return {
            **price_metrics,
            **product_metrics,
            **location_metrics,
            **maturity_metrics
        }
    
    def _calculate_posting_consistency(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate posting consistency metrics"""
        if 'date' not in vendor_data.columns or vendor_data.empty:
            return {'consistency_score': 0, 'posting_pattern': 'irregular'}
        
        # Group by day and count posts
        vendor_data['date_only'] = vendor_data['date'].dt.date
        daily_posts = vendor_data.groupby('date_only').size()
        
        # Calculate consistency score (lower variance = higher consistency)
        if len(daily_posts) > 1:
            consistency_score = 1 / (1 + daily_posts.var())
        else:
            consistency_score = 0.5
        
        # Determine posting pattern
        avg_posts_per_active_day = daily_posts.mean()
        if avg_posts_per_active_day >= 2:
            pattern = 'high_frequency'
        elif avg_posts_per_active_day >= 1:
            pattern = 'regular'
        elif avg_posts_per_active_day >= 0.5:
            pattern = 'moderate'
        else:
            pattern = 'irregular'
        
        return {
            'consistency_score': round(consistency_score, 3),
            'posting_pattern': pattern,
            'avg_posts_per_active_day': round(avg_posts_per_active_day, 2),
            'active_days': len(daily_posts)
        }
    
    def _calculate_recent_activity(self, vendor_data: pd.DataFrame) -> Dict[str, int]:
        """Calculate recent activity metrics"""
        if 'date' not in vendor_data.columns:
            return {'posts_30d': 0, 'posts_7d': 0}
        
        now = datetime.now()
        cutoff_30d = now - timedelta(days=30)
        cutoff_7d = now - timedelta(days=7)
        
        vendor_data['date'] = pd.to_datetime(vendor_data['date'])
        
        posts_30d = len(vendor_data[vendor_data['date'] > cutoff_30d])
        posts_7d = len(vendor_data[vendor_data['date'] > cutoff_7d])
        
        return {'posts_30d': posts_30d, 'posts_7d': posts_7d}
    
    def _calculate_views_metrics(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate views-related metrics"""
        if 'views' not in vendor_data.columns:
            return {
                'avg_views_per_post': 0,
                'max_views': 0,
                'min_views': 0,
                'total_views': 0,
                'views_std': 0
            }
        
        views = vendor_data['views'].fillna(0)
        
        return {
            'avg_views_per_post': round(views.mean(), 2),
            'max_views': int(views.max()),
            'min_views': int(views.min()),
            'total_views': int(views.sum()),
            'views_std': round(views.std(), 2),
            'median_views': round(views.median(), 2)
        }
    
    def _calculate_engagement_rates(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate engagement rates if data is available"""
        engagement_metrics = {}
        
        # Forward rate (if forwards column exists)
        if 'forwards' in vendor_data.columns and 'views' in vendor_data.columns:
            forwards = vendor_data['forwards'].fillna(0)
            views = vendor_data['views'].fillna(1)  # Avoid division by zero
            forward_rate = (forwards / views).mean()
            engagement_metrics['avg_forward_rate'] = round(forward_rate, 4)
        
        # Reply rate (if replies column exists)
        if 'replies' in vendor_data.columns and 'views' in vendor_data.columns:
            replies = vendor_data['replies'].fillna(0)
            views = vendor_data['views'].fillna(1)
            reply_rate = (replies / views).mean()
            engagement_metrics['avg_reply_rate'] = round(reply_rate, 4)
        
        return engagement_metrics
    
    def _calculate_content_performance(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate content performance metrics"""
        if vendor_data.empty:
            return {}
        
        # Text length analysis
        if 'text' in vendor_data.columns:
            text_lengths = vendor_data['text'].str.len()
            avg_text_length = text_lengths.mean()
            
            # Correlation between text length and views
            if 'views' in vendor_data.columns:
                correlation = text_lengths.corr(vendor_data['views'])
            else:
                correlation = 0
        else:
            avg_text_length = 0
            correlation = 0
        
        # Media usage
        media_usage = 0
        if 'has_media' in vendor_data.columns:
            media_usage = vendor_data['has_media'].mean()
        
        return {
            'avg_text_length': round(avg_text_length, 1),
            'text_length_views_correlation': round(correlation, 3),
            'media_usage_rate': round(media_usage, 3)
        }
    
    def _identify_top_performing_posts(self, vendor_data: pd.DataFrame, top_n: int = 3) -> List[Dict[str, Any]]:
        """Identify top performing posts"""
        if 'views' not in vendor_data.columns or vendor_data.empty:
            return []
        
        # Sort by views and get top N
        top_posts = vendor_data.nlargest(top_n, 'views')
        
        result = []
        for _, post in top_posts.iterrows():
            post_info = {
                'text': post.get('text', '')[:100] + '...' if len(post.get('text', '')) > 100 else post.get('text', ''),
                'views': int(post.get('views', 0)),
                'date': post.get('date').isoformat() if pd.notna(post.get('date')) else None
            }
            result.append(post_info)
        
        return result
    
    def _calculate_price_metrics(self, vendor_data: pd.DataFrame, 
                               extracted_entities: Dict[str, List] = None) -> Dict[str, Any]:
        """Calculate price-related metrics"""
        prices = []
        
        # Extract prices from entities or fallback to regex
        if extracted_entities and 'prices' in extracted_entities:
            for price_list in extracted_entities['prices']:
                if price_list:
                    for price_str in price_list:
                        price_num = self._extract_price_number(price_str)
                        if price_num:
                            prices.append(price_num)
        else:
            # Fallback: extract from text using regex
            if 'text' in vendor_data.columns:
                price_pattern = r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:ብር|birr|ETB)'
                for text in vendor_data['text'].fillna(''):
                    matches = re.findall(price_pattern, text, re.IGNORECASE)
                    for match in matches:
                        try:
                            price_num = float(match.replace(',', ''))
                            prices.append(price_num)
                        except ValueError:
                            continue
        
        if not prices:
            return {
                'avg_price_etb': 0,
                'min_price_etb': 0,
                'max_price_etb': 0,
                'price_range_etb': 0,
                'price_std_etb': 0,
                'total_priced_items': 0,
                'price_listing_rate': 0
            }
        
        # Calculate price statistics
        prices_array = np.array(prices)
        total_posts = len(vendor_data)
        posts_with_prices = len([p for p in prices if p > 0])
        
        return {
            'avg_price_etb': round(prices_array.mean(), 2),
            'min_price_etb': round(prices_array.min(), 2),
            'max_price_etb': round(prices_array.max(), 2),
            'price_range_etb': round(prices_array.max() - prices_array.min(), 2),
            'price_std_etb': round(prices_array.std(), 2),
            'median_price_etb': round(np.median(prices_array), 2),
            'total_priced_items': len(prices),
            'price_listing_rate': round(posts_with_prices / total_posts, 3) if total_posts > 0 else 0
        }
    
    def _calculate_product_metrics(self, vendor_data: pd.DataFrame, 
                                 extracted_entities: Dict[str, List] = None) -> Dict[str, Any]:
        """Calculate product diversity metrics"""
        products = []
        
        # Extract products from entities
        if extracted_entities and 'products' in extracted_entities:
            for product_list in extracted_entities['products']:
                if product_list:
                    products.extend(product_list)
        
        # Count unique products
        unique_products = len(set(products)) if products else 0
        total_product_mentions = len(products)
        
        # Product diversity score (0-1)
        diversity_score = min(unique_products / 10, 1.0)  # Normalize to max 10 products
        
        return {
            'total_product_mentions': total_product_mentions,
            'unique_products': unique_products,
            'product_diversity_score': round(diversity_score, 3),
            'avg_products_per_post': round(total_product_mentions / len(vendor_data), 2) if len(vendor_data) > 0 else 0
        }
    
    def _calculate_location_metrics(self, vendor_data: pd.DataFrame, 
                                  extracted_entities: Dict[str, List] = None) -> Dict[str, Any]:
        """Calculate location coverage metrics"""
        locations = []
        
        # Extract locations from entities
        if extracted_entities and 'locations' in extracted_entities:
            for location_list in extracted_entities['locations']:
                if location_list:
                    locations.extend(location_list)
        
        # Count unique locations
        unique_locations = len(set(locations)) if locations else 0
        total_location_mentions = len(locations)
        
        # Location coverage score
        coverage_score = min(unique_locations / 5, 1.0)  # Normalize to max 5 locations
        
        return {
            'total_location_mentions': total_location_mentions,
            'unique_locations': unique_locations,
            'location_coverage_score': round(coverage_score, 3),
            'location_mention_rate': round(total_location_mentions / len(vendor_data), 2) if len(vendor_data) > 0 else 0
        }
    
    def _calculate_business_maturity(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate business maturity indicators"""
        if vendor_data.empty:
            return {'business_maturity_score': 0}
        
        # Factors indicating business maturity
        maturity_factors = []
        
        # Consistent posting (already calculated)
        # Professional language use (simplified check)
        if 'text' in vendor_data.columns:
            # Check for business-like language patterns
            business_keywords = ['ዋጋ', 'ብር', 'አካባቢ', 'ይገኛል', 'ይሸጣል']
            business_language_score = 0
            
            for text in vendor_data['text'].fillna(''):
                keyword_count = sum(1 for keyword in business_keywords if keyword in text)
                business_language_score += keyword_count
            
            avg_business_language = business_language_score / len(vendor_data)
            maturity_factors.append(min(avg_business_language / 3, 1.0))
        
        # Price transparency
        if 'price_listing_rate' in vendor_data.columns:
            maturity_factors.append(vendor_data['price_listing_rate'].iloc[0] if len(vendor_data) > 0 else 0)
        
        # Calculate overall maturity score
        maturity_score = np.mean(maturity_factors) if maturity_factors else 0
        
        return {
            'business_maturity_score': round(maturity_score, 3),
            'maturity_factors_count': len(maturity_factors)
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
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty activity metrics"""
        return {
            'total_posts': 0,
            'date_range_days': 0,
            'posting_frequency_daily': 0,
            'posting_frequency_weekly': 0,
            'recent_activity_30d': 0,
            'recent_activity_7d': 0,
            'consistency_score': 0,
            'posting_pattern': 'no_data'
        }
    
    def _empty_engagement_metrics(self) -> Dict[str, Any]:
        """Return empty engagement metrics"""
        return {
            'avg_views_per_post': 0,
            'max_views': 0,
            'min_views': 0,
            'total_views': 0,
            'views_std': 0,
            'median_views': 0,
            'top_performing_posts': []
        }
    
    def _empty_business_metrics(self) -> Dict[str, Any]:
        """Return empty business metrics"""
        return {
            'avg_price_etb': 0,
            'min_price_etb': 0,
            'max_price_etb': 0,
            'price_range_etb': 0,
            'total_priced_items': 0,
            'price_listing_rate': 0,
            'unique_products': 0,
            'unique_locations': 0,
            'business_maturity_score': 0
        }
    
    def calculate_comprehensive_metrics(self, vendor_data: pd.DataFrame, 
                                      extracted_entities: Dict[str, List] = None) -> Dict[str, Any]:
        """Calculate all metrics for a vendor"""
        logger.info("Calculating comprehensive metrics")
        
        activity_metrics = self.calculate_activity_metrics(vendor_data)
        engagement_metrics = self.calculate_engagement_metrics(vendor_data)
        business_metrics = self.calculate_business_profile_metrics(vendor_data, extracted_entities)
        
        # Combine all metrics
        comprehensive_metrics = {
            **activity_metrics,
            **engagement_metrics,
            **business_metrics
        }
        
        logger.info("Comprehensive metrics calculation completed")
        return comprehensive_metrics


def main():
    """Main function for testing metrics calculation"""
    # Create sample data
    sample_data = pd.DataFrame([
        {
            'text': 'የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።',
            'views': 150,
            'date': '2024-01-15T10:00:00',
            'has_media': False
        },
        {
            'text': 'ሴቶች ጫማ 800 ብር። መርካቶ አካባቢ።',
            'views': 200,
            'date': '2024-01-16T14:30:00',
            'has_media': True
        }
    ])
    
    sample_data['date'] = pd.to_datetime(sample_data['date'])
    
    # Test metrics calculation
    calculator = MetricsCalculator()
    
    activity_metrics = calculator.calculate_activity_metrics(sample_data)
    engagement_metrics = calculator.calculate_engagement_metrics(sample_data)
    business_metrics = calculator.calculate_business_profile_metrics(sample_data)
    
    print("Activity Metrics:", activity_metrics)
    print("Engagement Metrics:", engagement_metrics)
    print("Business Metrics:", business_metrics)
    
    # Test comprehensive metrics
    comprehensive = calculator.calculate_comprehensive_metrics(sample_data)
    print("Comprehensive Metrics:", comprehensive)


if __name__ == "__main__":
    main()
