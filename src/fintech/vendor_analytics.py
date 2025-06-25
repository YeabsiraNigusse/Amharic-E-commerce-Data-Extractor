"""
Vendor Analytics Engine for EthioMart FinTech Analysis
Processes vendor data to calculate key performance metrics for micro-lending
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import re

from loguru import logger

class VendorAnalyticsEngine:
    """Analytics engine for vendor performance analysis"""
    
    def __init__(self, output_dir: str = "vendor_analytics"):
        """Initialize the vendor analytics engine"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vendor data storage
        self.vendor_data = {}
        self.processed_posts = []
        
        # Analytics results
        self.vendor_metrics = {}
        
        logger.info(f"Vendor analytics engine initialized: {output_dir}")
    
    def load_vendor_data(self, data_file: str) -> bool:
        """Load vendor data from file"""
        
        try:
            if data_file.endswith('.json'):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif data_file.endswith('.csv'):
                data = pd.read_csv(data_file).to_dict('records')
            else:
                logger.error(f"Unsupported file format: {data_file}")
                return False
            
            # Process and organize data by vendor
            self._organize_vendor_data(data)
            
            logger.info(f"Loaded data for {len(self.vendor_data)} vendors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vendor data: {e}")
            return False
    
    def _organize_vendor_data(self, data: List[Dict[str, Any]]):
        """Organize data by vendor/channel"""
        
        for post in data:
            # Extract vendor identifier (channel name or username)
            vendor_id = self._extract_vendor_id(post)
            
            if vendor_id not in self.vendor_data:
                self.vendor_data[vendor_id] = {
                    'posts': [],
                    'channel_info': {},
                    'total_posts': 0,
                    'date_range': {'start': None, 'end': None}
                }
            
            # Add post to vendor data
            self.vendor_data[vendor_id]['posts'].append(post)
            self.vendor_data[vendor_id]['total_posts'] += 1
            
            # Update date range
            post_date = self._extract_post_date(post)
            if post_date:
                if not self.vendor_data[vendor_id]['date_range']['start']:
                    self.vendor_data[vendor_id]['date_range']['start'] = post_date
                    self.vendor_data[vendor_id]['date_range']['end'] = post_date
                else:
                    if post_date < self.vendor_data[vendor_id]['date_range']['start']:
                        self.vendor_data[vendor_id]['date_range']['start'] = post_date
                    if post_date > self.vendor_data[vendor_id]['date_range']['end']:
                        self.vendor_data[vendor_id]['date_range']['end'] = post_date
    
    def _extract_vendor_id(self, post: Dict[str, Any]) -> str:
        """Extract vendor identifier from post"""
        
        # Try different fields for vendor identification
        vendor_id = (
            post.get('channel_username') or 
            post.get('channel_name') or 
            post.get('sender_username') or 
            post.get('vendor_id') or 
            f"vendor_{hash(str(post.get('text', '')))[:8]}"
        )
        
        return str(vendor_id).replace('@', '').lower()
    
    def _extract_post_date(self, post: Dict[str, Any]) -> Optional[datetime]:
        """Extract post date from post data"""
        
        date_fields = ['timestamp', 'date', 'created_at', 'post_date']
        
        for field in date_fields:
            if field in post and post[field]:
                try:
                    if isinstance(post[field], str):
                        # Try different date formats
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                            try:
                                return datetime.strptime(post[field], fmt)
                            except ValueError:
                                continue
                    elif isinstance(post[field], (int, float)):
                        # Assume Unix timestamp
                        return datetime.fromtimestamp(post[field])
                except Exception:
                    continue
        
        return None
    
    def calculate_vendor_metrics(self, vendor_id: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a vendor"""
        
        if vendor_id not in self.vendor_data:
            logger.error(f"Vendor {vendor_id} not found")
            return {}
        
        vendor_posts = self.vendor_data[vendor_id]['posts']
        date_range = self.vendor_data[vendor_id]['date_range']
        
        metrics = {
            'vendor_id': vendor_id,
            'activity_metrics': self._calculate_activity_metrics(vendor_posts, date_range),
            'engagement_metrics': self._calculate_engagement_metrics(vendor_posts),
            'business_metrics': self._calculate_business_metrics(vendor_posts),
            'consistency_metrics': self._calculate_consistency_metrics(vendor_posts, date_range),
            'risk_indicators': self._calculate_risk_indicators(vendor_posts)
        }
        
        return metrics
    
    def _calculate_activity_metrics(self, posts: List[Dict[str, Any]], 
                                  date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Calculate activity and posting frequency metrics"""
        
        total_posts = len(posts)
        
        # Calculate time span
        start_date = date_range.get('start')
        end_date = date_range.get('end')
        
        if start_date and end_date:
            time_span_days = (end_date - start_date).days + 1
            time_span_weeks = time_span_days / 7
        else:
            time_span_days = 30  # Default assumption
            time_span_weeks = time_span_days / 7
        
        # Posting frequency
        posts_per_day = total_posts / max(time_span_days, 1)
        posts_per_week = total_posts / max(time_span_weeks, 1)
        
        # Activity consistency (posts per week variance)
        weekly_posts = self._group_posts_by_week(posts)
        weekly_counts = [len(week_posts) for week_posts in weekly_posts.values()]
        activity_consistency = 1 - (np.std(weekly_counts) / max(np.mean(weekly_counts), 1))
        
        return {
            'total_posts': total_posts,
            'time_span_days': time_span_days,
            'posts_per_day': posts_per_day,
            'posts_per_week': posts_per_week,
            'activity_consistency': max(0, activity_consistency),
            'active_weeks': len(weekly_posts),
            'posting_frequency_score': min(posts_per_week / 5, 1.0)  # Normalized to 5 posts/week
        }
    
    def _calculate_engagement_metrics(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate engagement and reach metrics"""
        
        view_counts = []
        like_counts = []
        comment_counts = []
        share_counts = []
        
        for post in posts:
            # Extract engagement metrics
            views = self._extract_numeric_value(post, ['views', 'view_count', 'reach'])
            likes = self._extract_numeric_value(post, ['likes', 'like_count', 'reactions'])
            comments = self._extract_numeric_value(post, ['comments', 'comment_count', 'replies'])
            shares = self._extract_numeric_value(post, ['shares', 'share_count', 'forwards'])
            
            if views is not None:
                view_counts.append(views)
            if likes is not None:
                like_counts.append(likes)
            if comments is not None:
                comment_counts.append(comments)
            if shares is not None:
                share_counts.append(shares)
        
        # Calculate averages and find top performing post
        avg_views = np.mean(view_counts) if view_counts else 0
        avg_likes = np.mean(like_counts) if like_counts else 0
        avg_comments = np.mean(comment_counts) if comment_counts else 0
        avg_shares = np.mean(share_counts) if share_counts else 0
        
        # Find top performing post
        top_post = self._find_top_performing_post(posts)
        
        # Calculate engagement rate (likes + comments + shares) / views
        total_engagement = sum(like_counts) + sum(comment_counts) + sum(share_counts)
        total_views = sum(view_counts) if view_counts else 1
        engagement_rate = total_engagement / total_views
        
        return {
            'average_views_per_post': avg_views,
            'average_likes_per_post': avg_likes,
            'average_comments_per_post': avg_comments,
            'average_shares_per_post': avg_shares,
            'total_views': sum(view_counts),
            'engagement_rate': engagement_rate,
            'top_performing_post': top_post,
            'posts_with_views': len(view_counts),
            'market_reach_score': min(avg_views / 1000, 1.0)  # Normalized to 1000 views
        }
    
    def _calculate_business_metrics(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate business profile metrics using NER extracted entities"""
        
        prices = []
        products = []
        locations = []
        contact_info = []
        
        for post in posts:
            # Extract entities from NER results or text analysis
            entities = self._extract_entities_from_post(post)
            
            # Collect prices
            post_prices = entities.get('prices', [])
            prices.extend(post_prices)
            
            # Collect products
            post_products = entities.get('products', [])
            products.extend(post_products)
            
            # Collect locations
            post_locations = entities.get('locations', [])
            locations.extend(post_locations)
            
            # Collect contact info
            post_contacts = entities.get('contact_info', [])
            contact_info.extend(post_contacts)
        
        # Calculate price statistics
        if prices:
            avg_price = np.mean(prices)
            median_price = np.median(prices)
            price_range = max(prices) - min(prices)
            price_std = np.std(prices)
        else:
            avg_price = median_price = price_range = price_std = 0
        
        # Business profile classification
        business_profile = self._classify_business_profile(avg_price, len(products), len(posts))
        
        return {
            'average_price_etb': avg_price,
            'median_price_etb': median_price,
            'price_range_etb': price_range,
            'price_volatility': price_std / max(avg_price, 1),
            'total_products_mentioned': len(set(products)),
            'total_locations_mentioned': len(set(locations)),
            'contact_methods_count': len(set(contact_info)),
            'posts_with_prices': len([p for p in posts if self._extract_entities_from_post(p).get('prices')]),
            'business_profile': business_profile,
            'price_point_score': self._calculate_price_point_score(avg_price)
        }
    
    def _calculate_consistency_metrics(self, posts: List[Dict[str, Any]], 
                                     date_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Calculate consistency and reliability metrics"""
        
        # Posting time consistency
        posting_hours = []
        posting_days = []
        
        for post in posts:
            post_date = self._extract_post_date(post)
            if post_date:
                posting_hours.append(post_date.hour)
                posting_days.append(post_date.weekday())
        
        # Calculate consistency scores
        hour_consistency = self._calculate_time_consistency(posting_hours)
        day_consistency = self._calculate_time_consistency(posting_days)
        
        # Content consistency (similar products/prices)
        content_consistency = self._calculate_content_consistency(posts)
        
        return {
            'posting_hour_consistency': hour_consistency,
            'posting_day_consistency': day_consistency,
            'content_consistency': content_consistency,
            'overall_consistency': (hour_consistency + day_consistency + content_consistency) / 3
        }
    
    def _calculate_risk_indicators(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk indicators for lending assessment"""
        
        risk_factors = {
            'low_engagement_posts': 0,
            'price_inconsistency': 0,
            'irregular_posting': 0,
            'limited_contact_info': 0,
            'suspicious_content': 0
        }
        
        # Analyze each post for risk factors
        for post in posts:
            # Low engagement
            views = self._extract_numeric_value(post, ['views', 'view_count'])
            if views is not None and views < 10:
                risk_factors['low_engagement_posts'] += 1
            
            # Check for suspicious content patterns
            text = post.get('text', '').lower()
            suspicious_keywords = ['urgent', 'limited time', 'act now', 'guaranteed']
            if any(keyword in text for keyword in suspicious_keywords):
                risk_factors['suspicious_content'] += 1
        
        # Calculate overall risk score (lower is better)
        total_posts = len(posts)
        risk_score = sum(risk_factors.values()) / max(total_posts, 1)
        
        return {
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'risk_level': 'Low' if risk_score < 0.2 else 'Medium' if risk_score < 0.5 else 'High'
        }
    
    def _group_posts_by_week(self, posts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group posts by week"""
        
        weekly_posts = defaultdict(list)
        
        for post in posts:
            post_date = self._extract_post_date(post)
            if post_date:
                # Get week number
                week_key = f"{post_date.year}-W{post_date.isocalendar()[1]}"
                weekly_posts[week_key].append(post)
        
        return dict(weekly_posts)
    
    def _extract_numeric_value(self, post: Dict[str, Any], fields: List[str]) -> Optional[float]:
        """Extract numeric value from post fields"""
        
        for field in fields:
            if field in post and post[field] is not None:
                try:
                    return float(post[field])
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _find_top_performing_post(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the top performing post by views"""
        
        top_post = None
        max_views = 0
        
        for post in posts:
            views = self._extract_numeric_value(post, ['views', 'view_count'])
            if views and views > max_views:
                max_views = views
                top_post = {
                    'text': post.get('text', '')[:100] + '...',
                    'views': views,
                    'entities': self._extract_entities_from_post(post)
                }
        
        return top_post or {'text': 'No post data', 'views': 0, 'entities': {}}
    
    def _extract_entities_from_post(self, post: Dict[str, Any]) -> Dict[str, List]:
        """Extract entities from post using NER results or pattern matching"""
        
        entities = {
            'prices': [],
            'products': [],
            'locations': [],
            'contact_info': []
        }
        
        text = post.get('text', '')
        
        # Extract prices using regex patterns
        price_patterns = [
            r'(\d+)\s*ብር',
            r'ETB\s*(\d+)',
            r'ዋጋ\s*(\d+)',
            r'በ\s*(\d+)\s*ብር'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    entities['prices'].append(float(match))
                except ValueError:
                    continue
        
        # Extract locations
        location_keywords = ['አዲስ አበባ', 'ቦሌ', 'ፒያሳ', 'መርካቶ', 'ካዛንቺስ', 'ጎፋ', 'ኪርኮስ', 'ላፍቶ']
        for location in location_keywords:
            if location in text:
                entities['locations'].append(location)
        
        # Extract contact info
        phone_pattern = r'09\d{8}'
        username_pattern = r'@\w+'
        
        phones = re.findall(phone_pattern, text)
        usernames = re.findall(username_pattern, text)
        
        entities['contact_info'].extend(phones)
        entities['contact_info'].extend(usernames)
        
        # Extract products (simplified)
        product_keywords = ['ልብስ', 'ጫማ', 'ስልክ', 'ላፕቶፕ', 'ሸሚዝ', 'ቦርሳ', 'ሱሪ', 'ቀሚስ']
        for product in product_keywords:
            if product in text:
                entities['products'].append(product)
        
        return entities
    
    def _classify_business_profile(self, avg_price: float, product_count: int, post_count: int) -> str:
        """Classify business profile based on metrics"""
        
        if avg_price > 5000:
            return "High-value/Low-volume"
        elif avg_price > 1000:
            return "Medium-value/Medium-volume"
        else:
            return "Low-value/High-volume"
    
    def _calculate_price_point_score(self, avg_price: float) -> float:
        """Calculate price point score (normalized)"""
        
        # Normalize price to 0-1 scale (assuming max reasonable price is 50000 ETB)
        return min(avg_price / 50000, 1.0)
    
    def _calculate_time_consistency(self, time_values: List[int]) -> float:
        """Calculate consistency score for time-based values"""
        
        if not time_values:
            return 0.0
        
        # Calculate coefficient of variation (lower = more consistent)
        mean_val = np.mean(time_values)
        std_val = np.std(time_values)
        
        if mean_val == 0:
            return 1.0
        
        cv = std_val / mean_val
        return max(0, 1 - cv)  # Convert to consistency score
    
    def _calculate_content_consistency(self, posts: List[Dict[str, Any]]) -> float:
        """Calculate content consistency score"""
        
        # Simplified content consistency based on entity overlap
        all_entities = []
        
        for post in posts:
            entities = self._extract_entities_from_post(post)
            post_entities = (
                entities.get('products', []) + 
                entities.get('locations', [])
            )
            all_entities.append(set(post_entities))
        
        if len(all_entities) < 2:
            return 1.0
        
        # Calculate average pairwise overlap
        overlaps = []
        for i in range(len(all_entities)):
            for j in range(i + 1, len(all_entities)):
                if all_entities[i] or all_entities[j]:
                    overlap = len(all_entities[i] & all_entities[j]) / len(all_entities[i] | all_entities[j])
                    overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def process_all_vendors(self) -> Dict[str, Dict[str, Any]]:
        """Process all vendors and calculate metrics"""
        
        logger.info(f"Processing {len(self.vendor_data)} vendors")
        
        for vendor_id in self.vendor_data.keys():
            self.vendor_metrics[vendor_id] = self.calculate_vendor_metrics(vendor_id)
            logger.info(f"Processed vendor: {vendor_id}")
        
        return self.vendor_metrics
    
    def get_vendor_summary(self, vendor_id: str) -> Dict[str, Any]:
        """Get summary metrics for a vendor"""
        
        if vendor_id not in self.vendor_metrics:
            return {}
        
        metrics = self.vendor_metrics[vendor_id]
        
        return {
            'vendor_id': vendor_id,
            'posts_per_week': metrics['activity_metrics']['posts_per_week'],
            'average_views_per_post': metrics['engagement_metrics']['average_views_per_post'],
            'average_price_etb': metrics['business_metrics']['average_price_etb'],
            'business_profile': metrics['business_metrics']['business_profile'],
            'risk_level': metrics['risk_indicators']['risk_level'],
            'overall_consistency': metrics['consistency_metrics']['overall_consistency']
        }

def main():
    """Test the vendor analytics engine"""
    engine = VendorAnalyticsEngine()
    print("Vendor analytics engine initialized successfully!")

if __name__ == "__main__":
    main()
