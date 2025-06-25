"""
Lending Scorer for EthioMart Micro-Lending Assessment
Calculates lending scores based on vendor analytics and business metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

class LendingScorer:
    """Scoring system for micro-lending assessment"""
    
    def __init__(self, scoring_config: Dict[str, Any] = None):
        """Initialize the lending scorer with configuration"""
        
        # Default scoring configuration
        self.config = scoring_config or {
            'weights': {
                'activity_score': 0.25,
                'engagement_score': 0.30,
                'business_score': 0.25,
                'consistency_score': 0.15,
                'risk_adjustment': -0.05
            },
            'thresholds': {
                'min_posts_per_week': 1.0,
                'min_average_views': 50,
                'min_price_consistency': 0.3,
                'max_risk_score': 0.7
            },
            'scoring_ranges': {
                'excellent': (0.8, 1.0),
                'good': (0.6, 0.8),
                'fair': (0.4, 0.6),
                'poor': (0.0, 0.4)
            }
        }
        
        logger.info("Lending scorer initialized")
    
    def calculate_lending_score(self, vendor_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive lending score for a vendor"""
        
        vendor_id = vendor_metrics.get('vendor_id', 'unknown')
        logger.info(f"Calculating lending score for vendor: {vendor_id}")
        
        # Extract metric categories
        activity_metrics = vendor_metrics.get('activity_metrics', {})
        engagement_metrics = vendor_metrics.get('engagement_metrics', {})
        business_metrics = vendor_metrics.get('business_metrics', {})
        consistency_metrics = vendor_metrics.get('consistency_metrics', {})
        risk_indicators = vendor_metrics.get('risk_indicators', {})
        
        # Calculate component scores
        activity_score = self._calculate_activity_score(activity_metrics)
        engagement_score = self._calculate_engagement_score(engagement_metrics)
        business_score = self._calculate_business_score(business_metrics)
        consistency_score = self._calculate_consistency_score(consistency_metrics)
        risk_adjustment = self._calculate_risk_adjustment(risk_indicators)
        
        # Calculate weighted final score
        weights = self.config['weights']
        final_score = (
            activity_score * weights['activity_score'] +
            engagement_score * weights['engagement_score'] +
            business_score * weights['business_score'] +
            consistency_score * weights['consistency_score'] +
            risk_adjustment * weights['risk_adjustment']
        )
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, final_score))
        
        # Determine score category
        score_category = self._categorize_score(final_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            vendor_metrics, activity_score, engagement_score, 
            business_score, consistency_score, risk_adjustment
        )
        
        # Calculate loan amount recommendation
        loan_recommendation = self._calculate_loan_recommendation(final_score, business_metrics)
        
        return {
            'vendor_id': vendor_id,
            'final_lending_score': final_score,
            'score_category': score_category,
            'component_scores': {
                'activity_score': activity_score,
                'engagement_score': engagement_score,
                'business_score': business_score,
                'consistency_score': consistency_score,
                'risk_adjustment': risk_adjustment
            },
            'loan_recommendation': loan_recommendation,
            'recommendations': recommendations,
            'eligibility': self._assess_eligibility(final_score, vendor_metrics),
            'scoring_details': self._get_scoring_details(vendor_metrics)
        }
    
    def _calculate_activity_score(self, activity_metrics: Dict[str, Any]) -> float:
        """Calculate activity score based on posting frequency and consistency"""
        
        posts_per_week = activity_metrics.get('posts_per_week', 0)
        activity_consistency = activity_metrics.get('activity_consistency', 0)
        posting_frequency_score = activity_metrics.get('posting_frequency_score', 0)
        
        # Normalize posting frequency (target: 3-5 posts per week)
        frequency_score = min(posts_per_week / 5.0, 1.0)
        
        # Combine frequency and consistency
        activity_score = (frequency_score * 0.7) + (activity_consistency * 0.3)
        
        return min(activity_score, 1.0)
    
    def _calculate_engagement_score(self, engagement_metrics: Dict[str, Any]) -> float:
        """Calculate engagement score based on views and interactions"""
        
        avg_views = engagement_metrics.get('average_views_per_post', 0)
        engagement_rate = engagement_metrics.get('engagement_rate', 0)
        market_reach_score = engagement_metrics.get('market_reach_score', 0)
        
        # Normalize views (target: 500+ views per post)
        views_score = min(avg_views / 500.0, 1.0)
        
        # Normalize engagement rate (target: 5% engagement rate)
        engagement_rate_score = min(engagement_rate / 0.05, 1.0)
        
        # Combine metrics
        engagement_score = (views_score * 0.6) + (engagement_rate_score * 0.3) + (market_reach_score * 0.1)
        
        return min(engagement_score, 1.0)
    
    def _calculate_business_score(self, business_metrics: Dict[str, Any]) -> float:
        """Calculate business score based on pricing and product diversity"""
        
        avg_price = business_metrics.get('average_price_etb', 0)
        price_volatility = business_metrics.get('price_volatility', 1)
        products_count = business_metrics.get('total_products_mentioned', 0)
        posts_with_prices = business_metrics.get('posts_with_prices', 0)
        
        # Price consistency score (lower volatility is better)
        price_consistency_score = max(0, 1 - price_volatility)
        
        # Product diversity score
        diversity_score = min(products_count / 10.0, 1.0)
        
        # Price transparency score
        transparency_score = posts_with_prices / max(business_metrics.get('total_posts', 1), 1)
        
        # Price point score (moderate prices are preferred for micro-lending)
        if 100 <= avg_price <= 5000:
            price_point_score = 1.0
        elif avg_price < 100:
            price_point_score = avg_price / 100.0
        else:
            price_point_score = max(0.3, 5000 / avg_price)
        
        # Combine business metrics
        business_score = (
            price_consistency_score * 0.3 +
            diversity_score * 0.2 +
            transparency_score * 0.3 +
            price_point_score * 0.2
        )
        
        return min(business_score, 1.0)
    
    def _calculate_consistency_score(self, consistency_metrics: Dict[str, Any]) -> float:
        """Calculate consistency score"""
        
        overall_consistency = consistency_metrics.get('overall_consistency', 0)
        return min(overall_consistency, 1.0)
    
    def _calculate_risk_adjustment(self, risk_indicators: Dict[str, Any]) -> float:
        """Calculate risk adjustment (negative impact on score)"""
        
        risk_score = risk_indicators.get('risk_score', 0)
        
        # Convert risk score to adjustment (higher risk = more negative adjustment)
        risk_adjustment = -risk_score
        
        return max(-0.3, risk_adjustment)  # Cap negative adjustment at -0.3
    
    def _categorize_score(self, score: float) -> str:
        """Categorize the lending score"""
        
        ranges = self.config['scoring_ranges']
        
        for category, (min_score, max_score) in ranges.items():
            if min_score <= score <= max_score:
                return category
        
        return 'poor'
    
    def _generate_recommendations(self, vendor_metrics: Dict[str, Any],
                                activity_score: float, engagement_score: float,
                                business_score: float, consistency_score: float,
                                risk_adjustment: float) -> List[str]:
        """Generate recommendations for improvement"""
        
        recommendations = []
        
        # Activity recommendations
        if activity_score < 0.6:
            posts_per_week = vendor_metrics.get('activity_metrics', {}).get('posts_per_week', 0)
            if posts_per_week < 2:
                recommendations.append("Increase posting frequency to at least 2-3 posts per week")
            recommendations.append("Maintain more consistent posting schedule")
        
        # Engagement recommendations
        if engagement_score < 0.6:
            avg_views = vendor_metrics.get('engagement_metrics', {}).get('average_views_per_post', 0)
            if avg_views < 100:
                recommendations.append("Improve content quality to increase views and engagement")
            recommendations.append("Use more engaging product descriptions and images")
        
        # Business recommendations
        if business_score < 0.6:
            price_volatility = vendor_metrics.get('business_metrics', {}).get('price_volatility', 0)
            if price_volatility > 0.5:
                recommendations.append("Maintain more consistent pricing strategy")
            recommendations.append("Include clear pricing information in all product posts")
        
        # Consistency recommendations
        if consistency_score < 0.6:
            recommendations.append("Establish regular posting schedule and maintain consistent business hours")
        
        # Risk recommendations
        if risk_adjustment < -0.1:
            recommendations.append("Address risk factors: improve content quality and engagement")
        
        return recommendations
    
    def _calculate_loan_recommendation(self, score: float, business_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate recommended loan amount and terms"""
        
        avg_price = business_metrics.get('average_price_etb', 1000)
        
        # Base loan amount calculation
        if score >= 0.8:
            loan_multiplier = 20  # Excellent: 20x average product price
            interest_rate = 0.12  # 12% annual
            term_months = 12
        elif score >= 0.6:
            loan_multiplier = 15  # Good: 15x average product price
            interest_rate = 0.15  # 15% annual
            term_months = 9
        elif score >= 0.4:
            loan_multiplier = 10  # Fair: 10x average product price
            interest_rate = 0.18  # 18% annual
            term_months = 6
        else:
            loan_multiplier = 0   # Poor: Not recommended
            interest_rate = 0.25  # 25% annual (if approved)
            term_months = 3
        
        recommended_amount = avg_price * loan_multiplier
        
        # Cap loan amount
        max_loan = 100000  # 100,000 ETB maximum
        min_loan = 5000    # 5,000 ETB minimum
        
        if recommended_amount > max_loan:
            recommended_amount = max_loan
        elif recommended_amount < min_loan and loan_multiplier > 0:
            recommended_amount = min_loan
        
        return {
            'recommended_amount_etb': recommended_amount,
            'interest_rate_annual': interest_rate,
            'term_months': term_months,
            'monthly_payment_etb': self._calculate_monthly_payment(
                recommended_amount, interest_rate, term_months
            ),
            'loan_to_revenue_ratio': loan_multiplier,
            'approval_probability': min(score * 1.2, 1.0)
        }
    
    def _calculate_monthly_payment(self, principal: float, annual_rate: float, months: int) -> float:
        """Calculate monthly loan payment"""
        
        if principal == 0 or months == 0:
            return 0
        
        monthly_rate = annual_rate / 12
        
        if monthly_rate == 0:
            return principal / months
        
        payment = principal * (monthly_rate * (1 + monthly_rate) ** months) / \
                 ((1 + monthly_rate) ** months - 1)
        
        return round(payment, 2)
    
    def _assess_eligibility(self, score: float, vendor_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess loan eligibility based on score and thresholds"""
        
        thresholds = self.config['thresholds']
        
        # Check minimum requirements
        activity_metrics = vendor_metrics.get('activity_metrics', {})
        engagement_metrics = vendor_metrics.get('engagement_metrics', {})
        risk_indicators = vendor_metrics.get('risk_indicators', {})
        
        posts_per_week = activity_metrics.get('posts_per_week', 0)
        avg_views = engagement_metrics.get('average_views_per_post', 0)
        risk_score = risk_indicators.get('risk_score', 1)
        
        # Eligibility checks
        meets_activity = posts_per_week >= thresholds['min_posts_per_week']
        meets_engagement = avg_views >= thresholds['min_average_views']
        meets_risk = risk_score <= thresholds['max_risk_score']
        meets_score = score >= 0.4  # Minimum score threshold
        
        eligible = meets_activity and meets_engagement and meets_risk and meets_score
        
        # Eligibility reasons
        reasons = []
        if not meets_activity:
            reasons.append(f"Insufficient posting frequency ({posts_per_week:.1f} < {thresholds['min_posts_per_week']})")
        if not meets_engagement:
            reasons.append(f"Low engagement ({avg_views:.0f} < {thresholds['min_average_views']})")
        if not meets_risk:
            reasons.append(f"High risk score ({risk_score:.2f} > {thresholds['max_risk_score']})")
        if not meets_score:
            reasons.append(f"Low overall score ({score:.2f} < 0.4)")
        
        return {
            'eligible': eligible,
            'score': score,
            'requirements_met': {
                'activity': meets_activity,
                'engagement': meets_engagement,
                'risk': meets_risk,
                'score': meets_score
            },
            'rejection_reasons': reasons if not eligible else []
        }
    
    def _get_scoring_details(self, vendor_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed scoring breakdown"""
        
        return {
            'activity_details': {
                'posts_per_week': vendor_metrics.get('activity_metrics', {}).get('posts_per_week', 0),
                'consistency': vendor_metrics.get('activity_metrics', {}).get('activity_consistency', 0)
            },
            'engagement_details': {
                'avg_views': vendor_metrics.get('engagement_metrics', {}).get('average_views_per_post', 0),
                'engagement_rate': vendor_metrics.get('engagement_metrics', {}).get('engagement_rate', 0)
            },
            'business_details': {
                'avg_price': vendor_metrics.get('business_metrics', {}).get('average_price_etb', 0),
                'price_volatility': vendor_metrics.get('business_metrics', {}).get('price_volatility', 0),
                'product_diversity': vendor_metrics.get('business_metrics', {}).get('total_products_mentioned', 0)
            },
            'risk_details': {
                'risk_score': vendor_metrics.get('risk_indicators', {}).get('risk_score', 0),
                'risk_level': vendor_metrics.get('risk_indicators', {}).get('risk_level', 'Unknown')
            }
        }
    
    def batch_score_vendors(self, vendors_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Score multiple vendors"""
        
        logger.info(f"Scoring {len(vendors_metrics)} vendors")
        
        scored_vendors = {}
        
        for vendor_id, metrics in vendors_metrics.items():
            scored_vendors[vendor_id] = self.calculate_lending_score(metrics)
            logger.info(f"Scored vendor {vendor_id}: {scored_vendors[vendor_id]['final_lending_score']:.3f}")
        
        return scored_vendors
    
    def rank_vendors(self, scored_vendors: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float, str]]:
        """Rank vendors by lending score"""
        
        vendor_scores = [
            (vendor_id, score_data['final_lending_score'], score_data['score_category'])
            for vendor_id, score_data in scored_vendors.items()
        ]
        
        # Sort by score (descending)
        vendor_scores.sort(key=lambda x: x[1], reverse=True)
        
        return vendor_scores

def main():
    """Test the lending scorer"""
    scorer = LendingScorer()
    print("Lending scorer initialized successfully!")

if __name__ == "__main__":
    main()
