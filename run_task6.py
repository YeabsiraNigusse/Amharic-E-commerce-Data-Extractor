#!/usr/bin/env python3
"""
Main script to run Task 6: FinTech Vendor Scorecard for Micro-Lending
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from loguru import logger
from src.fintech.vendor_analytics import VendorAnalyticsEngine
from src.fintech.lending_scorer import LendingScorer
from src.fintech.scorecard_generator import ScorecardGenerator
from src.utils.data_utils import setup_logging

def create_mock_vendor_data():
    """Create mock vendor data for demonstration"""
    
    mock_vendors = [
        {
            'channel_username': 'ethio_electronics',
            'text': 'የስልክ ኬዝ 80 ብር። @ethio_electronics ላይ ይገናኙ። በቦሌ አካባቢ ይገኛል።',
            'views': 1250,
            'likes': 45,
            'timestamp': '2024-01-15 10:30:00',
            'entities': {'prices': [80], 'products': ['ስልክ ኬዝ'], 'locations': ['ቦሌ'], 'contact_info': ['@ethio_electronics']}
        },
        {
            'channel_username': 'ethio_electronics',
            'text': 'ላፕቶፕ ዋጋ 25000 ብር። ዴሊቨሪ ነፃ ነው። 0911234567 ይደውሉ።',
            'views': 2100,
            'likes': 78,
            'timestamp': '2024-01-16 14:20:00',
            'entities': {'prices': [25000], 'products': ['ላፕቶፕ'], 'contact_info': ['0911234567']}
        },
        {
            'channel_username': 'ethio_electronics',
            'text': 'ስማርት ፎን ETB 15000። ካዛንቺስ አካባቢ።',
            'views': 1800,
            'likes': 62,
            'timestamp': '2024-01-17 09:15:00',
            'entities': {'prices': [15000], 'products': ['ስማርት ፎን'], 'locations': ['ካዛንቺስ']}
        },
        {
            'channel_username': 'addis_fashion',
            'text': 'የሴቶች ልብስ በ 800 ብር። ፒያሳ አካባቢ ይገኛል። @addis_fashion',
            'views': 950,
            'likes': 32,
            'timestamp': '2024-01-15 11:45:00',
            'entities': {'prices': [800], 'products': ['ልብስ'], 'locations': ['ፒያሳ'], 'contact_info': ['@addis_fashion']}
        },
        {
            'channel_username': 'addis_fashion',
            'text': 'የወንዶች ሸሚዝ 300 ብር። በመርካቶ ይገኛል።',
            'views': 720,
            'likes': 18,
            'timestamp': '2024-01-16 16:30:00',
            'entities': {'prices': [300], 'products': ['ሸሚዝ'], 'locations': ['መርካቶ']}
        },
        {
            'channel_username': 'addis_fashion',
            'text': 'የሴቶች ቦርሳ 450 ብር። ዴሊቨሪ ክፍያ 50 ብር።',
            'views': 680,
            'likes': 25,
            'timestamp': '2024-01-17 13:10:00',
            'entities': {'prices': [450, 50], 'products': ['ቦርሳ']}
        },
        {
            'channel_username': 'book_store_et',
            'text': 'የመጽሐፍ መደብር በጎፋ። ዋጋ 200 ብር ነው።',
            'views': 320,
            'likes': 8,
            'timestamp': '2024-01-15 08:20:00',
            'entities': {'prices': [200], 'products': ['መጽሐፍ'], 'locations': ['ጎፋ']}
        },
        {
            'channel_username': 'book_store_et',
            'text': 'የሕፃናት መጽሐፍ 150 ብር። በላፍቶ ይገኛል።',
            'views': 280,
            'likes': 12,
            'timestamp': '2024-01-16 12:00:00',
            'entities': {'prices': [150], 'products': ['መጽሐፍ'], 'locations': ['ላፍቶ']}
        },
        {
            'channel_username': 'home_goods_addis',
            'text': 'የቤት እቃዎች በመርካቶ። ዋጋ 1500 ብር ነው። 0922334455',
            'views': 1100,
            'likes': 35,
            'timestamp': '2024-01-15 15:45:00',
            'entities': {'prices': [1500], 'products': ['ቤት እቃዎች'], 'locations': ['መርካቶ'], 'contact_info': ['0922334455']}
        },
        {
            'channel_username': 'home_goods_addis',
            'text': 'የኮምፒውተር አክሰሰሪዎች በ 1200 ብር። ኪርኮስ አካባቢ።',
            'views': 890,
            'likes': 28,
            'timestamp': '2024-01-16 10:30:00',
            'entities': {'prices': [1200], 'products': ['አክሰሰሪዎች'], 'locations': ['ኪርኮስ']}
        },
        {
            'channel_username': 'kids_toys_et',
            'text': 'የሕፃናት መጫወቻ 150 ብር። በላፍቶ ይገኛል።',
            'views': 450,
            'likes': 15,
            'timestamp': '2024-01-15 14:20:00',
            'entities': {'prices': [150], 'products': ['መጫወቻ'], 'locations': ['ላፍቶ']}
        },
        {
            'channel_username': 'kids_toys_et',
            'text': 'የወንዶች ሱሪ ዋጋ 600 ብር። ዴሊቨሪ 30 ብር።',
            'views': 380,
            'likes': 10,
            'timestamp': '2024-01-16 17:15:00',
            'entities': {'prices': [600, 30], 'products': ['ሱሪ']}
        }
    ]
    
    return mock_vendors

def run_vendor_analytics(data_file: str = None, use_mock: bool = False):
    """Run vendor analytics and scoring"""
    
    logger.info("Starting vendor analytics and scoring")
    
    # Initialize components
    analytics_engine = VendorAnalyticsEngine()
    lending_scorer = LendingScorer()
    scorecard_generator = ScorecardGenerator()
    
    if use_mock or not data_file or not Path(data_file).exists():
        logger.info("Using mock vendor data for demonstration")
        
        # Create mock data
        mock_data = create_mock_vendor_data()
        
        # Save mock data for processing
        mock_file = Path("mock_vendor_data.json")
        with open(mock_file, 'w', encoding='utf-8') as f:
            json.dump(mock_data, f, indent=2, ensure_ascii=False)
        
        data_file = str(mock_file)
    
    # Load vendor data
    if not analytics_engine.load_vendor_data(data_file):
        logger.error("Failed to load vendor data")
        return None
    
    # Process all vendors
    vendor_metrics = analytics_engine.process_all_vendors()
    
    # Calculate lending scores
    vendor_scores = lending_scorer.batch_score_vendors(vendor_metrics)
    
    # Generate scorecard
    scorecard = scorecard_generator.generate_vendor_scorecard(vendor_scores, vendor_metrics)
    
    return {
        'vendor_metrics': vendor_metrics,
        'vendor_scores': vendor_scores,
        'scorecard': scorecard
    }

def display_vendor_scorecard(results: dict):
    """Display vendor scorecard results"""
    
    if not results:
        logger.error("No results to display")
        return
    
    scorecard = results['scorecard']
    vendor_scores = results['vendor_scores']
    
    logger.info("=" * 80)
    logger.info("FINTECH VENDOR SCORECARD RESULTS")
    logger.info("=" * 80)
    
    # Executive Summary
    exec_summary = scorecard.get('executive_summary', {})
    
    logger.info("📊 EXECUTIVE SUMMARY:")
    logger.info(f"  Total Vendors Analyzed: {scorecard['metadata']['total_vendors']}")
    logger.info(f"  Eligible Vendors: {exec_summary.get('eligible_vendors', 0)}")
    logger.info(f"  Average Score: {exec_summary.get('average_score', 0):.3f}")
    
    # Top Vendors
    top_vendors = exec_summary.get('top_vendors', [])
    if top_vendors:
        logger.info("\n🏆 TOP VENDORS:")
        logger.info(f"{'Rank':<6} {'Vendor ID':<20} {'Score':<8} {'Category':<12}")
        logger.info("-" * 50)
        for i, (vendor_id, score, category) in enumerate(top_vendors[:5], 1):
            logger.info(f"{i:<6} {vendor_id:<20} {score:<8.3f} {category:<12}")
    
    # Vendor Summary Table
    summary_table = scorecard.get('summary_table', [])
    if summary_table:
        logger.info("\n📋 VENDOR SCORECARD:")
        logger.info(f"{'Vendor ID':<20} {'Score':<8} {'Posts/Week':<12} {'Avg Views':<12} {'Avg Price':<12} {'Eligible':<10}")
        logger.info("-" * 80)
        for row in summary_table[:10]:  # Show top 10
            logger.info(f"{row['Vendor ID']:<20} {row['Lending Score']:<8.3f} {row['Posts/Week']:<12.1f} "
                       f"{row['Avg Views/Post']:<12} {row['Avg Price (ETB)']:<12} {row['Eligible']:<10}")
    
    # Loan Recommendations
    eligible_vendors = [v for v in vendor_scores.values() if v['eligibility']['eligible']]
    if eligible_vendors:
        logger.info("\n💰 LOAN RECOMMENDATIONS:")
        logger.info(f"{'Vendor ID':<20} {'Loan Amount':<15} {'Interest Rate':<15} {'Term (months)':<15}")
        logger.info("-" * 70)
        for vendor_id, score_data in vendor_scores.items():
            if score_data['eligibility']['eligible']:
                loan_rec = score_data['loan_recommendation']
                logger.info(f"{vendor_id:<20} {loan_rec['recommended_amount_etb']:<15,.0f} "
                           f"{loan_rec['interest_rate_annual']*100:<15.1f}% {loan_rec['term_months']:<15}")
    
    # Risk Analysis
    risk_distribution = {}
    for score_data in vendor_scores.values():
        risk_level = score_data.get('scoring_details', {}).get('risk_details', {}).get('risk_level', 'Unknown')
        risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
    
    logger.info("\n⚠️  RISK ANALYSIS:")
    for risk_level, count in risk_distribution.items():
        logger.info(f"  {risk_level} Risk: {count} vendors")
    
    # Insights
    insights = scorecard.get('insights_and_recommendations', {})
    if insights:
        logger.info("\n💡 KEY INSIGHTS:")
        for category, insight_list in insights.items():
            if insight_list:
                logger.info(f"  {category.replace('_', ' ').title()}:")
                for insight in insight_list[:3]:  # Show top 3
                    logger.info(f"    • {insight}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="FinTech Vendor Scorecard for Micro-Lending")
    parser.add_argument('--data-file', '-d', 
                       help='Path to vendor data file (JSON or CSV)')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock data for demonstration')
    parser.add_argument('--output-dir', '-o', default='vendor_scorecards',
                       help='Output directory for scorecards')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("AMHARIC E-COMMERCE DATA EXTRACTOR - TASK 6")
    logger.info("FinTech Vendor Scorecard for Micro-Lending")
    logger.info("=" * 80)
    
    try:
        # Run vendor analytics
        results = run_vendor_analytics(
            data_file=args.data_file,
            use_mock=args.mock
        )
        
        if results:
            # Display results
            display_vendor_scorecard(results)
            
            logger.info("=" * 80)
            logger.info("TASK 6 COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info("Files generated:")
            logger.info(f"- {args.output_dir}/vendor_scorecard_*.json")
            logger.info(f"- {args.output_dir}/vendor_scorecard_*.html")
            logger.info(f"- {args.output_dir}/data/vendor_summary.csv")
            logger.info(f"- {args.output_dir}/visualizations/")
            
            logger.info("\nBusiness Impact:")
            logger.info("1. Data-driven micro-lending decisions")
            logger.info("2. Risk-adjusted loan recommendations")
            logger.info("3. Vendor performance benchmarking")
            logger.info("4. Automated eligibility assessment")
            
            logger.info("\nNext steps:")
            logger.info("1. Review vendor scorecards with business team")
            logger.info("2. Implement automated lending pipeline")
            logger.info("3. Monitor vendor performance over time")
            logger.info("4. Refine scoring algorithms based on outcomes")
        
        else:
            logger.error("Vendor analytics failed")
    
    except Exception as e:
        logger.error(f"Task 6 failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
