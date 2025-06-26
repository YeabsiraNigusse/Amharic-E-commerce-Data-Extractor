#!/usr/bin/env python3
"""
Task 6: FinTech Vendor Scorecard for Micro-Lending
Analyzes vendor performance and generates lending scorecards
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.analytics.vendor_analyzer import VendorAnalyzer
from src.analytics.scorecard_generator import ScorecardGenerator
from src.analytics.metrics_calculator import MetricsCalculator


def create_sample_telegram_data():
    """Create sample Telegram data for demonstration"""
    logger.info("Creating sample Telegram data for demonstration")
    
    sample_data = [
        # Vendor A - High performing vendor
        {
            'channel_name': 'Bole Fashion Store',
            'channel_username': '@bolefashion',
            'text': 'የሴቶች ልብስ ዋጋ 800 ብር ነው። በቦሌ አካባቢ ይገኛል። ጥራት ያለው ልብስ።',
            'views': 250,
            'forwards': 15,
            'replies': 8,
            'date': '2024-01-15T10:00:00',
            'has_media': True,
            'message_id': 1001
        },
        {
            'channel_name': 'Bole Fashion Store',
            'channel_username': '@bolefashion',
            'text': 'የወንዶች ሸሚዝ 600 ብር። ዘመናዊ ዲዛይን። ቦሌ አካባቢ።',
            'views': 180,
            'forwards': 12,
            'replies': 5,
            'date': '2024-01-16T14:30:00',
            'has_media': True,
            'message_id': 1002
        },
        {
            'channel_name': 'Bole Fashion Store',
            'channel_username': '@bolefashion',
            'text': 'ሴቶች ጫማ 1200 ብር። ከውጭ የመጣ። በቦሌ አካባቢ ይገኛል።',
            'views': 320,
            'forwards': 25,
            'replies': 12,
            'date': '2024-01-17T09:15:00',
            'has_media': True,
            'message_id': 1003
        },
        {
            'channel_name': 'Bole Fashion Store',
            'channel_username': '@bolefashion',
            'text': 'የሕፃናት ልብስ 400 ብር። ለ2-5 ዓመት ልጆች። ቦሌ አካባቢ።',
            'views': 150,
            'forwards': 8,
            'replies': 3,
            'date': '2024-01-18T16:45:00',
            'has_media': False,
            'message_id': 1004
        },
        
        # Vendor B - Medium performing vendor
        {
            'channel_name': 'Merkato Electronics',
            'channel_username': '@merkatotech',
            'text': 'የስልክ ፓወር ባንክ 450 ብር። መርካቶ አካባቢ ይገኛል።',
            'views': 120,
            'forwards': 5,
            'replies': 2,
            'date': '2024-01-15T11:30:00',
            'has_media': False,
            'message_id': 2001
        },
        {
            'channel_name': 'Merkato Electronics',
            'channel_username': '@merkatotech',
            'text': 'ስልክ ጆሮ ማዳመጫ 200 ብር። ጥራት ያለው። መርካቶ።',
            'views': 80,
            'forwards': 3,
            'replies': 1,
            'date': '2024-01-17T13:20:00',
            'has_media': False,
            'message_id': 2002
        },
        {
            'channel_name': 'Merkato Electronics',
            'channel_username': '@merkatotech',
            'text': 'ስልክ ኬዝ 150 ብር። የተለያዩ ዓይነት። መርካቶ አካባቢ።',
            'views': 95,
            'forwards': 4,
            'replies': 0,
            'date': '2024-01-19T10:10:00',
            'has_media': True,
            'message_id': 2003
        },
        
        # Vendor C - Lower performing vendor
        {
            'channel_name': 'Piassa Books',
            'channel_username': '@piassabooks',
            'text': 'መጽሐፍ 300 ብር። ፒያሳ አካባቢ።',
            'views': 45,
            'forwards': 1,
            'replies': 0,
            'date': '2024-01-16T15:00:00',
            'has_media': False,
            'message_id': 3001
        },
        {
            'channel_name': 'Piassa Books',
            'channel_username': '@piassabooks',
            'text': 'ትምህርታዊ መጽሐፍ 250 ብር። ፒያሳ።',
            'views': 30,
            'forwards': 0,
            'replies': 1,
            'date': '2024-01-20T12:30:00',
            'has_media': False,
            'message_id': 3002
        },
        
        # Vendor D - High activity, good engagement
        {
            'channel_name': 'Shiro Meda Cosmetics',
            'channel_username': '@shiromedacosmetics',
            'text': 'የሴቶች ሜክአፕ 500 ብር። ከውጭ የመጣ። ሽሮ መዳ አካባቢ።',
            'views': 200,
            'forwards': 18,
            'replies': 10,
            'date': '2024-01-15T08:00:00',
            'has_media': True,
            'message_id': 4001
        },
        {
            'channel_name': 'Shiro Meda Cosmetics',
            'channel_username': '@shiromedacosmetics',
            'text': 'ሻምፖ 350 ብር። ለሁሉም የፀጉር ዓይነት። ሽሮ መዳ።',
            'views': 160,
            'forwards': 12,
            'replies': 6,
            'date': '2024-01-16T19:30:00',
            'has_media': True,
            'message_id': 4002
        },
        {
            'channel_name': 'Shiro Meda Cosmetics',
            'channel_username': '@shiromedacosmetics',
            'text': 'የፊት ክሬም 400 ብር። ለደረቅ ቆዳ። ሽሮ መዳ አካባቢ ይገኛል።',
            'views': 180,
            'forwards': 15,
            'replies': 8,
            'date': '2024-01-18T07:45:00',
            'has_media': False,
            'message_id': 4003
        },
        {
            'channel_name': 'Shiro Meda Cosmetics',
            'channel_username': '@shiromedacosmetics',
            'text': 'ሊፕስቲክ 250 ብር። የተለያዩ ቀለም። ሽሮ መዳ።',
            'views': 140,
            'forwards': 10,
            'replies': 4,
            'date': '2024-01-19T20:15:00',
            'has_media': True,
            'message_id': 4004
        }
    ]
    
    return sample_data


def main():
    """Main function for Task 6: Vendor Scorecard"""
    logger.info("Starting Task 6: FinTech Vendor Scorecard for Micro-Lending")
    
    # Create necessary directories
    Path("data/vendor_analysis").mkdir(parents=True, exist_ok=True)
    Path("reports/vendor_scorecards").mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load or Create Sample Data
        logger.info("Step 1: Loading Telegram Data")
        
        # Check if real data exists, otherwise use sample data
        real_data_path = "data/raw/telegram_messages_latest.json"
        sample_data_path = "data/vendor_analysis/sample_telegram_data.json"
        
        if Path(real_data_path).exists():
            logger.info("Using real Telegram data")
            data_path = real_data_path
        else:
            logger.info("Creating sample data for demonstration")
            sample_data = create_sample_telegram_data()
            
            # Save sample data
            with open(sample_data_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            data_path = sample_data_path
        
        # Step 2: Initialize Vendor Analytics Engine
        logger.info("Step 2: Initializing Vendor Analytics Engine")
        
        # Try to load trained NER model if available
        ner_model_path = "models/ner_model"
        if Path(ner_model_path).exists():
            analyzer = VendorAnalyzer(ner_model_path)
            logger.info("Loaded trained NER model for entity extraction")
        else:
            analyzer = VendorAnalyzer()
            logger.info("Using fallback entity extraction (regex-based)")
        
        # Load data
        df = analyzer.load_telegram_data(data_path)
        logger.info(f"Loaded {len(df)} messages from {df['channel_name'].nunique()} vendors")
        
        # Step 3: Extract Entities from Messages
        logger.info("Step 3: Extracting Entities from Messages")
        df = analyzer.extract_entities_from_messages(df)
        
        # Step 4: Analyze Vendor Performance
        logger.info("Step 4: Analyzing Vendor Performance")
        vendor_metrics = analyzer.analyze_vendor_performance(df)
        
        # Step 5: Calculate Lending Scores
        logger.info("Step 5: Calculating Lending Scores")
        
        # Custom weights for lending score calculation
        lending_weights = {
            'avg_views': 0.3,           # Market reach importance
            'posting_frequency': 0.25,  # Business activity consistency
            'price_consistency': 0.2,   # Business transparency
            'recent_activity': 0.15,    # Current business status
            'business_diversity': 0.1   # Risk diversification
        }
        
        lending_scores = analyzer.calculate_lending_scores(vendor_metrics, lending_weights)
        
        # Step 6: Generate Vendor Scorecard
        logger.info("Step 6: Generating Vendor Scorecard")
        
        generator = ScorecardGenerator(analyzer)
        scorecard_df = generator.generate_vendor_scorecard_table(vendor_metrics, lending_scores)
        
        # Step 7: Create Visualizations
        logger.info("Step 7: Creating Visualizations")
        
        try:
            # Vendor ranking chart
            generator.create_vendor_ranking_chart(
                scorecard_df, 
                "reports/vendor_scorecards/vendor_ranking.html"
            )
            
            # Metrics comparison dashboard
            generator.create_metrics_comparison_chart(
                scorecard_df,
                "reports/vendor_scorecards/metrics_comparison.html"
            )
            
            # Risk assessment matrix
            generator.create_risk_assessment_matrix(
                scorecard_df,
                "reports/vendor_scorecards/risk_assessment.html"
            )
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.warning(f"Some visualizations failed: {e}")
        
        # Step 8: Generate Individual Vendor Reports
        logger.info("Step 8: Generating Individual Vendor Reports")
        
        for vendor_name in scorecard_df['Vendor'].head(3):  # Top 3 vendors
            try:
                individual_report = generator.generate_individual_vendor_report(
                    vendor_name,
                    f"reports/vendor_scorecards/{vendor_name.replace(' ', '_')}_report.txt"
                )
                logger.info(f"Generated report for {vendor_name}")
            except Exception as e:
                logger.warning(f"Failed to generate report for {vendor_name}: {e}")
        
        # Step 9: Generate Comprehensive Scorecard Report
        logger.info("Step 9: Generating Comprehensive Scorecard Report")
        
        comprehensive_report = generator.generate_comprehensive_scorecard_report(
            scorecard_df,
            "reports/vendor_scorecards/comprehensive_scorecard_report.txt"
        )
        
        # Step 10: Export Scorecard Data
        logger.info("Step 10: Exporting Scorecard Data")
        
        # Export in multiple formats
        generator.export_scorecard_data(scorecard_df, 'excel', "reports/vendor_scorecards/vendor_scorecard.xlsx")
        generator.export_scorecard_data(scorecard_df, 'csv', "reports/vendor_scorecards/vendor_scorecard.csv")
        generator.export_scorecard_data(scorecard_df, 'json', "reports/vendor_scorecards/vendor_scorecard.json")
        
        # Step 11: Generate Final Summary Report
        logger.info("Step 11: Generating Final Summary Report")
        
        # Calculate summary statistics
        total_vendors = len(scorecard_df)
        avg_lending_score = scorecard_df['Lending Score'].mean()
        top_vendor = scorecard_df.iloc[0]
        
        # Risk distribution
        risk_distribution = scorecard_df['Lending Score'].apply(
            lambda x: "Low Risk (70+)" if x >= 70 else 
                     "Medium Risk (50-69)" if x >= 50 else 
                     "High Risk (30-49)" if x >= 30 else 
                     "Very High Risk (<30)"
        ).value_counts()
        
        summary_report = f"""
# TASK 6: VENDOR SCORECARD FOR MICRO-LENDING - SUMMARY REPORT

## Executive Summary

EthioMart's Vendor Analytics Engine has successfully analyzed {total_vendors} vendors from Telegram channels to assess their suitability for micro-lending opportunities.

## Key Findings

### Overall Market Analysis
- **Total Vendors Analyzed**: {total_vendors}
- **Average Lending Score**: {avg_lending_score:.1f}/100
- **Top Performing Vendor**: {top_vendor['Vendor']} (Score: {top_vendor['Lending Score']})

### Risk Distribution
"""
        
        for risk_level, count in risk_distribution.items():
            percentage = (count / total_vendors) * 100
            summary_report += f"- **{risk_level}**: {count} vendors ({percentage:.1f}%)\n"
        
        summary_report += f"""

## Vendor Scorecard Table

| Vendor | Avg. Views/Post | Posts/Week | Avg. Price (ETB) | Lending Score |
|--------|----------------|------------|------------------|---------------|
"""
        
        for _, vendor in scorecard_df.iterrows():
            summary_report += f"| {vendor['Vendor']} | {vendor['Avg. Views/Post']:.1f} | {vendor['Posts/Week']:.1f} | {vendor['Avg. Price (ETB)']:.0f} | {vendor['Lending Score']:.1f} |\n"
        
        summary_report += f"""

## Top 3 Lending Candidates

"""
        
        for i, (_, vendor) in enumerate(scorecard_df.head(3).iterrows(), 1):
            summary_report += f"""
### {i}. {vendor['Vendor']} (Score: {vendor['Lending Score']:.1f})
- **Market Reach**: {vendor['Avg. Views/Post']:.0f} avg views per post
- **Activity Level**: {vendor['Posts/Week']:.1f} posts per week  
- **Price Point**: {vendor['Avg. Price (ETB)']:.0f} ETB average
- **Business Profile**: {vendor['Product Diversity']} unique products, {vendor['Location Coverage']} locations
"""
        
        summary_report += f"""

## Lending Recommendations

### Immediate Lending Candidates (Score ≥ 70)
"""
        high_score_vendors = scorecard_df[scorecard_df['Lending Score'] >= 70]
        if len(high_score_vendors) > 0:
            for _, vendor in high_score_vendors.iterrows():
                summary_report += f"- **{vendor['Vendor']}**: Strong business activity and market engagement\n"
        else:
            summary_report += "- No vendors currently meet the high-confidence threshold\n"
        
        summary_report += f"""

### Conditional Lending Candidates (Score 50-69)
"""
        medium_score_vendors = scorecard_df[(scorecard_df['Lending Score'] >= 50) & (scorecard_df['Lending Score'] < 70)]
        if len(medium_score_vendors) > 0:
            for _, vendor in medium_score_vendors.iterrows():
                summary_report += f"- **{vendor['Vendor']}**: Consider with additional requirements or collateral\n"
        else:
            summary_report += "- No vendors in this category\n"
        
        summary_report += f"""

## Key Metrics Analysis

### Market Engagement
- **Average Views per Post**: {scorecard_df['Avg. Views/Post'].mean():.1f}
- **Top Performer**: {scorecard_df.loc[scorecard_df['Avg. Views/Post'].idxmax(), 'Vendor']} ({scorecard_df['Avg. Views/Post'].max():.0f} views)

### Business Activity  
- **Average Posting Frequency**: {scorecard_df['Posts/Week'].mean():.1f} posts/week
- **Most Active**: {scorecard_df.loc[scorecard_df['Posts/Week'].idxmax(), 'Vendor']} ({scorecard_df['Posts/Week'].max():.1f} posts/week)

### Price Analysis
- **Average Price Point**: {scorecard_df['Avg. Price (ETB)'].mean():.0f} ETB
- **Price Range**: {scorecard_df['Avg. Price (ETB)'].min():.0f} - {scorecard_df['Avg. Price (ETB)'].max():.0f} ETB

## Methodology

### Lending Score Calculation
The lending score (0-100) is calculated using weighted metrics:
- **Market Reach (30%)**: Average views per post
- **Business Activity (25%)**: Posting frequency consistency  
- **Price Transparency (20%)**: Rate of price listing in posts
- **Recent Activity (15%)**: Activity in last 30 days
- **Business Diversity (10%)**: Product and location variety

### Data Sources
- **Telegram Channels**: {df['channel_name'].nunique()} vendor channels analyzed
- **Time Period**: {df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'} to {df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'}
- **Total Messages**: {len(df)} posts analyzed
- **Entity Extraction**: {"NER Model" if Path("models/ner_model").exists() else "Regex-based"} approach

## Files Generated
- **Comprehensive Report**: reports/vendor_scorecards/comprehensive_scorecard_report.txt
- **Scorecard Data**: reports/vendor_scorecards/vendor_scorecard.xlsx
- **Visualizations**: reports/vendor_scorecards/*.html
- **Individual Reports**: reports/vendor_scorecards/*_report.txt

## Next Steps
1. **Pilot Program**: Start with top 2-3 vendors for initial micro-lending
2. **Monitoring**: Track business performance post-lending
3. **Model Refinement**: Update scoring based on actual lending outcomes
4. **Scale**: Expand to more Telegram channels and vendors

---
*Generated by EthioMart Vendor Analytics Engine*
*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save summary report
        with open("reports/vendor_scorecards/task6_summary_report.md", 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        # Step 12: Display Results
        logger.info("Task 6 completed successfully!")
        
        print("\n" + "="*80)
        print("TASK 6: VENDOR SCORECARD FOR MICRO-LENDING - COMPLETED")
        print("="*80)
        print(f"Total Vendors Analyzed: {total_vendors}")
        print(f"Average Lending Score: {avg_lending_score:.1f}/100")
        print(f"Top Performer: {top_vendor['Vendor']} (Score: {top_vendor['Lending Score']:.1f})")
        print("\nVendor Scorecard Summary:")
        print(scorecard_df[['Vendor', 'Avg. Views/Post', 'Posts/Week', 'Avg. Price (ETB)', 'Lending Score']].to_string(index=False))
        print(f"\nDetailed reports saved to: reports/vendor_scorecards/")
        
        logger.info("Generated files:")
        logger.info("- reports/vendor_scorecards/vendor_scorecard.xlsx")
        logger.info("- reports/vendor_scorecards/comprehensive_scorecard_report.txt")
        logger.info("- reports/vendor_scorecards/task6_summary_report.md")
        logger.info("- reports/vendor_scorecards/*.html (visualizations)")
        
    except Exception as e:
        logger.error(f"Task 6 failed: {e}")
        raise


if __name__ == "__main__":
    main()
