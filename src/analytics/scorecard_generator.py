"""
Vendor Scorecard Generator for Micro-Lending Assessment
Creates comprehensive scorecards and reports for vendor evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from loguru import logger
from .vendor_analyzer import VendorAnalyzer


class ScorecardGenerator:
    """Generates comprehensive vendor scorecards for micro-lending"""
    
    def __init__(self, vendor_analyzer: VendorAnalyzer):
        self.analyzer = vendor_analyzer
        self.scorecard_data = {}
        
        # Setup logging
        logger.add("logs/scorecard_generation.log", rotation="1 day")
        
        # Configure plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_vendor_scorecard_table(self, vendor_metrics: Dict[str, Dict[str, Any]], 
                                      lending_scores: Dict[str, float]) -> pd.DataFrame:
        """Generate the main vendor scorecard table"""
        logger.info("Generating vendor scorecard table")
        
        scorecard_data = []
        
        for vendor_name, metrics in vendor_metrics.items():
            lending_score = lending_scores.get(vendor_name, 0)
            
            scorecard_row = {
                'Vendor': vendor_name,
                'Avg. Views/Post': metrics['avg_views_per_post'],
                'Posts/Week': metrics['posting_frequency_per_week'],
                'Avg. Price (ETB)': metrics['avg_price_etb'],
                'Lending Score': lending_score,
                'Total Posts': metrics['total_posts'],
                'Price Listing Rate': f"{metrics['price_listing_rate']:.1%}",
                'Product Diversity': metrics['unique_products'],
                'Location Coverage': metrics['unique_locations'],
                'Max Views': metrics['max_views'],
                'Recent Activity (30d)': metrics['recent_activity_30d']
            }
            
            scorecard_data.append(scorecard_row)
        
        # Create DataFrame and sort by lending score
        scorecard_df = pd.DataFrame(scorecard_data)
        scorecard_df = scorecard_df.sort_values('Lending Score', ascending=False)
        
        self.scorecard_data = scorecard_df
        logger.info(f"Generated scorecard for {len(scorecard_df)} vendors")
        
        return scorecard_df
    
    def create_vendor_ranking_chart(self, scorecard_df: pd.DataFrame, 
                                  output_path: str = None) -> None:
        """Create vendor ranking visualization"""
        logger.info("Creating vendor ranking chart")
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Color scale based on lending score
        # Use a simple color scale instead of RdYlGn
        
        fig.add_trace(go.Bar(
            y=scorecard_df['Vendor'],
            x=scorecard_df['Lending Score'],
            orientation='h',
            marker=dict(
                color=scorecard_df['Lending Score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Lending Score")
            ),
            text=scorecard_df['Lending Score'],
            textposition='inside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Lending Score: %{x}<br>' +
                         'Avg Views/Post: %{customdata[0]}<br>' +
                         'Posts/Week: %{customdata[1]}<br>' +
                         'Avg Price: %{customdata[2]} ETB<extra></extra>',
            customdata=scorecard_df[['Avg. Views/Post', 'Posts/Week', 'Avg. Price (ETB)']].values
        ))
        
        fig.update_layout(
            title='Vendor Lending Score Ranking',
            xaxis_title='Lending Score (0-100)',
            yaxis_title='Vendors',
            height=max(400, len(scorecard_df) * 40),
            showlegend=False
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Vendor ranking chart saved to {output_path}")
        
        fig.show()
    
    def create_metrics_comparison_chart(self, scorecard_df: pd.DataFrame, 
                                      output_path: str = None) -> None:
        """Create multi-metric comparison chart"""
        logger.info("Creating metrics comparison chart")
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Views per Post', 'Posting Frequency (Posts/Week)', 
                          'Average Price (ETB)', 'Lending Score Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average Views per Post
        fig.add_trace(
            go.Bar(x=scorecard_df['Vendor'], y=scorecard_df['Avg. Views/Post'],
                  name='Avg Views/Post', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Posting Frequency
        fig.add_trace(
            go.Bar(x=scorecard_df['Vendor'], y=scorecard_df['Posts/Week'],
                  name='Posts/Week', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Average Price
        fig.add_trace(
            go.Bar(x=scorecard_df['Vendor'], y=scorecard_df['Avg. Price (ETB)'],
                  name='Avg Price (ETB)', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Lending Score Distribution
        fig.add_trace(
            go.Histogram(x=scorecard_df['Lending Score'], nbinsx=10,
                        name='Score Distribution', marker_color='gold'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Vendor Metrics Comparison Dashboard',
            height=800,
            showlegend=False
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Metrics comparison chart saved to {output_path}")
        
        fig.show()
    
    def create_risk_assessment_matrix(self, scorecard_df: pd.DataFrame, 
                                    output_path: str = None) -> None:
        """Create risk assessment matrix visualization"""
        logger.info("Creating risk assessment matrix")
        
        # Create scatter plot with bubble size based on total posts
        fig = go.Figure()
        
        # Define risk categories
        def get_risk_category(score):
            if score >= 70:
                return "Low Risk"
            elif score >= 50:
                return "Medium Risk"
            elif score >= 30:
                return "High Risk"
            else:
                return "Very High Risk"
        
        scorecard_df['Risk Category'] = scorecard_df['Lending Score'].apply(get_risk_category)
        
        # Color mapping for risk categories
        color_map = {
            "Low Risk": "green",
            "Medium Risk": "yellow", 
            "High Risk": "orange",
            "Very High Risk": "red"
        }
        
        for risk_category in scorecard_df['Risk Category'].unique():
            subset = scorecard_df[scorecard_df['Risk Category'] == risk_category]
            
            fig.add_trace(go.Scatter(
                x=subset['Avg. Views/Post'],
                y=subset['Posts/Week'],
                mode='markers+text',
                marker=dict(
                    size=subset['Total Posts'] * 2,  # Bubble size based on total posts
                    color=color_map[risk_category],
                    opacity=0.7,
                    line=dict(width=2, color='black')
                ),
                text=subset['Vendor'],
                textposition='top center',
                name=risk_category,
                hovertemplate='<b>%{text}</b><br>' +
                             'Avg Views/Post: %{x}<br>' +
                             'Posts/Week: %{y}<br>' +
                             'Total Posts: %{marker.size}<br>' +
                             'Risk: %{fullData.name}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Vendor Risk Assessment Matrix',
            xaxis_title='Average Views per Post',
            yaxis_title='Posting Frequency (Posts/Week)',
            legend_title='Risk Category',
            height=600
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Risk assessment matrix saved to {output_path}")
        
        fig.show()
    
    def generate_individual_vendor_report(self, vendor_name: str, 
                                        output_path: str = None) -> str:
        """Generate detailed report for individual vendor"""
        logger.info(f"Generating individual report for {vendor_name}")
        
        vendor_summary = self.analyzer.get_vendor_summary(vendor_name)
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append(f"VENDOR SCORECARD: {vendor_name.upper()}")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY:")
        report_lines.append(f"  Lending Score: {vendor_summary['lending_score']}/100")
        report_lines.append(f"  Risk Level: {vendor_summary['risk_level']}")
        report_lines.append("")
        
        # Key Performance Metrics
        report_lines.append("KEY PERFORMANCE METRICS:")
        metrics = vendor_summary['key_metrics']
        report_lines.append(f"  Average Views per Post: {metrics['avg_views_per_post']}")
        report_lines.append(f"  Posting Frequency: {metrics['posting_frequency_per_week']} posts/week")
        report_lines.append(f"  Average Price Point: {metrics['avg_price_etb']} ETB")
        report_lines.append(f"  Price Listing Rate: {metrics['price_listing_rate']:.1%}")
        report_lines.append("")
        
        # Business Profile
        report_lines.append("BUSINESS PROFILE:")
        profile = vendor_summary['business_profile']
        report_lines.append(f"  Total Posts: {profile['total_posts']}")
        report_lines.append(f"  Product Diversity: {profile['unique_products']} unique products")
        report_lines.append(f"  Location Coverage: {profile['unique_locations']} locations")
        report_lines.append(f"  Price Range: {profile['price_range']}")
        report_lines.append("")
        
        # Top Performing Content
        if vendor_summary['top_performing_post']:
            top_post = vendor_summary['top_performing_post']
            report_lines.append("TOP PERFORMING POST:")
            report_lines.append(f"  Views: {top_post['views']}")
            report_lines.append(f"  Content: {top_post['text'][:100]}...")
            report_lines.append(f"  Date: {top_post['date']}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS FOR IMPROVEMENT:")
        for i, recommendation in enumerate(vendor_summary['recommendations'], 1):
            report_lines.append(f"  {i}. {recommendation}")
        report_lines.append("")
        
        # Lending Assessment
        report_lines.append("LENDING ASSESSMENT:")
        if vendor_summary['lending_score'] >= 70:
            report_lines.append("  ✅ APPROVED: Strong candidate for micro-lending")
            report_lines.append("  - Consistent business activity")
            report_lines.append("  - Good market engagement")
            report_lines.append("  - Transparent pricing")
        elif vendor_summary['lending_score'] >= 50:
            report_lines.append("  ⚠️  CONDITIONAL: Consider with additional requirements")
            report_lines.append("  - Moderate business activity")
            report_lines.append("  - May require collateral or guarantor")
        else:
            report_lines.append("  ❌ NOT RECOMMENDED: High risk for lending")
            report_lines.append("  - Insufficient business activity")
            report_lines.append("  - Recommend business development first")
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Individual vendor report saved to {output_path}")
        
        return report_content
    
    def generate_comprehensive_scorecard_report(self, scorecard_df: pd.DataFrame, 
                                              output_path: str = None) -> str:
        """Generate comprehensive scorecard report"""
        logger.info("Generating comprehensive scorecard report")
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("ETHIOMART VENDOR SCORECARD FOR MICRO-LENDING")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY:")
        report_lines.append(f"  Total Vendors Analyzed: {len(scorecard_df)}")
        report_lines.append(f"  Average Lending Score: {scorecard_df['Lending Score'].mean():.1f}")
        report_lines.append(f"  Top Performer: {scorecard_df.iloc[0]['Vendor']} (Score: {scorecard_df.iloc[0]['Lending Score']})")
        
        # Risk Distribution
        risk_counts = scorecard_df['Lending Score'].apply(
            lambda x: "Low Risk" if x >= 70 else "Medium Risk" if x >= 50 else "High Risk" if x >= 30 else "Very High Risk"
        ).value_counts()
        
        report_lines.append("")
        report_lines.append("RISK DISTRIBUTION:")
        for risk_level, count in risk_counts.items():
            percentage = (count / len(scorecard_df)) * 100
            report_lines.append(f"  {risk_level}: {count} vendors ({percentage:.1f}%)")
        
        # Main Scorecard Table
        report_lines.append("")
        report_lines.append("VENDOR SCORECARD:")
        report_lines.append("-" * 80)
        
        # Format table
        table_df = scorecard_df[['Vendor', 'Avg. Views/Post', 'Posts/Week', 'Avg. Price (ETB)', 'Lending Score']].copy()
        table_df.columns = ['Vendor', 'Avg Views/Post', 'Posts/Week', 'Avg Price (ETB)', 'Lending Score']
        
        # Convert to string representation
        table_str = table_df.to_string(index=False, float_format='%.1f')
        report_lines.append(table_str)
        
        # Key Insights
        report_lines.append("")
        report_lines.append("KEY INSIGHTS:")
        
        # Top performers
        top_3 = scorecard_df.head(3)
        report_lines.append("  Top 3 Lending Candidates:")
        for i, (_, vendor) in enumerate(top_3.iterrows(), 1):
            report_lines.append(f"    {i}. {vendor['Vendor']} (Score: {vendor['Lending Score']})")
        
        # Market analysis
        avg_views = scorecard_df['Avg. Views/Post'].mean()
        avg_frequency = scorecard_df['Posts/Week'].mean()
        avg_price = scorecard_df['Avg. Price (ETB)'].mean()
        
        report_lines.append("")
        report_lines.append("  Market Averages:")
        report_lines.append(f"    Average Views per Post: {avg_views:.1f}")
        report_lines.append(f"    Average Posting Frequency: {avg_frequency:.1f} posts/week")
        report_lines.append(f"    Average Price Point: {avg_price:.1f} ETB")
        
        # Recommendations
        report_lines.append("")
        report_lines.append("LENDING RECOMMENDATIONS:")
        report_lines.append("  1. Prioritize vendors with scores above 70 for immediate lending")
        report_lines.append("  2. Vendors with scores 50-70 may qualify with additional requirements")
        report_lines.append("  3. Vendors below 50 should focus on business development first")
        report_lines.append("  4. Monitor posting frequency and engagement trends over time")
        report_lines.append("  5. Consider seasonal variations in business activity")
        
        report_content = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Comprehensive scorecard report saved to {output_path}")
        
        return report_content
    
    def export_scorecard_data(self, scorecard_df: pd.DataFrame, 
                            format: str = 'excel', output_path: str = None) -> str:
        """Export scorecard data in various formats"""
        logger.info(f"Exporting scorecard data in {format} format")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/vendor_scorecard_{timestamp}.{format}"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'excel':
            scorecard_df.to_excel(output_path, index=False)
        elif format.lower() == 'csv':
            scorecard_df.to_csv(output_path, index=False, encoding='utf-8')
        elif format.lower() == 'json':
            scorecard_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Scorecard data exported to {output_path}")
        return output_path


def main():
    """Main function for testing scorecard generation"""
    from .vendor_analyzer import VendorAnalyzer
    
    # Create sample data
    sample_data = [
        {
            'channel_name': 'Vendor A',
            'text': 'የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።',
            'views': 150,
            'date': '2024-01-15T10:00:00'
        },
        {
            'channel_name': 'Vendor B', 
            'text': 'ሴቶች ጫማ 800 ብር። መርካቶ አካባቢ።',
            'views': 300,
            'date': '2024-01-16T14:30:00'
        }
    ]
    
    # Test scorecard generation
    analyzer = VendorAnalyzer()
    df = pd.DataFrame(sample_data)
    df['date'] = pd.to_datetime(df['date'])
    
    df = analyzer.extract_entities_from_messages(df)
    vendor_metrics = analyzer.analyze_vendor_performance(df)
    lending_scores = analyzer.calculate_lending_scores(vendor_metrics)
    
    # Generate scorecard
    generator = ScorecardGenerator(analyzer)
    scorecard_df = generator.generate_vendor_scorecard_table(vendor_metrics, lending_scores)
    
    print("Vendor Scorecard:")
    print(scorecard_df)
    
    # Generate reports
    report = generator.generate_comprehensive_scorecard_report(scorecard_df)
    print("\nComprehensive Report:")
    print(report)


if __name__ == "__main__":
    main()
