"""
Scorecard Generator for EthioMart Vendor Assessment
Creates comprehensive vendor scorecards for micro-lending decisions
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from loguru import logger

class ScorecardGenerator:
    """Generator for vendor scorecards and reports"""
    
    def __init__(self, output_dir: str = "vendor_scorecards"):
        """Initialize the scorecard generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        logger.info(f"Scorecard generator initialized: {output_dir}")
    
    def generate_vendor_scorecard(self, 
                                 vendor_scores: Dict[str, Dict[str, Any]],
                                 vendor_analytics: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive vendor scorecard"""
        
        logger.info(f"Generating scorecard for {len(vendor_scores)} vendors")
        
        # Create summary table
        summary_table = self._create_summary_table(vendor_scores, vendor_analytics)
        
        # Generate rankings
        rankings = self._generate_rankings(vendor_scores)
        
        # Create visualizations
        visualization_paths = self._create_visualizations(vendor_scores, summary_table)
        
        # Generate insights
        insights = self._generate_insights(vendor_scores, summary_table)
        
        # Create detailed vendor profiles
        vendor_profiles = self._create_vendor_profiles(vendor_scores, vendor_analytics)
        
        # Compile scorecard
        scorecard = {
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'total_vendors': len(vendor_scores),
                'analysis_type': 'micro_lending_assessment'
            },
            'executive_summary': {
                'top_vendors': rankings['top_vendors'][:5],
                'eligible_vendors': rankings['eligible_count'],
                'average_score': rankings['average_score'],
                'score_distribution': rankings['score_distribution']
            },
            'summary_table': summary_table,
            'vendor_rankings': rankings,
            'vendor_profiles': vendor_profiles,
            'insights_and_recommendations': insights,
            'visualizations': visualization_paths
        }
        
        # Save scorecard
        scorecard_file = self.output_dir / f"vendor_scorecard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(scorecard_file, 'w', encoding='utf-8') as f:
            json.dump(scorecard, f, indent=2, ensure_ascii=False, default=str)
        
        # Generate HTML report
        html_report = self._generate_html_scorecard(scorecard)
        html_file = scorecard_file.with_suffix('.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Generate CSV summary
        csv_file = self.output_dir / "data" / "vendor_summary.csv"
        summary_df = pd.DataFrame(summary_table)
        summary_df.to_csv(csv_file, index=False)
        
        logger.info(f"Scorecard generated: {scorecard_file}")
        logger.info(f"HTML report: {html_file}")
        logger.info(f"CSV summary: {csv_file}")
        
        return scorecard
    
    def _create_summary_table(self, 
                            vendor_scores: Dict[str, Dict[str, Any]],
                            vendor_analytics: Dict[str, Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Create vendor summary table"""
        
        summary_data = []
        
        for vendor_id, score_data in vendor_scores.items():
            # Get analytics data if available
            analytics = vendor_analytics.get(vendor_id, {}) if vendor_analytics else {}
            
            # Extract key metrics
            activity_metrics = analytics.get('activity_metrics', {})
            engagement_metrics = analytics.get('engagement_metrics', {})
            business_metrics = analytics.get('business_metrics', {})
            
            summary_row = {
                'Vendor ID': vendor_id,
                'Lending Score': round(score_data['final_lending_score'], 3),
                'Score Category': score_data['score_category'],
                'Posts/Week': round(activity_metrics.get('posts_per_week', 0), 1),
                'Avg Views/Post': int(engagement_metrics.get('average_views_per_post', 0)),
                'Avg Price (ETB)': int(business_metrics.get('average_price_etb', 0)),
                'Business Profile': business_metrics.get('business_profile', 'Unknown'),
                'Risk Level': score_data.get('scoring_details', {}).get('risk_details', {}).get('risk_level', 'Unknown'),
                'Eligible': 'Yes' if score_data['eligibility']['eligible'] else 'No',
                'Recommended Loan (ETB)': int(score_data['loan_recommendation']['recommended_amount_etb']),
                'Interest Rate (%)': round(score_data['loan_recommendation']['interest_rate_annual'] * 100, 1),
                'Approval Probability (%)': round(score_data['loan_recommendation']['approval_probability'] * 100, 1)
            }
            
            summary_data.append(summary_row)
        
        # Sort by lending score (descending)
        summary_data.sort(key=lambda x: x['Lending Score'], reverse=True)
        
        return summary_data
    
    def _generate_rankings(self, vendor_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate vendor rankings and statistics"""
        
        scores = [data['final_lending_score'] for data in vendor_scores.values()]
        categories = [data['score_category'] for data in vendor_scores.values()]
        eligible_vendors = [data for data in vendor_scores.values() if data['eligibility']['eligible']]
        
        # Top vendors
        top_vendors = sorted(
            [(vendor_id, data['final_lending_score'], data['score_category']) 
             for vendor_id, data in vendor_scores.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Score distribution
        score_distribution = {
            'excellent': len([s for s in scores if s >= 0.8]),
            'good': len([s for s in scores if 0.6 <= s < 0.8]),
            'fair': len([s for s in scores if 0.4 <= s < 0.6]),
            'poor': len([s for s in scores if s < 0.4])
        }
        
        return {
            'top_vendors': top_vendors,
            'eligible_count': len(eligible_vendors),
            'total_count': len(vendor_scores),
            'average_score': np.mean(scores),
            'median_score': np.median(scores),
            'score_std': np.std(scores),
            'score_distribution': score_distribution,
            'category_distribution': {cat: categories.count(cat) for cat in set(categories)}
        }
    
    def _create_visualizations(self, 
                             vendor_scores: Dict[str, Dict[str, Any]],
                             summary_table: List[Dict[str, Any]]) -> Dict[str, str]:
        """Create visualization files"""
        
        viz_paths = {}
        
        # Score distribution plot
        score_dist_plot = self._create_score_distribution_plot(vendor_scores)
        if score_dist_plot:
            viz_paths['score_distribution'] = str(score_dist_plot)
        
        # Vendor comparison plot
        comparison_plot = self._create_vendor_comparison_plot(summary_table)
        if comparison_plot:
            viz_paths['vendor_comparison'] = str(comparison_plot)
        
        # Risk vs Score plot
        risk_score_plot = self._create_risk_score_plot(vendor_scores)
        if risk_score_plot:
            viz_paths['risk_score_analysis'] = str(risk_score_plot)
        
        # Loan recommendation plot
        loan_plot = self._create_loan_recommendation_plot(summary_table)
        if loan_plot:
            viz_paths['loan_recommendations'] = str(loan_plot)
        
        return viz_paths
    
    def _create_score_distribution_plot(self, vendor_scores: Dict[str, Dict[str, Any]]) -> Optional[Path]:
        """Create score distribution visualization"""
        
        scores = [data['final_lending_score'] for data in vendor_scores.values()]
        categories = [data['score_category'] for data in vendor_scores.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of scores
        ax1.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Lending Score')
        ax1.set_ylabel('Number of Vendors')
        ax1.set_title('Distribution of Lending Scores')
        ax1.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
        ax1.legend()
        
        # Category distribution
        category_counts = pd.Series(categories).value_counts()
        colors = ['green', 'lightgreen', 'orange', 'red']
        ax2.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', 
                colors=colors[:len(category_counts)])
        ax2.set_title('Score Category Distribution')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "visualizations" / "score_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _create_vendor_comparison_plot(self, summary_table: List[Dict[str, Any]]) -> Optional[Path]:
        """Create vendor comparison visualization"""
        
        if not summary_table:
            return None
        
        df = pd.DataFrame(summary_table)
        
        # Select top 10 vendors for visualization
        top_vendors = df.head(10)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Lending Score comparison
        ax1.bar(range(len(top_vendors)), top_vendors['Lending Score'], color='lightblue')
        ax1.set_xlabel('Vendor Rank')
        ax1.set_ylabel('Lending Score')
        ax1.set_title('Top 10 Vendors - Lending Scores')
        ax1.set_xticks(range(len(top_vendors)))
        ax1.set_xticklabels([f"#{i+1}" for i in range(len(top_vendors))])
        
        # Posts per week vs Average views
        ax2.scatter(top_vendors['Posts/Week'], top_vendors['Avg Views/Post'], 
                   c=top_vendors['Lending Score'], cmap='viridis', s=100)
        ax2.set_xlabel('Posts per Week')
        ax2.set_ylabel('Average Views per Post')
        ax2.set_title('Activity vs Engagement')
        
        # Average price distribution
        ax3.hist(df['Avg Price (ETB)'], bins=15, alpha=0.7, color='lightcoral')
        ax3.set_xlabel('Average Price (ETB)')
        ax3.set_ylabel('Number of Vendors')
        ax3.set_title('Price Point Distribution')
        
        # Loan recommendations
        eligible_vendors = df[df['Eligible'] == 'Yes']
        if len(eligible_vendors) > 0:
            ax4.bar(range(len(eligible_vendors)), eligible_vendors['Recommended Loan (ETB)'], 
                   color='lightgreen')
            ax4.set_xlabel('Eligible Vendor Rank')
            ax4.set_ylabel('Recommended Loan Amount (ETB)')
            ax4.set_title('Loan Recommendations for Eligible Vendors')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "visualizations" / "vendor_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _create_risk_score_plot(self, vendor_scores: Dict[str, Dict[str, Any]]) -> Optional[Path]:
        """Create risk vs score analysis plot"""
        
        scores = []
        risk_levels = []
        risk_scores = []
        
        for data in vendor_scores.values():
            scores.append(data['final_lending_score'])
            risk_details = data.get('scoring_details', {}).get('risk_details', {})
            risk_levels.append(risk_details.get('risk_level', 'Unknown'))
            risk_scores.append(risk_details.get('risk_score', 0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk level distribution
        risk_counts = pd.Series(risk_levels).value_counts()
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Unknown': 'gray'}
        bar_colors = [colors.get(level, 'gray') for level in risk_counts.index]
        
        ax1.bar(risk_counts.index, risk_counts.values, color=bar_colors)
        ax1.set_xlabel('Risk Level')
        ax1.set_ylabel('Number of Vendors')
        ax1.set_title('Risk Level Distribution')
        
        # Risk score vs Lending score
        ax2.scatter(risk_scores, scores, alpha=0.7, color='purple')
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('Lending Score')
        ax2.set_title('Risk Score vs Lending Score')
        
        # Add trend line (with error handling)
        if len(risk_scores) > 1 and len(set(risk_scores)) > 1:
            try:
                z = np.polyfit(risk_scores, scores, 1)
                p = np.poly1d(z)
                ax2.plot(risk_scores, p(risk_scores), "r--", alpha=0.8)
            except np.linalg.LinAlgError:
                # Skip trend line if calculation fails
                pass
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "visualizations" / "risk_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _create_loan_recommendation_plot(self, summary_table: List[Dict[str, Any]]) -> Optional[Path]:
        """Create loan recommendation visualization"""
        
        if not summary_table:
            return None
        
        df = pd.DataFrame(summary_table)
        eligible_df = df[df['Eligible'] == 'Yes']
        
        if len(eligible_df) == 0:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loan amount vs Score
        ax1.scatter(eligible_df['Lending Score'], eligible_df['Recommended Loan (ETB)'], 
                   c=eligible_df['Interest Rate (%)'], cmap='RdYlGn_r', s=100)
        ax1.set_xlabel('Lending Score')
        ax1.set_ylabel('Recommended Loan Amount (ETB)')
        ax1.set_title('Loan Amount vs Lending Score')
        
        # Add colorbar
        scatter = ax1.scatter(eligible_df['Lending Score'], eligible_df['Recommended Loan (ETB)'], 
                            c=eligible_df['Interest Rate (%)'], cmap='RdYlGn_r', s=100)
        plt.colorbar(scatter, ax=ax1, label='Interest Rate (%)')
        
        # Interest rate distribution
        ax2.hist(eligible_df['Interest Rate (%)'], bins=10, alpha=0.7, color='lightblue')
        ax2.set_xlabel('Interest Rate (%)')
        ax2.set_ylabel('Number of Vendors')
        ax2.set_title('Interest Rate Distribution')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "visualizations" / "loan_recommendations.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _generate_insights(self, 
                         vendor_scores: Dict[str, Dict[str, Any]],
                         summary_table: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights and recommendations"""
        
        df = pd.DataFrame(summary_table)
        
        insights = {
            'market_insights': [],
            'lending_recommendations': [],
            'risk_observations': [],
            'business_patterns': []
        }
        
        # Market insights
        avg_score = df['Lending Score'].mean()
        eligible_rate = (df['Eligible'] == 'Yes').mean()
        
        insights['market_insights'].append(f"Average lending score: {avg_score:.3f}")
        insights['market_insights'].append(f"Vendor eligibility rate: {eligible_rate:.1%}")
        
        if avg_score > 0.6:
            insights['market_insights'].append("Overall market shows good lending potential")
        else:
            insights['market_insights'].append("Market needs development for micro-lending")
        
        # Lending recommendations
        eligible_vendors = df[df['Eligible'] == 'Yes']
        if len(eligible_vendors) > 0:
            avg_loan = eligible_vendors['Recommended Loan (ETB)'].mean()
            insights['lending_recommendations'].append(f"Average recommended loan: {avg_loan:,.0f} ETB")
            
            high_score_vendors = eligible_vendors[eligible_vendors['Lending Score'] >= 0.8]
            if len(high_score_vendors) > 0:
                insights['lending_recommendations'].append(
                    f"{len(high_score_vendors)} vendors qualify for premium lending terms"
                )
        
        # Risk observations
        high_risk_count = (df['Risk Level'] == 'High').sum()
        if high_risk_count > 0:
            insights['risk_observations'].append(f"{high_risk_count} vendors show high risk indicators")
        
        # Business patterns
        common_profile = df['Business Profile'].mode().iloc[0] if len(df) > 0 else "Unknown"
        insights['business_patterns'].append(f"Most common business profile: {common_profile}")
        
        avg_price = df['Avg Price (ETB)'].mean()
        insights['business_patterns'].append(f"Average product price: {avg_price:,.0f} ETB")
        
        return insights
    
    def _create_vendor_profiles(self, 
                              vendor_scores: Dict[str, Dict[str, Any]],
                              vendor_analytics: Dict[str, Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Create detailed vendor profiles"""
        
        profiles = {}
        
        for vendor_id, score_data in vendor_scores.items():
            analytics = vendor_analytics.get(vendor_id, {}) if vendor_analytics else {}
            
            profile = {
                'vendor_id': vendor_id,
                'lending_assessment': {
                    'score': score_data['final_lending_score'],
                    'category': score_data['score_category'],
                    'eligible': score_data['eligibility']['eligible'],
                    'loan_recommendation': score_data['loan_recommendation']
                },
                'business_metrics': analytics.get('business_metrics', {}),
                'activity_summary': analytics.get('activity_metrics', {}),
                'engagement_summary': analytics.get('engagement_metrics', {}),
                'recommendations': score_data.get('recommendations', []),
                'risk_assessment': score_data.get('scoring_details', {}).get('risk_details', {})
            }
            
            profiles[vendor_id] = profile
        
        return profiles

    def _generate_html_scorecard(self, scorecard: Dict[str, Any]) -> str:
        """Generate HTML version of the scorecard"""

        summary_table = scorecard.get('summary_table', [])
        executive_summary = scorecard.get('executive_summary', {})

        # Create HTML table
        table_html = "<table border='1' style='border-collapse: collapse; width: 100%;'>"

        if summary_table:
            # Header
            headers = summary_table[0].keys()
            table_html += "<tr style='background-color: #f2f2f2;'>"
            for header in headers:
                table_html += f"<th style='padding: 8px; text-align: left;'>{header}</th>"
            table_html += "</tr>"

            # Rows
            for row in summary_table[:10]:  # Show top 10
                table_html += "<tr>"
                for value in row.values():
                    table_html += f"<td style='padding: 8px;'>{value}</td>"
                table_html += "</tr>"

        table_html += "</table>"

        html = f"""<!DOCTYPE html>
<html><head><title>EthioMart Vendor Scorecard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
.header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
.section {{ margin: 20px 0; }}
.metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
</style></head><body>
<div class="header">
<h1>EthioMart Vendor Scorecard</h1>
<p>Generated: {scorecard['metadata']['generation_time']}</p>
<p>Total Vendors Analyzed: {scorecard['metadata']['total_vendors']}</p>
</div>
<div class="section">
<h2>Executive Summary</h2>
<div class="metric"><strong>Eligible Vendors:</strong> {executive_summary.get('eligible_vendors', 0)}</div>
<div class="metric"><strong>Average Score:</strong> {executive_summary.get('average_score', 0):.3f}</div>
</div>
<div class="section">
<h2>Vendor Summary (Top 10)</h2>
{table_html}
</div>
</body></html>"""

        return html

def main():
    """Test the scorecard generator"""
    generator = ScorecardGenerator()
    print("Scorecard generator initialized successfully!")

if __name__ == "__main__":
    main()
