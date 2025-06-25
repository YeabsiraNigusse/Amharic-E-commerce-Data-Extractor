"""
Interpretability Report Generator
Creates comprehensive reports combining SHAP, LIME, and other analysis
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger

class InterpretabilityReport:
    """Generator for comprehensive interpretability reports"""
    
    def __init__(self, output_dir: str = "interpretability_results"):
        """Initialize the report generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        logger.info(f"Interpretability report generator initialized: {output_dir}")
    
    def generate_comprehensive_report(self, 
                                    model_path: str,
                                    explanations: List[Dict[str, Any]],
                                    difficult_cases: Dict[str, Any],
                                    model_performance: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive interpretability report"""
        
        logger.info("Generating comprehensive interpretability report")
        
        # Analyze explanations
        explanation_analysis = self._analyze_explanations(explanations)
        
        # Create visualizations
        visualization_paths = self._create_visualizations(explanations, explanation_analysis)
        
        # Generate insights
        insights = self._generate_insights(explanation_analysis, difficult_cases)
        
        # Create executive summary
        executive_summary = self._create_executive_summary(
            explanation_analysis, difficult_cases, insights, model_performance
        )
        
        # Compile full report
        report = {
            'metadata': {
                'model_path': model_path,
                'generation_time': datetime.now().isoformat(),
                'total_explanations': len(explanations),
                'analysis_methods': ['SHAP', 'LIME', 'Gradient-based']
            },
            'executive_summary': executive_summary,
            'explanation_analysis': explanation_analysis,
            'difficult_cases_analysis': difficult_cases,
            'insights_and_recommendations': insights,
            'visualizations': visualization_paths,
            'detailed_explanations': explanations[:10]  # Include first 10 for reference
        }
        
        # Save report
        report_file = self.output_dir / f"interpretability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Generate HTML report
        html_report = self._generate_html_report(report)
        html_file = report_file.with_suffix('.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"Comprehensive report saved to {report_file}")
        logger.info(f"HTML report saved to {html_file}")
        
        return report
    
    def _analyze_explanations(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze explanation results across all methods"""
        
        analysis = {
            'method_comparison': {},
            'entity_analysis': {},
            'confidence_analysis': {},
            'feature_importance_patterns': {}
        }
        
        # Separate explanations by method
        shap_explanations = [exp for exp in explanations if exp.get('explanations', {}).get('shap')]
        lime_explanations = [exp for exp in explanations if exp.get('explanations', {}).get('lime')]
        
        # Analyze SHAP results
        if shap_explanations:
            analysis['method_comparison']['shap'] = self._analyze_shap_results(shap_explanations)
        
        # Analyze LIME results
        if lime_explanations:
            analysis['method_comparison']['lime'] = self._analyze_lime_results(lime_explanations)
        
        # Entity-level analysis
        analysis['entity_analysis'] = self._analyze_entity_patterns(explanations)
        
        # Confidence analysis
        analysis['confidence_analysis'] = self._analyze_confidence_patterns(explanations)
        
        return analysis
    
    def _analyze_shap_results(self, shap_explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze SHAP explanation results"""
        
        all_importance_scores = []
        entity_importance = {}
        
        for exp in shap_explanations:
            shap_data = exp.get('explanations', {}).get('shap', {})
            
            if 'token_explanations' in shap_data:
                for token_exp in shap_data['token_explanations']:
                    importance = token_exp.get('importance_score', 0)
                    all_importance_scores.append(importance)
                    
                    label = token_exp.get('predicted_label', 'O')
                    if label != 'O':
                        entity_type = label.split('-')[-1]
                        if entity_type not in entity_importance:
                            entity_importance[entity_type] = []
                        entity_importance[entity_type].append(importance)
        
        return {
            'total_tokens_analyzed': len(all_importance_scores),
            'average_importance': np.mean(all_importance_scores) if all_importance_scores else 0,
            'importance_std': np.std(all_importance_scores) if all_importance_scores else 0,
            'entity_importance_avg': {
                entity: np.mean(scores) for entity, scores in entity_importance.items()
            },
            'high_importance_threshold': np.percentile(all_importance_scores, 90) if all_importance_scores else 0
        }
    
    def _analyze_lime_results(self, lime_explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze LIME explanation results"""
        
        all_feature_importance = {}
        confidence_scores = []
        
        for exp in lime_explanations:
            lime_data = exp.get('explanations', {}).get('lime', {})
            
            if 'summary' in lime_data:
                summary = lime_data['summary']
                
                # Collect confidence scores
                avg_confidence = summary.get('average_prediction_confidence', 0)
                if avg_confidence > 0:
                    confidence_scores.append(avg_confidence)
                
                # Collect feature importance
                if 'most_important_features' in summary:
                    for feature, importance in summary['most_important_features']:
                        if feature not in all_feature_importance:
                            all_feature_importance[feature] = []
                        all_feature_importance[feature].append(importance)
        
        return {
            'total_features_analyzed': len(all_feature_importance),
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'top_global_features': {
                feature: np.mean(scores) 
                for feature, scores in sorted(all_feature_importance.items(), 
                                            key=lambda x: np.mean(x[1]), reverse=True)[:10]
            },
            'feature_consistency': self._calculate_feature_consistency(all_feature_importance)
        }
    
    def _calculate_feature_consistency(self, feature_importance: Dict[str, List[float]]) -> float:
        """Calculate consistency of feature importance across explanations"""
        
        if not feature_importance:
            return 0.0
        
        # Calculate coefficient of variation for each feature
        cvs = []
        for feature, scores in feature_importance.items():
            if len(scores) > 1:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                if mean_score != 0:
                    cv = std_score / abs(mean_score)
                    cvs.append(cv)
        
        # Return average consistency (lower CV = higher consistency)
        return 1 - np.mean(cvs) if cvs else 0.0
    
    def _analyze_entity_patterns(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns across entity types"""
        
        entity_stats = {}
        
        for exp in explanations:
            predictions = exp.get('prediction', {}).get('predictions', [])
            
            for pred in predictions:
                label = pred.get('label', 'O')
                if label != 'O':
                    entity_type = label.split('-')[-1]
                    
                    if entity_type not in entity_stats:
                        entity_stats[entity_type] = {
                            'count': 0,
                            'confidence_scores': [],
                            'tokens': []
                        }
                    
                    entity_stats[entity_type]['count'] += 1
                    entity_stats[entity_type]['confidence_scores'].append(pred.get('confidence', 0))
                    entity_stats[entity_type]['tokens'].append(pred.get('token', ''))
        
        # Calculate statistics
        entity_analysis = {}
        for entity_type, stats in entity_stats.items():
            entity_analysis[entity_type] = {
                'total_occurrences': stats['count'],
                'average_confidence': np.mean(stats['confidence_scores']),
                'confidence_std': np.std(stats['confidence_scores']),
                'unique_tokens': len(set(stats['tokens'])),
                'most_common_tokens': self._get_most_common_tokens(stats['tokens'])
            }
        
        return entity_analysis
    
    def _get_most_common_tokens(self, tokens: List[str], top_k: int = 5) -> List[Tuple[str, int]]:
        """Get most common tokens"""
        from collections import Counter
        return Counter(tokens).most_common(top_k)
    
    def _analyze_confidence_patterns(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze confidence patterns across predictions"""
        
        all_confidences = []
        entity_confidences = {}
        
        for exp in explanations:
            predictions = exp.get('prediction', {}).get('predictions', [])
            
            for pred in predictions:
                confidence = pred.get('confidence', 0)
                label = pred.get('label', 'O')
                
                all_confidences.append(confidence)
                
                if label != 'O':
                    entity_type = label.split('-')[-1]
                    if entity_type not in entity_confidences:
                        entity_confidences[entity_type] = []
                    entity_confidences[entity_type].append(confidence)
        
        return {
            'overall_confidence_stats': {
                'mean': np.mean(all_confidences),
                'std': np.std(all_confidences),
                'min': np.min(all_confidences),
                'max': np.max(all_confidences),
                'percentiles': {
                    '25': np.percentile(all_confidences, 25),
                    '50': np.percentile(all_confidences, 50),
                    '75': np.percentile(all_confidences, 75),
                    '90': np.percentile(all_confidences, 90)
                }
            },
            'entity_confidence_stats': {
                entity: {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'count': len(confidences)
                }
                for entity, confidences in entity_confidences.items()
            }
        }
    
    def _create_visualizations(self, explanations: List[Dict[str, Any]], 
                             analysis: Dict[str, Any]) -> Dict[str, str]:
        """Create visualization files"""
        
        viz_paths = {}
        
        # Confidence distribution plot
        confidence_plot = self._create_confidence_distribution_plot(analysis)
        if confidence_plot:
            viz_paths['confidence_distribution'] = str(confidence_plot)
        
        # Entity importance comparison
        entity_plot = self._create_entity_importance_plot(analysis)
        if entity_plot:
            viz_paths['entity_importance'] = str(entity_plot)
        
        # Method comparison plot
        method_plot = self._create_method_comparison_plot(analysis)
        if method_plot:
            viz_paths['method_comparison'] = str(method_plot)
        
        return viz_paths
    
    def _create_confidence_distribution_plot(self, analysis: Dict[str, Any]) -> Optional[Path]:
        """Create confidence distribution visualization"""
        
        confidence_stats = analysis.get('confidence_analysis', {}).get('overall_confidence_stats')
        if not confidence_stats:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram of confidence scores
        # Note: This is a simplified version - in practice, you'd need the raw data
        percentiles = confidence_stats.get('percentiles', {})
        
        if percentiles:
            x = [25, 50, 75, 90]
            y = [percentiles.get(str(p), 0) for p in x]
            
            ax.bar(x, y, alpha=0.7, color='skyblue')
            ax.set_xlabel('Confidence Percentiles')
            ax.set_ylabel('Confidence Score')
            ax.set_title('Model Confidence Distribution')
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "visualizations" / "confidence_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _create_entity_importance_plot(self, analysis: Dict[str, Any]) -> Optional[Path]:
        """Create entity importance comparison plot"""
        
        entity_analysis = analysis.get('entity_analysis', {})
        if not entity_analysis:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Entity occurrence counts
        entities = list(entity_analysis.keys())
        counts = [entity_analysis[e]['total_occurrences'] for e in entities]
        
        ax1.bar(entities, counts, color='lightcoral')
        ax1.set_xlabel('Entity Types')
        ax1.set_ylabel('Total Occurrences')
        ax1.set_title('Entity Type Frequency')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Average confidence by entity
        confidences = [entity_analysis[e]['average_confidence'] for e in entities]
        
        ax2.bar(entities, confidences, color='lightgreen')
        ax2.set_xlabel('Entity Types')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Average Confidence by Entity Type')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "visualizations" / "entity_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _create_method_comparison_plot(self, analysis: Dict[str, Any]) -> Optional[Path]:
        """Create method comparison visualization"""
        
        method_comparison = analysis.get('method_comparison', {})
        if not method_comparison:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(method_comparison.keys())
        
        # Compare average importance/confidence scores
        scores = []
        for method in methods:
            if method == 'shap':
                score = method_comparison[method].get('average_importance', 0)
            elif method == 'lime':
                score = method_comparison[method].get('average_confidence', 0)
            else:
                score = 0
            scores.append(score)
        
        ax.bar(methods, scores, color=['blue', 'orange'][:len(methods)])
        ax.set_xlabel('Explanation Methods')
        ax.set_ylabel('Average Score')
        ax.set_title('Method Comparison: Average Scores')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "visualizations" / "method_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

    def _generate_insights(self, analysis: Dict[str, Any], difficult_cases: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights and recommendations"""

        insights = {
            'model_strengths': [],
            'model_weaknesses': [],
            'recommendations': [],
            'interpretability_insights': []
        }

        # Analyze confidence patterns
        confidence_stats = analysis.get('confidence_analysis', {}).get('overall_confidence_stats', {})
        avg_confidence = confidence_stats.get('mean', 0)

        if avg_confidence > 0.8:
            insights['model_strengths'].append("High overall prediction confidence")
        elif avg_confidence < 0.6:
            insights['model_weaknesses'].append("Low overall prediction confidence")
            insights['recommendations'].append("Consider additional training or data augmentation")

        # Analyze entity performance
        entity_analysis = analysis.get('entity_analysis', {})
        for entity_type, stats in entity_analysis.items():
            avg_conf = stats.get('average_confidence', 0)
            if avg_conf > 0.85:
                insights['model_strengths'].append(f"Strong performance on {entity_type} entities")
            elif avg_conf < 0.65:
                insights['model_weaknesses'].append(f"Weak performance on {entity_type} entities")
                insights['recommendations'].append(f"Increase training data for {entity_type} entities")

        return insights

    def _create_executive_summary(self, analysis: Dict[str, Any],
                                difficult_cases: Dict[str, Any],
                                insights: Dict[str, Any],
                                model_performance: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create executive summary"""

        # Calculate key metrics
        total_cases = difficult_cases.get('total_cases', 0)
        difficult_count = difficult_cases.get('difficult_cases_count', 0)
        difficulty_rate = difficult_count / max(total_cases, 1)

        confidence_stats = analysis.get('confidence_analysis', {}).get('overall_confidence_stats', {})
        avg_confidence = confidence_stats.get('mean', 0)

        entity_count = len(analysis.get('entity_analysis', {}))

        summary = {
            'key_metrics': {
                'total_cases_analyzed': total_cases,
                'difficult_cases_rate': difficulty_rate,
                'average_confidence': avg_confidence,
                'entity_types_detected': entity_count
            },
            'model_assessment': {
                'overall_performance': 'Good' if avg_confidence > 0.75 and difficulty_rate < 0.2 else
                                     'Fair' if avg_confidence > 0.6 and difficulty_rate < 0.4 else 'Needs Improvement',
                'interpretability_score': self._calculate_interpretability_score(analysis),
                'transparency_level': 'High' if len(insights['interpretability_insights']) > 2 else 'Medium'
            },
            'top_recommendations': insights['recommendations'][:3],
            'critical_findings': self._identify_critical_findings(analysis, difficult_cases)
        }

        return summary

    def _calculate_interpretability_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate interpretability score (0-1)"""

        score = 0.0
        factors = 0

        # Factor 1: Method availability
        method_count = len(analysis.get('method_comparison', {}))
        if method_count > 0:
            score += min(method_count / 2, 1.0) * 0.3
            factors += 0.3

        # Factor 2: Confidence consistency
        confidence_stats = analysis.get('confidence_analysis', {}).get('overall_confidence_stats', {})
        confidence_std = confidence_stats.get('std', 1.0)
        if confidence_std < 0.2:
            score += 0.3
        elif confidence_std < 0.3:
            score += 0.2
        factors += 0.3

        return score / factors if factors > 0 else 0.0

    def _identify_critical_findings(self, analysis: Dict[str, Any],
                                  difficult_cases: Dict[str, Any]) -> List[str]:
        """Identify critical findings that need attention"""

        findings = []

        # Check for very low confidence entities
        entity_analysis = analysis.get('entity_analysis', {})
        for entity_type, stats in entity_analysis.items():
            if stats.get('average_confidence', 1.0) < 0.5:
                findings.append(f"Critical: Very low confidence for {entity_type} entities")

        # Check for high difficulty rate
        difficulty_rate = difficult_cases.get('difficult_cases_count', 0) / max(difficult_cases.get('total_cases', 1), 1)
        if difficulty_rate > 0.5:
            findings.append("Critical: More than 50% of cases are difficult for the model")

        return findings

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML version of the report"""

        html = f"""<!DOCTYPE html>
<html><head><title>Model Interpretability Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
.header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
.section {{ margin: 20px 0; }}
.metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
.recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
</style></head><body>
<div class="header">
<h1>Model Interpretability Report</h1>
<p>Generated: {report['metadata']['generation_time']}</p>
<p>Model: {report['metadata']['model_path']}</p>
</div>
<div class="section">
<h2>Executive Summary</h2>
<div class="metric"><strong>Overall Performance:</strong> {report['executive_summary']['model_assessment']['overall_performance']}</div>
<div class="metric"><strong>Cases Analyzed:</strong> {report['executive_summary']['key_metrics']['total_cases_analyzed']}</div>
</div>
</body></html>"""

        return html

def main():
    """Test the interpretability report generator"""
    import numpy as np
    report_gen = InterpretabilityReport()
    print("Interpretability report generator initialized successfully!")

if __name__ == "__main__":
    main()
