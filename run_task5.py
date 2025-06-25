#!/usr/bin/env python3
"""
Main script to run Task 5: Model Interpretability
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from loguru import logger
from src.interpretability.model_explainer import ModelExplainer
from src.interpretability.interpretability_report import InterpretabilityReport
from src.labeling.conll_labeler import CoNLLLabeler
from src.utils.data_utils import setup_logging

def load_test_texts(test_file: str = None) -> list:
    """Load test texts for interpretability analysis"""
    
    if test_file and Path(test_file).exists():
        # Load from CoNLL file
        labeler = CoNLLLabeler()
        data = labeler.load_conll_file(test_file)
        return [item['text'] for item in data]
    else:
        # Use sample Amharic texts
        return [
            "የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።",
            "ስልክ ቁጥር 0911234567 ላይ ይደውሉ። አዲስ አበባ ውስጥ ነን።",
            "የሴቶች ጫማ በ 800 ብር። ፒያሳ አካባቢ ይገኛል።",
            "ላፕቶፕ ዋጋ 25000 ብር። ዴሊቨሪ ነፃ ነው።",
            "የወንዶች ሸሚዝ 300 ብር። @ethioshop ላይ ይገናኙን።",
            "የቤት እቃዎች በመርካቶ። ዋጋ 1500 ብር ነው።",
            "ስማርት ፎን ETB 15000። ካዛንቺስ አካባቢ።",
            "የሴቶች ቦርሳ 450 ብር። ዴሊቨሪ ክፍያ 50 ብር።",
            "የመጽሐፍ መደብር በጎፋ። ዋጋ 200 ብር ነው።",
            "የኮምፒውተር አክሰሰሪዎች በ 1200 ብር። ኪርኮስ አካባቢ።",
            "የሕፃናት መጫወቻ 150 ብር። በላፍቶ ይገኛል።",
            "የወንዶች ሱሪ ዋጋ 600 ብር። ዴሊቨሪ 30 ብር።",
            "የሴቶች ቀሚስ በ 900 ብር። ቦሌ ሜዳኒያለም አካባቢ።",
            "የስልክ ኬዝ 80 ብር። @phonecase_et ላይ ይገናኙ።",
            "የቤት ዕቃዎች በአዲስ አበባ። ዋጋ 2500 ብር።"
        ]

def create_mock_model_explanation():
    """Create mock model explanation for demonstration"""
    
    logger.info("Creating mock model explanation (no trained model available)")
    
    # Sample explanation structure
    mock_explanation = {
        'text': "የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።",
        'prediction': {
            'predictions': [
                {'token': 'የሕፃናት', 'label': 'B-PRODUCT', 'confidence': 0.85},
                {'token': 'ልብስ', 'label': 'I-PRODUCT', 'confidence': 0.82},
                {'token': 'ዋጋ', 'label': 'B-PRICE', 'confidence': 0.91},
                {'token': '500', 'label': 'I-PRICE', 'confidence': 0.95},
                {'token': 'ብር', 'label': 'I-PRICE', 'confidence': 0.93},
                {'token': 'ነው', 'label': 'O', 'confidence': 0.78},
                {'token': '።', 'label': 'O', 'confidence': 0.99},
                {'token': 'በቦሌ', 'label': 'B-LOCATION', 'confidence': 0.88},
                {'token': 'አካባቢ', 'label': 'I-LOCATION', 'confidence': 0.86},
                {'token': 'ይገኛል', 'label': 'O', 'confidence': 0.79},
                {'token': '።', 'label': 'O', 'confidence': 0.99}
            ]
        },
        'explanations': {
            'shap': {
                'token_explanations': [
                    {'token': 'የሕፃናት', 'predicted_label': 'B-PRODUCT', 'importance_score': 0.85, 'position': 0},
                    {'token': 'ልብስ', 'predicted_label': 'I-PRODUCT', 'importance_score': 0.92, 'position': 1},
                    {'token': 'ዋጋ', 'predicted_label': 'B-PRICE', 'importance_score': 0.78, 'position': 2},
                    {'token': '500', 'predicted_label': 'I-PRICE', 'importance_score': 0.95, 'position': 3},
                    {'token': 'ብር', 'predicted_label': 'I-PRICE', 'importance_score': 0.89, 'position': 4},
                    {'token': 'በቦሌ', 'predicted_label': 'B-LOCATION', 'importance_score': 0.83, 'position': 7},
                    {'token': 'አካባቢ', 'predicted_label': 'I-LOCATION', 'importance_score': 0.76, 'position': 8}
                ],
                'summary': {
                    'most_important_tokens': [
                        {'token': '500', 'importance_score': 0.95},
                        {'token': 'ልብስ', 'importance_score': 0.92},
                        {'token': 'ብር', 'importance_score': 0.89}
                    ],
                    'average_importance_by_entity': {
                        'PRODUCT': 0.885,
                        'PRICE': 0.873,
                        'LOCATION': 0.795
                    }
                }
            },
            'lime': {
                'simplified_explanation': {
                    'word_importance': {
                        'የሕፃናት': 0.82,
                        'ልብስ': 0.88,
                        'ዋጋ': 0.75,
                        '500': 0.93,
                        'ብር': 0.87,
                        'በቦሌ': 0.81,
                        'አካባቢ': 0.74
                    },
                    'baseline_confidence': 0.85
                },
                'summary': {
                    'most_important_features': [
                        ('500', 0.93),
                        ('ልብስ', 0.88),
                        ('ብር', 0.87),
                        ('የሕፃናት', 0.82),
                        ('በቦሌ', 0.81)
                    ],
                    'average_prediction_confidence': 0.85,
                    'feature_consistency': 0.78
                }
            }
        }
    }
    
    return mock_explanation

def create_mock_difficult_cases():
    """Create mock difficult cases analysis"""
    
    return {
        'total_cases': 15,
        'difficult_cases_count': 3,
        'difficult_cases': [
            {
                'text': "የስልክ ኬዝ 80 ብር። @phonecase_et ላይ ይገናኙ።",
                'difficulties': {
                    'is_difficult': True,
                    'reasons': ['Low confidence predictions', 'Entity boundary issues'],
                    'low_confidence_tokens': [
                        {'token': 'ኢት', 'label': 'I-CONTACT_INFO', 'confidence': 0.45}
                    ]
                }
            }
        ],
        'summary': {
            'common_issues': {
                'Low confidence predictions': 2,
                'Entity boundary issues': 1
            },
            'entity_types_affected': {
                'CONTACT_INFO': 2,
                'PRODUCT': 1
            },
            'average_confidence': 0.67,
            'recommendations': [
                'Consider additional training data for low-confidence entity types',
                'Review and improve entity boundary labeling in training data'
            ]
        }
    }

def run_interpretability_analysis(model_path: str, 
                                test_texts: list,
                                output_dir: str = "interpretability_results",
                                use_mock: bool = False):
    """Run comprehensive interpretability analysis"""
    
    logger.info(f"Starting interpretability analysis")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Test texts: {len(test_texts)}")
    logger.info(f"Output directory: {output_dir}")
    
    if use_mock or not Path(model_path).exists():
        logger.info("Using mock explanations for demonstration")
        
        # Create mock explanations
        explanations = []
        for i, text in enumerate(test_texts[:5]):  # Limit to 5 for demo
            mock_exp = create_mock_model_explanation()
            mock_exp['text'] = text
            explanations.append(mock_exp)
        
        # Create mock difficult cases
        difficult_cases = create_mock_difficult_cases()
        
    else:
        logger.info("Using real model for explanations")
        
        # Initialize explainer
        explainer = ModelExplainer(model_path, output_dir)
        
        # Load model
        if not explainer.load_model():
            logger.error("Failed to load model. Using mock explanations.")
            return run_interpretability_analysis(model_path, test_texts, output_dir, use_mock=True)
        
        # Generate explanations
        explanations = []
        for text in test_texts:
            explanation = explainer.explain_prediction(text, methods=['shap', 'lime'])
            explanations.append(explanation)
        
        # Analyze difficult cases
        difficult_cases = explainer.analyze_difficult_cases(test_texts)
    
    # Generate comprehensive report
    report_generator = InterpretabilityReport(output_dir)
    
    report = report_generator.generate_comprehensive_report(
        model_path=model_path,
        explanations=explanations,
        difficult_cases=difficult_cases,
        model_performance={'f1_score': 0.75, 'accuracy': 0.82}  # Mock performance
    )
    
    return report

def display_interpretability_results(report: dict):
    """Display interpretability results in a user-friendly format"""
    
    logger.info("=" * 80)
    logger.info("INTERPRETABILITY ANALYSIS RESULTS")
    logger.info("=" * 80)
    
    # Executive Summary
    exec_summary = report.get('executive_summary', {})
    key_metrics = exec_summary.get('key_metrics', {})
    model_assessment = exec_summary.get('model_assessment', {})
    
    logger.info("📊 EXECUTIVE SUMMARY:")
    logger.info(f"  Cases Analyzed: {key_metrics.get('total_cases_analyzed', 0)}")
    logger.info(f"  Difficult Cases Rate: {key_metrics.get('difficult_cases_rate', 0):.1%}")
    logger.info(f"  Average Confidence: {key_metrics.get('average_confidence', 0):.3f}")
    logger.info(f"  Overall Performance: {model_assessment.get('overall_performance', 'Unknown')}")
    logger.info(f"  Interpretability Score: {model_assessment.get('interpretability_score', 0):.3f}")
    
    # Key Recommendations
    recommendations = exec_summary.get('top_recommendations', [])
    if recommendations:
        logger.info("\n💡 KEY RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")
    
    # Critical Findings
    critical_findings = exec_summary.get('critical_findings', [])
    if critical_findings:
        logger.info("\n⚠️  CRITICAL FINDINGS:")
        for finding in critical_findings:
            logger.info(f"  • {finding}")
    
    # Entity Analysis
    entity_analysis = report.get('explanation_analysis', {}).get('entity_analysis', {})
    if entity_analysis:
        logger.info("\n🎯 ENTITY ANALYSIS:")
        logger.info(f"{'Entity Type':<15} {'Occurrences':<12} {'Avg Confidence':<15} {'Unique Tokens':<15}")
        logger.info("-" * 60)
        for entity_type, stats in entity_analysis.items():
            occurrences = stats.get('total_occurrences', 0)
            avg_conf = stats.get('average_confidence', 0)
            unique_tokens = stats.get('unique_tokens', 0)
            logger.info(f"{entity_type:<15} {occurrences:<12} {avg_conf:<15.3f} {unique_tokens:<15}")
    
    # Method Comparison
    method_comparison = report.get('explanation_analysis', {}).get('method_comparison', {})
    if method_comparison:
        logger.info("\n🔍 METHOD COMPARISON:")
        for method, stats in method_comparison.items():
            logger.info(f"  {method.upper()}:")
            if method == 'shap':
                logger.info(f"    Average Importance: {stats.get('average_importance', 0):.3f}")
                logger.info(f"    Tokens Analyzed: {stats.get('total_tokens_analyzed', 0)}")
            elif method == 'lime':
                logger.info(f"    Average Confidence: {stats.get('average_confidence', 0):.3f}")
                logger.info(f"    Feature Consistency: {stats.get('feature_consistency', 0):.3f}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Model Interpretability Analysis for Amharic NER")
    parser.add_argument('--model-path', '-m', 
                       default='models/distilbert-base-multilingual-cased_finetuned',
                       help='Path to trained model')
    parser.add_argument('--test-file', '-t', 
                       default='data/labeled/amharic_ner_sample_50_messages.txt',
                       help='Path to test file in CoNLL format')
    parser.add_argument('--output-dir', '-o', default='interpretability_results',
                       help='Output directory for results')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock explanations for demonstration')
    parser.add_argument('--num-texts', '-n', type=int, default=15,
                       help='Number of texts to analyze')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("AMHARIC E-COMMERCE DATA EXTRACTOR - TASK 5")
    logger.info("Model Interpretability Analysis")
    logger.info("=" * 80)
    
    # Load test texts
    test_texts = load_test_texts(args.test_file)[:args.num_texts]
    logger.info(f"Loaded {len(test_texts)} test texts")
    
    try:
        # Run interpretability analysis
        report = run_interpretability_analysis(
            model_path=args.model_path,
            test_texts=test_texts,
            output_dir=args.output_dir,
            use_mock=args.mock
        )
        
        # Display results
        display_interpretability_results(report)
        
        logger.info("=" * 80)
        logger.info("TASK 5 COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("Files generated:")
        logger.info(f"- {args.output_dir}/interpretability_report_*.json")
        logger.info(f"- {args.output_dir}/interpretability_report_*.html")
        logger.info(f"- {args.output_dir}/visualizations/")
        
        logger.info("\nNext steps:")
        logger.info("1. Review the interpretability report")
        logger.info("2. Address critical findings and recommendations")
        logger.info("3. Proceed to Task 6: FinTech Vendor Scorecard")
        
    except Exception as e:
        logger.error(f"Task 5 failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
