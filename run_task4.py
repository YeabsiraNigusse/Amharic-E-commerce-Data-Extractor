#!/usr/bin/env python3
"""
Main script to run Task 4: Model Comparison & Selection
"""

import sys
import json
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from loguru import logger
from src.modeling.model_comparison import ModelComparison
from src.labeling.conll_labeler import CoNLLLabeler
from src.utils.data_utils import setup_logging

def load_test_data(test_file: str):
    """Load test data for evaluation"""
    if Path(test_file).exists():
        labeler = CoNLLLabeler()
        return labeler.load_conll_file(test_file)
    else:
        logger.warning(f"Test file not found: {test_file}")
        return []

def run_model_comparison(models: list, 
                        train_file: str, 
                        test_file: str = None,
                        parallel: bool = False,
                        val_split: float = 0.2):
    """Run comprehensive model comparison"""
    
    logger.info(f"Starting model comparison for: {models}")
    
    # Initialize comparison system
    comparison = ModelComparison()
    
    # Train multiple models
    training_results = comparison.train_multiple_models(
        model_names=models,
        train_file=train_file,
        val_split=val_split,
        parallel=parallel
    )
    
    # Report training results
    logger.info("Training Results Summary:")
    for model_name, result in training_results.items():
        if result.get('status') == 'success':
            eval_f1 = result.get('eval_results', {}).get('eval_f1', 'N/A')
            logger.info(f"✅ {model_name}: F1 = {eval_f1}")
        else:
            logger.error(f"❌ {model_name}: {result.get('error', 'Unknown error')}")
    
    # Load test data for evaluation
    if test_file and Path(test_file).exists():
        test_data = load_test_data(test_file)
    else:
        # Use a portion of training data as test data
        logger.info("Using validation data for final evaluation")
        test_data = load_test_data(train_file)
        # Take last 20% as test data
        test_size = max(1, len(test_data) // 5)
        test_data = test_data[-test_size:]
    
    if not test_data:
        logger.error("No test data available for evaluation")
        return None
    
    logger.info(f"Evaluating models on {len(test_data)} test samples")
    
    # Evaluate all trained models
    comparison_results = comparison.evaluate_trained_models(test_data)
    
    if not comparison_results:
        logger.error("Model evaluation failed")
        return None
    
    # Select best model
    best_model_selection = comparison.select_best_model(
        criteria="f1",
        min_f1=0.0,
        max_inference_time=10.0  # 10 seconds max
    )
    
    # Get recommendations
    recommendations = comparison.get_model_recommendations()
    
    return {
        'training_results': training_results,
        'comparison_results': comparison_results,
        'best_model_selection': best_model_selection,
        'recommendations': recommendations
    }

def display_results(results):
    """Display comparison results in a user-friendly format"""
    
    if not results:
        logger.error("No results to display")
        return
    
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 80)
    
    # Training Summary
    training_results = results.get('training_results', {})
    successful_models = [name for name, result in training_results.items() 
                        if result.get('status') == 'success']
    failed_models = [name for name, result in training_results.items() 
                    if result.get('status') == 'failed']
    
    logger.info(f"✅ Successfully trained: {len(successful_models)} models")
    logger.info(f"❌ Failed training: {len(failed_models)} models")
    
    if failed_models:
        logger.info(f"Failed models: {', '.join(failed_models)}")
    
    # Evaluation Results
    comparison_results = results.get('comparison_results', {})
    model_results = comparison_results.get('model_results', [])
    
    if model_results:
        logger.info("\n📊 EVALUATION METRICS:")
        logger.info("-" * 60)
        logger.info(f"{'Model':<30} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Speed(s)':<10}")
        logger.info("-" * 60)
        
        for result in model_results:
            if 'error' not in result:
                name = result['model_name'][:28]
                f1 = f"{result.get('overall_f1', 0):.4f}"
                precision = f"{result.get('overall_precision', 0):.4f}"
                recall = f"{result.get('overall_recall', 0):.4f}"
                speed = f"{result.get('inference_time_avg', 0):.4f}"
                
                logger.info(f"{name:<30} {f1:<8} {precision:<10} {recall:<8} {speed:<10}")
    
    # Best Model Selection
    best_selection = results.get('best_model_selection', {})
    if best_selection and 'selected_model' in best_selection:
        best_model = best_selection['selected_model']
        logger.info(f"\n🏆 BEST MODEL SELECTED:")
        logger.info(f"Model: {best_model['model_name']}")
        logger.info(f"F1 Score: {best_model.get('overall_f1', 'N/A'):.4f}")
        logger.info(f"Precision: {best_model.get('overall_precision', 'N/A'):.4f}")
        logger.info(f"Recall: {best_model.get('overall_recall', 'N/A'):.4f}")
        logger.info(f"Inference Time: {best_model.get('inference_time_avg', 'N/A'):.4f}s")
    
    # Recommendations
    recommendations = results.get('recommendations', {})
    if recommendations and 'error' not in recommendations:
        logger.info(f"\n💡 RECOMMENDATIONS:")
        for use_case, rec in recommendations.items():
            logger.info(f"{use_case.replace('_', ' ').title()}: {rec['model']} - {rec['reason']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compare and select NER models for Amharic e-commerce")
    parser.add_argument('--models', '-m', nargs='+', 
                       default=['xlm-roberta-base', 'bert-base-multilingual-cased', 'distilbert-base-multilingual-cased'],
                       help='Models to compare (default: xlm-roberta-base bert-base-multilingual-cased distilbert-base-multilingual-cased)')
    parser.add_argument('--train-file', '-t', default='data/labeled/amharic_ner_sample_50_messages.txt',
                       help='Path to training file in CoNLL format')
    parser.add_argument('--test-file', default=None,
                       help='Path to test file (optional, will use validation split if not provided)')
    parser.add_argument('--parallel', action='store_true',
                       help='Train models in parallel (requires sufficient GPU memory)')
    parser.add_argument('--val-split', '-v', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--quick', action='store_true',
                       help='Quick comparison with reduced training (for testing)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("AMHARIC E-COMMERCE DATA EXTRACTOR - TASK 4")
    logger.info("Model Comparison & Selection")
    logger.info("=" * 80)
    
    # Check if training file exists
    if not Path(args.train_file).exists():
        logger.error(f"Training file not found: {args.train_file}")
        logger.info("Please run Task 2 first to generate labeled data")
        return
    
    # Quick mode for testing
    if args.quick:
        logger.info("Running in quick mode (reduced training for testing)")
        args.models = args.models[:2]  # Limit to 2 models
        logger.info(f"Testing with models: {args.models}")
    
    try:
        # Run model comparison
        results = run_model_comparison(
            models=args.models,
            train_file=args.train_file,
            test_file=args.test_file,
            parallel=args.parallel,
            val_split=args.val_split
        )
        
        if results:
            # Display results
            display_results(results)
            
            # Save detailed results
            results_file = Path("comparison_results") / "detailed_comparison_results.json"
            results_file.parent.mkdir(exist_ok=True)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("=" * 80)
            logger.info("TASK 4 COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Detailed results saved to: {results_file}")
            logger.info("Files generated:")
            logger.info("- comparison_results/model_comparison.csv")
            logger.info("- comparison_results/evaluation_report.json")
            logger.info("- comparison_results/best_model_selection.json")
            logger.info("- comparison_results/model_comparison_plots.png")
            
            logger.info("\nNext steps:")
            logger.info("1. Review the best model selection")
            logger.info("2. Use the selected model for production")
            logger.info("3. Implement model interpretability (Task 5)")
            logger.info("4. Develop FinTech vendor scorecard (Task 6)")
        
        else:
            logger.error("Model comparison failed")
    
    except Exception as e:
        logger.error(f"Task 4 failed: {e}")
        logger.error("Please check the error logs and try again")

if __name__ == "__main__":
    main()
