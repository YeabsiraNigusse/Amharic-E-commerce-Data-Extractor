#!/usr/bin/env python3
"""
Demo script for model comparison system
Creates mock results to demonstrate the comparison functionality
"""

import sys
import json
from pathlib import Path
import random
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from loguru import logger
from src.modeling.model_evaluator import ModelEvaluator
from src.utils.data_utils import setup_logging

def create_mock_model_results():
    """Create mock model evaluation results for demonstration"""
    
    models = [
        "xlm-roberta-base",
        "bert-base-multilingual-cased", 
        "distilbert-base-multilingual-cased"
    ]
    
    mock_results = []
    
    for i, model_name in enumerate(models):
        # Create realistic but mock performance metrics
        base_f1 = 0.75 + random.uniform(-0.1, 0.15)  # F1 between 0.65-0.90
        precision = base_f1 + random.uniform(-0.05, 0.05)
        recall = base_f1 + random.uniform(-0.05, 0.05)
        accuracy = base_f1 + random.uniform(-0.02, 0.08)
        
        # Inference time varies by model size
        if "large" in model_name:
            inference_time = random.uniform(0.8, 1.2)
        elif "distil" in model_name:
            inference_time = random.uniform(0.2, 0.4)
        else:
            inference_time = random.uniform(0.4, 0.8)
        
        # Model parameters (approximate)
        if "xlm-roberta" in model_name:
            num_params = 270_000_000
        elif "distil" in model_name:
            num_params = 66_000_000
        else:
            num_params = 110_000_000
        
        # Entity-level metrics
        entity_metrics = {
            'PRICE': {
                'precision': precision + random.uniform(-0.1, 0.1),
                'recall': recall + random.uniform(-0.1, 0.1),
                'f1': base_f1 + random.uniform(-0.1, 0.1),
                'support': random.randint(80, 120)
            },
            'LOCATION': {
                'precision': precision + random.uniform(-0.15, 0.1),
                'recall': recall + random.uniform(-0.15, 0.1),
                'f1': base_f1 + random.uniform(-0.15, 0.1),
                'support': random.randint(30, 50)
            },
            'CONTACT_INFO': {
                'precision': precision + random.uniform(-0.2, 0.2),
                'recall': recall + random.uniform(-0.2, 0.2),
                'f1': base_f1 + random.uniform(-0.2, 0.2),
                'support': random.randint(3, 8)
            }
        }
        
        result = {
            'model_name': model_name,
            'model_path': f"models/{model_name.replace('/', '_')}_finetuned",
            'overall_precision': max(0, min(1, precision)),
            'overall_recall': max(0, min(1, recall)),
            'overall_f1': max(0, min(1, base_f1)),
            'overall_accuracy': max(0, min(1, accuracy)),
            'entity_metrics': entity_metrics,
            'inference_time_total': inference_time * 10,  # For 10 samples
            'inference_time_avg': inference_time,
            'num_parameters': num_params,
            'test_samples': 10
        }
        
        mock_results.append(result)
    
    return mock_results

def demo_model_comparison():
    """Demonstrate the model comparison functionality"""
    
    logger.info("=" * 60)
    logger.info("DEMO: Model Comparison System")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(str(output_dir))
    
    # Create mock results
    logger.info("Creating mock model evaluation results...")
    mock_results = create_mock_model_results()
    
    # Display individual results
    logger.info("\n📊 INDIVIDUAL MODEL RESULTS:")
    logger.info("-" * 80)
    for result in mock_results:
        logger.info(f"Model: {result['model_name']}")
        logger.info(f"  F1 Score: {result['overall_f1']:.4f}")
        logger.info(f"  Precision: {result['overall_precision']:.4f}")
        logger.info(f"  Recall: {result['overall_recall']:.4f}")
        logger.info(f"  Inference Time: {result['inference_time_avg']:.4f}s")
        logger.info(f"  Parameters: {result['num_parameters']:,}")
        logger.info("")
    
    # Create comparison report
    logger.info("Generating comparison report...")
    comparison_report = evaluator.create_evaluation_report(mock_results)
    
    # Display comparison table
    comparison_df = evaluator.compare_models(mock_results)
    logger.info("\n📋 MODEL COMPARISON TABLE:")
    logger.info("-" * 80)
    print(comparison_df.to_string(index=False))
    
    # Select best model
    logger.info("\n🏆 BEST MODEL SELECTION:")
    logger.info("-" * 40)
    
    # Best F1 score
    best_f1 = max(mock_results, key=lambda x: x['overall_f1'])
    logger.info(f"Best F1 Score: {best_f1['model_name']} ({best_f1['overall_f1']:.4f})")
    
    # Fastest inference
    fastest = min(mock_results, key=lambda x: x['inference_time_avg'])
    logger.info(f"Fastest Inference: {fastest['model_name']} ({fastest['inference_time_avg']:.4f}s)")
    
    # Most balanced (F1/speed ratio)
    balanced_scores = [(r, r['overall_f1'] / (r['inference_time_avg'] + 1)) for r in mock_results]
    best_balanced = max(balanced_scores, key=lambda x: x[1])
    logger.info(f"Most Balanced: {best_balanced[0]['model_name']} (ratio: {best_balanced[1]:.4f})")
    
    # Save results
    results_file = output_dir / "demo_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'description': 'Demo model comparison results',
            'model_results': mock_results,
            'comparison_summary': comparison_report
        }, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n💾 Results saved to: {results_file}")
    logger.info(f"📊 Plots saved to: {output_dir}/model_comparison_plots.png")
    logger.info(f"📋 CSV saved to: {output_dir}/model_comparison.csv")
    
    logger.info("\n✅ Demo completed successfully!")
    logger.info("This demonstrates the model comparison functionality.")
    logger.info("In actual usage, these would be real trained model results.")

def main():
    """Main function"""
    setup_logging()
    
    logger.info("Starting Model Comparison Demo")
    logger.info("This demo shows how the model comparison system works")
    logger.info("with mock data (no actual training required)")
    
    try:
        demo_model_comparison()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
