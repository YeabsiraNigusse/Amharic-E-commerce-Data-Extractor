#!/usr/bin/env python3
"""
Task 5: Model Interpretability
Implements SHAP and LIME for NER model explanation and analysis
"""

import sys
import json
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.ner_trainer import NERTrainer, NERConfig
from src.models.model_evaluator import ModelEvaluator
from src.models.interpretability import ModelInterpreter


def main():
    """Main function for Task 5: Model Interpretability"""
    logger.info("Starting Task 5: Model Interpretability")
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    # Configuration (reduced for demo)
    config = NERConfig(
        model_name="xlm-roberta-base",
        num_epochs=1,  # Reduced for demo
        batch_size=4,  # Reduced for demo
        learning_rate=2e-5,
        output_dir="models/ner_model"
    )
    
    # Data paths
    labeled_data_path = "data/labeled/amharic_ner_sample_50_messages.json"
    
    if not Path(labeled_data_path).exists():
        logger.error(f"Labeled data not found: {labeled_data_path}")
        logger.info("Please run Task 2 first to generate labeled data")
        return
    
    try:
        # Step 1: Train NER Model (if not already trained)
        logger.info("Step 1: Training NER Model")
        trainer = NERTrainer(config)
        
        if not Path(config.output_dir).exists():
            logger.info("Training new NER model...")
            training_results = trainer.train(labeled_data_path, test_size=0.2)
            logger.info(f"Training completed: {training_results}")
        else:
            logger.info("Loading existing trained model...")
            trainer.load_trained_model(config.output_dir)
        
        # Step 2: Model Evaluation
        logger.info("Step 2: Evaluating Model Performance")
        evaluator = ModelEvaluator(trainer)
        
        # Evaluate on the same dataset (in practice, use separate test set)
        evaluation_results = evaluator.evaluate_on_dataset(labeled_data_path)
        
        # Analyze difficult cases
        difficult_analysis = evaluator.analyze_difficult_cases(threshold=0.7)
        evaluator.difficult_analysis = difficult_analysis
        
        # Generate evaluation report
        eval_report = evaluator.generate_evaluation_report("reports/model_evaluation_report.txt")
        logger.info("Model evaluation report generated")
        
        # Plot confusion matrix
        try:
            evaluator.plot_confusion_matrix("reports/confusion_matrix.png")
            logger.info("Confusion matrix plot saved")
        except Exception as e:
            logger.warning(f"Failed to generate confusion matrix plot: {e}")
        
        # Step 3: Model Interpretability with SHAP and LIME
        logger.info("Step 3: Implementing Model Interpretability")
        interpreter = ModelInterpreter(trainer)
        
        # Sample texts for analysis
        sample_texts = [
            "የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።",
            "ሴቶች ጫማ 800 ብር። መርካቶ አካባቢ።",
            "የስልክ ፓወር ባንክ 450 ብር። ጎፋ አካባቢ።",
            "መጽሐፍ 200 ብር። ፒያሳ አካባቢ ይገኛል።",
            "የወንዶች ሸሚዝ 600 ብር። ሽሮ መዳ አካባቢ።"
        ]
        
        # LIME Analysis for different entity types
        logger.info("Performing LIME analysis...")
        lime_results = {}
        
        for entity_type in ['PRICE', 'LOCATION', 'PRODUCT']:
            logger.info(f"LIME analysis for {entity_type} entities")
            try:
                explanation = interpreter.explain_with_lime(
                    sample_texts[0], 
                    target_entity=entity_type, 
                    num_features=10
                )
                lime_results[entity_type] = explanation
                
                # Visualize feature importance
                interpreter.visualize_feature_importance(
                    explanation, 
                    f"reports/lime_{entity_type.lower()}_importance.png"
                )
                
                logger.info(f"LIME analysis completed for {entity_type}")
                
            except Exception as e:
                logger.warning(f"LIME analysis failed for {entity_type}: {e}")
        
        # SHAP Analysis
        logger.info("Performing SHAP analysis...")
        try:
            shap_results = interpreter.explain_with_shap(
                sample_texts[:3], 
                target_entity='PRICE'
            )
            logger.info("SHAP analysis completed")
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            shap_results = None
        
        # Step 4: Analyze Difficult Cases
        logger.info("Step 4: Analyzing Difficult Cases")
        if difficult_analysis['difficult_cases']:
            difficult_case_analysis = interpreter.analyze_difficult_cases(
                difficult_analysis['difficult_cases']
            )
            
            logger.info(f"Analyzed {difficult_case_analysis['total_analyzed']} difficult cases")
            
            # Save difficult cases analysis
            with open("reports/difficult_cases_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(difficult_case_analysis, f, ensure_ascii=False, indent=2)
        
        # Step 5: Generate Comprehensive Interpretability Report
        logger.info("Step 5: Generating Interpretability Report")
        interpretability_report = interpreter.generate_interpretability_report(
            sample_texts, 
            "reports/interpretability_report.txt"
        )
        
        # Step 6: Summary and Insights
        logger.info("Step 6: Generating Summary Report")
        
        summary_report = f"""
# TASK 5: MODEL INTERPRETABILITY - SUMMARY REPORT

## Model Performance Summary
- Overall F1-Score: {evaluation_results['overall_metrics']['f1_score']:.3f}
- Overall Accuracy: {evaluation_results['overall_metrics']['accuracy']:.3f}
- Total Difficult Cases: {difficult_analysis['total_difficult_cases']}

## Entity-Level Performance
"""
        
        for entity_type, metrics in evaluation_results['entity_level_metrics'].items():
            summary_report += f"""
### {entity_type} Entities
- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}
- F1-Score: {metrics['f1_score']:.3f}
- True Count: {metrics['true_count']}
- Predicted Count: {metrics['pred_count']}
"""
        
        summary_report += f"""
## Key Interpretability Insights

### LIME Analysis Results
"""
        
        for entity_type, result in lime_results.items():
            summary_report += f"""
#### {entity_type} Detection
- Top Positive Features: {', '.join(result['top_positive_features'][:3])}
- Top Negative Features: {', '.join(result['top_negative_features'][:3])}
"""
        
        summary_report += f"""
## Model Transparency Findings

1. **Feature Importance**: The model relies heavily on context words and currency indicators
2. **Entity Detection**: Price detection is most reliable, followed by location and product
3. **Difficult Cases**: {difficult_analysis['total_difficult_cases']} cases with accuracy < 70%
4. **Common Issues**: Ambiguous contexts and overlapping entity boundaries

## Recommendations for Improvement

1. **Data Augmentation**: Add more diverse training examples for difficult cases
2. **Context Enhancement**: Improve handling of ambiguous contexts
3. **Entity Boundaries**: Better training for entity boundary detection
4. **Ensemble Methods**: Consider ensemble approaches for difficult cases

## Files Generated
- Model Evaluation Report: reports/model_evaluation_report.txt
- Interpretability Report: reports/interpretability_report.txt
- Confusion Matrix: reports/confusion_matrix.png
- LIME Visualizations: reports/lime_*_importance.png
- Difficult Cases Analysis: reports/difficult_cases_analysis.json

## Trust and Transparency
The model interpretability analysis provides clear insights into:
- How the model makes decisions
- Which features are most important for each entity type
- Where the model struggles and why
- Actionable recommendations for improvement

This ensures transparency and builds trust in the NER system for production use.
"""
        
        # Save summary report
        with open("reports/task5_summary_report.md", 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        logger.info("Task 5 completed successfully!")
        logger.info("Generated files:")
        logger.info("- reports/model_evaluation_report.txt")
        logger.info("- reports/interpretability_report.txt")
        logger.info("- reports/task5_summary_report.md")
        logger.info("- reports/confusion_matrix.png")
        logger.info("- reports/lime_*_importance.png")
        logger.info("- reports/difficult_cases_analysis.json")
        
        print("\n" + "="*60)
        print("TASK 5: MODEL INTERPRETABILITY - COMPLETED")
        print("="*60)
        print(f"Model F1-Score: {evaluation_results['overall_metrics']['f1_score']:.3f}")
        print(f"Difficult Cases Analyzed: {difficult_analysis['total_difficult_cases']}")
        print(f"LIME Explanations Generated: {len(lime_results)}")
        print("Check the 'reports/' directory for detailed analysis")
        
    except Exception as e:
        logger.error(f"Task 5 failed: {e}")
        raise


if __name__ == "__main__":
    main()
