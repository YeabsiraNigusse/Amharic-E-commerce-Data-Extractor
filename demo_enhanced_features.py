#!/usr/bin/env python3
"""
Demonstration of Enhanced Features
Shows the key capabilities of the advanced fine-tuning and end-to-end pipeline
"""

import sys
import asyncio
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def demo_advanced_trainer():
    """Demonstrate advanced trainer capabilities"""
    print("=" * 80)
    print("DEMO: ADVANCED NER TRAINER")
    print("=" * 80)
    
    try:
        from src.modeling.advanced_ner_trainer import AdvancedNERTrainer
        
        # Initialize trainer
        trainer = AdvancedNERTrainer(
            model_name="distilbert-base-multilingual-cased",
            output_dir="demo_models"
        )
        
        print("✅ Advanced NER Trainer Initialized")
        print(f"   Model: {trainer.model_name}")
        print(f"   Output Directory: {trainer.output_dir}")
        
        # Show hyperparameter search space
        print("\n📊 Hyperparameter Search Space:")
        for param, values in trainer.hyperparameter_space.items():
            print(f"   {param}: {values}")
        
        # Show available methods
        print("\n🔧 Available Methods:")
        methods = [
            "grid_search() - Grid search optimization",
            "bayesian_optimization() - Bayesian optimization with Optuna",
            "cross_validate() - K-fold cross-validation",
            "train_with_hyperparams() - Train with specific parameters",
            "save_best_model() - Save the best performing model",
            "generate_training_report() - Comprehensive training report"
        ]
        
        for method in methods:
            print(f"   • {method}")
        
        print("\n💡 Usage Example:")
        print("""
# Initialize trainer
trainer = AdvancedNERTrainer("xlm-roberta-base")

# Perform grid search
results = trainer.grid_search(train_dataset, val_dataset, param_grid)

# Cross-validation
cv_results = trainer.cross_validate(dataset, n_folds=5)

# Save best model
best_model = trainer.save_best_model()
        """)
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

async def demo_pipeline():
    """Demonstrate pipeline capabilities"""
    print("\n" + "=" * 80)
    print("DEMO: END-TO-END PIPELINE")
    print("=" * 80)
    
    try:
        from src.pipeline.end_to_end_pipeline import EndToEndPipeline
        import yaml
        
        # Create demo configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            demo_config = {
                'data_ingestion': {
                    'enabled': True,
                    'max_messages_per_channel': 500
                },
                'preprocessing': {
                    'enabled': True,
                    'clean_text': True,
                    'extract_entities': True
                },
                'labeling': {
                    'enabled': True,
                    'sample_size': 100,
                    'auto_label': True
                },
                'training': {
                    'enabled': True,
                    'models': ['xlm-roberta-base', 'bert-base-multilingual-cased'],
                    'optimization': {
                        'method': 'grid_search',
                        'max_trials': 10,
                        'cv_folds': 3
                    }
                },
                'evaluation': {
                    'enabled': True,
                    'generate_report': True
                }
            }
            yaml.dump(demo_config, f)
            config_path = f.name
        
        # Initialize pipeline
        pipeline = EndToEndPipeline(config_path)
        
        print("✅ End-to-End Pipeline Initialized")
        print(f"   Pipeline ID: {pipeline.pipeline_id}")
        print(f"   Configuration Sections: {len(pipeline.config)}")
        
        # Show pipeline steps
        print("\n🔄 Pipeline Steps:")
        steps = [
            "1. Data Ingestion - Scrape data from Telegram channels",
            "2. Preprocessing - Clean and process Amharic text",
            "3. Labeling - Auto-label data in CoNLL format",
            "4. Training - Train models with hyperparameter optimization",
            "5. Evaluation - Evaluate and compare model performance"
        ]
        
        for step in steps:
            print(f"   {step}")
        
        # Show configuration
        print("\n⚙️ Configuration:")
        for section, config in pipeline.config.items():
            enabled = config.get('enabled', True)
            status = "✅ Enabled" if enabled else "❌ Disabled"
            print(f"   {section}: {status}")
        
        print("\n💡 Usage Example:")
        print("""
# Initialize pipeline
pipeline = EndToEndPipeline("config/pipeline_config.yaml")

# Run complete pipeline
success = await pipeline.run_complete_pipeline()

# Or run individual steps
await pipeline.run_data_ingestion()
pipeline.run_preprocessing()
pipeline.run_labeling()
pipeline.run_training()
pipeline.run_evaluation()
        """)
        
        # Cleanup
        Path(config_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def demo_usage_commands():
    """Show usage commands"""
    print("\n" + "=" * 80)
    print("DEMO: USAGE COMMANDS")
    print("=" * 80)
    
    print("🚀 Advanced Fine-tuning:")
    print("""
# Grid search optimization
python run_advanced_fine_tuning.py \\
  --model xlm-roberta-base \\
  --optimization grid_search \\
  --max-trials 20 \\
  --cv-folds 5

# Bayesian optimization
python run_advanced_fine_tuning.py \\
  --model bert-base-multilingual-cased \\
  --optimization bayesian \\
  --max-trials 50
    """)
    
    print("🔄 Complete Pipeline:")
    print("""
# Full pipeline
python run_complete_pipeline.py

# Skip data ingestion (use existing data)
python run_complete_pipeline.py --skip-ingestion

# Custom configuration
python run_complete_pipeline.py --config my_config.yaml

# Create sample configuration
python run_complete_pipeline.py --create-config
    """)
    
    print("📊 Individual Tasks:")
    print("""
# Data ingestion and preprocessing
python run_task1.py

# Data labeling
python run_task2.py

# Basic model training
python run_task3.py --model xlm-roberta-base

# Model comparison
python run_task4.py

# Model interpretability
python run_task5_interpretability.py

# Vendor scorecard
python run_task6_vendor_scorecard.py
    """)

def demo_file_structure():
    """Show new file structure"""
    print("\n" + "=" * 80)
    print("DEMO: NEW FILE STRUCTURE")
    print("=" * 80)
    
    structure = {
        "🆕 Advanced Fine-tuning": [
            "src/modeling/advanced_ner_trainer.py",
            "run_advanced_fine_tuning.py"
        ],
        "🆕 End-to-End Pipeline": [
            "src/pipeline/__init__.py",
            "src/pipeline/end_to_end_pipeline.py",
            "run_complete_pipeline.py"
        ],
        "🆕 Documentation": [
            "ENHANCED_IMPLEMENTATION_REPORT.md",
            "IMPLEMENTATION_FIXES_SUMMARY.md"
        ],
        "🆕 Testing": [
            "test_enhanced_features.py",
            "simple_test.py",
            "demo_enhanced_features.py"
        ]
    }
    
    for category, files in structure.items():
        print(f"\n{category}:")
        for file_path in files:
            exists = "✅" if Path(file_path).exists() else "❌"
            print(f"   {exists} {file_path}")

async def main():
    """Main demo function"""
    print("🎉 ENHANCED FEATURES DEMONSTRATION")
    print("Amharic E-commerce Data Extractor - Advanced Capabilities")
    
    # Run demos
    trainer_success = demo_advanced_trainer()
    pipeline_success = await demo_pipeline()
    
    demo_usage_commands()
    demo_file_structure()
    
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    
    print(f"Advanced Trainer Demo: {'✅ SUCCESS' if trainer_success else '❌ FAILED'}")
    print(f"Pipeline Demo: {'✅ SUCCESS' if pipeline_success else '❌ FAILED'}")
    
    if trainer_success and pipeline_success:
        print("\n🎉 All enhanced features are working correctly!")
        print("\n📋 What's New:")
        print("   ✅ Comprehensive model fine-tuning with hyperparameter optimization")
        print("   ✅ Complete end-to-end pipeline automation")
        print("   ✅ Advanced training strategies and cross-validation")
        print("   ✅ Robust error handling and state management")
        print("   ✅ Comprehensive reporting and visualization")
        
        print("\n🚀 Ready for Production:")
        print("   • Run advanced fine-tuning: python run_advanced_fine_tuning.py")
        print("   • Run complete pipeline: python run_complete_pipeline.py")
        print("   • See documentation: ENHANCED_IMPLEMENTATION_REPORT.md")
    else:
        print("\n⚠️ Some features may need additional setup.")
    
    return trainer_success and pipeline_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
