#!/usr/bin/env python3
"""
Complete End-to-End Pipeline Runner
Demonstrates the full workflow from data ingestion to model deployment
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from loguru import logger
from src.pipeline.end_to_end_pipeline import EndToEndPipeline
from src.utils.data_utils import setup_logging

async def run_complete_pipeline(
    config_path: str = "config/pipeline_config.yaml",
    skip_ingestion: bool = False,
    skip_preprocessing: bool = False,
    skip_labeling: bool = False,
    skip_training: bool = False,
    skip_evaluation: bool = False
):
    """Run the complete end-to-end pipeline with optional step skipping"""
    
    logger.info("=" * 100)
    logger.info("AMHARIC E-COMMERCE DATA EXTRACTOR - COMPLETE PIPELINE")
    logger.info("=" * 100)
    
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables if ingestion is enabled
    if not skip_ingestion:
        required_vars = ['TELEGRAM_API_ID', 'TELEGRAM_API_HASH', 'TELEGRAM_PHONE']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            logger.info("Please set these variables in your .env file or environment")
            logger.info("You can skip data ingestion with --skip-ingestion if you have existing data")
            return False
    
    try:
        # Initialize pipeline
        pipeline = EndToEndPipeline(config_path)
        
        # Modify configuration based on skip flags
        if skip_ingestion:
            pipeline.config['data_ingestion']['enabled'] = False
            logger.info("Data ingestion disabled")
        
        if skip_preprocessing:
            pipeline.config['preprocessing']['enabled'] = False
            logger.info("Preprocessing disabled")
        
        if skip_labeling:
            pipeline.config['labeling']['enabled'] = False
            logger.info("Labeling disabled")
        
        if skip_training:
            pipeline.config['training']['enabled'] = False
            logger.info("Training disabled")
        
        if skip_evaluation:
            pipeline.config['evaluation']['enabled'] = False
            logger.info("Evaluation disabled")
        
        # Display pipeline configuration
        logger.info("\nPipeline Configuration:")
        logger.info(f"  Data Ingestion: {'Enabled' if pipeline.config['data_ingestion']['enabled'] else 'Disabled'}")
        logger.info(f"  Preprocessing: {'Enabled' if pipeline.config['preprocessing']['enabled'] else 'Disabled'}")
        logger.info(f"  Labeling: {'Enabled' if pipeline.config['labeling']['enabled'] else 'Disabled'}")
        logger.info(f"  Training: {'Enabled' if pipeline.config['training']['enabled'] else 'Disabled'}")
        logger.info(f"  Evaluation: {'Enabled' if pipeline.config['evaluation']['enabled'] else 'Disabled'}")
        
        if pipeline.config['training']['enabled']:
            logger.info(f"  Models to train: {pipeline.config['training']['models']}")
            logger.info(f"  Optimization method: {pipeline.config['training']['optimization']['method']}")
        
        # Run the complete pipeline
        success = await pipeline.run_complete_pipeline()
        
        if success:
            logger.info("\n" + "=" * 100)
            logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 100)
            
            # Display summary
            state = pipeline.pipeline_state
            logger.info(f"Pipeline ID: {state['pipeline_id']}")
            logger.info(f"Status: {state['status']}")
            logger.info(f"Steps completed: {len(state['steps_completed'])}/5")
            logger.info(f"Completed steps: {', '.join(state['steps_completed'])}")
            
            # Display key results
            if 'data_ingestion' in state['results']:
                result = state['results']['data_ingestion']
                logger.info(f"Messages collected: {result.get('messages_collected', 'N/A')}")
            
            if 'preprocessing' in state['results']:
                result = state['results']['preprocessing']
                logger.info(f"Messages processed: {result.get('total_messages', 'N/A')}")
                logger.info(f"Amharic messages: {result.get('amharic_messages', 'N/A')}")
            
            if 'labeling' in state['results']:
                result = state['results']['labeling']
                logger.info(f"Messages labeled: {result.get('total_messages_labeled', 'N/A')}")
            
            if 'training' in state['results']:
                result = state['results']['training']
                logger.info(f"Models trained: {len(result.get('models_trained', []))}")
            
            if 'evaluation' in state['results']:
                result = state['results']['evaluation']
                eval_results = result.get('evaluation_results', {})
                if 'comparison_report' in eval_results:
                    comp_report = eval_results['comparison_report']
                    logger.info(f"Best model: {comp_report.get('best_model', 'N/A')}")
                    logger.info(f"Best F1 score: {comp_report.get('best_f1_score', 'N/A'):.4f}")
            
            logger.info("\nGenerated files and reports:")
            
            # List key output files
            output_dirs = [
                "data/raw",
                "data/processed", 
                "data/labeled",
                "models",
                "reports",
                "pipeline_states"
            ]
            
            for dir_path in output_dirs:
                if Path(dir_path).exists():
                    files = list(Path(dir_path).glob("*"))
                    if files:
                        logger.info(f"  {dir_path}/: {len(files)} files")
            
            logger.info("\nNext steps:")
            logger.info("1. Review the generated reports in the 'reports/' directory")
            logger.info("2. Test the trained models with new data")
            logger.info("3. Deploy the best model for production use")
            logger.info("4. Set up monitoring and retraining schedules")
            
            return True
        
        else:
            logger.error("\n" + "=" * 100)
            logger.error("PIPELINE EXECUTION FAILED!")
            logger.error("=" * 100)
            
            # Display error information
            state = pipeline.pipeline_state
            logger.error(f"Pipeline ID: {state['pipeline_id']}")
            logger.error(f"Status: {state['status']}")
            logger.error(f"Steps completed: {len(state['steps_completed'])}/5")
            
            if state['errors']:
                logger.error("Errors encountered:")
                for error in state['errors']:
                    logger.error(f"  {error['step']}: {error['error']}")
            
            logger.info("\nTroubleshooting:")
            logger.info("1. Check the logs for detailed error messages")
            logger.info("2. Verify your configuration and environment variables")
            logger.info("3. Ensure all required data files are present")
            logger.info("4. Try running individual steps to isolate the issue")
            
            return False
    
    except Exception as e:
        logger.error(f"Pipeline execution failed with unexpected error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def create_sample_config():
    """Create a sample pipeline configuration file"""
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    sample_config = {
        'data_ingestion': {
            'enabled': True,
            'max_messages_per_channel': 500,
            'date_range': {
                'start_date': '2024-01-01',
                'end_date': '2024-12-31'
            }
        },
        'preprocessing': {
            'enabled': True,
            'clean_text': True,
            'extract_entities': True,
            'filter_amharic': True
        },
        'labeling': {
            'enabled': True,
            'sample_size': 100,
            'auto_label': True,
            'manual_review': False
        },
        'training': {
            'enabled': True,
            'models': ['xlm-roberta-base', 'bert-base-multilingual-cased'],
            'optimization': {
                'method': 'grid_search',
                'max_trials': 10,
                'cv_folds': 3
            },
            'hyperparameters': {
                'learning_rate': [1e-5, 2e-5, 3e-5],
                'batch_size': [8, 16],
                'num_epochs': [3, 5],
                'dropout': [0.1, 0.2]
            }
        },
        'evaluation': {
            'enabled': True,
            'metrics': ['precision', 'recall', 'f1', 'accuracy'],
            'generate_report': True
        }
    }
    
    import yaml
    config_file = config_dir / "pipeline_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)
    
    logger.info(f"Sample configuration created: {config_file}")
    return str(config_file)

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Complete End-to-End Pipeline for Amharic E-commerce Data Extraction")
    
    parser.add_argument('--config', '-c', default='config/pipeline_config.yaml',
                       help='Path to pipeline configuration file')
    parser.add_argument('--skip-ingestion', action='store_true',
                       help='Skip data ingestion step')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing step')
    parser.add_argument('--skip-labeling', action='store_true',
                       help='Skip labeling step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation step')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a sample configuration file and exit')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if args.create_config:
        create_sample_config()
        return
    
    # Create config if it doesn't exist
    if not Path(args.config).exists():
        logger.info(f"Configuration file not found: {args.config}")
        logger.info("Creating sample configuration...")
        args.config = create_sample_config()
    
    # Run pipeline
    success = await run_complete_pipeline(
        config_path=args.config,
        skip_ingestion=args.skip_ingestion,
        skip_preprocessing=args.skip_preprocessing,
        skip_labeling=args.skip_labeling,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )
    
    if success:
        logger.info("Complete pipeline execution finished successfully!")
        sys.exit(0)
    else:
        logger.error("Complete pipeline execution failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
