"""
Complete End-to-End Pipeline for Amharic E-commerce Data Extraction
Orchestrates the entire workflow from data ingestion to model deployment
"""

import os
import sys
import json
import yaml
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import modules with error handling for optional dependencies
try:
    from data_ingestion.telegram_scraper import TelegramScraper
except ImportError:
    TelegramScraper = None

try:
    from preprocessing.text_processor import AmharicTextProcessor
except ImportError:
    AmharicTextProcessor = None

try:
    from labeling.conll_labeler import CoNLLLabeler
except ImportError:
    CoNLLLabeler = None

try:
    from modeling.advanced_ner_trainer import AdvancedNERTrainer
except ImportError:
    AdvancedNERTrainer = None

try:
    from modeling.data_loader import CoNLLDataLoader
except ImportError:
    CoNLLDataLoader = None

try:
    from modeling.model_evaluator import ModelEvaluator
except ImportError:
    ModelEvaluator = None

try:
    from utils.data_utils import setup_logging, create_directory_structure
except ImportError:
    def setup_logging():
        pass
    def create_directory_structure():
        pass

class EndToEndPipeline:
    """Complete end-to-end pipeline orchestrator"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """Initialize the pipeline with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.scraper = None
        self.text_processor = None
        self.labeler = None
        self.trainer = None
        self.evaluator = None
        
        # Pipeline state
        self.pipeline_state = {
            'pipeline_id': self.pipeline_id,
            'status': 'initialized',
            'steps_completed': [],
            'current_step': None,
            'start_time': None,
            'end_time': None,
            'results': {},
            'errors': []
        }
        
        # Setup logging and directories
        setup_logging()
        create_directory_structure()
        
        logger.info(f"Initialized End-to-End Pipeline: {self.pipeline_id}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'data_ingestion': {
                    'enabled': True,
                    'max_messages_per_channel': 1000,
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
                        'method': 'grid_search',  # 'grid_search', 'bayesian', 'none'
                        'max_trials': 20,
                        'cv_folds': 5
                    },
                    'hyperparameters': {
                        'learning_rate': [1e-5, 2e-5, 3e-5],
                        'batch_size': [8, 16, 32],
                        'num_epochs': [3, 5],
                        'dropout': [0.1, 0.2]
                    }
                },
                'evaluation': {
                    'enabled': True,
                    'metrics': ['precision', 'recall', 'f1', 'accuracy'],
                    'generate_report': True
                },
                'deployment': {
                    'enabled': False,
                    'save_best_model': True,
                    'model_format': 'pytorch'
                }
            }
    
    def _update_pipeline_state(self, step: str, status: str, results: Dict[str, Any] = None, error: str = None):
        """Update pipeline state"""
        self.pipeline_state['current_step'] = step
        self.pipeline_state['status'] = status
        
        if status == 'completed' and step not in self.pipeline_state['steps_completed']:
            self.pipeline_state['steps_completed'].append(step)
        
        if results:
            self.pipeline_state['results'][step] = results
        
        if error:
            self.pipeline_state['errors'].append({
                'step': step,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
        
        # Save state
        self._save_pipeline_state()
    
    def _save_pipeline_state(self):
        """Save pipeline state to file"""
        state_dir = Path("pipeline_states")
        state_dir.mkdir(exist_ok=True)
        
        state_file = state_dir / f"{self.pipeline_id}_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2, default=str)
    
    async def run_data_ingestion(self) -> bool:
        """Step 1: Data Ingestion from Telegram channels"""
        if not self.config['data_ingestion']['enabled']:
            logger.info("Data ingestion disabled, skipping...")
            return True
        
        logger.info("=" * 60)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("=" * 60)
        
        self._update_pipeline_state('data_ingestion', 'running')
        
        try:
            # Check if TelegramScraper is available
            if TelegramScraper is None:
                error_msg = "TelegramScraper not available. Please install telethon: pip install telethon"
                logger.error(error_msg)
                self._update_pipeline_state('data_ingestion', 'failed', error=error_msg)
                return False

            # Check for required environment variables
            required_vars = ['TELEGRAM_API_ID', 'TELEGRAM_API_HASH', 'TELEGRAM_PHONE']
            missing_vars = [var for var in required_vars if not os.getenv(var)]

            if missing_vars:
                error_msg = f"Missing environment variables: {missing_vars}"
                logger.error(error_msg)
                self._update_pipeline_state('data_ingestion', 'failed', error=error_msg)
                return False

            # Initialize scraper
            self.scraper = TelegramScraper()
            
            # Initialize Telegram client
            await self.scraper.initialize_client(
                os.getenv('TELEGRAM_API_ID'),
                os.getenv('TELEGRAM_API_HASH'),
                os.getenv('TELEGRAM_PHONE')
            )
            
            # Scrape all channels
            logger.info("Starting data collection from Telegram channels...")
            df = await self.scraper.scrape_all_channels()
            
            if len(df) == 0:
                error_msg = "No messages collected. Check your channel configuration."
                logger.warning(error_msg)
                self._update_pipeline_state('data_ingestion', 'failed', error=error_msg)
                return False
            
            # Close scraper
            await self.scraper.close()
            
            results = {
                'messages_collected': len(df),
                'channels_scraped': len(self.scraper.config['channels']),
                'data_file': str(self.scraper.data_dir / f"telegram_messages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            }
            
            logger.info(f"Successfully collected {len(df)} messages")
            self._update_pipeline_state('data_ingestion', 'completed', results)
            return True
            
        except Exception as e:
            error_msg = f"Data ingestion failed: {str(e)}"
            logger.error(error_msg)
            self._update_pipeline_state('data_ingestion', 'failed', error=error_msg)
            return False
    
    def run_preprocessing(self) -> bool:
        """Step 2: Text Preprocessing"""
        if not self.config['preprocessing']['enabled']:
            logger.info("Preprocessing disabled, skipping...")
            return True
        
        logger.info("=" * 60)
        logger.info("STEP 2: TEXT PREPROCESSING")
        logger.info("=" * 60)
        
        self._update_pipeline_state('preprocessing', 'running')
        
        try:
            # Check if AmharicTextProcessor is available
            if AmharicTextProcessor is None:
                error_msg = "AmharicTextProcessor not available. Please check preprocessing module."
                logger.error(error_msg)
                self._update_pipeline_state('preprocessing', 'failed', error=error_msg)
                return False

            # Initialize text processor
            self.text_processor = AmharicTextProcessor()
            
            # Find latest raw data file
            raw_data_dir = Path("data/raw")
            if not raw_data_dir.exists():
                error_msg = "No raw data directory found"
                logger.error(error_msg)
                self._update_pipeline_state('preprocessing', 'failed', error=error_msg)
                return False
            
            # Get latest JSON file
            json_files = list(raw_data_dir.glob("*.json"))
            if not json_files:
                error_msg = "No raw data files found"
                logger.error(error_msg)
                self._update_pipeline_state('preprocessing', 'failed', error=error_msg)
                return False
            
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Processing file: {latest_file}")
            
            # Process the data
            output_file = str(latest_file).replace("raw", "processed")
            processed_df = self.text_processor.preprocess_dataset(str(latest_file), output_file)
            
            # Generate statistics
            amharic_messages = processed_df[processed_df['is_amharic'] == True]
            messages_with_prices = processed_df[processed_df['has_price_indicators'] == True]
            messages_with_locations = processed_df[processed_df['has_location_indicators'] == True]
            
            results = {
                'total_messages': len(processed_df),
                'amharic_messages': len(amharic_messages),
                'messages_with_prices': len(messages_with_prices),
                'messages_with_locations': len(messages_with_locations),
                'processed_file': output_file
            }
            
            logger.info(f"Successfully preprocessed {len(processed_df)} messages")
            logger.info(f"Amharic messages: {len(amharic_messages)}")
            logger.info(f"Messages with price indicators: {len(messages_with_prices)}")
            
            self._update_pipeline_state('preprocessing', 'completed', results)
            return True
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            logger.error(error_msg)
            self._update_pipeline_state('preprocessing', 'failed', error=error_msg)
            return False
    
    def run_labeling(self) -> bool:
        """Step 3: Data Labeling"""
        if not self.config['labeling']['enabled']:
            logger.info("Labeling disabled, skipping...")
            return True
        
        logger.info("=" * 60)
        logger.info("STEP 3: DATA LABELING")
        logger.info("=" * 60)
        
        self._update_pipeline_state('labeling', 'running')
        
        try:
            # Initialize labeler
            self.labeler = CoNLLLabeler()
            
            # Find latest processed data file
            processed_data_dir = Path("data/processed")
            if not processed_data_dir.exists():
                error_msg = "No processed data directory found"
                logger.error(error_msg)
                self._update_pipeline_state('labeling', 'failed', error=error_msg)
                return False
            
            json_files = list(processed_data_dir.glob("*.json"))
            if not json_files:
                error_msg = "No processed data files found"
                logger.error(error_msg)
                self._update_pipeline_state('labeling', 'failed', error=error_msg)
                return False
            
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Labeling file: {latest_file}")
            
            # Load and sample data
            df = pd.read_json(latest_file)
            sample_size = self.config['labeling']['sample_size']
            
            # Filter for Amharic messages
            amharic_df = df[df['is_amharic'] == True]
            if len(amharic_df) < sample_size:
                sample_df = amharic_df
                logger.warning(f"Only {len(amharic_df)} Amharic messages available, using all")
            else:
                sample_df = amharic_df.sample(n=sample_size, random_state=42)
            
            # Auto-label the sample
            labeled_data = []
            for _, row in sample_df.iterrows():
                labeled_message = self.labeler.auto_label_message(row['cleaned_text'])
                labeled_data.append(labeled_message)
            
            # Save labeled data in CoNLL format
            output_file = f"data/labeled/amharic_ner_sample_{len(labeled_data)}_messages.txt"
            self.labeler.save_conll_format(labeled_data, output_file)
            
            results = {
                'total_messages_labeled': len(labeled_data),
                'labeled_file': output_file,
                'source_file': str(latest_file)
            }
            
            logger.info(f"Successfully labeled {len(labeled_data)} messages")
            logger.info(f"Labeled data saved to: {output_file}")
            
            self._update_pipeline_state('labeling', 'completed', results)
            return True

        except Exception as e:
            error_msg = f"Labeling failed: {str(e)}"
            logger.error(error_msg)
            self._update_pipeline_state('labeling', 'failed', error=error_msg)
            return False

    def run_training(self) -> bool:
        """Step 4: Model Training and Optimization"""
        if not self.config['training']['enabled']:
            logger.info("Training disabled, skipping...")
            return True

        logger.info("=" * 60)
        logger.info("STEP 4: MODEL TRAINING AND OPTIMIZATION")
        logger.info("=" * 60)

        self._update_pipeline_state('training', 'running')

        try:
            # Find labeled data file
            labeled_data_dir = Path("data/labeled")
            if not labeled_data_dir.exists():
                error_msg = "No labeled data directory found"
                logger.error(error_msg)
                self._update_pipeline_state('training', 'failed', error=error_msg)
                return False

            conll_files = list(labeled_data_dir.glob("*.txt"))
            if not conll_files:
                error_msg = "No labeled data files found"
                logger.error(error_msg)
                self._update_pipeline_state('training', 'failed', error=error_msg)
                return False

            latest_file = max(conll_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Training with file: {latest_file}")

            training_results = {}

            # Train models
            for model_name in self.config['training']['models']:
                logger.info(f"Training model: {model_name}")

                # Initialize trainer
                self.trainer = AdvancedNERTrainer(model_name=model_name, output_dir="models")

                # Load data
                data_loader = CoNLLDataLoader(tokenizer_name=model_name)
                self.trainer.data_loader = data_loader

                train_dataset, val_dataset = data_loader.prepare_datasets(str(latest_file), val_split=0.2)

                # Perform optimization based on configuration
                optimization_method = self.config['training']['optimization']['method']

                if optimization_method == 'grid_search':
                    logger.info("Performing grid search optimization...")
                    hyperparams = self.config['training']['hyperparameters']
                    max_trials = self.config['training']['optimization']['max_trials']

                    grid_results = self.trainer.grid_search(
                        train_dataset, val_dataset, hyperparams, max_trials
                    )
                    training_results[model_name] = {
                        'optimization_method': 'grid_search',
                        'results': grid_results
                    }

                elif optimization_method == 'bayesian':
                    logger.info("Performing Bayesian optimization...")
                    n_trials = self.config['training']['optimization']['max_trials']

                    bayesian_results = self.trainer.bayesian_optimization(
                        train_dataset, val_dataset, n_trials
                    )
                    training_results[model_name] = {
                        'optimization_method': 'bayesian',
                        'results': bayesian_results
                    }

                else:  # No optimization, use default parameters
                    logger.info("Training with default parameters...")
                    default_params = {
                        'learning_rate': 2e-5,
                        'batch_size': 16,
                        'num_epochs': 3,
                        'warmup_ratio': 0.1,
                        'weight_decay': 0.01,
                        'dropout': 0.1,
                        'scheduler_type': 'linear'
                    }

                    results = self.trainer.train_with_hyperparams(
                        train_dataset, val_dataset, default_params
                    )
                    training_results[model_name] = {
                        'optimization_method': 'none',
                        'results': results
                    }

                # Perform cross-validation if enabled
                cv_folds = self.config['training']['optimization']['cv_folds']
                if cv_folds > 1:
                    logger.info(f"Performing {cv_folds}-fold cross-validation...")
                    cv_results = self.trainer.cross_validate(
                        train_dataset, n_folds=cv_folds
                    )
                    training_results[model_name]['cv_results'] = cv_results

                # Save best model
                best_model_path = self.trainer.save_best_model(
                    f"models/best_{model_name.replace('/', '_')}"
                )
                training_results[model_name]['best_model_path'] = best_model_path

                # Generate training report
                report = self.trainer.generate_training_report()
                training_results[model_name]['training_report'] = report

                logger.info(f"Completed training for {model_name}")

            results = {
                'models_trained': list(training_results.keys()),
                'training_results': training_results,
                'labeled_data_file': str(latest_file)
            }

            logger.info(f"Successfully trained {len(training_results)} models")
            self._update_pipeline_state('training', 'completed', results)
            return True

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(error_msg)
            self._update_pipeline_state('training', 'failed', error=error_msg)
            return False

    def run_evaluation(self) -> bool:
        """Step 5: Model Evaluation and Comparison"""
        if not self.config['evaluation']['enabled']:
            logger.info("Evaluation disabled, skipping...")
            return True

        logger.info("=" * 60)
        logger.info("STEP 5: MODEL EVALUATION AND COMPARISON")
        logger.info("=" * 60)

        self._update_pipeline_state('evaluation', 'running')

        try:
            # Initialize evaluator
            self.evaluator = ModelEvaluator()

            # Get training results
            training_results = self.pipeline_state['results'].get('training', {}).get('training_results', {})

            if not training_results:
                error_msg = "No training results found for evaluation"
                logger.error(error_msg)
                self._update_pipeline_state('evaluation', 'failed', error=error_msg)
                return False

            evaluation_results = {}

            # Evaluate each trained model
            for model_name, model_results in training_results.items():
                logger.info(f"Evaluating model: {model_name}")

                best_model_path = model_results.get('best_model_path')
                if not best_model_path or not Path(best_model_path).exists():
                    logger.warning(f"Best model path not found for {model_name}, skipping evaluation")
                    continue

                # Load test data (use validation split for now)
                labeled_data_file = self.pipeline_state['results']['labeling']['labeled_file']
                data_loader = CoNLLDataLoader(tokenizer_name=model_name)
                _, test_dataset = data_loader.prepare_datasets(labeled_data_file, val_split=0.2)

                # Evaluate model
                eval_metrics = self.evaluator.evaluate_model(best_model_path, test_dataset)

                evaluation_results[model_name] = {
                    'metrics': eval_metrics,
                    'model_path': best_model_path
                }

                logger.info(f"Evaluation completed for {model_name}")
                logger.info(f"F1 Score: {eval_metrics.get('f1', 'N/A')}")

            # Generate comparison report
            if len(evaluation_results) > 1:
                comparison_report = self._generate_model_comparison_report(evaluation_results)
                evaluation_results['comparison_report'] = comparison_report

            results = {
                'models_evaluated': list(evaluation_results.keys()),
                'evaluation_results': evaluation_results
            }

            logger.info(f"Successfully evaluated {len(evaluation_results)} models")
            self._update_pipeline_state('evaluation', 'completed', results)
            return True

        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            logger.error(error_msg)
            self._update_pipeline_state('evaluation', 'failed', error=error_msg)
            return False

    def _generate_model_comparison_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model comparison report"""

        models_data = []
        for model_name, results in evaluation_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                models_data.append({
                    'model': model_name,
                    'f1': metrics.get('f1', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'accuracy': metrics.get('accuracy', 0)
                })

        if not models_data:
            return {'error': 'No valid model metrics found'}

        # Sort by F1 score
        models_data.sort(key=lambda x: x['f1'], reverse=True)

        comparison_report = {
            'best_model': models_data[0]['model'],
            'best_f1_score': models_data[0]['f1'],
            'model_rankings': models_data,
            'summary': {
                'total_models': len(models_data),
                'best_f1': max(m['f1'] for m in models_data),
                'worst_f1': min(m['f1'] for m in models_data),
                'mean_f1': sum(m['f1'] for m in models_data) / len(models_data)
            }
        }

        return comparison_report

    async def run_complete_pipeline(self) -> bool:
        """Run the complete end-to-end pipeline"""

        logger.info("=" * 80)
        logger.info("STARTING COMPLETE END-TO-END PIPELINE")
        logger.info(f"Pipeline ID: {self.pipeline_id}")
        logger.info("=" * 80)

        self.pipeline_state['start_time'] = datetime.now().isoformat()
        self.pipeline_state['status'] = 'running'

        # Define pipeline steps
        steps = [
            ('data_ingestion', self.run_data_ingestion),
            ('preprocessing', self.run_preprocessing),
            ('labeling', self.run_labeling),
            ('training', self.run_training),
            ('evaluation', self.run_evaluation)
        ]

        # Execute steps
        for step_name, step_function in steps:
            logger.info(f"\nExecuting step: {step_name}")

            try:
                if asyncio.iscoroutinefunction(step_function):
                    success = await step_function()
                else:
                    success = step_function()

                if not success:
                    logger.error(f"Step {step_name} failed. Pipeline stopped.")
                    self.pipeline_state['status'] = 'failed'
                    self.pipeline_state['end_time'] = datetime.now().isoformat()
                    self._save_pipeline_state()
                    return False

                logger.info(f"Step {step_name} completed successfully")

            except Exception as e:
                error_msg = f"Unexpected error in step {step_name}: {str(e)}"
                logger.error(error_msg)
                self._update_pipeline_state(step_name, 'failed', error=error_msg)
                self.pipeline_state['status'] = 'failed'
                self.pipeline_state['end_time'] = datetime.now().isoformat()
                self._save_pipeline_state()
                return False

        # Pipeline completed successfully
        self.pipeline_state['status'] = 'completed'
        self.pipeline_state['end_time'] = datetime.now().isoformat()
        self._save_pipeline_state()

        # Generate final report
        self._generate_final_report()

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Pipeline ID: {self.pipeline_id}")
        logger.info("=" * 80)

        return True

    def _generate_final_report(self):
        """Generate final pipeline report"""

        report = {
            'pipeline_id': self.pipeline_id,
            'execution_summary': {
                'status': self.pipeline_state['status'],
                'start_time': self.pipeline_state['start_time'],
                'end_time': self.pipeline_state['end_time'],
                'steps_completed': self.pipeline_state['steps_completed'],
                'total_steps': 5,
                'success_rate': len(self.pipeline_state['steps_completed']) / 5 * 100
            },
            'results_summary': {},
            'errors': self.pipeline_state['errors'],
            'configuration': self.config
        }

        # Add results summary
        for step, results in self.pipeline_state['results'].items():
            if step == 'data_ingestion':
                report['results_summary']['data_ingestion'] = {
                    'messages_collected': results.get('messages_collected', 0),
                    'channels_scraped': results.get('channels_scraped', 0)
                }
            elif step == 'preprocessing':
                report['results_summary']['preprocessing'] = {
                    'total_messages': results.get('total_messages', 0),
                    'amharic_messages': results.get('amharic_messages', 0)
                }
            elif step == 'labeling':
                report['results_summary']['labeling'] = {
                    'messages_labeled': results.get('total_messages_labeled', 0)
                }
            elif step == 'training':
                report['results_summary']['training'] = {
                    'models_trained': len(results.get('models_trained', []))
                }
            elif step == 'evaluation':
                eval_results = results.get('evaluation_results', {})
                if 'comparison_report' in eval_results:
                    comp_report = eval_results['comparison_report']
                    report['results_summary']['evaluation'] = {
                        'models_evaluated': len(results.get('models_evaluated', [])),
                        'best_model': comp_report.get('best_model', 'N/A'),
                        'best_f1_score': comp_report.get('best_f1_score', 0)
                    }

        # Save final report
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        report_file = reports_dir / f"{self.pipeline_id}_final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate markdown report
        self._generate_markdown_final_report(report)

        logger.info(f"Final report saved to {report_file}")

    def _generate_markdown_final_report(self, report: Dict[str, Any]):
        """Generate markdown final report"""

        md_content = f"""# End-to-End Pipeline Report

## Pipeline Information
- **Pipeline ID**: {report['pipeline_id']}
- **Status**: {report['execution_summary']['status']}
- **Start Time**: {report['execution_summary']['start_time']}
- **End Time**: {report['execution_summary']['end_time']}
- **Success Rate**: {report['execution_summary']['success_rate']:.1f}%

## Execution Summary
- **Steps Completed**: {len(report['execution_summary']['steps_completed'])}/{report['execution_summary']['total_steps']}
- **Completed Steps**: {', '.join(report['execution_summary']['steps_completed'])}

## Results Summary
"""

        for step, summary in report['results_summary'].items():
            md_content += f"\n### {step.replace('_', ' ').title()}\n"
            for key, value in summary.items():
                md_content += f"- **{key.replace('_', ' ').title()}**: {value}\n"

        if report['errors']:
            md_content += "\n## Errors\n"
            for error in report['errors']:
                md_content += f"- **{error['step']}**: {error['error']} ({error['timestamp']})\n"

        md_content += "\n## Configuration\n```yaml\n"
        md_content += yaml.dump(report['configuration'], default_flow_style=False)
        md_content += "\n```\n"

        # Save markdown report
        reports_dir = Path("reports")
        md_file = reports_dir / f"{self.pipeline_id}_final_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Markdown report saved to {md_file}")


async def main():
    """Main function to run the complete pipeline"""

    # Create pipeline configuration if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    pipeline_config_path = config_dir / "pipeline_config.yaml"
    if not pipeline_config_path.exists():
        # Create default configuration
        default_config = {
            'data_ingestion': {'enabled': True},
            'preprocessing': {'enabled': True},
            'labeling': {'enabled': True, 'sample_size': 50},
            'training': {
                'enabled': True,
                'models': ['xlm-roberta-base'],
                'optimization': {'method': 'grid_search', 'max_trials': 4, 'cv_folds': 3}
            },
            'evaluation': {'enabled': True}
        }

        with open(pipeline_config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        logger.info(f"Created default pipeline configuration: {pipeline_config_path}")

    # Initialize and run pipeline
    pipeline = EndToEndPipeline(str(pipeline_config_path))

    success = await pipeline.run_complete_pipeline()

    if success:
        logger.info("Pipeline completed successfully!")
    else:
        logger.error("Pipeline failed!")

    return success


if __name__ == "__main__":
    asyncio.run(main())
