"""
Utility functions for data handling and processing
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml

from loguru import logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

def save_json(data: Any, file_path: str, indent: int = 2):
    """Save data to JSON file with proper encoding"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    logger.info(f"Data saved to {file_path}")

def load_json(file_path: str) -> Any:
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_directory_structure():
    """Create necessary directory structure for the project"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/labeled",
        "logs",
        "config",
        "models",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created")

def get_latest_file(directory: str, pattern: str = "*") -> Optional[str]:
    """Get the most recently created file matching pattern in directory"""
    directory_path = Path(directory)
    if not directory_path.exists():
        return None
    
    files = list(directory_path.glob(pattern))
    if not files:
        return None
    
    # Sort by modification time and return the latest
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)

def validate_dataset(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that dataset has required columns"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    logger.info("Dataset validation passed")
    return True

def filter_amharic_messages(df: pd.DataFrame, min_amharic_ratio: float = 0.3) -> pd.DataFrame:
    """Filter messages that contain significant Amharic content"""
    if 'is_amharic' not in df.columns:
        logger.warning("'is_amharic' column not found. Cannot filter Amharic messages.")
        return df
    
    amharic_df = df[df['is_amharic'] == True].copy()
    logger.info(f"Filtered {len(amharic_df)} Amharic messages from {len(df)} total messages")
    
    return amharic_df

def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Get basic statistics about the dataset"""
    stats = {
        'total_messages': len(df),
        'date_range': {
            'start': df['date'].min() if 'date' in df.columns else None,
            'end': df['date'].max() if 'date' in df.columns else None
        },
        'channels': df['channel_name'].nunique() if 'channel_name' in df.columns else 0,
        'avg_text_length': df['text'].str.len().mean() if 'text' in df.columns else 0,
        'messages_with_media': df['has_media'].sum() if 'has_media' in df.columns else 0,
        'amharic_messages': df['is_amharic'].sum() if 'is_amharic' in df.columns else 0,
    }
    
    return stats

def create_sample_dataset(df: pd.DataFrame, sample_size: int = 100, random_state: int = 42) -> pd.DataFrame:
    """Create a sample dataset for labeling"""
    if len(df) <= sample_size:
        return df.copy()
    
    # Stratified sampling by channel if possible
    if 'channel_name' in df.columns:
        sample_df = df.groupby('channel_name').apply(
            lambda x: x.sample(min(len(x), sample_size // df['channel_name'].nunique() + 1), 
                             random_state=random_state)
        ).reset_index(drop=True)
        
        # If we still have too many, randomly sample
        if len(sample_df) > sample_size:
            sample_df = sample_df.sample(sample_size, random_state=random_state)
    else:
        sample_df = df.sample(sample_size, random_state=random_state)
    
    logger.info(f"Created sample dataset with {len(sample_df)} messages")
    return sample_df

def export_for_labeling(df: pd.DataFrame, output_file: str, text_column: str = 'cleaned_text'):
    """Export dataset in format suitable for manual labeling"""
    # Select relevant columns for labeling
    labeling_columns = ['message_id', 'channel_name', text_column, 'entity_hints']
    
    available_columns = [col for col in labeling_columns if col in df.columns]
    labeling_df = df[available_columns].copy()
    
    # Save as JSON for better Unicode support
    labeling_df.to_json(output_file, orient='records', force_ascii=False, indent=2)
    logger.info(f"Dataset exported for labeling: {output_file}")

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    
    # Add file handler
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="30 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    
    # Add console handler
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
    )
    
    logger.info("Logging setup completed")

class DataPipeline:
    """Data processing pipeline for the project"""
    
    def __init__(self, config_path: str = "config/telegram_channels.yaml"):
        self.config = load_config(config_path)
        setup_logging()
        create_directory_structure()
    
    def run_ingestion(self):
        """Run data ingestion pipeline"""
        logger.info("Starting data ingestion pipeline")
        # This would call the telegram scraper
        # Implementation depends on the scraper module
        pass
    
    def run_preprocessing(self, input_file: str = None):
        """Run preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline")
        
        if input_file is None:
            input_file = get_latest_file("data/raw", "*.json")
            if input_file is None:
                logger.error("No input file found for preprocessing")
                return
        
        from src.preprocessing.text_processor import AmharicTextProcessor
        processor = AmharicTextProcessor()
        
        output_file = input_file.replace("raw", "processed")
        processed_df = processor.preprocess_dataset(input_file, output_file)
        
        # Generate statistics
        stats = get_dataset_statistics(processed_df)
        stats_file = output_file.replace('.json', '_stats.json')
        save_json(stats, stats_file)
        
        logger.info("Preprocessing pipeline completed")
        return processed_df
    
    def prepare_for_labeling(self, input_file: str = None, sample_size: int = 50):
        """Prepare dataset for manual labeling"""
        logger.info("Preparing dataset for labeling")
        
        if input_file is None:
            input_file = get_latest_file("data/processed", "*.json")
            if input_file is None:
                logger.error("No processed file found")
                return
        
        # Load processed data
        df = pd.read_json(input_file)
        
        # Filter for Amharic messages
        amharic_df = filter_amharic_messages(df)
        
        # Create sample for labeling
        sample_df = create_sample_dataset(amharic_df, sample_size)
        
        # Export for labeling
        output_file = f"data/labeled/sample_for_labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_for_labeling(sample_df, output_file)
        
        return sample_df

if __name__ == "__main__":
    # Test the pipeline
    pipeline = DataPipeline()
    print("Data pipeline utilities loaded successfully")
