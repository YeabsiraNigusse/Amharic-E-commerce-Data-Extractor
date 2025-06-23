#!/usr/bin/env python3
"""
Main script to run Task 1: Data Ingestion and Preprocessing
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from loguru import logger

from src.data_ingestion.telegram_scraper import TelegramScraper
from src.preprocessing.text_processor import AmharicTextProcessor
from src.utils.data_utils import DataPipeline, setup_logging

async def run_data_ingestion():
    """Run the data ingestion process"""
    logger.info("Starting Task 1: Data Ingestion and Preprocessing")
    
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ['TELEGRAM_API_ID', 'TELEGRAM_API_HASH', 'TELEGRAM_PHONE']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please copy .env.example to .env and fill in your Telegram API credentials")
        return False
    
    try:
        # Initialize scraper
        scraper = TelegramScraper()
        
        # Initialize Telegram client
        await scraper.initialize_client(
            os.getenv('TELEGRAM_API_ID'),
            os.getenv('TELEGRAM_API_HASH'),
            os.getenv('TELEGRAM_PHONE')
        )
        
        # Scrape all channels
        logger.info("Starting data collection from Telegram channels...")
        df = await scraper.scrape_all_channels()
        
        if len(df) == 0:
            logger.warning("No messages collected. Check your channel configuration.")
            return False
        
        logger.info(f"Successfully collected {len(df)} messages")
        
        # Close scraper
        await scraper.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return False

def run_preprocessing():
    """Run the preprocessing process"""
    logger.info("Starting text preprocessing...")
    
    try:
        # Initialize pipeline
        pipeline = DataPipeline()
        
        # Run preprocessing
        processed_df = pipeline.run_preprocessing()
        
        if processed_df is not None:
            logger.info(f"Successfully preprocessed {len(processed_df)} messages")
            
            # Prepare sample for labeling
            sample_df = pipeline.prepare_for_labeling(sample_size=50)
            logger.info(f"Prepared {len(sample_df)} messages for labeling")
            
            return True
        else:
            logger.error("Preprocessing failed")
            return False
            
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return False

async def main():
    """Main function"""
    # Setup logging
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("AMHARIC E-COMMERCE DATA EXTRACTOR - TASK 1")
    logger.info("=" * 60)
    
    # Step 1: Data Ingestion
    logger.info("Step 1: Data Ingestion from Telegram Channels")
    ingestion_success = await run_data_ingestion()
    
    if not ingestion_success:
        logger.error("Data ingestion failed. Exiting.")
        return
    
    # Step 2: Text Preprocessing
    logger.info("Step 2: Text Preprocessing")
    preprocessing_success = run_preprocessing()
    
    if not preprocessing_success:
        logger.error("Text preprocessing failed. Exiting.")
        return
    
    logger.info("=" * 60)
    logger.info("TASK 1 COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Review the processed data in data/processed/")
    logger.info("2. Check the sample data prepared for labeling in data/labeled/")
    logger.info("3. Run Task 2 for CoNLL format labeling")

if __name__ == "__main__":
    asyncio.run(main())
