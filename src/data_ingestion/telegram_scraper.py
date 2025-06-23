"""
Telegram Data Scraper for Ethiopian E-commerce Channels
Collects messages, images, and documents from specified Telegram channels
"""

import asyncio
import json
import os
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from telethon import TelegramClient
from telethon.tl.types import Message, MessageMediaPhoto, MessageMediaDocument
import pandas as pd
from loguru import logger

class TelegramScraper:
    """Scraper for collecting data from Ethiopian e-commerce Telegram channels"""
    
    def __init__(self, config_path: str = "config/telegram_channels.yaml"):
        """Initialize the scraper with configuration"""
        self.config = self._load_config(config_path)
        self.client = None
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logger.add("logs/telegram_scraper.log", rotation="1 day")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
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
    
    async def initialize_client(self, api_id: str, api_hash: str, phone: str):
        """Initialize Telegram client"""
        try:
            self.client = TelegramClient('session', api_id, api_hash)
            await self.client.start(phone=phone)
            logger.info("Telegram client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram client: {e}")
            raise
    
    async def scrape_channel(self, channel_info: Dict[str, str]) -> List[Dict[str, Any]]:
        """Scrape messages from a single channel"""
        if not self.client:
            raise ValueError("Telegram client not initialized")
        
        channel_username = channel_info['username']
        channel_name = channel_info['name']
        max_messages = self.config['scraping']['max_messages_per_channel']
        
        logger.info(f"Starting to scrape channel: {channel_name} ({channel_username})")
        
        messages_data = []
        
        try:
            entity = await self.client.get_entity(channel_username)
            
            # Calculate date range
            start_date = datetime.strptime(
                self.config['scraping']['date_range']['start_date'], 
                "%Y-%m-%d"
            )
            end_date = datetime.strptime(
                self.config['scraping']['date_range']['end_date'], 
                "%Y-%m-%d"
            )
            
            message_count = 0
            async for message in self.client.iter_messages(
                entity, 
                limit=max_messages,
                offset_date=end_date
            ):
                if message.date < start_date:
                    break
                    
                if message_count >= max_messages:
                    break
                
                # Extract message data
                message_data = await self._extract_message_data(
                    message, channel_info
                )
                
                if message_data:
                    messages_data.append(message_data)
                    message_count += 1
                    
                    if message_count % 100 == 0:
                        logger.info(f"Scraped {message_count} messages from {channel_name}")
            
            logger.info(f"Completed scraping {channel_name}: {len(messages_data)} messages")
            return messages_data
            
        except Exception as e:
            logger.error(f"Error scraping channel {channel_name}: {e}")
            return []
    
    async def _extract_message_data(
        self, 
        message: Message, 
        channel_info: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Extract relevant data from a message"""
        
        # Skip messages without text content
        if not message.text and not message.media:
            return None
        
        message_data = {
            'message_id': message.id,
            'channel_name': channel_info['name'],
            'channel_username': channel_info['username'],
            'channel_category': channel_info['category'],
            'date': message.date.isoformat(),
            'text': message.text or "",
            'sender_id': getattr(message.sender, 'id', None) if message.sender else None,
            'views': getattr(message, 'views', 0),
            'forwards': getattr(message, 'forwards', 0),
            'replies': getattr(message.replies, 'replies', 0) if message.replies else 0,
            'has_media': bool(message.media),
            'media_type': None,
            'media_path': None
        }
        
        # Handle media
        if message.media:
            if isinstance(message.media, MessageMediaPhoto):
                message_data['media_type'] = 'photo'
                if self.config['processing']['save_images']:
                    # Save image logic would go here
                    pass
            elif isinstance(message.media, MessageMediaDocument):
                message_data['media_type'] = 'document'
        
        return message_data
    
    async def scrape_all_channels(self) -> pd.DataFrame:
        """Scrape all configured channels"""
        all_messages = []
        
        for channel_info in self.config['channels']:
            channel_messages = await self.scrape_channel(channel_info)
            all_messages.extend(channel_messages)
            
            # Add delay between channels to avoid rate limiting
            await asyncio.sleep(2)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_messages)
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.data_dir / f"telegram_messages_{timestamp}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Also save as JSON for better Unicode support
        json_file = self.data_dir / f"telegram_messages_{timestamp}.json"
        df.to_json(json_file, orient='records', force_ascii=False, indent=2)
        
        logger.info(f"Saved {len(df)} messages to {output_file}")
        return df
    
    async def close(self):
        """Close the Telegram client"""
        if self.client:
            await self.client.disconnect()
            logger.info("Telegram client disconnected")

async def main():
    """Main function to run the scraper"""
    # You need to get these from https://my.telegram.org/apps
    API_ID = os.getenv('TELEGRAM_API_ID')
    API_HASH = os.getenv('TELEGRAM_API_HASH')
    PHONE = os.getenv('TELEGRAM_PHONE')
    
    if not all([API_ID, API_HASH, PHONE]):
        logger.error("Please set TELEGRAM_API_ID, TELEGRAM_API_HASH, and TELEGRAM_PHONE environment variables")
        return
    
    scraper = TelegramScraper()
    
    try:
        await scraper.initialize_client(API_ID, API_HASH, PHONE)
        df = await scraper.scrape_all_channels()
        logger.info(f"Successfully scraped {len(df)} messages from all channels")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(main())
