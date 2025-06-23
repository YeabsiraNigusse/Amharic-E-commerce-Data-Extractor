#!/usr/bin/env python3
"""
Main script to run Task 2: CoNLL Format Labeling
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from loguru import logger
from src.labeling.conll_labeler import CoNLLLabeler
from src.labeling.interactive_labeler import InteractiveLabeler
from src.utils.data_utils import setup_logging, get_latest_file

def create_sample_labeled_dataset():
    """Create sample labeled dataset with 30-50 messages"""
    logger.info("Creating sample labeled dataset...")
    
    labeler = CoNLLLabeler()
    
    # Extended sample messages in Amharic
    sample_messages = [
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
        "የቤት ዕቃዎች በአዲስ አበባ። ዋጋ 2500 ብር።",
        "የሴቶች ሰዓት 1200 ብር። ፒያሳ ማዕከል።",
        "የወንዶች ጫማ በ 1100 ብር። ዴሊቨሪ ነፃ።",
        "የሕፃናት ልብስ ስብስብ 800 ብር። መርካቶ አካባቢ።",
        "የኮምፒውተር ማውስ 250 ብር። ካዛንቺስ ውስጥ።",
        "የሴቶች ቦርሳ ስብስብ 1500 ብር። ጎፋ አካባቢ።",
        "የወንዶች ቲሸርት 180 ብር። ዴሊቨሪ 25 ብር።",
        "የሕፃናት ጫማ በ 350 ብር። ኪርኮስ ውስጥ።",
        "የስልክ ቻርጀር 120 ብር። @accessories_et ላይ።",
        "የቤት ማስዋቢያ 450 ብር። ላፍቶ አካባቢ።",
        "የሴቶች ሸሚዝ ዋጋ 400 ብር። ቦሌ ውስጥ።",
        "የወንዶች ቀሚስ ሱሪ 1200 ብር। አዲስ አበባ።",
        "የሕፃናት መጫወቻ ስብስብ 600 ብር። ፒያሳ ውስጥ።",
        "የኮምፒውተር ኪቦርድ 800 ብር። መርካቶ አካባቢ።",
        "የሴቶች ጌጣጌጥ 300 ብር። ዴሊቨሪ 20 ብር።",
        "የወንዶች ቦርሳ በ 950 ብር። ካዛንቺስ አካባቢ።",
        "የሕፃናት መጽሐፍ 85 ብር። ጎፋ ውስጥ።",
        "የስልክ ስክሪን ፕሮቴክተር 45 ብር። ኪርኮስ።",
        "የቤት ማጽጃ እቃዎች 280 ብር። ላፍቶ አካባቢ።",
        "የሴቶች ሱሪ ዋጋ 520 ብር። ቦሌ ውስጥ።",
        "የወንዶች ጃኬት በ 1800 ብር። አዲስ አበባ።",
        "የሕፃናት ሱሪ 120 ብር። ፒያሳ አካባቢ።",
        "የኮምፒውተር ስፒከር 650 ብር። መርካቶ ውስጥ።",
        "የሴቶች ሰንደል 380 ብር। ዴሊቨሪ 35 ብር።",
        "የወንዶች ሰዓት ዋጋ 2200 ብር። ካዛንቺስ።",
        "የሕፃናት ቀሚስ በ 220 ብር። ጎፋ አካባቢ።",
        "የስልክ ኢርፎን 180 ብር। @audio_et ላይ።",
        "የቤት ማብሰያ እቃዎች 1100 ብር። ኪርኮስ ውስጥ።",
        "የሴቶች ኮት ዋጋ 1600 ብር። ላፍቶ አካባቢ።",
        "የወንዶች ሱሪ ስብስብ 1400 ብር። ቦሌ ውስጥ።",
        "የሕፃናት ጫማ ስብስብ 750 ብር። አዲስ አበባ።",
        "የኮምፒውተር ካሜራ 950 ብር። ፒያሳ ውስጥ።",
        "የሴቶች ቦርሳ ዋጋ 680 ብር። መርካቶ አካባቢ።",
        "የወንዶች ቤልት በ 320 ብር። ዴሊቨሪ 15 ብር።",
        "የሕፃናት መጫወቻ መኪና 280 ብር። ካዛንቺስ ውስጥ።",
        "የስልክ ፓወር ባንክ 450 ብር። ጎፋ አካባቢ።"
    ]
    
    # Create labeled data
    labeled_data = []
    for i, message in enumerate(sample_messages):
        labeled_msg = labeler.label_message(message, auto_label=True)
        labeled_msg['message_id'] = f"sample_{i+1:03d}"
        labeled_data.append(labeled_msg)
    
    # Save in CoNLL format
    output_file = "data/labeled/amharic_ner_sample_50_messages.txt"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    labeler.save_conll_file(labeled_data, output_file)
    
    # Also save as JSON for analysis
    json_file = output_file.replace('.txt', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(labeled_data, f, ensure_ascii=False, indent=2)
    
    # Generate statistics
    stats = labeler.get_entity_statistics(labeled_data)
    
    logger.info(f"Created labeled dataset with {len(labeled_data)} messages")
    logger.info(f"Statistics: {stats}")
    
    return labeled_data, output_file

def run_interactive_labeling():
    """Run interactive labeling session"""
    logger.info("Starting interactive labeling session...")
    
    # Check if there's processed data to label
    processed_file = get_latest_file("data/processed", "*.json")
    
    interactive_labeler = InteractiveLabeler()
    
    if processed_file:
        logger.info(f"Found processed data: {processed_file}")
        choice = input(f"Use processed data from {processed_file}? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_labeler.run_interactive_session(processed_file, max_messages=30)
        else:
            interactive_labeler.run_interactive_session(max_messages=30)
    else:
        logger.info("No processed data found. Using sample data.")
        interactive_labeler.run_interactive_session(max_messages=30)

def validate_conll_format(file_path: str):
    """Validate CoNLL format file"""
    logger.info(f"Validating CoNLL format file: {file_path}")
    
    labeler = CoNLLLabeler()
    
    try:
        labeled_data = labeler.load_conll_file(file_path)
        stats = labeler.get_entity_statistics(labeled_data)
        
        logger.info("CoNLL file validation successful!")
        logger.info(f"Loaded {len(labeled_data)} messages")
        logger.info(f"Statistics: {stats}")
        
        return True
    except Exception as e:
        logger.error(f"CoNLL file validation failed: {e}")
        return False

def main():
    """Main function"""
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("AMHARIC E-COMMERCE DATA EXTRACTOR - TASK 2")
    logger.info("CoNLL Format Labeling")
    logger.info("=" * 60)
    
    print("\nTask 2 Options:")
    print("1. Create sample labeled dataset (50 messages)")
    print("2. Run interactive labeling session")
    print("3. Validate existing CoNLL file")
    print("4. Exit")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            logger.info("Creating sample labeled dataset...")
            labeled_data, output_file = create_sample_labeled_dataset()
            
            print(f"\n✅ Sample dataset created successfully!")
            print(f"📁 CoNLL file: {output_file}")
            print(f"📊 Messages labeled: {len(labeled_data)}")
            
            # Validate the created file
            validate_conll_format(output_file)
            break
            
        elif choice == '2':
            logger.info("Starting interactive labeling...")
            run_interactive_labeling()
            break
            
        elif choice == '3':
            file_path = input("Enter path to CoNLL file: ").strip()
            if Path(file_path).exists():
                validate_conll_format(file_path)
            else:
                logger.error(f"File not found: {file_path}")
            break
            
        elif choice == '4':
            logger.info("Exiting...")
            break
            
        else:
            print("Invalid choice. Please select 1-4.")
    
    logger.info("=" * 60)
    logger.info("TASK 2 COMPLETED!")
    logger.info("=" * 60)
    logger.info("Output files:")
    logger.info("- CoNLL format: data/labeled/amharic_ner_sample_50_messages.txt")
    logger.info("- JSON format: data/labeled/amharic_ner_sample_50_messages.json")
    logger.info("\nNext steps:")
    logger.info("1. Review the labeled data")
    logger.info("2. Use the data for NER model training")
    logger.info("3. Evaluate model performance")

if __name__ == "__main__":
    main()
