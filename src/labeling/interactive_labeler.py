"""
Interactive CoNLL Labeling Tool
Provides a command-line interface for manual labeling of Amharic text
"""

import json
import sys
from typing import List, Dict, Any
from pathlib import Path

from loguru import logger
from .conll_labeler import CoNLLLabeler

class InteractiveLabeler:
    """Interactive tool for manual CoNLL labeling"""
    
    def __init__(self):
        self.labeler = CoNLLLabeler()
        self.entity_types = ['PRODUCT', 'PRICE', 'LOCATION', 'DELIVERY_FEE', 'CONTACT_INFO']
        self.current_session = []
        
    def display_help(self):
        """Display help information"""
        help_text = """
=== AMHARIC CONLL LABELING TOOL ===

Entity Types:
- PRODUCT: Product names or types (e.g., ልብስ, ጫማ, ስልክ)
- PRICE: Monetary values (e.g., 500 ብር, ዋጋ 1000)
- LOCATION: Geographic locations (e.g., አዲስ አበባ, ቦሌ)
- DELIVERY_FEE: Delivery costs (e.g., ዴሊቨሪ 50 ብር)
- CONTACT_INFO: Contact information (e.g., 0911234567, @username)

Labeling Format:
- B-ENTITY: Beginning of entity
- I-ENTITY: Inside entity (continuation)
- O: Outside any entity

Commands:
- help: Show this help
- auto: Auto-label current message
- manual: Manual labeling mode
- save: Save current session
- load: Load previous session
- stats: Show labeling statistics
- quit: Exit the tool

Example:
Text: የሕፃናት ልብስ ዋጋ 500 ብር
Labels: B-PRODUCT I-PRODUCT B-PRICE I-PRICE I-PRICE
        """
        print(help_text)
    
    def display_message(self, message_data: Dict[str, Any], show_tokens: bool = True):
        """Display message for labeling"""
        print("\n" + "="*60)
        print(f"Message ID: {message_data.get('message_id', 'unknown')}")
        print(f"Text: {message_data['text']}")
        
        if show_tokens:
            tokens = message_data.get('tokens', [])
            print(f"\nTokens ({len(tokens)}):")
            for i, token in enumerate(tokens):
                print(f"{i+1:2d}: {token}")
    
    def get_user_labels(self, tokens: List[str]) -> List[str]:
        """Get labels from user input"""
        labels = []
        
        print(f"\nLabel each token (1-{len(tokens)}):")
        print("Available entities: " + ", ".join(self.entity_types))
        print("Format: B-ENTITY, I-ENTITY, or O")
        print("Type 'auto' for automatic labeling, 'skip' to skip this message")
        
        for i, token in enumerate(tokens):
            while True:
                label = input(f"Token {i+1} '{token}': ").strip().upper()
                
                if label == 'AUTO':
                    # Auto-label remaining tokens
                    auto_labels = self.labeler.auto_label_entities(tokens)
                    labels.extend(auto_labels[i:])
                    break
                elif label == 'SKIP':
                    return None
                elif label == 'O':
                    labels.append(label)
                    break
                elif label.startswith('B-') or label.startswith('I-'):
                    entity_type = label[2:]
                    if entity_type in self.entity_types:
                        labels.append(label)
                        break
                    else:
                        print(f"Invalid entity type: {entity_type}")
                        print(f"Available: {', '.join(self.entity_types)}")
                else:
                    print("Invalid format. Use B-ENTITY, I-ENTITY, or O")
        
        return labels
    
    def label_message_interactive(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Interactively label a single message"""
        self.display_message(message_data)
        
        tokens = message_data.get('tokens', [])
        if not tokens:
            tokens = self.labeler.tokenize_amharic_for_conll(message_data['text'])
            message_data['tokens'] = tokens
        
        while True:
            command = input("\nEnter command (label/auto/manual/help/skip): ").strip().lower()
            
            if command == 'help':
                self.display_help()
            elif command == 'auto':
                labels = self.labeler.auto_label_entities(tokens)
                break
            elif command in ['label', 'manual', '']:
                labels = self.get_user_labels(tokens)
                if labels is None:  # User chose to skip
                    return None
                break
            elif command == 'skip':
                return None
            else:
                print("Invalid command. Type 'help' for assistance.")
        
        # Validate labels
        labels = self.labeler.validate_labels(labels)
        
        # Create labeled data
        labeled_data = {
            'message_id': message_data.get('message_id'),
            'text': message_data['text'],
            'tokens': tokens,
            'labels': labels,
            'conll_format': self.labeler.create_conll_format(tokens, labels),
            'token_count': len(tokens)
        }
        
        # Display result
        print("\nLabeled result:")
        print(labeled_data['conll_format'])
        
        return labeled_data
    
    def load_messages_for_labeling(self, input_file: str) -> List[Dict[str, Any]]:
        """Load messages from file for labeling"""
        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return []
        
        if input_file.endswith('.json'):
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to expected format
            messages = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    message = {
                        'message_id': item.get('message_id', f'msg_{i+1}'),
                        'text': item.get('cleaned_text', item.get('text', '')),
                        'tokens': item.get('tokens', [])
                    }
                    messages.append(message)
            
            return messages
        else:
            logger.error("Only JSON input files are supported")
            return []
    
    def run_interactive_session(self, input_file: str = None, max_messages: int = 50):
        """Run interactive labeling session"""
        print("=== AMHARIC CONLL INTERACTIVE LABELER ===")
        
        if input_file:
            messages = self.load_messages_for_labeling(input_file)
            if not messages:
                print("No messages loaded. Creating sample data...")
                messages = [
                    {'message_id': f'sample_{i+1}', 'text': text, 'tokens': []}
                    for i, text in enumerate([
                        "የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።",
                        "ስልክ ቁጥር 0911234567 ላይ ይደውሉ። አዲስ አበባ ውስጥ ነን።",
                        "የሴቶች ጫማ በ 800 ብር። ፒያሳ አካባቢ ይገኛል።"
                    ])
                ]
        else:
            # Use sample data
            sample_data = self.labeler.create_sample_labeled_data()
            messages = [
                {'message_id': data['message_id'], 'text': data['text'], 'tokens': data['tokens']}
                for data in sample_data
            ]
        
        print(f"Loaded {len(messages)} messages for labeling")
        print("Type 'help' for instructions")
        
        labeled_count = 0
        for i, message in enumerate(messages[:max_messages]):
            print(f"\n--- Message {i+1}/{min(len(messages), max_messages)} ---")
            
            labeled_data = self.label_message_interactive(message)
            
            if labeled_data:
                self.current_session.append(labeled_data)
                labeled_count += 1
                
                # Ask if user wants to continue
                if i < min(len(messages), max_messages) - 1:
                    continue_labeling = input("\nContinue to next message? (y/n/save): ").strip().lower()
                    if continue_labeling == 'n':
                        break
                    elif continue_labeling == 'save':
                        self.save_session()
                        break
        
        print(f"\nLabeling session completed. Labeled {labeled_count} messages.")
        
        if self.current_session:
            save_choice = input("Save labeled data? (y/n): ").strip().lower()
            if save_choice == 'y':
                self.save_session()
    
    def save_session(self, output_file: str = None):
        """Save current labeling session"""
        if not self.current_session:
            print("No labeled data to save.")
            return
        
        if output_file is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/labeled/interactive_session_{timestamp}.txt"
        
        # Ensure directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save in CoNLL format
        self.labeler.save_conll_file(self.current_session, output_file)
        
        # Also save as JSON
        json_file = output_file.replace('.txt', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_session, f, ensure_ascii=False, indent=2)
        
        # Show statistics
        stats = self.labeler.get_entity_statistics(self.current_session)
        print(f"\nSession saved to: {output_file}")
        print(f"Statistics: {stats}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive CoNLL Labeler for Amharic NER")
    parser.add_argument('--input', '-i', help="Input JSON file with messages to label")
    parser.add_argument('--max-messages', '-m', type=int, default=50, help="Maximum messages to label")
    
    args = parser.parse_args()
    
    labeler = InteractiveLabeler()
    labeler.run_interactive_session(args.input, args.max_messages)

if __name__ == "__main__":
    main()
