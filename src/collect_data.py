"""
Data Collection Module for Braille ETL Pipeline
Collects unstructured data from various sources including web content and sample documents
"""

import os
import requests
from bs4 import BeautifulSoup
import time
import json
from typing import List, Dict
import logging
from urllib.parse import urljoin, urlparse
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample sources for educational content
        self.sample_sources = {
            'english': [
                'https://www.gutenberg.org/files/74/74-0.txt',  # Adventures of Tom Sawyer
                'https://www.gutenberg.org/files/1342/1342-0.txt',  # Pride and Prejudice
                'https://www.gutenberg.org/files/11/11-0.txt',  # Alice in Wonderland
            ],
            'web_content': [
                'https://en.wikipedia.org/wiki/Braille',
                'https://en.wikipedia.org/wiki/Accessibility',
                'https://en.wikipedia.org/wiki/Assistive_technology'
            ]
        }
        
        # Sample Hindi content (for demonstration)
        self.hindi_samples = [
            "यह एक नमूना हिंदी पाठ है। ब्रेल अनुवाद के लिए उपयोग किया जाता है।",
            "शिक्षा सभी के लिए सुलभ होनी चाहिए। दृष्टिबाधित व्यक्तियों के लिए ब्रेल एक महत्वपूर्ण साधन है।",
            "प्रौद्योगिकी की सहायता से हम बेहतर पहुंच बना सकते हैं।"
        ]

    def collect_web_content(self, url: str, max_paragraphs: int = 10) -> Dict:
        """
        Collect clean text content from web pages
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Extract paragraphs
            paragraphs = []
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if len(text) > 50:  # Filter out short paragraphs
                    paragraphs.append(text)
                    if len(paragraphs) >= max_paragraphs:
                        break
            
            return {
                'url': url,
                'title': soup.title.string if soup.title else '',
                'paragraphs': paragraphs,
                'word_count': sum(len(p.split()) for p in paragraphs),
                'collection_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error collecting from {url}: {e}")
            return None

    def collect_gutenberg_text(self, url: str) -> Dict:
        """
        Collect text from Project Gutenberg
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            text = response.text
            
            # Remove Project Gutenberg header/footer
            start_markers = ['*** START OF', '***START OF']
            end_markers = ['*** END OF', '***END OF']
            
            for marker in start_markers:
                if marker in text:
                    text = text.split(marker, 1)[1]
                    break
            
            for marker in end_markers:
                if marker in text:
                    text = text.split(marker, 1)[0]
                    break
            
            # Split into paragraphs and clean
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
            
            # Limit to reasonable number for processing
            paragraphs = paragraphs[:20]
            
            return {
                'source': url,
                'type': 'gutenberg',
                'paragraphs': paragraphs,
                'word_count': sum(len(p.split()) for p in paragraphs),
                'collection_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error collecting Gutenberg text from {url}: {e}")
            return None

    def create_sample_documents(self) -> List[Dict]:
        """
        Create sample documents for testing when web sources aren't available
        """
        samples = []
        
        # English samples
        english_texts = [
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
            "Education should be accessible to everyone, regardless of their physical abilities. Braille technology helps bridge this gap.",
            "Artificial intelligence is transforming how we process and understand information. Machine learning models can help translate text into accessible formats.",
            "Books have the power to transport us to different worlds and expand our understanding of life.",
            "Technology has revolutionized the way we communicate and share information across the globe."
        ]
        
        for i, text in enumerate(english_texts):
            samples.append({
                'id': f'sample_en_{i+1}',
                'language': 'english',
                'text': text,
                'word_count': len(text.split()),
                'type': 'sample',
                'collection_time': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Hindi samples
        for i, text in enumerate(self.hindi_samples):
            samples.append({
                'id': f'sample_hi_{i+1}',
                'language': 'hindi',
                'text': text,
                'word_count': len(text.split()),
                'type': 'sample',
                'collection_time': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return samples

    def collect_all_data(self, target_samples: int = 25) -> None:
        """
        Main collection method - gathers data from all sources
        """
        logger.info("Starting data collection...")
        collected_data = []
        
        # Try to collect from web sources
        logger.info("Collecting from web sources...")
        for url in self.sample_sources['web_content']:
            data = self.collect_web_content(url)
            if data and data['paragraphs']:
                # Split into individual samples
                for i, paragraph in enumerate(data['paragraphs'][:5]):  # Limit per source
                    collected_data.append({
                        'id': f"web_{len(collected_data)+1}",
                        'source': url,
                        'text': paragraph,
                        'language': 'english',
                        'type': 'web_content',
                        'word_count': len(paragraph.split()),
                        'collection_time': data['collection_time']
                    })
            time.sleep(1)  # Be respectful to servers
        
        # Try to collect from Gutenberg (just one source to avoid overload)
        logger.info("Collecting from Project Gutenberg...")
        gutenberg_data = self.collect_gutenberg_text(self.sample_sources['english'][0])
        if gutenberg_data and gutenberg_data['paragraphs']:
            for i, paragraph in enumerate(gutenberg_data['paragraphs'][:10]):
                collected_data.append({
                    'id': f"gutenberg_{i+1}",
                    'source': gutenberg_data['source'],
                    'text': paragraph,
                    'language': 'english',
                    'type': 'literature',
                    'word_count': len(paragraph.split()),
                    'collection_time': gutenberg_data['collection_time']
                })
        
        # Add sample documents to reach target
        logger.info("Adding sample documents...")
        samples = self.create_sample_documents()
        collected_data.extend(samples)
        
        # Ensure we have the target number of samples
        if len(collected_data) < target_samples:
            # Generate additional samples by modifying existing ones
            additional_needed = target_samples - len(collected_data)
            base_samples = collected_data[-5:]  # Use last 5 as base
            
            for i in range(additional_needed):
                base_sample = base_samples[i % len(base_samples)].copy()
                base_sample['id'] = f"generated_{i+1}"
                base_sample['type'] = 'generated'
                # Slightly modify the text
                text = base_sample['text']
                if len(text) > 100:
                    # Take a different portion of the text
                    words = text.split()
                    start_idx = min(10, len(words) // 4)
                    base_sample['text'] = ' '.join(words[start_idx:start_idx + 20])
                    base_sample['word_count'] = len(base_sample['text'].split())
                
                collected_data.append(base_sample)
        
        # Limit to target number and save
        collected_data = collected_data[:target_samples]
        
        # Save collected data
        output_file = os.path.join(self.output_dir, 'collected_data.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(collected_data, f, indent=2, ensure_ascii=False)
        
        # Save individual files for easier processing
        for item in collected_data:
            filename = f"{item['id']}.txt"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(item['text'])
        
        logger.info(f"Collected {len(collected_data)} samples")
        logger.info(f"Data saved to {output_file}")
        
        # Print summary
        english_count = sum(1 for item in collected_data if item.get('language') == 'english')
        hindi_count = sum(1 for item in collected_data if item.get('language') == 'hindi')
        total_words = sum(item.get('word_count', 0) for item in collected_data)
        
        print(f"\nCollection Summary:")
        print(f"Total samples: {len(collected_data)}")
        print(f"English samples: {english_count}")
        print(f"Hindi samples: {hindi_count}")
        print(f"Total words: {total_words}")

def main():
    collector = DataCollector()
    collector.collect_all_data(target_samples=25)

if __name__ == "__main__":
    main()