"""
Corpus Structuring Module for Braille ETL Pipeline
Creates structured training datasets from parallel text-Braille corpus
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional
import random
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorpusStructurer:
    def __init__(self, input_dir: str = "data/output", output_dir: str = "data/structured"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset split ratios
        self.split_ratios = {
            'train': 0.7,
            'validation': 0.15,
            'test': 0.15
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'high': 0.7,
            'medium': 0.4,
            'minimum': 0.2
        }

    def load_corpus_data(self) -> Dict:
        """
        Load the parallel corpus data
        """
        corpus_file = os.path.join(self.input_dir, 'parallel_corpus.json')
        if not os.path.exists(corpus_file):
            logger.error(f"Parallel corpus file not found: {corpus_file}")
            return {}
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_sentence_pairs(self, text: str, braille: str, metadata: Dict) -> List[Dict]:
        """
        Break down paragraph-level pairs into sentence-level pairs
        """
        pairs = []
        
        # Split text into sentences
        import re
        
        # Sentence splitting patterns for English and Hindi
        english_pattern = r'[.!?]+\s+'
        hindi_pattern = r'[।.!?]+\s+'
        
        language = metadata.get('translation_metadata', {}).get('language', 'english')
        
        if language == 'hindi':
            text_sentences = re.split(hindi_pattern, text.strip())
            # For Braille, we'll need to estimate splits (simplified approach)
            braille_sentences = re.split(r'\s{2,}', braille.strip())
        else:
            text_sentences = re.split(english_pattern, text.strip())
            braille_sentences = re.split(r'\s{2,}', braille.strip())
        
        # Clean empty sentences
        text_sentences = [s.strip() for s in text_sentences if s.strip()]
        braille_sentences = [s.strip() for s in braille_sentences if s.strip()]
        
        # If we have matching number of sentences, pair them directly
        if len(text_sentences) == len(braille_sentences):
            for i, (text_sent, braille_sent) in enumerate(zip(text_sentences, braille_sentences)):
                if len(text_sent.split()) >= 3:  # Minimum length filter
                    pairs.append({
                        'text': text_sent,
                        'braille': braille_sent,
                        'sentence_id': i + 1,
                        'parent_id': metadata.get('id', 'unknown'),
                        'language': language
                    })
        else:
            # If sentences don't align, create one pair for the entire text
            pairs.append({
                'text': text,
                'braille': braille,
                'sentence_id': 1,
                'parent_id': metadata.get('id', 'unknown'),
                'language': language
            })
        
        return pairs

    def filter_by_quality(self, pairs: List[Dict], min_quality: float = 0.2) -> List[Dict]:
        """
        Filter pairs based on quality metrics
        """
        filtered_pairs = []
        
        for pair in pairs:
            quality_score = self.calculate_pair_quality(pair)
            
            if quality_score >= min_quality:
                pair['quality_score'] = quality_score
                filtered_pairs.append(pair)
            else:
                logger.debug(f"Filtered out low quality pair: {quality_score:.2f}")
        
        return filtered_pairs

    def calculate_pair_quality(self, pair: Dict) -> float:
        """
        Calculate quality score for a text-Braille pair
        """
        text = pair.get('text', '')
        braille = pair.get('braille', '')
        
        if not text or not braille:
            return 0.0
        
        score = 0.0
        
        # Length alignment score (0.3)
        text_len = len(text)
        braille_len = len(braille)
        if text_len > 0 and braille_len > 0:
            ratio = min(braille_len / text_len, text_len / braille_len)
            length_score = ratio * 0.3
            score += length_score
        
        # Word count score (0.2)
        text_words = len(text.split())
        braille_words = len(braille.split())
        if text_words >= 3 and braille_words >= 3:
            word_score = 0.2
            score += word_score
        
        # Character diversity score (0.2)
        text_chars = len(set(text.lower()))
        braille_chars = len(set(braille))
        if text_chars >= 10 and braille_chars >= 5:
            char_score = 0.2
            score += char_score
        
        # Language consistency score (0.3)
        language = pair.get('language', 'english')
        if language == 'english':
            # Check for English characteristics
            english_chars = sum(c.isalpha() and ord(c) < 128 for c in text)
            if english_chars / len(text) > 0.8:
                score += 0.3
        elif language == 'hindi':
            # Check for Hindi characteristics
            hindi_chars = sum('\u0900' <= c <= '\u097F' for c in text)
            if hindi_chars > 0:
                score += 0.3
        
        return min(score, 1.0)

    def create_training_splits(self, pairs: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Split data into training, validation, and test sets
        """
        # Shuffle pairs for random split
        random.seed(42)  # For reproducibility
        shuffled_pairs = pairs.copy()
        random.shuffle(shuffled_pairs)
        
        total_pairs = len(shuffled_pairs)
        train_size = int(total_pairs * self.split_ratios['train'])
        val_size = int(total_pairs * self.split_ratios['validation'])
        
        splits = {
            'train': shuffled_pairs[:train_size],
            'validation': shuffled_pairs[train_size:train_size + val_size],
            'test': shuffled_pairs[train_size + val_size:]
        }
        
        return splits

    def create_metadata_annotations(self, pairs: List[Dict]) -> List[Dict]:
        """
        Add comprehensive metadata to each pair
        """
        annotated_pairs = []
        
        for i, pair in enumerate(pairs):
            annotated_pair = pair.copy()
            
            # Add unique identifier
            pair_hash = hashlib.md5(
                (pair['text'] + pair['braille']).encode('utf-8')
            ).hexdigest()[:8]
            
            annotated_pair.update({
                'pair_id': f"{pair.get('language', 'en')}_{i+1:04d}_{pair_hash}",
                'created_at': datetime.now().isoformat(),
                'text_statistics': self.calculate_text_statistics(pair['text']),
                'braille_statistics': self.calculate_braille_statistics(pair['braille']),
                'alignment_score': self.calculate_alignment_score(pair['text'], pair['braille']),
                'complexity_level': self.assess_complexity_level(pair['text'])
            })
            
            annotated_pairs.append(annotated_pair)
        
        return annotated_pairs

    def calculate_text_statistics(self, text: str) -> Dict:
        """
        Calculate detailed statistics for text
        """
        words = text.split()
        sentences = len([s for s in text.split('.') if s.strip()])
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': max(sentences, 1),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / max(sentences, 1),
            'unique_words': len(set(word.lower() for word in words)),
            'vocabulary_richness': len(set(word.lower() for word in words)) / len(words) if words else 0
        }

    def calculate_braille_statistics(self, braille: str) -> Dict:
        """
        Calculate statistics for Braille text
        """
        words = braille.split()
        braille_chars = [c for c in braille if ord(c) >= 0x2800 and ord(c) <= 0x28FF]
        
        return {
            'character_count': len(braille),
            'braille_character_count': len(braille_chars),
            'word_count': len(words),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'unique_patterns': len(set(braille_chars)),
            'pattern_density': len(braille_chars) / len(braille) if braille else 0
        }

    def calculate_alignment_score(self, text: str, braille: str) -> float:
        """
        Calculate how well text and Braille are aligned
        """
        text_words = len(text.split())
        braille_words = len(braille.split())
        
        if text_words == 0 or braille_words == 0:
            return 0.0
        
        # Word count alignment
        word_ratio = min(text_words, braille_words) / max(text_words, braille_words)
        
        # Length alignment
        length_ratio = min(len(text), len(braille)) / max(len(text), len(braille))
        
        return (word_ratio + length_ratio) / 2

    def assess_complexity_level(self, text: str) -> str:
        """
        Assess the complexity level of the text
        """
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        sentence_count = max(1, len([s for s in text.split('.') if s.strip()]))
        avg_sentence_length = len(words) / sentence_count
        
        complexity_score = 0
        
        # Word length complexity
        if avg_word_length > 6:
            complexity_score += 2
        elif avg_word_length > 4:
            complexity_score += 1
        
        # Sentence length complexity
        if avg_sentence_length > 20:
            complexity_score += 2
        elif avg_sentence_length > 10:
            complexity_score += 1
        
        # Vocabulary complexity (simplified)
        unique_ratio = len(set(word.lower() for word in words)) / len(words) if words else 0
        if unique_ratio > 0.8:
            complexity_score += 1
        
        if complexity_score >= 4:
            return 'high'
        elif complexity_score >= 2:
            return 'medium'
        else:
            return 'low'

    def export_structured_datasets(self, splits: Dict[str, List[Dict]], 
                                 language: str) -> None:
        """
        Export structured datasets in multiple formats
        """
        lang_dir = os.path.join(self.output_dir, language)
        os.makedirs(lang_dir, exist_ok=True)
        
        for split_name, pairs in splits.items():
            if not pairs:
                continue
            
            split_dir = os.path.join(lang_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # JSON format (complete data)
            json_file = os.path.join(split_dir, f'{split_name}_data.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
            
            # Separate text and Braille files
            text_file = os.path.join(split_dir, f'{split_name}_text.txt')
            braille_file = os.path.join(split_dir, f'{split_name}_braille.txt')
            
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(pair['text'] for pair in pairs))
            
            with open(braille_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(pair['braille'] for pair in pairs))
            
            # CSV format for easy analysis
            csv_file = os.path.join(split_dir, f'{split_name}_pairs.csv')
            self.export_to_csv(pairs, csv_file)
            
            logger.info(f"Exported {len(pairs)} pairs to {split_name} set for {language}")

    def export_to_csv(self, pairs: List[Dict], filename: str) -> None:
        """
        Export pairs to CSV format
        """
        import csv
        
        if not pairs:
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'pair_id', 'text', 'braille', 'language', 'quality_score',
                'text_word_count', 'braille_word_count', 'complexity_level',
                'alignment_score', 'parent_id'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for pair in pairs:
                row = {
                    'pair_id': pair.get('pair_id', ''),
                    'text': pair.get('text', ''),
                    'braille': pair.get('braille', ''),
                    'language': pair.get('language', ''),
                    'quality_score': pair.get('quality_score', 0),
                    'text_word_count': pair.get('text_statistics', {}).get('word_count', 0),
                    'braille_word_count': pair.get('braille_statistics', {}).get('word_count', 0),
                    'complexity_level': pair.get('complexity_level', ''),
                    'alignment_score': pair.get('alignment_score', 0),
                    'parent_id': pair.get('parent_id', '')
                }
                writer.writerow(row)

    def create_dataset_summary(self, all_structured_data: Dict) -> None:
        """
        Create comprehensive dataset summary
        """
        summary = {
            'creation_date': datetime.now().isoformat(),
            'total_languages': len(all_structured_data),
            'languages': {},
            'overall_statistics': {
                'total_pairs': 0,
                'total_words': 0,
                'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'complexity_distribution': {'high': 0, 'medium': 0, 'low': 0}
            }
        }
        
        for language, data in all_structured_data.items():
            lang_stats = {
                'total_pairs': 0,
                'splits': {},
                'quality_stats': {'high': 0, 'medium': 0, 'low': 0},
                'complexity_stats': {'high': 0, 'medium': 0, 'low': 0}
            }
            
            for split_name, pairs in data.items():
                split_stats = {
                    'pair_count': len(pairs),
                    'avg_text_length': 0,
                    'avg_braille_length': 0,
                    'avg_quality_score': 0
                }
                
                if pairs:
                    split_stats['avg_text_length'] = sum(
                        len(p.get('text', '')) for p in pairs
                    ) / len(pairs)
                    
                    split_stats['avg_braille_length'] = sum(
                        len(p.get('braille', '')) for p in pairs
                    ) / len(pairs)
                    
                    split_stats['avg_quality_score'] = sum(
                        p.get('quality_score', 0) for p in pairs
                    ) / len(pairs)
                    
                    # Count quality and complexity distributions
                    for pair in pairs:
                        quality = pair.get('quality_score', 0)
                        if quality >= 0.7:
                            lang_stats['quality_stats']['high'] += 1
                        elif quality >= 0.4:
                            lang_stats['quality_stats']['medium'] += 1
                        else:
                            lang_stats['quality_stats']['low'] += 1
                        
                        complexity = pair.get('complexity_level', 'low')
                        lang_stats['complexity_stats'][complexity] += 1
                
                lang_stats['splits'][split_name] = split_stats
                lang_stats['total_pairs'] += len(pairs)
            
            summary['languages'][language] = lang_stats
            summary['overall_statistics']['total_pairs'] += lang_stats['total_pairs']
            
            # Aggregate quality and complexity stats
            for quality_level in ['high', 'medium', 'low']:
                summary['overall_statistics']['quality_distribution'][quality_level] += \
                    lang_stats['quality_stats'][quality_level]
                summary['overall_statistics']['complexity_distribution'][quality_level] += \
                    lang_stats['complexity_stats'][quality_level]
        
        # Save summary
        summary_file = os.path.join(self.output_dir, 'dataset_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset summary saved to {summary_file}")
        self.print_summary_report(summary)

    def print_summary_report(self, summary: Dict) -> None:
        """
        Print a formatted summary report to console
        """
        print("\n" + "="*60)
        print("BRAILLE CORPUS DATASET SUMMARY")
        print("="*60)
        
        print(f"Created: {summary['creation_date']}")
        print(f"Total Languages: {summary['total_languages']}")
        print(f"Total Pairs: {summary['overall_statistics']['total_pairs']}")
        
        print("\nLanguage Breakdown:")
        for lang, stats in summary['languages'].items():
            print(f"  {lang.upper()}:")
            print(f"    Total Pairs: {stats['total_pairs']}")
            for split, split_stats in stats['splits'].items():
                print(f"    {split.capitalize()}: {split_stats['pair_count']} pairs")
        
        print("\nQuality Distribution:")
        quality_dist = summary['overall_statistics']['quality_distribution']
        total_pairs = summary['overall_statistics']['total_pairs']
        if total_pairs > 0:
            for level, count in quality_dist.items():
                percentage = (count / total_pairs) * 100
                print(f"  {level.capitalize()}: {count} ({percentage:.1f}%)")
        
        print("\nComplexity Distribution:")
        complexity_dist = summary['overall_statistics']['complexity_distribution']
        if total_pairs > 0:
            for level, count in complexity_dist.items():
                percentage = (count / total_pairs) * 100
                print(f"  {level.capitalize()}: {count} ({percentage:.1f}%)")
        
        print("="*60 + "\n")

    def structure_corpus(self, languages: List[str] = ['english', 'hindi']) -> Dict:
        """
        Main method to structure the entire corpus
        """
        logger.info("Starting corpus structuring process...")
        
        # Load corpus data
        corpus_data = self.load_corpus_data()
        if not corpus_data:
            logger.error("No corpus data loaded. Exiting.")
            return {}
        
        all_structured_data = {}
        
        for language in languages:
            logger.info(f"Processing {language} corpus...")
            
            # Filter corpus data by language
            lang_data = [
                item for item in corpus_data.get('entries', [])
                if item.get('metadata', {}).get('translation_metadata', {}).get('language', '').lower() == language.lower()
            ]
            
            if not lang_data:
                logger.warning(f"No data found for language: {language}")
                continue
            
            # Process each entry
            all_pairs = []
            for entry in lang_data:
                text = entry.get('text', '')
                braille = entry.get('braille', '')
                metadata = entry.get('metadata', {})
                
                if text and braille:
                    pairs = self.create_sentence_pairs(text, braille, metadata)
                    all_pairs.extend(pairs)
            
            if not all_pairs:
                logger.warning(f"No valid pairs created for {language}")
                continue
            
            logger.info(f"Created {len(all_pairs)} initial pairs for {language}")
            
            # Filter by quality
            quality_pairs = self.filter_by_quality(all_pairs, self.quality_thresholds['minimum'])
            logger.info(f"Retained {len(quality_pairs)} quality pairs for {language}")
            
            # Add comprehensive metadata
            annotated_pairs = self.create_metadata_annotations(quality_pairs)
            
            # Create training splits
            splits = self.create_training_splits(annotated_pairs)
            
            # Export structured datasets
            self.export_structured_datasets(splits, language)
            
            all_structured_data[language] = splits
        
        # Create overall dataset summary
        if all_structured_data:
            self.create_dataset_summary(all_structured_data)
        
        logger.info("Corpus structuring completed successfully!")
        return all_structured_data


def main():
    """
    Main execution function
    """
    # Initialize corpus structurer
    structurer = CorpusStructurer(
        input_dir="data/output",
        output_dir="data/structured"
    )
    
    # Structure corpus for both languages
    structured_data = structurer.structure_corpus(['english', 'hindi'])
    
    if structured_data:
        print(f"Successfully structured corpus for {len(structured_data)} languages")
        for lang, data in structured_data.items():
            total_pairs = sum(len(pairs) for pairs in data.values())
            print(f"  {lang}: {total_pairs} total pairs")
    else:
        print("No structured data created")


if __name__ == "__main__":
    main()