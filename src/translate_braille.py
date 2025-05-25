"""
Braille Translation Module for Braille ETL Pipeline
Handles text-to-Braille translation using Liblouis CLI for both English and Hindi
"""

import os
import json
import logging
import subprocess
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def liblouis_cli_translate(text: str, table: str) -> str:
    table_path = f"/usr/share/liblouis/tables/{table}"
    try:
        result = subprocess.run(
            ["lou_translate", table_path],
            input=text.encode(),
            capture_output=True,
            check=True
        )
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Translation failed: {e.stderr.decode()}")

class BrailleTranslator:
    def __init__(self, input_dir: str = "data/processed", output_dir: str = "data/output"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.braille_tables = {
            'english': {
                'grade1': 'en-us-g1.ctb',
                'grade2': 'en-us-g2.ctb',
                'computer': 'en-us-comp6.ctb'
            },
            'hindi': {
                'grade1': 'hi-in-g1.utb',
                'grade2': 'hi-in-g1.utb'
            }
        }

    def get_available_tables(self) -> Dict[str, List[str]]:
        available_tables = {}

        for language, tables in self.braille_tables.items():
            available_tables[language] = []
            for grade, table_name in tables.items():
                table_path = f"/usr/share/liblouis/tables/{table_name}"
                if os.path.exists(table_path):
                    available_tables[language].append(f"{grade}: {table_name}")
                else:
                    logger.warning(f"Table not available: {table_name}")
        return available_tables

    def translate_to_braille(self, text: str, language: str = 'english', grade: int = 2) -> Tuple[str, Dict]:
        if not text.strip():
            return "", {"error": "Empty text provided"}

        table_name = self._get_table_name(language, grade)
        if not table_name:
            return "", {"error": f"No table available for {language} grade {grade}"}

        try:
            braille_text = liblouis_cli_translate(text, table_name)
            metadata = {
                'original_length': len(text),
                'braille_length': len(braille_text),
                'table_used': table_name,
                'language': language,
                'grade': grade,
                'word_count': len(text.split()),
                'compression_ratio': len(braille_text) / len(text) if text else 0
            }
            return braille_text, metadata
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return "", {"error": str(e)}

    def _get_table_name(self, language: str, grade: int) -> Optional[str]:
        if language not in self.braille_tables:
            return None
        tables = self.braille_tables[language]
        if grade == 1:
            return tables.get('grade1')
        elif grade == 2:
            return tables.get('grade2')
        else:
            return tables.get('grade1')

    def translate_with_fallback(self, text: str, language: str = 'english') -> Tuple[str, Dict]:
        braille_text, metadata = self.translate_to_braille(text, language)
        if braille_text and not metadata.get('error'):
            return braille_text, metadata

        fallback_options = []
        if language == 'english':
            fallback_options = [(language, 1), (language, 2)]
        elif language == 'hindi':
            fallback_options = [(language, 1), ('english', 1)]
        else:
            fallback_options = [('english', 1)]

        for fallback_lang, fallback_grade in fallback_options:
            try:
                braille_text, metadata = self.translate_to_braille(text, fallback_lang, fallback_grade)
                if braille_text and not metadata.get('error'):
                    metadata['fallback_used'] = f"{fallback_lang}_grade{fallback_grade}"
                    return braille_text, metadata
            except Exception as e:
                logger.warning(f"Fallback {fallback_lang} grade {fallback_grade} failed: {e}")

        return self._create_ascii_braille(text), {
            'method': 'ascii_fallback',
            'original_length': len(text),
            'warning': 'Used ASCII fallback - not actual Braille'
        }

    def _create_ascii_braille(self, text: str) -> str:
        ascii_mapping = {
            'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
            'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
            'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
            'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
            'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵',
            ' ': ' ', '.': '⠲', ',': '⠂', '!': '⠖', '?': '⠦'
        }
        result = []
        for char in text.lower():
            result.append(ascii_mapping.get(char, '⠿'))
        return ''.join(result)

    def process_extracted_data(self) -> None:
        logger.info("Starting Braille translation...")
        input_file = os.path.join(self.input_dir, 'extracted_cleaned.json')
        if not os.path.exists(input_file):
            logger.error(f"Processed data file not found: {input_file}")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)

        translated_data = []
        translation_stats = {
            'total_processed': 0,
            'successful_translations': 0,
            'fallback_used': 0,
            'failed_translations': 0,
            'languages': {'english': 0, 'hindi': 0}
        }

        for item in processed_data:
            try:
                text = item.get('cleaned_text', '')
                language = item.get('language', 'english')

                if not text:
                    logger.warning(f"No text found for item {item.get('id')}")
                    continue

                braille_text, metadata = self.translate_with_fallback(text, language)

                translated_item = {
                    'id': item['id'],
                    'original_text': text,
                    'braille_text': braille_text,
                    'language': language,
                    'translation_metadata': metadata,
                    'source_info': {
                        'source': item.get('source', 'unknown'),
                        'type': item.get('type', 'text'),
                        'quality_score': item.get('quality_score', 0),
                        'word_count': item.get('word_count', 0)
                    }
                }

                translated_data.append(translated_item)
                translation_stats['total_processed'] += 1
                translation_stats['languages'][language] += 1

                if metadata.get('error'):
                    translation_stats['failed_translations'] += 1
                elif metadata.get('fallback_used'):
                    translation_stats['fallback_used'] += 1
                    translation_stats['successful_translations'] += 1
                else:
                    translation_stats['successful_translations'] += 1

                logger.info(f"Translated {item['id']} ({language})")

            except Exception as e:
                logger.error(f"Error translating {item.get('id', 'unknown')}: {e}")
                translation_stats['failed_translations'] += 1

        output_file = os.path.join(self.output_dir, 'braille_translations.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, indent=2, ensure_ascii=False)

        for item in translated_data:
            if item.get('braille_text'):
                filename = f"{item['id']}_braille.txt"
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(item['braille_text'])

        stats_file = os.path.join(self.output_dir, 'translation_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(translation_stats, f, indent=2)

        logger.info(f"Translation complete. Results saved to {output_file}")
        self._print_translation_summary(translation_stats, translated_data)

    def create_parallel_corpus(self) -> None:
        logger.info("Creating parallel corpus...")
        translated_file = os.path.join(self.output_dir, 'braille_translations.json')
        if not os.path.exists(translated_file):
            logger.error(f"Translated data file not found: {translated_file}")
            return

        with open(translated_file, 'r', encoding='utf-8') as f:
            translated_data = json.load(f)

        parallel_corpus = {
            'english': {'text': [], 'braille': [], 'metadata': []},
            'hindi': {'text': [], 'braille': [], 'metadata': []}
        }

        for item in translated_data:
            language = item.get('language', 'english')
            original_text = item.get('original_text', '')
            braille_text = item.get('braille_text', '')

            if original_text and braille_text and not item.get('translation_metadata', {}).get('error'):
                parallel_corpus[language]['text'].append(original_text)
                parallel_corpus[language]['braille'].append(braille_text)
                parallel_corpus[language]['metadata'].append({
                    'id': item['id'],
                    'source': item.get('source_info', {}).get('source', 'unknown'),
                    'quality_score': item.get('source_info', {}).get('quality_score', 0),
                    'translation_metadata': item.get('translation_metadata', {})
                })

        corpus_file = os.path.join(self.output_dir, 'parallel_corpus.json')
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump(parallel_corpus, f, indent=2, ensure_ascii=False)

        for language, data in parallel_corpus.items():
            if data['text']:
                lang_dir = os.path.join(self.output_dir, language)
                os.makedirs(lang_dir, exist_ok=True)

                with open(os.path.join(lang_dir, 'source_text.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(data['text']))

                with open(os.path.join(lang_dir, 'target_braille.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(data['braille']))

                with open(os.path.join(lang_dir, 'corpus_metadata.json'), 'w', encoding='utf-8') as f:
                    json.dump(data['metadata'], f, indent=2, ensure_ascii=False)

        logger.info(f"Parallel corpus created: {corpus_file}")
        self._print_corpus_summary(parallel_corpus)

    def _print_translation_summary(self, stats: Dict, translated_data: List[Dict]) -> None:
        print(f"\nTranslation Summary:")
        print(f"Total items processed: {stats['total_processed']}")
        print(f"Successful translations: {stats['successful_translations']}")
        print(f"Fallback translations used: {stats['fallback_used']}")
        print(f"Failed translations: {stats['failed_translations']}")
        print(f"English items: {stats['languages']['english']}")
        print(f"Hindi items: {stats['languages']['hindi']}")

        compression_ratios = [
            item.get('translation_metadata', {}).get('compression_ratio', 0)
            for item in translated_data
            if item.get('translation_metadata', {}).get('compression_ratio', 0) > 0
        ]

        if compression_ratios:
            avg_compression = sum(compression_ratios) / len(compression_ratios)
            print(f"Average Braille compression ratio: {avg_compression:.2f}")

    def _print_corpus_summary(self, corpus: Dict) -> None:
        print(f"\nParallel Corpus Summary:")
        for language, data in corpus.items():
            count = len(data['text'])
            if count > 0:
                total_words = sum(len(text.split()) for text in data['text'])
                avg_length = total_words / count if count > 0 else 0
                print(f"{language.title()}: {count} pairs, {total_words} total words, {avg_length:.1f} avg words/pair")

def main():
    translator = BrailleTranslator()
    available = translator.get_available_tables()
    print("Available Braille tables:")
    for lang, tables in available.items():
        print(f"  {lang}: {tables}")

    translator.process_extracted_data()
    translator.create_parallel_corpus()

if __name__ == "__main__":
    main()

