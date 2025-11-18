#!/usr/bin/env python3
"""
NLP-based caveman compression without LLM.
Fast, free, deterministic - uses stop word removal and grammar stripping.
Supports multiple languages via spaCy.
"""

import sys
import argparse
from pathlib import Path

try:
    import spacy
    from spacy.language import Language
except ImportError:
    print("Error: spaCy not installed. Install with:", file=sys.stderr)
    print("  pip install spacy", file=sys.stderr)
    print("  python -m spacy download en_core_web_sm", file=sys.stderr)
    print("  python -m spacy download xx_ent_wiki_sm  # for other languages", file=sys.stderr)
    sys.exit(1)

# Language model cache
_nlp_models = {}

def get_nlp_model(lang='en'):
    """Load or retrieve cached spaCy model"""
    if lang in _nlp_models:
        return _nlp_models[lang]

    # Try to load language-specific model
    model_names = {
        'en': 'en_core_web_sm',
        'es': 'es_core_news_sm',
        'de': 'de_core_news_sm',
        'fr': 'fr_core_news_sm',
        'it': 'it_core_news_sm',
        'pt': 'pt_core_news_sm',
        'nl': 'nl_core_news_sm',
        'el': 'el_core_news_sm',
        'nb': 'nb_core_news_sm',
        'lt': 'lt_core_news_sm',
        'ja': 'ja_core_news_sm',
        'zh': 'zh_core_web_sm',
        'pl': 'pl_core_news_sm',
        'ro': 'ro_core_news_sm',
        'ru': 'ru_core_news_sm',
    }

    model_name = model_names.get(lang, 'xx_ent_wiki_sm')  # multilingual fallback

    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Warning: Model '{model_name}' not found. Using multilingual model.", file=sys.stderr)
        try:
            nlp = spacy.load('xx_ent_wiki_sm')
        except OSError:
            print("Error: No spaCy models found. Install with:", file=sys.stderr)
            print(f"  python -m spacy download {model_names.get('en', 'en_core_web_sm')}", file=sys.stderr)
            sys.exit(1)

    _nlp_models[lang] = nlp
    return nlp

def count_tokens(text):
    """Estimate token count: characters / 4"""
    return len(text.strip()) // 4

def compress_text(text, lang='en'):
    """Apply NLP-based compression using spaCy"""
    nlp = get_nlp_model(lang)
    doc = nlp(text)

    compressed_sentences = []

    for sent in doc.sents:
        kept_tokens = []

        for token in sent:
            # Skip punctuation (except important ones like numbers with decimals)
            if token.is_punct and token.text not in ['-', '/', ':', '%', '$', '€', '£']:
                continue

            # Skip stop words (articles, conjunctions, etc.)
            if token.is_stop:
                continue

            # Skip auxiliary verbs (is, are, was, were, have, has, etc.)
            if token.pos_ == 'AUX':
                continue

            # Skip determiners (the, a, an, this, that, etc.)
            if token.pos_ == 'DET':
                continue

            # Skip some adverbs (very, really, quite, etc.) but keep important ones
            if token.pos_ == 'ADV' and token.text.lower() in {
                'very', 'really', 'quite', 'extremely', 'incredibly', 'absolutely',
                'totally', 'completely', 'utterly', 'highly', 'particularly',
                'especially', 'truly', 'actually', 'basically', 'essentially'
            }:
                continue

            # Skip coordinating conjunctions in some cases (and, but, or)
            if token.pos_ == 'CCONJ' and token.text.lower() in {'and', 'or'}:
                continue

            # Keep everything else: nouns, verbs, adjectives, numbers, proper nouns, etc.
            kept_tokens.append(token.text)

        # Join kept tokens
        if kept_tokens:
            compressed_sentences.append(' '.join(kept_tokens) + '.')

    result = ' '.join(compressed_sentences)

    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:]

    return result

def decompress_text(text):
    """Simple decompression - just capitalize properly and clean up"""
    # Split into sentences
    sentences = text.split('.')
    result = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Capitalize first letter
            sentence = sentence[0].upper() + sentence[1:]
            result.append(sentence)

    return '. '.join(result) + '.'

def main():
    parser = argparse.ArgumentParser(
        description='NLP-based caveman compression (no LLM required, multilingual)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Compress text (auto-detect language)
  python caveman_compress_nlp.py compress "Your text here"

  # Compress from file
  python caveman_compress_nlp.py compress -f input.txt

  # Specify language
  python caveman_compress_nlp.py compress -f input.txt -l es

  # Save to file
  python caveman_compress_nlp.py compress -f input.txt -o output.txt

  # Decompress
  python caveman_compress_nlp.py decompress "compressed text"

Supported languages: en, es, de, fr, it, pt, nl, el, nb, lt, ja, zh, pl, ro, ru
(and many more via multilingual model)
        '''
    )

    parser.add_argument('mode', choices=['compress', 'decompress'],
                       help='compress or decompress')
    parser.add_argument('text', nargs='?', help='text to process')
    parser.add_argument('-f', '--file', help='input file path')
    parser.add_argument('-o', '--output', help='output file path')
    parser.add_argument('-l', '--lang', default='en',
                       help='language code (en, es, de, fr, etc.)')

    args = parser.parse_args()

    # Get input text
    if args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        input_text = input_path.read_text()
    elif args.text:
        input_text = args.text
    else:
        print("Error: Provide text or use -f for file input", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Process
    print("=" * 60)
    print(f"MODE: {args.mode.upper()} (NLP-BASED)")
    print(f"LANGUAGE: {args.lang}")
    print("=" * 60)
    print()

    if args.mode == 'compress':
        print("ORIGINAL TEXT:")
        print(input_text)
        print()
        print("Compressing...")
        print()

        output_text = compress_text(input_text, args.lang)

        print("CAVEMAN COMPRESSED:")
        print(output_text)
        print()
        print("=" * 60)
        print("STATISTICS:")
        orig_tokens = count_tokens(input_text)
        comp_tokens = count_tokens(output_text)
        reduction = ((orig_tokens - comp_tokens) / orig_tokens * 100) if orig_tokens > 0 else 0
        print(f"  Original:    {len(input_text)} chars ≈ {orig_tokens} tokens")
        print(f"  Compressed:  {len(output_text)} chars ≈ {comp_tokens} tokens")
        print(f"  Reduction:  {reduction:.1f}%")
        print("=" * 60)

    else:  # decompress
        print("COMPRESSED TEXT:")
        print(input_text)
        print()
        print("Decompressing...")
        print()

        output_text = decompress_text(input_text)

        print("DECOMPRESSED:")
        print(output_text)
        print()

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_text)
        print(f"\nSaved to: {args.output}")

if __name__ == '__main__':
    main()
