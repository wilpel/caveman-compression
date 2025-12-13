#!/usr/bin/env python3
"""
Caveman Compression Tool
Converts normal English to caveman compression and vice versa using OpenAI API
"""

import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError, AuthenticationError
import spacy
import numpy as np

from utils import load_api_key

# Load prompts from files
PROMPTS_DIR = Path(__file__).parent / 'prompts'
MAX_TEXT_LENGTH = 100000 # Maximum characters for input text

def load_prompt(filename):
    """Load prompt from prompts directory"""
    prompt_path = PROMPTS_DIR / filename
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}", file=sys.stderr)
        print(f"Please ensure '{filename}' exists in the '{PROMPTS_DIR}' directory.", file=sys.stderr)
        sys.exit(1)
    try:
        return prompt_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        print(f"Error: Could not read prompt file '{prompt_path}' due to invalid UTF-8 encoding.", file=sys.stderr)
        sys.exit(1)

COMPRESSION_PROMPT = load_prompt('compression.txt')
DECOMPRESSION_PROMPT = load_prompt('decompression.txt')


# Language model cache
_nlp_models = {}

def get_nlp_model(lang='en'):
    """Load or retrieve cached spaCy model"""
    if lang in _nlp_models:
        return _nlp_models[lang]

    model_name = 'en_core_web_sm'  # Hardcoding for this script

    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Error: spaCy model '{model_name}' not found.", file=sys.stderr)
        print(f"Please install it by running:", file=sys.stderr)
        print(f"  python -m spacy download {model_name}", file=sys.stderr)
        sys.exit(1)

    _nlp_models[lang] = nlp
    return nlp


def count_tokens(text):
    """Estimate tokens using character count / 4"""
    return len(text.strip()) // 4


def is_text_content(text):
    """Detect if content is natural language text vs code/structured data"""
    # Check for common code indicators
    code_indicators = [
        'def ', 'class ', 'function ', 'import ', 'const ', 'let ', 'var ',
        'public ', 'private ', 'protected ', '#include', 'package ',
        '=>', '->', '::', '!=', '==', '<=', '>=', '&&', '||',
        'print(', 'console.log(', 'return ', 'if (', 'for (', 'while (',
    ]

    # Count code-like patterns
    code_score = sum(1 for indicator in code_indicators if indicator in text)

    # Check for balanced braces/brackets (common in code)
    brace_count = text.count('{') + text.count('}') + text.count('[') + text.count(']') + text.count('(') + text.count(')')

    # Check for natural language indicators
    words = text.split()
    if len(words) < 5:
        return True  # Short text, treat as natural language

    # If high code indicators or many braces, treat as code
    if code_score >= 2 or brace_count > len(words) * 0.2:
        return False

    return True


def split_sentences(text):
    """Split text into sentences using spaCy"""
    nlp = get_nlp_model()
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def get_embedding(client, text, model="text-embedding-3-large"):
    """Get embedding vector for text"""
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Warning: Failed to get embedding: {e}", file=sys.stderr)
        return None


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2)


def calculate_embedding_loss(client, original_text, compressed_text, model="text-embedding-3-large"):
    """Calculate embedding similarity loss between original and compressed text"""
    original_emb = get_embedding(client, original_text, model)
    compressed_emb = get_embedding(client, compressed_text, model)

    if original_emb is None or compressed_emb is None:
        return None

    similarity = cosine_similarity(original_emb, compressed_emb)
    # Loss is 1 - similarity (0 = perfect preservation, 1 = total loss)
    loss = 1 - similarity
    return loss, similarity


def compress_text(text, model="gpt-4o", calculate_embeddings=True):
    """Compress normal English to caveman compression"""
    client = OpenAI(api_key=load_api_key())

    # Detect if this is natural language text
    is_text = is_text_content(text)

    compressed = ""
    try:
        # If it's text, use sentence-by-sentence compression with gpt-4o-mini
        if is_text:
            sentences = split_sentences(text)

            # If only one sentence or very short, compress as whole
            if len(sentences) <= 1:
                model = "gpt-4o-mini"
                prompt = COMPRESSION_PROMPT.format(text=text)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert at caveman compression. Always compress the provided text, never ask for clarification."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                )
                compressed = response.choices[0].message.content.strip()
            else:
                # Compress sentence by sentence with gpt-4o-mini
                compressed_sentences = []
                for sentence in sentences:
                    if not sentence.strip():
                        continue

                    prompt = COMPRESSION_PROMPT.format(text=sentence)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert at caveman compression. Always compress the provided text, never ask for clarification."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                    )
                    compressed_sent = response.choices[0].message.content.strip()
                    compressed_sentences.append(compressed_sent)

                compressed = ' '.join(compressed_sentences)
        else:
            # For code/structured data, use original model and compress as whole
            prompt = COMPRESSION_PROMPT.format(text=text)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at caveman compression. Always compress the provided text, never ask for clarification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            compressed = response.choices[0].message.content.strip()

    except AuthenticationError:
        print("Error: OpenAI authentication failed. Please check your API key.", file=sys.stderr)
        sys.exit(1)
    except APIStatusError as e:
        if e.status_code == 429:
            print("Error: OpenAI rate limit exceeded. Please wait and try again later.", file=sys.stderr)
        elif e.status_code >= 500:
            print(f"Error: OpenAI server error ({e.status_code}). Service may be temporarily unavailable.", file=sys.stderr)
        else:
            print(f"Error: OpenAI API returned an unexpected status code {e.status_code}: {e.response}", file=sys.stderr)
        sys.exit(1)
    except APITimeoutError:
        print("Error: OpenAI API request timed out. Please check your internet connection or try again later.", file=sys.stderr)
        sys.exit(1)
    except APIConnectionError:
        print("Error: Could not connect to OpenAI API. Please check your internet connection.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during compression: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate statistics
    original_tokens = count_tokens(text)
    compressed_tokens = count_tokens(compressed)
    reduction = ((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens > 0 else 0

    # Calculate embedding loss if requested
    embedding_loss = None
    embedding_similarity = None
    if calculate_embeddings:
        result = calculate_embedding_loss(client, text, compressed)
        if result is not None:
            embedding_loss, embedding_similarity = result

    return compressed, original_tokens, compressed_tokens, reduction, embedding_loss, embedding_similarity


def decompress_text(text, model="gpt-4o"):
    """Decompress caveman compression to normal English"""
    client = OpenAI(api_key=load_api_key())

    decompressed = ""
    try:
        prompt = DECOMPRESSION_PROMPT.format(text=text)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at expanding compressed text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        decompressed = response.choices[0].message.content.strip()

    except AuthenticationError:
        print("Error: OpenAI authentication failed. Please check your API key.", file=sys.stderr)
        sys.exit(1)
    except APIStatusError as e:
        if e.status_code == 429:
            print("Error: OpenAI rate limit exceeded. Please wait and try again later.", file=sys.stderr)
        elif e.status_code >= 500:
            print(f"Error: OpenAI server error ({e.status_code}). Service may be temporarily unavailable.", file=sys.stderr)
        else:
            print(f"Error: OpenAI API returned an unexpected status code {e.status_code}: {e.response}", file=sys.stderr)
        sys.exit(1)
    except APITimeoutError:
        print("Error: OpenAI API request timed out. Please check your internet connection or try again later.", file=sys.stderr)
        sys.exit(1)
    except APIConnectionError:
        print("Error: Could not connect to OpenAI API. Please check your internet connection.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during decompression: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate statistics
    caveman_tokens = count_tokens(text)
    normal_tokens = count_tokens(decompressed)
    expansion = ((normal_tokens - caveman_tokens) / caveman_tokens * 100) if caveman_tokens > 0 else 0

    return decompressed, caveman_tokens, normal_tokens, expansion


def main():
    parser = argparse.ArgumentParser(
        description='Caveman Compression Tool - Compress or decompress text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress normal English to caveman
  python caveman_compress.py compress "In order to optimize the database..."

  # Decompress caveman to normal English
  python caveman_compress.py decompress "Need fast queries. Add index..."

  # Read from file
  python caveman_compress.py compress -f input.txt

  # Save output to file
  python caveman_compress.py compress -f input.txt -o output.txt
        """
    )

    parser.add_argument(
        'mode',
        choices=['compress', 'decompress', 'c', 'd'],
        help='Mode: compress (c) or decompress (d)'
    )
    parser.add_argument(
        'text',
        nargs='?',
        help='Text to process (omit if using -f)'
    )
    parser.add_argument(
        '-f', '--file',
        help='Read input from file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Write output to file'
    )
    parser.add_argument(
        '-m', '--model',
        default='gpt-4o',
        help='OpenAI model to use (default: gpt-4o)'
    )

    args = parser.parse_args()

    # Normalize mode
    mode = 'compress' if args.mode in ['compress', 'c'] else 'decompress'

    # Get input text
    input_text = ""
    if args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        try:
            input_text = input_path.read_text(encoding='utf-8').strip()
        except PermissionError:
            print(f"Error: Permission denied when reading file: {args.file}", file=sys.stderr)
            sys.exit(1)
        except UnicodeDecodeError:
            print(f"Error: Input file '{args.file}' contains invalid UTF-8 encoding.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: An unexpected error occurred while reading file '{args.file}': {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        input_text = args.text
    else:
        parser.error("Must provide either text argument or -f/--file option")

    # Process text
    print(f"\n{'='*60}")
    print(f"MODE: {mode.upper()}")
    print(f"MODEL: {args.model}")
    print(f"{'='*60}\n")

    if mode == 'compress':
        print("ORIGINAL TEXT:")
        print(f"{input_text}\n")

        print("Compressing...\n")
        result, orig_tokens, comp_tokens, reduction, emb_loss, emb_sim = compress_text(input_text, args.model)

        print("CAVEMAN COMPRESSED:")
        print(f"{result}\n")

        print(f"{'='*60}")
        print("STATISTICS:")
        print(f"  Original:   {len(input_text):4d} chars \u2248 {orig_tokens:3d} tokens")
        print(f"  Compressed: {len(result):4d} chars \u2248 {comp_tokens:3d} tokens")
        print(f"  Reduction:  {reduction:.1f}%")

        if emb_loss is not None and emb_sim is not None:
            print(f"\n  Embedding Similarity: {emb_sim:.4f}")
            print(f"  Embedding Loss:       {emb_loss:.4f}")
            if emb_sim >= 0.95:
                print(f"  Quality: Excellent - virtually identical semantic meaning")
            elif emb_sim >= 0.90:
                print(f"  Quality: Good - minor semantic drift")
            elif emb_sim >= 0.85:
                print(f"  Quality: Moderate - noticeable drift")
            else:
                print(f"  Quality: Poor - significant semantic drift")

        print(f"{'='*60}\n")

    else:  # decompress
        print("CAVEMAN TEXT:")
        print(f"{input_text}\n")

        print("Decompressing...\n")
        result, cave_tokens, norm_tokens, expansion = decompress_text(input_text, args.model)

        print("NORMAL ENGLISH:")
        print(f"{result}\n")

        print(f"{'='*60}")
        print("STATISTICS:")
        print(f"  Caveman:  {len(input_text):4d} chars \u2248 {cave_tokens:3d} tokens")
        print(f"  Normal:   {len(result):4d} chars \u2248 {norm_tokens:3d} tokens")
        print(f"  Expansion: {expansion:.1f}%")
        print(f"{'='*60}\n")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_path.write_text(result, encoding='utf-8')
            print(f"Output saved to: {args.output}\n")
        except PermissionError:
            print(f"Error: Permission denied when writing to file: {args.output}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: An unexpected error occurred while writing to file '{args.output}': {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()