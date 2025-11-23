#!/usr/bin/env python3
"""
Caveman Compression Tool
Converts normal English to caveman compression and vice versa using OpenAI API
"""

import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI
import numpy as np

from .utils import load_api_key

# Load prompts from files
PROMPTS_DIR = Path(__file__).parent / 'prompts'

def load_prompt(filename):
    """Load prompt from prompts directory"""
    prompt_path = PROMPTS_DIR / filename
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    return prompt_path.read_text()

COMPRESSION_PROMPT = load_prompt('compression.txt')
DECOMPRESSION_PROMPT = load_prompt('decompression.txt')


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
    ]

    # Count code-like patterns
    code_score = sum(1 for indicator in code_indicators if indicator in text)

    # Check for balanced braces/brackets (common in code)
    brace_count = text.count('{') + text.count('}') + text.count('[') + text.count(']')

    # Check for natural language indicators
    words = text.split()
    if len(words) < 5:
        return True  # Short text, treat as natural language

    # If high code indicators or many braces, treat as code
    if code_score >= 2 or brace_count > len(words) * 0.2:
        return False

    return True


def split_sentences(text):
    """Split text into sentences by period, preserving sentence boundaries"""
    # If no period in text, return as single sentence
    if '.' not in text:
        return [text]

    # Simple sentence splitting by periods followed by space or end of string
    import re
    # Split on '. ' or '.\n' or '. \n' but keep the period
    sentences = re.split(r'\.(\s+)', text)

    # Reconstruct sentences with their periods
    result = []
    i = 0
    while i < len(sentences):
        if sentences[i].strip():
            sentence = sentences[i]
            # Add back the period if this isn't the last fragment
            if i + 1 < len(sentences):
                sentence = sentence + '.'
            result.append(sentence.strip())
        i += 2 if i + 1 < len(sentences) else 1

    return result if result else [text]


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
    if args.file:
        with open(args.file, 'r') as f:
            input_text = f.read().strip()
    elif args.text:
        input_text = args.text
    else:
        parser.error("Must provide either text argument or -f/--file option")

    if not input_text:
        print("Error: Input text is empty", file=sys.stderr)
        sys.exit(1)

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
        print(f"  Original:   {len(input_text):4d} chars ≈ {orig_tokens:3d} tokens")
        print(f"  Compressed: {len(result):4d} chars ≈ {comp_tokens:3d} tokens")
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
        print(f"  Caveman:  {len(input_text):4d} chars ≈ {cave_tokens:3d} tokens")
        print(f"  Normal:   {len(result):4d} chars ≈ {norm_tokens:3d} tokens")
        print(f"  Expansion: {expansion:.1f}%")
        print(f"{'='*60}\n")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"Output saved to: {args.output}\n")


if __name__ == "__main__":
    main()
