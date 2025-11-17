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

# Try to get API key from environment variable or matrix-social .env file
API_KEY = os.getenv('OPENAI_API_KEY')

if not API_KEY:
    # Try to read from matrix-social .env file
    matrix_social_env = Path(__file__).parent.parent / 'matrix-social' / 'backend' / '.env'
    if matrix_social_env.exists():
        with open(matrix_social_env, 'r') as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    API_KEY = line.split('=', 1)[1].strip()
                    break

if not API_KEY:
    print("Error: OPENAI_API_KEY not found. Set environment variable or create .env file", file=sys.stderr)
    sys.exit(1)

COMPRESSION_PROMPT = """You are a caveman compression expert. Convert the following normal English text into caveman compression following these rules:

CRITICAL REQUIREMENT - ZERO INFORMATION LOSS:
You MUST preserve 100% of semantic information. Every fact, number, adjective with meaning, constraint, and detail must appear in the output. Only remove grammatical scaffolding, NOT content.

CORE PRINCIPLES:
1. Strip connectives (therefore, however, consequently, because, due to, in order to)
2. Minimize words per statement (target 2-5 words per sentence)
3. Use action verbs (do, make, fix, check, test, find, need, use, try)
4. Be concrete over abstract (never "values in range 5-6", always "five or six" or "test five, test six")
5. No passive voice (never "value is calculated", always "calculate value")
6. Strip ONLY decorative adjectives/adverbs - KEEP ones with semantic meaning

WHAT TO KEEP (these carry information):
- Numbers, quantities, sizes (small, medium, large, 5, ten, many)
- Meaningful adjectives (fast, slow, broken, expensive, critical, optional)
- Names, places, roles, titles
- Constraints and conditions
- All facts and logical steps

WHAT TO REMOVE (these are just grammar):
- Articles: a, an, the
- Decorative words: very, quite, rather, somewhat, really
- Connectives: because, therefore, however, although
- Passive voice constructions
- Pronouns (replace with nouns)

EXAMPLES:
"I am a 26 year old CTO at a medium large company"
→ "Am 26 years old. Am CTO. Company medium large."
(Keeps: 26, CTO, medium large)

"The very important database needs to be optimized"
→ "Important database needs optimization."
(Keeps: important - has meaning. Removes: very - decorative)

"Use binary search instead of sequential scanning"
→ "Use binary search. Not sequential scanning."
(Keeps BOTH methods - what to use AND what to avoid)

REMEMBER: If original mentions what something replaces or contrasts with, KEEP BOTH parts.

Output ONLY the caveman compressed text, nothing else.

TEXT TO COMPRESS:
{text}"""

DECOMPRESSION_PROMPT = """You are a language expansion expert. Convert the following caveman-compressed text back into proper, fluent English while preserving ALL semantic information.

The caveman text uses:
- Very short sentences (2-5 words)
- No connectives
- Active voice
- Concrete language
- Minimal articles

Your task:
1. Expand sentences to natural English length
2. Add appropriate connectives (because, therefore, however, etc.)
3. Add articles (a, an, the) where natural
4. Ensure smooth flow between sentences
5. Maintain all facts, constraints, and logical steps
6. Use proper grammar and style

Output ONLY the expanded English text, nothing else.

CAVEMAN TEXT TO EXPAND:
{text}"""


def count_tokens(text):
    """Estimate tokens using character count / 4"""
    return len(text.strip()) // 4


def compress_text(text, model="gpt-4o"):
    """Compress normal English to caveman compression"""
    client = OpenAI(api_key=API_KEY)

    prompt = COMPRESSION_PROMPT.format(text=text)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at caveman compression."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    compressed = response.choices[0].message.content.strip()

    # Calculate statistics
    original_tokens = count_tokens(text)
    compressed_tokens = count_tokens(compressed)
    reduction = ((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens > 0 else 0

    return compressed, original_tokens, compressed_tokens, reduction


def decompress_text(text, model="gpt-4o"):
    """Decompress caveman compression to normal English"""
    client = OpenAI(api_key=API_KEY)

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
        result, orig_tokens, comp_tokens, reduction = compress_text(input_text, args.model)

        print("CAVEMAN COMPRESSED:")
        print(f"{result}\n")

        print(f"{'='*60}")
        print("STATISTICS:")
        print(f"  Original:   {len(input_text):4d} chars ≈ {orig_tokens:3d} tokens")
        print(f"  Compressed: {len(result):4d} chars ≈ {comp_tokens:3d} tokens")
        print(f"  Reduction:  {reduction:.1f}%")
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
