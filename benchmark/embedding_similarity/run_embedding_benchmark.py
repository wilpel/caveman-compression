#!/usr/bin/env python3
"""
Embedding Similarity Benchmark for LLM-based Compression

Tests how well semantic similarity is preserved in vector embeddings
after caveman compression.
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

import os
import subprocess

from utils import load_api_key


def compress_text(text, model="gpt-4o-mini"):
    """Compress text using LLM-based compression"""
    script_path = Path(__file__).parent.parent.parent / 'caveman_compress.py'

    result = subprocess.run(
        ['python3', str(script_path), 'compress', '--model', model, text],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        print(f"Compression failed: {result.stderr}", file=sys.stderr)
        return None

    output = result.stdout

    # Extract compressed text
    if "CAVEMAN COMPRESSED:" in output:
        parts = output.split("CAVEMAN COMPRESSED:")
        if len(parts) > 1:
            compressed_section = parts[1]
            if "====" in compressed_section:
                compressed_text = compressed_section.split("====")[0].strip()
            else:
                compressed_text = compressed_section.strip()
            return compressed_text

    return None


def get_embedding(client, text, model="text-embedding-3-large"):
    """Get embedding vector for text"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2)


def run_benchmark(test_texts, compression_model="gpt-4o-mini", embedding_model="text-embedding-3-large", output_path=None, verbose=True):
    """Run embedding similarity benchmark"""

    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    if verbose:
        print("=" * 70)
        print("EMBEDDING SIMILARITY BENCHMARK (LLM-based Compression)")
        print("=" * 70)
        print(f"Compression Model: {compression_model}")
        print(f"Embedding Model: {embedding_model}")
        print(f"Test Cases: {len(test_texts)}")
        print()

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'compression_type': 'llm',
            'compression_model': compression_model,
            'embedding_model': embedding_model,
            'total_test_cases': len(test_texts)
        },
        'test_results': [],
        'summary': {}
    }

    similarities = []
    compression_ratios = []

    for i, text_data in enumerate(test_texts, 1):
        text_id = text_data.get('id', f'test_{i}')
        original_text = text_data['text']

        if verbose:
            print(f"[{i}/{len(test_texts)}] Testing: {text_id}")
            print(f"  Original length: {len(original_text)} chars")

        # Compress text
        compressed_text = compress_text(original_text, compression_model)
        if compressed_text is None:
            if verbose:
                print(f"  ❌ Compression failed, skipping")
            continue

        compression_ratio = (1 - len(compressed_text) / len(original_text)) * 100

        if verbose:
            print(f"  Compressed length: {len(compressed_text)} chars")
            print(f"  Compression ratio: {compression_ratio:.1f}%")
            print(f"  Generating embeddings...")

        # Get embeddings
        original_embedding = get_embedding(client, original_text, embedding_model)
        compressed_embedding = get_embedding(client, compressed_text, embedding_model)

        # Calculate similarity
        similarity = cosine_similarity(original_embedding, compressed_embedding)
        similarities.append(similarity)
        compression_ratios.append(compression_ratio)

        if verbose:
            print(f"  Cosine similarity: {similarity:.4f}")
            if similarity >= 0.95:
                print(f"  ✓ Excellent - virtually identical semantic meaning")
            elif similarity >= 0.90:
                print(f"  ✓ Good - minor semantic drift")
            elif similarity >= 0.85:
                print(f"  ⚠ Moderate - noticeable drift")
            else:
                print(f"  ✗ Poor - significant semantic drift")
            print()

        results['test_results'].append({
            'id': text_id,
            'original_text': original_text,
            'compressed_text': compressed_text,
            'original_length': len(original_text),
            'compressed_length': len(compressed_text),
            'compression_ratio': compression_ratio,
            'cosine_similarity': float(similarity)
        })

    # Calculate summary statistics
    if similarities:
        results['summary'] = {
            'average_similarity': float(np.mean(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'average_compression_ratio': float(np.mean(compression_ratios)),
            'tests_completed': len(similarities)
        }

    # Save results
    if output_path:
        output_file = Path(output_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(__file__).parent / "reports" / f"embedding_similarity_{timestamp}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    if verbose and similarities:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Average Cosine Similarity: {results['summary']['average_similarity']:.4f}")
        print(f"Similarity Range: {results['summary']['min_similarity']:.4f} - {results['summary']['max_similarity']:.4f}")
        print(f"Average Compression Ratio: {results['summary']['average_compression_ratio']:.1f}%")
        print()
        print(f"Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark embedding similarity preservation in LLM-based caveman compression"
    )
    parser.add_argument(
        '--test-data',
        default=Path(__file__).parent / 'test_texts.json',
        help='Path to test data JSON file'
    )
    parser.add_argument(
        '--output',
        help='Output file path for results'
    )
    parser.add_argument(
        '--compression-model',
        default='gpt-4o-mini',
        help='Model for compression (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--embedding-model',
        default='text-embedding-3-large',
        help='Model for embeddings (default: text-embedding-3-large)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Load test data
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        print(f"Error: Test data file not found: {test_data_path}", file=sys.stderr)
        sys.exit(1)

    with open(test_data_path, 'r') as f:
        data = json.load(f)

    test_texts = data.get('test_texts', [])
    if not test_texts:
        print("Error: No test texts found in data file", file=sys.stderr)
        sys.exit(1)

    # Run benchmark
    try:
        run_benchmark(
            test_texts,
            args.compression_model,
            args.embedding_model,
            args.output,
            verbose=not args.quiet
        )
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n\nBenchmark failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
