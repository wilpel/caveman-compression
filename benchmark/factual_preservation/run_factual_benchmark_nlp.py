#!/usr/bin/env python3
"""
Factual Preservation Benchmark for NLP-based Compression

Tests how well facts are preserved after NLP-based caveman compression by using an LLM
to verify that specific facts from the original text are still present in the
compressed version.

Usage:
    python benchmark/factual_preservation/run_factual_benchmark_nlp.py
    python benchmark/factual_preservation/run_factual_benchmark_nlp.py --output results.json
    python benchmark/factual_preservation/run_factual_benchmark_nlp.py --lang es
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import caveman_compress_nlp
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

import os


def load_api_key():
    """Load OpenAI API key from environment or .env file"""
    # Try environment variable first
    api_key = os.environ.get('OPENAI_API_KEY')

    # Try local .env file
    if not api_key:
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.split('=', 1)[1].strip().strip('"\'')
                    break

    if not api_key:
        print("Error: OPENAI_API_KEY not found. Set OPENAI_API_KEY environment variable or create .env file", file=sys.stderr)
        sys.exit(1)

    return api_key


def compress_text(text, lang='en'):
    """Compress text using NLP-based caveman compression"""
    import subprocess

    try:
        result = subprocess.run(
            ["python3", "caveman_compress_nlp.py", "compress", "--lang", lang, text],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"Compression failed: {result.stderr}", file=sys.stderr)
            return None

        output = result.stdout.strip()

        # Extract only the compressed text from the output
        # Format is: header, original, "Compressing...", "CAVEMAN COMPRESSED:", compressed_text, statistics
        if "CAVEMAN COMPRESSED:" in output:
            # Split at the compressed marker and get everything after it
            parts = output.split("CAVEMAN COMPRESSED:")
            if len(parts) > 1:
                # Get the part after "CAVEMAN COMPRESSED:" and before "===="
                compressed_section = parts[1]
                # Split by "====" and take the first part (the actual compressed text)
                if "====" in compressed_section:
                    compressed_text = compressed_section.split("====")[0].strip()
                else:
                    compressed_text = compressed_section.strip()
                return compressed_text

        # Fallback: return full output if parsing fails
        return output
    except subprocess.TimeoutExpired:
        print("Compression timeout", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Compression error: {e}", file=sys.stderr)
        return None


def check_fact_preservation(original_text, compressed_text, fact, question, is_exact_quote, client, model="gpt-4o"):
    """
    Use LLM Q&A to check if a specific fact from the original text is preserved
    in the compressed version by asking a question and verifying the answer.

    Returns:
        dict: {
            'fact': str,
            'question': str,
            'answer_from_compressed': str,
            'preserved': bool,
            'confidence': str,
            'explanation': str,
            'is_exact_quote': bool
        }
    """

    # Step 2: Ask the LLM to answer the question using ONLY the compressed text
    qa_prompt = f"""Answer the following question using ONLY the information provided in the text below. Be specific and complete.

TEXT:
{compressed_text}

QUESTION: {question}

Provide a direct, factual answer. If the information is not available or unclear, say "Information not found" or "Unclear"."""

    try:
        # Get answer from compressed text
        qa_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise question-answering assistant. Answer based only on the provided text."},
                {"role": "user", "content": qa_prompt}
            ],
            temperature=0.1,
        )

        answer = qa_response.choices[0].message.content.strip()

        # Step 3: Verify if the answer contains all the required information from the fact
        verification_prompt = f"""You are verifying if an answer derived from compressed text contains all essential information from an original fact.

ORIGINAL FACT:
{fact}

QUESTION ASKED:
{question}

ANSWER FROM COMPRESSED TEXT:
{answer}

Your task:
1. Check if the answer contains ALL essential information from the original fact
2. For exact quotes, verify the quote itself is word-for-word identical
3. For numbers/dates, verify they are exact
4. For names, verify they are present and correct
5. Accept minor grammatical variations but NOT missing key details

IMPORTANT FOR EXACT QUOTES:
- If fact says "exact quote: 'X'", only verify the quote 'X' itself is preserved exactly
- ALLOW additional context, surrounding sentences, or explanatory text in the answer
- The quote can appear within a longer answer as long as the exact words are present
- Example: Fact requires quote "hello world" - Answer "He said 'hello world' yesterday" is VALID

Respond in JSON format:
{{
    "preserved": true/false,
    "confidence": "HIGH/MEDIUM/LOW",
    "explanation": "brief explanation noting what was preserved or what is missing",
    "missing_details": "list any missing information, or 'none' if complete"
}}

STRICT CRITERIA:
- If answer is "Information not found" or similar, mark preserved as FALSE
- If ANY key detail from the fact is missing in the answer, mark preserved as FALSE
- For exact quotes: verify the quote appears word-for-word (additional context is OK)
- Only mark preserved as TRUE if answer fully supports the fact"""

        verification_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict fact verification assistant. Respond only with valid JSON."},
                {"role": "user", "content": verification_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        result = json.loads(verification_response.choices[0].message.content)
        result['fact'] = fact
        result['question'] = question
        result['answer_from_compressed'] = answer
        result['is_exact_quote'] = is_exact_quote
        return result

    except Exception as e:
        print(f"Error checking fact: {e}", file=sys.stderr)
        return {
            'fact': fact,
            'question': question,
            'answer_from_compressed': 'Error',
            'preserved': False,
            'confidence': 'LOW',
            'explanation': f'Error during evaluation: {str(e)}',
            'missing_details': 'evaluation failed',
            'is_exact_quote': is_exact_quote
        }


def run_benchmark(test_data_path, output_path=None, lang='en', model="gpt-4o", verbose=True):
    """
    Run the factual preservation benchmark for NLP compression

    Args:
        test_data_path: Path to test_data.json
        output_path: Path to save results (optional)
        lang: Language code for NLP compression
        model: OpenAI model to use for verification
        verbose: Print progress information
    """

    # Load API key and initialize client
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    # Load test data
    with open(test_data_path, 'r') as f:
        data = json.load(f)

    test_cases = data['test_cases']

    if verbose:
        print(f"Running Factual Preservation Benchmark (NLP-based Compression)")
        print(f"Language: {lang}")
        print(f"Verification Model: {model}")
        print(f"Test Cases: {len(test_cases)}")
        print("=" * 70)

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'compression_type': 'nlp',
            'language': lang,
            'verification_model': model,
            'total_test_cases': len(test_cases),
            'test_data_file': str(test_data_path)
        },
        'test_results': [],
        'summary': {}
    }

    total_facts = 0
    total_preserved = 0

    # Run each test case
    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case['id']
        original_text = test_case['text']
        fact_checks = test_case['fact_checks']

        if verbose:
            print(f"\n[{i}/{len(test_cases)}] Testing: {test_id}")
            print(f"  Facts to verify: {len(fact_checks)}")

        # Compress the text
        if verbose:
            print(f"  Compressing text (NLP)...")

        compressed_text = compress_text(original_text, lang)

        if compressed_text is None:
            if verbose:
                print(f"  ❌ Compression failed, skipping test case")
            continue

        # Calculate compression ratio
        original_size = len(original_text)
        compressed_size = len(compressed_text)
        compression_ratio = (1 - compressed_size / original_size) * 100

        if verbose:
            print(f"  Original: {original_size} chars")
            print(f"  Compressed: {compressed_size} chars")
            print(f"  Compression: {compression_ratio:.1f}%")
            print(f"  Checking facts...")

        # Check each fact
        fact_results = []
        preserved_count = 0

        for j, fact_check in enumerate(fact_checks, 1):
            fact = fact_check['fact']
            question = fact_check['question']
            is_exact_quote = fact_check.get('is_exact_quote', False)

            if verbose:
                print(f"    [{j}/{len(fact_checks)}] Fact: {fact[:80]}..." if len(fact) > 80 else f"    [{j}/{len(fact_checks)}] Fact: {fact}")

            fact_result = check_fact_preservation(
                original_text,
                compressed_text,
                fact,
                question,
                is_exact_quote,
                client,
                model
            )

            fact_results.append(fact_result)

            if fact_result['preserved']:
                preserved_count += 1

            if verbose:
                status = "✓" if fact_result['preserved'] else "✗"
                print(f"      Q: {fact_result.get('question', 'N/A')}")
                print(f"      A: {fact_result.get('answer_from_compressed', 'N/A')[:100]}...")
                print(f"      {status} {fact_result['confidence']} - {fact_result['explanation'][:80]}...")
                if not fact_result['preserved'] and 'missing_details' in fact_result:
                    print(f"      Missing: {fact_result['missing_details']}")

        preservation_rate = (preserved_count / len(fact_checks)) * 100 if fact_checks else 0

        test_result = {
            'id': test_id,
            'original_text': original_text,
            'compressed_text': compressed_text,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'total_facts': len(fact_checks),
            'facts_preserved': preserved_count,
            'preservation_rate': preservation_rate,
            'fact_results': fact_results
        }

        results['test_results'].append(test_result)

        total_facts += len(fact_checks)
        total_preserved += preserved_count

        if verbose:
            print(f"  Results: {preserved_count}/{len(fact_checks)} facts preserved ({preservation_rate:.1f}%)")

    # Calculate summary statistics
    if results['test_results']:
        avg_compression = sum(r['compression_ratio'] for r in results['test_results']) / len(results['test_results'])
        avg_preservation = sum(r['preservation_rate'] for r in results['test_results']) / len(results['test_results'])
        overall_preservation = (total_preserved / total_facts * 100) if total_facts > 0 else 0

        results['summary'] = {
            'total_facts_tested': total_facts,
            'total_facts_preserved': total_preserved,
            'overall_preservation_rate': overall_preservation,
            'average_compression_ratio': avg_compression,
            'average_preservation_rate_per_test': avg_preservation,
            'tests_completed': len(results['test_results']),
            'tests_failed': len(test_cases) - len(results['test_results'])
        }

    # Save results
    if output_path:
        output_file = Path(output_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(__file__).parent / "reports" / f"factual_preservation_nlp_{timestamp}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY (NLP-based Compression)")
        print("=" * 70)
        print(f"Total Facts Tested: {total_facts}")
        print(f"Total Facts Preserved: {total_preserved}")

        if results['test_results']:
            print(f"Overall Preservation Rate: {overall_preservation:.1f}%")
            print(f"Average Compression Ratio: {avg_compression:.1f}%")
            print(f"Average Preservation Rate: {avg_preservation:.1f}%")
        else:
            print("No test results available")

        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark factual preservation in NLP-based caveman compression"
    )
    parser.add_argument(
        '--test-data',
        default=Path(__file__).parent / 'test_data.json',
        help='Path to test data JSON file (default: test_data.json)'
    )
    parser.add_argument(
        '--output',
        help='Output file path for results (default: results_nlp_TIMESTAMP.json)'
    )
    parser.add_argument(
        '--lang',
        default='en',
        help='Language code for NLP compression (default: en)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o',
        help='OpenAI model to use for verification (default: gpt-4o)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Verify test data exists
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        print(f"Error: Test data file not found: {test_data_path}", file=sys.stderr)
        sys.exit(1)

    # Run benchmark
    try:
        run_benchmark(
            test_data_path,
            args.output,
            args.lang,
            args.model,
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
