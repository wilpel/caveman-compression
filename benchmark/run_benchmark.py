#!/usr/bin/env python3
"""
Benchmark script to test semantic losslessness of caveman compression.

Tests whether an LLM can answer comprehension questions equally well
from compressed vs original text.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import compression modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

# Import compression function
import subprocess

def load_api_key():
    """Load OpenAI API key from environment or .env file"""
    import os

    # Try environment variable first
    api_key = os.environ.get('OPENAI_API_KEY')

    # Try local .env file
    if not api_key:
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.split('=', 1)[1].strip().strip('"\'')
                    break

    if not api_key:
        print("Error: OPENAI_API_KEY not found. Set OPENAI_API_KEY environment variable or create .env file", file=sys.stderr)
        sys.exit(1)

    return api_key

def compress_text(text):
    """Compress text using the LLM-based compressor"""
    script_path = Path(__file__).parent.parent / 'caveman_compress.py'

    result = subprocess.run(
        ['python', str(script_path), 'compress', text],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Compression failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Extract compressed text from output
    output = result.stdout
    lines = output.split('\n')

    # Find "CAVEMAN COMPRESSED:" section
    compressed_start = None
    for i, line in enumerate(lines):
        if 'CAVEMAN COMPRESSED:' in line:
            compressed_start = i + 1
            break

    if compressed_start is None:
        print("Error: Could not find compressed text in output", file=sys.stderr)
        sys.exit(1)

    # Find the end (statistics section)
    compressed_end = None
    for i in range(compressed_start, len(lines)):
        if '=' in lines[i] and len(lines[i].strip('=')) == 0:
            compressed_end = i
            break

    compressed_text = '\n'.join(lines[compressed_start:compressed_end]).strip()
    return compressed_text

def ask_question(client, context, question, model='gpt-4o'):
    """Ask the LLM a question about the given context"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions about provided text. Answer concisely and accurately based only on the information given."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            }
        ],
        temperature=0  # Deterministic answers
    )

    return response.choices[0].message.content.strip()

def compare_answers(client, answer1, answer2, expected_answer, model='gpt-4o'):
    """Use LLM to compare semantic similarity of answers"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an objective evaluator. Compare two answers to see if they convey the same semantic meaning. Respond with a JSON object containing 'match' (boolean), 'score' (0-100), and 'explanation' (brief reason)."
            },
            {
                "role": "user",
                "content": f"""Expected answer: {expected_answer}

Answer from original text: {answer1}

Answer from compressed text: {answer2}

Do these answers convey the same essential information? Rate the semantic match from 0-100."""
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    return result

def parse_questions(questions_file):
    """Parse questions and expected answers from file"""
    content = questions_file.read_text()
    questions = []

    for block in content.split('\n\n'):
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        if len(lines) >= 2:
            question_line = lines[0]
            answer_line = lines[1]

            # Remove question number
            question = question_line.split('.', 1)[1].strip() if '.' in question_line else question_line

            # Extract answer
            answer = answer_line.replace('ANSWER:', '').strip()

            questions.append({
                'question': question,
                'expected_answer': answer
            })

    return questions

def main():
    print("=" * 80)
    print("CAVEMAN COMPRESSION SEMANTIC LOSSLESSNESS BENCHMARK")
    print("=" * 80)
    print()

    # Load files
    benchmark_dir = Path(__file__).parent
    story_file = benchmark_dir / 'story.txt'
    questions_file = benchmark_dir / 'questions.txt'

    if not story_file.exists() or not questions_file.exists():
        print("Error: story.txt or questions.txt not found", file=sys.stderr)
        sys.exit(1)

    original_text = story_file.read_text()
    questions = parse_questions(questions_file)

    print(f"Loaded story: {len(original_text)} characters")
    print(f"Loaded {len(questions)} questions")
    print()

    # Compress the story
    print("Compressing story...")
    compressed_text = compress_text(original_text)
    print(f"Compressed: {len(compressed_text)} characters ({100 - len(compressed_text)/len(original_text)*100:.1f}% reduction)")
    print()

    # Initialize OpenAI client
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    # Test each question
    results = []
    print("Testing questions...")
    print()

    for i, q in enumerate(questions, 1):
        print(f"Question {i}: {q['question'][:60]}...")

        # Ask question on original text
        answer_original = ask_question(client, original_text, q['question'])

        # Ask question on compressed text
        answer_compressed = ask_question(client, compressed_text, q['question'])

        # Compare answers
        comparison = compare_answers(
            client,
            answer_original,
            answer_compressed,
            q['expected_answer']
        )

        result = {
            'question': q['question'],
            'expected_answer': q['expected_answer'],
            'answer_original': answer_original,
            'answer_compressed': answer_compressed,
            'score': comparison.get('score', 0),
            'match': comparison.get('match', False),
            'explanation': comparison.get('explanation', '')
        }

        results.append(result)

        print(f"  Score: {result['score']}/100 - {result['explanation']}")
        print()

    # Calculate overall statistics
    avg_score = sum(r['score'] for r in results) / len(results)
    match_count = sum(1 for r in results if r['match'])

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Average semantic match score: {avg_score:.1f}/100")
    print(f"Questions with matching answers: {match_count}/{len(results)} ({match_count/len(results)*100:.1f}%)")
    print(f"Character reduction: {100 - len(compressed_text)/len(original_text)*100:.1f}%")
    print()

    # Save detailed results
    output_file = benchmark_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'story_length_original': len(original_text),
            'story_length_compressed': len(compressed_text),
            'compression_ratio': len(compressed_text) / len(original_text),
            'average_score': avg_score,
            'match_rate': match_count / len(results),
            'questions': results
        }, f, indent=2)

    print(f"Detailed results saved to: {output_file}")

    # Print verdict
    print()
    print("=" * 80)
    if avg_score >= 90:
        print("✅ VERDICT: Compression is semantically lossless (score >= 90)")
    elif avg_score >= 75:
        print("⚠️  VERDICT: Compression has minimal semantic loss (score 75-89)")
    else:
        print("❌ VERDICT: Compression has significant semantic loss (score < 75)")
    print("=" * 80)

if __name__ == '__main__':
    main()
