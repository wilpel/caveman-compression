# Semantic Losslessness Benchmark

Tests whether caveman compression preserves semantic information by comparing an LLM's ability to answer comprehension questions from both original and compressed text.

## Methodology

1. **Compress** the story using caveman compression
2. **Ask** 8 comprehension questions to LLM using original text
3. **Ask** same questions using compressed text
4. **Compare** answers using LLM as semantic similarity judge
5. **Score** each answer pair (0-100) for semantic match

## Files

- `story.txt` - Test story: "The Cartographer's Dilemma"
- `questions.txt` - 8 comprehension questions with expected answers
- `run_benchmark.py` - Automated benchmark script
- `results_*.json` - Timestamped results from benchmark runs

## Usage

```bash
# Run benchmark
python benchmark/run_benchmark.py

# Results are saved to benchmark/results_<timestamp>.json
```

## Scoring

- **≥90**: Semantically lossless
- **75-89**: Minimal semantic loss
- **<75**: Significant semantic loss

## Example Output

```
Average semantic match score: 92.5/100
Questions with matching answers: 7/8 (87.5%)
Character reduction: 45.2%

✅ VERDICT: Compression is semantically lossless (score >= 90)
```
