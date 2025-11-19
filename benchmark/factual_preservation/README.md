# Factual Preservation Benchmark

Tests how well factual information is preserved after caveman compression using question-answering verification.

**Methodology:** Instead of just checking if facts are present, the benchmark:
1. Converts each fact into a natural question
2. Asks an LLM to answer using ONLY the compressed text
3. Verifies the answer contains all essential information from the original fact

This ensures facts are not just present but actually usable and retrievable from the compressed text.

## Structure

```
benchmark/factual_preservation/
├── README.md                       # This file
├── test_data.json                 # Test cases with texts and facts to verify
├── run_factual_benchmark.py       # Benchmark script for LLM-based compression
├── run_factual_benchmark_nlp.py   # Benchmark script for NLP-based compression
└── reports/                       # Generated benchmark reports (gitignored)
    ├── factual_preservation_TIMESTAMP.json
    └── factual_preservation_nlp_TIMESTAMP.json
```

## Usage

### LLM-based Compression Benchmark

```bash
# Run the benchmark (uses gpt-4o for compression)
python benchmark/factual_preservation/run_factual_benchmark.py

# Specify output file
python benchmark/factual_preservation/run_factual_benchmark.py --output my_results.json

# Use different model
python benchmark/factual_preservation/run_factual_benchmark.py --model gpt-4o-mini

# Quiet mode
python benchmark/factual_preservation/run_factual_benchmark.py --quiet
```

### NLP-based Compression Benchmark

```bash
# Run the NLP benchmark (no LLM for compression, free and fast)
python benchmark/factual_preservation/run_factual_benchmark_nlp.py

# Specify language for NLP compression
python benchmark/factual_preservation/run_factual_benchmark_nlp.py --lang es

# Specify output file
python benchmark/factual_preservation/run_factual_benchmark_nlp.py --output my_results.json

# Use different model for verification (not compression)
python benchmark/factual_preservation/run_factual_benchmark_nlp.py --model gpt-4o-mini

# Quiet mode
python benchmark/factual_preservation/run_factual_benchmark_nlp.py --quiet
```

## Test Data Format

The `test_data.json` file contains test cases with:
- **text**: The original text to compress
- **facts**: List of specific facts to verify in the compressed version
- **id**: Unique identifier for the test case

## How It Works

1. Loads test texts and their associated facts
2. Compresses each text using caveman compression
3. For each fact:
   - Generates a natural question that would elicit that fact
   - Asks LLM to answer using ONLY the compressed text as context
   - Verifies the answer contains all essential information
4. Generates a detailed report with preservation rates and compression ratios

## Example

**Fact:** "Henderson said the exact quote: 'I heard what sounded like a gavel'"

**Question Generated:** "What exact quote did Henderson say?"

**Answer from Compressed Text:** "I heard what sounded like a gavel," Henderson told reporters...

**Verification:** ✓ HIGH confidence - Exact quote preserved in its entirety

## Output

Results are automatically saved to `reports/factual_preservation_TIMESTAMP.json` and include:
- Compression ratios
- Fact preservation rates (overall and per-test)
- For each fact:
  - The question generated
  - The answer extracted from compressed text
  - Verification result with confidence level
  - Explanation and missing details (if any)

## Evaluation Strictness

- **Preserved (TRUE)**: Answer from compressed text contains ALL essential information
- **Not Preserved (FALSE)**: ANY key detail missing or answer is "Information not found"
- **HIGH confidence**: All details present and accurate
- **MEDIUM confidence**: All details present but minor inference needed
- **LOW confidence**: Key details missing or altered
- **Exact quotes**: Must match word-for-word
