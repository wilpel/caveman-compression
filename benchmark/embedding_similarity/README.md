# Embedding Similarity Benchmark

Tests how much semantic similarity is preserved in vector embeddings after caveman compression.

**Methodology:**
1. Generate embeddings for original text using OpenAI's text-embedding model
2. Compress the text using caveman compression
3. Generate embeddings for compressed text
4. Calculate cosine similarity between original and compressed embeddings
5. Compare compression ratio vs semantic drift

This measures whether compressed text maintains semantic meaning in vector space, which is critical for RAG systems and semantic search.

## Usage

### LLM-based Compression

```bash
python benchmark/embedding_similarity/run_embedding_benchmark.py

# Specify output file
python benchmark/embedding_similarity/run_embedding_benchmark.py --output results.json

# Use different models
python benchmark/embedding_similarity/run_embedding_benchmark.py --compression-model gpt-4o-mini --embedding-model text-embedding-3-small
```

### NLP-based Compression

```bash
python benchmark/embedding_similarity/run_embedding_benchmark_nlp.py

# Specify language
python benchmark/embedding_similarity/run_embedding_benchmark_nlp.py --lang es
```

## Metrics

- **Cosine Similarity**: 0.0 (completely different) to 1.0 (identical)
  - **≥0.95**: Excellent - virtually identical semantic meaning
  - **0.90-0.94**: Good - minor semantic drift
  - **0.85-0.89**: Moderate - noticeable but acceptable drift
  - **<0.85**: Poor - significant semantic drift

## Benchmark Results

### Latest Results (LLM-based Compression)

```
Model: gpt-4o-mini
Embedding Model: text-embedding-3-large
Test Cases: 5

[1/5] technical_documentation
  Compression: 34.0% (329 → 217 chars)
  Cosine similarity: 0.8409 ✗ Poor

[2/5] system_prompt
  Compression: 24.7% (417 → 314 chars)
  Cosine similarity: 0.8813 ⚠ Moderate

[3/5] product_description
  Compression: 13.8% (391 → 337 chars)
  Cosine similarity: 0.9679 ✓ Excellent

[4/5] news_article
  Compression: 17.8% (473 → 389 chars)
  Cosine similarity: 0.9704 ✓ Excellent

[5/5] instructions
  Compression: 24.5% (428 → 323 chars)
  Cosine similarity: 0.9501 ✓ Excellent

Average Cosine Similarity: 0.9221 (Good)
Average Compression Ratio: 23.0%
```

**Key Finding:** Compression maintains **≥0.92 average semantic similarity** with 23% size reduction, proving semantic meaning is well-preserved in vector space. This is critical for RAG systems where compressed documents need to maintain searchability.
