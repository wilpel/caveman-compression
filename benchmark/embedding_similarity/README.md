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

## Expected Results

High-quality compression should maintain **cosine similarity ≥0.90** even with 20-40% token reduction, proving that semantic meaning is preserved in vector space.
