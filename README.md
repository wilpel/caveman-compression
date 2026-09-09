# opencode-caveman-compress

MLM-based text compression for LLM context optimization. Fork of [wilpel/caveman-compression](https://github.com/wilpel/caveman-compression) with additional features.

**Why use many token when few token do trick?**

## Features

- **MLM compression** using masked language models (RoBERTa for English, CamemBERT for French)
- **NLP fallback** for active languages when MLM is unavailable (spaCy-based)
- **Auto language detection** via fastText (170+ languages)
- **Dynamic presets** calibrated from NLP compression ratio
- **Sentence or text mode** for different compression contexts
- **Custom MLM models** via build argument
- **HTTP API** and **MCP server** for integration
- **Docker support** for easy deployment

## Quick Start

### Docker

Models are provisioned at build time, so the image is immutable and the
runtime is offline (it never downloads models). No volumes are mounted.

```bash
# Build (provisions the default en/fr models into the image)
docker build -t caveman-compression .

# Run
docker run -p 3000:3000 caveman-compression

# Build with an additional catalogue language (de/es)
docker build --build-arg LANGUAGES=en,fr,de -t caveman-compression .

# Or with docker-compose
docker-compose up -d
```

`LANGUAGES` and `CUSTOM_MLM_MODELS` are build arguments, not runtime settings.
During the build, `download_models.py` provisions the requested models and writes
a *language manifest* (`/app/models/languages.json`) recording exactly which
languages were successfully provisioned. The runtime reads that manifest and
never consults `LANGUAGES`/`CUSTOM_MLM_MODELS` environment variables, so the
active-language set is frozen to the image. The container runs as a non-root
user, is fully offline, and performs no model provisioning after build.

### API Usage

```bash
# Compress text (default: preset=lite, mode=sentence)
curl -X POST http://localhost:3000/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "The reason your React component is re-rendering is likely because you are creating a new object reference on each render cycle."}'

# Response
{
  "compressed": "reason your React is re-rendering is likely creating a object reference on each render cycle.",
  "language": "en",
  "model": "caveman-mlm-en",
  "preset": "lite",
  "mode": "sentence",
  "original_size": 127,
  "compressed_size": 93,
  "compression_ratio": 0.73
}
```

## Presets

Presets are dynamically calibrated from NLP compression ratio. The NLP compressor removes stop words, determiners, and auxiliaries. MLM presets are calculated relative to NLP:

| Preset | Formula | Effect |
|--------|---------|--------|
| `lite` | = NLP | Similar to NLP compression (~30% removed) |
| `full` | = NLP × 1.5 | More aggressive (~45% removed) |
| `ultra` | = NLP × 2 | Maximum compression (~60% removed) |

**Default:** `lite` (fast, simple, effective)

## Modes

| Mode | NLP Context | MLM Context | Performance | Use Case |
|------|-------------|-------------|-------------|----------|
| `sentence` | Per sentence | Word in sentence | Fast | Default, preserves structure |
| `text` | Full text | Word in text | Slower | Long texts, global context |

**Default:** `sentence`

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to compress |
| `language` | string | auto | Language code (en, fr, de, etc.) |
| `preset` | string | `lite` | Compression preset: `lite`, `full`, `ultra` |
| `mode` | string | `sentence` | Compression mode: `sentence`, `text` |
| `method` | string | `mlm` | Compression method: `mlm`, `nlp` |

### Method

- `mlm` (default): Uses MLM for active languages (the ones listed in
  `LANGUAGES`). Inactive languages are returned unchanged.
- `nlp`: Force NLP mode (spaCy-based, no MLM) for an active language.

## Examples

### French

```bash
curl -X POST http://localhost:3000/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, je travaille sur un projet complexe qui nécessite une attention particulière aux détails et une compréhension approfondie des exigences du client.", "preset": "lite"}'

# Response
{
  "compressed": "Bonjour, je travaille sur un projet complexe nécessite particulière détails et compréhension approfondie exigences du client.",
  "language": "fr",
  "model": "caveman-mlm-fr",
  "preset": "lite",
  "mode": "sentence",
  "original_size": 155,
  "compressed_size": 126,
  "compression_ratio": 0.81
}
```

### English

```bash
curl -X POST http://localhost:3000/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "The reason your React component is re-rendering is likely because you are creating a new object reference on each render cycle.", "preset": "full"}'

# Response
{
  "compressed": "reason React is re-rendering is likely creating a reference on cycle.",
  "language": "en",
  "model": "caveman-mlm-en",
  "preset": "full",
  "mode": "sentence",
  "original_size": 127,
  "compressed_size": 69,
  "compression_ratio": 0.54
}
```

## Custom Models

Add custom MLM models via the `CUSTOM_MLM_MODELS` build argument (models are
provisioned during the build):

```bash
docker build \
  --build-arg LANGUAGES=en,fr,sv \
  --build-arg CUSTOM_MLM_MODELS='{"sv": {"model": "KB/bert-base-swedish-cased", "spacy": "sv_core_news_sm"}}' \
  -t caveman-compression .
```

Each custom entry must provide both a `model` and a `spacy` pipeline. Entries
missing either are skipped with a warning, and any language listed in
`LANGUAGES` without an available spaCy pipeline fails the build instead of
baking an image that cannot run. Custom models are baked into the image at
build time and recorded in the language manifest; the runtime never downloads
models.

## MCP Server

The MCP server provides the same functionality via the Model Context Protocol:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "caveman_compress",
    "arguments": {
      "text": "Your text here",
      "preset": "lite",
      "mode": "sentence"
    }
  }
}
```

## Language Detection

Language detection uses fastText as the primary method, with a word-list
heuristic fallback (English vs French) when fastText is unavailable or the
model fails to load:

- **fastText**: Fast, accurate, 170+ languages
- **Word-list heuristic**: Last resort for unknown text (en/fr only)

When neither method can determine a language - including the word-list
heuristic finding no positive English or French evidence - the text is
returned unchanged rather than being misclassified.

## Performance

| Method | 100 words | 1000 words |
|--------|-----------|------------|
| NLP | ~100ms | ~1s |
| MLM (sentence mode) | ~10s | ~100s |
| MLM (text mode) | ~10s | ~100s |

MLM is ~100x slower than NLP due to neural network inference per word.

## Supported Languages

The MLM languages are controlled by the `LANGUAGES` build argument (default
`en,fr`). Only languages listed there are provisioned at build time and
activated at runtime (via the baked-in language manifest). Inactive languages -
including catalogue languages not listed in `LANGUAGES` and unknown codes - are
returned unchanged (no compression and no NLP fallback).

### Catalogue (LT n-gram)

The full catalogue is defined in `language_catalog.py` and maps each language
to an MLM model and a spaCy pipeline:

| Language | Model | spaCy | Default |
|----------|-------|-------|---------|
| English | roberta-base | en_core_web_sm | yes |
| French | camembert-base | fr_core_news_sm | yes |
| German | bert-base-german-cased | de_core_news_sm | no |
| Spanish | dccuchile/bert-base-spanish-wwm-cased | es_core_news_sm | no |

Enable `de`/`es` by listing them in `LANGUAGES` (e.g.
`--build-arg LANGUAGES=en,fr,de`); no `CUSTOM_MLM_MODELS` is required because
they are already in the catalogue. Turkish is not supported: spaCy ships no
Turkish pipeline, and every language needs a spaCy pipeline for tokenization
and NER.

### NLP Models (fallback)

spaCy models for 15+ languages with multilingual fallback, used when
`method=nlp` is requested for an active language or MLM is unavailable for an
active language.

## Environment Variables

`LANGUAGES` and `CUSTOM_MLM_MODELS` are **build arguments only** (see above).
They are consumed during `docker build`, and the result is recorded in the
language manifest baked into the image. They are *not* read at runtime, so a
running container's active-language set cannot be changed by setting environment
variables. The only runtime environment variable is `PORT`.

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGUAGES` | `en,fr` | Languages provisioned at build time (build argument) |
| `CUSTOM_MLM_MODELS` | (empty) | JSON mapping of custom language codes to `{"model": ..., "spacy": ...}` (build argument) |
| `LANGUAGE_MANIFEST_PATH` | `/app/models/languages.json` | Location of the baked-in language manifest (local-dev override) |
| `PORT` | `3000` | HTTP server port (honored by the gunicorn bind and the health check) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/compress` | POST | Compress text |

## License

Same as upstream [wilpel/caveman-compression](https://github.com/wilpel/caveman-compression).
