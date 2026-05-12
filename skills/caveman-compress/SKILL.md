---
name: caveman-compress
description: Compress verbose prompts, system prompts, RAG chunks, or any text destined for an LLM into a token-efficient form while preserving facts (numbers, names, technical terms, constraints). Use when (1) the user asks to compress, shrink, or shorten a prompt or document for LLM input, (2) the user complains about token cost or context window pressure on an input prompt, (3) the user pastes a long system prompt and wants it tightened, (4) the user mentions caveman-compression, lossless semantic compression, or wants a free/offline grammar-stripping pass. Do NOT use for compressing the assistant's own output — this skill compresses INPUT text only.
---

# caveman-compress

This skill wraps the `caveman_compress_nlp.py` and `caveman_compress_mlm.py` scripts shipped with this plugin. It strips predictable grammar (articles, connectives, passive voice, filler) from input text while keeping unpredictable content (numbers, names, dates, technical terms, constraints) intact.

## When to invoke

- User asks: "compress this prompt", "shrink this system prompt", "make this shorter for the LLM", "reduce tokens on this RAG chunk".
- User pastes a long block and asks for a tightened version to feed to an LLM.
- User mentions caveman-compression, semantic compression, or token-cost on input.

## When NOT to invoke

- The user wants the *assistant's reply* to be terse — that is the `caveman` skill, not this one.
- The user wants lossy summarization or paraphrase — this preserves facts only; structure can drop.
- The user wants binary file compression.

## How to invoke

Use the `/caveman-compress` slash command. Default engine is `nlp` (free, offline, 15-30% reduction). Switch to `mlm` for predictability-aware compression (free, offline, 20-30%) or `llm` for highest reduction (40-58%, requires `OPENAI_API_KEY`).

```
/caveman-compress --engine nlp --file path/to/prompt.txt
/caveman-compress --engine mlm "long verbose text here"
/caveman-compress --engine llm --file system_prompt.md
```

If the venv is not bootstrapped, the slash command bootstraps it automatically. To bootstrap manually, run `/caveman-setup nlp` (or `mlm` / `llm`).

## Engine choice heuristics

| Need | Engine |
|------|--------|
| No API key, fastest install | `nlp` |
| Better ratio, still offline | `mlm` |
| Maximum reduction, OK with API call | `llm` |
| Non-English text | `nlp` with `-l <lang>` |

## Verifying output

After compression, the script reports token counts. Sanity-check that facts (numbers, proper nouns, technical terms) survived. If a critical fact was dropped, switch engines (LLM tends to preserve more) or pre-mark facts in the input.
