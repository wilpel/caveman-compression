---
description: Compress text with caveman-compression (default engine nlp)
argument-hint: "[--engine nlp|mlm|llm] [--file PATH | TEXT]"
allowed-tools: Bash, Read
---

Compress the provided text or file using a caveman-compression engine.

Parse `$ARGUMENTS`:

- `--engine nlp|mlm|llm` selects the engine (default `nlp`).
- `--file PATH` reads input from PATH.
- Otherwise treat the remaining text as the literal input string.

Steps:

1. If `${CLAUDE_PLUGIN_ROOT}/.venv` does not exist, auto-bootstrap by running:

   ```bash
   "${CLAUDE_PLUGIN_ROOT}/bin/caveman-bootstrap.sh" <engine>
   ```

   If bootstrap fails, stop and print the manual fallback from `/caveman-setup`.

2. Run the engine:

   ```bash
   source "${CLAUDE_PLUGIN_ROOT}/.venv/bin/activate"
   case "<engine>" in
       nlp) python "${CLAUDE_PLUGIN_ROOT}/caveman_compress_nlp.py" compress <args> ;;
       mlm) python "${CLAUDE_PLUGIN_ROOT}/caveman_compress_mlm.py" compress <args> ;;
       llm) python "${CLAUDE_PLUGIN_ROOT}/caveman_compress.py"     compress <args> ;;
   esac
   ```

   Pass `-f PATH` if `--file` was given; otherwise pass the text positionally (quoted).

3. Print the compressed output verbatim. If the engine printed token counts, surface them.

Notes:

- The `llm` engine needs `OPENAI_API_KEY` in env. If missing, instruct the user to set it.
- For non-English input with `nlp`, suggest `-l <lang>` (e.g. `-l es`).
