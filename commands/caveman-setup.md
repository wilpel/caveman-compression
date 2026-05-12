---
description: Bootstrap a Python venv for caveman-compression engines (nlp|mlm|llm)
argument-hint: "[nlp|mlm|llm]"
allowed-tools: Bash
---

Bootstrap the caveman-compression engine for engine `$ARGUMENTS` (default `nlp`).

Run the bootstrap script and report the result. The script is idempotent — re-running is safe.

```bash
"${CLAUDE_PLUGIN_ROOT}/bin/caveman-bootstrap.sh" ${ARGUMENTS:-nlp}
```

If the script fails:

1. Show the user the exact error.
2. Print the manual fallback:

   ```bash
   cd "${CLAUDE_PLUGIN_ROOT}"
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-nlp.txt   # or requirements-mlm.txt / requirements.txt
   python -m spacy download en_core_web_sm   # for nlp/mlm only
   ```

After success, tell the user they can now use `/caveman-compress`.
