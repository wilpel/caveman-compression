# Examples

All examples validated using `caveman_compress.py` with GPT-4o.

## Files

- `resume-normal.txt` / `resume-caveman.txt` - Professional resume (201→152 tokens, 24%)
- `system-prompt-normal.txt` / `system-prompt-caveman.txt` - AI assistant prompt (171→74 tokens, 57%)
- `api-documentation-normal.txt` / `api-documentation-caveman.txt` - API auth docs (137→82 tokens, 40%)

## Usage

```bash
# Compress an example
python ../caveman_compress.py compress -f resume-normal.txt

# Compare with validated output
cat resume-caveman.txt
```
