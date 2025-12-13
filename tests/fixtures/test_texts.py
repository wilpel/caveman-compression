# tests/fixtures/test_texts.py

# Basic text for simple compression and processing
SIMPLE_TEXT = "This is a simple sentence for testing purposes. It has two sentences."

# Text with common abbreviations that can challenge sentence splitters
ABBREVIATION_TEXT = "Dr. Smith lives in the U.S.A. He works for Acme Inc. and his office is on Main St. The price is approx. $9.99."

# Text containing numbers and decimals
NUMBERS_AND_DECIMALS = "The item costs 9.99 dollars. The model number is 123.45.67. There are 5,000 units."

# Text with a URL
URL_TEXT = "Please visit our website at https://www.example.com for more information. You can also check http://sub.domain.org/path?query=value."

# Text with ellipses and other punctuation
PUNCTUATION_TEXT = "What is this...? I don't know! Is it good? Maybe... we should see."

# Text containing a code block
CODE_BLOCK_TEXT = """
Here is some text.
And here is a code block:

```python
def hello_world():
    print("Hello, world!")
```

This is the end of the text.
"""

# Very short code snippet that might be mistaken for text
SHORT_CODE_SNIPPET = "x = y + 1"

# Text that is just a single, long paragraph
LONG_PARAGRAPH = "This is a very long paragraph designed to test the wrapping and splitting capabilities of the text processing utilities. It contains multiple clauses, phrases, and words, all strung together without any line breaks, to simulate a dense block of text that might be found in academic papers or legal documents. The purpose is to ensure that tokenizers, sentence splitters, and other NLP tools can handle such structures gracefully without errors."

# Text for testing stop word removal
STOP_WORDS_TEXT = "This is a test of the stop word removal. We are trying to see if it works as expected."

# Text with named entities
NAMED_ENTITY_TEXT = "John Smith traveled to Paris from New York on a British Airways flight. He met with representatives from Google."

# Empty and whitespace-only text for input validation tests
EMPTY_TEXT = ""
WHITESPACE_TEXT = "   \t\n   "

# Text with non-ASCII characters for UTF-8 tests
UTF8_TEXT = "Résumé in café. Это тест на русском. こんにちは世界。"

# A dictionary to easily access all test texts
ALL_TEXTS = {
    "simple": SIMPLE_TEXT,
    "abbreviations": ABBREVIATION_TEXT,
    "numbers": NUMBERS_AND_DECIMALS,
    "url": URL_TEXT,
    "punctuation": PUNCTUATION_TEXT,
    "code": CODE_BLOCK_TEXT,
    "short_code": SHORT_CODE_SNIPPET,
    "long_paragraph": LONG_PARAGRAPH,
    "stop_words": STOP_WORDS_TEXT,
    "entities": NAMED_ENTITY_TEXT,
    "empty": EMPTY_TEXT,
    "whitespace": WHITESPACE_TEXT,
    "utf8": UTF8_TEXT,
}
