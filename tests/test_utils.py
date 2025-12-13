# tests/test_utils.py
import pytest
from caveman_compress import count_tokens, is_text_content, split_sentences
from tests.fixtures.test_texts import ALL_TEXTS

# --- Tests for count_tokens (4 tests) ---

def test_count_tokens_simple():
    """Test token count for simple text."""
    text = "This is four tokens."
    assert count_tokens(text) == 5  # 20 chars / 4 = 5

def test_count_tokens_empty():
    """Test token count for empty string."""
    assert count_tokens("") == 0

def test_count_tokens_whitespace():
    """Test token count for string with only whitespace."""
    assert count_tokens("   \t\n   ") == 0

def test_count_tokens_long_text():
    """Test token count for a longer piece of text."""
    text = ALL_TEXTS["long_paragraph"]
    expected_tokens = len(text) // 4
    assert count_tokens(text) == expected_tokens

# --- Tests for is_text_content (6 tests) ---

def test_is_text_content_natural_language():
    """Test that natural language is correctly identified."""
    assert is_text_content(ALL_TEXTS["simple"]) is True
    assert is_text_content(ALL_TEXTS["long_paragraph"]) is True

def test_is_text_content_code_block():
    """Test that a block of code is correctly identified as not text."""
    assert is_text_content(ALL_TEXTS["code"]) is False

def test_is_text_content_short_code_snippet_is_text():
    """Test that a very short code snippet is treated as text (by design)."""
    assert is_text_content(ALL_TEXTS["short_code"]) is True

def test_is_text_content_with_some_code_symbols():
    """Test text with some symbols that might appear in code. Should be detected as code."""
    text = "This is text with => and -> arrows."
    assert is_text_content(text) is False

def test_is_text_content_very_short_text():
    """Test that very short text is always considered text content."""
    assert is_text_content("Hi.") is True
    assert is_text_content("x=1") is True

def test_is_text_content_with_many_braces():
    """Test that text with a high density of braces is considered code."""
    text = "{'key': 'value', 'a': {'b': [1, 2]}} and some other text"
    assert is_text_content(text) is False

# --- Tests for split_sentences (6 tests) ---

def test_split_sentences_simple():
    """Test splitting simple sentences."""
    text = "First sentence. Second sentence."
    expected = ["First sentence.", "Second sentence."]
    assert split_sentences(text) == expected

def test_split_sentences_no_period():
    """Test splitting text with no periods."""
    text = "This is a single block of text"
    assert split_sentences(text) == [text]

def test_split_sentences_with_abbreviations_known_limitation():
    """Test splitting with abbreviations, documenting the known limitation."""
    text = "Dr. Smith went to the U.S.A. It was sunny."
    # The current simple regex will split "Dr." and "U.S.A."
    expected = ["Dr.", "Smith went to the U.S.A.", "It was sunny."]
    assert split_sentences(text) == expected

def test_split_sentences_with_newlines():
    """Test splitting sentences separated by newlines."""
    text = "First sentence.\nSecond sentence. \n\nThird sentence."
    expected = ["First sentence.", "Second sentence.", "Third sentence."]
    assert split_sentences(text) == expected

def test_split_sentences_ending_without_space():
    """Test splitting when text ends with a period but no trailing space."""
    text = "This is a test."
    assert split_sentences(text) == ["This is a test."]

def test_split_sentences_empty_string():
    """Test splitting an empty string."""
    assert split_sentences("") == [""]
