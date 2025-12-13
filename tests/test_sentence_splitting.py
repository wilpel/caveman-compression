# tests/test_sentence_splitting.py
import pytest
from caveman_compress import split_sentences
from tests.fixtures.test_texts import ALL_TEXTS

# --- Comprehensive tests for split_sentences (15 tests) ---

def test_split_basic_sentences():
    """Test with two simple sentences."""
    text = "This is the first sentence. This is the second one."
    assert split_sentences(text) == ["This is the first sentence.", "This is the second one."]

def test_split_no_period():
    """Test text without any periods."""
    text = "A long line of text without any terminating punctuation"
    assert split_sentences(text) == [text]

def test_split_multiple_spaces():
    """Test splitting with multiple spaces after a period."""
    text = "A sentence.   Another sentence."
    assert split_sentences(text) == ["A sentence.", "Another sentence."]

def test_split_newlines_and_tabs():
    """Test splitting with newlines and tabs after a period."""
    text = "First line.\nSecond line.\tThird line."
    assert split_sentences(text) == ["First line.", "Second line.", "Third line."]

def test_split_dr_abbreviation_limitation():
    """Test known limitation with 'Dr.' abbreviation."""
    text = "Dr. Smith visited the patient. The patient was fine."
    assert split_sentences(text) == ["Dr.", "Smith visited the patient.", "The patient was fine."]

def test_split_mr_abbreviation_limitation():
    """Test known limitation with 'Mr.' abbreviation."""
    text = "Mr. Jones was happy. He smiled."
    assert split_sentences(text) == ["Mr.", "Jones was happy.", "He smiled."]

def test_split_usa_abbreviation_limitation():
    """Test known limitation with 'U.S.A.' abbreviation."""
    text = "He went to the U.S.A. It was a long trip."
    assert split_sentences(text) == ["He went to the U.S.A.", "It was a long trip."]

def test_split_company_inc_limitation():
    """Test known limitation with 'Inc.' suffix."""
    text = "He works at Acme Inc. The company is large."
    assert split_sentences(text) == ["He works at Acme Inc.", "The company is large."]

def test_split_eg_abbreviation_limitation():
    """Test known limitation with 'e.g.' abbreviation."""
    text = "Use a tool, e.g. a hammer. It works well."
    assert split_sentences(text) == ["Use a tool, e.g.", "a hammer.", "It works well."]

def test_split_ellipsis_limitation():
    """Test how it handles ellipsis (not well)."""
    text = "I wonder... what will happen. This is a test."
    # The current implementation doesn't handle ellipsis.
    assert split_sentences(text) == ["I wonder...", "what will happen.", "This is a test."]

def test_split_question_mark_and_exclamation_no_split():
    """Test that it does not split on ? or !"""
    text = "Is this a test? Yes it is! This is another sentence."
    assert split_sentences(text) == ["Is this a test? Yes it is! This is another sentence."]

def test_split_url_limitation():
    """Test that a URL is not correctly handled."""
    text = "Visit example.com. It is a good site."
    assert split_sentences(text) == ["Visit example.com.", "It is a good site."]

def test_split_decimal_number_limitation():
    """Test limitation with decimal numbers."""
    text = "The price is $9.99. It is a good price."
    assert split_sentences(text) == ["The price is $9.99.", "It is a good price."]

def test_split_sentence_ending_with_number():
    """Test sentence ending with a number before the period."""
    text = "The year is 2023. The next year is 2024."
    assert split_sentences(text) == ["The year is 2023.", "The next year is 2024."]

def test_split_empty_and_whitespace_string():
    """Test with empty and whitespace-only strings."""
    assert split_sentences("") == [""]
    assert split_sentences("   \t ") == ["   \t "]
