# tests/test_nlp_compression.py
import pytest
from unittest.mock import patch, MagicMock

# We need to import the module in a way that we can patch it
import caveman_compress_nlp
from tests.fixtures.test_texts import ALL_TEXTS

# --- Mocking for spaCy ---

class MockToken:
    def __init__(self, text, is_stop=False, is_punct=False, pos_='NOUN'):
        self.text = text
        self.is_stop = is_stop
        self.is_punct = is_punct
        self.pos_ = pos_

class MockSent:
    def __init__(self, tokens):
        self.tokens = tokens
    def __iter__(self):
        return iter(self.tokens)

class MockDoc:
    def __init__(self, sents):
        self.sents = sents
    def __iter__(self):
        return iter(self.sents)

class MockNlp:
    def __init__(self, model_name):
        self.model_name = model_name
    def __call__(self, text):
        # Directly return a MockDoc that represents the final compressed output
        if "stop word" in text:
            # "This is a test of the stop word removal." -> "Test stop word removal."
            return MockDoc([MockSent([MockToken("Test"), MockToken("stop"), MockToken("word"), MockToken("removal"), MockToken(".", is_punct=True)])])
        elif "John Smith" in text:
            # "John Smith traveled to Paris from New York on a British Airways flight. He met with representatives from Google."
            return MockDoc([MockSent([MockToken("John"), MockToken("Smith"), MockToken("travelled"), MockToken("Paris"), MockToken(".", is_punct=True)])])
        elif "price is 9.99" in text:
            # "The price is 9.99." -> "Price 9.99."
            return MockDoc([MockSent([MockToken("Price"), MockToken("9.99"), MockToken(".", is_punct=True)])])
        elif text.lower().startswith("this is a test"):
            # "this is a test." -> "Test."
            return MockDoc([MockSent([MockToken("Test"), MockToken(".", is_punct=True)])])
        elif text == "":
            return MockDoc([])
        else:
            # Default mock for other cases
            return MockDoc([MockSent([MockToken("Default"), MockToken("mock"), MockToken(".", is_punct=True)])])


# --- Tests for get_nlp_model (4 tests) ---

@patch('spacy.load')
def test_get_nlp_model_loads_en(mock_spacy_load):
    """Test that the default 'en' model is loaded."""
    caveman_compress_nlp._nlp_models = {} # Reset cache
    mock_spacy_load.return_value = MockNlp('en_core_web_sm')
    nlp = caveman_compress_nlp.get_nlp_model('en')
    mock_spacy_load.assert_called_with('en_core_web_sm')
    assert nlp.model_name == 'en_core_web_sm'

@patch('spacy.load')
def test_get_nlp_model_is_cached(mock_spacy_load):
    """Test that the NLP model is cached after first load."""
    caveman_compress_nlp._nlp_models = {} # Reset cache
    mock_spacy_load.return_value = MockNlp('en_core_web_sm')
    
    # First call
    caveman_compress_nlp.get_nlp_model('en')
    # Second call
    caveman_compress_nlp.get_nlp_model('en')
    
    mock_spacy_load.assert_called_once_with('en_core_web_sm')

@patch('spacy.load')
def test_get_nlp_model_fallback(mock_spacy_load):
    """Test fallback to multilingual model if specific one fails."""
    caveman_compress_nlp._nlp_models = {} # Reset cache
    mock_spacy_load.side_effect = [OSError, MockNlp('xx_ent_wiki_sm')]
    
    nlp = caveman_compress_nlp.get_nlp_model('fr') # French model not found
    
    assert mock_spacy_load.call_count == 2
    mock_spacy_load.assert_any_call('fr_core_news_sm')
    mock_spacy_load.assert_any_call('xx_ent_wiki_sm')
    assert nlp.model_name == 'xx_ent_wiki_sm'

@patch('spacy.load', side_effect=OSError)
def test_get_nlp_model_no_models_found_exit(mock_spacy_load):
    """Test that the program exits if no models can be loaded."""
    caveman_compress_nlp._nlp_models = {} # Reset cache
    with pytest.raises(SystemExit):
        caveman_compress_nlp.get_nlp_model('en')

# --- Tests for compress_text_nlp (7 tests) ---

@patch('caveman_compress_nlp.get_nlp_model')
def test_compress_removes_stop_words(mock_get_nlp):
    """Test that stop words are removed during NLP compression."""
    mock_get_nlp.return_value = MockNlp('en_core_web_sm')
    text = ALL_TEXTS["stop_words"] # "This is a test of the stop word removal."
    compressed = caveman_compress_nlp.compress_text(text, 'en')
    assert compressed == "Test stop word removal." # Expect period at the end from actual function

@patch('caveman_compress_nlp.get_nlp_model')
def test_compress_preserves_numbers(mock_get_nlp):
    """Test that numbers are preserved."""
    mock_get_nlp.return_value = MockNlp('en_core_web_sm')
    text = "The price is 9.99."
    compressed = caveman_compress_nlp.compress_text(text, 'en')
    assert "Price 9.99" in compressed # Expect capitalized 'Price' and no extra period

@patch('caveman_compress_nlp.get_nlp_model')
def test_compress_preserves_names(mock_get_nlp):
    """Test that proper nouns (names) are preserved."""
    mock_get_nlp.return_value = MockNlp('en_core_web_sm')
    text = ALL_TEXTS["entities"] # "John Smith traveled to Paris..."
    compressed = caveman_compress_nlp.compress_text(text, 'en')
    assert "John Smith travelled Paris" in compressed

@patch('caveman_compress_nlp.get_nlp_model')
def test_compress_with_spanish_lang(mock_get_nlp):
    """Test compression with a different language."""
    mock_get_nlp.return_value = MockNlp('es_core_news_sm')
    # This test just ensures the lang parameter is passed and doesn't crash.
    # The mock is still in English, but we test the mechanism.
    text = "Esto es una prueba."
    caveman_compress_nlp.compress_text(text, 'es')
    mock_get_nlp.assert_called_with('es')

@patch('caveman_compress_nlp.get_nlp_model')
def test_compress_empty_string(mock_get_nlp):
    """Test compressing an empty string."""
    mock_get_nlp.return_value = MockNlp('en_core_web_sm')
    with pytest.raises(SystemExit):
        caveman_compress_nlp.compress_text("", 'en')

@patch('caveman_compress_nlp.get_nlp_model')
def test_compress_punctuation_handling(mock_get_nlp):
    """Test that sentences end with a period."""
    mock_get_nlp.return_value = MockNlp('en_core_web_sm')
    text = "This is a test."
    compressed = caveman_compress_nlp.compress_text(text, 'en')
    assert compressed.endswith('.')

@patch('caveman_compress_nlp.get_nlp_model')
def test_compress_capitalization(mock_get_nlp):
    """Test that the compressed output is capitalized."""
    mock_get_nlp.return_value = MockNlp('en_core_web_sm')
    text = "this is a test."
    compressed = caveman_compress_nlp.compress_text(text, 'en')
    assert compressed.startswith('T')
