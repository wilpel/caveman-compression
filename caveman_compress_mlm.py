#!/usr/bin/env python3
"""
MLM-based caveman compression using RoBERTa (English) and CamemBERT (French).
Removes highly predictable tokens based on masked language model probabilities.
No LLM API required - uses local models for deterministic compression.
"""

import sys
import os
import argparse
from pathlib import Path

from caveman_compress_nlp import ModelUnavailableError
from language_catalog import MLM_LANGUAGE_CATALOGUE, load_active_languages

try:
    import torch
    import torch.nn.functional as F
    from transformers import RobertaForMaskedLM, RobertaTokenizer, CamembertForMaskedLM, CamembertTokenizer
    import spacy
except ImportError as e:
    raise ImportError(
        "Required packages not installed. Install with:\n"
        "  pip install torch transformers spacy\n"
        "  python -m spacy download en_core_web_sm\n"
        "  python -m spacy download fr_core_news_sm"
    ) from e

# Model cache
_models = {}
_device = None
_nlp_models = {}
_fasttext_model = None

# Core input bound: reject oversized inputs with a clear error instead of
# silently truncating. Aligned with the HTTP/MCP MAX_INPUT_LENGTH.
MAX_INPUT_LENGTH = 4096
# Maximum token sequence accepted for a single MLM inference pass.
MAX_SEQ_LENGTH = 512


class InputTooLongError(ValueError):
    """Raised when the tokenized input exceeds MAX_SEQ_LENGTH for MLM inference."""

# SUPPORTED_LANGUAGES is derived from the checked-in-image manifest written at
# build time (see language_catalog.load_active_languages). The runtime never
# activates languages from mutable environment variables: LANGUAGES and
# CUSTOM_MLM_MODELS are build-time configuration only. When the manifest is
# absent (local development without a build), the fallback is the default
# en/fr pair resolved from the catalogue.
SUPPORTED_LANGUAGES = load_active_languages()

def get_fasttext_model():
    """Get or load the fastText language detection model.

    A corrupt or unloadable artifact is treated as unavailable and skipped, so
    language detection falls back to the word-list heuristic instead of raising
    or leaving a broken model cached.
    """
    global _fasttext_model
    if _fasttext_model is None:
        try:
            import fasttext
        except ImportError:
            print(
                "Warning: fasttext not installed, falling back to word-list heuristic",
                file=sys.stderr,
            )
            return None
        # Try to find the model in common locations
        model_paths = [
            "/app/models/lid.176.bin",
            os.path.expanduser("~/.cache/fasttext/lid.176.bin"),
            "lid.176.bin",
        ]
        for path in model_paths:
            if os.path.exists(path):
                try:
                    _fasttext_model = fasttext.load_model(path)
                    print(f"fastText model loaded from {path}", file=sys.stderr)
                    return _fasttext_model
                except Exception as e:
                    print(
                        f"Warning: fastText model at {path} failed to load: {e}",
                        file=sys.stderr,
                    )
        print(
            "Warning: fastText model not found, falling back to word-list heuristic",
            file=sys.stderr,
        )
    return _fasttext_model

def detect_language(text):
    """Detect language using fastText (primary) or a word-list heuristic (fallback).

    Returns a language code ("en"/"fr"/...), or None when no language can be
    determined. The word-list heuristic only classifies English or French, so
    text with no positive evidence for either returns None instead of being
    misclassified as English; callers then return the input unchanged.
    """
    # Try fastText first (faster and more accurate)
    ft_model = get_fasttext_model()
    if ft_model is not None:
        try:
            # fastText expects single line text
            clean_text = text[:1000].replace('\n', ' ')
            predictions = ft_model.predict(clean_text)
            lang_code = predictions[0][0].replace('__label__', '')
            confidence = predictions[1][0]
            if confidence > 0.3:  # Minimum confidence threshold
                return lang_code
        except Exception:
            pass

    # Simple heuristic fallback (en/fr only). With no positive English or
    # French evidence, return None so the caller leaves the text unchanged
    # rather than guessing English.
    french_words = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "sont", "avoir", "être", "faire"}
    english_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does"}

    words = set(text.lower().split())
    french_count = len(words & french_words)
    english_count = len(words & english_words)

    if french_count == 0 and english_count == 0:
        return None
    return "fr" if french_count > english_count else "en"

def get_nlp_model(lang_code):
    """Get or load the spaCy model configured for the language."""
    if lang_code not in _nlp_models:
        config = SUPPORTED_LANGUAGES.get(lang_code)
        if config is None:
            raise ModelUnavailableError(
                f"No NLP model configured for unsupported language '{lang_code}'"
            )
        model_name = config.get("spacy")
        if not model_name:
            raise ModelUnavailableError(
                f"No spaCy model configured for language '{lang_code}'"
            )
        try:
            _nlp_models[lang_code] = spacy.load(model_name)
        except OSError as e:
            raise ModelUnavailableError(
                f"spaCy model '{model_name}' not available for language '{lang_code}': {e}"
            ) from e
    return _nlp_models[lang_code]

def get_mlm_model(lang_code):
    """Get or load the MLM model configured for the language.

    Raises ModelUnavailableError (an OSError subclass) when the language has no
    configured model or the model files cannot be found, so callers can fall
    back to NLP. Unexpected runtime errors propagate unchanged.
    """
    if lang_code not in _models:
        config = SUPPORTED_LANGUAGES.get(lang_code)
        if config is None or not config.get("model"):
            raise ModelUnavailableError(
                f"No MLM model configured for unsupported language '{lang_code}'"
            )
        model_name = config["model"]

        print(f"Loading MLM model '{model_name}' for language '{lang_code}'...", file=sys.stderr)

        # Use Auto classes for generic model loading. local_files_only=True
        # enforces offline loading: the runtime must use the models provisioned
        # into the image at build time and never reach the HuggingFace hub.
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)
        except OSError as e:
            raise ModelUnavailableError(
                f"MLM model '{model_name}' not available for language '{lang_code}': {e}"
            ) from e

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        _models[lang_code] = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
        }
        print(f"Model '{model_name}' loaded.", file=sys.stderr)

    return _models[lang_code]

def get_mlm_probability(lang_code, sentence, word_idx):
    """Get MLM probability for a specific whitespace-split word in a sentence.

    Raises InputTooLongError if the masked sentence exceeds MAX_SEQ_LENGTH tokens.
    """
    model_data = get_mlm_model(lang_code)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    device = model_data["device"]
    
    words = sentence.split()
    if word_idx >= len(words):
        return 0.0

    target_word = words[word_idx]
    masked_words = words.copy()
    masked_words[word_idx] = tokenizer.mask_token
    masked_sentence = " ".join(masked_words)

    inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
    if inputs.input_ids.size(1) > MAX_SEQ_LENGTH:
        raise InputTooLongError(
            f"Input sentence too long for MLM inference "
            f"({inputs.input_ids.size(1)} tokens > {MAX_SEQ_LENGTH} max)"
        )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    if len(mask_token_index) == 0:
        return 0.0

    mask_token_logits = logits[0, mask_token_index[0], :]
    probs = F.softmax(mask_token_logits, dim=0)

    target_tokens = tokenizer.encode((" " if word_idx > 0 else "") + target_word, add_special_tokens=False)
    if len(target_tokens) == 0:
        return 0.0

    target_token_id = target_tokens[0]
    prob = float(probs[target_token_id].item())

    return prob

def compress_text(text, prob_threshold=1e-5, no_adjacent_removal=False, protect_ner=True, *, language=None, drop_ratio=None, preset="lite", mode="sentence", calibrate_from_nlp=True):
    """
    Apply MLM-based compression by removing words whose predictability exceeds threshold.

    Languages not activated in the build-time manifest (default "en,fr") are
    returned unchanged: catalogued-but-inactive languages (de/es) and unknown
    codes pass through without touching the MLM/NLP models and without erroring.

    Args:
        text: Input text to compress
        prob_threshold: Minimum MLM probability for a removal candidate. Without
            adaptive calibration, every candidate at or above this value is
            removed; adaptive modes use it to filter the drop-ratio candidates.
        language: Language code (auto-detect if None)
        drop_ratio: Fraction of words to drop (0.0-1.0). If set, uses adaptive threshold.
        preset: Preset name (lite/full/ultra) used with calibrate_from_nlp
        mode: "sentence" (NLP/MLM per sentence) or "text" (NLP/MLM on full text)
        calibrate_from_nlp: If True, calibrate drop_ratio from NLP compression
        no_adjacent_removal: If True, don't remove adjacent words
        protect_ner: If True, don't remove named entities

    Raises:
        ValueError: If text exceeds MAX_INPUT_LENGTH, or text mode is requested
            for Chinese (zh).
        InputTooLongError: If the tokenized input exceeds MAX_SEQ_LENGTH for a
            single MLM inference pass (text-mode full-text preflight, or a
            per-sentence input in get_mlm_probability).
        ModelUnavailableError: If an activated language's model cannot be loaded.
    """
    if not text or not text.strip():
        return text

    # Auto-detect language if not specified
    if language is None:
        language = detect_language(text)

    # Inactive or undetermined language: return the input unchanged. Only
    # languages activated in the build-time manifest (default "en,fr") are
    # compressed. Catalogued-but-inactive languages (de/es), unknown codes, and
    # a None result from auto-detection pass through without touching the
    # MLM/NLP models and without erroring.
    if language not in SUPPORTED_LANGUAGES:
        return text

    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input text too long: {len(text)} chars exceeds limit of {MAX_INPUT_LENGTH}"
        )

    # Chinese text mode cannot split words on whitespace, so it would return
    # the input unchanged. Reject it explicitly instead.
    if mode == "text" and language == "zh":
        raise ValueError(
            "text mode is not supported for Chinese (zh); use mode='sentence'"
        )

    # Text mode feeds the full text into MLM inference per word, so verify the
    # tokenized input fits the model's sequence bound before compressing. No
    # truncation is applied. Sentence mode skips this preflight: each sentence
    # is bound-checked individually in get_mlm_probability, so a long
    # multi-sentence input is fine as long as each sentence fits.
    if mode == "text":
        model_data = get_mlm_model(language)
        tokenizer = model_data["tokenizer"]
        n_tokens = len(tokenizer.encode(text, truncation=False))
        if n_tokens > MAX_SEQ_LENGTH:
            raise InputTooLongError(
                f"Input text too long for MLM inference: {n_tokens} tokens exceeds "
                f"MAX_SEQ_LENGTH of {MAX_SEQ_LENGTH}"
            )

    # Get NLP model for tokenization and NER
    nlp = get_nlp_model(language)
    doc = nlp(text)
    
    # Calculate drop_ratio from NLP if needed. Sentence mode performs this
    # calibration independently for each sentence in _compress_sentence_mode.
    if mode == "text" and drop_ratio is None and calibrate_from_nlp:
        import math
        try:
            from caveman_compress_nlp import compress_text as compress_text_nlp
            
            if mode == "text":
                # NLP on full text
                nlp_result = compress_text_nlp(text, lang=language)
                nlp_kept_ratio = len(nlp_result) / len(text) if text else 1.0
                nlp_drop = 1.0 - nlp_kept_ratio
                nlp_drop = round(nlp_drop * 10) / 10
                
                if preset == "lite":
                    drop_ratio = nlp_drop
                elif preset == "full":
                    drop_ratio = min(0.9, math.ceil(nlp_drop * 1.5 * 10) / 10)
                elif preset == "ultra":
                    drop_ratio = min(0.9, math.ceil(nlp_drop * 2 * 10) / 10)
                else:
                    drop_ratio = nlp_drop
        except Exception:
            drop_ratio = 0.3  # fallback
    
    # Process based on mode
    if mode == "text":
        # Text mode: NLP on full text, MLM on full text context
        return _compress_text_mode(
            text, doc, language, drop_ratio, no_adjacent_removal, protect_ner,
            prob_threshold=prob_threshold,
        )
    else:
        # Sentence mode: NLP per sentence, MLM per sentence context (default)
        return _compress_sentence_mode(
            text, doc, language, preset, drop_ratio, calibrate_from_nlp,
            no_adjacent_removal, protect_ner, prob_threshold=prob_threshold,
        )

def _word_char_offsets(text, words):
    """Return (start, end) character offsets for whitespace-split words in text."""
    offsets = []
    pos = 0
    for word in words:
        start = text.find(word, pos)
        if start == -1:
            start = pos
        offsets.append((start, start + len(word)))
        pos = start + len(word)
    return offsets


def _ner_covered_words(word_offsets, ner_ranges):
    """Return word indices whose character range overlaps any (start, end) range."""
    protected = set()
    for start, end in ner_ranges:
        for i, (word_start, word_end) in enumerate(word_offsets):
            if word_start < end and word_end > start:
                protected.add(i)
    return protected


def _compress_text_mode(text, doc, language, drop_ratio, no_adjacent_removal, protect_ner, prob_threshold=1e-5):
    """Text mode: compress using full text context"""
    words = text.split()
    if len(words) < 3:
        return text

    # NER protection: mark whitespace-split words covered by a named entity.
    ner_protected = set()
    if protect_ner:
        word_offsets = _word_char_offsets(text, words)
        ner_ranges = [(ent.start_char, ent.end_char) for ent in doc.ents]
        ner_protected = _ner_covered_words(word_offsets, ner_ranges)

    # Calculate probabilities for all words
    word_probs = []
    for i, word in enumerate(words):
        if len(word) <= 2 or not word.isalpha():
            word_probs.append((i, 0.0))
        elif i in ner_protected:
            word_probs.append((i, 0.0))
        else:
            prob = get_mlm_probability(language, text, i)
            word_probs.append((i, prob))

    if drop_ratio is None:
        # Legacy threshold mode: remove every eligible candidate at or above
        # the threshold.
        sortable_probs = [
            (i, p) for i, p in word_probs
            if p > 0 and p >= prob_threshold
        ]
        to_remove = {i for i, _ in sortable_probs}
    else:
        # Adaptive mode: remove the most predictable words up to drop_ratio.
        num_to_drop = int(len(words) * drop_ratio)
        sortable_probs = [
            (i, p) for i, p in word_probs
            if p > 0 and p >= prob_threshold
        ]
        sortable_probs.sort(key=lambda x: x[1], reverse=True)

        to_remove = set()
        for i, p in sortable_probs:
            if len(to_remove) >= num_to_drop:
                break
            to_remove.add(i)

    # Apply no_adjacent_removal constraint
    if no_adjacent_removal:
        filtered_remove = set()
        prev_removed = False
        for i in range(len(words)):
            if i in to_remove:
                if not prev_removed:
                    filtered_remove.add(i)
                    prev_removed = True
            else:
                prev_removed = False
        to_remove = filtered_remove

    # Reconstruct
    compressed_words = [w for i, w in enumerate(words) if i not in to_remove]
    return " ".join(compressed_words)

def _compress_sentence_mode(text, doc, language, preset, drop_ratio, calibrate_from_nlp, no_adjacent_removal, protect_ner, prob_threshold=1e-5):
    """Sentence mode: compress per sentence (default)"""
    import math
    from caveman_compress_nlp import compress_text as compress_text_nlp
    
    result_parts = []
    
    for sent in doc.sents:
        raw_sent = sent.text
        leading = len(raw_sent) - len(raw_sent.lstrip())
        sent_text = raw_sent.strip()
        if not sent_text:
            continue
        
        words = sent_text.split()
        if len(words) < 3:
            result_parts.append(sent_text)
            continue
        
        # Only calibrate per-sentence drop_ratio from NLP when no explicit
        # drop_ratio was provided AND calibration is enabled. Otherwise use
        # the caller-supplied value unchanged.
        if drop_ratio is not None:
            sent_drop_ratio = drop_ratio
        elif not calibrate_from_nlp:
            sent_drop_ratio = None
        else:
            sent_drop_ratio = drop_ratio
            try:
                nlp_result = compress_text_nlp(sent_text, lang=language)
                nlp_kept_ratio = len(nlp_result) / len(sent_text) if sent_text else 1.0
                nlp_drop = 1.0 - nlp_kept_ratio
                nlp_drop = round(nlp_drop * 10) / 10

                if preset == "lite":
                    sent_drop_ratio = nlp_drop
                elif preset == "full":
                    sent_drop_ratio = min(0.9, math.ceil(nlp_drop * 1.5 * 10) / 10)
                elif preset == "ultra":
                    sent_drop_ratio = min(0.9, math.ceil(nlp_drop * 2 * 10) / 10)
                else:
                    sent_drop_ratio = nlp_drop
            except Exception:
                sent_drop_ratio = 0.5  # safe fallback
        
        # NER protection: mark whitespace-split words covered by a named entity.
        ner_protected = set()
        if protect_ner:
            sent_start_char = sent.start_char + leading
            word_offsets = _word_char_offsets(sent_text, words)
            ner_ranges = [
                (ent.start_char - sent_start_char, ent.end_char - sent_start_char)
                for ent in doc.ents
            ]
            ner_protected = _ner_covered_words(word_offsets, ner_ranges)

        # Calculate probabilities
        word_probs = []
        for i, word in enumerate(words):
            if len(word) <= 2 or not word.isalpha():
                word_probs.append((i, 0.0))
            elif i in ner_protected:
                word_probs.append((i, 0.0))
            else:
                prob = get_mlm_probability(language, sent_text, i)
                word_probs.append((i, prob))
        
        if sent_drop_ratio is None:
            # Legacy threshold mode: remove every eligible candidate at or
            # above the threshold.
            sortable_probs = [
                (i, p) for i, p in word_probs
                if p > 0 and p >= prob_threshold
            ]
            to_remove = {i for i, _ in sortable_probs}
        else:
            # Adaptive mode: remove the most predictable words up to the
            # calibrated or explicitly requested drop ratio.
            num_to_drop = int(len(words) * sent_drop_ratio)
            sortable_probs = [
                (i, p) for i, p in word_probs
                if p > 0 and p >= prob_threshold
            ]
            sortable_probs.sort(key=lambda x: x[1], reverse=True)

            to_remove = set()
            for i, p in sortable_probs:
                if len(to_remove) >= num_to_drop:
                    break
                to_remove.add(i)
        
        # Apply no_adjacent_removal
        if no_adjacent_removal:
            filtered_remove = set()
            prev_removed = False
            for i in range(len(words)):
                if i in to_remove:
                    if not prev_removed:
                        filtered_remove.add(i)
                        prev_removed = True
                else:
                    prev_removed = False
            to_remove = filtered_remove
        
        # Reconstruct
        compressed_words = [w for i, w in enumerate(words) if i not in to_remove]
        result_parts.append(" ".join(compressed_words))
    
    return " ".join(result_parts)

def count_tokens(text):
    """Estimate token count: characters / 4"""
    return len(text.strip()) // 4

def main():
    parser = argparse.ArgumentParser(description="MLM-based caveman compression")
    parser.add_argument("action", choices=["compress"], help="Action to perform")
    parser.add_argument("text", nargs="?", help="Text to compress")
    parser.add_argument("-f", "--file", help="Input file")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-l", "--language", help="Language code (en, fr)")
    parser.add_argument("--no-adjacent", action="store_true", help="Don't remove adjacent words")
    parser.add_argument("--no-ner", action="store_true", help="Don't protect named entities")
    
    args = parser.parse_args()
    
    # Get input text
    if args.file:
        with open(args.file, "r") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Error: Provide text or use -f for file", file=sys.stderr)
        sys.exit(1)
    
    if args.action == "compress":
        compressed = compress_text(
            text,
            language=args.language,
            no_adjacent_removal=args.no_adjacent,
            protect_ner=not args.no_ner
        )
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(compressed)
            print(f"Compressed text written to {args.output}", file=sys.stderr)
        else:
            print(compressed)

if __name__ == "__main__":
    main()
