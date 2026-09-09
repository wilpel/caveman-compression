"""Focused regression tests for caveman_compress_mlm fixes.

Covers:
- ModelUnavailableError contract (OSError subclass, shared between modules).
- Frozen runtime config: active languages come from the build-time manifest,
  never from mutable LANGUAGES / CUSTOM_MLM_MODELS environment variables.
- Unsupported language no longer silently mapped to English.
- Inactive-language passthrough: catalogued-but-inactive (de/es) and unknown
  languages return the input unchanged without touching models or erroring.
- Punctuation-before-NER char-offset protection (index alignment).
- Core input-length bound (ValueError, no silent truncation).
- Per-sentence MLM inference over MAX_SEQ_LENGTH raises InputTooLongError.
- Offline model loading (local_files_only=True).
- Unknown-language heuristic: no en/fr evidence returns None -> unchanged text.
"""

import os
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The server/MCP boundary tests stub the backend modules in sys.modules with
# bare ModuleType objects. Drop such stubs so this module exercises the real
# backend implementations instead of the stubs.
for _name in ("caveman_compress_mlm", "caveman_compress_nlp"):
    _mod = sys.modules.get(_name)
    if _mod is not None and getattr(_mod, "__file__", None) is None:
        del sys.modules[_name]

import spacy  # noqa: E402

import caveman_compress_mlm as mlm  # noqa: E402
import language_catalog  # noqa: E402
from caveman_compress_nlp import ModelUnavailableError  # noqa: E402

HAS_EN_MODEL = spacy.util.is_package("en_core_web_sm")


class TestModelUnavailableError(unittest.TestCase):
    def test_subclasses_oserror(self):
        self.assertTrue(issubclass(ModelUnavailableError, OSError))

    def test_shared_between_modules(self):
        self.assertIs(mlm.ModelUnavailableError, ModelUnavailableError)


class TestUnsupportedLanguage(unittest.TestCase):
    def test_get_nlp_model_raises(self):
        with self.assertRaises(ModelUnavailableError):
            mlm.get_nlp_model("xx_unsupported")

    def test_get_mlm_model_raises(self):
        with self.assertRaises(ModelUnavailableError):
            mlm.get_mlm_model("xx_unsupported")


class TestInputBound(unittest.TestCase):
    def test_too_long_raises_valueerror(self):
        text = "word " * (mlm.MAX_INPUT_LENGTH + 1)
        with self.assertRaises(ValueError):
            mlm.compress_text(text, language="en")

    def test_empty_text_passthrough(self):
        self.assertEqual(mlm.compress_text(""), "")

    def test_active_language_over_length_raises_valueerror(self):
        # For an active language the length bound is enforced after language
        # resolution (inactive languages return unchanged before this check).
        text = "word " * (mlm.MAX_INPUT_LENGTH + 1)
        with self.assertRaises(ValueError):
            mlm.compress_text(text, language="fr")


@unittest.skipUnless(HAS_EN_MODEL, "en_core_web_sm not installed")
class TestNerProtectionWithPunctuation(unittest.TestCase):
    def _doc(self, text):
        return spacy.load("en_core_web_sm")(text)

    def test_punctuation_before_entity_aligns_char_offsets(self):
        text = "Hello, John Smith lives in Paris."
        doc = self._doc(text)
        sent = next(doc.sents)
        raw = sent.text
        leading = len(raw) - len(raw.lstrip())
        sent_text = raw.strip()
        words = sent_text.split()
        # "John" is whitespace word index 1 despite the preceding comma token.
        self.assertEqual(words, ["Hello,", "John", "Smith", "lives", "in", "Paris."])

        sent_start_char = sent.start_char + leading
        word_offsets = mlm._word_char_offsets(sent_text, words)
        ner_ranges = [
            (ent.start_char - sent_start_char, ent.end_char - sent_start_char)
            for ent in doc.ents
        ]
        protected = mlm._ner_covered_words(word_offsets, ner_ranges)
        self.assertIn(1, protected)  # "John"
        self.assertIn(2, protected)  # "Smith"
        self.assertIn(5, protected)  # "Paris."

    def test_compress_text_mode_keeps_entities_and_no_nameerror(self):
        text = "Hello, John Smith lives in Paris. He likes New York."
        doc = self._doc(text)

        original = mlm.get_mlm_probability
        mlm.get_mlm_probability = lambda lang, sentence, word_idx: 0.9
        try:
            out = mlm._compress_text_mode(
                text, doc, "en", drop_ratio=0.5,
                no_adjacent_removal=False, protect_ner=True,
            )
        finally:
            mlm.get_mlm_probability = original

        # Named entities must survive even at a high drop ratio.
        for word in ("John", "Smith", "Paris.", "New", "York."):
            self.assertIn(word, out)


class _FakeTokenizer:
    """Returns a fixed token count regardless of input, for token-bound tests."""

    def __init__(self, n_tokens):
        self.n_tokens = n_tokens

    def encode(self, text, truncation=False):
        return list(range(self.n_tokens))


class _FakeMaskTensor:
    """A minimal tensor stand-in whose size(dim=1) is a fixed token count."""

    def __init__(self, n_tokens):
        self.n_tokens = n_tokens

    def size(self, dim=None):
        return self.n_tokens


class _FakeMaskedInputs(dict):
    """Tokenizer output stand-in supporting .to(device), **unpacking and .input_ids.size()."""

    def __init__(self, n_tokens):
        super().__init__()
        self.input_ids = _FakeMaskTensor(n_tokens)

    def to(self, device):
        return self


class _FakeMaskTokenizer:
    """Returns a masked-input stand-in with a fixed token count."""

    mask_token = "<mask>"
    mask_token_id = 50264

    def __init__(self, n_tokens):
        self.n_tokens = n_tokens

    def __call__(self, text, return_tensors=None):
        return _FakeMaskedInputs(self.n_tokens)


class TestTokenBound(unittest.TestCase):
    def test_input_too_long_error_is_exported_valueerror(self):
        self.assertTrue(issubclass(mlm.InputTooLongError, ValueError))

    def test_text_mode_over_512_tokens_raises(self):
        # Short in characters (far below MAX_INPUT_LENGTH) but the tokenizer
        # output exceeds MAX_SEQ_LENGTH: in text mode the token bound, not the
        # char bound, must reject the input.
        text = "word " * 200  # 1000 chars, well under 4096
        fake = _FakeTokenizer(mlm.MAX_SEQ_LENGTH + 1)
        with mock.patch.object(mlm, "get_mlm_model", return_value={"tokenizer": fake}):
            with self.assertRaises(mlm.InputTooLongError):
                mlm.compress_text(text, language="en", mode="text")

    def test_text_mode_at_limit_does_not_raise_token_error(self):
        # Exactly MAX_SEQ_LENGTH tokens must pass the text-mode token bound and
        # continue into the pipeline (reaching the NLP stage here).
        fake = _FakeTokenizer(mlm.MAX_SEQ_LENGTH)

        def _reach_nlp(lang):
            raise RuntimeError("reached nlp stage")

        with mock.patch.object(mlm, "get_mlm_model", return_value={"tokenizer": fake}), \
                mock.patch.object(mlm, "get_nlp_model", side_effect=_reach_nlp):
            with self.assertRaises(RuntimeError):
                mlm.compress_text("test", language="en", mode="text")

    def test_sentence_mode_skips_full_text_preflight(self):
        # A multi-sentence input whose full text exceeds MAX_SEQ_LENGTH must
        # not be rejected up front in sentence mode: each sentence is
        # bound-checked individually in get_mlm_probability. Reaching the NLP
        # stage (RuntimeError) proves the preflight was skipped instead of
        # raising InputTooLongError.
        text = " ".join(["short sentence words here."] * 100)
        fake = _FakeTokenizer(mlm.MAX_SEQ_LENGTH + 1)

        def _reach_nlp(lang):
            raise RuntimeError("reached nlp stage")

        with mock.patch.object(mlm, "get_mlm_model", return_value={"tokenizer": fake}), \
                mock.patch.object(mlm, "get_nlp_model", side_effect=_reach_nlp):
            with self.assertRaises(RuntimeError):
                mlm.compress_text(text, language="en", mode="sentence")

    def test_per_sentence_over_512_tokens_raises_input_too_long_error(self):
        # A masked per-sentence input over MAX_SEQ_LENGTH must raise the
        # dedicated InputTooLongError, not a generic ValueError.
        fake_tokenizer = _FakeMaskTokenizer(mlm.MAX_SEQ_LENGTH + 1)
        model_data = {"model": mock.Mock(), "tokenizer": fake_tokenizer, "device": "cpu"}
        with mock.patch.object(mlm, "get_mlm_model", return_value=model_data):
            with self.assertRaises(mlm.InputTooLongError):
                mlm.get_mlm_probability("en", "word " * 10, 0)

    def test_per_sentence_at_limit_does_not_raise(self):
        # Exactly MAX_SEQ_LENGTH tokens must pass the per-sentence bound; the
        # subsequent inference then hits the mocked model and stops there,
        # proving the bound check itself accepted the input.
        fake_tokenizer = _FakeMaskTokenizer(mlm.MAX_SEQ_LENGTH)

        def _boom(*args, **kwargs):
            raise RuntimeError("reached model inference")

        model_data = {"model": mock.Mock(side_effect=_boom), "tokenizer": fake_tokenizer, "device": "cpu"}
        with mock.patch.object(mlm, "get_mlm_model", return_value=model_data):
            with self.assertRaises(RuntimeError):
                mlm.get_mlm_probability("en", "word " * 10, 0)


class TestCompressionApiCompatibility(unittest.TestCase):
    def test_legacy_positional_parameters_precede_keyword_only_controls(self):
        parameters = list(inspect.signature(mlm.compress_text).parameters.values())

        self.assertEqual(
            [parameter.name for parameter in parameters[:4]],
            ["text", "prob_threshold", "no_adjacent_removal", "protect_ner"],
        )
        self.assertEqual(
            [parameter.kind for parameter in parameters[4:]],
            [inspect.Parameter.KEYWORD_ONLY] * 5,
        )
        self.assertEqual(
            [parameter.name for parameter in parameters[4:]],
            ["language", "drop_ratio", "preset", "mode", "calibrate_from_nlp"],
        )

    def test_numeric_second_positional_argument_is_not_language(self):
        text = "The quick brown fox"
        doc = SimpleNamespace(sents=[], ents=[])

        with mock.patch.object(mlm, "detect_language", return_value="en") as detect, \
                mock.patch.object(mlm, "get_nlp_model", return_value=lambda value: doc), \
                mock.patch.object(mlm, "_compress_sentence_mode", return_value="compressed") as compress:
            result = mlm.compress_text(text, 0.2, True, False, calibrate_from_nlp=False)

        self.assertEqual(result, "compressed")
        detect.assert_called_once_with(text)
        self.assertEqual(compress.call_args.args[2], "en")
        self.assertTrue(compress.call_args.args[6])
        self.assertFalse(compress.call_args.args[7])

    def test_prob_threshold_changes_legacy_removals(self):
        text = "Alpha beta gamma delta"
        doc = SimpleNamespace(
            sents=[SimpleNamespace(text=text, start_char=0)],
            ents=[],
        )
        probabilities = [0.1, 0.9, 0.2, 0.8]

        def mlm_probability(language, sentence, word_idx):
            return probabilities[word_idx]

        with mock.patch.object(mlm, "get_nlp_model", return_value=lambda value: doc), \
                mock.patch.object(mlm, "get_mlm_probability", side_effect=mlm_probability):
            conservative = mlm.compress_text(
                text,
                prob_threshold=0.95,
                language="en",
                calibrate_from_nlp=False,
                protect_ner=False,
            )
            aggressive = mlm.compress_text(
                text,
                prob_threshold=0.5,
                language="en",
                calibrate_from_nlp=False,
                protect_ner=False,
            )

        self.assertEqual(conservative, text)
        self.assertEqual(aggressive, "Alpha gamma")


class TestSentenceCalibration(unittest.TestCase):
    def _doc(self):
        first = "Alpha beta gamma delta"
        second = "Epsilon zeta eta theta"
        return SimpleNamespace(
            sents=[
                SimpleNamespace(text=first, start_char=0),
                SimpleNamespace(text=second, start_char=len(first) + 1),
            ],
            ents=[],
        )

    def test_default_sentence_mode_calibrates_each_sentence(self):
        text = "Alpha beta gamma delta. Epsilon zeta eta theta"
        doc = self._doc()

        def nlp_compress(sentence, lang):
            return "Alpha delta" if sentence.startswith("Alpha") else "Epsilon theta"

        with mock.patch.object(mlm, "get_nlp_model", return_value=lambda value: doc), \
                mock.patch("caveman_compress_nlp.compress_text", side_effect=nlp_compress) as nlp, \
                mock.patch.object(mlm, "get_mlm_probability", return_value=0.9):
            mlm.compress_text(text, language="en")

        self.assertEqual(
            [call.args for call in nlp.call_args_list],
            [
                ("Alpha beta gamma delta",),
                ("Epsilon zeta eta theta",),
            ],
        )
        self.assertEqual(
            [call.kwargs for call in nlp.call_args_list],
            [{"lang": "en"}, {"lang": "en"}],
        )

    def test_explicit_drop_ratio_is_not_replaced_by_sentence_calibration(self):
        text = "Alpha beta gamma delta. Epsilon zeta eta theta"
        doc = self._doc()

        with mock.patch.object(mlm, "get_nlp_model", return_value=lambda value: doc), \
                mock.patch("caveman_compress_nlp.compress_text") as nlp, \
                mock.patch.object(mlm, "get_mlm_probability", return_value=0.9):
            result = mlm.compress_text(text, language="en", drop_ratio=0.0)

        nlp.assert_not_called()
        self.assertEqual(result, "Alpha beta gamma delta Epsilon zeta eta theta")


class TestFrozenRuntimeConfig(unittest.TestCase):
    """Active languages come from the build manifest, never runtime env."""

    def test_catalogue_has_all_lt_ngram_languages(self):
        self.assertEqual(
            set(mlm.MLM_LANGUAGE_CATALOGUE),
            {"en", "de", "fr", "es"},
        )

    def test_supported_languages_are_manifest_derived(self):
        # The module-level SUPPORTED_LANGUAGES is fixed at import time from the
        # manifest (or its en/fr fallback in local development). The core no
        # longer exposes env-based activation helpers.
        self.assertEqual(set(mlm.SUPPORTED_LANGUAGES), {"en", "fr"})
        self.assertFalse(hasattr(mlm, "_active_supported_languages"))
        self.assertFalse(hasattr(mlm, "_load_custom_models"))

    def test_manifest_absent_falls_back_to_en_fr(self):
        self.assertEqual(
            set(language_catalog.load_active_languages("/nonexistent/manifest.json")),
            {"en", "fr"},
        )

    def test_manifest_activates_catalogue_language(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "languages.json")
            language_catalog.write_language_manifest(
                path,
                {
                    "en": {"model": "roberta-base", "spacy": "en_core_web_sm"},
                    "fr": {"model": "camembert-base", "spacy": "fr_core_news_sm"},
                    "de": {"model": "bert-base-german-cased", "spacy": "de_core_news_sm"},
                },
            )
            active = language_catalog.load_active_languages(path)
        self.assertIn("de", active)
        self.assertEqual(active["de"]["model"], "bert-base-german-cased")

    def test_manifest_includes_custom_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "languages.json")
            language_catalog.write_language_manifest(
                path,
                {
                    "en": {"model": "roberta-base", "spacy": "en_core_web_sm"},
                    "fr": {"model": "camembert-base", "spacy": "fr_core_news_sm"},
                    "sv": {"model": "KB/bert-base-swedish-cased", "spacy": "sv_core_news_sm"},
                },
            )
            active = language_catalog.load_active_languages(path)
        self.assertEqual(
            active["sv"],
            {"model": "KB/bert-base-swedish-cased", "spacy": "sv_core_news_sm"},
        )

    def test_runtime_env_does_not_affect_active_languages(self):
        # Setting LANGUAGES/CUSTOM_MLM_MODELS at runtime must not change which
        # languages are active: resolution reads the manifest (or en/fr
        # fallback), never the environment.
        os.environ["LANGUAGES"] = "de,es"
        os.environ["CUSTOM_MLM_MODELS"] = (
            '{"sv": {"model": "KB/x", "spacy": "sv_core_news_sm"}}'
        )
        try:
            active = language_catalog.load_active_languages("/nonexistent/manifest.json")
        finally:
            os.environ.pop("LANGUAGES", None)
            os.environ.pop("CUSTOM_MLM_MODELS", None)
        self.assertEqual(set(active), {"en", "fr"})
        self.assertNotIn("de", active)
        self.assertNotIn("sv", active)


class TestOfflineModelLoading(unittest.TestCase):
    """MLM models must be loaded offline (local_files_only=True)."""

    def test_mlm_load_passes_local_files_only(self):
        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock.Mock()
        ) as tok, mock.patch(
            "transformers.AutoModelForMaskedLM.from_pretrained", return_value=mock.Mock()
        ) as mod:
            mlm._models.pop("en", None)
            mlm.get_mlm_model("en")
        tok.assert_called_once_with("roberta-base", local_files_only=True)
        mod.assert_called_once_with("roberta-base", local_files_only=True)


class TestUnknownLanguageHeuristic(unittest.TestCase):
    """The word-list heuristic returns None with no en/fr evidence."""

    def test_no_english_french_evidence_returns_none(self):
        with mock.patch.object(mlm, "get_fasttext_model", return_value=None):
            self.assertIsNone(mlm.detect_language("zzz qqq xxx yyy"))

    def test_french_evidence_returns_fr(self):
        with mock.patch.object(mlm, "get_fasttext_model", return_value=None):
            self.assertEqual(mlm.detect_language("le chat et le chien sont ici"), "fr")

    def test_english_evidence_returns_en(self):
        with mock.patch.object(mlm, "get_fasttext_model", return_value=None):
            self.assertEqual(mlm.detect_language("the cat and the dog are here"), "en")

    def test_unknown_detection_returns_original_text(self):
        # When no language can be determined, compress_text returns the input
        # unchanged without touching the MLM/NLP models.
        text = "zzz qqq xxx yyy"
        with mock.patch.object(mlm, "get_fasttext_model", return_value=None), \
                mock.patch.object(mlm, "get_nlp_model") as nlp, \
                mock.patch.object(mlm, "get_mlm_model") as mlm_model:
            out = mlm.compress_text(text)
        self.assertEqual(out, text)
        nlp.assert_not_called()
        mlm_model.assert_not_called()


class TestInactiveLanguagePassthrough(unittest.TestCase):
    """Direct compress_text returns inactive-language input unchanged, never erroring."""

    def _assert_passthrough(self, language, text):
        # The input must come back unchanged and no model may be consulted:
        # get_nlp_model / get_mlm_model are never called for an inactive language.
        with mock.patch.object(mlm, "get_nlp_model") as nlp, \
                mock.patch.object(mlm, "get_mlm_model") as mlm_model:
            out = mlm.compress_text(text, language=language)
        self.assertEqual(out, text)
        nlp.assert_not_called()
        mlm_model.assert_not_called()

    def test_catalogued_but_inactive_de_es_return_unchanged(self):
        # de/es are in the catalogue but not activated by default (LANGUAGES="en,fr").
        text = "Der schnelle braune Fuchs springt über den faulen Hund."
        for code in ("de", "es"):
            with self.subTest(code=code):
                self._assert_passthrough(code, text)

    def test_unknown_language_returns_unchanged(self):
        self._assert_passthrough("xx_unknown", "some words here to compress")

    def test_zh_inactive_returns_unchanged(self):
        # zh has no MLM model in the catalogue, so it is inactive by default
        # and passes through unchanged even in text mode.
        self._assert_passthrough("zh", "你好世界 hello world")

    def test_inactive_language_ignores_length_bound(self):
        # "Never error": an inactive language returns the input unchanged even
        # when it exceeds MAX_INPUT_LENGTH, rather than raising ValueError.
        self._assert_passthrough("de", "word " * (mlm.MAX_INPUT_LENGTH + 1))

    def test_explicit_inactive_language_skips_autodetection(self):
        # An explicit language code must not trigger language detection, and
        # the text is returned unchanged without touching any model.
        text = "hello world"
        with mock.patch.object(mlm, "detect_language") as det, \
                mock.patch.object(mlm, "get_nlp_model") as nlp:
            out = mlm.compress_text(text, language="es")
        self.assertEqual(out, text)
        det.assert_not_called()
        nlp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
