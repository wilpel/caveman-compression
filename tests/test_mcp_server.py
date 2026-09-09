"""Protocol boundary tests for the MCP server (mcp_server.py).

The compression backend modules are stubbed only inside an isolated import
scope (see _import_target), which restores sys.modules immediately. The stubs
therefore never leak into unittest collection, so the real backend tests never
see them regardless of discovery order.
"""

import importlib
import json
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BACKENDS = ("caveman_compress_mlm", "caveman_compress_nlp")


class _Behavior:
    mlm_error = None
    mlm_result = "mlm-compressed"
    nlp_result = "nlp-compressed"
    nlp_calls = []
    detected = "en"


def _fake_mlm_compress(text, language=None, preset="lite", mode="sentence"):
    if _Behavior.mlm_error is not None:
        raise _Behavior.mlm_error
    return _Behavior.mlm_result


def _fake_nlp_compress(text, lang="en"):
    _Behavior.nlp_calls.append(lang)
    return _Behavior.nlp_result


_mlm = types.ModuleType("caveman_compress_mlm")
_mlm.SUPPORTED_LANGUAGES = {
    "en": {},
    "fr": {},
    "sv": {"model": "KB/bert-base-swedish-cased", "spacy": "sv_core_news_sm"},
}
_mlm.detect_language = lambda text: _Behavior.detected
_mlm.compress_text = _fake_mlm_compress
_mlm.get_mlm_model = lambda lang: None
_mlm.InputTooLongError = type("InputTooLongError", (ValueError,), {})

_nlp = types.ModuleType("caveman_compress_nlp")
_nlp.compress_text = _fake_nlp_compress


def _import_target(name):
    """Import `name` fresh under the backend stubs, then restore sys.modules.

    The stubs and the target module live only for the duration of this call;
    afterwards sys.modules is returned to its prior state so nothing leaks into
    collection or contaminates the real backend tests.
    """
    prior = {key: sys.modules.pop(key, None) for key in (*_BACKENDS, name)}
    try:
        sys.modules["caveman_compress_mlm"] = _mlm
        sys.modules["caveman_compress_nlp"] = _nlp
        return importlib.import_module(name)
    finally:
        for key, value in prior.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class McpProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mcp_server = _import_target("mcp_server")

    def setUp(self):
        _Behavior.mlm_error = None
        _Behavior.nlp_calls = []
        _Behavior.detected = "en"

    def _call(self, arguments, tool="caveman_compress", request_id=3):
        req = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        return json.loads(self.mcp_server.process_line(json.dumps(req)))

    # --- process_line: transport framing ---

    def test_parse_error_invalid_json(self):
        msg = json.loads(self.mcp_server.process_line("{not json"))
        self.assertEqual(msg["error"]["code"], -32700)
        self.assertIsNone(msg["id"])

    def test_invalid_request_non_object(self):
        msg = json.loads(self.mcp_server.process_line("[1, 2, 3]"))
        self.assertEqual(msg["error"]["code"], -32600)
        self.assertIsNone(msg["id"])

    def test_invalid_request_missing_method(self):
        msg = json.loads(self.mcp_server.process_line(json.dumps({"id": 1})))
        self.assertEqual(msg["error"]["code"], -32600)
        self.assertEqual(msg["id"], 1)

    def test_notification_emits_no_response(self):
        out = self.mcp_server.process_line(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        )
        self.assertIsNone(out)

    def test_blank_line_emits_no_response(self):
        self.assertIsNone(self.mcp_server.process_line("   "))

    # --- method dispatch ---

    def test_initialize(self):
        out = self.mcp_server.process_line(
            json.dumps({"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {}})
        )
        msg = json.loads(out)
        self.assertEqual(msg["id"], 7)
        self.assertEqual(msg["jsonrpc"], "2.0")
        self.assertIn("result", msg)
        self.assertEqual(msg["result"]["protocolVersion"], "2024-11-05")

    def test_tools_list(self):
        out = self.mcp_server.process_line(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        )
        msg = json.loads(out)
        self.assertEqual(msg["id"], 1)
        self.assertIn("result", msg)
        self.assertIn("tools", msg["result"])

    def test_unknown_method(self):
        out = self.mcp_server.process_line(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "bogus"}))
        msg = json.loads(out)
        self.assertEqual(msg["error"]["code"], -32601)

    # --- tools/call validation ---

    def test_tools_call_success(self):
        msg = self._call({"text": "hello world"})
        self.assertNotIn("error", msg)
        self.assertIn("result", msg)
        self.assertEqual(msg["result"]["content"][0]["type"], "text")
        self.assertEqual(msg["result"]["metadata"]["model"], "caveman-mlm-en")
        self.assertFalse(msg["result"]["metadata"]["fallback"])
        self.assertFalse(msg["result"]["metadata"]["uncompressed"])

    def test_tools_call_missing_text(self):
        self.assertEqual(self._call({})["error"]["code"], -32602)

    def test_tools_call_text_wrong_type(self):
        self.assertEqual(self._call({"text": 99})["error"]["code"], -32602)

    def test_tools_call_text_empty(self):
        self.assertEqual(self._call({"text": "  "})["error"]["code"], -32602)

    def test_tools_call_text_too_long(self):
        self.assertEqual(self._call({"text": "a" * 4097})["error"]["code"], -32602)

    def test_json_rpc_transport_allows_at_least_64_kib(self):
        self.assertGreaterEqual(self.mcp_server.MAX_LINE_LENGTH, 64 * 1024)

    def test_tools_call_accepts_exactly_4096_unicode_characters(self):
        text = "😀" * 4096
        req = {
            "jsonrpc": "2.0",
            "id": 4096,
            "method": "tools/call",
            "params": {
                "name": "caveman_compress",
                "arguments": {"text": text},
            },
        }
        line = json.dumps(req, ensure_ascii=False)
        self.assertEqual(len(text), self.mcp_server.MAX_INPUT_LENGTH)
        self.assertLessEqual(len(line.encode("utf-8")), self.mcp_server.MAX_LINE_LENGTH)
        msg = json.loads(self.mcp_server.process_line(line))
        self.assertEqual(msg["id"], 4096)
        self.assertNotIn("error", msg)
        self.assertEqual(msg["result"]["metadata"]["original_size"], 4096)

    def test_tools_call_invalid_preset(self):
        self.assertEqual(self._call({"text": "hi", "preset": "nope"})["error"]["code"], -32602)

    def test_tools_call_invalid_mode(self):
        self.assertEqual(self._call({"text": "hi", "mode": "nope"})["error"]["code"], -32602)

    def test_tools_call_invalid_method(self):
        self.assertEqual(self._call({"text": "hi", "method": "nope"})["error"]["code"], -32602)

    def test_tools_call_language_wrong_type(self):
        self.assertEqual(self._call({"text": "hi", "language": 5})["error"]["code"], -32602)

    def test_tools_call_arguments_not_object(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "caveman_compress", "arguments": [1, 2]},
        }
        msg = json.loads(self.mcp_server.process_line(json.dumps(req)))
        self.assertEqual(msg["error"]["code"], -32602)

    def test_tools_call_unknown_tool(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
        msg = json.loads(self.mcp_server.process_line(json.dumps(req)))
        self.assertEqual(msg["error"]["code"], -32601)

    # --- MLM -> NLP fallback ---

    def test_mlm_fallback_on_oserror(self):
        _Behavior.mlm_error = OSError("model unavailable")
        msg = self._call({"text": "hello world"})
        self.assertNotIn("error", msg)
        self.assertEqual(msg["result"]["metadata"]["model"], "caveman-nlp-en")
        self.assertTrue(msg["result"]["metadata"]["fallback"])

    def test_mlm_fallback_reports_effective_preset_mode(self):
        # When MLM falls back to NLP, the metadata must report the actual
        # effective preset='lite' and mode='sentence', not the requested
        # (unfulfilled) MLM values.
        _Behavior.mlm_error = OSError("model unavailable")
        msg = self._call({"text": "hello world", "preset": "ultra", "mode": "text"})
        self.assertNotIn("error", msg)
        meta = msg["result"]["metadata"]
        self.assertEqual(meta["model"], "caveman-nlp-en")
        self.assertTrue(meta["fallback"])
        self.assertEqual(meta["preset"], "lite")
        self.assertEqual(meta["mode"], "sentence")

    def test_custom_active_language_forced_nlp_uses_manifest_language(self):
        _Behavior.detected = "sv"
        msg = self._call({"text": "hej världen", "method": "nlp"})
        self.assertNotIn("error", msg)
        self.assertEqual(msg["result"]["metadata"]["model"], "caveman-nlp-sv")
        self.assertEqual(_Behavior.nlp_calls, ["sv"])

    def test_custom_active_language_mlm_fallback_uses_manifest_language(self):
        _Behavior.detected = "sv"
        _Behavior.mlm_error = OSError("model unavailable")
        msg = self._call({"text": "hej världen"})
        self.assertNotIn("error", msg)
        self.assertEqual(msg["result"]["metadata"]["model"], "caveman-nlp-sv")
        self.assertTrue(msg["result"]["metadata"]["fallback"])
        self.assertEqual(_Behavior.nlp_calls, ["sv"])

    def test_non_active_language_returns_unchanged(self):
        # Spanish is not an active MLM language (default only en/fr), so the
        # input is returned unchanged rather than NLP-compressed or errored.
        _Behavior.detected = "es"
        msg = self._call({"text": "hola mundo"})
        self.assertNotIn("error", msg)
        result = msg["result"]
        self.assertEqual(result["content"][0]["text"], "hola mundo")
        self.assertEqual(result["metadata"]["model"], "caveman-uncompressed-es")
        self.assertFalse(result["metadata"]["fallback"])
        self.assertTrue(result["metadata"]["uncompressed"])
        self.assertEqual(result["metadata"]["compression_ratio"], 1.0)
        self.assertEqual(result["metadata"]["compressed_size"], result["metadata"]["original_size"])

    def test_non_active_explicit_language_returns_unchanged(self):
        msg = self._call({"text": "hola mundo", "language": "es"})
        self.assertNotIn("error", msg)
        result = msg["result"]
        self.assertEqual(result["content"][0]["text"], "hola mundo")
        self.assertEqual(result["metadata"]["model"], "caveman-uncompressed-es")
        self.assertTrue(result["metadata"]["uncompressed"])
        self.assertEqual(result["metadata"]["compression_ratio"], 1.0)

    def test_non_active_language_nlp_method_returns_unchanged(self):
        # Even an explicit method="nlp" must not NLP-compress a non-active
        # language; the input is returned unchanged instead.
        _Behavior.detected = "es"
        msg = self._call({"text": "hola mundo", "method": "nlp"})
        self.assertNotIn("error", msg)
        result = msg["result"]
        self.assertEqual(result["content"][0]["text"], "hola mundo")
        self.assertEqual(result["metadata"]["model"], "caveman-uncompressed-es")
        self.assertTrue(result["metadata"]["uncompressed"])
        self.assertEqual(result["metadata"]["compression_ratio"], 1.0)

    def test_mlm_unexpected_error_is_internal(self):
        _Behavior.mlm_error = RuntimeError("boom")
        msg = self._call({"text": "hello world"})
        self.assertEqual(msg["error"]["code"], -32000)

    def test_detect_language_none_returns_unknown_unchanged(self):
        # A detector that returns None must be normalized to "unknown" and the
        # input returned unchanged with a stable model tag (not "None").
        _Behavior.detected = None
        msg = self._call({"text": "some text"})
        self.assertNotIn("error", msg)
        result = msg["result"]
        self.assertEqual(result["content"][0]["text"], "some text")
        self.assertEqual(result["metadata"]["language"], "unknown")
        self.assertEqual(result["metadata"]["model"], "caveman-uncompressed-unknown")
        self.assertFalse(result["metadata"]["fallback"])
        self.assertTrue(result["metadata"]["uncompressed"])
        self.assertEqual(result["metadata"]["compression_ratio"], 1.0)

    def test_detect_language_unknown_string_returns_unknown_unchanged(self):
        # A detector that returns the literal "unknown" is unchanged in the
        # response and the input is returned unchanged.
        _Behavior.detected = "unknown"
        msg = self._call({"text": "some text"})
        self.assertNotIn("error", msg)
        result = msg["result"]
        self.assertEqual(result["content"][0]["text"], "some text")
        self.assertEqual(result["metadata"]["language"], "unknown")
        self.assertEqual(result["metadata"]["model"], "caveman-uncompressed-unknown")
        self.assertTrue(result["metadata"]["uncompressed"])
        self.assertEqual(result["metadata"]["compression_ratio"], 1.0)

    def test_mlm_input_too_long_returns_invalid_params(self):
        # InputTooLongError raised by an active sentence-mode MLM core call is
        # surfaced as a JSON-RPC -32602 invalid-params error (not a generic
        # -32000 internal error).
        _Behavior.mlm_error = self.mcp_server.InputTooLongError("sequence too long")
        msg = self._call({"text": "hello world"})
        self.assertEqual(msg["error"]["code"], -32602)

    # --- caveman_stats ---

    def test_caveman_stats(self):
        msg = self._call({"text": "hello world this is a test"}, tool="caveman_stats")
        self.assertNotIn("error", msg)
        self.assertIn("result", msg)
        self.assertEqual(msg["result"]["metadata"]["language"], "en")

    def test_no_stub_leaks_into_sys_modules(self):
        # The target was imported under stubs in an isolated scope and
        # sys.modules was restored immediately, so no backend stub may remain.
        for name in _BACKENDS:
            mod = sys.modules.get(name)
            self.assertFalse(
                mod is not None and getattr(mod, "__file__", None) is None,
                f"{name} leaked a stub into sys.modules",
            )
        self.assertNotIn("mcp_server", sys.modules)


if __name__ == "__main__":
    unittest.main()
