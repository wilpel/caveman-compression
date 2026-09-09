#!/usr/bin/env python3
"""
Caveman Compression MCP Server
Provides MLM-based text compression via MCP protocol.
"""

import json
import sys
import os

# Add the caveman-compression directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caveman_compress_mlm import compress_text, detect_language, SUPPORTED_LANGUAGES, InputTooLongError
from caveman_compress_nlp import compress_text as compress_text_nlp

# MCP Protocol Constants
MCP_VERSION = "2024-11-05"

# Input constraints
MAX_INPUT_LENGTH = 4096
MAX_LINE_LENGTH = 64 * 1024  # bytes per stdin line (JSON-RPC message)
PRESETS = ("lite", "full", "ultra")
MODES = ("sentence", "text")
METHODS = ("mlm", "nlp")

# Languages with MLM models available (derived from SUPPORTED_LANGUAGES)
MLM_LANGUAGES = set(SUPPORTED_LANGUAGES.keys())


def error_response(code, message, request_id=None):
    """Build a JSON-RPC error response dict."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _invalid_params(message):
    return {"error": {"code": -32602, "message": message}}


def handle_request(request):
    """Handle a decoded JSON-RPC request object."""
    if request.get("jsonrpc") != "2.0":
        return {"error": {"code": -32600, "message": "Invalid Request: 'jsonrpc' must be '2.0'"}}

    request_id = request.get("id")
    if "id" in request and not (
        isinstance(request_id, str)
        or (isinstance(request_id, int) and not isinstance(request_id, bool))
    ):
        return {"error": {"code": -32600, "message": "Invalid Request: 'id' must be a string or a non-boolean integer"}}

    method = request.get("method")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "caveman-compression",
                "version": "1.0.0"
            }
        }

    elif method == "tools/list":
        return {
            "tools": [
                {
                    "name": "caveman_compress",
                    "description": "Compress text using Caveman compression (MLM for active languages such as English/French, returns input unchanged for others)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to compress"
                            },
                            "language": {
                                "type": "string",
                                "description": "Language code (en, fr, de, etc.). Auto-detect if not specified."
                            },
                            "preset": {
                                "type": "string",
                                "description": "Compression preset: lite (NLP level), full (NLP×1.5), ultra (NLP×2)",
                                "default": "lite",
                                "enum": ["lite", "full", "ultra"]
                            },
                            "mode": {
                                "type": "string",
                                "description": "Compression mode: sentence (per sentence, faster), text (full text, more context)",
                                "default": "sentence",
                                "enum": ["sentence", "text"]
                            },
                            "method": {
                                "type": "string",
                                "description": "Compression method: mlm (default) or nlp (force NLP); non-active languages are returned unchanged",
                                "default": "mlm",
                                "enum": ["mlm", "nlp"]
                            }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "caveman_stats",
                    "description": "Get compression statistics for text",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to analyze"
                            }
                        },
                        "required": ["text"]
                    }
                }
            ]
        }

    elif method == "tools/call":
        return handle_tools_call(params)

    elif method == "notifications/initialized":
        return None

    else:
        return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}


def handle_tools_call(params):
    """Handle a tools/call request."""
    if not isinstance(params, dict):
        return _invalid_params("Invalid params: expected an object")

    name = params.get("name")
    if not isinstance(name, str):
        return _invalid_params("Invalid params: 'name' must be a string")

    arguments = params.get("arguments", {})
    if not isinstance(arguments, dict):
        return _invalid_params("Invalid params: 'arguments' must be an object")

    if name == "caveman_compress":
        return call_caveman_compress(arguments)
    elif name == "caveman_stats":
        return call_caveman_stats(arguments)
    else:
        return {"error": {"code": -32601, "message": f"Unknown tool: {name}"}}


def call_caveman_compress(arguments):
    text = arguments.get("text")
    if not isinstance(text, str):
        return _invalid_params("Invalid params: 'text' must be a non-empty string")
    if not text.strip():
        return _invalid_params("Invalid params: 'text' must not be empty")
    if len(text) > MAX_INPUT_LENGTH:
        return _invalid_params(f"Invalid params: 'text' exceeds maximum length of {MAX_INPUT_LENGTH} characters")

    language = arguments.get("language")
    if language is not None and not isinstance(language, str):
        return _invalid_params("Invalid params: 'language' must be a string")

    preset = arguments.get("preset", "lite")
    if not isinstance(preset, str) or preset not in PRESETS:
        return _invalid_params(f"Invalid params: 'preset' must be one of {list(PRESETS)}")

    mode = arguments.get("mode", "sentence")
    if not isinstance(mode, str) or mode not in MODES:
        return _invalid_params(f"Invalid params: 'mode' must be one of {list(MODES)}")

    method = arguments.get("method", "mlm")  # "mlm" (default) or "nlp"
    if not isinstance(method, str) or method not in METHODS:
        return _invalid_params(f"Invalid params: 'method' must be one of {list(METHODS)}")

    if method == "nlp" and (preset != "lite" or mode != "sentence"):
        return _invalid_params("Invalid params: 'method' 'nlp' only supports preset='lite' and mode='sentence'")

    try:
        lang = language or detect_language(text) or "unknown"

        fallback = False
        uncompressed = False

        if lang not in MLM_LANGUAGES:
            # Non-active language: return the input unchanged. There is no
            # provisioned model for this language, so no compression (and no
            # NLP fallback) is attempted.
            compressed = text
            model = f"caveman-uncompressed-{lang}"
            uncompressed = True
        else:
            # Reject incompatible language/mode combinations before inference
            if lang == "zh" and mode == "text":
                return _invalid_params("Invalid params: 'mode' 'text' is not supported for Chinese (zh); use mode='sentence'")

            if method == "nlp":
                compressed = compress_text_nlp(text, lang=lang)
                model = f"caveman-nlp-{lang}"
            else:
                # method="mlm" (default): try MLM, fallback to NLP
                try:
                    compressed = compress_text(text, language=lang, preset=preset, mode=mode)
                    model = f"caveman-mlm-{lang}"
                except OSError as e:
                    # MLM model unavailable for this language: expected fallback.
                    print(f"Warning: MLM unavailable for {lang}, falling back to NLP: {e}", file=sys.stderr)
                    compressed = compress_text_nlp(text, lang=lang)
                    model = f"caveman-nlp-{lang}"
                    fallback = True
                    # NLP fallback is always lite/sentence; report the actual
                    # effective preset/mode rather than the requested MLM values.
                    preset = "lite"
                    mode = "sentence"
                except InputTooLongError:
                    return _invalid_params("Invalid params: 'text' exceeds the MLM sequence length")

        return {
            "content": [{"type": "text", "text": compressed}],
            "metadata": {
                "language": lang,
                "model": model,
                "preset": preset,
                "mode": mode,
                "fallback": fallback,
                "uncompressed": uncompressed,
                "original_size": len(text),
                "compressed_size": len(compressed),
                "compression_ratio": len(compressed) / len(text) if text else 0
            }
        }
    except Exception as e:
        print(f"Error during compression: {e}", file=sys.stderr)
        return {"error": {"code": -32000, "message": "Compression failed"}}


def call_caveman_stats(arguments):
    text = arguments.get("text")
    if not isinstance(text, str):
        return _invalid_params("Invalid params: 'text' must be a non-empty string")
    if not text.strip():
        return _invalid_params("Invalid params: 'text' must not be empty")
    if len(text) > MAX_INPUT_LENGTH:
        return _invalid_params(f"Invalid params: 'text' exceeds maximum length of {MAX_INPUT_LENGTH} characters")

    try:
        lang = detect_language(text)
        orig_tokens = len(text.strip()) // 4

        return {
            "content": [{"type": "text", "text": f"Language: {lang}, Tokens: ~{orig_tokens}"}],
            "metadata": {
                "language": lang,
                "original_size": len(text),
                "estimated_tokens": orig_tokens
            }
        }
    except Exception as e:
        print(f"Error during stats: {e}", file=sys.stderr)
        return {"error": {"code": -32000, "message": "Internal error"}}


def process_line(line):
    """
    Process one line of JSON-RPC input.

    Returns a JSON-RPC response string to print, or None when no response is
    required (blank line, notification, or anything without a valid id).
    Never raises; all failures are converted to JSON-RPC error responses.
    """
    # Reject oversized lines (content over MAX_LINE_LENGTH bytes) and drain the
    # remainder so the stream stays aligned for the next message.
    content = line[:-1] if line.endswith("\n") else line
    if len(content.encode("utf-8")) > MAX_LINE_LENGTH:
        while not line.endswith("\n"):
            line = sys.stdin.readline(MAX_LINE_LENGTH)
            if not line:
                break
        return json.dumps(error_response(-32600, "Invalid Request: line too long", None))

    line = line.strip()
    if not line:
        return None

    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        return json.dumps(error_response(-32700, "Parse error", None))

    if not isinstance(request, dict):
        return json.dumps(error_response(-32600, "Invalid Request", None))

    # A request without an "id" member is a notification: never respond.
    if "id" not in request:
        return None

    request_id = request.get("id")

    if not isinstance(request.get("method"), str):
        return json.dumps(error_response(-32600, "Invalid Request", request_id))

    try:
        result = handle_request(request)
    except Exception as e:
        print(f"Error handling request: {e}", file=sys.stderr)
        return json.dumps(error_response(-32603, "Internal error", request_id))

    if result is None:
        return None

    if "error" in result:
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": result["error"],
        }
    else:
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
    return json.dumps(response)


def main():
    """Main MCP server loop"""
    try:
        from caveman_compress_mlm import get_mlm_model
        get_mlm_model("en")
        get_mlm_model("fr")
        print("MLM models loaded (en, fr).", file=sys.stderr)
    except Exception as e:
        print(f"Warning: MLM models failed to load: {e}", file=sys.stderr)

    print("Caveman Compression MCP Server ready", file=sys.stderr)

    while True:
        line = sys.stdin.readline(MAX_LINE_LENGTH + 1)
        if not line:
            break
        response = process_line(line)
        if response is not None:
            print(response, flush=True)


if __name__ == "__main__":
    main()
