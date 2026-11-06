"""The llama.cpp rewrite backend builds the same stateless request as fp16.

These tests never start ``llama-server`` and need no GGUF: they cover the
request/response plumbing, the error messages that tell you how to stage the
binary or the weights, and the loader-path parsing used for CUDA runtime fixes.
"""

from __future__ import annotations

import pytest

from src.rewriter_llama import (
    LlamaCppRewriter,
    RewriterError,
    build_chat_payload,
    parse_chat_response,
    parse_loader_path,
)

MESSAGES = [
    {"role": "system", "content": "Rewrite this."},
    {"role": "user", "content": "A piano track."},
]


def test_payload_is_greedy_and_stateless():
    payload = build_chat_payload(MESSAGES, 384)
    assert payload["messages"] == MESSAGES
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 384
    assert payload["stream"] is False
    # No history, no extra fields the model could key on.
    assert set(payload) == {"messages", "temperature", "max_tokens", "stream"}


def test_parse_response_strips_text():
    payload = {"choices": [{"message": {"content": "  Clean prose.  "}}]}
    assert parse_chat_response(payload) == "Clean prose."


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {"content": "   "}}]},
    ],
)
def test_parse_response_rejects_empty(payload):
    with pytest.raises(RewriterError):
        parse_chat_response(payload)


def test_parse_loader_path():
    out = "some noise\nexport LD_LIBRARY_PATH=/a:/b\nmore\n"
    assert parse_loader_path(out) == "/a:/b"
    assert parse_loader_path("nothing here") == ""


def test_missing_server_binary_is_actionable(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")
    rewriter = LlamaCppRewriter(
        model_path=model, server_bin=tmp_path / "missing" / "llama-server"
    )
    with pytest.raises(RewriterError, match="fetch_llamacpp_text"):
        rewriter.rewrite(MESSAGES)


def test_missing_gguf_is_actionable(tmp_path):
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    rewriter = LlamaCppRewriter(
        model_path=tmp_path / "absent.gguf", server_bin=server
    )
    with pytest.raises(RewriterError, match="GGUF model not found"):
        rewriter.rewrite(MESSAGES)


def test_empty_messages_rejected(tmp_path):
    rewriter = LlamaCppRewriter(model_path=tmp_path / "m.gguf")
    with pytest.raises(RewriterError, match="empty message list"):
        rewriter.rewrite([])


def test_name_and_unload_without_server(tmp_path):
    rewriter = LlamaCppRewriter(model_path=tmp_path / "q8_0.gguf")
    assert rewriter.name == "gguf:q8_0.gguf"
    assert rewriter.prefetch_model_id is None
    rewriter.unload()  # no process: must not raise
