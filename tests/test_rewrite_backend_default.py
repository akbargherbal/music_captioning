"""The GGUF rewrite backend fails fast, before captioning costs GPU time."""

from __future__ import annotations

import pytest

from scripts.batch import (
    DEFAULT_LLAMACPP_SERVER,
    DEFAULT_REWRITE_BACKEND,
    DEFAULT_REWRITE_GGUF,
    check_llamacpp_paths,
)


def test_gguf_is_the_default_backend():
    assert DEFAULT_REWRITE_BACKEND == "llamacpp"
    assert DEFAULT_REWRITE_GGUF.name == "Qwen3-4B-Instruct-2507-Q8_0.gguf"
    assert DEFAULT_LLAMACPP_SERVER.name == "llama-server"


def test_missing_weights_reported_before_start(tmp_path):
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n")
    with pytest.raises(FileNotFoundError, match="GGUF weights not found"):
        check_llamacpp_paths(tmp_path / "absent.gguf", server)


def test_missing_server_reported(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="llama-server not found"):
        check_llamacpp_paths(gguf, tmp_path / "no-server")


def test_both_missing_lists_both(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        check_llamacpp_paths(tmp_path / "absent.gguf", tmp_path / "no-server")
    message = str(excinfo.value)
    assert "GGUF weights" in message and "llama-server" in message


def test_present_files_pass(tmp_path):
    gguf = tmp_path / "m.gguf"
    server = tmp_path / "llama-server"
    gguf.write_bytes(b"")
    server.write_bytes(b"")
    check_llamacpp_paths(gguf, server)  # must not raise
