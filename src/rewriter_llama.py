"""llama.cpp backend for the stateless caption rewrite.

Same protocol as :class:`src.rewriter_hf.HFRewriter` (``rewrite(messages)`` ->
plain text), but the model runs as a **GGUF** file under ``llama-server``
instead of fp16 transformers. Nothing about the prompt or the verification
changes: only the inference engine does, so the two backends are directly
comparable.

One ``llama-server`` process is started lazily on first use and reused for every
track (loading a 4B GGUF takes seconds; the HTTP call per track is cheap). All
calls are stateless: a fresh ``system`` + ``user`` message list, no history, and
greedy decoding (``temperature: 0``) so the output is reproducible.

T4 notes: ``sm_75`` builds come from ``plumbing/fetch_llamacpp_text.sh``; the
helper resolves CUDA runtime libraries through ``cuda_loader_path.sh`` before
spawning the server, so a binary built against a different CUDA minor still
finds its ``libcudart``/``libcublas``.

No third-party HTTP client is needed (stdlib ``urllib`` only), so this module
imports on a CPU-only runtime without llama.cpp installed.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .rewrite import REWRITE_PROMPT_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_BIN = REPO_ROOT / "llama.cpp-text" / "build" / "bin" / "llama-server"
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_CTX_SIZE = 4096


class RewriterError(RuntimeError):
    """Raised when the server cannot start, dies, or returns nothing."""


def build_chat_payload(
    messages: list[dict[str, str]], max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> dict[str, object]:
    """OpenAI-compatible chat request: greedy, one shot, no history."""
    return {
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "stream": False,
    }


def parse_chat_response(payload: dict) -> str:
    """Pull the assistant text out of a ``/v1/chat/completions`` response."""
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RewriterError(f"unexpected llama-server response: {payload!r}"[:400]) from exc
    message = choice.get("message") or {}
    text = (message.get("content") or "").strip()
    if not text:
        raise RewriterError("llama-server returned an empty response")
    return text


def parse_loader_path(output: str) -> str:
    """Extract the ``LD_LIBRARY_PATH`` value printed by ``cuda_loader_path.sh``."""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("export LD_LIBRARY_PATH="):
            return line.split("=", 1)[1].strip()
    return ""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class LlamaCppRewriter:
    """Rewrite text with a GGUF model served by ``llama-server``."""

    model_path: Path
    server_bin: Path = DEFAULT_SERVER_BIN
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ctx_size: int = DEFAULT_CTX_SIZE
    n_gpu_layers: int = 99
    threads: int = 0  # 0 = llama.cpp default
    startup_timeout: float = 300.0
    request_timeout: float = 600.0
    port: int = 0  # 0 = pick a free port per run
    _proc: object = field(default=None, repr=False, compare=False)
    _port: int = field(default=0, repr=False, compare=False)
    _log_path: Path | None = field(default=None, repr=False, compare=False)
    _last_run: dict[str, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.server_bin = Path(self.server_bin)

    @property
    def name(self) -> str:
        return f"gguf:{self.model_path.name}"

    @property
    def prefetch_model_id(self) -> None:
        """GGUF weights are not fetched through ``huggingface_hub``."""
        return None

    # -- process management ----------------------------------------------
    def _child_env(self) -> dict[str, str]:
        """Environment for the server, with CUDA runtime libs on the path."""
        env = dict(os.environ)
        helper = REPO_ROOT / "cuda_loader_path.sh"
        if helper.exists():
            try:
                out = subprocess.run(
                    ["bash", str(helper)],
                    capture_output=True, text=True, timeout=60, check=False,
                ).stdout
                libs = parse_loader_path(out)
                if libs:
                    env["LD_LIBRARY_PATH"] = libs
            except (OSError, subprocess.SubprocessError):
                pass  # the binary may not need it (same CUDA version)
        return env

    def _server_args(self, port: int) -> list[str]:
        args = [
            str(self.server_bin),
            "-m", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx_size),
            "--jinja",  # use the GGUF's own chat template
        ]
        if self.threads > 0:
            args += ["-t", str(self.threads)]
        return args

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.server_bin.exists():
            raise RewriterError(
                f"llama-server not found at {self.server_bin}; run "
                "./plumbing/fetch_llamacpp_text.sh to stage it"
            )
        if not self.model_path.exists():
            raise RewriterError(
                f"GGUF model not found at {self.model_path}; download it with "
                "'hf download <repo> <file>.gguf --local-dir models'"
            )

        port = self.port or _free_port()
        self._log_path = self.model_path.with_suffix(
            self.model_path.suffix + ".server.log"
        )
        log = self._log_path.open("w", encoding="utf-8")
        try:
            self._proc = subprocess.Popen(
                self._server_args(port),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self._child_env(),
                start_new_session=True,
            )
        finally:
            log.close()
        self._port = port
        try:
            self._wait_ready(port)
        except RewriterError:
            self.unload()  # do not leave a half-started server holding VRAM
            raise

    def _wait_ready(self, port: int) -> None:
        deadline = time.time() + self.startup_timeout
        url = f"http://127.0.0.1:{port}/health"
        while time.time() < deadline:
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                raise RewriterError(
                    "llama-server exited during startup "
                    f"(code {proc.returncode}): {self._log_tail()}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(0.5)
        raise RewriterError(
            f"llama-server did not become ready in {self.startup_timeout:.0f}s: "
            f"{self._log_tail()}"
        )

    def _log_tail(self, lines: int = 20) -> str:
        if self._log_path is None or not self._log_path.exists():
            return "(no log)"
        text = self._log_path.read_text(encoding="utf-8", errors="replace")
        return " | ".join(text.strip().splitlines()[-lines:])[:1200]

    # -- generation ------------------------------------------------------
    def rewrite(self, messages: list[dict[str, str]]) -> str:
        """Generate one rewrite from a fresh message list (no history)."""
        if not messages:
            raise RewriterError("empty message list")
        self._start()
        body = json.dumps(build_chat_payload(messages, self.max_new_tokens)).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RewriterError(
                f"llama-server HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RewriterError(f"llama-server request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RewriterError(f"llama-server returned invalid JSON: {exc}") from exc

        text = parse_chat_response(payload)
        self._last_run = {
            "model": self.name,
            "prompt_version": REWRITE_PROMPT_VERSION,
            "quantization": "gguf",
            "input_words": len(messages[-1]["content"].split()),
            "output_words": len(text.split()),
            "seconds": round(time.time() - started, 2),
        }
        return text

    @property
    def last_run(self) -> dict[str, object]:
        """Metadata from the most recent rewrite."""
        return dict(self._last_run)

    def unload(self) -> None:
        """Stop the server process and free the GPU."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                pass
        except OSError:  # pragma: no cover - already gone
            pass
