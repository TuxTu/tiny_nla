"""SGLang-based RL rollout — batched embedding generation with injection.

Replaces the sequential HF generate() loop with SGLang's continuous batching
server.  The server runs alongside the trainer on the same GPU, time-sharing:

  1. Launch SGLang server on GPU
  2. Pre-compute input embeddings (with injection vector) for all prompts
  3. POST all embeddings to SGLang → batched, continuous-batching generation
  4. Collect responses, shutdown server, free VRAM
  5. PyTorch training step (policy gradient + critic update)
  6. Weight sync: save checkpoint → SGLang picks up fresh weights next step

SGLang v0.3+ supports input_embeds natively via the /generate endpoint when
launched with --disable-radix-cache.  No C++ patches needed.

Usage:
    rollout = SGLangRollout(
        model_path="checkpoints/actor_sft",
        mem_fraction=0.7,   # leave 30% VRAM for critic + training
    )
    with rollout:
        responses = rollout.generate(
            input_embeds_list=[emb1, emb2, ...],  # pre-injected
            max_new_tokens=300,
        )
    # Server automatically stopped; VRAM freed for training
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

_DEFAULT_PORT = 30000
_HEALTH_TIMEOUT = 120  # seconds to wait for server startup
_GENERATE_TIMEOUT = 300  # seconds per generate call


# ---------------------------------------------------------------------------
# SGLang server lifecycle
# ---------------------------------------------------------------------------


class SGLangRollout:
    """Manage an SGLang server for embedding-based RL rollout.

    Server is launched as a subprocess pointing at a model checkpoint.
    After each training step, save the updated actor and relaunch to pick
    up fresh weights.  For 0.6B this adds ~3 s overhead per step; for 4B
    it's ~10 s — negligible compared to the generation-time savings.
    """

    def __init__(
        self,
        model_path: str,
        *,
        mem_fraction: float = 0.70,
        port: int = _DEFAULT_PORT,
        device: str = "cuda",
        gpu_id: int | None = None,
        log_dir: str | None = None,
    ):
        # Resolve local paths but keep HF repo names (e.g. "org/model") as-is
        p = Path(model_path)
        if p.exists():
            self.model_path = str(p.resolve())
        else:
            self.model_path = model_path
        self.mem_fraction = mem_fraction
        self.port = port
        self.device = device
        self.gpu_id = gpu_id
        self.log_dir = log_dir
        self._process: subprocess.Popen | None = None
        self._base_url = f"http://localhost:{port}"

    # -- start / stop ---------------------------------------------------------

    def start(self) -> None:
        """Launch SGLang server and block until healthy."""
        if self._process is not None:
            return  # already running

        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--mem-fraction-static", str(self.mem_fraction),
            "--disable-radix-cache",   # required for input_embeds
            "--trust-remote-code",
        ]
        env = os.environ.copy()
        # Determine which GPU SGLang should use:
        #   gpu_id=N  → CUDA_VISIBLE_DEVICES=N
        #   device="cuda:N"  → extract N
        #   otherwise keep existing CUDA_VISIBLE_DEVICES
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        elif self.device.startswith("cuda:") and self.device != "cuda":
            env["CUDA_VISIBLE_DEVICES"] = self.device.split(":")[1]
        else:
            env.setdefault("CUDA_VISIBLE_DEVICES", "0" if self.device == "cuda" else "")

        if self.log_dir is None:
            self.log_dir = str(Path(self.model_path).parent / "sglang_logs")
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        log_file = open(Path(self.log_dir) / "sglang_server.log", "w")
        self._process = subprocess.Popen(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        )
        self._wait_ready()

    def wait_ready(self) -> None:
        """Public entry point for externally-managed SGLang. Just poll /health."""
        self._wait_ready()

    def _wait_ready(self) -> None:
        """Poll /health until the server responds or timeout."""
        import urllib.request

        deadline = time.time() + _HEALTH_TIMEOUT
        last_err = None
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                # Server exited — tail the log
                tail = self._tail_log(20)
                raise RuntimeError(
                    f"SGLang server exited with code {self._process.returncode}.\n"
                    f"Last log lines:\n{tail}"
                )
            try:
                req = urllib.request.urlopen(
                    f"{self._base_url}/health", timeout=2
                )
                if req.status == 200:
                    # Give it a moment to finish model warmup
                    time.sleep(2)
                    return
            except Exception as e:
                last_err = e
                time.sleep(1)
        raise RuntimeError(
            f"SGLang did not become healthy within {_HEALTH_TIMEOUT}s. "
            f"Last error: {last_err}"
        )

    def stop(self) -> None:
        """Gracefully stop the SGLang server and free GPU memory."""
        if self._process is None:
            return
        try:
            self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None
        # Force CUDA to release any cached memory SGLang held
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _tail_log(self, n_lines: int = 20) -> str:
        """Return last N lines of the server log (for error reporting)."""
        if self.log_dir is None:
            return ""
        log_path = Path(self.log_dir) / "sglang_server.log"
        if not log_path.exists():
            return "(no log file)"
        lines = log_path.read_text().splitlines()
        return "\n".join(lines[-n_lines:])

    # -- context manager ------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        return False

    # -- weight sync ----------------------------------------------------------

    def update_weights(self, model_path: str | None = None) -> None:
        """Hot-reload model weights from disk without restarting the server.

        Calls SGLang's ``/update_weights_from_disk`` endpoint.  This avoids the
        ~3-10s server restart overhead per training step — the server stays
        alive and continuous-batching cache is preserved.

        Parameters
        ----------
        model_path :
            Path to updated checkpoint.  Defaults to the original model_path
            the server was launched with.
        """
        import urllib.request

        target = str(Path(model_path or self.model_path).resolve())
        body = json.dumps({"model_path": target}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/update_weights_from_disk",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            if not result.get("success", False):
                print(f"[SGLang] update_weights failed: {result}")
        except Exception as e:
            print(f"[SGLang] update_weights error: {e}")

    # -- generation -----------------------------------------------------------

    def generate(
        self,
        input_embeds_list: list[torch.Tensor],
        *,
        max_new_tokens: int = 300,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_concurrent: int = 16,
    ) -> list[str]:
        """Generate text from pre-computed input embeddings.

        Each embedding in the list produces one response.  Embeddings MUST
        already have the activation vector injected at the marker position
        (use ``inject_at_marked_positions`` before calling).

        Requests are sent concurrently (ThreadPoolExecutor) so SGLang's
        continuous batching can interleave generation across the batch.

        Parameters
        ----------
        input_embeds_list :
            List of [1, seq_len, d_model] embeddings, one per prompt.
        max_new_tokens :
            Maximum tokens to generate per sequence.
        temperature, top_p :
            Sampling parameters passed directly to SGLang.
        max_concurrent :
            Max concurrent HTTP requests to SGLang (default 16).

        Returns
        -------
        List of decoded response strings, same length as input_embeds_list.
        """
        import urllib.request

        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_token_ids": [],  # SGLang will use EOS by default
        }

        # Pre-serialize all embeddings on the main thread, then fire
        # concurrent HTTP requests so SGLang continuous-batches them.
        payloads: list[bytes] = []
        for i, embeds in enumerate(input_embeds_list):
            assert embeds.ndim == 3, (
                f"input_embeds[{i}] must be [1, seq_len, d_model], "
                f"got shape {tuple(embeds.shape)}"
            )
            assert embeds.shape[0] == 1, (
                f"input_embeds[{i}] batch dim must be 1, got {embeds.shape[0]}"
            )
            buf = io.BytesIO()
            torch.save(embeds.cpu(), buf)
            payload_b64 = _encode_bytes(buf.getvalue())
            body = json.dumps({
                "input_embeds": payload_b64,
                "input_embeds_shape": list(embeds.shape),
                "sampling_params": sampling_params,
            }).encode("utf-8")
            payloads.append(body)

        # Concurrent requests — SGLang receives them near-simultaneously
        # and continuous-batches across the entire batch.
        results: dict[int, str] = {}

        def _send_one(index: int, body: bytes) -> tuple[int, str]:
            req = urllib.request.Request(
                f"{self._base_url}/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=_GENERATE_TIMEOUT) as resp:
                    result = json.loads(resp.read())
                return index, result.get("text", "")
            except Exception as e:
                print(f"[SGLang] generate failed for prompt {index}: {e}")
                return index, ""

        with ThreadPoolExecutor(max_workers=min(max_concurrent, len(payloads))) as pool:
            futures = [
                pool.submit(_send_one, i, body)
                for i, body in enumerate(payloads)
            ]
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text

        # Preserve input order
        return [results[i] for i in range(len(input_embeds_list))]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _encode_bytes(data: bytes) -> str:
    """Base64-encode binary data, returning a plain ASCII string."""
    import base64
    return base64.b64encode(data).decode("ascii")


def _decode_bytes(encoded: str) -> bytes:
    """Decode a base64 string back to bytes."""
    import base64
    return base64.b64decode(encoded)


def prepare_embeddings(
    prompt_ids: torch.Tensor,       # [1, seq_len]
    vector: torch.Tensor,            # [d_model]
    embed_layer: torch.nn.Module,
    *,
    inj_token_id: int,
    left_neighbor_id: int,
    right_neighbor_id: int,
    injection_scale: float | None = None,
) -> torch.Tensor:
    """Compute input embeddings with activation vector injected.

    This is the preprocessing step done BEFORE sending to SGLang.
    The resulting embedding tensor can be serialized and sent as-is.

    Parameters
    ----------
    prompt_ids : [1, seq_len]
        Tokenized prompt (should contain the injection character).
    vector : [d_model]
        Raw activation vector.
    embed_layer :
        Model's token embedding layer (model.get_input_embeddings()).
    inj_token_id, left_neighbor_id, right_neighbor_id :
        Token metadata from the sidecar.
    injection_scale :
        Passed to normalize_activation (None = raw, mean = scaled).

    Returns
    -------
    input_embeds : [1, seq_len, d_model]
        Embedding tensor with vector injected at the marker position.
    """
    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import normalize_activation

    d_model = embed_layer.weight.shape[1]

    # Normalize if needed
    if injection_scale is not None:
        vector = normalize_activation(vector.unsqueeze(0), injection_scale).squeeze(0)

    # Compute base embeddings on CPU, then inject
    # Keep on CPU until injection to avoid double GPU memory
    with torch.no_grad():
        embeddings = embed_layer(prompt_ids)  # [1, seq_len, d]
        embeddings = inject_at_marked_positions(
            prompt_ids, embeddings,
            vector.unsqueeze(0).to(embeddings.device, embeddings.dtype),
            inj_token_id, left_neighbor_id, right_neighbor_id,
        )

    return embeddings.detach()


def prepare_batch_embeddings(
    tokenizer,
    prompts: list[list[dict]],   # messages format
    vectors: torch.Tensor,        # [B, d_model]
    embed_layer: torch.nn.Module,
    *,
    injection_char: str,
    inj_token_id: int,
    left_neighbor_id: int,
    right_neighbor_id: int,
    injection_scale: float | None = None,
    max_length: int = 2048,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Tokenize prompts, compute embeddings, inject vectors — all batched.

    Returns a list of [1, seq_len, d_model] tensors ready for SGLang.
    """
    from nla.training.injection import inject_at_marked_positions
    from nla.training.schema import normalize_activation

    # Swap placeholder for real injection char
    prompt_texts = [
        tokenizer.apply_chat_template(
            [
                {"role": m["role"], "content": m["content"].replace(
                    "<INJECT>", injection_char
                )}
                for m in msgs
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for msgs in prompts
    ]

    results: list[torch.Tensor] = []
    for i, (ptext, vec) in enumerate(zip(prompt_texts, vectors)):
        enc = tokenizer(ptext, return_tensors="pt", truncation=True,
                        max_length=max_length)
        p_ids = enc["input_ids"].to(device)

        v = vec.to(device)
        if injection_scale is not None:
            v = normalize_activation(v.unsqueeze(0), injection_scale).squeeze(0)

        with torch.no_grad():
            embs = embed_layer(p_ids)  # [1, seq, d]
            embs = inject_at_marked_positions(
                p_ids, embs, v.unsqueeze(0),
                inj_token_id, left_neighbor_id, right_neighbor_id,
            )

        results.append(embs.cpu())

    return results
