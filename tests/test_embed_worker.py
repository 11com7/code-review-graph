"""Tests for the embedding worker process and its client.

The worker is exercised as a real subprocess, but with a fake
``sentence_transformers`` package injected via PYTHONPATH so the tests run
without the heavy optional dependency (and without loading any C-extension
DLLs, which is the whole point of the worker).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from code_review_graph.embeddings import (
    LocalEmbeddingProvider,
    _EmbedWorkerClient,
    _use_worker_process,
)

FAKE_ST = '''
"""Fake sentence_transformers for worker tests."""


class SentenceTransformer:
    def __init__(self, model_name, trust_remote_code=False):
        if model_name == "boom":
            raise OSError("model not found")
        self.model_name = model_name

    def get_sentence_embedding_dimension(self):
        return 4

    class _Vec(list):
        def tolist(self):
            return list(self)

    def encode(self, texts, show_progress_bar=False):
        return [self._Vec([float(len(t)), 1.0, 2.0, 3.0]) for t in texts]
'''


@pytest.fixture()
def fake_st_env(tmp_path: Path) -> dict:
    """Environment whose PYTHONPATH provides a fake sentence_transformers."""
    pkg = tmp_path / "sentence_transformers"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(FAKE_ST, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHOME"}
    env["PYTHONPATH"] = str(tmp_path)
    return env


def _worker_cmd(model: str) -> list[str]:
    return [sys.executable, "-m", "code_review_graph.embed_worker", "--model", model]


class TestEmbedWorkerProtocol:
    def test_ready_then_embed(self, fake_st_env):
        proc = subprocess.Popen(
            _worker_cmd("test-model"),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", env=fake_st_env,
        )
        try:
            ready = json.loads(proc.stdout.readline())
            assert ready == {"ready": True, "dimension": 4, "model": "test-model"}

            proc.stdin.write(json.dumps({"op": "embed", "texts": ["ab", "c"]}) + "\n")
            proc.stdin.flush()
            resp = json.loads(proc.stdout.readline())
            assert resp["vectors"] == [[2.0, 1.0, 2.0, 3.0], [1.0, 1.0, 2.0, 3.0]]

            proc.stdin.write(json.dumps({"op": "ping"}) + "\n")
            proc.stdin.flush()
            assert json.loads(proc.stdout.readline()) == {"ok": True}
        finally:
            proc.kill()

    def test_load_failure_reports_error(self, fake_st_env):
        proc = subprocess.Popen(
            _worker_cmd("boom"),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", env=fake_st_env,
        )
        try:
            msg = json.loads(proc.stdout.readline())
            assert "error" in msg
            assert "model not found" in msg["error"]
            assert proc.wait(timeout=10) == 1
        finally:
            proc.kill()

    def test_exits_when_stdin_closes(self, fake_st_env):
        proc = subprocess.Popen(
            _worker_cmd("test-model"),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", env=fake_st_env,
        )
        proc.stdout.readline()  # ready line
        proc.stdin.close()
        assert proc.wait(timeout=10) == 0


class TestEmbedWorkerClient:
    def _client(self, fake_st_env, model="test-model"):
        return _EmbedWorkerClient(model, cmd=_worker_cmd(model), env=fake_st_env)

    def test_embed_roundtrip(self, fake_st_env):
        client = self._client(fake_st_env)
        assert client.embed(["xyz"]) == [[3.0, 1.0, 2.0, 3.0]]
        assert client.dimension() == 4

    def test_start_is_non_blocking_and_idempotent(self, fake_st_env):
        client = self._client(fake_st_env)
        client.start()
        client.start()
        assert client.embed(["a", "bb"]) == [
            [1.0, 1.0, 2.0, 3.0], [2.0, 1.0, 2.0, 3.0],
        ]

    def test_load_failure_raises(self, fake_st_env):
        client = self._client(fake_st_env, model="boom")
        with pytest.raises(RuntimeError, match="model not found"):
            client.embed(["a"])

    def test_restart_after_worker_death(self, fake_st_env):
        client = self._client(fake_st_env)
        assert client.embed(["a"]) == [[1.0, 1.0, 2.0, 3.0]]
        client._proc.kill()
        client._proc.wait(timeout=10)
        assert client.embed(["bb"]) == [[2.0, 1.0, 2.0, 3.0]]


class TestUseWorkerProcess:
    def test_default_matches_platform(self, monkeypatch):
        monkeypatch.delenv("CRG_EMBED_WORKER", raising=False)
        assert _use_worker_process() == (sys.platform == "win32")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CRG_EMBED_WORKER", "1")
        assert _use_worker_process() is True
        monkeypatch.setenv("CRG_EMBED_WORKER", "0")
        assert _use_worker_process() is False


class TestProviderRouting:
    def test_provider_uses_worker_when_enabled(self, fake_st_env, monkeypatch):
        import code_review_graph.embeddings as emb

        monkeypatch.setenv("CRG_EMBED_WORKER", "1")

        provider = LocalEmbeddingProvider(model_name="test-model")
        # Patch the registry so the worker spawns with the fake package env.
        client = _EmbedWorkerClient(
            "test-model", cmd=_worker_cmd("test-model"), env=fake_st_env,
        )
        monkeypatch.setattr(emb, "_get_worker_client", lambda name: client)

        assert provider.embed(["hello"]) == [[5.0, 1.0, 2.0, 3.0]]
        assert provider.embed_query("hi") == [2.0, 1.0, 2.0, 3.0]
        assert provider.dimension == 4
