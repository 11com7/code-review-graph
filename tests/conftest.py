"""Shared test configuration."""

import os

# Force the in-process embedding path during tests. On Windows the production
# default routes LocalEmbeddingProvider through an embed_worker subprocess
# (see embeddings._use_worker_process), which would make every embedding test
# spawn a real worker loading the real model — slow and memory-hungry. Tests
# that exercise the worker explicitly (test_embed_worker.py) construct
# _EmbedWorkerClient directly or override this variable.
os.environ.setdefault("CRG_EMBED_WORKER", "0")
