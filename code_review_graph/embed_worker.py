"""Embedding worker process for Windows.

Loads the sentence-transformers model on the process **main thread** and
serves embedding requests over a JSON-lines stdio protocol.

Why a separate process: on Windows, importing numpy/torch C extensions from
any non-main thread deadlocks on the OS DLL Loader Lock — the loading thread
holds the lock while torch's import machinery spawns helper threads whose
``DllMain(DLL_THREAD_ATTACH)`` callbacks need that same lock. This was 100%
reproducible in the MCP server regardless of pre-warm strategy (see #46,
#136). Loading on a dedicated process's main thread is the only arrangement
that reliably works, so the MCP server process never imports
sentence-transformers / torch / numpy at all and delegates to this worker.

Protocol (one JSON object per line):

    -> (on startup)            {"ready": true, "dimension": 384, "model": "..."}
       or on failure           {"error": "..."} followed by exit code 1
    <- {"op": "embed", "texts": [...]}
    -> {"vectors": [[...], ...]}
    <- {"op": "ping"}
    -> {"ok": true}

The worker exits when stdin closes (parent process gone) or on "shutdown".
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="code-review-graph embedding worker")
    parser.add_argument("--model", required=True, help="sentence-transformers model name")
    args = parser.parse_args()

    # stdout carries the protocol exclusively; route any library chatter away
    # from it and disable progress bars that could interleave with JSON lines.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    out = sys.stdout

    def send(obj: dict) -> None:
        out.write(json.dumps(obj) + "\n")
        out.flush()

    try:
        from sentence_transformers import SentenceTransformer

        _rce_val = os.environ.get("CRG_ALLOW_REMOTE_CODE", "0")
        allow_remote_code = _rce_val.lower() in ("1", "true", "yes")
        model = SentenceTransformer(args.model, trust_remote_code=allow_remote_code)
        dimension = model.get_sentence_embedding_dimension()
    except BaseException as exc:  # noqa: BLE001 - report any load failure to parent
        send({"error": f"{type(exc).__name__}: {exc}"})
        return 1

    send({"ready": True, "dimension": dimension, "model": args.model})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "embed":
                vectors = model.encode(req["texts"], show_progress_bar=False)
                send({"vectors": [v.tolist() for v in vectors]})
            elif op == "ping":
                send({"ok": True})
            elif op == "shutdown":
                break
            else:
                send({"error": f"unknown op: {op!r}"})
        except BaseException as exc:  # noqa: BLE001 - keep serving after bad requests
            send({"error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
