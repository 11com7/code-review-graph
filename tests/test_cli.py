"""Tests for CLI helpers and MCP serve command wiring."""

import logging
import sys
from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock, patch

from code_review_graph import cli


def test_get_version_logs_and_falls_back_to_dev(monkeypatch, caplog):
    def _raise_package_not_found(_dist_name: str) -> str:
        raise PackageNotFoundError("code-review-graph")

    monkeypatch.setattr(cli, "pkg_version", _raise_package_not_found)

    with caplog.at_level(logging.DEBUG, logger="code_review_graph.cli"):
        version = cli._get_version()

    assert version == "dev"
    assert "Package metadata unavailable" in caplog.text


class TestUpdateNoGitExitsZero:
    """Regression tests for #312: running ``update`` or ``detect-changes``
    in a directory with no git repository must exit 0 (with a warning
    to stderr) so Claude Code's PostToolUse hook does not report a
    failure on every Write / Edit / Bash tool call in monorepos where
    the workspace root has no ``.git``.

    We mock ``find_repo_root`` to return ``None`` explicitly so these
    tests do not depend on the test runner's ambient git hierarchy
    (e.g. a ``.git`` directory in the user's home, which would make the
    unbounded ancestor walk find it and skip the no-git branch we want
    to test — same hazard addressed by #241's ``stop_at`` parameter).
    """

    def _invoke(self, command: str, capsys, monkeypatch):
        """Drive ``cli.main`` through the no-git branch by forcing
        ``find_repo_root`` to return ``None``, and capture the stderr
        warning + exit code."""
        import pytest as _pytest

        monkeypatch.setattr(
            "code_review_graph.incremental.find_repo_root",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(sys, "argv", ["code-review-graph", command])
        with _pytest.raises(SystemExit) as excinfo:
            cli.main()
        captured = capsys.readouterr()
        return excinfo.value.code, captured.out, captured.err

    def test_update_exits_zero_without_git(self, capsys, monkeypatch):
        """Before #312 this exited 1, causing
        ``PostToolUse:Edit hook error`` noise on every tool call."""
        code, _out, err = self._invoke("update", capsys, monkeypatch)
        assert code == 0, f"expected exit 0, got {code}; stderr: {err!r}"

    def test_update_still_warns_about_missing_git(self, capsys, monkeypatch):
        """Exit 0 must not be silent — an interactive user still gets
        told why the update did nothing.  The warning goes to stderr so
        MCP stdio transport is not corrupted."""
        _code, out, err = self._invoke("update", capsys, monkeypatch)
        # Warning must be visible in stderr (hook/MCP-safe location).
        assert "git" in err.lower(), (
            f"expected a 'git' hint in stderr; got stdout={out!r} stderr={err!r}"
        )
        # And stdout must NOT contain the warning (would corrupt MCP stdio).
        assert "git" not in out.lower() or "not in a git" not in out.lower()

    def test_detect_changes_also_exits_zero_without_git(
        self, capsys, monkeypatch,
    ):
        """Same non-failing semantics for the sibling ``detect-changes``
        subcommand — otherwise hooks that wrap it get the same error."""
        code, _out, err = self._invoke("detect-changes", capsys, monkeypatch)
        assert code == 0, (
            f"expected exit 0, got {code}; stderr: {err!r}"
        )


class TestServeCommand:
    def test_serve_passes_auto_watch_flag(self):
        argv = [
            "code-review-graph",
            "serve",
            "--repo",
            "repo-root",
            "--auto-watch",
        ]
        with patch.object(sys, "argv", argv):
            with patch("code_review_graph.main.main") as mock_serve:
                cli.main()

        mock_serve.assert_called_once_with(
            repo_root="repo-root",
            auto_watch=True,
            tools=None,
        )

    def test_mcp_alias_maps_to_serve(self):
        argv = [
            "code-review-graph",
            "mcp",
            "--repo",
            "repo-root",
        ]
        with patch.object(sys, "argv", argv):
            with patch("code_review_graph.main.main") as mock_serve:
                cli.main()

        mock_serve.assert_called_once_with(
            repo_root="repo-root",
            auto_watch=False,
        )


class TestWatchInteraction:
    def test_watch_exits_when_lock_is_held(self):
        argv = ["code-review-graph", "watch", "--repo", "repo-root"]
        with patch.object(sys, "argv", argv):
            with patch("code_review_graph.graph.GraphStore") as mock_store:
                mock_store.return_value = MagicMock()
                with patch("code_review_graph.incremental.get_db_path") as mock_db:
                    mock_db.return_value = MagicMock()
                    with patch("code_review_graph.incremental.watch") as mock_watch:
                        mock_watch.side_effect = RuntimeError("watcher already running")
                        try:
                            cli.main()
                            assert False, "Expected SystemExit"
                        except SystemExit as exc:
                            assert exc.code == 1


class TestEmbedCommand:
    """``code-review-graph embed`` runs (re-)embedding outside the MCP
    request path, so large repos are not bottlenecked by client timeouts."""

    def setup_method(self):
        import tempfile
        from pathlib import Path

        from code_review_graph.graph import GraphStore
        from code_review_graph.parser import NodeInfo

        self.tmp_dir = tempfile.mkdtemp()
        self.root = Path(self.tmp_dir).resolve()
        (self.root / ".git").mkdir()
        (self.root / ".code-review-graph").mkdir()

        store = GraphStore(str(self.root / ".code-review-graph" / "graph.db"))
        for name in ("get_users", "authenticate"):
            store.upsert_node(NodeInfo(
                kind="Function", name=name, file_path="api.py",
                line_start=1, line_end=10, language="python",
            ), file_hash="abc123")
        store._conn.commit()
        store.close()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _embedding_rows(self):
        import sqlite3

        conn = sqlite3.connect(
            str(self.root / ".code-review-graph" / "graph.db")
        )
        try:
            return conn.execute(
                "SELECT qualified_name, provider FROM embeddings"
            ).fetchall()
        finally:
            conn.close()

    def test_embed_command_embeds_all_nodes(self, capsys):
        mock_provider = MagicMock()
        mock_provider.name = "local:test-model"
        mock_provider.embed.side_effect = (
            lambda texts: [[0.1, 0.2] for _ in texts]
        )

        argv = ["code-review-graph", "embed", "--repo", str(self.root)]
        with patch.object(sys, "argv", argv):
            with patch(
                "code_review_graph.embeddings.get_provider",
                return_value=mock_provider,
            ):
                cli.main()

        out = capsys.readouterr().out
        assert "local:test-model" in out
        rows = self._embedding_rows()
        assert len(rows) == 2
        assert all(provider == "local:test-model" for _, provider in rows)

    def test_embed_command_exits_1_when_provider_unavailable(self, capsys):
        argv = ["code-review-graph", "embed", "--repo", str(self.root)]
        with patch.object(sys, "argv", argv):
            with patch(
                "code_review_graph.embeddings.get_provider",
                return_value=None,
            ):
                try:
                    cli.main()
                    assert False, "Expected SystemExit"
                except SystemExit as exc:
                    assert exc.code == 1

