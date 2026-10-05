from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=None, env=None):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=True)


@pytest.fixture
def deployment(tmp_path):
    origin, repo, binaries = (tmp_path / n for n in ("origin", "repo", "bin"))
    run("git", "init", "--bare", "-q", str(origin))
    run("git", "clone", "-q", str(origin), str(repo))
    run("git", "config", "user.email", "test@example.com", cwd=repo)
    run("git", "config", "user.name", "Test", cwd=repo)
    run("git", "checkout", "-b", "deploy", cwd=repo)
    (repo / "deploy").mkdir()
    for name in ("deploy.sh", "poll-deploy.sh"):
        shutil.copy(ROOT / "deploy" / name, repo / "deploy" / name)
    (repo / ".gitignore").write_text("data/\n")
    run("git", "add", ".", cwd=repo)
    run("git", "commit", "-qm", "test fixture", cwd=repo)
    run("git", "push", "-qu", "origin", "deploy", cwd=repo)
    binaries.mkdir()
    scripts = {
        "flock": "exit 0",  # macOS lacks flock; this test covers retry state.
        "pgrep": 'test -f "$FUND_DIR/active-run"',
        "sleep": "exit 0",
        "uv": 'echo install >> "$FUND_DIR/data/events"\ntest ! -f "$FUND_DIR/fail-install"',
        "sudo": 'echo restart >> "$FUND_DIR/data/events"\ntest ! -f "$FUND_DIR/fail-restart"',
        "systemctl": 'test ! -f "$FUND_DIR/fail-health"',
    }
    for name, body in scripts.items():
        path = binaries / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
    (tmp_path / "FinanceData").mkdir()
    env = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "FUND_DIR": str(repo),
           "FINANCEDATA_DIR": str(tmp_path / "FinanceData")}
    return repo, env


@pytest.mark.parametrize("failure", ["fail-install", "fail-restart", "fail-health"])
def test_same_revision_retries_until_success(deployment, failure):
    repo, env = deployment
    marker = repo / "data/deployed-revision"
    fault = repo / failure
    fault.touch()
    first = subprocess.run(["bash", str(repo / "deploy/deploy.sh")], env=env, capture_output=True)
    assert first.returncode != 0
    assert not marker.exists()
    fault.unlink()
    # Polling also retries when HEAD already equals origin/deploy.
    run("bash", str(repo / "deploy/poll-deploy.sh"), env=env)
    assert marker.read_text().strip() == run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    events = (repo / "data/events").read_text()
    run("bash", str(repo / "deploy/deploy.sh"), env=env)
    assert (repo / "data/events").read_text() == events


def test_busy_fund_aborts_before_installing(deployment):
    repo, env = deployment
    (repo / "active-run").touch()
    result = subprocess.run(["bash", str(repo / "deploy/deploy.sh")], env=env, capture_output=True)
    assert result.returncode != 0
    assert not (repo / "data/events").exists()
    assert not (repo / "data/deployed-revision").exists()


def test_deploy_ref_must_match_tested_revision(deployment):
    repo, env = deployment
    env["EXPECTED_REVISION"] = "0" * 40
    result = subprocess.run(["bash", str(repo / "deploy/deploy.sh")], env=env, capture_output=True)
    assert result.returncode != 0
    assert not (repo / "data/events").exists()
    assert not (repo / "data/deployed-revision").exists()
