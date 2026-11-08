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


def test_macos_restarts_the_launchd_agents(deployment):
    repo, env = deployment
    binaries = Path(env["PATH"].split(":")[0])
    for name, body in {
        "uname": "echo Darwin",
        "launchctl": 'echo "launchctl $*" >> "$FUND_DIR/data/events"\n'
                     'case "$1" in print) test ! -f "$FUND_DIR/fail-health" && echo "state = running";; esac',
    }.items():
        (binaries / name).write_text("#!/bin/sh\n" + body + "\n")
        (binaries / name).chmod(0o755)
    run("bash", str(repo / "deploy/deploy.sh"), env=env)
    events = (repo / "data/events").read_text()
    assert "restart" not in events  # no sudo systemctl on a Mac
    for svc in ("bot", "web", "global-web"):
        assert f"kickstart -k gui/{os.getuid()}/com.fundmgr.{svc}" in events
    assert (repo / "data/deployed-revision").exists()

    (repo / "data/deployed-revision").unlink()
    (repo / "fail-health").touch()
    failed = subprocess.run(["bash", str(repo / "deploy/deploy.sh")], env=env,
                            capture_output=True, check=False)
    assert failed.returncode != 0
    assert not (repo / "data/deployed-revision").exists()


def test_a_changed_deploy_script_runs_its_new_steps_in_the_same_deploy(deployment):
    """git reset swaps deploy.sh for a new file while bash still reads the old
    one, so a step added to the script used to skip the deploy that shipped it."""
    repo, env = deployment
    script = repo / "deploy/deploy.sh"
    old = script.read_text()
    marker = 'log "Restarting services…"'
    script.write_text(old.replace(marker, 'echo new-step >> "$FUND_DIR/data/events"\n' + marker))
    run("git", "commit", "-qam", "add a deploy step", cwd=repo)
    run("git", "push", "-q", "origin", "deploy", cwd=repo)
    run("git", "reset", "-q", "--hard", "HEAD~1", cwd=repo)          # the Pi, one deploy behind
    assert script.read_text() == old

    run("bash", str(script), env=env)
    events = (repo / "data/events").read_text().split()
    assert events.count("new-step") == 1
    assert events.count("install") == 1          # one deploy, not two
    assert (repo / "data/deployed-revision").read_text().strip() == \
        run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()


def test_a_retired_host_refuses_every_deploy(deployment):
    repo, env = deployment
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data/moved-to-mac").write_text("2026-11-14, mac-studio")
    result = subprocess.run(["bash", str(repo / "deploy/deploy.sh")], env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "retired" in (repo / "data/logs/deploy.log").read_text()
    assert not (repo / "data/events").exists()  # nothing installed, nothing restarted
