from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = "/home/alexander/Documents/ai-fund-manager"

BEFORE = f"""MAILTO=""
# DeepSwing
*/15 9-17 * * 1-5 cd /home/alexander/Documents/DeepSwing && .venv/bin/deepswing tick >> logs/tick.log 2>&1
0 6 * * * cd /home/alexander && python -m financedata.universe refresh >> fd.log 2>&1
# Fund
30 9 * * 1 cd {REPO} && .venv/bin/fund run >> data/logs/weekly.log 2>&1
# 0 9 * * 1 cd {REPO} && .venv/bin/fund paper-watch >> data/logs/x.log 2>&1
0 3 * * * cd {REPO} && deploy/backup.sh >> data/logs/backup.log 2>&1
# >>> fundmgr managed (deploy/managed_cron.sh) — edit there, not here >>>
# Broker accounts
*/30 9-22 * * 1-5 cd {REPO} && .venv/bin/fund account-sync --all >> data/logs/accounts.log 2>&1
# <<< fundmgr managed <<<
"""


@pytest.fixture
def pi(tmp_path):
    repo = tmp_path / "ai-fund-manager"
    (repo / "deploy").mkdir(parents=True)
    for name in ("pi-crontab.sh", "cron.pi-watchdog"):
        shutil.copy(ROOT / "deploy" / name, repo / "deploy" / name)
    table, binaries = tmp_path / "table", tmp_path / "bin"
    binaries.mkdir()
    (binaries / "crontab").write_text(
        "#!/bin/sh\n"
        f'T="{table}"\n'
        'case "$1" in -l) [ -f "$T" ] && cat "$T" || exit 1;; -) cat > "$T";; *) cp "$1" "$T";; esac\n')
    (binaries / "crontab").chmod(0o755)
    table.write_text(BEFORE.replace(REPO, str(repo)))
    env = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "HOME": str(tmp_path),
           "FUND_DIR": str(repo)}

    def run(cmd: str) -> str:
        return subprocess.run(["bash", str(repo / "deploy" / "pi-crontab.sh"), cmd], env=env,
                              check=True, capture_output=True, text=True).stdout
    return run, table, tmp_path, repo


def test_strip_removes_only_the_funds_jobs(pi):
    run, table, home, _ = pi
    out = run("strip")
    after = table.read_text()
    assert "DeepSwing" in after and "financedata.universe refresh" in after and 'MAILTO=""' in after
    assert "fund " not in after and "backup.sh" not in after and "fundmgr managed" not in after
    assert "removed 2 fund job(s); kept 2 other job(s)" in out
    assert (home / "crontab.before-mac").read_text().strip() == BEFORE.replace(
        REPO, str(pi[3])).strip()
    run("strip")  # a second strip never overwrites the saved original
    assert "fund run" in (home / "crontab.before-mac").read_text()


def test_watchdog_is_added_once_beside_the_rest_and_restore_undoes_it_all(pi):
    run, table, _, repo = pi
    run("strip")
    run("watchdog")
    run("watchdog")
    after = table.read_text()
    assert after.count("# >>> fundmgr watchdog") == 1
    assert after.count("deploy/watchdog.sh") == 3
    assert f"cd {repo} && deploy/watchdog.sh" in after and "DeepSwing" in after
    run("restore")
    assert "fund run" in table.read_text() and "watchdog.sh" not in table.read_text()
