from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("migrate", ROOT / "deploy" / "migrate.py")
migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate)


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_db(path: Path, rows: int, *, wal: bool = False) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE transactions (id INTEGER PRIMARY KEY, note TEXT)")
    conn.executemany("INSERT INTO transactions (note) VALUES (?)", [(f"t{i}",) for i in range(rows)])
    conn.commit()
    return conn


def count(path: Path, table: str = "transactions") -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture
def pi(tmp_path, monkeypatch):
    """A checkout with state where the Pi keeps it, and a home with a FinanceData cache."""
    repo, home = tmp_path / "pi" / "ai-fund-manager", tmp_path / "pi-home"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "T")
    (repo / ".gitignore").write_text(".env\n/data/\n.venv/\n__pycache__/\nconfig/compiled/searches/\n")
    (repo / "config").mkdir()
    (repo / "config" / "config.yaml").write_text("db_path: data/fund.db\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "init")

    # Kept open with WAL writes not yet checkpointed: a plain copy of fund.db would lose them.
    live = make_db(repo / "data" / "fund.db", 3, wal=True)
    live.execute("INSERT INTO transactions (note) VALUES ('only in the wal')")
    live.commit()
    make_db(repo / "data" / "paper" / "kf-chokepoint.db", 2).close()
    (repo / "data" / "whatif").mkdir(parents=True)
    (repo / "data" / "whatif" / "run.json").write_text('{"buys": 4}')
    (repo / "data" / "shadow" / "fund" / "case1").mkdir(parents=True)
    (repo / "data" / "shadow" / "fund" / "case1" / "registration.json").write_text("{}")
    (repo / "data" / "ticker_aliases.json").write_text("{}")
    (repo / "data" / "logs").mkdir()
    (repo / "data" / "logs" / "weekly.log").write_text(f"ran in {repo}\n")
    (repo / "config" / "compiled" / "searches" / "fund").mkdir(parents=True)
    (repo / "config" / "compiled" / "active.json").write_text('{"instructions": "x"}')
    (repo / "config" / "compiled" / "searches" / "fund" / "abc.json").write_text("{}")
    (repo / ".env").write_text("OPENAI_API_KEY=sk-test\n")
    (repo / "data" / "montrose").mkdir()
    (repo / "data" / "montrose" / "token.json").write_text('{"refresh_token": "r"}')
    (repo / "data" / "montrose" / "instruments.json").write_text("{}")
    # Regenerable: must stay behind.
    (repo / ".venv" / "lib").mkdir(parents=True)
    (repo / ".venv" / "lib" / "big.so").write_text("x")
    (repo / "data" / "backups").mkdir()
    (repo / "data" / "backups" / "fund-backup-1.tgz").write_text("x")
    (repo / "data" / "deploy.lock").write_text("")
    (repo / "src" / "__pycache__").mkdir(parents=True)
    (repo / "src" / "__pycache__" / "m.cpython-312.pyc").write_text("x")

    make_db(home / ".financedata" / "cache.db", 5).close()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(migrate, "running_fund_processes", list)
    monkeypatch.setattr(migrate, "active_cron_lines", lambda root: [])
    yield repo, home
    live.close()


@pytest.fixture
def mac(tmp_path):
    repo, home = tmp_path / "mac" / "ai-fund-manager", tmp_path / "mac-home"
    repo.mkdir(parents=True)
    home.mkdir()
    return repo, home


def export(repo: Path, out: Path, *extra: str) -> int:
    return migrate.main(["--root", str(repo), "export", "--out", str(out), *extra])


def restore(repo: Path, home: Path, archive: Path, *extra: str) -> int:
    return migrate.main(["--root", str(repo), "restore", str(archive), "--home", str(home), *extra])


def test_round_trip_carries_every_kind_of_state_and_leaves_the_regenerable(pi, mac, tmp_path):
    (src, _), (dst, home) = pi, mac
    archive = tmp_path / "move.tgz"
    assert export(src, archive) == 0
    assert archive.stat().st_mode & 0o777 == 0o600

    assert restore(dst, home, archive) == 0
    assert count(dst / "data" / "fund.db") == 4  # the WAL-only row came along
    assert count(dst / "data" / "paper" / "kf-chokepoint.db") == 2
    assert count(home / ".financedata" / "cache.db") == 5
    for rel in ("data/whatif/run.json", "data/shadow/fund/case1/registration.json",
                "data/ticker_aliases.json", "data/logs/weekly.log",
                "config/compiled/active.json", "config/compiled/searches/fund/abc.json"):
        assert (dst / rel).read_bytes() == (src / rel).read_bytes(), rel
    assert (dst / ".env").stat().st_mode & 0o777 == 0o600
    assert (dst / "data/montrose/token.json").stat().st_mode & 0o777 == 0o600
    assert (dst / "data/montrose").stat().st_mode & 0o777 == 0o700
    assert (dst / "data/montrose/instruments.json").exists()
    for rel in (".venv/lib/big.so", "data/backups/fund-backup-1.tgz", "data/deploy.lock",
                "src/__pycache__/m.cpython-312.pyc", "config/config.yaml"):
        assert not (dst / rel).exists(), rel
    assert not list(dst.rglob("*-wal"))

    manifest = json.loads((dst / "data" / "migration" / "MANIFEST.json").read_text())
    fund = next(f for f in manifest["files"] if f["path"] == "repo/data/fund.db")
    assert fund["sqlite"] == {"integrity": "ok", "rows": {"transactions": 4}, "user_version": 0}
    assert manifest["source_root"] == str(src) and not manifest["live"]
    assert migrate.main(["--root", str(dst), "verify", "--home", str(home), "--strict"]) == 0


def test_export_refuses_while_the_fund_can_still_write(pi, tmp_path, monkeypatch, capsys):
    src, _ = pi
    monkeypatch.setattr(migrate, "running_fund_processes", lambda: ["42 python -m fundmgr.notify.telegram_bot"])
    archive = tmp_path / "move.tgz"
    assert export(src, archive) == 1
    assert not archive.exists()
    assert "telegram_bot" in capsys.readouterr().err

    assert export(src, archive, "--live") == 0
    with tarfile.open(archive) as tar:
        assert json.load(tar.extractfile(migrate.MANIFEST))["live"] is True


def test_backup_flags_leave_keys_cache_and_logs_behind(pi, tmp_path):
    src, _ = pi
    archive = tmp_path / "move.tgz"
    assert export(src, archive, "--no-secrets", "--no-financedata-cache", "--no-logs") == 0
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "repo/.env" not in names and "repo/data/logs/weekly.log" not in names
    assert "repo/data/paper/kf-chokepoint.db" in names
    assert not any(n.startswith("home/") for n in names)


def test_restore_never_overwrites_silently(pi, mac, tmp_path):
    (src, _), (dst, home) = pi, mac
    archive = tmp_path / "move.tgz"
    assert export(src, archive) == 0
    (dst / "data").mkdir()
    make_db(dst / "data" / "fund.db", 1).close()

    assert restore(dst, home, archive) == 1
    assert count(dst / "data" / "fund.db") == 1
    assert not (dst / "data" / "paper").exists()  # all or nothing

    assert restore(dst, home, archive, "--force") == 0
    assert count(dst / "data" / "fund.db") == 4
    assert count(dst / "data" / "fund.db.pre-migration") == 1


def test_a_damaged_archive_restores_nothing(pi, mac, tmp_path):
    (src, _), (dst, home) = pi, mac
    good, bad = tmp_path / "good.tgz", tmp_path / "bad.tgz"
    assert export(src, good) == 0
    with tarfile.open(good) as tin, tarfile.open(bad, "w:gz") as tout:
        for m in tin.getmembers():
            data = tin.extractfile(m).read()
            if m.name == "repo/data/whatif/run.json":
                data = b'{"buys": 5}'
                m.size = len(data)
            tout.addfile(m, io.BytesIO(data))
    assert restore(dst, home, bad) == 1
    assert not (dst / "data").exists()


def test_unsafe_members_are_refused(mac, tmp_path):
    dst, home = mac
    bad = tmp_path / "evil.tgz"
    with tarfile.open(bad, "w:gz") as tar:
        info = tarfile.TarInfo("../outside.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        restore(dst, home, bad)
    assert not (tmp_path / "mac" / "outside.txt").exists()


def test_later_verify_tolerates_a_running_fund_but_not_loss(pi, mac, tmp_path, capsys):
    (src, _), (dst, home) = pi, mac
    archive = tmp_path / "move.tgz"
    assert export(src, archive) == 0
    assert restore(dst, home, archive) == 0
    verify = ["--root", str(dst), "verify", "--home", str(home)]

    conn = sqlite3.connect(dst / "data" / "fund.db")
    conn.execute("DELETE FROM transactions WHERE id = 1")
    conn.commit()
    conn.close()
    (dst / "data" / "logs" / "weekly.log").write_text("more\n")
    capsys.readouterr()
    assert migrate.main(verify) == 0
    out = capsys.readouterr().out
    assert "fewer rows: transactions 4→3" in out and "weekly.log" in out
    assert migrate.main([*verify, "--strict"]) == 1

    (dst / "data" / "whatif" / "run.json").unlink()
    assert migrate.main(verify) == 1


def test_uncommitted_edits_to_tracked_files_travel_as_a_patch(pi, mac, tmp_path):
    (src, _), (dst, home) = pi, mac
    (src / "config" / "config.yaml").write_text("db_path: data/fund.db\ntop_n: 80\n")
    archive = tmp_path / "move.tgz"
    assert export(src, archive) == 0
    assert restore(dst, home, archive) == 0
    assert "+top_n: 80" in (dst / "data" / "migration" / migrate.PATCH).read_text()


def test_backup_script_takes_a_live_export_without_secrets(pi, mac):
    (src, _), (dst, home) = pi, mac
    import shutil

    (src / "deploy").mkdir()
    for name in ("backup.sh", "migrate.py"):
        shutil.copy(ROOT / "deploy" / name, src / "deploy" / name)
    git(src, "add", "deploy")
    git(src, "commit", "-qm", "deploy scripts")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "FUND_DIR": str(src)}
    out = subprocess.run(["bash", str(src / "deploy" / "backup.sh")], env=env, check=True,
                         capture_output=True, text=True).stdout
    assert "rclone not installed" in out
    archive = next((src / "data" / "backups").glob("fund-backup-2*.tgz"))
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
        manifest = json.load(tar.extractfile(migrate.MANIFEST))
    assert {"repo/data/fund.db", "repo/data/paper/kf-chokepoint.db",
            "repo/config/compiled/active.json"} <= names
    assert "repo/.env" not in names and not any(n.startswith("home/") for n in names)
    assert "repo/data/montrose/token.json" not in names
    assert "repo/data/montrose/instruments.json" in names
    assert not any("backups/" in n for n in names)
    assert manifest["live"] is True
    assert restore(dst, home, archive) == 0
    assert count(dst / "data" / "fund.db") == 4


def test_a_financedata_cache_moved_by_financedata_db_is_carried(pi, mac, tmp_path, monkeypatch):
    (src, home), (dst, mac_home) = pi, mac
    make_db(home / "data" / "fd.db", 7).close()
    (src / ".env").write_text(f'OPENAI_API_KEY=x\nexport FINANCEDATA_DB="{home}/data/fd.db"\n')
    monkeypatch.delenv("FINANCEDATA_DB", raising=False)
    archive = tmp_path / "move.tgz"
    assert export(src, archive) == 0
    assert restore(dst, mac_home, archive) == 0
    assert count(mac_home / "data" / "fd.db") == 7
    assert count(mac_home / ".financedata" / "cache.db") == 5  # the default dir still comes too


def test_a_financedata_cache_outside_home_is_refused_not_dropped(pi, tmp_path, monkeypatch):
    src, _ = pi
    outside = tmp_path / "srv" / "fd.db"
    make_db(outside, 1).close()
    monkeypatch.setenv("FINANCEDATA_DB", str(outside))
    archive = tmp_path / "move.tgz"
    assert export(src, archive) == 1
    assert not archive.exists()
    assert export(src, archive, "--no-financedata-cache") == 0
