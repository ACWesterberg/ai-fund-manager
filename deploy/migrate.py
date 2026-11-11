#!/usr/bin/env python3
"""Move the fund manager's state from one machine to another without losing any of it.

    # On the Pi, with services and cron stopped (or --live for a trial copy):
    python3 deploy/migrate.py export
    # Copy the archive across, then on the Mac, from a fresh clone:
    python3 deploy/migrate.py restore ~/fund-migration-*.tgz
    python3 deploy/migrate.py verify          # any time later: still what was exported?

Everything git does not hold is state, so the export takes every untracked and ignored
file in the checkout (not a list of known paths: `config/compiled/` was never in the
backups' list of databases, and live sleeves under `data/paper/` were missed entirely by
the daily backup until this was written). SQLite files are snapshotted through the online
backup API, so a WAL that has not been checkpointed comes along consolidated. The
FinanceData cache in `~/.financedata` rides along too; it is regenerable for prices, but
not for news that has aged off the providers.

A manifest records each file's hash and each database's row counts and integrity check;
`restore` refuses to finish unless the unpacked files match it, and refuses to overwrite a
file that already exists with different content.

`deploy/backup.sh` takes the same export with --live every night, so a backup holds
exactly what a migration would, and restores the same way.

Stdlib only, Python 3.9+: it has to run on the new machine before the venv exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "MANIFEST.json"
PATCH = "local-changes.patch"
SQLITE_MAGIC = b"SQLite format 3\x00"
HOME_PREFIX = "home/"
REPO_PREFIX = "repo/"
FORMAT = 1

# Regenerable or machine-specific: a venv, caches, build output, and SQLite side files the
# snapshot already folds in. data/backups holds old tarballs already on Google Drive.
EXCLUDED_DIRS = {".venv", "__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache",
                 ".cache", "build", "dist", ".git"}
EXCLUDED_PREFIXES = ("data/backups/",)
EXCLUDED_NAMES = {"data/deploy.lock", "data/deploy.pid", "data/deployed-revision",
                  "data/moved-to-mac", ".DS_Store"}
# Credentials: API keys, and the Montrose OAuth token (its refresh token reaches the
# real broker account without BankID). A migration needs them; a backup bound for cloud
# storage does not, so --no-secrets leaves every one out.
SECRETS = (".env", "data/montrose/token.json", "data/web/session.key")
EXCLUDED_SUFFIXES = ("-wal", "-shm", "-journal", ".pyc", ".pyo")


def _excluded(rel: str) -> bool:
    parts = rel.split("/")
    return (any(p in EXCLUDED_DIRS or p.endswith(".egg-info") for p in parts[:-1])
            or rel.startswith(EXCLUDED_PREFIXES) or rel in EXCLUDED_NAMES
            or parts[-1] in EXCLUDED_NAMES or rel.endswith(EXCLUDED_SUFFIXES))


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def state_files(root: Path) -> list[str]:
    """Every file in the checkout that git does not track, minus the regenerable."""
    listed = _git(root, "ls-files", "-z", "--others")
    if listed is None:
        # No git: fall back to the places the code is known to write.
        found = [p for base in ("data", "config/compiled") for p in (root / base).rglob("*")]
        found += [root / ".env"]
        rels = [p.relative_to(root).as_posix() for p in found if p.is_file()]
    else:
        rels = [r for r in listed.split("\0") if r]
    return sorted(r for r in rels if not _excluded(r) and (root / r).is_file())


def is_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(16) == SQLITE_MAGIC
    except OSError:
        return False


def snapshot_sqlite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = sqlite3.connect(src)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        s.close()
        d.close()


def describe_sqlite(path: Path) -> dict:
    conn = sqlite3.connect(path)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name")]
        rows = {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
        return {"integrity": integrity, "rows": rows,
                "user_version": conn.execute("PRAGMA user_version").fetchone()[0]}
    finally:
        conn.close()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def running_fund_processes() -> list[str]:
    try:
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    me = str(os.getpid())
    hits = []
    for line in out.splitlines():
        pid, _, args = line.strip().partition(" ")
        if pid == me or "migrate.py" in args:
            continue
        if "fundmgr" in args or "/bin/fund " in args or args.endswith("/bin/fund"):
            hits.append(line.strip())
    return hits


def active_cron_lines(root: Path) -> list[str]:
    try:
        out = subprocess.run(["crontab", "-l"], check=False, capture_output=True,
                             text=True).stdout
    except OSError:
        return []
    marks = (str(root), "ai-fund-manager")
    return [ln for ln in out.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#") and any(m in ln for m in marks)]


def _entry(path: Path, arc: str) -> dict:
    entry = {"path": arc, "size": path.stat().st_size, "sha256": sha256(path)}
    if is_sqlite(path):
        entry["sqlite"] = describe_sqlite(path)
    return entry


def build_stage(root: Path, stage: Path, *, include_env: bool, home_dirs: list[Path],
                live: bool, include_logs: bool = True,
                include_run_archive: bool = True) -> dict:
    files: list[dict] = []
    for rel in state_files(root):
        if (rel in SECRETS and not include_env) or (rel.startswith("data/logs/") and not include_logs):
            continue
        if rel.startswith("data/archive/") and not include_run_archive:
            continue
        src, dst = root / rel, stage / REPO_PREFIX / rel
        if is_sqlite(src):
            snapshot_sqlite(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        files.append(_entry(dst, REPO_PREFIX + rel))

    home = Path.home()
    for base in home_dirs:
        if not base.is_dir():
            continue
        for src in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = src.relative_to(home).as_posix()
            if _excluded(rel):
                continue
            dst = stage / HOME_PREFIX / rel
            if is_sqlite(src):
                snapshot_sqlite(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            files.append(_entry(dst, HOME_PREFIX + rel))

    patch = _git(root, "diff", "--binary", "HEAD")
    if patch:
        (stage / PATCH).write_text(patch)
    return {
        "format": FORMAT,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017 (3.9)
        "host": socket.gethostname(),
        "source_root": str(root),
        "source_home": str(home),
        "git_head": (_git(root, "rev-parse", "HEAD") or "").strip() or None,
        "git_branch": (_git(root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "local_changes": bool(patch),
        "live": live,
        "files": files,
    }


def summarize(manifest: dict) -> list[str]:
    lines = []
    dbs = [f for f in manifest["files"] if "sqlite" in f]
    total = sum(f["size"] for f in manifest["files"])
    lines.append(f"{len(manifest['files'])} files, {total / 1e6:.1f} MB, {len(dbs)} SQLite databases")
    for f in dbs:
        rows = f["sqlite"]["rows"]
        busiest = sorted(rows.items(), key=lambda kv: -kv[1])[:4]
        shown = ", ".join(f"{t} {n}" for t, n in busiest)
        lines.append(f"  {f['path']}: {sum(rows.values())} rows ({shown})")
    return lines


def cmd_export(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    if not args.live:
        procs, cron = running_fund_processes(), active_cron_lines(root)
        if procs or cron:
            print("Refusing to export while the fund can still write: anything written after "
                  "the snapshot would stay behind on this machine.", file=sys.stderr)
            for p in procs:
                print(f"  running: {p}", file=sys.stderr)
            for c in cron:
                print(f"  cron:    {c}", file=sys.stderr)
            print("Stop the services and comment out the crontab (deploy/MIGRATE-TO-MAC.md), "
                  "or pass --live for a trial copy.", file=sys.stderr)
            return 1
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or Path.home() / f"fund-migration-{stamp}.tgz").expanduser()
    home_dirs = [] if args.no_financedata_cache else [Path.home() / ".financedata"]
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        manifest = build_stage(root, stage, include_env=not args.no_secrets, home_dirs=home_dirs,
                               live=args.live, include_logs=not args.no_logs,
                               include_run_archive=not args.no_run_archive)
        (stage / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
        bad = [f["path"] for f in manifest["files"]
               if "sqlite" in f and f["sqlite"]["integrity"] != "ok"]
        out.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh, tarfile.open(fileobj=fh, mode="w:gz") as tar:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    tar.add(p, arcname=p.relative_to(stage).as_posix(), recursive=False)
    for line in summarize(manifest):
        print(line)
    if manifest["local_changes"]:
        print(f"  ⚠ tracked files differ from git HEAD; saved as {PATCH} in the archive")
    if bad:
        print(f"  ⚠ integrity check failed on the source copy of: {', '.join(bad)}")
    if not args.no_secrets:
        print("  ⚠ the archive contains .env and the Montrose token: copy it machine to machine, not to "
              "cloud storage, and delete it once restored")
    print(f"{'live (trial/backup) ' if args.live else ''}archive: {out}")
    return 1 if bad else 0


def _safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = []
    for m in tar.getmembers():
        p = Path(m.name)
        if p.is_absolute() or ".." in p.parts or not (m.isfile() or m.isdir()):
            raise ValueError(f"refusing unsafe archive member: {m.name}")
        members.append(m)
    return members


def destination(arc: str, root: Path, home: Path) -> Path:
    if arc.startswith(REPO_PREFIX):
        return root / arc[len(REPO_PREFIX):]
    if arc.startswith(HOME_PREFIX):
        return home / arc[len(HOME_PREFIX):]
    raise ValueError(f"unexpected manifest path: {arc}")


def check(manifest: dict, locate, *, strict: bool = True) -> list[str]:
    """Problems found comparing files on disk with the manifest; empty means all match.

    Strict (restore): every byte must match. Otherwise (a later `verify`, once the fund
    has been running here) logs grow and positions close, so only a missing file or a
    failed integrity check is a problem; changes are listed for a human to read.
    """
    problems = []
    for f in manifest["files"]:
        path = locate(f["path"])
        if not path.is_file():
            problems.append(f"missing: {path}")
            continue
        if sha256(path) == f["sha256"]:
            continue
        if strict:
            problems.append(f"content differs: {path}")
            continue
        if "sqlite" not in f:
            print(f"  changed since export: {path}")
            continue
        now = describe_sqlite(path) if is_sqlite(path) else {"integrity": "not a database",
                                                             "rows": {}}
        if now["integrity"] != "ok":
            problems.append(f"integrity {now['integrity']}: {path}")
            continue
        fewer = [f"{t} {n}→{now['rows'].get(t, 0)}" for t, n in sorted(f["sqlite"]["rows"].items())
                 if now["rows"].get(t, 0) < n]
        print(f"  changed since export: {path}" + (
            f" (fewer rows: {', '.join(fewer)})" if fewer else " (no table has fewer rows)"))
    return problems


def mentions_old_root(manifest: dict, locate) -> list[Path]:
    old = manifest.get("source_root", "").encode()
    hits = []
    for f in manifest["files"]:
        path = locate(f["path"])
        if (old and "sqlite" not in f and path.is_file() and f["size"] < 50_000_000
                and old in path.read_bytes()):
            hits.append(path)
    return hits


def cmd_restore(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    home = Path(args.home).expanduser().resolve() if args.home else Path.home()
    archive = Path(args.archive).expanduser()
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(archive, "r:gz") as tar:
        stage = Path(tmp)
        # The "data" filter (3.12+) adds its own refusal of links and odd modes.
        extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(stage, members=_safe_members(tar), **extra)
        manifest = json.loads((stage / MANIFEST).read_text())
        if manifest.get("format") != FORMAT:
            print(f"unknown archive format {manifest.get('format')}", file=sys.stderr)
            return 1
        if manifest.get("live"):
            print("⚠ taken while the fund was running (a backup or a trial copy): the source "
                  "may have written more since. For the move itself, restore the final export "
                  "over this one with --force.")
        problems = check(manifest, lambda arc: stage / arc)
        if problems:
            print("Archive does not match its own manifest; nothing restored:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 1
        clashes = []
        for f in manifest["files"]:
            dst = destination(f["path"], root, home)
            if dst.exists() and sha256(dst) != f["sha256"]:
                clashes.append(dst)
        if clashes and not args.force:
            print("These already exist with different content; nothing restored "
                  "(--force replaces them, keeping each as <name>.pre-migration):",
                  file=sys.stderr)
            for c in clashes:
                print(f"  {c}", file=sys.stderr)
            return 1
        for f in manifest["files"]:
            src, dst = stage / f["path"], destination(f["path"], root, home)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst in clashes:
                shutil.move(str(dst), str(dst) + ".pre-migration")
                for side in ("-wal", "-shm"):
                    stale = Path(str(dst) + side)
                    if stale.exists():
                        stale.unlink()
            shutil.copy2(src, dst)
            if f["path"][len(REPO_PREFIX):] in SECRETS:
                dst.chmod(0o600)
                if dst.parent != root:
                    dst.parent.chmod(0o700)
        record = root / "data" / "migration"
        record.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage / MANIFEST, record / f"MANIFEST-{manifest['created'][:19].replace(':', '')}.json")
        shutil.copy2(stage / MANIFEST, record / MANIFEST)
        if (stage / PATCH).exists():
            shutil.copy2(stage / PATCH, record / PATCH)

    problems = check(manifest, lambda arc: destination(arc, root, home))
    for line in summarize(manifest):
        print(line)
    if manifest.get("git_head"):
        here = (_git(root, "rev-parse", "HEAD") or "").strip()
        note = "same" if here == manifest["git_head"] else f"this checkout is {here[:12] or '?'}"
        print(f"  source ran {manifest.get('git_branch')} @ {manifest['git_head'][:12]} ({note})")
    if manifest.get("local_changes"):
        print(f"  ⚠ the source had uncommitted edits to tracked files: {record / PATCH}")
    stale = mentions_old_root(manifest, lambda arc: destination(arc, root, home))
    if stale and manifest["source_root"] != str(root):
        print(f"  note: {len(stale)} restored file(s) mention {manifest['source_root']} "
              "(checkpoint and report records; informational, nothing reads them as paths)")
    if problems:
        print("Restored files do not match the manifest:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"restored into {root} and verified against the manifest")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    home = Path(args.home).expanduser().resolve() if args.home else Path.home()
    path = Path(args.manifest) if args.manifest else root / "data" / "migration" / MANIFEST
    manifest = json.loads(path.read_text())
    problems = check(manifest, lambda arc: destination(arc, root, home), strict=args.strict)
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    print(f"{len(manifest['files'])} files checked against the export of {manifest['created']}: "
          + ("ok" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(ROOT), help="fund manager checkout (default: this one)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="snapshot all state into one archive")
    ex.add_argument("--out", help="archive path (default ~/fund-migration-<stamp>.tgz)")
    ex.add_argument("--live", action="store_true",
                    help="allow running services and cron (backups, trial copies); "
                         "recorded in the manifest")
    ex.add_argument("--no-logs", action="store_true", help="leave data/logs out")
    ex.add_argument("--no-run-archive", action="store_true",
                    help="leave data/archive out (daily backups sync it incrementally instead)")
    ex.add_argument("--no-secrets", "--no-env", dest="no_secrets", action="store_true",
                    help="leave credentials out: .env and the Montrose token")
    ex.add_argument("--no-financedata-cache", action="store_true",
                    help="leave ~/.financedata out (regenerable apart from aged-off news)")
    ex.set_defaults(func=cmd_export)

    rs = sub.add_parser("restore", help="unpack an archive into this checkout and verify it")
    rs.add_argument("archive")
    rs.add_argument("--force", action="store_true",
                    help="replace differing files (old copies kept as *.pre-migration)")
    rs.add_argument("--home", help="where ~/.financedata goes (default: $HOME)")
    rs.set_defaults(func=cmd_restore)

    vf = sub.add_parser("verify", help="check restored state against the manifest")
    vf.add_argument("--manifest", help="default data/migration/MANIFEST.json")
    vf.add_argument("--home", help="default: $HOME")
    vf.add_argument("--strict", action="store_true",
                    help="every file byte-identical (right after restore, before anything runs)")
    vf.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
