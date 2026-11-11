#!/usr/bin/env bash
# backup.sh — snapshot all fund state and push it to Google Drive.
#
# The archive is a `deploy/migrate.py export --live`: every file git does not hold
# (all SQLite databases under data/, live sleeves in data/paper/ included, What-If
# results, shadow comparisons, ticker aliases, config/compiled guidance and search
# checkpoints), each SQLite file taken through the online backup API so a run writing
# at the time cannot leave a half-written copy, with a manifest of hashes and row
# counts. Until 2026-11 this only took data/*.db, so live sleeves were never backed up.
#
# Restore with: python3 deploy/migrate.py restore <archive>   (deploy/BACKUP.md)
#
# Env overrides:
#   RCLONE_REMOTE    rclone remote:path         (default "gdrive:fund-manager-backups")
#   RETENTION_DAYS   keep backups newer than N  (default 14)
#   BACKUP_ENV       "1" to also back up .env and the Montrose token (default 0 — secrets!)
#   RCLONE_ARCHIVE_REMOTE  where data/archive/ goes (default "<RCLONE_REMOTE>-archive").
#                    Kept apart from RCLONE_REMOTE because that path is pruned
#                    by age, and the run archive is history that cannot be refetched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${FUND_DIR:-$SCRIPT_DIR/..}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:fund-manager-backups}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
RCLONE_ARCHIVE_REMOTE="${RCLONE_ARCHIVE_REMOTE:-${RCLONE_REMOTE}-archive}"
BACKUP_ENV="${BACKUP_ENV:-0}"

cd "$REPO_DIR"
LOG() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

STAMP="$(date '+%Y%m%d-%H%M%S')"
ARCHIVE="fund-backup-$STAMP.tgz"
mkdir -p data/backups
LOG "=== Backup $STAMP ==="

PY="$REPO_DIR/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
args=(export --live --no-logs --no-financedata-cache --no-run-archive --out "data/backups/$ARCHIVE")
[ "$BACKUP_ENV" = "1" ] || args+=(--no-secrets)
# A database failing its integrity check makes export exit 1 after writing the archive:
# upload it anyway (a damaged copy of one file beats no copy of all of them) and say so.
if ! "$PY" deploy/migrate.py --root "$REPO_DIR" "${args[@]}" | sed 's/^/  /'; then
    [ -f "data/backups/$ARCHIVE" ] || { LOG "  ✗ export failed, no archive written"; exit 1; }
    LOG "  ⚠ export reported a problem (above); keeping and uploading the archive anyway"
fi
LOG "  archived data/backups/$ARCHIVE ($(du -h "data/backups/$ARCHIVE" | cut -f1))"

if command -v rclone >/dev/null 2>&1; then
    rclone copy "data/backups/$ARCHIVE" "$RCLONE_REMOTE" --quiet
    LOG "  uploaded → $RCLONE_REMOTE/$ARCHIVE"
    rclone delete "$RCLONE_REMOTE" --min-age "${RETENTION_DAYS}d" --quiet 2>/dev/null || true
else
    LOG "  ⚠ rclone not installed — kept local archive only (see deploy/BACKUP.md)"
fi

# Run archive (data/archive/<fund>/<run_id>.json.gz): immutable, one file per run,
# hundreds of MB a year. Left out of the daily tarball (fourteen copies of a growing
# history) and copied incrementally to its own, never-pruned remote instead. A full
# migration export still carries it.
if [ -d data/archive ] && command -v rclone >/dev/null 2>&1; then
    rclone copy data/archive "$RCLONE_ARCHIVE_REMOTE" --ignore-existing --quiet
    LOG "  run archive synced → $RCLONE_ARCHIVE_REMOTE"
fi

find data/backups -name 'fund-backup-*.tgz' -mtime "+${RETENTION_DAYS}" -delete 2>/dev/null || true
LOG "=== Backup complete ==="
