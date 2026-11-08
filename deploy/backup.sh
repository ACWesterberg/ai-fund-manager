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
#   BACKUP_ENV       "1" to also back up .env    (default 0 — contains secrets!)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${FUND_DIR:-$SCRIPT_DIR/..}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:fund-manager-backups}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
BACKUP_ENV="${BACKUP_ENV:-0}"

cd "$REPO_DIR"
LOG() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

STAMP="$(date '+%Y%m%d-%H%M%S')"
ARCHIVE="fund-backup-$STAMP.tgz"
mkdir -p data/backups
LOG "=== Backup $STAMP ==="

PY="$REPO_DIR/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
args=(export --live --no-logs --no-financedata-cache --out "data/backups/$ARCHIVE")
[ "$BACKUP_ENV" = "1" ] || args+=(--no-env)
"$PY" deploy/migrate.py --root "$REPO_DIR" "${args[@]}" | sed 's/^/  /'
LOG "  archived data/backups/$ARCHIVE ($(du -h "data/backups/$ARCHIVE" | cut -f1))"

if command -v rclone >/dev/null 2>&1; then
    rclone copy "data/backups/$ARCHIVE" "$RCLONE_REMOTE" --quiet
    LOG "  uploaded → $RCLONE_REMOTE/$ARCHIVE"
    rclone delete "$RCLONE_REMOTE" --min-age "${RETENTION_DAYS}d" --quiet 2>/dev/null || true
else
    LOG "  ⚠ rclone not installed — kept local archive only (see deploy/BACKUP.md)"
fi

find data/backups -name 'fund-backup-*.tgz' -mtime "+${RETENTION_DAYS}" -delete 2>/dev/null || true
LOG "=== Backup complete ==="
