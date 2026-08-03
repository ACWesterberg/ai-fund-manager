#!/usr/bin/env bash
# backup.sh — snapshot the fund SQLite DBs + configs and push to Google Drive.
#
# Uses SQLite's online .backup (consistent even while a run is writing) so we
# never ship a half-written DB — the likely cause of card corruption. Uploads a
# timestamped tarball via rclone and prunes old copies locally + remotely.
#
# One-time setup and restore steps: see deploy/BACKUP.md
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

# Timestamp without Date.now quirks — plain date on the Pi.
STAMP="$(date '+%Y%m%d-%H%M%S')"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

LOG "=== Backup $STAMP ==="

# 1. Consistent hot-copy of each SQLite DB via Python's online backup API.
PY="$REPO_DIR/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
shopt -s nullglob
dbs=(data/*.db)
if [ ${#dbs[@]} -eq 0 ]; then
    LOG "  ⚠ no data/*.db found — nothing to back up"; exit 0
fi
for db in "${dbs[@]}"; do
    name="$(basename "$db")"
    "$PY" - "$db" "$STAGE/$name" <<'PY'
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
s = sqlite3.connect(src); d = sqlite3.connect(dst)
try:
    with d:
        s.backup(d)          # atomic, consistent snapshot
finally:
    s.close(); d.close()
PY
    LOG "  snapshot $name"
done

# 2. Include configs (also in git, but handy for a one-shot restore).
cp -r config "$STAGE/config" 2>/dev/null || true
[ "$BACKUP_ENV" = "1" ] && [ -f .env ] && cp .env "$STAGE/.env"

# 3. Tarball + upload.
ARCHIVE="fund-backup-$STAMP.tgz"
mkdir -p data/backups
tar -czf "data/backups/$ARCHIVE" -C "$STAGE" .
LOG "  archived data/backups/$ARCHIVE ($(du -h "data/backups/$ARCHIVE" | cut -f1))"

if command -v rclone >/dev/null 2>&1; then
    rclone copy "data/backups/$ARCHIVE" "$RCLONE_REMOTE" --quiet
    LOG "  uploaded → $RCLONE_REMOTE/$ARCHIVE"
    # Prune remote copies older than retention.
    rclone delete "$RCLONE_REMOTE" --min-age "${RETENTION_DAYS}d" --quiet 2>/dev/null || true
else
    LOG "  ⚠ rclone not installed — kept local archive only (see deploy/BACKUP.md)"
fi

# 4. Prune old local archives.
find data/backups -name 'fund-backup-*.tgz' -mtime "+${RETENTION_DAYS}" -delete 2>/dev/null || true
LOG "=== Backup complete ==="
